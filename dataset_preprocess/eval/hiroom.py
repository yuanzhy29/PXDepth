"""Convert the HiRoom validation scenes into PXDepth evaluation data.

Expected input directory::

    input_dir/
      selected_scene_list_val.txt
      data/<scene>/
        image/<frame>.jpg
        depth/<frame>.png
        pose/<frame>.npy
        aliasing_mask/<frame>.png
        cam_K.npy

``input_dir`` may point at the nested ``data`` directory; its parent is then
used automatically. HiRoom depth encodes 0--100 m into uint16 0--65535.
Aliasing-mask pixels and non-positive depth become unknown NaN. ``pose`` files
store world-to-camera matrices and are inverted before writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import BaseDataset
from depth import DepthData
from runner import add_common_args, processor_kwargs


class HiRoom(BaseDataset[tuple[str, str]]):
    """Convert all complete frames from the listed HiRoom validation scenes."""

    name = "HiRoom"
    resize_rgb_to_depth = True

    def __init__(self, *args, **kwargs) -> None:
        """Resolve a dataset root provided either above or at ``data/``.

        Args:
            *args: Common positional arguments.
            **kwargs: Common keyword arguments.
        """

        super().__init__(*args, **kwargs)
        if self.input_dir.name == "data" and (self.input_dir.parent / "selected_scene_list_val.txt").is_file():
            self.input_dir = self.input_dir.parent

    def discover(self) -> list[tuple[str, str]]:
        """Match RGB, depth, pose, mask, and calibration files.

        Returns:
            ``(scene, frame_id)`` tasks.
        """

        list_path = self.input_dir / "selected_scene_list_val.txt"
        scenes = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tasks = []
        for scene in scenes:
            root = self.input_dir / "data" / scene
            image_root = root / "image"
            if not image_root.is_dir():
                continue
            for image in sorted(path for path in image_root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
                frame = image.stem
                if all((root / folder / f"{frame}{suffix}").is_file() for folder, suffix in (("depth", ".png"), ("pose", ".npy"), ("aliasing_mask", ".png"))) and (root / "cam_K.npy").is_file():
                    tasks.append((scene, frame))
        return tasks

    def _root(self, frame: tuple[str, str]) -> Path:
        """Return the selected validation scene root."""

        return self.input_dir / "data" / frame[0]

    def key(self, frame: tuple[str, str]) -> str:
        """Return scene/frame output key."""

        return f"{frame[0]}/{frame[1]}"

    def read_rgb(self, frame: tuple[str, str]) -> Path:
        """Resolve the RGB frame regardless of JPEG or PNG suffix."""

        return next(path for path in (self._root(frame) / "image").glob(f"{frame[1]}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def read_depth(self, frame: tuple[str, str]) -> DepthData:
        """Decode 0--100 m depth and mark aliasing pixels unknown."""

        root, token = self._root(frame), frame[1]
        raw = cv2.imread(str(root / "depth" / f"{token}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(root / "aliasing_mask" / f"{token}.png"), cv2.IMREAD_UNCHANGED)
        if raw is None or mask is None:
            raise OSError(f"Cannot read HiRoom frame {frame[0]}/{token}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return DepthData(raw, scale=100.0 / 65535.0, invalid=mask > 0)

    def read_intrinsics(self, frame: tuple[str, str], width: int, height: int) -> np.ndarray:
        """Load the scene pixel-space camera matrix."""

        return np.load(self._root(frame) / "cam_K.npy").astype(np.float32)

    def read_pose(self, frame: tuple[str, str]) -> np.ndarray:
        """Invert the stored world-to-camera matrix."""

        return np.linalg.inv(np.load(self._root(frame) / "pose" / f"{frame[1]}.npy").astype(np.float32))

    def metadata(self, frame: tuple[str, str]) -> dict[str, str]:
        """Record aliasing-mask provenance and metric units."""

        return {"source_aliasing_mask": f"aliasing_mask/{frame[1]}.png", "depth_unit": "meter"}


def main() -> None:
    """Parse command-line arguments and run the HiRoom converter."""

    parser = add_common_args(argparse.ArgumentParser(description=__doc__), workers=1)
    HiRoom(**processor_kwargs(parser.parse_args())).run()


if __name__ == "__main__":
    main()
