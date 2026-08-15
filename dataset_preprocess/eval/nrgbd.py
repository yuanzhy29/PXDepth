"""Convert Neural RGB-D (NRGBD) scenes into PXDepth evaluation data.

Expected input directory::

    input_dir/
      <scene>/
        images/img0.png
        depth/depth0.png
        poses.txt

``poses.txt`` contains consecutive four-line 4x4 camera poses. Depth PNGs are
millimeters. Values below 1 mm or above 10 m become unknown NaN. The converter
uses NRGBD's fixed camera intrinsics, changes pose axes from OpenGL to OpenCV,
and keeps every 100th frame by default.
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


def _poses(path: Path) -> np.ndarray:
    """Parse four-line NRGBD pose matrices.

    Args:
        path: Scene ``poses.txt`` path.

    Returns:
        ``float32 [N,4,4]`` camera poses. NaN blocks become identity.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    values = []
    for start in range(0, len(lines), 4):
        block = lines[start : start + 4]
        if len(block) < 4 or any("nan" in line.lower() for line in block):
            values.append(np.eye(4, dtype=np.float32))
        else:
            values.append(np.array([[float(item) for item in line.split()] for line in block], dtype=np.float32))
    return np.stack(values) if values else np.empty((0, 4, 4), dtype=np.float32)


class NRGBD(BaseDataset[tuple[str, int, list[list[float]]]]):
    """Convert temporally thinned NRGBD scene frames."""

    name = "NRGBD"
    resize_rgb_to_depth = True

    def __init__(self, *args, stride: int = 100, **kwargs) -> None:
        """Configure the nominal frame interval.

        Args:
            stride: Maximum keyframe interval; short scenes retain at least two
                potential samples through the original benchmark rule.
            *args: Common positional arguments.
            **kwargs: Common keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.stride = max(1, int(stride))

    def discover(self) -> list[tuple[str, int, list[list[float]]]]:
        """Pair image indices with available poses for every scene.

        Returns:
            ``(scene, frame_index, pose)`` tasks.
        """

        tasks = []
        for scene in sorted(path for path in self.input_dir.iterdir() if path.is_dir()):
            if not (scene / "images").is_dir() or not (scene / "poses.txt").is_file():
                continue
            poses = _poses(scene / "poses.txt")
            image_count = len(list((scene / "images").glob("img*.png")))
            step = min(self.stride, max(1, image_count // 2))
            for frame in range(0, min(image_count, len(poses)), step):
                tasks.append((scene.name, frame, poses[frame].tolist()))
        return tasks

    def _root(self, frame: tuple[str, int, list[list[float]]]) -> Path:
        """Return the raw scene directory."""

        return self.input_dir / frame[0]

    def key(self, frame: tuple[str, int, list[list[float]]]) -> str:
        """Return scene/frame output key."""

        return f"{frame[0]}/{frame[1]}"

    def read_rgb(self, frame: tuple[str, int, list[list[float]]]) -> Path:
        """Return the indexed RGB frame path."""

        return self._root(frame) / "images" / f"img{frame[1]}.png"

    def read_depth(self, frame: tuple[str, int, list[list[float]]]) -> DepthData:
        """Decode millimeter depth and apply NRGBD's valid range."""

        path = self._root(frame) / "depth" / f"depth{frame[1]}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"Cannot read {path}")
        return DepthData(raw, scale=0.001, min_depth=0.001, max_depth=10.0)

    def read_intrinsics(self, frame: tuple[str, int, list[list[float]]], width: int, height: int) -> np.ndarray:
        """Return NRGBD's fixed pixel-space calibration."""

        return np.array([[554.2562584220408, 0, 320], [0, 554.2562584220408, 240], [0, 0, 1]], dtype=np.float32)

    def read_pose(self, frame: tuple[str, int, list[list[float]]]) -> np.ndarray:
        """Convert the stored OpenGL-like pose axes to OpenCV convention."""

        pose = np.asarray(frame[2], dtype=np.float32).copy()
        pose[:, 1:3] *= -1
        return pose

    def metadata(self, frame: tuple[str, int, list[list[float]]]) -> dict[str, str]:
        """Record metric depth units."""

        return {"depth_unit": "meter"}


def main() -> None:
    """Parse command-line arguments and run the NRGBD converter."""

    parser = add_common_args(argparse.ArgumentParser(description=__doc__), workers=1)
    parser.add_argument("--stride", type=int, default=100)
    args = parser.parse_args()
    NRGBD(**processor_kwargs(args), stride=args.stride).run()


if __name__ == "__main__":
    main()
