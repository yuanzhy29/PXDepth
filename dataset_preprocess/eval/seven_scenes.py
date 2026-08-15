"""Convert the 7Scenes test split into PXDepth evaluation data.

Expected input directory::

    input_dir/
      chess/
        TestSplit.txt
        seq-01/
          frame-000000.color.png
          frame-000000.depth.proj.png
          frame-000000.pose.txt
      fire/ ...

DA3-BENCH mirrors that omit ``TestSplit.txt`` are also accepted; every
directory containing ``frame-*.color.png`` is then used. Depth PNGs are uint16
millimeters, with 65535, values below 1 mm, and values above 10 m treated as
unknown NaN. The benchmark uses the fixed 7Scenes intrinsics and keeps every
200th frame by default.
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


class SevenScenes(BaseDataset[tuple[str, str]]):
    """Convert selected 7Scenes sequences and frames."""

    name = "7Scenes"
    resize_rgb_to_depth = True

    def __init__(self, *args, split: str = "test", stride: int = 200, **kwargs) -> None:
        """Configure split-file selection and frame stride.

        Args:
            split: ``train`` or ``test`` split file.
            stride: Keep every Nth frame in each sequence.
            *args: Common positional arguments.
            **kwargs: Common keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.split, self.stride = split, max(1, int(stride))

    def discover(self) -> list[tuple[str, str]]:
        """Read official split files or discover complete mirror sequences.

        Returns:
            ``(sequence_relative_path, frame_id)`` tasks.
        """

        sequences = []
        split_name = "TrainSplit.txt" if self.split == "train" else "TestSplit.txt"
        for scene in sorted(path for path in self.input_dir.iterdir() if path.is_dir()):
            split = scene / split_name
            if split.is_file():
                for line in split.read_text(encoding="utf-8").splitlines():
                    digits = "".join(character for character in line if character.isdigit())
                    if digits:
                        sequences.append(scene / f"seq-{digits.zfill(2)}")
        if not sequences:
            sequences = sorted({path.parent for path in self.input_dir.rglob("frame-*.color.png")})
        tasks = []
        for sequence in sequences:
            images = sorted(sequence.glob("frame-*.color.png"))[:: self.stride]
            for image in images:
                frame = image.name[len("frame-") : -len(".color.png")]
                tasks.append((sequence.relative_to(self.input_dir).as_posix(), frame))
        return tasks

    def _root(self, frame: tuple[str, str]) -> Path:
        """Return the raw sequence directory."""

        return self.input_dir / frame[0]

    def key(self, frame: tuple[str, str]) -> str:
        """Return sequence/frame output key."""

        return f"{frame[0]}/{frame[1]}"

    def read_rgb(self, frame: tuple[str, str]) -> Path:
        """Return the source color frame path."""

        return self._root(frame) / f"frame-{frame[1]}.color.png"

    def _depth_path(self, frame: tuple[str, str]) -> Path:
        """Resolve projected depth with fallback to the shorter mirror name."""

        root, token = self._root(frame), frame[1]
        depth_path = root / f"frame-{token}.depth.proj.png"
        if not depth_path.is_file():
            depth_path = root / f"frame-{token}.depth.png"
        return depth_path

    def read_depth(self, frame: tuple[str, str]) -> DepthData:
        """Decode millimeter depth and reject the official invalid range."""

        depth_path = self._depth_path(frame)
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"Cannot read {depth_path}")
        return DepthData(raw, scale=0.001, invalid=raw == 65535, min_depth=0.001, max_depth=10.0)

    def read_intrinsics(self, frame: tuple[str, str], width: int, height: int) -> np.ndarray:
        """Return the fixed 7Scenes pixel-space calibration."""

        return np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)

    def read_pose(self, frame: tuple[str, str]) -> np.ndarray:
        """Load the camera-to-world pose text matrix."""

        path = self._root(frame) / f"frame-{frame[1]}.pose.txt"
        return np.loadtxt(path).astype(np.float32)

    def metadata(self, frame: tuple[str, str]) -> dict[str, str]:
        """Record selected source depth path and metric units."""

        return {"source_depth": self._depth_path(frame).relative_to(self.input_dir).as_posix(), "depth_unit": "meter"}


def main() -> None:
    """Parse command-line arguments and run the 7Scenes converter."""

    parser = add_common_args(argparse.ArgumentParser(description=__doc__), workers=1)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--stride", type=int, default=200)
    args = parser.parse_args()
    SevenScenes(**processor_kwargs(args), split=args.split, stride=args.stride).run()


if __name__ == "__main__":
    main()
