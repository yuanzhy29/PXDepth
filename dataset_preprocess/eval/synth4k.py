"""Convert the five Synth4K subsets into PXDepth evaluation data.

Expected input follows the official InfiniDepth download and metadata layout::

    input_dir/
      datasets/
        cyberpunk/<files referenced by val.txt>
        spiderman2/<files referenced by val.txt>
        spidermanmm/<files referenced by val.txt>
        deadisland/<files referenced by val.txt>
        watchdoglegion/<files referenced by val.txt>
      processed_datasets/
        cyberpunk/val.txt
        spiderman2/val.txt
        spidermanmm/val.txt
        deadisland/val.txt
        watchdoglegion/val.txt

Each non-comment manifest line is ``rgb_rel_path depth_rel_path`` and paths are
relative to that subset's directory under ``datasets``. ``--input_dir`` may
instead point directly at ``datasets`` when ``--meta_dir`` points at
``processed_datasets``. Depth supports NPY, NPZ, HDF5, EXR, TIFF, and PNG and is
preserved as metric depth by default. Invalid or non-positive values become
NaN. Evaluation-range filtering remains in ``all_benchmarks.json``.

The output contains independent ``Synth4K-1`` through ``Synth4K-5`` roots,
each with its own ``.index.txt`` as required by the released eval config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import BaseDataset
from depth import DepthData
from runner import add_common_args, processor_kwargs


SUBSETS = {
    "Synth4K-1": "cyberpunk",
    "Synth4K-2": "spiderman2",
    "Synth4K-3": "spidermanmm",
    "Synth4K-4": "deadisland",
    "Synth4K-5": "watchdoglegion",
}


def _read_depth(path: Path, scale: float) -> np.ndarray:
    """Read one metric depth map from a supported Synth4K container.

    Args:
        path: Source depth path referenced by a manifest.
        scale: Unit multiplier applied after decoding.

    Returns:
        ``float32 [H,W]`` metric depth with invalid support set to NaN.
    """

    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if not archive.files:
                raise ValueError(f"Empty NPZ depth: {path}")
            key = "data" if "data" in archive.files else archive.files[0]
            depth = archive[key]
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as file:
            key = "dataset" if "dataset" in file else next(iter(file.keys()))
            depth = file[key][()]
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise OSError(f"Cannot decode Synth4K depth: {path}")
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape [H,W], got {depth.shape}: {path}")
    depth = depth.astype(np.float32, copy=True) * np.float32(scale)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def _key(root: Path, depth_path: Path) -> str:
    """Build a stable output key from a manifest depth path.

    Args:
        root: Raw subset root.
        depth_path: Resolved source depth path.

    Returns:
        Safe relative key with modality-directory components removed.
    """

    try:
        relative = depth_path.relative_to(root).with_suffix("")
    except ValueError:
        relative = Path(depth_path.stem)
    ignored = {"depth", "depths", "depth_map", "depth_maps", "dpt"}
    parts = [part for part in relative.parts if part.lower() not in ignored]
    return Path(*(parts or [depth_path.stem])).as_posix()


class Synth4K(BaseDataset[tuple[str, str, str]]):
    """Convert one manifest-backed Synth4K game subset."""

    def __init__(self, *args, subset: str, manifest: str | Path, depth_scale: float = 1.0, focal_px: float = 1440.0, **kwargs) -> None:
        """Configure one subset and its official validation manifest.

        Args:
            subset: Released output name, from ``Synth4K-1`` to ``Synth4K-5``.
            manifest: Text file containing relative RGB/depth pairs.
            depth_scale: Multiplier converting source depth to meters.
            focal_px: Fallback focal length used by the released benchmark.
            *args: Common preprocessor positional arguments.
            **kwargs: Common preprocessor keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.subset = subset
        self.manifest = Path(manifest).expanduser().resolve()
        self.depth_scale = float(depth_scale)
        self.focal_px = float(focal_px)
        self.name = subset

    def discover(self) -> list[tuple[str, str, str]]:
        """Read complete RGB/depth pairs from the official manifest.

        Returns:
            Tuples of relative RGB path, relative depth path, and output key.
        """

        if not self.manifest.is_file():
            raise FileNotFoundError(f"Synth4K manifest does not exist: {self.manifest}")
        tasks = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            image = Path(fields[0])
            depth = Path(fields[1])
            image_path = image if image.is_absolute() else self.input_dir / image
            depth_path = depth if depth.is_absolute() else self.input_dir / depth
            if image_path.is_file() and depth_path.is_file():
                tasks.append((str(image_path), str(depth_path), _key(self.input_dir, depth_path)))
        return tasks

    def key(self, frame: tuple[str, str, str]) -> str:
        """Return the manifest-derived stable sample key."""

        return frame[2]

    def read_rgb(self, frame: tuple[str, str, str]) -> Path:
        """Return the absolute RGB path from the manifest."""

        return Path(frame[0])

    def read_depth(self, frame: tuple[str, str, str]) -> DepthData:
        """Decode one metric Synth4K depth source."""

        return DepthData(_read_depth(Path(frame[1]), self.depth_scale))

    def read_intrinsics(self, frame: tuple[str, str, str], width: int, height: int) -> np.ndarray:
        """Build the released centered fallback camera calibration."""

        focal = self.focal_px if self.focal_px > 0 else float(max(width, height))
        return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float32)

    def metadata(self, frame: tuple[str, str, str]) -> dict[str, str]:
        """Record released subset name and metric depth units."""

        return {"dataset": self.subset, "depth_unit": "meter"}


def _roots(input_dir: Path, meta_dir: Path | None) -> tuple[Path, Path]:
    """Resolve dataset and manifest roots from the two supported CLI forms.

    Args:
        input_dir: Commonspace root or its nested ``datasets`` directory.
        meta_dir: Explicit processed-dataset metadata root, when provided.

    Returns:
        ``(datasets_root, manifests_root)`` absolute paths.
    """

    if (input_dir / "datasets").is_dir():
        datasets = input_dir / "datasets"
        manifests = meta_dir or input_dir / "processed_datasets"
    else:
        datasets = input_dir
        manifests = meta_dir or input_dir.parent / "processed_datasets"
    return datasets.resolve(), manifests.resolve()


def main() -> None:
    """Parse options and preprocess selected Synth4K subsets."""

    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--meta_dir", type=Path, default=None, help="Root containing one <game>/val.txt per subset.")
    parser.add_argument("--subsets", nargs="*", choices=tuple(SUBSETS), default=list(SUBSETS), help="Output subsets to process.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--focal_px", type=float, default=1440.0)
    args = parser.parse_args()
    datasets, manifests = _roots(Path(args.input_dir).expanduser().resolve(), args.meta_dir)
    common = processor_kwargs(args)
    for subset in args.subsets:
        game = SUBSETS[subset]
        common_subset = dict(common)
        common_subset["input_dir"] = datasets / game
        common_subset["output_dir"] = Path(args.output_dir) / subset
        manifest_candidates = (manifests / game / "val.txt", manifests / game / "val_new.txt")
        manifest = next((path for path in manifest_candidates if path.is_file()), manifest_candidates[0])
        Synth4K(
            **common_subset,
            subset=subset,
            manifest=manifest,
            depth_scale=args.depth_scale,
            focal_px=args.focal_px,
        ).run()


if __name__ == "__main__":
    main()
