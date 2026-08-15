"""Convert the five Synth4K subsets into PXDepth evaluation data.

Expected input directory::

    input_dir/
      cyberpunk/<RGB and depth files>
      spiderman2/<RGB and depth files>
      spidermanmm/<RGB and depth files>
      deadisland/<RGB and depth files>
      watchdoglegion/<RGB and depth files>

All discoverable RGB/depth pairs are converted by default, matching the
original research preprocessor. An InfiniDepth ``processed_datasets`` root can
optionally be supplied through ``--meta_dir`` to restrict conversion using
``val_new.txt``, ``val.txt``, or ``test.txt``. Depth supports NPY, NPZ, HDF5,
EXR, TIFF, and PNG and is preserved as metric depth by default. Invalid or
non-positive values become NaN. Evaluation-range filtering remains in
``all_benchmarks.json``.

The output contains independent ``Synth4K-1`` through ``Synth4K-5`` roots,
each with its own ``.index.txt`` as required by the released eval config.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import BaseDataset
from depth import DepthData
from runner import add_common_args, processor_kwargs


SUBSETS = {
    "Synth4K-1": ("cyberpunk", ("cyberpunk", "CyberPunk", "Synth4K-1", "synth4k-1")),
    "Synth4K-2": ("spiderman2", ("spiderman2", "spider_man_2", "SpiderMan2", "Synth4K-2", "synth4k-2")),
    "Synth4K-3": ("spidermanmm", ("spidermanmm", "spiderman_miles_morales", "SpiderManMM", "Synth4K-3", "synth4k-3")),
    "Synth4K-4": ("deadisland", ("deadisland", "dead_island", "DeadIsland", "Synth4K-4", "synth4k-4")),
    "Synth4K-5": ("watchdoglegion", ("watchdoglegion", "watch_dogs_legion", "WatchDogLegion", "Synth4K-5", "synth4k-5")),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEPTH_EXTS = {".png", ".exr", ".npy", ".npz", ".hdf5", ".h5"}
IMAGE_DIRS = {"rgb", "image", "images", "color", "colors", "left", "frame", "frames"}
DEPTH_DIRS = {"depth", "depths", "depth_map", "depth_maps", "dpt"}
IGNORE_IMAGE_DIRS = DEPTH_DIRS | {"mask", "masks", "segmentation", "semantic", "normal", "normals"}


def _subset_root(root: Path, aliases: tuple[str, ...]) -> Path | None:
    """Resolve one game folder from its accepted raw names."""

    aliases_lower = {name.lower() for name in aliases}
    if root.name.lower() in aliases_lower:
        return root
    children = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    return next((children[name.lower()] for name in aliases if name.lower() in children), None)


def _manifest(meta_dir: Path | None, subset: str, game: str, aliases: tuple[str, ...]) -> Path | None:
    """Find an optional official split file without making it mandatory."""

    if meta_dir is None:
        return None
    candidates = []
    for name in (game, subset, *aliases):
        candidates.extend(meta_dir / name / filename for filename in ("val_new.txt", "val.txt", "test.txt"))
    candidates.extend(meta_dir / filename for filename in ("val_new.txt", "val.txt", "test.txt"))
    return next((path for path in candidates if path.is_file()), None)


def _resolve_path(root: Path, value: str) -> Path:
    """Resolve a manifest path relative to the game folder or its parents."""

    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [root / path, *(parent / path for parent in root.parents)]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _image_candidates(depth: Path) -> Iterable[Path]:
    """Generate structural RGB candidates for one depth file."""

    depth_dir = next((parent for parent in (depth.parent, *depth.parents) if parent.name.lower() in DEPTH_DIRS), None)
    if depth_dir is None:
        return ()
    relative = depth.relative_to(depth_dir)
    stems = {
        depth.stem,
        depth.stem.replace("_depth", ""),
        depth.stem.replace("-depth", ""),
        depth.stem.replace(".depth", ""),
    }
    candidates = []
    for directory in IMAGE_DIRS:
        image_dir = depth_dir.parent / directory
        candidates.extend((image_dir / relative).with_suffix(ext) for ext in IMAGE_EXTS)
        candidates.extend(image_dir / relative.parent / f"{stem}{ext}" for stem in stems for ext in IMAGE_EXTS)
    return candidates


def _discover_pairs(root: Path) -> list[tuple[str, str, str]]:
    """Discover every RGB/depth pair using the original converter rules."""

    images: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = {part.lower() for part in path.parts}
        if parts & IGNORE_IMAGE_DIRS or any("mask" in part for part in parts):
            continue
        images.setdefault(path.stem, []).append(path)

    depths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DEPTH_EXTS or "mask" in path.stem.lower():
            continue
        parts = {part.lower() for part in path.parts}
        if parts & DEPTH_DIRS or "depth" in path.stem.lower():
            depths.append(path)

    pairs = []
    for depth in sorted(depths):
        image = next((candidate for candidate in _image_candidates(depth) if candidate.is_file()), None)
        if image is None:
            stems = (
                depth.stem,
                depth.stem.replace("_depth", ""),
                depth.stem.replace("-depth", ""),
                depth.stem.replace(".depth", ""),
            )
            image = next(
                (sorted(images[stem], key=lambda path: len(path.parts))[0] for stem in stems if images.get(stem)),
                None,
            )
        if image is not None:
            pairs.append((str(image), str(depth), _key(root, depth)))
    return pairs


def _manifest_pairs(root: Path, manifest: Path) -> list[tuple[str, str, str]]:
    """Read complete RGB/depth pairs from an optional split manifest."""

    pairs = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        image, depth = _resolve_path(root, fields[0]), _resolve_path(root, fields[1])
        if image.is_file() and depth.is_file():
            pairs.append((str(image), str(depth), _key(root, depth)))
    return pairs


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
    """Convert one Synth4K game subset, optionally restricted by a manifest."""

    def __init__(self, *args, subset: str, manifest: str | Path | None = None, depth_scale: float = 1.0, focal_px: float = 1440.0, **kwargs) -> None:
        """Configure one subset and an optional validation manifest.

        Args:
            subset: Released output name, from ``Synth4K-1`` to ``Synth4K-5``.
            manifest: Optional text file containing relative RGB/depth pairs.
            depth_scale: Multiplier converting source depth to meters.
            focal_px: Fallback focal length used by the released benchmark.
            *args: Common preprocessor positional arguments.
            **kwargs: Common preprocessor keyword arguments.
        """

        super().__init__(*args, **kwargs)
        self.subset = subset
        self.manifest = None if manifest is None else Path(manifest).expanduser().resolve()
        self.depth_scale = float(depth_scale)
        self.focal_px = float(focal_px)
        self.name = subset

    def discover(self) -> list[tuple[str, str, str]]:
        """Read a supplied manifest or discover every complete pair.

        Returns:
            Tuples of relative RGB path, relative depth path, and output key.
        """

        if self.manifest is not None:
            return _manifest_pairs(self.input_dir, self.manifest)
        return _discover_pairs(self.input_dir)

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


def main() -> None:
    """Parse options and preprocess selected Synth4K subsets."""

    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--meta_dir", type=Path, default=None, help="Optional root containing split manifests.")
    parser.add_argument("--subsets", nargs="*", choices=tuple(SUBSETS), default=list(SUBSETS), help="Output subsets to process.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--focal_px", type=float, default=1440.0)
    args = parser.parse_args()
    input_root = Path(args.input_dir).expanduser().resolve()
    meta_dir = None if args.meta_dir is None else args.meta_dir.expanduser().resolve()
    common = processor_kwargs(args)
    for subset in args.subsets:
        game, aliases = SUBSETS[subset]
        root = _subset_root(input_root, aliases)
        if root is None:
            print(f"[{subset}] subset folder not found under {input_root}; skipping")
            continue
        common_subset = dict(common)
        common_subset["input_dir"] = root
        common_subset["output_dir"] = Path(args.output_dir) / subset
        manifest = _manifest(meta_dir, subset, game, aliases)
        if manifest is None:
            print(f"[{subset}] discovering all RGB/depth pairs under {root}")
        else:
            print(f"[{subset}] using split manifest {manifest}")
        Synth4K(
            **common_subset,
            subset=subset,
            manifest=manifest,
            depth_scale=args.depth_scale,
            focal_px=args.focal_px,
        ).run()


if __name__ == "__main__":
    main()
