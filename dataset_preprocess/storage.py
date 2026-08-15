"""Serialization helpers shared by dataset preprocessors.

Every converter writes the same directory-per-sample representation. This
module validates that representation and keeps depth validity as three states:

* a finite positive number is an observed depth value
* ``NaN`` is unknown geometry and is ignored by evaluation
* ``Inf`` is known infinite geometry and may supervise the validity head

Depth values may be stored in dataset-native units. The corresponding
``depth_unit`` in the evaluation config converts them to meters.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pxdepth.utils.io import write_depth, write_image, write_json, write_segmentation


@dataclass(frozen=True)
class Sample:
    """One converted sample before it is serialized.

    Args:
        key: Relative output path such as ``scene/frame``. Absolute paths and
            parent traversal are rejected.
        image: RGB ``uint8 [H,W,3]`` array or a source image path.
        depth: Depth ``float [H,W]`` in the unit documented by ``meta`` and the
            dataset config. NaN and Inf retain the module-level semantics.
        intrinsics: Normalized pinhole matrix ``float [3,3]``. Its first row is
            divided by image width and its second row by image height.
        meta: Additional JSON-serializable metadata. The writer supplies
            ``intrinsics``, ``width``, and ``height``.
        pose: Optional camera-to-world matrix ``float [4,4]``.
        extra_depths: Additional depth maps such as ``lidar_depth.png``. Each
            value must be ``float [H,W]``.
        segmentation: Optional integer label map ``[H,W]``.
        segmentation_labels: Optional mapping from names to integer IDs.
        copy_jpeg: Copy a JPEG source byte-for-byte instead of encoding it.

    Returns:
        Immutable data container consumed by :func:`write_sample`.
    """

    key: str
    image: np.ndarray | Path | str
    depth: np.ndarray
    intrinsics: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)
    pose: np.ndarray | None = None
    extra_depths: Mapping[str, np.ndarray] = field(default_factory=dict)
    segmentation: np.ndarray | None = None
    segmentation_labels: Mapping[str, int] | None = None
    copy_jpeg: bool = False


def normalize_intrinsics(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel-space pinhole intrinsics to normalized coordinates.

    Args:
        intrinsics: Pixel-space camera matrix ``float [3,3]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``float32 [3,3]`` matrix whose first row is divided by ``width`` and
        second row by ``height``. The last row is set to ``[0,0,1]``.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    matrix = np.asarray(intrinsics, dtype=np.float32).copy()
    if matrix.shape != (3, 3):
        raise ValueError(f"Intrinsics must have shape (3, 3), got {matrix.shape}.")
    matrix[0] /= float(width)
    matrix[1] /= float(height)
    matrix[2] = (0.0, 0.0, 1.0)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Intrinsics contain non-finite or non-positive focal values.")
    return matrix


def read_rgb(path: str | Path) -> np.ndarray:
    """Decode an image as RGB uint8.

    Args:
        path: Image path supported by OpenCV.

    Returns:
        RGB ``uint8 [H,W,3]`` image.
    """

    path = Path(path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def safe_key(value: str | Path) -> str:
    """Validate a relative sample path used in ``.index.txt``.

    Args:
        value: Relative path-like sample identifier.

    Returns:
        POSIX-form relative path.
    """

    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe sample key: {value!r}")
    return path.as_posix()


def _image_array(image: np.ndarray | Path | str) -> np.ndarray:
    """Resolve an image source into RGB uint8 ``[H,W,3]``.

    Args:
        image: RGB array or image path from a :class:`Sample`.

    Returns:
        Contiguous RGB ``uint8 [H,W,3]`` array.
    """

    if isinstance(image, (str, Path)):
        return read_rgb(image)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB image must have shape [H,W,3], got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def write_sample(root: str | Path, sample: Sample, *, jpeg_quality: int = 95) -> str:
    """Validate and serialize one processed sample.

    Args:
        root: Processed dataset root.
        sample: RGB, depth, intrinsics, and optional annotations.
        jpeg_quality: Quality used when the source image must be encoded.

    Returns:
        Relative key written below ``root``.
    """

    key = safe_key(sample.key)
    depth = np.asarray(sample.depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape [H,W], got {depth.shape} for {key}.")
    image = _image_array(sample.image)
    height, width = image.shape[:2]
    if depth.shape != (height, width):
        raise ValueError(f"RGB/depth shape mismatch for {key}: {image.shape[:2]} vs {depth.shape}.")
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise ValueError(f"Sample has no finite positive depth: {key}")

    intrinsics = np.asarray(sample.intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"Normalized intrinsics must be finite [3,3] for {key}.")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"Normalized intrinsics have non-positive focal length for {key}.")

    meta = dict(sample.meta)
    meta.update({"intrinsics": intrinsics.tolist(), "width": int(width), "height": int(height)})
    if sample.pose is not None:
        pose = np.asarray(sample.pose, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Pose must be finite [4,4] for {key}.")
        meta["pose"] = pose.tolist()

    output = Path(root) / key
    output.mkdir(parents=True, exist_ok=True)
    source = Path(sample.image) if isinstance(sample.image, (str, Path)) else None
    if sample.copy_jpeg and source is not None and source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, output / "image.jpg")
    else:
        write_image(output / "image.jpg", image, quality=int(jpeg_quality))
    write_depth(output / "depth.png", depth)

    for name, extra_depth in sample.extra_depths.items():
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError(f"Extra depth name must be a plain .png filename, got {name!r}.")
        extra_depth = np.asarray(extra_depth, dtype=np.float32)
        if extra_depth.shape != depth.shape:
            raise ValueError(f"Extra depth {name} has shape {extra_depth.shape}, expected {depth.shape}.")
        write_depth(output / name, extra_depth)

    if sample.segmentation is not None:
        segmentation = np.asarray(sample.segmentation)
        if segmentation.shape != depth.shape:
            raise ValueError(f"Segmentation has shape {segmentation.shape}, expected {depth.shape}.")
        if segmentation.dtype not in (np.uint8, np.uint16):
            maximum = int(segmentation.max()) if segmentation.size else 0
            segmentation = segmentation.astype(np.uint8 if maximum <= 255 else np.uint16)
        labels = dict(sample.segmentation_labels) if sample.segmentation_labels is not None else None
        write_segmentation(output / "segmentation.png", segmentation, labels=labels)

    write_json(output / "meta.json", meta)
    return key


def write_index(root: str | Path, entries: Iterable[str], name: str = ".index.txt") -> Path:
    """Write a sorted duplicate-free sample index through an atomic rename.

    Args:
        root: Processed dataset root.
        entries: Relative sample keys.
        name: Index filename.

    Returns:
        Final index path.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    values = sorted({safe_key(entry) for entry in entries})
    target = root / name
    temporary = root / f".{name.lstrip('.')}.tmp"
    temporary.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")
    temporary.replace(target)
    return target
