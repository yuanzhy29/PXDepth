"""I/O for the processed benchmark format used by PXDepth evaluation.

Benchmark samples store RGB as JPEG or PNG, depth as a logarithmically encoded
16-bit PNG, optional semantic labels as PNG metadata, and camera information in
JSON. The matching writers are shared by the released evaluation-dataset
converters so their outputs can be consumed directly by the benchmark loader.
"""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, PngImagePlugin


PathOrBinary = Union[str, os.PathLike, IO[bytes]]
JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def _read_bytes(path: PathOrBinary) -> bytes:
    """Read encoded data from a filesystem path or binary stream.

    Args:
        path: File path or a binary stream exposing ``read()``.

    Returns:
        Encoded file contents as ``bytes``.
    """
    if isinstance(path, (str, os.PathLike)):
        return Path(path).read_bytes()
    return path.read()


def read_image(path: PathOrBinary) -> np.ndarray:
    """Decode an RGB image.

    Args:
        path: JPEG/PNG path or readable binary stream.

    Returns:
        RGB uint8 array with shape ``[H, W, 3]``.

    Raises:
        ValueError: If OpenCV cannot decode the input.
    """
    image = cv2.imdecode(np.frombuffer(_read_bytes(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path: Union[str, os.PathLike, IO[bytes]], image: np.ndarray, quality: int = 95) -> None:
    """Encode an RGB image as JPEG.

    Args:
        path: Destination path or writable binary stream.
        image: RGB uint8 array with shape ``[H, W, 3]``.
        quality: JPEG quality passed to OpenCV.
    """
    encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )[1].tobytes()
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_bytes(encoded)
    else:
        path.write(encoded)


def read_depth(path: PathOrBinary) -> np.ndarray:
    """Decode the logarithmic 16-bit PNG depth representation.

    Args:
        path: Encoded depth PNG path or readable binary stream. The PNG must
            contain ``near`` and ``far`` text metadata.

    Returns:
        Float32 depth array ``[H, W]``. Code 0 maps to NaN, code 65535 maps to
        positive infinity, and codes 1 through 65534 map to finite depth.
    """
    image = Image.open(io.BytesIO(_read_bytes(path)))
    near = float(image.info["near"])
    far = float(image.info["far"])
    encoded = np.asarray(image)
    mask_nan = encoded == 0
    mask_inf = encoded == 65535
    value = (encoded.astype(np.float32) - 1.0) / 65533.0
    depth = near ** (1.0 - value) * far**value
    if "unit" in image.info:
        depth *= float(image.info["unit"])
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth


def write_depth(
    path: Union[str, os.PathLike, IO[bytes]],
    depth: np.ndarray,
    max_range: float = 1e5,
    compression_level: int = 7,
) -> None:
    """Encode depth as logarithmic 16-bit PNG with NaN/Inf sentinels.

    Args:
        path: Destination path or writable binary stream.
        depth: Float depth array ``[H, W]``. NaN stores unknown geometry and
            positive infinity stores known infinite geometry.
        max_range: Maximum finite ``far / near`` encoding ratio.
        compression_level: PNG compression level from zero through nine.
    """
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    mask_nan = np.isnan(depth)
    mask_inf = np.isinf(depth)
    if not np.any(finite):
        raise ValueError("Depth encoding requires at least one finite value.")
    near = max(float(depth[finite].min()), 1e-5)
    far = max(near * 1.1, min(float(depth[finite].max()), near * float(max_range)))
    clipped = np.nan_to_num(depth, nan=near, posinf=far, neginf=near).clip(near, far)
    encoded = 1 + np.round(np.log(clipped / near) / np.log(far / near) * 65533).astype(np.uint16)
    encoded[mask_nan] = 0
    encoded[mask_inf] = 65535
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("near", str(near))
    pnginfo.add_text("far", str(far))
    Image.fromarray(encoded).save(path, pnginfo=pnginfo, compress_level=int(compression_level))


def read_segmentation(path: PathOrBinary) -> Tuple[np.ndarray, Optional[Dict[str, int]]]:
    """Decode an integer segmentation PNG and its optional label mapping.

    Args:
        path: Segmentation PNG path or readable binary stream.

    Returns:
        A pair of ``mask`` and ``labels``. ``mask`` has shape ``[H, W]`` and
        retains the PNG integer dtype. ``labels`` maps names to IDs when the
        PNG contains label metadata, otherwise it is ``None``.
    """
    image = Image.open(io.BytesIO(_read_bytes(path)))
    labels = json.loads(image.info["labels"]) if "labels" in image.info else None
    return np.asarray(image), labels


def write_segmentation(
    path: Union[str, os.PathLike, IO[bytes]],
    mask: np.ndarray,
    labels: Optional[Dict[str, int]] = None,
    compression_level: int = 7,
) -> None:
    """Write an integer segmentation PNG and optional label mapping.

    Args:
        path: Destination path or writable binary stream.
        mask: Integer label array ``[H, W]`` with uint8 or uint16 dtype.
        labels: Optional mapping from label names to integer IDs.
        compression_level: PNG compression level from zero through nine.
    """
    mask = np.asarray(mask)
    if mask.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Segmentation must be uint8 or uint16, got {mask.dtype}.")
    pnginfo = PngImagePlugin.PngInfo()
    if labels is not None:
        pnginfo.add_text("labels", json.dumps(labels, ensure_ascii=True, separators=(",", ":")))
    Image.fromarray(mask).save(path, pnginfo=pnginfo, compress_level=int(compression_level))


def read_json(path: Union[str, os.PathLike, IO[str]]) -> JsonValue:
    """Parse JSON from a path or readable text stream.

    Args:
        path: JSON path or text stream exposing ``read()``.

    Returns:
        Parsed JSON-compatible Python value.
    """
    text = Path(path).read_text() if isinstance(path, (str, os.PathLike)) else path.read()
    return json.loads(text)


def write_json(path: Union[str, os.PathLike, IO[str]], content: JsonValue) -> None:
    """Serialize a JSON-compatible value.

    Args:
        path: Destination path or writable text stream.
        content: JSON-compatible scalar, list, or dictionary.
    """
    text = json.dumps(content)
    if isinstance(path, (str, os.PathLike)):
        Path(path).write_text(text)
    else:
        path.write(text)


__all__ = [
    "read_depth",
    "read_image",
    "read_json",
    "read_segmentation",
    "write_depth",
    "write_image",
    "write_json",
    "write_segmentation",
]
