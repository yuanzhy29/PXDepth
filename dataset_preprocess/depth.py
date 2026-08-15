"""Depth decoding and validity normalization shared by all datasets.

The processed format distinguishes finite observations, unknown NaN pixels,
and explicitly known infinite geometry. Dataset adapters describe native units
and sentinels with :class:`DepthData`; :func:`prepare_depth` applies the policy
once before serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthData:
    """Raw depth values plus the source dataset's validity semantics.

    Args:
        values: Raw numeric depth array with shape ``[H,W]``.
        scale: Multiplier applied before range checks, for example ``0.001``
            when the source stores millimeters.
        invalid: Optional boolean ``[H,W]`` mask for unknown pixels. They are
            serialized as NaN and ignored by evaluation metrics.
        infinite: Optional boolean ``[H,W]`` mask for explicitly known
            infinite geometry such as renderer sky. It is serialized as Inf
            and takes precedence over ``invalid``.
        min_depth: Optional inclusive lower bound after applying ``scale``.
            Non-positive values always become NaN.
        max_depth: Optional inclusive upper bound after applying ``scale``.
        min_valid_ratio: Required fraction of finite positive pixels. A frame
            below this ratio is rejected before entering ``.index.txt``.

    Returns:
        Immutable source-depth description consumed by
        :func:`prepare_depth`.
    """

    values: np.ndarray
    scale: float = 1.0
    invalid: np.ndarray | None = None
    infinite: np.ndarray | None = None
    min_depth: float | None = 0.0
    max_depth: float | None = None
    min_valid_ratio: float = 0.0


def prepare_depth(value: DepthData | np.ndarray) -> np.ndarray:
    """Convert source depth and masks to the common three-state depth map.

    Args:
        value: :class:`DepthData` or a plain numeric ``[H,W]`` array. Plain
            arrays use the default rule where positive finite values are kept
            and every non-positive or non-finite value becomes NaN.

    Returns:
        Independent ``float32 [H,W]`` array. Finite positive values are known
        measurements, NaN is unknown geometry, and Inf is known infinity.

    Raises:
        ValueError: If values or masks have invalid shapes, the scale is not
            positive, or finite support is below ``min_valid_ratio``.
    """

    data = value if isinstance(value, DepthData) else DepthData(np.asarray(value))
    depth = np.asarray(data.values, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape [H,W], got {depth.shape}.")
    if not np.isfinite(data.scale) or data.scale <= 0:
        raise ValueError(f"Depth scale must be finite and positive, got {data.scale}.")
    depth = depth.copy() * np.float32(data.scale)

    invalid = ~np.isfinite(depth) | (depth <= 0)
    if data.invalid is not None:
        source_invalid = np.asarray(data.invalid, dtype=bool)
        if source_invalid.shape != depth.shape:
            raise ValueError(f"Invalid mask has shape {source_invalid.shape}, expected {depth.shape}.")
        invalid |= source_invalid
    finite = np.isfinite(depth)
    if data.min_depth is not None:
        invalid |= finite & (depth < float(data.min_depth))
    if data.max_depth is not None:
        invalid |= finite & (depth > float(data.max_depth))

    infinite = np.zeros(depth.shape, dtype=bool)
    if data.infinite is not None:
        infinite = np.asarray(data.infinite, dtype=bool)
        if infinite.shape != depth.shape:
            raise ValueError(f"Infinity mask has shape {infinite.shape}, expected {depth.shape}.")

    depth[invalid] = np.nan
    depth[infinite] = np.inf
    ratio = float((np.isfinite(depth) & (depth > 0)).mean())
    if ratio < float(data.min_valid_ratio):
        raise ValueError(f"Valid depth ratio {ratio:.6f} is below {data.min_valid_ratio:.6f}.")
    return depth


def read_depth_image(path: str | Path, *, scale: float = 1.0, channel: int = 0) -> np.ndarray:
    """Decode a PNG, TIFF, or EXR depth image without guessing sentinels.

    Args:
        path: Depth image path supported by the active OpenCV build.
        scale: Scalar multiplied into decoded values.
        channel: Selected channel when decoding returns ``[H,W,C]``.

    Returns:
        Raw ``float32 [H,W]`` depth. The dataset adapter still describes which
        source values are invalid or known infinity.
    """

    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Cannot decode depth image: {path}")
    raw = np.asarray(raw)
    if raw.ndim == 3:
        if not 0 <= channel < raw.shape[-1]:
            raise ValueError(f"Depth channel {channel} is unavailable in shape {raw.shape}.")
        raw = raw[..., channel]
    if raw.ndim != 2:
        raise ValueError(f"Depth must decode to [H,W], got {raw.shape}: {path}")
    return raw.astype(np.float32) * np.float32(scale)
