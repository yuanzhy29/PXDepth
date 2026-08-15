"""NumPy visualization mappings for geometry and semantic predictions.

Depth, disparity, normals, labels, and scalar error arrays are normalized with
explicit valid masks and converted to display-ready RGB images. Unknown or
infinite regions use stable colors shared by training and evaluation outputs.
"""

from typing import Optional, Tuple

import numpy as np
import matplotlib


def colorize_depth(depth: np.ndarray, mask: Optional[np.ndarray] = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    """Colorize positive depth through inverse-depth ordering.

    Args:
        depth: Depth map ``[H,W]``.
        mask: Optional boolean validity mask ``[H,W]``.
        normalize: Quantile-normalize disparity before colormap lookup.
        cmap: Matplotlib colormap name.

    Returns:
        RGB uint8 visualization ``[H,W,3]``; invalid pixels are black.
    """
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_depth_affine(depth: np.ndarray, mask: Optional[np.ndarray] = None, cmap: str = 'Spectral') -> np.ndarray:
    """Colorize depth after direct affine quantile normalization.

    Args:
        depth: Scalar depth-like map ``[H,W]``.
        mask: Optional boolean validity mask ``[H,W]``.
        cmap: Matplotlib colormap name.

    Returns:
        RGB uint8 visualization ``[H,W,3]``.
    """
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_depth_shifted_disparity(
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True,
    cmap: str = 'Spectral',
    eps: float = 1.0,
) -> np.ndarray:
    """Colorize depth using disparity shifted by the nearest finite value.

    Args:
        depth: Depth-like map ``[H,W]`` that may include negative values.
        mask: Optional boolean validity mask ``[H,W]``.
        normalize: Quantile-normalize shifted disparity.
        cmap: Matplotlib colormap name.
        eps: Positive offset preventing division by zero at minimum depth.

    Returns:
        RGB uint8 visualization ``[H,W,3]``.
    """
    if mask is not None:
        depth = np.where(mask, depth, np.nan)
    else:
        depth = np.where(np.isfinite(depth), depth, np.nan)
    if not np.isfinite(depth).any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    min_depth = np.nanmin(depth)
    disp = 1.0 / (depth - min_depth + eps)
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        if max_disp > min_disp:
            disp = (disp - min_disp) / (max_disp - min_disp)
        else:
            disp = np.zeros_like(disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_disparity(disparity: np.ndarray, mask: Optional[np.ndarray] = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    """Colorize a disparity map with optional quantile normalization.

    Args:
        disparity: Disparity array ``[H,W]``.
        mask: Optional boolean validity mask ``[H,W]``.
        normalize: Normalize the 0.1%--99.9% quantile interval.
        cmap: Matplotlib colormap name.

    Returns:
        RGB uint8 visualization ``[H,W,3]``.
    """
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)

    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_segmentation(segmentation: np.ndarray, cmap: str = 'Set1') -> np.ndarray:
    """Assign repeating categorical colors to integer segmentation IDs.

    Args:
        segmentation: Integer label map ``[H,W]``.
        cmap: Matplotlib categorical colormap name.

    Returns:
        RGB uint8 visualization ``[H,W,3]``.
    """
    colored = matplotlib.colormaps[cmap]((segmentation % 20) / 20)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_normal(normal: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Map camera-space unit normals to conventional RGB colors.

    Args:
        normal: Normal map ``[H,W,3]`` with components near ``[-1,1]``.
        mask: Optional boolean validity mask ``[H,W]``.

    Returns:
        RGB uint8 normal visualization ``[H,W,3]``.
    """
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def colorize_error_map(error_map: np.ndarray, mask: Optional[np.ndarray] = None, cmap: str = 'plasma', value_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Colorize a scalar error map over an explicit or observed value range.

    Args:
        error_map: Scalar error array ``[H,W]``.
        mask: Optional boolean validity mask ``[H,W]``.
        cmap: Matplotlib colormap name.
        value_range: Optional ``(minimum,maximum)`` normalization bounds.

    Returns:
        RGB uint8 error visualization ``[H,W,3]``.
    """
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map
