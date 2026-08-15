"""Public preprocessing and raw-forward helpers for PXDepth inference.

The package centralizes fixed-size and equal-area image resizing, patch-grid
alignment, output restoration, and timed model execution. These helpers return
raw normalized predictions and avoid embedding benchmark-specific alignment in
the model forward path.
"""

from .runner import predict_raw
from .resize import area_size, area_size_from_area, parse_size, patch_size, resize_image, resize_map

__all__ = [
    "predict_raw",
    "area_size",
    "area_size_from_area",
    "parse_size",
    "patch_size",
    "resize_image",
    "resize_map",
]
