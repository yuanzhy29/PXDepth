"""Top-level public API for PXDepth monocular depth estimation.

Importing this package exposes the :class:`PXDepth` model without pulling
evaluation entry points into user code. The model accepts RGB
tensors and returns normalized depth plus a finite-depth probability map.
"""

from .build import build_model
from .model import PXDepth
from .registry import ENCODERS, MODELS, PREDICTORS

__all__ = [
    "PXDepth",
    "build_model",
    "MODELS",
    "ENCODERS",
    "PREDICTORS",
]
