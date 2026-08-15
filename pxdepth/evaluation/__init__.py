"""Public benchmark-evaluation API for PXDepth.

The exported loader produces geometry-aware evaluation samples and the metric
function aligns raw predictions before computing depth, point-cloud, camera,
local-structure, and optional boundary measurements. Command-line orchestration
is kept in ``scripts/eval.py``.
"""

from .dataloader import EvalDataLoaderPipeline
from .metrics import compute_metrics

__all__ = ["EvalDataLoaderPipeline", "compute_metrics"]
