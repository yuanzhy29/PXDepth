"""Public construction helpers for config-driven PXDepth components."""

from typing import Any, Dict

import torch.nn as nn

from .registry import MODELS

# Import built-in modules once so their registration decorators run. External
# components are imported by ``load_config`` before these builders are called.
from . import model as _model  # noqa: F401,E402


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build a model from its JSON-compatible configuration.

    Args:
        config: Model dictionary. ``type`` defaults to ``PXDepth`` for old
            public configs and checkpoints.

    Returns:
        Constructed ``torch.nn.Module`` registered in :data:`MODELS`.
    """
    config = dict(config)
    config.setdefault("type", "PXDepth")
    return MODELS.build(config)
