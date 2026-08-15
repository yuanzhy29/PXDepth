"""Load model-only checkpoints written in the public PXDepth format."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, IO, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


ModelT = TypeVar("ModelT", bound=nn.Module)


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge model-constructor overrides into a copied config.

    Args:
        base: Original nested model configuration. The dictionary is not
            modified.
        update: User overrides. Nested dictionaries update individual fields,
            while lists and scalar values replace their counterparts.

    Returns:
        A new merged dictionary suitable for constructing the model.
    """
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_pretrained(
    model_class: Type[ModelT],
    path_or_repo: Union[str, Path, IO[bytes]],
    model_kwargs: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    **hf_kwargs: Any,
) -> ModelT:
    """Construct a model from a local or Hugging Face ``model.pt`` file.

    Args:
        model_class: Model class whose constructor accepts the canonical public
            configuration.
        path_or_repo: Existing local checkpoint path, binary file object, or
            Hugging Face model repository identifier.
        model_kwargs: Optional nested constructor overrides. Nested encoder or
            predictor fields are merged without discarding sibling settings.
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`. Published
            checkpoints should normally use the default strict loading.
        **hf_kwargs: Extra arguments forwarded to ``hf_hub_download`` when
            ``path_or_repo`` is a repository identifier.

    Returns:
        An initialized model instance on CPU.
    """
    path = Path(path_or_repo) if isinstance(path_or_repo, (str, Path)) else None
    if path is not None and path.exists():
        checkpoint_path: Union[Path, IO[bytes]] = path
    elif isinstance(path_or_repo, str):
        checkpoint_path = Path(
            hf_hub_download(path_or_repo, repo_type="model", filename="model.pt", **hf_kwargs)
        )
    else:
        checkpoint_path = path_or_repo

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("PXDepth checkpoints must contain 'model_config' and 'model'.")
    model_config = deepcopy(checkpoint["model_config"])
    model_type = model_config.pop("type", None)
    if model_type != model_class.__name__:
        raise ValueError(
            f"Expected a {model_class.__name__} checkpoint, got model type {model_type!r}."
        )
    if model_kwargs:
        model_config = _merge(model_config, model_kwargs)
    model = model_class(**model_config)
    model.load_state_dict(checkpoint["model"], strict=strict)
    return model
