"""Load, compose, expand, and validate PXDepth JSON configurations.

The loader deliberately stays JSON based.  It adds only three conveniences that
are useful to external users: optional ``_base_`` composition, environment
variable expansion in strings, and optional module imports for custom registry
entries.  There is no framework-specific config object or runtime magic.
"""

import json
import os
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable


def _merge(base: Any, update: Any) -> Any:
    """Recursively merge one configuration value into another.

    Args:
        base: Existing value inherited from a base config.
        update: Value from the child config. Dictionaries merge recursively,
            while lists and scalar values replace ``base`` completely.

    Returns:
        A deep-copied merged value. Neither input object is mutated.
    """
    if not isinstance(base, dict) or not isinstance(update, dict):
        return deepcopy(update)
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = _merge(result[key], value) if key in result else deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    """Expand filesystem shorthand throughout a nested config structure.

    Args:
        value: Arbitrarily nested dictionaries, lists, strings, and scalar
            values loaded from JSON.

    Returns:
        A matching nested structure where every string has environment
        variables and a leading ``~`` expanded. Non-string values are retained.
    """
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _read(path: Path, stack: tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Read one JSON file and recursively compose optional base files.

    Args:
        path: JSON file to load. Relative ``_base_`` entries resolve beside it.
        stack: Internal chain of resolved paths used to detect inheritance
            cycles. Callers normally leave this empty.

    Returns:
        Merged plain dictionary before string expansion and validation.

    Raises:
        ValueError: If configs form a circular ``_base_`` dependency.
    """
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    config = json.loads(path.read_text())
    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = Path(os.path.expanduser(os.path.expandvars(str(base))))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge(merged, _read(base_path, (*stack, path)))
    return _merge(merged, config)


def import_modules(names: Iterable[str]) -> None:
    """Import extension modules so their registry decorators execute.

    Args:
        names: Iterable of importable Python module names, such as
            ``my_project.components``.

    Returns:
        ``None``. Imports are performed for their registration side effects.
    """
    for name in names:
        import_module(str(name))


def validate_config(config: Dict[str, Any], kind: str | None = None) -> None:
    """Fail early for missing or structurally invalid public configuration fields.

    Args:
        config: Fully composed configuration dictionary.
        kind: Optional ``'eval'`` validation profile.

    Returns:
        ``None``. A descriptive ``ValueError`` is raised for invalid structure.
    """
    if not isinstance(config, dict):
        raise ValueError("The config root must be a JSON object.")
    if kind not in {None, "eval"}:
        raise ValueError(f"Unsupported config kind: {kind!r}")
    if kind == "eval" and not config:
        raise ValueError("Evaluation config must contain at least one benchmark.")


def load_config(path: str | Path, kind: str | None = None) -> Dict[str, Any]:
    """Load a resolved config and import optional external extension modules.

    Args:
        path: JSON config path. Relative ``_base_`` paths resolve beside this file.
        kind: Optional validation profile passed to :func:`validate_config`.

    Returns:
        Plain nested dictionaries/lists suitable for JSON serialization. The
        optional top-level ``imports`` list is retained for external registry
        extensions.
    """
    config = _expand(_read(Path(path)))
    imports = config.get("imports", [])
    if isinstance(imports, str):
        imports = [imports]
    import_modules(imports)
    validate_config(config, kind=kind)
    return config
