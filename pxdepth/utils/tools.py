"""General-purpose helpers used by evaluation modules.

The public evaluation path currently uses recursive scalar aggregation. The
small standalone interface avoids coupling metrics to command-line code.
"""

import math
from numbers import Number
from typing import Any, Dict


def key_average(items: list) -> Dict[str, Any]:
    """Recursively average matching finite numeric dictionary leaves.

    Args:
        items: List of nested dictionaries. Missing keys are ignored, nested
            dictionaries recurse, numeric NaNs are excluded, and unsupported
            leaf types are omitted.

    Returns:
        Dictionary with the union of supported keys and arithmetic means at
        numeric leaves. Returns an empty dictionary for an empty list.
    """
    if not items:
        return {}
    keys = set().union(*(item.keys() for item in items))
    result: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if not values:
            continue
        if isinstance(values[0], dict):
            result[key] = key_average(values)
        elif isinstance(values[0], Number):
            finite_values = [value for value in values if not math.isnan(float(value))]
            result[key] = sum(finite_values) / len(finite_values) if finite_values else float("nan")
    return result
