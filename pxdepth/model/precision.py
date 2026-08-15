"""Precision-policy helpers shared by PXDepth model components.

The context managers define where full precision is required and where
attention-heavy regions may use FP16 or BF16 autocast. Centralizing this policy
keeps numerical behavior consistent between evaluation and public inference
entry points.
"""

from contextlib import nullcontext
from typing import Optional

import torch


def reduced_precision(device: torch.device, dtype: Optional[torch.dtype] = torch.bfloat16):
    """Create an autocast context for attention-heavy model regions.

    Args:
        device: Device on which enclosed tensor operations execute.
        dtype: CUDA autocast dtype, normally BF16 or FP16. ``None`` requests
            full precision.

    Returns:
        Context manager enabling CUDA autocast when applicable, otherwise a
        no-op context manager.
    """
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return nullcontext()


def full_precision(device: torch.device):
    """Create a context that disables an enclosing autocast region.

    Args:
        device: Device type used to construct the autocast context.

    Returns:
        Context manager that executes enclosed operators in their explicit
        dtypes, or a no-op context on unsupported devices.
    """
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def inference_dtype(use_fp16: bool = False, use_fp32: bool = False) -> Optional[torch.dtype]:
    """Resolve public inference precision flags to an autocast dtype.

    Args:
        use_fp16: Select FP16 attention and encoder execution.
        use_fp32: Disable reduced precision. Mutually exclusive with FP16.

    Returns:
        ``torch.float16`` for FP16, ``None`` for FP32, and
        ``torch.bfloat16`` for the default path.
    """
    if use_fp16 and use_fp32:
        raise ValueError("use_fp16 and use_fp32 are mutually exclusive")
    if use_fp32:
        return None
    if use_fp16:
        return torch.float16
    return torch.bfloat16
