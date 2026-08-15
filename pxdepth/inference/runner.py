"""Timed raw PXDepth forward path used by evaluation tools.

This module applies the selected fixed-size or equal-area preprocessing,
measures only model-forward latency, and restores predictions to input
resolution. It intentionally returns only genuine network outputs rather than
performing metric alignment or estimating camera intrinsics.
"""

import time
from typing import Dict, Optional, Tuple

import torch

from ..model import PXDepth
from .resize import resize_image, resize_map


def synchronize(device: torch.device) -> None:
    """Synchronize pending CUDA work before or after timing model forward.

    Args:
        device: Model device. CPU and other devices require no action.

    Returns:
        ``None``.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_raw(
    model: PXDepth,
    image: torch.Tensor,
    input_size: Optional[Tuple[int, int]] = (1022, 770),
    resize_by_area: bool = True,
    use_fp16: bool = False,
    use_fp32: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run raw model forward and return outputs at the original image size.

    Only ``model.forward`` is included in ``inference_time``. Resizing,
    synchronization overhead, and output packaging are excluded. No GT depth
    alignment or camera-intrinsics prediction is performed.

    Args:
        model: Evaluation-mode :class:`PXDepth` model.
        image: RGB tensor ``[3,H,W]`` or ``[B,3,H,W]`` in ``[0,1]``.
        input_size: Exact/reference tuple ``(width,height)``. Defaults to
            ``(1022,770)``.
        resize_by_area: Preserve aspect ratio at ``input_size`` area. Enabled
            by default.
        use_fp16: Use FP16 for attention-heavy model regions.
        use_fp32: Force full-precision model execution.

    Returns:
        Dictionary with raw normalized log-depth ``depth_affine_invariant``
        ``[B,H,W]``, ``depth_affine_space='log'``, boolean ``mask`` ``[B,H,W]``,
        and scalar forward time. The leading batch dimension is removed for
        unbatched input.
    """
    image, original_size = resize_image(image, input_size, resize_by_area, model.patch_size)
    batched = image.ndim == 4
    model_input = image if batched else image.unsqueeze(0)
    model_input = model_input.to(device=model.device, dtype=torch.float32)

    synchronize(model.device)
    start = time.perf_counter()
    output = model.forward(model_input, use_fp16=use_fp16, use_fp32=use_fp32)
    synchronize(model.device)
    elapsed = time.perf_counter() - start

    depth = resize_map(output["depth"], original_size)
    mask = resize_map(output["mask"], original_size, is_mask=True)

    pred = {
        "depth_affine_invariant": depth,
        "depth_affine_space": "log",
        "mask": mask,
        "inference_time": elapsed,
    }
    if not batched:
        pred = {key: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value for key, value in pred.items()}
    return pred
