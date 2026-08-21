"""Metric-scale inference helpers used by :class:`PXDepth`.

The network predicts normalized log-depth. This module keeps reference-model
loading, low-resolution log-space alignment, camera reconstruction, and mask
application outside the architecture file while preserving the released
``model.infer`` behavior.
"""

from numbers import Number
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils3d

from ..utils.alignment import align_depth_affine
from .precision import full_precision


def _reference_model(model: nn.Module) -> nn.Module:
    """Load and cache the optional MoGe-2 reference model.

    Args:
        model: PXDepth-like module exposing ``device``, ``dtype``, and a
            mutable ``_reference_model`` attribute.

    Returns:
        An evaluation-mode MoGe-2 module placed on the same device and storage
        dtype as ``model``.

    Raises:
        RuntimeError: If the optional MoGe-2 dependency is unavailable.
    """
    if model._reference_model is None:
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise RuntimeError(
                "Metric-scale visualization requires MoGe-2. Install the optional `reference` dependencies "
                "or pass gt_depth and intrinsics to infer()."
            ) from exc
        reference = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        model._reference_model = reference.to(device=model.device, dtype=model.dtype).eval()
    return model._reference_model


def _patch_size(model: nn.Module) -> Optional[int]:
    """Resolve a reference model's scalar image patch size.

    Args:
        model: Reference model potentially exposing ``patch_size`` directly or
            through ``encoder.backbone``.

    Returns:
        A positive integer patch size, or ``None`` when it cannot be resolved.
    """
    patch = getattr(model, "patch_size", None)
    if patch is None:
        patch = getattr(getattr(getattr(model, "encoder", None), "backbone", None), "patch_size", None)
    if isinstance(patch, (tuple, list)):
        patch = patch[0]
    return int(patch) if isinstance(patch, Number) and int(patch) > 0 else None


def _prepare_reference_image(image: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize an RGB batch to satisfy a reference model's patch constraint.

    Args:
        image: RGB tensor ``[B, 3, H, W]``.
        model: Reference depth model inspected for its patch size.

    Returns:
        A pair containing the bilinearly resized RGB tensor and original
        ``(H, W)``. The input is returned unchanged when already divisible.
    """
    patch = _patch_size(model)
    size = tuple(image.shape[-2:])
    if patch is None:
        return image, size
    height, width = size
    target = (
        height if height % patch == 0 else max(patch, height // patch * patch),
        width if width % patch == 0 else max(patch, width // patch * patch),
    )
    if target == size:
        return image, size
    return F.interpolate(image, target, mode="bilinear", align_corners=False), size


@torch.inference_mode()
def infer(
    model: nn.Module,
    image: torch.Tensor,
    gt_depth: Optional[torch.Tensor] = None,
    intrinsics: Optional[torch.Tensor] = None,
    fov_x: Optional[Union[Number, torch.Tensor]] = None,
    ref_image: Optional[torch.Tensor] = None,
    apply_mask: bool = True,
    use_fp16: bool = True,
    use_fp32: bool = False,
) -> Dict[str, torch.Tensor]:
    """Recover aligned depth, validity, camera intrinsics, and 3D points.

    Raw normalized log-depth is affine-aligned in log space to ``gt_depth``
    when supplied, otherwise to a lazily loaded MoGe-2 reference. Alignment is
    estimated from a masked-nearest 64x64 representation, matching the released
    evaluation and visualization behavior.

    Args:
        model: PXDepth-like module exposing ``forward``, ``device``,
            ``dtype``, and ``mask_threshold``.
        image: RGB tensor ``[3,H,W]`` or ``[B,3,H,W]`` in ``[0,1]``.
        gt_depth: Optional reference depth ``[H,W]`` or ``[B,H,W]``. Finite
            positive values define log-space alignment.
        intrinsics: Optional normalized camera matrix ``[3,3]`` or batch
            ``[B,3,3]`` corresponding to the reference depth.
        fov_x: Optional horizontal field of view in degrees, scalar or ``[B]``.
        ref_image: Optional original-resolution RGB input used only by MoGe-2.
        apply_mask: Replace invalid predicted depth and points with infinity.
        use_fp16: Use FP16 in attention-heavy model regions.
        use_fp32: Force full precision and disable reduced-precision autocast.

    Returns:
        Dictionary containing aligned ``depth`` ``[B,H,W]``, normalized
        log-depth ``depth_log1p_affine_invariant`` ``[B,H,W]``, boolean ``mask``
        ``[B,H,W]``, ``points`` ``[B,H,W,3]``, normalized ``intrinsics``
        ``[B,3,3]``, and horizontal ``fov_x`` ``[B]``. The leading batch
        dimension is removed when ``image`` is unbatched.
    """
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    image = image.to(device=model.device, dtype=model.dtype)
    if ref_image is not None and ref_image.ndim == 3:
        ref_image = ref_image.unsqueeze(0)
    if ref_image is not None:
        ref_image = ref_image.to(device=model.device, dtype=model.dtype)
    if gt_depth is not None and gt_depth.ndim == 2:
        gt_depth = gt_depth.unsqueeze(0)
    if gt_depth is not None:
        gt_depth = gt_depth.to(device=model.device, dtype=torch.float32)
    if intrinsics is not None and intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics is not None:
        intrinsics = intrinsics.to(device=model.device, dtype=torch.float32)

    height, width = image.shape[-2:]
    aspect = width / height
    output = model.forward(image, use_fp16=use_fp16, use_fp32=use_fp32)

    with full_precision(model.device):
        pred = output["depth"].float()
        mask = output["mask"].float()
        ref_depth, ref_intrinsics, ref_fov = gt_depth, intrinsics, fov_x
        if ref_depth is None:
            reference = _reference_model(model)
            reference_input = image if ref_image is None else ref_image
            reference_input, reference_size = _prepare_reference_image(reference_input, reference)
            ref = reference.infer(reference_input, apply_mask=True, use_fp16=use_fp16 and not use_fp32)
            ref_depth = ref["depth"].float()
            if ref_depth.ndim == 2:
                ref_depth = ref_depth.unsqueeze(0)
            if ref_depth.shape[-2:] != reference_size:
                ref_depth = F.interpolate(ref_depth.unsqueeze(1), reference_size, mode="nearest").squeeze(1)
            ref_intrinsics = ref.get("intrinsics")
            ref_fov = ref.get("fov_x")
            if ref_intrinsics is not None:
                ref_intrinsics = ref_intrinsics.float()
            if ref_fov is not None:
                ref_fov = ref_fov.float()
        if ref_depth.shape[-2:] != pred.shape[-2:]:
            ref_depth = F.interpolate(ref_depth.unsqueeze(1), pred.shape[-2:], mode="nearest").squeeze(1)

        ref_valid = torch.isfinite(ref_depth) & (ref_depth > 0)
        ref_log = torch.where(ref_valid, torch.log1p(ref_depth), 0.0)
        scale = torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype)
        shift = torch.zeros_like(scale)
        valid = torch.isfinite(pred) & ref_valid
        for index in range(pred.shape[0]):
            low_mask, nearest = utils3d.pt.masked_nearest_resize(
                mask=valid[index], size=(64, 64), return_index=True
            )
            if not low_mask.any():
                continue
            pred_low = pred[index][nearest][low_mask]
            ref_log_low = ref_log[index][nearest][low_mask]
            ref_depth_low = ref_depth[index][nearest][low_mask]
            a, b = align_depth_affine(
                pred_low.unsqueeze(0),
                ref_log_low.unsqueeze(0),
                (1.0 / ref_depth_low.clamp_min(1e-5)).unsqueeze(0),
            )
            scale[index], shift[index] = a.squeeze(0), b.squeeze(0)
        depth = torch.expm1(scale[:, None, None] * pred + shift[:, None, None])

        if ref_intrinsics is None:
            if ref_fov is None:
                fx = torch.ones(depth.shape[0], device=depth.device)
                fy = torch.ones_like(fx)
                ref_fov = 2.0 * torch.atan(0.5 / fx).rad2deg()
            else:
                ref_fov = torch.as_tensor(ref_fov, device=depth.device, dtype=depth.dtype)
                focal = aspect / (1.0 + aspect**2) ** 0.5 / torch.tan(torch.deg2rad(ref_fov / 2.0))
                if focal.ndim == 0:
                    focal = focal[None].expand(depth.shape[0])
                fx = focal / 2.0 * (1.0 + aspect**2) ** 0.5 / aspect
                fy = focal / 2.0 * (1.0 + aspect**2) ** 0.5
            ref_intrinsics = utils3d.pt.intrinsics_from_focal_center(
                fx,
                fy,
                torch.tensor(0.5, device=depth.device),
                torch.tensor(0.5, device=depth.device),
            )
        else:
            ref_fov = 2.0 * torch.atan(0.5 / ref_intrinsics[..., 0, 0]).rad2deg()

        mask_binary = (mask > model.mask_threshold) & torch.isfinite(depth) & (depth > 0)
        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=ref_intrinsics)
        if apply_mask:
            depth = torch.where(mask_binary, depth, torch.inf)
            points = torch.where(mask_binary[..., None], points, torch.inf)
        result = {
            "depth": depth,
            "depth_log1p_affine_invariant": pred,
            "mask": mask_binary,
            "points": points,
            "intrinsics": ref_intrinsics,
            "fov_x": ref_fov,
        }
    return {key: value.squeeze(0) for key, value in result.items()} if squeeze else result
