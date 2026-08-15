"""Core PXDepth architecture and stable public model API.

The module connects the Global Context Encoder to the Pixel-Space Depth
Predictor and defines raw forward computation. Checkpoint translation and
metric-scale inference live in focused helper modules, while their familiar
``from_pretrained`` and ``infer`` entry points remain methods on this class.
"""

from pathlib import Path
from typing import Any, Dict, IO, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Global_Context_Encoder import GlobalContextEncoder
from .Pixel_Space_Depth_Predictor import PixelSpaceDepthPredictor
from .checkpoint import load_pretrained
from .inference import infer as infer_model
from .precision import full_precision, inference_dtype, reduced_precision
from ..registry import ENCODERS, MODELS, PREDICTORS


@MODELS.register()
class PXDepth(nn.Module):
    """Complete PXDepth monocular depth model.

    A Global Context Encoder extracts semantic patch features and a Pixel-Space
    Depth Predictor estimates full-resolution normalized log-depth together
    with a finite-depth probability. ``forward`` exposes raw network outputs,
    while ``infer`` aligns them to a GT or MoGe-2 reference for metric-scale
    visualization and point-cloud reconstruction.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, Dict[str, Any]],
        predictor: Union[nn.Module, Dict[str, Any]],
        remap_output: str = "linear",
        mask_threshold: float = 0.5,
    ) -> None:
        """Construct the encoder and CM-PiT pixel predictor.

        Args:
            encoder: Encoder module or registry config.
            predictor: Pixel predictor module or registry config. Its context
                patch size and channel width default to the encoder contract.
            remap_output: Output remapping applied to normalized log-depth.
                The released model uses ``'linear'``.
            mask_threshold: Probability threshold used by :meth:`infer`.

        Returns:
            ``None``. Model modules and ImageNet normalization buffers are
            registered on the instance.
        """
        super().__init__()
        if remap_output not in {"linear", "elu"}:
            raise ValueError(f"Unsupported remap_output: {remap_output}")

        self.remap_output = remap_output
        self.mask_threshold = float(mask_threshold)
        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        else:
            encoder_config = dict(encoder)
            encoder_config.setdefault("type", "GlobalContextEncoder")
            self.encoder = ENCODERS.build(encoder_config)
        if not hasattr(self.encoder, "patch_size"):
            raise TypeError("The encoder must expose an integer patch_size attribute.")
        self.patch_size = self.encoder.patch_size
        self.p_enc = self.patch_size
        dim_ctx = getattr(self.encoder, "dim_out", None)
        if isinstance(predictor, nn.Module):
            self.predictor = predictor
        else:
            predictor_config = dict(predictor)
            predictor_config.setdefault("type", "PixelSpaceDepthPredictor")
            predictor_config.setdefault("in_channels", 3)
            predictor_config.setdefault("ctx_patch_size", self.patch_size)
            if dim_ctx is not None:
                predictor_config.setdefault("dim_ctx", int(dim_ctx))
            self.predictor = PREDICTORS.build(predictor_config)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._reference_model: Optional[nn.Module] = None

    @property
    def device(self) -> torch.device:
        """Return the device hosting PXDepth learnable parameters.

        No inputs are required. The value is inferred from the model's first
        parameter and is used when moving inference inputs or reference models.

        Returns:
            ``torch.device`` for the current model placement.
        """
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the storage dtype of PXDepth learnable parameters.

        No inputs are required. This reports parameter storage, which is
        independent from local autocast contexts used inside attention.

        Returns:
            ``torch.dtype`` of the model's first parameter.
        """
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        **hf_kwargs: Any,
    ) -> "PXDepth":
        """Create a model from a local or Hugging Face ``model.pt`` checkpoint.

        Args:
            path_or_repo: Local checkpoint path, binary file object, or Hugging
                Face model repository identifier.
            model_kwargs: Optional constructor overrides applied after reading
                ``model_config`` from the checkpoint.
            strict: Forwarded to ``load_state_dict``. Published checkpoints
                should use the default exact matching.
            **hf_kwargs: Additional keyword arguments forwarded to
                ``huggingface_hub.hf_hub_download`` for remote repositories.

        Returns:
            Initialized :class:`PXDepth` instance on CPU.
        """
        return load_pretrained(
            cls,
            path_or_repo,
            model_kwargs=model_kwargs,
            strict=strict,
            **hf_kwargs,
        )

    def init_weights(self) -> None:
        """Initialize the Global Context Encoder from official DINOv2 weights.

        Predictor parameters retain the initialization created by their own
        module constructors.

        Returns:
            ``None``. Encoder parameters are updated in place.
        """
        self.encoder.init_weights()

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing in both encoder and predictor.

        This reduces saved activation memory during backward at the cost of
        recomputing transformer blocks.

        Returns:
            ``None``. Child module runtime behavior is updated in place.
        """
        self.encoder.enable_gradient_checkpointing()
        self.predictor.enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self) -> None:
        """Enable the optimized SDPA attention path in the DINOv2 backbone.

        Decoder CM-PiT attention already uses PyTorch SDPA directly and is not
        modified by this method.

        Returns:
            ``None``. Encoder attention modules are wrapped in place.
        """
        self.encoder.enable_pytorch_native_sdpa()

    def _remap(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply the configured output activation to raw depth predictions.

        Args:
            depth: Raw normalized-depth tensor with arbitrary batch/spatial
                shape, normally ``[B, H, W]``.

        Returns:
            Tensor with the same shape. The released ``linear`` setting returns
            the input unchanged.
        """
        return F.elu(depth) if self.remap_output == "elu" else depth

    def forward(
        self,
        image: torch.Tensor,
        use_fp16: bool = False,
        use_fp32: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the network without metric-scale alignment.

        Args:
            image: RGB tensor ``[B, 3, H, W]`` with values in ``[0, 1]``. ``H``
                and ``W`` must be divisible by the encoder patch size.
            use_fp16: Run attention-heavy encoder and predictor regions under
                FP16 autocast.
            use_fp32: Disable reduced-precision autocast. It is mutually
                exclusive with ``use_fp16``.

        Returns:
            Dictionary with normalized log-depth ``depth`` and finite-depth
            probability ``mask``, both FP32 tensors ``[B, H, W]``.
        """
        height, width = image.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input resolution ({height}, {width}) must be divisible by patch size {self.patch_size}"
            )
        dtype = inference_dtype(use_fp16=use_fp16, use_fp32=use_fp32)

        with full_precision(image.device):
            image_norm = (image.float() - self.image_mean.float()) / self.image_std.float()
        with reduced_precision(image.device, dtype):
            context = self.encoder(image, height // self.patch_size, width // self.patch_size)
        context = context.flatten(2).permute(0, 2, 1).contiguous()
        depth, mask = self.predictor(image_norm, context, autocast_dtype=dtype)

        with full_precision(image.device):
            depth = self._remap(depth.float().squeeze(1))
            mask = mask.float().squeeze(1).sigmoid()
        return {"depth": depth, "mask": mask}

    def infer(
        self,
        image: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        fov_x: Optional[Union[float, torch.Tensor]] = None,
        ref_image: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
        use_fp16: bool = True,
        use_fp32: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Recover metric-scale depth, validity, intrinsics, and 3D points.

        Raw normalized log-depth is affine-aligned in log space to ``gt_depth``
        when supplied, otherwise to a lazily loaded MoGe-2 reference. Alignment
        parameters are estimated on a 64x64 nearest-resized valid subset. The
        aligned depth is exponentiated and back-projected with normalized camera
        intrinsics.

        Args:
            image: RGB tensor ``[3,H,W]`` or batch ``[B,3,H,W]`` in ``[0,1]``.
            gt_depth: Optional reference depth ``[H,W]`` or ``[B,H,W]``. Finite
                positive pixels define log-space alignment.
            intrinsics: Optional normalized camera matrices ``[3,3]`` or
                ``[B,3,3]`` corresponding to ``gt_depth``.
            fov_x: Optional horizontal field of view in degrees, scalar or
                tensor ``[B]``, used when intrinsics are unavailable.
            ref_image: Optional original-resolution RGB tensor used only by the
                reference model; PXDepth still consumes ``image``.
            apply_mask: Replace invalid predicted depth/points with infinity.
            use_fp16: Use FP16 for attention-heavy model regions.
            use_fp32: Force those regions to FP32 and override the BF16 default.

        Returns:
            Dictionary containing aligned ``depth`` ``[B,H,W]``, boolean
            ``mask`` ``[B,H,W]``, point map ``points`` ``[B,H,W,3]``, normalized
            ``intrinsics`` ``[B,3,3]``, and horizontal ``fov_x`` ``[B]``. For an
            unbatched input, the leading batch dimension is removed.
        """
        return infer_model(
            self,
            image,
            gt_depth=gt_depth,
            intrinsics=intrinsics,
            fov_x=fov_x,
            ref_image=ref_image,
            apply_mask=apply_mask,
            use_fp16=use_fp16,
            use_fp32=use_fp32,
        )
