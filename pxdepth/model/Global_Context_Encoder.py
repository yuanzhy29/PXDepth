"""Global Context Encoder used to condition pixel-space depth prediction.

A DINOv2 vision transformer extracts selected intermediate patch-token maps.
Each map is normalized, reshaped to its image grid, projected to a common
channel width, and summed into the context feature consumed by CM-PiT adaptive
normalization layers.
"""

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import ENCODERS
from .dinov2.hub import backbones
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing


@ENCODERS.register()
class GlobalContextEncoder(nn.Module):
    """Global Context Encoder based on intermediate DINOv2 features.

    The encoder extracts several normalized patch-token maps from a ViT,
    projects each map to a shared channel width with a 1x1 convolution, and
    sums the projected maps. The resulting grid provides global semantic
    context for Context-Guided Adaptive Normalization in the pixel predictor.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitl14",
        intermediate_layers: Union[int, Sequence[int]] = (5, 11, 17, 23),
        dim_out: int = 1024,
    ) -> None:
        """Construct the DINOv2 backbone and intermediate projections.

        Args:
            backbone: Name of a constructor exposed by ``dinov2.hub.backbones``.
            intermediate_layers: Explicit zero-based block indices or an
                integer requesting the last ``n`` intermediate layers.
            dim_out: Channel count ``C_ctx`` of every projected context map.

        Returns:
            ``None``. The backbone, output projections, and ImageNet
            normalization buffers are registered on the module.
        """
        super().__init__()
        if not hasattr(backbones, backbone):
            raise ValueError(f"Unsupported DINOv2 backbone: {backbone}")

        self.backbone_name = backbone
        self.intermediate_layers = list(intermediate_layers) if not isinstance(intermediate_layers, int) else intermediate_layers
        self.backbone = getattr(backbones, backbone)(pretrained=False)
        if hasattr(self.backbone, "mask_token"):
            self.backbone.mask_token.requires_grad_(False)

        patch_size = getattr(self.backbone, "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.dim_features = int(getattr(self.backbone, "embed_dim"))
        self.dim_out = int(dim_out)
        count = self.intermediate_layers if isinstance(self.intermediate_layers, int) else len(self.intermediate_layers)
        self.output_projections = nn.ModuleList(
            nn.Conv2d(self.dim_features, dim_out, kernel_size=1) for _ in range(count)
        )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._onnx_compatible_mode = False

    @property
    def onnx_compatible_mode(self) -> bool:
        """Report whether ONNX-compatible resize behavior is enabled.

        Returns:
            Boolean flag controlling antialiasing and the vendored backbone's
            ONNX compatibility path.
        """
        return self._onnx_compatible_mode

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, enabled: bool) -> None:
        """Enable or disable ONNX-compatible encoder operators.

        Args:
            enabled: Boolean state propagated to the DINOv2 backbone.

        Returns:
            ``None``. Runtime flags are updated in place.
        """
        self._onnx_compatible_mode = bool(enabled)
        self.backbone.onnx_compatible_mode = bool(enabled)

    def init_weights(self) -> None:
        """Load official pretrained weights for the configured DINOv2 backbone.

        Returns:
            ``None``. Backbone parameters are replaced in place while the
            PXDepth-specific 1x1 projections keep their initialization.
        """
        state = getattr(backbones, self.backbone_name)(pretrained=True).state_dict()
        self.backbone.load_state_dict(state, strict=True)

    def enable_gradient_checkpointing(self) -> None:
        """Wrap every DINO transformer block with activation checkpointing.

        Parameter names and numerical block behavior remain unchanged; only
        activation storage during training is affected.

        Returns:
            ``None``. Each backbone block is modified in place.
        """
        for block in self.backbone.blocks:
            wrap_module_with_gradient_checkpointing(block)

    def enable_pytorch_native_sdpa(self) -> None:
        """Replace DINO attention forward methods with SDPA-compatible paths.

        Returns:
            ``None``. Attention modules are wrapped in place and use
            Flash-Attention when the installed runtime supports it.
        """
        for block in self.backbone.blocks:
            wrap_dinov2_attention_with_sdpa(block.attn)

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int,
        token_cols: int,
        return_feature_maps: bool = False,
        return_class_token: bool = False,
    ):
        """Encode RGB images into a summed global context feature map.

        Args:
            image: RGB tensor ``[B, 3, H_in, W_in]`` with values in ``[0, 1]``.
            token_rows: Requested context-grid height ``H_ctx``.
            token_cols: Requested context-grid width ``W_ctx``.
            return_feature_maps: Also return the list of individually projected
                feature maps when ``True``.
            return_class_token: Also return the final selected DINO class token
                ``[B, C_vit]`` when ``True``.

        Returns:
            By default, a context map ``[B, C_ctx, H_ctx, W_ctx]``. Optional
            outputs are appended as a tuple in the order ``feature_maps`` then
            ``class_token``. Each feature map has shape
            ``[B, C_ctx, H_ctx, W_ctx]``.
        """
        target_size = (token_rows * self.patch_size, token_cols * self.patch_size)
        if image.shape[-2:] != target_size:
            image = F.interpolate(
                image,
                size=target_size,
                mode="bilinear",
                align_corners=False,
                antialias=not self.onnx_compatible_mode,
            )
        image = (image - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(
            image,
            n=self.intermediate_layers,
            return_class_token=True,
            norm=True,
        )
        maps = []
        context = None
        for projection, (tokens, _) in zip(self.output_projections, features):
            feature = tokens.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous()
            projected = projection(feature)
            context = projected if context is None else context + projected
            if return_feature_maps:
                maps.append(projected)
        if context is None:
            raise RuntimeError("Global Context Encoder did not receive any intermediate features.")

        outputs: List[object] = [context]
        if return_feature_maps:
            outputs.append(maps)
        if return_class_token:
            outputs.append(features[-1][1])
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
