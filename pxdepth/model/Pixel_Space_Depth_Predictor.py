"""Pixel-Space Depth Predictor that preserves the dense image lattice.

Normalized RGB is embedded with a 1x1 projection and processed by shared
CM-PiT trunk blocks before branching into depth and finite-mask predictors.
Linear patch compression is used only within transformer blocks, after which
features are expanded back to per-pixel tokens for dense output heads.
"""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..registry import PREDICTORS
from .CM_PiT import CMPiTBlock
from .RoPE import PositionGetter, RotaryPositionEmbedding2D
from .precision import full_precision, reduced_precision


@PREDICTORS.register()
class PixelSpaceDepthPredictor(nn.Module):
    """Pixel-Space Depth Predictor built from cascaded CM-PiT blocks.

    A 1x1 projection first embeds normalized RGB into dense pixel features.
    Shared trunk blocks refine those features, after which independent depth
    and validity branches predict normalized log-depth and finite-depth logits.
    No convolution larger than 1x1 is applied to the pixel representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim_ctx: int = 1024,
        attn_dim: int = 1536,
        ctx_patch_size: int = 14,
        dim_pix: int = 16,
        trunk_patch_size: int = 14,
        depth_patch_size: int = 7,
        mask_patch_size: int = 14,
        num_heads: int = 24,
        trunk_depth: int = 4,
        depth_depth: int = 4,
        mask_depth: int = 2,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        rope_frequency: float = 100.0,
        eps: float = 1e-6,
        gradient_checkpointing: bool = True,
    ) -> None:
        """Construct the shared trunk and two prediction branches.

        Args:
            in_channels: Number of image channels, equal to three for RGB.
            dim_ctx: Global context-token channel count ``C_ctx``.
            attn_dim: Channel count ``D`` after linear patch compression.
            ctx_patch_size: Encoder patch size ``P_ctx`` in image pixels.
            dim_pix: Channel count ``C_pix`` of the dense pixel feature map.
            trunk_patch_size: Attention patch size used by shared trunk blocks.
            depth_patch_size: Attention patch size used by depth blocks.
            mask_patch_size: Attention patch size used by validity-mask blocks.
            num_heads: Number of gated-attention heads.
            trunk_depth: Number of shared CM-PiT blocks.
            depth_depth: Number of depth-branch CM-PiT blocks.
            mask_depth: Number of validity-branch CM-PiT blocks.
            mlp_ratio: SwiGLU expansion ratio inside every block.
            qk_norm: Enable FP32 RMSNorm for attention queries and keys.
            rope_frequency: Base frequency of the shared 2D RoPE module.
            eps: Numerical epsilon for normalization layers.
            gradient_checkpointing: Recompute CM-PiT blocks during backward to
                reduce activation memory.

        Returns:
            ``None``. The complete pixel predictor is registered on the module.
        """
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})")
        for name, patch_size in {
            "trunk_patch_size": trunk_patch_size,
            "depth_patch_size": depth_patch_size,
            "mask_patch_size": mask_patch_size,
        }.items():
            if patch_size <= 0 or ctx_patch_size % patch_size != 0:
                raise ValueError(
                    f"{name} ({patch_size}) must be positive and divide ctx_patch_size ({ctx_patch_size})"
                )

        self.dim_ctx = dim_ctx
        self.dim_pix = dim_pix
        self.attn_dim = attn_dim
        self.ctx_patch_size = ctx_patch_size
        self.trunk_patch_size = trunk_patch_size
        self.depth_patch_size = depth_patch_size
        self.mask_patch_size = mask_patch_size
        self.gradient_checkpointing = gradient_checkpointing

        self.pos = PositionGetter()
        self.rope = RotaryPositionEmbedding2D(frequency=rope_frequency)
        self.input_proj = nn.Conv2d(in_channels, dim_pix, kernel_size=1, bias=True)

        block_args = dict(
            dim_ctx=dim_ctx,
            ctx_patch_size=ctx_patch_size,
            dim_pix=dim_pix,
            attn_dim=attn_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            rope=self.rope,
            eps=eps,
        )
        self.trunk_blocks = nn.ModuleList(
            CMPiTBlock(patch_size=trunk_patch_size, **block_args) for _ in range(trunk_depth)
        )
        self.depth_blocks = nn.ModuleList(
            CMPiTBlock(patch_size=depth_patch_size, **block_args) for _ in range(depth_depth)
        )
        self.mask_blocks = nn.ModuleList(
            CMPiTBlock(patch_size=mask_patch_size, **block_args) for _ in range(mask_depth)
        )
        self.depth_head = nn.Conv2d(dim_pix, 1, kernel_size=1, bias=True)
        self.mask_head = nn.Conv2d(dim_pix, 1, kernel_size=1, bias=True)
        self.reset_parameters()

    def _blocks(self) -> Iterable[CMPiTBlock]:
        """Iterate over every CM-PiT block in execution-independent order.

        Returns:
            Iterable containing shared trunk, depth, and mask blocks. The method
            takes no tensor inputs and is used for parameter initialization.
        """
        return (*self.trunk_blocks, *self.depth_blocks, *self.mask_blocks)

    def reset_parameters(self) -> None:
        """Initialize projections and start adaptive modulation at identity.

        Linear and 1x1 convolution weights use Xavier uniform initialization.
        Normalization scales start at one. The final adaptive-normalization
        projections are zeroed so every CM-PiT residual branch initially has
        zero modulation and zero gate.

        Returns:
            ``None``. Parameters are modified in place.
        """
        def init(module: nn.Module) -> None:
            """Initialize one child module visited by :meth:`nn.Module.apply`.

            Args:
                module: Child ``nn.Module`` to initialize in place.

            Returns:
                ``None``.
            """
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)
        for block in self._blocks():
            nn.init.zeros_(block.ada_norm.proj[-1].weight)
            nn.init.zeros_(block.ada_norm.proj[-1].bias)

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation recomputation for CM-PiT blocks.

        The flag is consulted only while the module is in training mode.

        Returns:
            ``None``. The runtime flag is changed in place.
        """
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable activation recomputation for CM-PiT blocks.

        Subsequent training forwards retain block activations for backward.

        Returns:
            ``None``. The runtime flag is changed in place.
        """
        self.gradient_checkpointing = False

    def _position(self, batch: int, height: int, width: int, patch_size: int, device: torch.device):
        """Create cached 2D coordinates for one decoder patch grid.

        Args:
            batch: Batch size ``B``.
            height: Dense image-feature height ``H``.
            width: Dense image-feature width ``W``.
            patch_size: Block patch side length ``P``.
            device: Device on which coordinates are allocated.

        Returns:
            Integer position tensor ``[B, (H/P)*(W/P), 2]``.
        """
        return self.pos(batch, height // patch_size, width // patch_size, device=device).to(device)

    def _run(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        ctx: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Run a sequence of CM-PiT blocks with optional checkpointing.

        Args:
            x: Dense pixel features ``[B, C_pix, H, W]``.
            blocks: Ordered CM-PiT block collection for one branch.
            ctx: Global context tokens ``[B, N_ctx, C_ctx]``.
            pos: 2D positions ``[B, N, 2]`` matching the blocks' patch grid.

        Returns:
            Refined dense features ``[B, C_pix, H, W]``.
        """
        for block in blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, ctx, pos, use_reentrant=False)
            else:
                x = block(x, ctx, pos)
        return x

    def forward(
        self,
        image: torch.Tensor,
        ctx: torch.Tensor,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict normalized log-depth and finite-depth logits from RGB.

        Args:
            image: ImageNet-normalized RGB tensor ``[B, 3, H, W]``.
            ctx: Global context tokens ``[B, (H/P_ctx)*(W/P_ctx), C_ctx]``.
            autocast_dtype: Decoder attention dtype. Use ``None`` for FP32,
                ``torch.float16`` for FP16, or ``torch.bfloat16`` for BF16.

        Returns:
            depth: Raw normalized log-depth tensor ``[B, 1, H, W]``.
            mask: Raw finite-depth logit tensor ``[B, 1, H, W]``.
        """
        batch, _, height, width = image.shape
        p_ctx = self.ctx_patch_size
        if height % p_ctx != 0 or width % p_ctx != 0:
            raise ValueError(f"Input resolution ({height}, {width}) must be divisible by {p_ctx}")
        expected = (height // p_ctx) * (width // p_ctx)
        if tuple(ctx.shape) != (batch, expected, self.dim_ctx):
            raise ValueError(
                f"Context shape {tuple(ctx.shape)} does not match ({batch}, {expected}, {self.dim_ctx})"
            )

        with full_precision(image.device):
            pix = self.input_proj(image.float())

        with reduced_precision(image.device, autocast_dtype):
            trunk_pos = self._position(batch, height, width, self.trunk_patch_size, image.device)
            pix = self._run(pix, self.trunk_blocks, ctx, trunk_pos)

            depth_pos = self._position(batch, height, width, self.depth_patch_size, image.device)
            depth_feat = self._run(pix, self.depth_blocks, ctx, depth_pos)

            mask_pos = self._position(batch, height, width, self.mask_patch_size, image.device)
            mask_feat = self._run(pix, self.mask_blocks, ctx, mask_pos)

        with full_precision(image.device):
            depth = self.depth_head(depth_feat.float())
            mask = self.mask_head(mask_feat.float())
        return depth, mask
