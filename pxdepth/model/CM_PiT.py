"""Context-Modulated Pixel Transformer (CM-PiT) building blocks.

CM-PiT compresses local dense pixel features into attention tokens, processes
them with gated self-attention and SwiGLU, and expands them back without losing
the original pixel lattice. Global encoder tokens generate adaptive shift,
scale, and residual gates that condition both transformer sublayers.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .Gated_Attention import GatedAttention
from .RoPE import RotaryPositionEmbedding2D
from .precision import full_precision


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply affine context modulation without changing the tensor layout.

    Args:
        x: Normalized pixel tokens with shape ``[B, N, P, C]``.
        shift: Context-predicted additive offsets with shape ``[B, N, P, C]``.
        scale: Context-predicted residual scales with shape ``[B, N, P, C]``.

    Returns:
        Modulated tokens ``x * (1 + scale) + shift`` with shape
        ``[B, N, P, C]``.
    """
    return x * (1.0 + scale) + shift


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer operating independently on every pixel token.

    The first projection creates value and gate branches, SiLU activates the
    gate, and the second projection returns to the pixel-channel dimension.
    Spatial and patch axes are preserved throughout the module.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Construct the gated feed-forward projections.

        Args:
            dim: Input and output channel count ``C``.
            hidden_dim: Width of each hidden value/gate branch.

        Returns:
            ``None``. Learnable linear layers are registered on the module.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform pixel tokens with a SiLU-gated hidden representation.

        Args:
            x: Floating tensor with arbitrary leading dimensions and final
                channel dimension ``C=dim``. CM-PiT supplies ``[B,N,P,C]``.

        Returns:
            Tensor with the same shape and dtype as ``x``.
        """
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(value * F.silu(gate))


class ContextAdaNorm(nn.Module):
    """Predict Context-Guided Adaptive Normalization parameters.

    Each global context token produces shift, scale, and residual-gate values
    for both the attention and MLP sublayers over every pixel represented by
    that encoder token. The six parameter groups are unpacked by
    :class:`CMPiTBlock`.
    """

    def __init__(self, dim_ctx: int, patch_size: int, dim_pix: int) -> None:
        """Create the context-to-modulation projection.

        Args:
            dim_ctx: Channel count of each Global Context Encoder token.
            patch_size: Encoder patch side length ``P_ctx`` in image pixels.
            dim_pix: Pixel-feature channel count ``C_pix``.

        Returns:
            ``None``. The projection outputs ``6 * P_ctx^2 * C_pix`` values per
            context token.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_ctx, 6 * patch_size * patch_size * dim_pix),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """Project context tokens into six dense pixel-wise parameter fields.

        Args:
            ctx: Context token tensor ``[B, N_ctx, C_ctx]``.

        Returns:
            Modulation tensor ``[B, N_ctx, 6 * P_ctx^2 * C_pix]``.
        """
        return self.proj(ctx)


class CMPiTBlock(nn.Module):
    """Context-Modulated Pixel Transformer block.

    The block groups a dense pixel feature map into local patches, linearly
    compresses every patch to an attention token, applies gated global
    self-attention, expands the token back to pixel features, and follows it
    with a per-pixel SwiGLU MLP. Both residual branches use Context-Guided
    Adaptive Normalization generated from DINO context tokens.
    """

    def __init__(
        self,
        dim_ctx: int,
        ctx_patch_size: int,
        dim_pix: int,
        patch_size: int,
        attn_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        rope: Optional[RotaryPositionEmbedding2D] = None,
        eps: float = 1e-6,
    ) -> None:
        """Configure one CM-PiT block.

        Args:
            dim_ctx: Context-token channel count ``C_ctx``.
            ctx_patch_size: Image patch size ``P_ctx`` represented by one
                context token.
            dim_pix: Dense pixel-feature channel count ``C_pix``.
            patch_size: Side length ``P`` grouped into one attention token.
                It must divide ``ctx_patch_size``.
            attn_dim: Compressed attention-token channel count ``D``.
            num_heads: Number of attention heads. ``D`` must be divisible by it.
            mlp_ratio: Expansion ratio controlling the SwiGLU hidden width.
            qk_norm: Whether to apply FP32 RMSNorm to each query/key head.
            rope: Optional 2D rotary position embedding shared by decoder blocks.
            eps: Numerical epsilon used by RMSNorm layers.

        Returns:
            ``None``. Attention, modulation, MLP, and projection layers are
            registered on the block.
        """
        super().__init__()
        if ctx_patch_size % patch_size != 0:
            raise ValueError(
                f"ctx_patch_size ({ctx_patch_size}) must be divisible by patch_size ({patch_size})"
            )

        self.dim_ctx = dim_ctx
        self.dim_pix = dim_pix
        self.ctx_patch_size = ctx_patch_size
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size * dim_pix

        self.norm1 = nn.RMSNorm(dim_pix, eps=eps)
        self.linear_compress = nn.Linear(patch_dim, attn_dim)
        self.attn = GatedAttention(attn_dim, num_heads, qk_norm=qk_norm, rope=rope, eps=eps)
        self.linear_expand = nn.Linear(attn_dim, patch_dim)
        self.norm2 = nn.RMSNorm(dim_pix, eps=eps)
        hidden_dim = max(1, int(round(dim_pix * mlp_ratio * 2.0 / 3.0)))
        self.mlp = SwiGLU(dim_pix, hidden_dim)
        self.ada_norm = ContextAdaNorm(dim_ctx, ctx_patch_size, dim_pix)

    @staticmethod
    def _norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Evaluate a normalization layer in FP32 and restore input dtype.

        Args:
            norm: Normalization module acting on the final channel dimension.
            x: Pixel tokens ``[B, N, P^2, C_pix]`` in the active model dtype.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.
        """
        dtype = x.dtype
        with full_precision(x.device):
            out = norm(x.float())
        return out.to(dtype)

    def _modulation(self, ctx: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Align context modulation fields with the block's pixel patches.

        Args:
            ctx: Global context tokens ``[B, H_ctx*W_ctx, C_ctx]``.
            height: Dense pixel-map height ``H``.
            width: Dense pixel-map width ``W``.

        Returns:
            Six modulation groups with shape
            ``[B, (H/P)*(W/P), 6, P^2, C_pix]``. Rearrangement is exact and
            contains no interpolation.
        """
        batch = ctx.shape[0]
        p_ctx, p = self.ctx_patch_size, self.patch_size
        ctx_h, ctx_w = height // p_ctx, width // p_ctx
        if ctx.shape[1] != ctx_h * ctx_w:
            raise ValueError(
                f"Context token count ({ctx.shape[1]}) does not match grid ({ctx_h}x{ctx_w})"
            )

        mod = self.ada_norm(ctx).view(batch, ctx_h, ctx_w, 6, p_ctx, p_ctx, self.dim_pix)
        if p_ctx == p:
            return rearrange(mod, "b h w m ph pw c -> b (h w) m (ph pw) c")

        ratio = p_ctx // p
        return rearrange(
            mod,
            "b h w m (rh ph) (rw pw) c -> b (h rh w rw) m (ph pw) c",
            rh=ratio,
            rw=ratio,
            ph=p,
            pw=p,
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply context-modulated attention and MLP residual updates.

        Args:
            x: Dense pixel features ``[B, C_pix, H, W]``.
            ctx: Global context tokens ``[B, (H/P_ctx)*(W/P_ctx), C_ctx]``.
            pos: Integer 2D token positions ``[B, (H/P)*(W/P), 2]`` used by
                rotary position embedding in self-attention.

        Returns:
            Updated dense pixel features ``[B, C_pix, H, W]``.
        """
        batch, _, height, width = x.shape
        p = self.patch_size
        pix = rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw) c", ph=p, pw=p)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self._modulation(
            ctx, height, width
        ).unbind(dim=2)

        out = modulate(self._norm(self.norm1, pix), shift_attn, scale_attn)
        out = self.linear_compress(out.flatten(2))
        out = self.attn(out, pos=pos)
        out = self.linear_expand(out).view(batch, -1, p * p, self.dim_pix)
        pix = pix + gate_attn * out

        out = modulate(self._norm(self.norm2, pix), shift_mlp, scale_mlp)
        pix = pix + gate_mlp * self.mlp(out)
        return rearrange(
            pix,
            "b (h w) (ph pw) c -> b c (h ph) (w pw)",
            h=height // p,
            w=width // p,
            ph=p,
            pw=p,
        )
