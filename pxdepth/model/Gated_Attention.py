"""Gated multi-head self-attention used inside CM-PiT blocks.

The implementation performs optional FP32 query/key normalization, applies
two-dimensional rotary position embeddings, and delegates attention to PyTorch
SDPA. A learned token-channel sigmoid gate modulates the attended features
before the output projection.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .RoPE import RotaryPositionEmbedding2D
from .precision import full_precision


class GatedAttention(nn.Module):
    """Multi-head self-attention followed by a learned token-channel gate.

    The Q/K normalization and 2D RoPE calculations follow the numerical path
    used for the released model. Q/K normalization and RoPE are evaluated in
    FP32, while SDPA follows the active decoder autocast dtype.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        rope: Optional[RotaryPositionEmbedding2D] = None,
        eps: float = 1e-6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Construct gated multi-head self-attention.

        Args:
            dim: Token channel count ``D``.
            num_heads: Number of attention heads. ``D`` must be divisible by it.
            qk_norm: Enable per-head RMSNorm for queries and keys.
            rope: Optional 2D rotary position embedding module.
            eps: Epsilon used by query/key RMSNorm.
            attn_drop: Attention-probability dropout used during training.
            proj_drop: Dropout applied after the output projection.

        Returns:
            ``None``. QKV, gate, output, and optional normalization layers are
            registered on the module.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.rope = rope
        self.attn_drop = float(attn_drop)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply self-attention and token-channel gating.

        Args:
            x: Compressed patch tokens ``[B, N, D]``.
            pos: Optional integer grid coordinates ``[B, N, 2]``. They are
                required when a rotary position embedding is configured.

        Returns:
            Gated and projected attention output ``[B, N, D]``.
        """
        batch, length, dim = x.shape
        gate = torch.sigmoid(self.gate(x))
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if not isinstance(self.q_norm, nn.Identity):
            dtype = q.dtype
            with full_precision(q.device):
                q = self.q_norm(q.float())
                k = self.k_norm(k.float())
            q, k = q.to(dtype), k.to(dtype)

        if self.rope is not None:
            dtype = q.dtype
            with full_precision(q.device):
                q = self.rope(q.float(), pos)
                k = self.rope(k.float(), pos)
            q, k = q.to(dtype), k.to(dtype)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = out * gate.to(out.dtype)
        return self.proj_drop(self.proj(out))
