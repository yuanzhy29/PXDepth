# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch.nn.functional as F
from torch import Tensor
from torch import nn
import torch


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # # Deprecated implementation, extremely slow
    # def forward(self, x: Tensor, attn_bias=None) -> Tensor:
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    #     attn = q @ k.transpose(-2, -1)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, H, N, C // H)

        q, k, v = qkv.unbind(0)      # (B, H, N, C // H)

        use_flash_attn = (
            FLASH_ATTN_AVAILABLE
            and attn_bias is None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash_attn:
            q_f = q.permute(0, 2, 1, 3).contiguous()
            k_f = k.permute(0, 2, 1, 3).contiguous()
            v_f = v.permute(0, 2, 1, 3).contiguous()
            x = flash_attn_func(
                q_f,
                k_f,
                v_f,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            x = x.reshape(B, N, C)
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        use_flash_attn = (
            FLASH_ATTN_AVAILABLE
            and attn_bias is None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash_attn:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.unbind(2)
            x = flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, attn_bias=attn_bias)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
