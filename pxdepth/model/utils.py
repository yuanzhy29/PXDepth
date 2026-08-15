"""Runtime wrappers for DINOv2 checkpointing and optimized attention.

The vendored DINOv2 source is kept close to upstream.  These two helpers apply
PXDepth-specific runtime behavior without editing every upstream block.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


def wrap_module_with_gradient_checkpointing(module: nn.Module) -> nn.Module:
    """Recompute a module's forward pass during backward to save memory.

    A dynamic subclass replaces only the supplied module instance's class. Its
    original ``forward`` remains the computation being checkpointed, so
    parameters and state-dict names are unchanged.

    Args:
        module: DINO transformer block to modify in place. Its forward inputs
            and output retain the upstream block contract, normally token
            tensors ``[B, N, C]`` plus optional attention metadata.

    Returns:
        The same module instance with a checkpointed ``forward`` method.
    """
    from torch.utils.checkpoint import checkpoint

    class _CheckpointingWrapper(module.__class__):
        """Per-instance subclass that checkpoints the inherited forward call.

        It introduces no parameters or buffers, preserving the wrapped DINO
        block's state-dict schema and tensor interface.

        Instances are created by mutating the class of one existing block.
        """

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Run the inherited module under non-reentrant checkpointing.

            Args:
                *args: Positional arguments accepted by the wrapped block,
                    typically a token tensor ``[B,N,C]``.
                **kwargs: Keyword arguments accepted by the wrapped block.

            Returns:
                The wrapped block's original output, normally ``[B,N,C]``.
            """
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


def wrap_dinov2_attention_with_sdpa(module: nn.Module) -> nn.Module:
    """Replace one DINOv2 attention forward path with Flash-Attention or SDPA.

    Flash-Attention is selected only for CUDA FP16/BF16 tensors without an
    additive attention bias.  Other inputs use PyTorch scaled dot-product
    attention.  The wrapper reuses the upstream QKV and output projections, so
    checkpoint keys and numerical semantics remain compatible.

    Args:
        module: DINOv2 attention module with ``qkv``, ``num_heads``, ``scale``,
            ``proj``, and ``proj_drop`` attributes.

    Returns:
        The same attention module instance with an optimized forward method.
    """
    if torch.__version__ < "2.0":
        raise RuntimeError("SDPA requires PyTorch 2.0 or later")

    class _AttentionWrapper(module.__class__):
        """Per-instance attention subclass dispatching to optimized kernels.

        It reuses all upstream projections and introduces no new checkpoint
        state, so wrapping does not affect serialization compatibility.

        Instances are created by mutating one existing DINO attention module.
        """

        def forward(self, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
            """Apply multi-head self-attention to DINO patch and special tokens.

            Args:
                x: Input token tensor ``[B, N, C]``.
                attn_bias: Optional bias broadcastable to ``[B,H,N,N]``. A
                    non-``None`` value disables the external Flash-Attention path.

            Returns:
                Projected token tensor ``[B, N, C]``.
            """
            batch, length, channels = x.shape
            qkv = self.qkv(x).reshape(
                batch,
                length,
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            query, key, value = torch.unbind(qkv, 0)

            use_flash = (
                FLASH_ATTN_AVAILABLE
                and attn_bias is None
                and query.is_cuda
                and query.dtype in (torch.float16, torch.bfloat16)
            )
            if use_flash:
                query = query.permute(0, 2, 1, 3).contiguous()
                key = key.permute(0, 2, 1, 3).contiguous()
                value = value.permute(0, 2, 1, 3).contiguous()
                out = flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=False,
                ).reshape(batch, length, channels)
            else:
                out = F.scaled_dot_product_attention(query, key, value, attn_bias)
                out = out.permute(0, 2, 1, 3).reshape(batch, length, channels)
            return self.proj_drop(self.proj(out))

    module.__class__ = _AttentionWrapper
    return module
