"""Robust scalar, affine-depth, and point-cloud alignment primitives.

These routines estimate scale and shift parameters under missing data and
outliers, using low-resolution robust solvers where required by the evaluation
protocol. Returned parameters are batched tensors that callers apply to
full-resolution depth maps or point clouds.
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch


def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor) -> torch.return_types.min:
    """Scatter-reduce values by minimum and recover source indices.

    Args:
        size: Length of the reduced output dimension.
        dim: Dimension of ``src`` along which group indices apply.
        index: Long tensor broadcast-compatible with ``src`` assigning each
            source value to an output group.
        src: Floating source tensor of arbitrary shape.

    Returns:
        ``torch.return_types.min`` containing grouped minimum values and source
        indices, each shaped like ``src`` with dimension ``dim`` replaced by
        ``size``.
    """
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))


def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    """Evaluate a tensor function in chunks along its leading batch axis.

    Args:
        fn: Callable accepting the supplied positional and keyword arguments.
        chunk_size: Maximum leading-axis tensor length per invocation.
        *args: Tensor arguments sharing leading batch length, or constants
            repeated for every chunk.
        **kwargs: Keyword equivalents of ``args``.

    Returns:
        Concatenated tensor result, or tuple of concatenated tensors when ``fn``
        returns a tuple.
    """
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)


def _pad_inf(x_: torch.Tensor):
    """Pad a sorted sequence with negative and positive infinity.

    Args:
        x_: Tensor ``[...,N]``.

    Returns:
        Tensor ``[...,N+2]`` with sentinels at both ends.
    """
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    """Pad a cumulative sum with zero and its final total.

    Args:
        cumsum: Cumulative values ``[...,N]``.

    Returns:
        Tensor ``[...,N+2]`` equal to ``[0,cumsum,total]``.
    """
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    """Evaluate a truncated weighted absolute residual for candidate scales.

    Args:
        a: Candidate scales ``[K,1]``.
        xyw: Stacked source, target, and weight values ``[K,N,3]``.
        trunc: Scalar upper bound applied to every weighted residual.

    Returns:
        Objective values ``[K]``.
    """
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)


def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, trunc: Optional[Union[float, torch.Tensor]] = None, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """Solve robust weighted L1 scale alignment without iterative optimization.

    Args:
        x: Source values ``[...,N]``.
        y: Target values ``[...,N]``.
        w: Nonnegative correspondence weights ``[...,N]``.
        trunc: Optional scalar/tensor cap for each weighted absolute residual.
        eps: Lower bound protecting divisions by near-zero source values.

    Returns:
        scale: Differentiable optimal scale tensor ``[...]``.
        loss: Detached objective value tensor ``[...]``.
        index: Long tensor ``[...]`` identifying the correspondence whose ratio
            reproduces each selected optimum.
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)

    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)    # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)

            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                       # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]               # (num_extrema,)
            MAX_ELEMENTS = 4096 ** 2      # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])          # (num_extrema,)

            # Find minima among corresponding extrema
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)        # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_depth_affine(depth_src: torch.Tensor, depth_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """Fit robust affine scale and shift between paired scalar values.

    Args:
        depth_src: Source depth/log-depth values ``[...,N]``.
        depth_tgt: Target values ``[...,N]``.
        weight: Nonnegative correspondence weights ``[...,N]``.
        trunc: Optional robust residual cap forwarded to :func:`align`.

    Returns:
        scale: Scalar multiplier tensor ``[...]``.
        shift: Additive offset tensor ``[...]``.
    """

    # Flatten batch dimensions for simplicity
    batch_shape, n = depth_src.shape[:-1], depth_src.shape[-1]
    batch_size = math.prod(batch_shape)
    depth_src, depth_tgt, weight = depth_src.reshape(batch_size, n), depth_tgt.reshape(batch_size, n), weight.reshape(batch_size, n)

    # Here, we take anchors only for non-zero weights.
    # Although the results will be still correct even anchor points have zero weight,
    # it is wasting computation and may cause instability in some cases, e.g. too many extrema.
    anchors_where_batch, anchors_where_n = torch.where(weight > 0)

    # Stop gradient when solving optimal anchors
    with torch.no_grad():
        depth_src_anchor = depth_src[anchors_where_batch, anchors_where_n]                              # (anchors)
        depth_tgt_anchor = depth_tgt[anchors_where_batch, anchors_where_n]                              # (anchors)

        depth_src_anchored = depth_src[anchors_where_batch, :] - depth_src_anchor[..., None]            # (anchors, n)
        depth_tgt_anchored = depth_tgt[anchors_where_batch, :] - depth_tgt_anchor[..., None]            # (anchors, n)
        weight_anchored = weight[anchors_where_batch, :]                                                # (anchors, n)

        scale, loss, index = align(depth_src_anchored, depth_tgt_anchored, weight_anchored, trunc)      # (anchors)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchors_where_batch, src=loss)   # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_1 = anchors_where_n[index_anchor]      # (batch_size,)
    index_2 = index[index_anchor]                # (batch_size,)

    tgt_1, src_1 = torch.gather(depth_tgt, dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(depth_tgt, dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(depth_src, dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1e-7)
    shift = tgt_1 - scale * src_1

    scale, shift = scale.reshape(batch_shape), shift.reshape(batch_shape)

    return scale, shift


def align_points_scale_xyz_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None, max_iters: int = 30, eps: float = 1e-6):
    """Fit one isotropic scale and a three-axis translation to point pairs.

    Args:
        points_src: Source camera-space points ``[...,N,3]``.
        points_tgt: Target points ``[...,N,3]`` with one-to-one correspondence.
        weight: Nonnegative correspondence weights ``[...,N]``.
        trunc: Optional robust residual cap.
        max_iters: Compatibility argument retained from the research API.
        eps: Compatibility stabilizer retained from the research API.

    Returns:
        scale: Isotropic scale tensor ``[...]``.
        shift: XYZ translation tensor ``[...,3]``.
    """

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]          # (anchors, 3)
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]          # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // 2, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        # Get optimal scale and shift for each batch element
        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    src_1, tgt_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2, tgt_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)

    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def align_affine_lstsq(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit weighted affine scale and shift by linear least squares.

    Args:
        x: Source scalar values ``[...,N]``.
        y: Target scalar values ``[...,N]``.
        w: Optional nonnegative least-squares weights ``[...,N]``. ``None``
            assigns unit weight.

    Returns:
        scale: Least-squares multiplier tensor ``[...]``.
        shift: Least-squares additive offset tensor ``[...]``.
    """
    w_sqrt = torch.ones_like(x) if w is None else w.sqrt()
    A = torch.stack([w_sqrt * x, torch.ones_like(x)], dim=-1)
    B = (w_sqrt * y)[..., None]
    a, b = torch.linalg.lstsq(A, B)[0].squeeze(-1).unbind(-1)
    return a, b
