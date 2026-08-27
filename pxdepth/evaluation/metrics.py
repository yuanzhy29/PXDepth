"""Depth, point-cloud, local-structure, and boundary metrics.

Raw model outputs are aligned with protocol-specific robust affine procedures.
Depth-space, log-depth-space, and disparity-space predictions are converted to
positive depth before common metrics and point-cloud reconstruction are
evaluated.
"""

from typing import Dict, Tuple, Union
from numbers import Number

import cv2
import torch
import numpy as np
import utils3d

from ..utils.alignment import (
    align_depth_affine,
    align_disparity_affine,
    align_points_affine,
    align_points_scale_xyz_shift,
)
from ..utils.tools import key_average


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Compute mean absolute relative depth error.

    Args:
        pred: Positive predicted depths at selected pixels, tensor ``[N]``.
        gt: Positive ground-truth depths at the same pixels, tensor ``[N]``.
        eps: Denominator stabilizer for near-zero GT values.

    Returns:
        Python float containing ``mean(abs(pred-gt)/(gt+eps))``.
    """
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Compute the fraction of depth ratios below ``1.25``.

    Args:
        pred: Positive predicted depth tensor ``[N]``.
        gt: Positive ground-truth depth tensor ``[N]``.
        eps: Compatibility argument retained by the public metric API.

    Returns:
        Python float in ``[0,1]``; larger is better.
    """
    delta1 = (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean()
    return delta1.item()


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Compute 3D endpoint error relative to GT camera-space radius.

    Args:
        pred: Predicted camera-space points ``[N,3]``.
        gt: Corresponding ground-truth points ``[N,3]``.
        eps: Stabilizer added to each GT point radius.

    Returns:
        Python float mean relative Euclidean point error.
    """
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()


def delta1_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Compute the MoGe point accuracy under a 25% radial tolerance.

    Args:
        pred: Predicted camera-space points ``[N,3]``.
        gt: Corresponding ground-truth points ``[N,3]``.
        eps: Compatibility argument retained by the metric API.

    Returns:
        Python float fraction whose 3D error is below 25% of the smaller
        predicted/GT camera-space radius.
    """
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta1.item()


def rel_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    """Normalize local 3D endpoint error by an object's GT diameter.

    Args:
        pred: Locally aligned predicted points ``[N,3]``.
        gt: Ground-truth points ``[N,3]`` for the same region.
        diameter: Scalar tensor containing the largest GT bounding-box extent.

    Returns:
        Python float mean error divided by ``diameter``.
    """
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    """Compute local point accuracy at one quarter of object diameter.

    Args:
        pred: Locally aligned predicted points ``[N,3]``.
        gt: Ground-truth points ``[N,3]``.
        diameter: Scalar GT region diameter.

    Returns:
        Python float fraction with Euclidean error below ``0.25*diameter``.
    """
    dist_err = torch.norm(pred - gt, dim=-1)
    delta1 = (dist_err < 0.25 * diameter).float().mean()
    return delta1.item()


def _nan_boundary_metrics() -> Dict[str, float]:
    """Create a complete boundary metric record for invalid edge samples.

    A stable key set keeps aggregation and JSON schemas consistent.

    Returns:
        Dictionary whose boundary accuracy and Chamfer distance are both NaN.
    """
    return {
        'acc': float('nan'),
        'cd': float('nan'),
    }


def _canny_boundary_mask(gt_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Extract the Canny GT depth edge mask used for boundary evaluation.

    Args:
        gt_depth: Ground-truth depth map ``[H,W]`` in meters.
        mask: Boolean valid-depth mask ``[H,W]``.

    Returns:
        Boolean edge tensor ``[H,W]`` on ``gt_depth.device``. Depth is clipped
        to ``[0.1,65]`` meters and invalid support is dilated by a 2x2 kernel
        before Canny thresholds 100/200 are applied.
    """
    depth_np = gt_depth.detach().float().cpu().numpy()
    valid_np = mask.detach().cpu().numpy().astype(bool)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=65.0, neginf=0.0)

    depth_gt_clamp = np.clip(depth_np, 0.1, 65.0)
    min_val = depth_gt_clamp.min()
    max_val = depth_gt_clamp.max()
    norm_depth = (depth_gt_clamp - min_val) / (max_val - min_val + 1e-5)
    norm_depth = np.clip(norm_depth, 0.0, 1.0)
    depth_uint8 = (norm_depth * 255).astype(np.uint8)

    edge = cv2.Canny(depth_uint8, 100, 200) > 0.5
    kernel = np.ones((2, 2), np.uint8)
    valid_np = cv2.dilate(1 - valid_np.astype(np.uint8), kernel, iterations=1) < 0.5
    edge = edge & valid_np
    return torch.from_numpy(edge).to(device=gt_depth.device, dtype=torch.bool)


def _as_o3d_point_cloud(points: np.ndarray):
    """Convert an XYZ NumPy array to an Open3D point cloud.

    Args:
        points: Finite point array ``float [N,3]``.

    Returns:
        ``open3d.geometry.PointCloud`` containing the supplied XYZ positions.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def boundary_edge_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
    return_misc: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, torch.Tensor]]]:
    """Evaluate edge depth and 3D boundary point-cloud quality.

    GT Canny edges select the boundary point clouds. The predicted cloud is
    rigidly refined to GT with point-to-point ICP, then bidirectional
    nearest-neighbor distances produce accuracy and symmetric Chamfer distance
    in millimeters.

    Args:
        pred_depth: Globally aligned predicted depth ``[H,W]`` in meters.
        gt_depth: Ground-truth depth ``[H,W]`` in meters.
        mask: Boolean GT valid-depth mask ``[H,W]``.
        intrinsics: Normalized camera matrix ``[3,3]``.
        return_misc: Also return edge masks, aligned clouds, and ICP transform.

    Returns:
        metrics: Dictionary containing ``acc`` and ``cd`` in millimeters.
        misc: Returned only when requested. Contains ``edge_mask`` ``[H,W]``,
            edge point arrays ``[N,3]``, and ``icp_transform`` ``[4,4]``.
    """
    from scipy.spatial import cKDTree as KDTree
    import open3d as o3d

    metrics = _nan_boundary_metrics()
    misc: Dict[str, torch.Tensor] = {}

    def _finish():
        """Package the current metric state according to ``return_misc``.

        The closure captures the partially populated dictionaries by reference.

        Returns:
            Metrics dictionary alone, or ``(metrics,misc)`` when requested.
        """
        return (metrics, misc) if return_misc else metrics

    valid = mask & torch.isfinite(gt_depth) & (gt_depth > 0)
    pred_valid = torch.isfinite(pred_depth) & (pred_depth > 0)
    edge = _canny_boundary_mask(gt_depth, valid)
    gt_edge_mask = edge & valid
    pred_edge_mask = gt_edge_mask & pred_valid
    if return_misc:
        misc['edge_mask'] = edge
    if gt_edge_mask.sum().item() < 10 or pred_edge_mask.sum().item() < 10:
        return _finish()

    pred_depth_clean = pred_depth.float().clone()
    pred_depth_clean[~pred_valid] = 1.0
    gt_depth_clean = gt_depth.float().clone()
    gt_depth_clean[~valid] = 1.0

    pred_points_full = utils3d.pt.depth_map_to_point_map(pred_depth_clean, intrinsics=intrinsics)
    gt_points_full = utils3d.pt.depth_map_to_point_map(gt_depth_clean, intrinsics=intrinsics)

    pred_points = pred_points_full[pred_edge_mask].detach().float().cpu().numpy()
    gt_points = gt_points_full[gt_edge_mask].detach().float().cpu().numpy()
    pred_points = pred_points[np.isfinite(pred_points).all(axis=1)]
    gt_points = gt_points[np.isfinite(gt_points).all(axis=1)]
    if pred_points.shape[0] < 10 or gt_points.shape[0] < 10:
        return _finish()

    pcd = _as_o3d_point_cloud(pred_points)
    pcd_gt = _as_o3d_point_cloud(gt_points)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transform = reg_p2p.transformation
    pcd.transform(transform)
    pred_points_aligned = np.asarray(pcd.points)

    gt_tree = KDTree(gt_points)
    acc_distances, _ = gt_tree.query(pred_points_aligned, workers=-1)
    pred_tree = KDTree(pred_points_aligned)
    comp_distances, _ = pred_tree.query(gt_points, workers=-1)

    acc = float(np.mean(acc_distances))
    comp = float(np.mean(comp_distances))
    cd = (acc + comp) / 2.0
    if np.isfinite(acc) and np.isfinite(cd):
        metrics['acc'] = acc * 1000.0
        metrics['cd'] = cd * 1000.0

    if return_misc:
        misc['pred_edge_points'] = torch.from_numpy(pred_points_aligned).to(device=gt_depth.device, dtype=torch.float32)
        misc['gt_edge_points'] = torch.from_numpy(gt_points).to(device=gt_depth.device, dtype=torch.float32)
        misc['icp_transform'] = torch.from_numpy(transform.copy()).to(device=gt_depth.device, dtype=torch.float32)
    return _finish()


def compute_metrics(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    vis: bool = False,
    compute_boundary: bool = True,
) -> Tuple[Dict[str, Dict[str, Number]], Dict[str, torch.Tensor]]:
    """Align one prediction and compute all applicable benchmark metrics.

    Args:
        pred: Prediction dictionary. It may contain raw normalized log-depth
            ``depth_log1p_affine_invariant`` ``[H,W]``, raw
            ``disparity_affine_invariant`` ``[H,W]``, optional point map
            ``points_affine_invariant`` ``[H,W,3]``, and predicted ``mask``
            ``[H,W]``.
        gt: Ground-truth sample containing depth/mask ``[H,W]``, point map
            ``[H,W,3]``, normalized intrinsics ``[3,3]``, metric/boundary flags,
            and optional segmentation annotations.
        vis: Include aligned depth/points and boundary visualization tensors in the
            auxiliary output.
        compute_boundary: Evaluate boundary metrics when the dataset is marked
            ``has_sharp_boundary``.

    Returns:
        metrics: Nested Python-number dictionary for depth, points, local points,
            and optional boundary quality.
        misc: Tensor dictionary containing aligned maps and optional boundary
            visualization data when ``vis=True``.
    """
    metrics = {}
    misc = {}

    mask = gt['depth_mask']
    gt_depth = gt['depth']
    gt_points = gt['points']

    valid_depth = mask & torch.isfinite(gt_depth) & (gt_depth > 0)
    pred_depth_aligned = None
    pred_points_aligned = None

    if 'depth_log1p_affine_invariant' in pred:
        raw_log_depth = pred['depth_log1p_affine_invariant'].float()
        fit_mask = valid_depth & torch.isfinite(raw_log_depth)
        target_log = torch.log1p(gt_depth)
        aligned_log, ok = align_depth_affine(raw_log_depth, target_log, fit_mask, gt_depth)
        pred_depth_aligned = torch.expm1(aligned_log if ok else raw_log_depth)

        metric_mask = fit_mask
        if metric_mask.any():
            metrics['depth_affine_invariant'] = {
                'rel': rel_depth(pred_depth_aligned[metric_mask], gt_depth[metric_mask]),
                'delta1': delta1_depth(pred_depth_aligned[metric_mask], gt_depth[metric_mask]),
            }

    elif 'disparity_affine_invariant' in pred:
        raw_disparity = pred['disparity_affine_invariant'].float()
        fit_mask = valid_depth & torch.isfinite(raw_disparity)
        gt_disparity = torch.where(valid_depth, gt_depth.reciprocal(), torch.zeros_like(gt_depth))
        aligned_disparity, ok = align_disparity_affine(raw_disparity, gt_disparity, fit_mask)
        aligned_disparity = aligned_disparity if ok else raw_disparity
        if fit_mask.any():
            max_depth = gt_depth[fit_mask].max()
            pred_depth_metric = aligned_disparity.clamp_min(max_depth.reciprocal()).reciprocal()
        else:
            pred_depth_metric = aligned_disparity.clamp_min(1e-6).reciprocal()
        pred_depth_aligned = pred_depth_metric
        metric_mask = fit_mask & torch.isfinite(pred_depth_metric)
        if metric_mask.any():
            metrics['depth_affine_invariant'] = {
                'rel': rel_depth(pred_depth_metric[metric_mask], gt_depth[metric_mask]),
                'delta1': delta1_depth(pred_depth_metric[metric_mask], gt_depth[metric_mask]),
            }

    pred_points_affine_invariant = pred.get('points_affine_invariant', None)
    if pred_points_affine_invariant is None and pred_depth_aligned is not None:
        point_intrinsics = gt['intrinsics'].to(
            device=pred_depth_aligned.device,
            dtype=pred_depth_aligned.dtype,
        )
        pred_points_affine_invariant = utils3d.pt.depth_map_to_point_map(
            pred_depth_aligned,
            intrinsics=point_intrinsics,
        )

    if pred_points_affine_invariant is not None:
        point_mask = (
            valid_depth
            & torch.isfinite(pred_points_affine_invariant).all(dim=-1)
            & torch.isfinite(gt_points).all(dim=-1)
        )
        if point_mask.any():
            aligned_points, ok = align_points_affine(pred_points_affine_invariant, gt_points, point_mask)
            pred_points_aligned = aligned_points if ok else pred_points_affine_invariant
            metrics['points_affine_invariant'] = {
                'rel': rel_point(pred_points_aligned[point_mask], gt_points[point_mask]),
                'delta1': delta1_point(pred_points_aligned[point_mask], gt_points[point_mask]),
            }

    # Local points
    if 'segmentation_mask' in gt and 'points' in gt and pred_points_affine_invariant is not None:
        pred_points = pred_points_affine_invariant
        gt_points = gt['points']
        segmentation_mask = gt['segmentation_mask']
        segmentation_labels = gt['segmentation_labels']
        local_points_metrics = []
        for _, seg_id in segmentation_labels.items():
            valid_mask = (
                (segmentation_mask == seg_id)
                & valid_depth
                & torch.isfinite(pred_points).all(dim=-1)
                & torch.isfinite(gt_points).all(dim=-1)
            )
            if valid_mask.sum().item() < 10:
                continue

            try:
                pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
                    torch.where(valid_mask[..., None], pred_points.float(), torch.zeros_like(pred_points, dtype=torch.float32)),
                    torch.where(valid_mask[..., None], gt_points.float(), torch.zeros_like(gt_points, dtype=torch.float32)),
                    mask=valid_mask,
                    size=(64, 64),
                )
                pred_points_masked = pred_lr[mask_lr]
                gt_points_masked = gt_lr[mask_lr]
                if pred_points_masked.shape[0] < 10:
                    continue
                diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()
                scale, shift = align_points_scale_xyz_shift(
                    pred_points_masked.unsqueeze(0),
                    gt_points_masked.unsqueeze(0),
                    diameter.clamp_min(1e-6).reciprocal().expand(1, gt_points_masked.shape[0]),
                )
                pred_points_masked = pred_points[valid_mask] * scale.squeeze() + shift.squeeze()
                gt_points_masked = gt_points[valid_mask]
            except Exception:
                pred_points_masked = pred_points[valid_mask]
                gt_points_masked = gt_points[valid_mask]
                diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()

            local_points_metrics.append({
                'rel': rel_point_local(pred_points_masked, gt_points_masked, diameter),
                'delta1': delta1_point_local(pred_points_masked, gt_points_masked, diameter),
            })

        metrics['local_points'] = key_average(local_points_metrics)

    # Boundary Acc/CD with Canny depth edges.
    boundary_depth = pred_depth_aligned
    if compute_boundary and boundary_depth is not None and gt['has_sharp_boundary']:
        if vis:
            boundary_metrics, boundary_misc = boundary_edge_metrics(
                boundary_depth,
                gt_depth,
                mask,
                gt['intrinsics'],
                return_misc=True,
            )
        else:
            boundary_metrics = boundary_edge_metrics(
                boundary_depth,
                gt_depth,
                mask,
                gt['intrinsics'],
            )
            boundary_misc = {}
        metrics['boundary'] = boundary_metrics
        if vis:
            if 'edge_mask' in boundary_misc:
                misc['boundary_edge_mask'] = boundary_misc['edge_mask']
            if 'pred_edge_points' in boundary_misc:
                misc['boundary_pred_edge_points'] = boundary_misc['pred_edge_points']
            if 'gt_edge_points' in boundary_misc:
                misc['boundary_gt_edge_points'] = boundary_misc['gt_edge_points']
            if 'icp_transform' in boundary_misc:
                misc['boundary_icp_transform'] = boundary_misc['icp_transform']

    if vis:
        if pred_points_aligned is not None:
            misc['pred_points'] = pred_points_aligned
        elif pred_depth_aligned is not None:
            misc['pred_points'] = utils3d.pt.depth_map_to_point_map(pred_depth_aligned, intrinsics=gt['intrinsics'])
        if pred_depth_aligned is not None:
            misc['pred_depth'] = pred_depth_aligned

    return metrics, misc
