"""Geometry-aware RGB-depth-camera transforms and blur primitives.

Every geometric function updates normalized camera intrinsics together with RGB
and depth. Mixed depth interpolation uses bilinear disparity on smooth surfaces
and nearest sampling around geometric discontinuities to avoid flying points.
The evaluation loader uses the geometry-preserving transforms in this module.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import utils3d
from scipy.signal import fftconvolve

def sample_perspective(
    src_intrinsics: np.ndarray,
    tgt_aspect: float,
    center_augmentation: float,
    fov_range_absolute: Tuple[float, float],
    fov_range_relative: Tuple[float, float],
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a valid target pinhole view inside a source camera frustum.

    Args:
        src_intrinsics: Normalized source camera matrix ``float [3,3]``.
        tgt_aspect: Target width divided by target height.
        center_augmentation: Fraction controlling random optical-axis movement.
        fov_range_absolute: Minimum/maximum target FoV in degrees.
        fov_range_relative: Multipliers limiting target FoV relative to source.
        rng: NumPy random generator used for FoV and center sampling.

    Returns:
        tgt_intrinsics: Normalized target camera matrix ``float32 [3,3]``.
        rotation: Camera-space rotation ``float32 [3,3]`` mapping source rays
            into the sampled target view.
    """
    raw_horizontal, raw_vertical = abs(1.0 / src_intrinsics[0, 0]), abs(1.0 / src_intrinsics[1, 1])
    raw_fov_x, raw_fov_y = utils3d.np.intrinsics_to_fov(src_intrinsics)

    # 1. set target fov
    fov_range_absolute_min, fov_range_absolute_max = fov_range_absolute
    fov_range_relative_min, fov_range_relative_max = fov_range_relative
    tgt_fov_x_min = min(fov_range_relative_min * raw_fov_x, utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_min * raw_fov_y) / tgt_aspect))
    tgt_fov_x_max = min(fov_range_relative_max * raw_fov_x, utils3d.focal_to_fov(utils3d.fov_to_focal(fov_range_relative_max * raw_fov_y) / tgt_aspect))
    tgt_fov_x_min, tgt_fov_max = max(np.deg2rad(fov_range_absolute_min), tgt_fov_x_min), min(np.deg2rad(fov_range_absolute_max), tgt_fov_x_max)
    tgt_fov_x = rng.uniform(min(tgt_fov_x_min, tgt_fov_x_max), tgt_fov_x_max)
    tgt_fov_y = utils3d.focal_to_fov(utils3d.np.fov_to_focal(tgt_fov_x) * tgt_aspect)

    # 2. set target image center (principal point) and the corresponding z-direction in raw camera space
    center_dtheta = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_x - tgt_fov_x)
    center_dphi = center_augmentation * rng.uniform(-0.5, 0.5) * (raw_fov_y - tgt_fov_y)
    cu, cv = 0.5 + 0.5 * np.tan(center_dtheta) / np.tan(raw_fov_x / 2), 0.5 + 0.5 *  np.tan(center_dphi) / np.tan(raw_fov_y / 2)
    direction = utils3d.np.unproject_cv(np.array([[cu, cv]], dtype=np.float32), np.array([1.0], dtype=np.float32), intrinsics=src_intrinsics)[0]

    # 3. obtain the rotation matrix for homography warping (new_ext = R * old_ext)
    R = utils3d.np.rotation_matrix_from_vectors(direction, np.array([0, 0, 1], dtype=np.float32))

    # 4. shrink the target view to fit into the warped image
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    corners = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1) @ (np.linalg.inv(src_intrinsics).T @ R.T)   # corners in viewport's camera plane
    corners = corners[:, :2] / corners[:, 2:3]
    tgt_horizontal, tgt_vertical = np.tan(tgt_fov_x / 2) * 2, np.tan(tgt_fov_y / 2) * 2
    warp_horizontal, warp_vertical = float('inf'), float('inf')
    for i in range(4):
        intersection, _ = utils3d.np.ray_intersection(
            np.array([0., 0.]), np.array([[tgt_aspect, 1.0], [tgt_aspect, -1.0]]),
            corners[i - 1], corners[i] - corners[i - 1],
        )
        warp_horizontal, warp_vertical = min(warp_horizontal, 2 * np.abs(intersection[:, 0]).min()), min(warp_vertical, 2 * np.abs(intersection[:, 1]).min())
    tgt_horizontal, tgt_vertical = min(tgt_horizontal, warp_horizontal), min(tgt_vertical, warp_vertical)

    # 5. obtain the target intrinsics
    fx, fy = 1 / tgt_horizontal, 1 / tgt_vertical
    tgt_intrinsics = utils3d.np.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).astype(np.float32)

    return tgt_intrinsics, R


def warp_perspective(
    src_map: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    tgt_size: Optional[Tuple[int, int]] = None,
    interpolation: Literal['nearest', 'bilinear', 'lanczos'] = 'nearest',
    sparse_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warp an image-like array through a normalized planar homography.

    Lanczos downsampling first reduces the source to avoid aliasing. Sparse
    nearest-neighbor input optionally uses mask-aware pre-resizing so isolated
    samples are not discarded.

    Args:
        src_map: Source array ``[H,W]`` or ``[H,W,C]``.
        transform: Normalized 3x3 homography satisfying
            ``p_target = transform @ p_source``.
        tgt_size: Output tuple ``(height,width)``.
        interpolation: ``nearest``, ``bilinear``, or ``lanczos``.
        sparse_mask: Optional boolean source support ``[H,W]`` for sparse
            nearest-neighbor maps.

    Returns:
        Warped array ``[H_t,W_t]`` or ``[H_t,W_t,C]``.
    """

    tgt_height, tgt_width = tgt_size
    src_height, src_width = src_map.shape[:2]

    # source to target transform
    transform_pixel = np.array([[tgt_width, 0, -0.5], [0, tgt_height, -0.5], [0, 0, 1]], dtype=np.float32) @ transform @ np.array([[1 / src_width, 0, 0.5 / src_width], [0, 1 / src_height, 0.5 / src_height], [0, 0, 1]], dtype=np.float32)
    # Get scale factor at the target center
    w = np.dot(np.linalg.inv(transform_pixel)[2, :], np.array([tgt_width / 2, tgt_height / 2, 1], dtype=np.float32))
    scale_x, scale_y = w * np.linalg.norm(transform_pixel[:2, :2], axis=0)

    if interpolation == 'lanczos' and (scale_x < 0.8 or scale_y < 0.8):
        # If lanczos & downsampling, use PIL to resize first to reduce aliasing
        src_height, src_width = max(round(src_height * scale_y * 1.25), 16), max(round(src_width * scale_x * 1.25), 16)
        src_map = np.array(Image.fromarray(src_map).resize((src_width, src_height), Image.Resampling.LANCZOS))
    elif interpolation == 'nearest' and sparse_mask is not None and (scale_x < 1 or scale_y < 1):
        # If nearest and sparse, use mask-aware nearest resize first to avoid losing points
        src_height, src_width = max(round(src_height * scale_y), 16), max(round(src_width * scale_x), 16)
        src_map, _ = utils3d.np.masked_nearest_resize(src_map, mask=sparse_mask, size=(src_height, src_width))

    # Recompute the pixel-space transform after resizing
    transform_pixel = np.array([[tgt_width, 0, -0.5], [0, tgt_height, -0.5], [0, 0, 1]], dtype=np.float32) @ transform @ np.array([[1 / src_width, 0, 0.5 / src_width], [0, 1 / src_height, 0.5 / src_height], [0, 0, 1]], dtype=np.float32)

    # Remap
    cv2_interpolation = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'lanczos': cv2.INTER_LANCZOS4}[interpolation]
    tgt_map = cv2.warpPerspective(src_map, transform_pixel, (tgt_width, tgt_height), flags=cv2_interpolation)

    return tgt_map


def crop_resize_view(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    src_intrinsics: np.ndarray,
    tgt_size: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    random_crop: bool = True,
    image_interpolation: Literal['nearest', 'bilinear', 'lanczos'] = 'lanczos',
    depth_interpolation: Literal['nearest', 'mixed'] = 'nearest',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a source view and resize it to an exact target resolution.

    Large images are cropped directly at target size. Smaller images first take
    the largest crop matching target aspect ratio and then resize. Normalized
    intrinsics are updated for crop geometry; a full-image resize alone does not
    alter normalized values.

    Args:
        src_image: RGB uint8 image ``[H_s,W_s,3]``.
        src_depth: Depth map ``[H_s,W_s]`` with NaN/Inf semantics.
        src_intrinsics: Normalized source matrix ``[3,3]``.
        tgt_size: Output tuple ``(height,width)``.
        rng: Optional random generator for crop offsets.
        random_crop: Randomize offsets instead of center cropping.
        image_interpolation: RGB resampling method.
        depth_interpolation: ``nearest`` or edge-aware ``mixed``.

    Returns:
        image: RGB array ``[H_t,W_t,3]``.
        depth: Depth array ``float32 [H_t,W_t]``.
        intrinsics: Updated normalized matrix ``float32 [3,3]``.
    """
    if depth_interpolation not in {'nearest', 'mixed'}:
        raise ValueError(f"crop_resize_view only supports nearest/mixed depth interpolation, got {depth_interpolation}.")

    tgt_height, tgt_width = tgt_size
    src_height, src_width = src_image.shape[:2]
    if src_height <= 0 or src_width <= 0 or tgt_height <= 0 or tgt_width <= 0:
        raise ValueError(
            f"Invalid source/target size: src=({src_height}, {src_width}), tgt=({tgt_height}, {tgt_width})."
        )

    if src_width >= tgt_width and src_height >= tgt_height:
        crop_width = tgt_width
        crop_height = tgt_height
    else:
        tgt_aspect = tgt_width / tgt_height
        if src_width / src_height >= tgt_aspect:
            crop_height = src_height
            crop_width = max(1, min(src_width, int(np.floor(src_height * tgt_aspect))))
        else:
            crop_width = src_width
            crop_height = max(1, min(src_height, int(np.floor(src_width / tgt_aspect))))

    max_x = max(0, src_width - crop_width)
    max_y = max(0, src_height - crop_height)
    if random_crop:
        if rng is None:
            rng = np.random.default_rng()
        x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    else:
        x0 = max_x // 2
        y0 = max_y // 2

    x1, y1 = x0 + crop_width, y0 + crop_height
    cropped_image = src_image[y0:y1, x0:x1]
    cropped_depth = src_depth[y0:y1, x0:x1]

    if crop_width != tgt_width or crop_height != tgt_height:
        if image_interpolation == 'lanczos':
            tgt_image = np.array(
                Image.fromarray(cropped_image).resize((tgt_width, tgt_height), Image.Resampling.LANCZOS)
            )
        else:
            cv2_interpolation = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
            }[image_interpolation]
            tgt_image = cv2.resize(cropped_image, (tgt_width, tgt_height), interpolation=cv2_interpolation)

        cropped_valid = np.isfinite(cropped_depth)
        cropped_depth_values = np.where(cropped_valid, cropped_depth, 0).astype(np.float32)
        tgt_depth_nearest = cv2.resize(cropped_depth_values, (tgt_width, tgt_height), interpolation=cv2.INTER_NEAREST)
        tgt_depth_valid = cv2.resize(cropped_valid.astype(np.uint8), (tgt_width, tgt_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        if depth_interpolation == 'mixed':
            depth_edge_mask = utils3d.np.depth_map_edge(cropped_depth, mask=cropped_valid, kernel_size=5, ltol=0.005)
            depth_bilinear_mask = cropped_valid & ~depth_edge_mask
            tgt_depth_bilinear_mask = cv2.resize(
                depth_bilinear_mask.astype(np.float32),
                (tgt_width, tgt_height),
                interpolation=cv2.INTER_LINEAR,
            )
            cropped_disp = np.where(cropped_valid & (cropped_depth > 0), 1.0 / cropped_depth, 0.0).astype(np.float32)
            tgt_disp_bilinear = cv2.resize(cropped_disp, (tgt_width, tgt_height), interpolation=cv2.INTER_LINEAR)
            tgt_depth_bilinear = np.where(tgt_disp_bilinear > 0, 1.0 / tgt_disp_bilinear, np.inf).astype(np.float32)
            tgt_depth = np.where(tgt_depth_bilinear_mask == 1.0, tgt_depth_bilinear, tgt_depth_nearest)
        else:
            tgt_depth = tgt_depth_nearest

        tgt_depth = np.where(tgt_depth_valid, tgt_depth, np.inf).astype(np.float32)
    else:
        tgt_image = cropped_image.copy()
        tgt_depth = cropped_depth.copy().astype(np.float32)

    tgt_intrinsics = src_intrinsics.astype(np.float32).copy()
    tgt_intrinsics[0, 0] = src_intrinsics[0, 0] * src_width / crop_width
    tgt_intrinsics[0, 1] = src_intrinsics[0, 1] * src_width / crop_width
    tgt_intrinsics[0, 2] = (src_intrinsics[0, 2] * src_width - x0) / crop_width
    tgt_intrinsics[1, 0] = 0.0
    tgt_intrinsics[1, 1] = src_intrinsics[1, 1] * src_height / crop_height
    tgt_intrinsics[1, 2] = (src_intrinsics[1, 2] * src_height - y0) / crop_height
    tgt_intrinsics[2, 0] = 0.0
    tgt_intrinsics[2, 1] = 0.0
    tgt_intrinsics[2, 2] = 1.0

    return tgt_image, tgt_depth, tgt_intrinsics


def resize_then_crop_view(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    src_intrinsics: np.ndarray,
    resize_size: Tuple[int, int],
    crop_size: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    random_crop: bool = True,
    image_interpolation: Literal['nearest', 'bilinear', 'area'] = 'area',
    depth_interpolation: Literal['nearest'] = 'nearest',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize the full view first, then crop an exact output window.

    Args:
        src_image: RGB uint8 image ``[H_s,W_s,3]``.
        src_depth: Depth map ``[H_s,W_s]``.
        src_intrinsics: Normalized source matrix ``[3,3]``.
        resize_size: Intermediate tuple ``(height,width)``.
        crop_size: Final tuple ``(height,width)`` no larger than resize size.
        rng: Optional random generator for crop offsets.
        random_crop: Randomize offsets instead of center cropping.
        image_interpolation: Intermediate RGB resampling method.
        depth_interpolation: Depth method; only nearest is supported.

    Returns:
        image: Cropped RGB ``[H_c,W_c,3]``.
        depth: Cropped depth ``float32 [H_c,W_c]``.
        intrinsics: Crop-adjusted normalized matrix ``float32 [3,3]``.
    """
    if depth_interpolation != 'nearest':
        raise ValueError(f"resize_then_crop_view only supports nearest depth interpolation, got {depth_interpolation}.")

    resize_height, resize_width = resize_size
    crop_height, crop_width = crop_size
    src_height, src_width = src_image.shape[:2]
    if src_height <= 0 or src_width <= 0 or resize_height <= 0 or resize_width <= 0 or crop_height <= 0 or crop_width <= 0:
        raise ValueError(
            f"Invalid source/resize/crop size: src=({src_height}, {src_width}), "
            f"resize=({resize_height}, {resize_width}), crop=({crop_height}, {crop_width})."
        )
    if crop_height > resize_height or crop_width > resize_width:
        raise ValueError(
            f"Crop size ({crop_height}, {crop_width}) must not exceed resize size ({resize_height}, {resize_width})."
        )

    image_cv2_interpolation = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
    }[image_interpolation]
    resized_image = cv2.resize(src_image, (resize_width, resize_height), interpolation=image_cv2_interpolation)

    resized_depth = cv2.resize(src_depth.astype(np.float32), (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

    max_x = max(0, resize_width - crop_width)
    max_y = max(0, resize_height - crop_height)
    if random_crop:
        if rng is None:
            rng = np.random.default_rng()
        x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    else:
        x0 = max_x // 2
        y0 = max_y // 2

    x1, y1 = x0 + crop_width, y0 + crop_height
    tgt_image = resized_image[y0:y1, x0:x1].copy()
    tgt_depth = resized_depth[y0:y1, x0:x1].copy().astype(np.float32)

    tgt_intrinsics = src_intrinsics.astype(np.float32).copy()
    tgt_intrinsics[0, 0] = src_intrinsics[0, 0] * resize_width / crop_width
    tgt_intrinsics[0, 1] = src_intrinsics[0, 1] * resize_width / crop_width
    tgt_intrinsics[0, 2] = (src_intrinsics[0, 2] * resize_width - x0) / crop_width
    tgt_intrinsics[1, 0] = 0.0
    tgt_intrinsics[1, 1] = src_intrinsics[1, 1] * resize_height / crop_height
    tgt_intrinsics[1, 2] = (src_intrinsics[1, 2] * resize_height - y0) / crop_height
    tgt_intrinsics[2, 0] = 0.0
    tgt_intrinsics[2, 1] = 0.0
    tgt_intrinsics[2, 2] = 1.0

    return tgt_image, tgt_depth, tgt_intrinsics


def resize_to_cover_center_crop_view(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    src_intrinsics: np.ndarray,
    tgt_size: Tuple[int, int],
    image_interpolation: Literal['nearest', 'bilinear', 'lanczos'] = 'lanczos',
    depth_interpolation: Literal['nearest', 'mixed'] = 'mixed',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize a view to cover the target, then take a centered crop.

    Args:
        src_image: RGB uint8 image ``[H_s,W_s,3]``.
        src_depth: Depth map ``[H_s,W_s]`` with invalid sentinels.
        src_intrinsics: Normalized source matrix ``[3,3]``.
        tgt_size: Output tuple ``(height,width)``.
        image_interpolation: RGB resampling method.
        depth_interpolation: ``nearest`` or edge-aware ``mixed``.

    Returns:
        image: Center-cropped RGB ``[H_t,W_t,3]``.
        depth: Center-cropped depth ``float32 [H_t,W_t]``.
        intrinsics: Resize/crop-adjusted normalized matrix ``float32 [3,3]``.
    """
    if depth_interpolation not in {'nearest', 'mixed'}:
        raise ValueError(
            f"resize_to_cover_center_crop_view only supports nearest/mixed depth interpolation, got {depth_interpolation}."
        )

    tgt_height, tgt_width = tgt_size
    src_height, src_width = src_image.shape[:2]
    if src_height <= 0 or src_width <= 0 or tgt_height <= 0 or tgt_width <= 0:
        raise ValueError(
            f"Invalid source/target size: src=({src_height}, {src_width}), tgt=({tgt_height}, {tgt_width})."
        )

    scale = max(tgt_width / src_width, tgt_height / src_height)
    resized_width = max(tgt_width, int(round(src_width * scale)))
    resized_height = max(tgt_height, int(round(src_height * scale)))

    if image_interpolation == 'lanczos':
        resized_image = np.array(
            Image.fromarray(src_image).resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        )
    else:
        cv2_interpolation = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
        }[image_interpolation]
        resized_image = cv2.resize(src_image, (resized_width, resized_height), interpolation=cv2_interpolation)

    resized_valid = np.isfinite(src_depth)
    resized_depth_values = cv2.resize(
        np.where(resized_valid, src_depth, 0).astype(np.float32),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_depth_valid = cv2.resize(
        resized_valid.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized_depth = np.where(resized_depth_valid, resized_depth_values, np.inf).astype(np.float32)

    if depth_interpolation == 'mixed':
        depth_edge_mask = utils3d.np.depth_map_edge(src_depth, mask=np.isfinite(src_depth), kernel_size=5, ltol=0.005)
        depth_bilinear_mask = np.isfinite(src_depth) & ~depth_edge_mask
        resized_bilinear_mask = cv2.resize(
            depth_bilinear_mask.astype(np.float32),
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        disp = np.where(np.isfinite(src_depth) & (src_depth > 0), 1.0 / src_depth, 0.0).astype(np.float32)
        resized_disp = cv2.resize(disp, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        resized_depth_bilinear = np.where(resized_disp > 0, 1.0 / resized_disp, np.inf).astype(np.float32)
        resized_depth = np.where(resized_bilinear_mask == 1.0, resized_depth_bilinear, resized_depth)
        resized_depth = np.where(resized_depth_valid, resized_depth, np.inf).astype(np.float32)

    x0 = max(0, (resized_width - tgt_width) // 2)
    y0 = max(0, (resized_height - tgt_height) // 2)
    x1, y1 = x0 + tgt_width, y0 + tgt_height

    tgt_image = resized_image[y0:y1, x0:x1].copy()
    tgt_depth = resized_depth[y0:y1, x0:x1].copy().astype(np.float32)

    tgt_intrinsics = src_intrinsics.astype(np.float32).copy()
    tgt_intrinsics[0, 0] = src_intrinsics[0, 0] * resized_width / tgt_width
    tgt_intrinsics[0, 1] = src_intrinsics[0, 1] * resized_width / tgt_width
    tgt_intrinsics[0, 2] = (src_intrinsics[0, 2] * resized_width - x0) / tgt_width
    tgt_intrinsics[1, 0] = 0.0
    tgt_intrinsics[1, 1] = src_intrinsics[1, 1] * resized_height / tgt_height
    tgt_intrinsics[1, 2] = (src_intrinsics[1, 2] * resized_height - y0) / tgt_height
    tgt_intrinsics[2, 0] = 0.0
    tgt_intrinsics[2, 1] = 0.0
    tgt_intrinsics[2, 2] = 1.0

    return tgt_image, tgt_depth, tgt_intrinsics


def resize_view(
    src_image: np.ndarray,
    src_depth: np.ndarray,
    src_intrinsics: np.ndarray,
    tgt_size: Tuple[int, int],
    image_interpolation: Literal['nearest', 'bilinear', 'lanczos'] = 'lanczos',
    depth_interpolation: Literal['nearest', 'mixed'] = 'mixed',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize RGB and depth directly to a target width and height.

    Args:
        src_image: RGB uint8 image ``[H_s,W_s,3]``.
        src_depth: Depth map ``[H_s,W_s]`` with invalid sentinels.
        src_intrinsics: Normalized source matrix ``[3,3]``.
        tgt_size: Exact output tuple ``(height,width)``.
        image_interpolation: RGB resampling method.
        depth_interpolation: ``nearest`` or edge-aware ``mixed``.

    Returns:
        image: Resized RGB ``[H_t,W_t,3]``.
        depth: Resized depth ``float32 [H_t,W_t]``.
        intrinsics: Normalized matrix ``float32 [3,3]``. Direct full-image
            resizing leaves normalized focal lengths and center unchanged.
    """
    if depth_interpolation not in {'nearest', 'mixed'}:
        raise ValueError(f"resize_view only supports nearest/mixed depth interpolation, got {depth_interpolation}.")

    tgt_height, tgt_width = tgt_size
    src_height, src_width = src_image.shape[:2]
    if src_height <= 0 or src_width <= 0 or tgt_height <= 0 or tgt_width <= 0:
        raise ValueError(
            f"Invalid source/target size: src=({src_height}, {src_width}), tgt=({tgt_height}, {tgt_width})."
        )

    if image_interpolation == 'lanczos':
        tgt_image = np.array(Image.fromarray(src_image).resize((tgt_width, tgt_height), Image.Resampling.LANCZOS))
    else:
        cv2_interpolation = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
        }[image_interpolation]
        tgt_image = cv2.resize(src_image, (tgt_width, tgt_height), interpolation=cv2_interpolation)

    src_valid = np.isfinite(src_depth)
    tgt_depth_values = cv2.resize(
        np.where(src_valid, src_depth, 0).astype(np.float32),
        (tgt_width, tgt_height),
        interpolation=cv2.INTER_NEAREST,
    )
    tgt_depth_valid = cv2.resize(
        src_valid.astype(np.uint8),
        (tgt_width, tgt_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    tgt_depth = np.where(tgt_depth_valid, tgt_depth_values, np.inf).astype(np.float32)

    if depth_interpolation == 'mixed':
        depth_edge_mask = utils3d.np.depth_map_edge(src_depth, mask=src_valid, kernel_size=5, ltol=0.005)
        depth_bilinear_mask = src_valid & ~depth_edge_mask
        tgt_depth_bilinear_mask = cv2.resize(
            depth_bilinear_mask.astype(np.float32),
            (tgt_width, tgt_height),
            interpolation=cv2.INTER_LINEAR,
        )
        src_disp = np.where(src_valid & (src_depth > 0), 1.0 / src_depth, 0.0).astype(np.float32)
        tgt_disp_bilinear = cv2.resize(src_disp, (tgt_width, tgt_height), interpolation=cv2.INTER_LINEAR)
        tgt_depth_bilinear = np.where(tgt_disp_bilinear > 0, 1.0 / tgt_disp_bilinear, np.inf).astype(np.float32)
        tgt_depth = np.where(tgt_depth_bilinear_mask == 1.0, tgt_depth_bilinear, tgt_depth)
        tgt_depth = np.where(tgt_depth_valid, tgt_depth, np.inf).astype(np.float32)

    tgt_intrinsics = src_intrinsics.astype(np.float32).copy()
    tgt_intrinsics[1, 0] = 0.0
    tgt_intrinsics[2, 0] = 0.0
    tgt_intrinsics[2, 1] = 0.0
    tgt_intrinsics[2, 2] = 1.0

    return tgt_image, tgt_depth, tgt_intrinsics


def disk_kernel(radius: int) -> np.ndarray:
    """Generate a normalized circular convolution kernel.

    Args:
        radius: Nonnegative disk radius in pixels.

    Returns:
        Float32 kernel ``[2*radius+1,2*radius+1]`` summing to one.
    """
    # Create coordinate grid centered at (0,0)
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    # Generate disk: region inside circle with radius R is 1
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply a circular point-spread function with FFT convolution.

    Args:
        image: Scalar ``[H,W]`` or channel image ``[H,W,C]``.
        radius: Nonnegative blur radius in pixels.

    Returns:
        Blurred floating array with the same shape as ``image``.
    """
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    if image.ndim == 2:
        blurred = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            blurred_channel = fftconvolve(image[..., i], kernel, mode='same')
            channels.append(blurred_channel)
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D or 3D.")
    return blurred


def depth_of_field(
    img: np.ndarray,
    disp: np.ndarray,
    focus_disp : float,
    max_blur_radius : int = 10,
) -> np.ndarray:
    """Synthesize depth of field from a disparity map and focus plane.

    Args:
        img: RGB image ``[H,W,3]``.
        disp: Positive disparity map ``[H,W]`` aligned to ``img``.
        focus_disp: Disparity value lying on the simulated focus plane.
        max_blur_radius: Largest circular blur radius in pixels.

    Returns:
        Depth-of-field image with shape ``[H,W,3]`` and ``img`` dtype.
    """
    # Precalculate dialated depth map for each blur radius
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp
    dilated_disp = []
    for radius in range(max_blur_radius + 1):
        dilated_disp.append(cv2.dilate(disp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)), iterations=1))

    # Determine the blur radius for each pixel based on the depth map
    blur_radii = np.clip(np.abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
    for radius in range(max_blur_radius + 1):
        dialted_blur_radii = np.clip(np.abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
        mask = (dialted_blur_radii >= radius) & (dialted_blur_radii >= blur_radii) & (dilated_disp[radius] > disp)
        blur_radii[mask] = dialted_blur_radii[mask]
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = cv2.blur(blur_radii, (5, 5))

    # Precalculate the blured image for each blur radius
    unique_radii = np.unique(blur_radii)
    precomputed = {}
    for radius in range(max_blur_radius + 1):
        if radius not in unique_radii:
            continue
        precomputed[radius] = disk_blur(img, radius)

    # Composit the blured image for each pixel
    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]

    return output
