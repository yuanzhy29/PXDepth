"""Asynchronous RGB-depth benchmark loader with geometry-aware resizing.

Evaluation samples follow the processed benchmark directory contract. This
module loads RGB, depth, normalized intrinsics, and optional segmentation,
applies the benchmark-configured view transformation, and returns aligned
PyTorch tensors plus an organized ground-truth point map.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import utils3d
import pipeline

from ..utils.io import read_depth, read_image, read_json, read_segmentation
from ..utils.data_augmentation import sample_perspective, warp_perspective, resize_to_cover_center_crop_view


def _resolve_image_path(instance_path: Union[str, Path]) -> Path:
    """Resolve a processed sample's RGB path with PNG precedence.

    Args:
        instance_path: Directory containing one processed benchmark sample.

    Returns:
        ``image.png`` when present, otherwise ``image.jpg``.
    """
    instance_path = Path(instance_path)
    image_png = instance_path / 'image.png'
    if image_png.exists():
        return image_png
    return instance_path / 'image.jpg'


def _resize_to_cover_center_crop_mask(mask: np.ndarray, raw_width: int, raw_height: int, tgt_width: int, tgt_height: int) -> np.ndarray:
    """Apply resize-to-cover and center crop to a discrete label mask.

    Args:
        mask: Integer/boolean source mask ``[H_raw,W_raw]``.
        raw_width: Source image width.
        raw_height: Source image height.
        tgt_width: Output width.
        tgt_height: Output height.

    Returns:
        Nearest-resized and center-cropped mask ``[H_tgt,W_tgt]``.
    """
    scale = max(tgt_width / raw_width, tgt_height / raw_height)
    resized_width = max(tgt_width, int(round(raw_width * scale)))
    resized_height = max(tgt_height, int(round(raw_height * scale)))
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    x0 = max(0, (resized_width - tgt_width) // 2)
    y0 = max(0, (resized_height - tgt_height) // 2)
    return resized_mask[y0:y0 + tgt_height, x0:x0 + tgt_width].copy()


def _intrinsics_normalized_to_pixel(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized camera intrinsics to pixel coordinates.

    Args:
        intrinsics: Floating camera matrix ``[3,3]`` normalized by image size.
        width: Image width used to scale the first matrix row.
        height: Image height used to scale the second matrix row.

    Returns:
        Pixel-space ``float32 [3,3]`` camera matrix.
    """
    intrinsics_px = intrinsics.astype(np.float32).copy()
    intrinsics_px[0, :] *= float(width)
    intrinsics_px[1, :] *= float(height)
    intrinsics_px[2, 0] = 0.0
    intrinsics_px[2, 1] = 0.0
    intrinsics_px[2, 2] = 1.0
    return intrinsics_px


def _intrinsics_pixel_to_normalized(intrinsics_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel camera intrinsics to normalized coordinates.

    Args:
        intrinsics_px: Pixel-space camera matrix ``[3,3]``.
        width: Image width used to normalize the first matrix row.
        height: Image height used to normalize the second matrix row.

    Returns:
        Normalized ``float32 [3,3]`` camera matrix.
    """
    intrinsics = intrinsics_px.astype(np.float32).copy()
    intrinsics[0, :] /= float(width)
    intrinsics[1, :] /= float(height)
    intrinsics[2, 0] = 0.0
    intrinsics[2, 1] = 0.0
    intrinsics[2, 2] = 1.0
    return intrinsics


def _resize_depth_nearest_preserve_nan(depth: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-resize positive finite depth while preserving invalid support.

    Args:
        depth: Source depth ``float [H,W]`` with NaN/Inf invalid values.
        size: OpenCV target tuple ``(width,height)``.

    Returns:
        ``float32 [height,width]`` depth. Pixels whose nearest source was invalid
        are represented by NaN.
    """
    width, height = size
    valid = np.isfinite(depth) & (depth > 0)
    resized_depth = cv2.resize(
        np.where(valid, depth, 0.0).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_valid = cv2.resize(
        valid.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return np.where(resized_valid, resized_depth, np.nan).astype(np.float32)


def _mda_boundary_view(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    target_size: Tuple[int, int],
    segmentation_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Create the principal-point-centered view used by boundary benchmarks.

    The image is first cropped symmetrically around the principal point, then
    resized to cover the target and center-cropped. RGB uses Lanczos for
    downsampling and bicubic for upsampling; depth and segmentation use nearest
    interpolation. Pixel intrinsics are updated after every crop and resize.

    Args:
        image: RGB uint8 array ``[H,W,3]``.
        depth: Depth array ``[H,W]`` with NaN invalid values.
        intrinsics: Normalized camera matrix ``[3,3]``.
        target_size: Output ``(height,width)``.
        segmentation_mask: Optional integer labels ``[H,W]``.

    Returns:
        image: Transformed RGB ``[H_t,W_t,3]``.
        depth: Transformed depth ``float32 [H_t,W_t]``.
        intrinsics: Updated normalized matrix ``float32 [3,3]``.
        segmentation_mask: Transformed labels ``[H_t,W_t]`` or ``None``.
    """
    tgt_height, tgt_width = target_size
    raw_height, raw_width = image.shape[:2]
    intrinsics_px = _intrinsics_normalized_to_pixel(intrinsics, raw_width, raw_height)

    cx = float(intrinsics_px[0, 2])
    cy = float(intrinsics_px[1, 2])
    margin_x = max(1.0, min(cx, raw_width - cx))
    margin_y = max(1.0, min(cy, raw_height - cy))
    crop_left = max(0, int(round(cx - margin_x)))
    crop_right = min(raw_width, int(round(cx + margin_x)))
    crop_top = max(0, int(round(cy - margin_y)))
    crop_bottom = min(raw_height, int(round(cy + margin_y)))

    if crop_right - crop_left < 2 or crop_bottom - crop_top < 2:
        crop_left, crop_top = 0, 0
        crop_right, crop_bottom = raw_width, raw_height

    image = image[crop_top:crop_bottom, crop_left:crop_right].copy()
    depth = depth[crop_top:crop_bottom, crop_left:crop_right].copy()
    if segmentation_mask is not None:
        segmentation_mask = segmentation_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
    intrinsics_px[0, 2] -= float(crop_left)
    intrinsics_px[1, 2] -= float(crop_top)

    crop_height, crop_width = image.shape[:2]
    scale = max(tgt_width / crop_width, tgt_height / crop_height)
    resized_width = max(tgt_width, int(np.floor(crop_width * scale)))
    resized_height = max(tgt_height, int(np.floor(crop_height * scale)))
    if resized_width < tgt_width or resized_height < tgt_height:
        resized_width = max(tgt_width, int(np.ceil(crop_width * scale)))
        resized_height = max(tgt_height, int(np.ceil(crop_height * scale)))

    image_resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
    resized_image = np.array(Image.fromarray(image).resize((resized_width, resized_height), image_resample))
    resized_depth = _resize_depth_nearest_preserve_nan(depth, (resized_width, resized_height))
    resized_segmentation_mask = None
    if segmentation_mask is not None:
        resized_segmentation_mask = cv2.resize(
            segmentation_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
    intrinsics_px[:2, :] *= float(scale)

    x0 = int(round((resized_width - tgt_width) * 0.5))
    y0 = int(round((resized_height - tgt_height) * 0.5))
    x0 = min(max(x0, 0), resized_width - tgt_width)
    y0 = min(max(y0, 0), resized_height - tgt_height)
    x1, y1 = x0 + tgt_width, y0 + tgt_height

    tgt_image = resized_image[y0:y1, x0:x1].copy()
    tgt_depth = resized_depth[y0:y1, x0:x1].copy().astype(np.float32)
    tgt_segmentation_mask = None
    if resized_segmentation_mask is not None:
        tgt_segmentation_mask = resized_segmentation_mask[y0:y1, x0:x1].copy()
    intrinsics_px[0, 2] -= float(x0)
    intrinsics_px[1, 2] -= float(y0)
    tgt_intrinsics = _intrinsics_pixel_to_normalized(intrinsics_px, tgt_width, tgt_height)

    return tgt_image, tgt_depth, tgt_intrinsics, tgt_segmentation_mask


class EvalDataLoaderPipeline:
    """Asynchronously load and geometrically standardize one benchmark dataset.

    The pipeline emits one sample at a time. It supports exact resolutions,
    center-crop sizes, or aspect-preserving token budgets, and can optionally
    include segmentation or normal annotations for local and boundary metrics.
    """

    def __init__(
        self,
        path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        center_crop_size: Optional[int] = None,
        split: int = '.index.txt',
        drop_max_depth: float = 1000.,
        num_load_workers: int = 4,
        num_process_workers: int = 8,
        include_segmentation: bool = False,
        include_normal: bool = False,
        depth_to_normal: bool = False,
        max_segments: int = 100,
        min_seg_area: int = 1000,
        depth_unit: str = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        has_sharp_boundary = False,
        subset: int = None,
        filenames: Optional[List[str]] = None,
        num_tokens: Optional[int] = None,
        patch_size: Optional[int] = None,
        disable_augmentations: bool = True,
        disable_perspective: bool = True,
        resize_to_cover_center_crop: bool = False,
        mda_boundary_transform: bool = False,
    ):
        """Configure benchmark indexing, transforms, and worker stages.

        Args:
            path: Processed benchmark root containing the split index.
            width: Exact output width when token/crop modes are disabled.
            height: Exact output height when token/crop modes are disabled.
            center_crop_size: Optional square output side length.
            split: Relative index filename under ``path``.
            drop_max_depth: Relative dynamic-range multiplier used to suppress
                extreme finite values after transformation.
            num_load_workers: Number of parallel disk readers.
            num_process_workers: Number of parallel geometry workers.
            include_segmentation: Load ``segmentation.png`` and label metadata.
            include_normal: Derive a normal map from depth.
            depth_to_normal: Retained benchmark compatibility flag.
            max_segments: Maximum segmentation labels retained by area.
            min_seg_area: Minimum number of pixels for a retained label.
            depth_unit: Optional scalar converting stored depth to metric units.
            min_depth: Optional lower valid-depth bound in converted units.
            max_depth: Optional upper valid-depth bound in converted units.
            has_sharp_boundary: Mark samples for boundary metric computation.
            subset: Optional number of leading index entries to evaluate.
            filenames: Optional explicit relative paths replacing the split file.
            num_tokens: Optional approximate image-token count.
            patch_size: Patch divisibility used with token sampling.
            disable_augmentations: Disable random flip/color transforms.
            disable_perspective: Use identity perspective mapping.
            resize_to_cover_center_crop: Resize to cover then center crop.
            mda_boundary_transform: Use the principal-point-centered boundary
                benchmark transformation.

        Returns:
            ``None``. Workers start when the context manager is entered.
        """
        if filenames is None:
            filenames = Path(path).joinpath(split).read_text(encoding='utf-8').splitlines()
        else:
            filenames = list(filenames)
        if subset is not None:
            subset = int(subset)
            if subset > 0:
                filenames = filenames[:subset]
        self.width = int(width) if width is not None else None
        self.height = int(height) if height is not None else None
        self.center_crop_size = int(center_crop_size) if center_crop_size is not None else None
        self.drop_max_depth = drop_max_depth
        self.path = Path(path)
        self.filenames = filenames
        self.include_segmentation = include_segmentation
        self.include_normal = include_normal
        self.max_segments = max_segments
        self.min_seg_area = min_seg_area
        self.depth_to_normal = depth_to_normal
        self.depth_unit = depth_unit
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.has_sharp_boundary = has_sharp_boundary
        self.num_tokens = int(num_tokens) if num_tokens is not None else None
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.disable_augmentations = bool(disable_augmentations)
        self.disable_perspective = bool(disable_perspective)
        self.resize_to_cover_center_crop = bool(resize_to_cover_center_crop)
        self.mda_boundary_transform = bool(mda_boundary_transform)

        self.rng = np.random.default_rng(seed=0)

        self.pipeline = pipeline.Sequential([
            self._generator,
            pipeline.Parallel([self._load_instance] * num_load_workers),
            pipeline.Parallel([self._process_instance] * num_process_workers),
            pipeline.Buffer(4)
        ])

    def __len__(self):
        """Return the number of configured benchmark samples.

        The value reflects explicit filenames and optional subset truncation.

        Returns:
            Integer length of the selected filename list.
        """
        return len(self.filenames)

    def _resolve_target_size(self, raw_width: int, raw_height: int) -> Tuple[int, int]:
        """Resolve output dimensions from exact, crop, or token settings.

        Args:
            raw_width: Source image width.
            raw_height: Source image height.

        Returns:
            Integer tuple ``(target_width,target_height)``, optionally rounded to
            patch multiples.
        """
        if self.num_tokens is not None:
            if self.patch_size is None:
                raise ValueError("patch_size must be set when using num_tokens.")
            aspect_ratio = raw_width / raw_height
            target_area = self.num_tokens * (self.patch_size ** 2)
            tgt_width = int(round((target_area * aspect_ratio) ** 0.5))
            tgt_height = int(round(tgt_width / aspect_ratio))
        elif self.center_crop_size is not None:
            tgt_width = self.center_crop_size
            tgt_height = self.center_crop_size
        else:
            if self.width is None or self.height is None:
                raise ValueError("width/height or center_crop_size must be set when num_tokens is not provided.")
            tgt_width, tgt_height = self.width, self.height
        if self.patch_size is not None:
            tgt_width = max(self.patch_size, int(round(tgt_width / self.patch_size)) * self.patch_size)
            tgt_height = max(self.patch_size, int(round(tgt_height / self.patch_size)) * self.patch_size)
        return tgt_width, tgt_height

    def _generator(self):
        """Yield sequential sample indices to the asynchronous pipeline.

        Disk loading and processing are parallelized after this ordered stage.

        Yields:
            Integer indices from zero through ``len(self)-1``.
        """
        for idx in range(len(self)):
            yield idx

    def _load_instance(self, idx):
        """Read one indexed RGB-depth sample and optional segmentation.

        Args:
            idx: Integer index into ``self.filenames``.

        Returns:
            Dictionary containing RGB ``uint8 [H,W,3]``, depth ``float [H,W]``,
            normalized intrinsics ``float32 [3,3]``, masks, and optional
            segmentation; ``None`` for an out-of-range index.
        """
        if idx >= len(self.filenames):
            return None

        path = self.path.joinpath(self.filenames[idx])

        instance = {
            'filename': self.filenames[idx],
        }
        instance['image'] = read_image(_resolve_image_path(path))

        depth = read_depth(Path(path, 'depth.png'))  # ignore depth unit from depth file, use config instead
        instance.update({
            'depth': depth,
            'depth_mask': np.isfinite(depth) & (depth > 0),
            'depth_mask_inf': np.isinf(depth),
        })

        if self.include_segmentation:
            segmentation_mask, segmentation_labels = read_segmentation(Path(path,'segmentation.png'))
            instance.update({
                'segmentation_mask': segmentation_mask,
                'segmentation_labels': segmentation_labels,
            })

        meta = read_json(Path(path, 'meta.json'))
        instance['intrinsics'] = np.array(meta['intrinsics'], dtype=np.float32)

        return instance

    def _process_instance(self, instance: dict):
        """Transform one loaded instance and build its ground-truth point map.

        Args:
            instance: Raw dictionary returned by :meth:`_load_instance`, or
                ``None`` propagated from a failed/out-of-range stage.

        Returns:
            Processed dictionary with image ``float32 [3,H,W]``, depth and masks
            ``[H,W]``, normalized intrinsics ``[3,3]``, point map ``[H,W,3]``,
            metadata flags, and optional normals/segmentation tensors. Returns
            ``None`` when the input is ``None``.
        """
        if instance is None:
            return None

        image = instance['image']
        depth = instance['depth']
        intrinsics = instance['intrinsics']
        segmentation_mask = instance.get('segmentation_mask', None)
        segmentation_labels = instance.get('segmentation_labels', None)

        raw_height, raw_width = image.shape[:2]
        tgt_width, tgt_height = self._resolve_target_size(raw_width, raw_height)
        tgt_aspect = tgt_width / tgt_height

        raw_depth_mask = np.isfinite(depth) & (depth > 0)
        raw_depth_ratio = raw_depth_mask.mean()
        if raw_depth_ratio < 0.001:
            depth = np.ones_like(depth, dtype=np.float32)
            raw_depth_mask = np.isfinite(depth)
        else:
            depth = np.where(raw_depth_mask, depth, np.nan)

        if self.include_normal:
            raw_normal, raw_normal_mask = utils3d.np.depth_map_to_normal_map(
                depth, intrinsics=intrinsics, mask=raw_depth_mask, edge_threshold=88
            )
            raw_normal = np.where(raw_normal_mask[..., None], raw_normal, np.nan)
        else:
            raw_normal = None

        if self.mda_boundary_transform:
            tgt_image, tgt_depth, tgt_intrinsics, tgt_segmentation_mask = _mda_boundary_view(
                image,
                depth,
                intrinsics,
                (tgt_height, tgt_width),
                segmentation_mask=segmentation_mask,
            )
            if self.include_normal:
                tgt_normal, tgt_normal_mask = utils3d.np.depth_map_to_normal_map(
                    tgt_depth, intrinsics=tgt_intrinsics, mask=np.isfinite(tgt_depth) & (tgt_depth > 0), edge_threshold=88
                )
                tgt_normal = np.where(tgt_normal_mask[..., None], tgt_normal, np.nan)
            else:
                tgt_normal = None
        elif self.resize_to_cover_center_crop:
            tgt_image, tgt_depth, tgt_intrinsics = resize_to_cover_center_crop_view(
                image,
                depth,
                intrinsics,
                (tgt_height, tgt_width),
                image_interpolation='lanczos',
                depth_interpolation='mixed',
            )
            if self.include_normal:
                tgt_normal, tgt_normal_mask = utils3d.np.depth_map_to_normal_map(
                    tgt_depth, intrinsics=tgt_intrinsics, mask=np.isfinite(tgt_depth) & (tgt_depth > 0), edge_threshold=88
                )
                tgt_normal = np.where(tgt_normal_mask[..., None], tgt_normal, np.nan)
            else:
                tgt_normal = None
            tgt_segmentation_mask = None
            if segmentation_mask is not None:
                tgt_segmentation_mask = _resize_to_cover_center_crop_mask(
                    segmentation_mask,
                    raw_width,
                    raw_height,
                    tgt_width,
                    tgt_height,
                )
        elif self.disable_perspective:
            tgt_intrinsics = intrinsics.copy()
            R = np.eye(3, dtype=np.float32)
            transform = np.eye(3, dtype=np.float32)
        else:
            tgt_intrinsics, R = sample_perspective(
                intrinsics,
                tgt_aspect=tgt_aspect,
                center_augmentation=0.0,
                fov_range_absolute=(1, 179),
                fov_range_relative=(1.0, 1.0),
                rng=self.rng,
            )
            transform = tgt_intrinsics @ R @ np.linalg.inv(intrinsics)

        if not self.resize_to_cover_center_crop and not self.mda_boundary_transform:
            tgt_image = warp_perspective(image, transform, (tgt_height, tgt_width), interpolation='lanczos')

            depth_edge_mask = utils3d.np.depth_map_edge(depth, mask=raw_depth_mask, kernel_size=5, ltol=0.01)
            depth_bilinear_mask = raw_depth_mask & ~depth_edge_mask
            warped_depth_bilinear_mask = warp_perspective(
                depth_bilinear_mask.astype(np.float32),
                transform,
                (tgt_height, tgt_width),
                interpolation='bilinear',
            )
            warped_depth_nearest = warp_perspective(
                depth,
                transform,
                (tgt_height, tgt_width),
                interpolation='nearest',
                sparse_mask=~np.isnan(depth),
            )
            warped_depth_bilinear = 1 / warp_perspective(
                1 / depth,
                transform,
                (tgt_height, tgt_width),
                interpolation='bilinear',
            )
            warped_depth = np.where(warped_depth_bilinear_mask == 1.0, warped_depth_bilinear, warped_depth_nearest)
            tgt_uvhomo = np.concatenate(
                [utils3d.np.uv_map((tgt_height, tgt_width)), np.ones((tgt_height, tgt_width, 1), dtype=np.float32)],
                axis=-1,
            )
            tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])

            if raw_normal is not None:
                warped_normal = warp_perspective(raw_normal, transform, (tgt_height, tgt_width), interpolation='bilinear')
                tgt_normal = warped_normal @ R.T
            else:
                tgt_normal = None

            if segmentation_mask is not None:
                tgt_segmentation_mask = warp_perspective(
                    segmentation_mask, transform, (tgt_height, tgt_width), interpolation='nearest'
                )
            else:
                tgt_segmentation_mask = None

        if not self.disable_augmentations:
            if self.rng.choice([True, False]):
                tgt_image = np.flip(tgt_image, axis=1).copy()
                tgt_depth = np.flip(tgt_depth, axis=1).copy()
                if tgt_normal is not None:
                    tgt_normal = np.flip(tgt_normal, axis=1).copy() * [-1, 1, 1]

        if self.depth_unit is not None:
            tgt_depth *= self.depth_unit
            is_metric = True
        else:
            is_metric = False

        depth_range_mask = np.isfinite(tgt_depth) & (tgt_depth > 0)
        if self.min_depth is not None:
            depth_range_mask &= tgt_depth >= self.min_depth
        if self.max_depth is not None:
            depth_range_mask &= tgt_depth <= self.max_depth
        tgt_depth = np.where(depth_range_mask, tgt_depth, np.nan)

        drop_max_depth = np.nanquantile(np.where(np.isfinite(tgt_depth), tgt_depth, np.nan), 0.01) * self.drop_max_depth
        tgt_depth = np.where(np.isfinite(tgt_depth), np.clip(tgt_depth, 0, drop_max_depth), tgt_depth)

        tgt_depth_mask_inf = np.isinf(tgt_depth)
        tgt_depth_mask = np.isfinite(tgt_depth) & (tgt_depth > 0)
        if not np.any(tgt_depth_mask):
            tgt_depth_mask = np.ones_like(tgt_depth_mask)
            tgt_depth = np.ones_like(tgt_depth)

        tgt_points = utils3d.np.depth_map_to_point_map(tgt_depth, intrinsics=tgt_intrinsics)

        if self.include_segmentation and tgt_segmentation_mask is not None:
            for k in ['undefined', 'unannotated', 'background', 'sky']:
                if k in segmentation_labels:
                    del segmentation_labels[k]
            seg_id2count = dict(zip(*np.unique(tgt_segmentation_mask, return_counts=True)))
            sorted_labels = sorted(segmentation_labels.keys(), key=lambda x: seg_id2count.get(segmentation_labels[x], 0), reverse=True)
            segmentation_labels = {
                k: segmentation_labels[k]
                for k in sorted_labels[:self.max_segments]
                if seg_id2count.get(segmentation_labels[k], 0) >= self.min_seg_area
            }

        instance.update({
            'image': torch.from_numpy(tgt_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'depth': torch.from_numpy(tgt_depth).float(),
            'depth_mask': torch.from_numpy(tgt_depth_mask).bool(),
            'depth_mask_inf': torch.from_numpy(tgt_depth_mask_inf).bool(),
            'intrinsics': torch.from_numpy(tgt_intrinsics).float(),
            'points': torch.from_numpy(tgt_points).float(),
            'segmentation_mask': torch.from_numpy(tgt_segmentation_mask).long() if tgt_segmentation_mask is not None else None,
            'segmentation_labels': segmentation_labels,
            'is_metric': is_metric,
            'has_sharp_boundary': self.has_sharp_boundary,
        })
        if tgt_normal is not None:
            instance['normal'] = torch.from_numpy(tgt_normal).float()

        instance = {k: v for k, v in instance.items() if v is not None}

        return instance

    def start(self):
        """Start asynchronous loader workers.

        Call this before :meth:`get` when not using the context manager.

        Returns:
            ``None``.
        """
        self.pipeline.start()

    def stop(self):
        """Stop asynchronous loader workers and release resources.

        Any prefetched samples are discarded by the pipeline implementation.

        Returns:
            ``None``.
        """
        self.pipeline.stop()

    def __enter__(self):
        """Start the pipeline and return it as a context-manager value.

        This is equivalent to an explicit :meth:`start` call.

        Returns:
            This :class:`EvalDataLoaderPipeline` instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the pipeline when leaving its context.

        Args:
            exc_type: Exception class raised inside the context, if any.
            exc_value: Exception instance raised inside the context, if any.
            traceback: Associated traceback object, if any.

        Returns:
            ``None``; exceptions are not suppressed.
        """
        self.stop()

    def get(self):
        """Block until the next processed evaluation sample is available.

        Worker-side exceptions are surfaced by the underlying pipeline call.

        Returns:
            Processed sample dictionary documented by :meth:`_process_instance`.
        """
        return self.pipeline.get()
