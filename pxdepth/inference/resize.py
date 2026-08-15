"""Image and prediction resizing shared by evaluation and inference.

The functions parse user-facing sizes, derive patch-compatible equal-area
shapes, resize RGB tensors for a model, and restore depth or mask maps to the
source resolution. Their return values retain the original image dimensions so
camera-normalized geometry remains consistent after restoration.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def parse_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a CLI image-size string in ``WIDTHxHEIGHT`` notation.

    Args:
        value: Size string, or ``None``/empty text when no target is requested.

    Returns:
        Positive integer tuple ``(width, height)``, or ``None``.
    """
    if value is None or not str(value).strip():
        return None
    parts = str(value).lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("Image size must be WIDTHxHEIGHT, for example 1022x770.")
    width, height = map(int, parts)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    return width, height


def area_size(
    height: int,
    width: int,
    target_width: int,
    target_height: int,
    patch_size: int,
) -> Tuple[int, int]:
    """Preserve aspect ratio while matching a reference width-height area.

    Args:
        height: Original image height ``H``.
        width: Original image width ``W``.
        target_width: Width defining the desired reference area.
        target_height: Height defining the desired reference area.
        patch_size: Required divisibility of both output dimensions.

    Returns:
        Integer ``(new_height, new_width)`` with approximately
        ``target_width*target_height`` pixels and the original aspect ratio.
    """
    area = int(target_width) * int(target_height)
    return area_size_from_area(height, width, area, patch_size)


def area_size_from_area(
    height: int,
    width: int,
    target_area: int,
    patch_size: int,
) -> Tuple[int, int]:
    """Preserve aspect ratio while matching an explicit target pixel area.

    Args:
        height: Original image height ``H``.
        width: Original image width ``W``.
        target_area: Desired number of input pixels before patch rounding.
        patch_size: Required divisibility of both output dimensions.

    Returns:
        Integer ``(new_height, new_width)`` rounded to patch multiples.
    """
    area = int(target_area)
    if height <= 0 or width <= 0 or area <= 0:
        raise ValueError("Image dimensions and target area must be positive.")
    aspect = width / height
    new_width = int(round((area * aspect) ** 0.5))
    new_height = int(round(new_width / aspect))
    new_width = max(patch_size, int(round(new_width / patch_size)) * patch_size)
    new_height = max(patch_size, int(round(new_height / patch_size)) * patch_size)
    return new_height, new_width


def patch_size(height: int, width: int, patch: int) -> Tuple[int, int]:
    """Round spatial dimensions down to valid patch multiples.

    Args:
        height: Original image height.
        width: Original image width.
        patch: Positive encoder patch side length.

    Returns:
        Integer ``(new_height, new_width)``. Already divisible dimensions are
        unchanged; smaller results are clamped to one patch.
    """
    new_height = height if height % patch == 0 else max(patch, height // patch * patch)
    new_width = width if width % patch == 0 else max(patch, width // patch * patch)
    return new_height, new_width


def resize_image(
    image: torch.Tensor,
    target: Optional[Tuple[int, int]],
    resize_by_area: bool,
    patch: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize a CHW/BCHW image to the model input resolution.

    Args:
        image: RGB tensor ``[3,H,W]`` or ``[B,3,H,W]``.
        target: Optional ``(width,height)`` reference size.
        resize_by_area: Preserve aspect ratio and use only ``target`` area when
            ``True``; otherwise force the exact target dimensions.
        patch: Required encoder patch divisibility.

    Returns:
        resized: Bilinearly resized tensor preserving the input batch layout.
        original_size: Original integer tuple ``(H,W)``.
    """
    original = tuple(image.shape[-2:])
    if target is None:
        height, width = patch_size(*original, patch)
    elif resize_by_area:
        height, width = area_size(*original, target[0], target[1], patch)
    else:
        width, height = target
        if height % patch or width % patch:
            raise ValueError(f"Fixed input size {width}x{height} must be divisible by patch size {patch}.")
    if (height, width) == original:
        return image, original
    batched = image.ndim == 4
    source = image if batched else image.unsqueeze(0)
    resized = F.interpolate(source, (height, width), mode="bilinear", align_corners=False)
    return (resized if batched else resized[0]), original


def resize_map(value: torch.Tensor, size: Tuple[int, int], is_mask: bool = False) -> torch.Tensor:
    """Nearest-resize a depth/probability map while preserving batch layout.

    Args:
        value: Map tensor ``[H,W]`` or batch ``[B,H,W]``.
        size: Target ``(height,width)``.
        is_mask: Threshold resized values at ``0.5`` and return boolean output.

    Returns:
        Tensor ``[H_t,W_t]`` or ``[B,H_t,W_t]``. Non-mask output is floating;
        mask output is boolean.
    """
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    batched = value.ndim == 3
    source = value.float().unsqueeze(1) if batched else value.float()[None, None]
    output = F.interpolate(source, size=size, mode="nearest")
    output = output[:, 0] if batched else output[0, 0]
    return output > 0.5 if is_mask else output
