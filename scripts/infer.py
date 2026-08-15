#!/usr/bin/env python3
"""Command-line PXDepth inference for individual images or directories.

Inputs are resized with the selected fixed-size or equal-area policy, passed
through the model, and restored to source resolution. The script can save RGB,
masked depth, finite masks, and aligned colored point clouds without requiring
benchmark ground truth.
"""

import sys
from pathlib import Path
from typing import Optional

import click

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True), required=True)
@click.option("--output", "output_path", type=click.Path(), default="output")
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--device", default="cuda:0")
@click.option("--input-size", default="1022x770", show_default=True, help="Model input WIDTHxHEIGHT.")
@click.option(
    "--resize-by-area/--fixed-size",
    default=True,
    show_default=True,
    help="Preserve aspect ratio at the area specified by --input-size.",
)
@click.option("--fp16", is_flag=True)
@click.option("--fp32", is_flag=True)
@click.option("--fov-x", type=float, default=None, help="Optional horizontal FoV in degrees.")
def main(
    input_path: str,
    output_path: str,
    checkpoint: str,
    device: str,
    input_size: Optional[str],
    resize_by_area: bool,
    fp16: bool,
    fp32: bool,
    fov_x: Optional[float],
) -> None:
    """Infer depth and colored point clouds for one image or an image tree.

    Args:
        input_path: Input image path or directory recursively searched for common
            JPEG/PNG extensions.
        output_path: Root directory mirroring the input tree by image stem.
        checkpoint: Public PXDepth ``model.pt`` checkpoint.
        device: PyTorch device string such as ``cuda:0`` or ``cpu``.
        input_size: ``WIDTHxHEIGHT`` model-input specification. Defaults to
            ``1022x770``.
        resize_by_area: Preserve source aspect ratio while matching input area.
            Enabled by default.
        fp16: Use FP16 in attention-heavy model regions.
        fp32: Force full-precision execution; mutually exclusive with FP16.
        fov_x: Optional horizontal field of view in degrees. If omitted, the
            optional MoGe-2 reference supplies intrinsics and scale.

    Returns:
        ``None``. Each output directory receives ``image.jpg``, ``mask.png``,
        colorized ``depth.png``, and aligned ``points.ply``.
    """
    import cv2
    import numpy as np
    import torch
    import utils3d
    from tqdm import tqdm

    from pxdepth.inference import parse_size, resize_image, resize_map
    from pxdepth.model import PXDepth
    from pxdepth.utils.ply import write_point_cloud_ply
    from pxdepth.utils.vis import colorize_depth

    if fp16 and fp32:
        raise click.ClickException("--fp16 and --fp32 are mutually exclusive.")
    size = parse_size(input_size)
    if resize_by_area and size is None:
        raise click.ClickException("--resize-by-area requires --input-size.")

    model = PXDepth.from_pretrained(checkpoint, strict=True).to(device).eval()
    if fp32:
        model.float()
        if torch.device(device).type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    source = Path(input_path).expanduser()
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = sorted(path for path in source.rglob("*") if path.suffix in suffixes) if source.is_dir() else [source]
    if not images:
        raise click.ClickException(f"No images found under {source}")
    output_root = Path(output_path)

    for path in tqdm(images, desc="Inference"):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            click.echo(f"Skip unreadable image: {path}", err=True)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        image = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        model_image, _ = resize_image(image, size, resize_by_area, model.patch_size)
        result = model.infer(
            model_image,
            fov_x=fov_x,
            ref_image=image,
            apply_mask=False,
            use_fp16=fp16,
            use_fp32=fp32,
        )

        depth = resize_map(result["depth"], original_size)
        mask = resize_map(result["mask"], original_size, is_mask=True)
        intrinsics = result["intrinsics"].detach().float()
        valid = torch.isfinite(depth) & (depth > 0)
        points = utils3d.pt.depth_map_to_point_map(
            torch.where(valid, depth, torch.zeros_like(depth)),
            intrinsics=intrinsics,
        )
        points = torch.where(valid[..., None], points, torch.inf)

        relative = path.relative_to(source).parent if source.is_dir() else Path()
        directory = output_root / relative / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "image.jpg"), image_bgr)
        mask_np = mask.detach().cpu().numpy().astype(bool)
        cv2.imwrite(str(directory / "mask.png"), mask_np.astype(np.uint8) * 255)
        depth_np = depth.detach().cpu().numpy()
        depth_vis = colorize_depth(np.where(mask_np, depth_np, np.inf), mask=None)
        cv2.imwrite(str(directory / "depth.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        points_np = points.detach().cpu().numpy().reshape(-1, 3)
        colors = image_rgb.reshape(-1, 3).astype(np.float32) / 255.0
        point_mask = mask_np.reshape(-1) & np.isfinite(points_np).all(axis=1)
        write_point_cloud_ply(directory / "points.ply", points_np[point_mask], colors[point_mask])


if __name__ == "__main__":
    main()
