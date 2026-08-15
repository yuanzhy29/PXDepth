#!/usr/bin/env python3
"""Command-line benchmark evaluation for the released PXDepth model.

The script loads a checkpoint, iterates datasets from one JSON benchmark file,
runs timed raw inference, performs protocol-specific alignment and metric
computation, and optionally writes depth, mask, RGB, and point-cloud artifacts.
Aggregated results are serialized as JSON.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _loader_config(config: Dict[str, Any], boundary: bool) -> Dict[str, Any]:
    """Create the effective loader configuration for one benchmark.

    Args:
        config: Dataset entry from ``configs/eval/all_benchmarks.json``.
        boundary: Whether the dataset requests the principal-point-centered
            boundary evaluation view.

    Returns:
        New loader dictionary. The input config is not mutated.
    """
    result = dict(config)
    result.pop("boundary_width", None)
    result.pop("boundary_height", None)
    if boundary:
        result.update(
            mda_boundary_transform=True,
            disable_augmentations=True,
            disable_perspective=True,
            resize_to_cover_center_crop=False,
            include_normal=False,
            depth_to_normal=False,
        )
        result.pop("center_crop_size", None)
        result.pop("num_tokens", None)
        result.pop("patch_size", None)
    return result


def _write_cloud(path: Path, points, image, mask=None) -> None:
    """Write finite organized points with aligned RGB colors.

    Args:
        path: Destination PLY path.
        points: NumPy point map ``[H,W,3]`` or array ``[N,3]``.
        image: RGB uint8 image ``[H,W,3]`` aligned to organized points.
        mask: Optional boolean/binary map ``[H,W]`` selecting valid output.

    Returns:
        ``None``. Invalid points are omitted from the written PLY.
    """
    import numpy as np
    from pxdepth.utils.ply import write_point_cloud_ply

    points = points.reshape(-1, 3)
    colors = image.reshape(-1, 3).astype(np.float32) / 255.0
    valid = np.isfinite(points).all(axis=1)
    if mask is not None:
        valid &= mask.reshape(-1).astype(bool)
    write_point_cloud_ply(path, points[valid], colors[valid])


def _dump(path: Path, image, pred, sample, misc, dump_pred: bool, dump_gt: bool) -> None:
    """Save prediction, ground truth, and optional boundary visualizations.

    Args:
        path: Per-sample output directory.
        image: RGB tensor ``[3,H,W]`` in ``[0,1]``.
        pred: Raw prediction dictionary, including optional mask ``[H,W]``.
        sample: Ground-truth evaluation sample with depth, mask, and points.
        misc: Aligned maps and optional boundary tensors from ``compute_metrics``.
        dump_pred: Save aligned predicted depth and point cloud.
        dump_gt: Save GT depth and point cloud.

    Returns:
        ``None``. Requested artifacts are written below ``path``.
    """
    import cv2
    import numpy as np
    from pxdepth.utils.vis import colorize_depth

    image_rgb = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    if dump_pred:
        directory = path / "pred"
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "image.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        mask = pred.get("mask")
        mask_np = None if mask is None else mask.detach().cpu().numpy().astype(bool)
        if mask_np is not None:
            cv2.imwrite(str(directory / "mask.png"), mask_np.astype(np.uint8) * 255)
        if "pred_depth" in misc:
            depth = misc["pred_depth"].detach().cpu().numpy()
            masked = np.where(mask_np, depth, np.inf) if mask_np is not None else depth
            cv2.imwrite(str(directory / "depth_wo_mask.png"), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(directory / "depth.png"), cv2.cvtColor(colorize_depth(masked), cv2.COLOR_RGB2BGR))
        if "pred_points" in misc:
            _write_cloud(directory / "points.ply", misc["pred_points"].detach().cpu().numpy(), image_rgb, mask_np)

    if dump_gt:
        directory = path / "gt"
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "image.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        depth = sample["depth"].detach().cpu().numpy()
        mask = sample["depth_mask"].detach().cpu().numpy().astype(bool)
        cv2.imwrite(str(directory / "depth.png"), cv2.cvtColor(colorize_depth(depth, mask=mask), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(directory / "mask.png"), mask.astype(np.uint8) * 255)
        _write_cloud(directory / "points.ply", sample["points"].detach().cpu().numpy(), image_rgb, mask)

    if "boundary_edge_mask" in misc:
        directory = path / "boundary"
        directory.mkdir(parents=True, exist_ok=True)
        edge = misc["boundary_edge_mask"].detach().cpu().numpy().astype(np.uint8) * 255
        cv2.imwrite(str(directory / "edge.png"), edge)
        for source, filename, color in (
            ("boundary_pred_edge_points", "pred_edge_aligned.ply", (1.0, 0.25, 0.25)),
            ("boundary_gt_edge_points", "gt_edge.ply", (0.25, 1.0, 0.25)),
        ):
            if source in misc:
                points = misc[source].detach().cpu().numpy()
                colors = np.tile(np.asarray(color, np.float32)[None], (len(points), 1))
                from pxdepth.utils.ply import write_point_cloud_ply
                write_point_cloud_ply(directory / filename, points, colors)


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default="configs/eval/all_benchmarks.json")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True)
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
@click.option("--dump-pred", is_flag=True)
@click.option("--dump-gt", is_flag=True)
def main(
    checkpoint: str,
    config_path: str,
    output_path: str,
    device: str,
    input_size: Optional[str],
    resize_by_area: bool,
    fp16: bool,
    fp32: bool,
    dump_pred: bool,
    dump_gt: bool,
) -> None:
    """Evaluate a checkpoint over all configured benchmark datasets.

    Args:
        checkpoint: Public PXDepth ``model.pt`` checkpoint.
        config_path: Benchmark JSON path.
        output_path: Destination aggregate metrics JSON path.
        device: PyTorch device string.
        input_size: ``WIDTHxHEIGHT`` model input specification. Defaults to
            ``1022x770``.
        resize_by_area: Preserve source aspect ratio at ``input_size`` area.
            Enabled by default.
        fp16: Use FP16 in attention-heavy model regions.
        fp32: Force full-precision inference; mutually exclusive with FP16.
        dump_pred: Save aligned prediction visualizations and point clouds.
        dump_gt: Save GT visualizations and point clouds.

    Returns:
        ``None``. Per-dataset and mean metrics are incrementally written as JSON.
    """
    import torch
    from tqdm import tqdm

    from pxdepth.config import load_config
    from pxdepth.evaluation import EvalDataLoaderPipeline, compute_metrics
    from pxdepth.inference import parse_size, predict_raw
    from pxdepth.model import PXDepth
    from pxdepth.utils.tools import key_average

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

    benchmarks = load_config(config_path, kind="eval")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Any] = {}
    for name, benchmark in tqdm(benchmarks.items(), desc="Benchmarks"):
        boundary = bool(benchmark.get("has_sharp_boundary", False))
        records = []
        with EvalDataLoaderPipeline(**_loader_config(benchmark, boundary)) as loader:
            for index in tqdm(range(len(loader)), desc=name, leave=False):
                sample = loader.get()
                sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
                pred = predict_raw(
                    model,
                    sample["image"],
                    input_size=size,
                    resize_by_area=resize_by_area,
                    use_fp16=fp16,
                    use_fp32=fp32,
                )
                metrics, misc = compute_metrics(pred, sample, vis=dump_pred or dump_gt, compute_boundary=boundary)
                metrics["inference_time"] = pred["inference_time"]
                records.append(metrics)
                if dump_pred or dump_gt:
                    dump_root = Path(str(output).removesuffix(".json") + "_dump")
                    sample_path = dump_root / name / sample["filename"].replace(".zip", "")
                    _dump(sample_path, sample["image"], pred, sample, misc, dump_pred, dump_gt)
                if index % 100 == 0 or index + 1 == len(loader):
                    output.write_text(json.dumps({**all_metrics, name: key_average(records)}, indent=2))
        all_metrics[name] = key_average(records)
    all_metrics["mean"] = key_average(list(all_metrics.values()))
    output.write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
