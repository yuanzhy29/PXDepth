"""Hugging Face Gradio demo for PXDepth.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Import `spaces` before torch.  On ZeroGPU this enables Hugging Face's CUDA
# emulation during module initialization, so models can be placed on CUDA once
# at startup and automatically use the real allocated GPU inside @spaces.GPU.
try:
    import spaces

    _GPU_DECORATOR = spaces.GPU(duration=120)
except ImportError:
    spaces = None
    _GPU_DECORATOR = lambda fn: fn

import gradio as gr
import numpy as np
import torch
import utils3d

from pxdepth.inference import resize_image, resize_map
from pxdepth.model import PXDepth
from pxdepth.utils.ply import write_point_cloud_ply
from pxdepth.utils.vis import colorize_depth


PXDEPTH_REPO = "yuanzhy29/PXDepth"
MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"
DEFAULT_CHECKPOINT = Path("checkpoints/pxdepth/model.pt")
DEFAULT_REFERENCE_CHECKPOINT = Path("checkpoints/moge-2-vitl-normal/model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed inference reference area. The source aspect ratio is preserved and the
# resulting H/W are rounded to multiples of the PXDepth encoder patch size.
TARGET_SIZE = (1022, 770)

CSS = """
#pxdepth-demo {
    max-width: 1280px;
    margin: 0 auto;
}
#img-display-input {
    max-height: 72vh;
}
#img-display-output {
    max-height: 72vh;
}
#img-display-output img {
    object-fit: contain !important;
}
#model-3d {
    min-height: 60vh;
}
"""


def _local_checkpoint(path: str | Path) -> Optional[Path]:
    """Resolve a checkpoint path relative to either the shell or this repository."""
    path = Path(path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / path)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _load_models(
    checkpoint: str | Path,
    reference_checkpoint: str | Path,
) -> tuple[PXDepth, torch.nn.Module]:
    """Load local checkpoints when available, otherwise use Hugging Face."""
    local_checkpoint = _local_checkpoint(checkpoint)
    model_source: str | Path = local_checkpoint or PXDEPTH_REPO
    if local_checkpoint is None:
        print(f"PXDepth checkpoint not found at {checkpoint}; downloading {PXDEPTH_REPO}.")
    else:
        print(f"Loading PXDepth from {local_checkpoint}.")
    model = PXDepth.from_pretrained(model_source, strict=True).eval()

    print("Loading MoGe-2...")
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError(
            "MoGe-2 is required for the metric-scale demo. "
            "Install PXDepth with the `demo` (or `reference`) extra."
        ) from exc

    local_reference = _local_checkpoint(reference_checkpoint)
    reference_source: str | Path = local_reference or MOGE2_REPO
    if local_reference is None:
        print(
            f"MoGe-2 checkpoint not found at {reference_checkpoint}; "
            f"downloading {MOGE2_REPO}."
        )
    else:
        print(f"Loading MoGe-2 from {local_reference}.")
    reference = MoGeModel.from_pretrained(reference_source).eval()

    # PXDepth.infer() checks this cache before lazily loading MoGe-2 itself.
    # Assigning the reference model here also registers it as a child module,
    # so MODEL.to(DEVICE) moves both networks together.
    model._reference_model = reference
    model = model.to(DEVICE).eval()
    reference = model._reference_model

    print(f"Models loaded on {DEVICE}.")
    return model, reference


MODEL: Optional[PXDepth] = None
REFERENCE_MODEL: Optional[torch.nn.Module] = None


def _session_dir(request: Optional[gr.Request]) -> Path:
    """Create one output directory per browser session."""
    session_hash = getattr(request, "session_hash", None) or "local"
    root = Path(tempfile.gettempdir()) / "pxdepth-demo"
    output = root / session_hash
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _sample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample a dense point cloud for browser rendering."""
    if points.shape[0] <= max_points:
        return points, colors

    indices = np.linspace(
        0,
        points.shape[0] - 1,
        num=max_points,
        dtype=np.int64,
    )
    return points[indices], colors[indices]


@_GPU_DECORATOR
@torch.inference_mode()
def on_submit(
    image: Optional[np.ndarray],
    max_points: int,
    apply_mask: bool,
    request: gr.Request,
):
    """Run PXDepth, recover metric scale with MoGe-2, and build demo outputs."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    if MODEL is None:
        raise gr.Error("The model has not been loaded.")

    if image.ndim != 3 or image.shape[-1] < 3:
        raise gr.Error("The input must be an RGB image.")

    image = np.ascontiguousarray(image[..., :3].astype(np.uint8))

    use_fp16 = DEVICE.type == "cuda"
    use_fp32 = DEVICE.type != "cuda"

    image_tensor = (
        torch.from_numpy(image.copy())
        .to(device=DEVICE, dtype=torch.float32)
        .permute(2, 0, 1)
        / 255.0
    )

    model_image, original_size = resize_image(
        image_tensor,
        TARGET_SIZE,
        True,
        MODEL.patch_size,
    )

    # This is the same public metric-scale path used by scripts/infer.py.
    result = MODEL.infer(
        model_image,
        ref_image=image_tensor,
        apply_mask=False,
        use_fp16=use_fp16,
        use_fp32=use_fp32,
    )

    depth = resize_map(result["depth"], original_size).float()
    mask = resize_map(result["mask"], original_size, is_mask=True)
    intrinsics = result["intrinsics"].detach().float()

    finite_depth = torch.isfinite(depth) & (depth > 0)
    point_valid = finite_depth & mask if apply_mask else finite_depth

    # Reconstruct at the original image resolution using normalized camera
    # intrinsics returned by PXDepth/MoGe-2.
    point_map = utils3d.pt.depth_map_to_point_map(
        torch.where(finite_depth, depth, torch.zeros_like(depth)),
        intrinsics=intrinsics,
    )

    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    point_valid_np = point_valid.detach().cpu().numpy().astype(bool)

    # Match the repository's existing depth visualization.
    depth_vis = colorize_depth(
        np.where(mask_np, depth_np, np.inf),
        mask=None,
    )

    output_dir = _session_dir(request)

    # Save raw outputs for download.
    raw_depth_path = output_dir / "metric_depth.npy"
    np.save(raw_depth_path, depth_np)

    depth_vis_path = output_dir / "depth.png"
    from PIL import Image

    Image.fromarray(depth_vis).save(depth_vis_path)

    mask_path = output_dir / "mask.png"
    Image.fromarray((mask_np.astype(np.uint8) * 255), mode="L").save(mask_path)

    # Save a full valid PLY for download.
    points_np = point_map.detach().cpu().numpy().reshape(-1, 3)
    colors_np = image.reshape(-1, 3).astype(np.float32) / 255.0
    keep = point_valid_np.reshape(-1) & np.isfinite(points_np).all(axis=1)

    points_full = points_np[keep]
    colors_full = colors_np[keep]
    if points_full.shape[0] == 0:
        raise gr.Error("No valid 3D points were produced for this image.")

    ply_path = output_dir / "pointcloud.ply"
    write_point_cloud_ply(ply_path, points_full, colors_full)

    # A smaller point cloud keeps the interactive browser viewer responsive.
    viewer_points, viewer_colors = _sample_point_cloud(
        points_full,
        colors_full,
        int(max_points),
    )
    # Gradio's 3D viewer is most reliable with GLB output.  Keep the
    # downloaded PLY in PXDepth's native camera convention, while flipping
    # the display-only cloud to the conventional web-viewer orientation.
    import trimesh

    viewer_vertices = viewer_points * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    viewer_rgb = np.clip(viewer_colors * 255.0, 0, 255).astype(np.uint8)
    viewer_glb_path = output_dir / "pointcloud_viewer.glb"
    trimesh.PointCloud(
        vertices=viewer_vertices,
        colors=viewer_rgb,
    ).export(viewer_glb_path)

    download_files = [
        str(depth_vis_path),
        str(raw_depth_path),
        str(mask_path),
        str(ply_path),
    ]

    return (
        (image, depth_vis),
        str(viewer_glb_path),
        download_files,
    )


def build_demo() -> gr.Blocks:
    title = "# PXDepth"
    description = """
Official demo for **PXDepth: Pixel-Space Modeling for Structure Preserving Monocular Depth Estimation**.  
Please refer to our [paper](https://arxiv.org/abs/2608.16984),
[project page](https://yuanzhy29.github.io/PXDepth-Page/), and
[GitHub](https://github.com/yuanzhy29/PXDepth) for more details.
"""

    with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
        with gr.Column(elem_id="pxdepth-demo"):
            gr.Markdown(title)
            gr.Markdown(description)
            gr.Markdown("### Point Cloud & Depth Prediction demo")

            with gr.Row():
                # Left: image, settings, predict button.
                with gr.Column():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGB",
                        type="numpy",
                        elem_id="img-display-input",
                    )

                    with gr.Accordion(label="Settings", open=False):
                        max_points = gr.Slider(
                            minimum=50_000,
                            maximum=500_000,
                            value=200_000,
                            step=50_000,
                            label="3D Viewer Max Points",
                            info="Only affects the interactive viewer; the downloaded PLY keeps all valid points.",
                        )
                        apply_mask = gr.Checkbox(
                            label="Apply valid-depth mask to point cloud",
                            value=True,
                        )

                    submit_btn = gr.Button(
                        value="Predict",
                        variant="primary",
                    )

                # Right: same three-tab organization as Pixel-Perfect-Depth.
                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab("3D View"):
                            model_3d = gr.Model3D(
                                label="3D Point Map",
                                clear_color=(1.0, 1.0, 1.0, 1.0),
                                height="60vh",
                                elem_id="model-3d",
                            )

                        with gr.Tab("Depth"):
                            depth_map = gr.ImageSlider(
                                label="RGB / Metric Depth",
                                image_mode="RGB",
                                type="numpy",
                                slider_position=50,
                                elem_id="img-display-output",
                            )

                        with gr.Tab("Download"):
                            download_files = gr.File(
                                label="Download Files",
                                file_count="multiple",
                                type="filepath",
                            )

            example_dir = Path("example_images")
            example_files = []
            if example_dir.exists():
                example_files = sorted(
                    str(path)
                    for path in example_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                )

            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=input_image,
                    label="Examples",
                    cache_examples=False,
                )

            submit_btn.click(
                fn=on_submit,
                inputs=[
                    input_image,
                    max_points,
                    apply_mask,
                ],
                outputs=[
                    depth_map,
                    model_3d,
                    download_files,
                ],
                show_progress="full",
            )

    return demo


demo = build_demo()


def main() -> None:
    """Load checkpoints and launch the local or Hugging Face Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the PXDepth Gradio demo.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=(
            "PXDepth model.pt path. If the file does not exist, the released "
            "checkpoint is downloaded from Hugging Face."
        ),
    )
    parser.add_argument(
        "--reference-checkpoint",
        default=str(DEFAULT_REFERENCE_CHECKPOINT),
        help=(
            "MoGe-2 model.pt path. If the file does not exist, the released "
            "checkpoint is downloaded from Hugging Face."
        ),
    )
    args = parser.parse_args()

    global MODEL, REFERENCE_MODEL
    MODEL, REFERENCE_MODEL = _load_models(
        args.checkpoint,
        args.reference_checkpoint,
    )
    demo.queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
