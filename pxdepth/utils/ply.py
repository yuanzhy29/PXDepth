"""Minimal binary little-endian PLY export for dense colored point clouds.

Finite XYZ samples and their corresponding RGB values are flattened, filtered,
and serialized with a standards-compliant vertex header. The implementation is
dependency-light and is used by inference and evaluation dumps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write XYZ points and RGB colors as a binary PLY vertex list.

    Args:
        path: Destination file. Parent directories are created automatically.
        points: Floating point positions ``[N,3]``.
        colors: RGB values ``[N,3]``. Floating arrays are interpreted in
            ``[0,1]``; integer arrays are interpreted in ``[0,255]``.

    Returns:
        ``None``. A binary little-endian PLY file is written.
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must have shape [N, 3], got {colors.shape}")
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    if colors.dtype.kind == "f":
        colors = np.clip(colors, 0.0, 1.0) * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)
