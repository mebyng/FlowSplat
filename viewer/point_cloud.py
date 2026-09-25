"""Conversion from RGB-D images to world-space colored point clouds."""

from __future__ import annotations

import torch


def depth_to_point_cloud(
    image: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project an RGB image and z-depth map through a camera-to-world pose.

    Args:
        image: RGB tensor shaped ``(3, H, W)``. Values are retained as colors.
        depth: Positive depth along the camera's forward (-Z) axis, shaped
            ``(H, W)`` or ``(1, H, W)``.
        intrinsics: 3x3 camera intrinsic matrix.
        camera_pose: 4x4 camera-to-world pose from ``render_scene.look_at``.
        min_depth: Depths at or below this value are discarded.
        max_depth: Optional upper depth threshold.

    Returns:
        ``(points, colors)`` with shapes ``(N, 3)`` and ``(N, 3)`` in world
        coordinates and RGB respectively. Pixel centers match ``compute_plucker``.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image shape (3, H, W), got {tuple(image.shape)}")
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"expected depth shape (H, W) or (1, H, W), got {tuple(depth.shape)}")
    if image.shape[1:] != depth.shape:
        raise ValueError("image and depth spatial dimensions must match")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"expected intrinsics shape (3, 3), got {tuple(intrinsics.shape)}")
    if camera_pose.shape != (4, 4):
        raise ValueError(f"expected camera_pose shape (4, 4), got {tuple(camera_pose.shape)}")

    device, dtype = depth.device, depth.dtype
    image = image.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    camera_pose = camera_pose.to(device=device, dtype=dtype)
    height, width = depth.shape
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (columns - cx) / fx * depth
    y = (cy - rows) / fy * depth
    # Pyrender uses an OpenGL camera: visible scene points have negative local Z.
    camera_points = torch.stack([x, y, -depth], dim=-1)

    valid = torch.isfinite(depth) & (depth > min_depth)
    if max_depth is not None:
        valid &= depth <= max_depth

    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    points = camera_points[valid] @ rotation.T + translation
    colors = image.permute(1, 2, 0)[valid]
    return points, colors
