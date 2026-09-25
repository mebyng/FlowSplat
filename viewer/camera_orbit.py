"""Orbit cameras using the same camera-to-world convention as dataset rendering."""

from __future__ import annotations

import math

import torch


class OrbitCamera:
    """A camera constrained to a sphere around the world origin.

    The generated pose has the same convention as ``data.render_scene.look_at``:
    it is a camera-to-world matrix whose translation is the camera position and
    whose local ``-Z`` axis points toward the origin, as required by Pyrender's
    OpenGL camera convention.
    """

    def __init__(
        self,
        radius: float = 2.0,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        elevation_limit: float = math.pi / 2 - 1e-3,
    ):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.azimuth = float(azimuth)
        self.elevation_limit = float(elevation_limit)
        self.elevation = max(-self.elevation_limit, min(float(elevation), self.elevation_limit))

    @property
    def position(self) -> torch.Tensor:
        cos_elevation = math.cos(self.elevation)
        return torch.tensor(
            [
                self.radius * cos_elevation * math.cos(self.azimuth),
                self.radius * cos_elevation * math.sin(self.azimuth),
                self.radius * math.sin(self.elevation),
            ],
            dtype=torch.float32,
        )

    def rotate(self, delta_azimuth: float, delta_elevation: float) -> None:
        """Change orbit angles while keeping the camera distance fixed."""
        self.azimuth = (self.azimuth + float(delta_azimuth)) % (2 * math.pi)
        self.elevation = max(
            -self.elevation_limit,
            min(self.elevation + float(delta_elevation), self.elevation_limit),
        )

    def pose(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """Return a 4x4 camera-to-world pose matching ``render_scene.look_at``."""
        eye = self.position.to(device=device, dtype=dtype)
        target = torch.zeros(3, device=device, dtype=dtype)
        forward = target - eye
        forward = forward / torch.linalg.vector_norm(forward)

        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        if torch.abs(torch.dot(forward, up)) > 0.9999:
            up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

        right = torch.linalg.cross(forward, up)
        right = right / torch.linalg.vector_norm(right)
        true_up = torch.linalg.cross(right, forward)

        pose = torch.eye(4, device=device, dtype=dtype)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return pose

    @classmethod
    def from_pose(cls, pose: torch.Tensor) -> "OrbitCamera":
        """Create a fixed-radius orbit camera from a camera-to-world pose."""
        if pose.shape != (4, 4):
            raise ValueError(f"expected a (4, 4) pose, got {tuple(pose.shape)}")
        position = pose[:3, 3].detach().cpu().float()
        radius = torch.linalg.vector_norm(position).item()
        if radius <= 0:
            raise ValueError("the pose translation must not be at the origin")
        azimuth = math.atan2(position[1].item(), position[0].item())
        elevation = math.asin(max(-1.0, min(1.0, position[2].item() / radius)))
        return cls(radius=radius, azimuth=azimuth, elevation=elevation)
