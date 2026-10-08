"""Trainable fixed-size 3D Gaussian scene with differentiable gsplat rendering."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from gsplat import rasterization


class GaussianScene(nn.Module):
    """One scene, shared across all cameras; colors are direct RGB, without SH.

    Positions are initialized uniformly inside a sphere. Scale is stored in log
    space, color/opacity in logit space, and rotations as wxyz quaternions.
    Rendering requires gsplat and CUDA; constructing/loading the model does not.
    """

    def __init__(
        self,
        num_gaussians=512,
        radius=0.5,
        initial_scale=0.05,
        initial_opacity=0.1,
        background=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        if not isinstance(num_gaussians, int) or num_gaussians <= 0:
            raise ValueError("num_gaussians must be a positive integer")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite")
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be positive and finite")
        if not 0 < initial_opacity < 1:
            raise ValueError("initial_opacity must be in (0, 1)")

        background = torch.as_tensor(background, dtype=torch.float32)
        if background.shape != (3,) or not torch.isfinite(background).all():
            raise ValueError("background must contain three finite RGB values")
        if not ((background >= 0) & (background <= 1)).all():
            raise ValueError("background values must be in [0, 1]")

        directions = F.normalize(torch.randn(num_gaussians, 3), dim=-1)
        radii = radius * torch.rand(num_gaussians, 1).pow(1 / 3)
        self.means = nn.Parameter(directions * radii)
        self.log_scales = nn.Parameter(
            torch.full((num_gaussians, 3), math.log(initial_scale))
        )
        quaternions = torch.zeros(num_gaussians, 4)
        quaternions[:, 0] = 1
        self.quaternions = nn.Parameter(quaternions)
        self.color_logits = nn.Parameter(
            torch.logit(torch.rand(num_gaussians, 3).clamp(1e-4, 1 - 1e-4))
        )
        self.opacity_logits = nn.Parameter(
            torch.full(
                (num_gaussians,), math.log(initial_opacity / (1 - initial_opacity))
            )
        )
        self.register_buffer("background", background.clone())

    @property
    def scales(self):
        return self.log_scales.exp()

    @property
    def rotations(self):
        return F.normalize(self.quaternions, dim=-1)

    @property
    def colors(self):
        return self.color_logits.sigmoid()

    @property
    def opacities(self):
        return self.opacity_logits.sigmoid()

    def device(self):
        return self.means.device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load weights into a scene constructed with the same Gaussian count."""
        checkpoint = torch.load(path, map_location=self.device(), weights_only=True)
        self.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    def _prepare_cameras(self, camera_pose, intrinsics):
        """Convert OpenGL camera-to-world poses to OpenCV world-to-camera matrices."""
        camera_pose = camera_pose.to(self.device())
        intrinsics = intrinsics.to(self.device())
        if camera_pose.ndim == 2:
            camera_pose = camera_pose.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if camera_pose.ndim != 3 or camera_pose.shape[1:] != (4, 4):
            raise ValueError("camera_pose must have shape (4, 4) or (B, 4, 4)")
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3) or (B, 3, 3)")
        if camera_pose.shape[0] == 0 or intrinsics.shape[0] not in (
            1,
            camera_pose.shape[0],
        ):
            raise ValueError("intrinsics must be shared or match the camera batch")
        intrinsics = intrinsics.expand(camera_pose.shape[0], -1, -1).contiguous()
        conversion = torch.diag(self.means.new_tensor([1, -1, -1, 1]))
        viewmats = conversion @ torch.linalg.inv(camera_pose)
        return viewmats.contiguous(), intrinsics

    def render(
        self,
        camera_pose,
        intrinsics,
        height=128,
        width=128,
        near_plane=0.01,
        far_plane=100.0,
    ):
        """Render RGB, expected forward z-depth, and accumulated opacity.

        Inputs follow dataset conventions (OpenGL camera-to-world, pixel-space K).
        Results always retain a batch dimension: RGB (B,3,H,W), depth/alpha
        (B,1,H,W). Empty pixels have zero depth. Gradients reach scene parameters.
        """
        viewmats, intrinsics = self._prepare_cameras(camera_pose, intrinsics)

        rendered, alpha, _ = rasterization(
            means=self.means,
            quats=self.rotations,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+ED",
            packed=True,
        )
        # gsplat 1.5.3 validates packed backgrounds without the camera axis.
        # Composite RGB here to support batched cameras and leave depth alone.
        rgb = rendered[..., :3] + (1 - alpha) * self.background
        depth = torch.where(alpha > 0, rendered[..., 3:4], 0)
        return {
            "rgb": rgb.permute(0, 3, 1, 2).contiguous(),
            "depth": depth.permute(0, 3, 1, 2).contiguous(),
            "alpha": alpha.permute(0, 3, 1, 2).contiguous(),
        }

    def forward(self, camera_pose, intrinsics, **kwargs):
        return self.render(camera_pose, intrinsics, **kwargs)
