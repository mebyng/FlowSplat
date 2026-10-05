"""A minimal interactive, rotation-only colored point-cloud viewer."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .camera_orbit import OrbitCamera


class OrbitPointCloudViewer:
    """Display colored points from a fixed-distance orbit camera.

    Drag the left mouse button to rotate. Pan and zoom are deliberately absent so
    every selected view remains on the same radius-2 camera sphere used by dataset
    generation. Pressing Space invokes ``on_space`` with the current C2W pose.
    """

    def __init__(
        self,
        radius: float = 2.0,
        width: int = 960,
        height: int = 720,
        point_size: float = 2.0,
        on_space: Callable[[torch.Tensor], None] | None = None,
    ):
        self.camera = OrbitCamera(radius=radius)
        self.width = width
        self.height = height
        self.point_size = point_size
        self.on_space = on_space
        self._points = torch.empty((0, 3), dtype=torch.float32)
        self._colors = torch.empty((0, 3), dtype=torch.float32)
        self._window = None
        self._cloud_version = 0

    @property
    def camera_pose(self) -> torch.Tensor:
        """Current camera-to-world pose for use as a flow-model target camera."""
        return self.camera.pose()

    def set_point_cloud(self, points: torch.Tensor, colors: torch.Tensor) -> None:
        self._points, self._colors = self._validate_cloud(points, colors)
        self._cloud_version += 1

    def add_point_cloud(self, points: torch.Tensor, colors: torch.Tensor) -> None:
        points, colors = self._validate_cloud(points, colors)
        self._points = torch.cat([self._points, points], dim=0)
        self._colors = torch.cat([self._colors, colors], dim=0)
        self._cloud_version += 1

    @staticmethod
    def _validate_cloud(points: torch.Tensor, colors: torch.Tensor):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected points shape (N, 3), got {tuple(points.shape)}")
        if colors.shape != points.shape:
            raise ValueError("colors must have the same (N, 3) shape as points")
        return (
            points.detach().cpu().float().contiguous(),
            colors.detach().cpu().float().contiguous(),
        )

    def run(self) -> None:
        """Open the viewer. This blocks until the viewer window is closed."""
        try:
            import pyglet
            from .renderer import PointCloudRenderer
        except ImportError as exc:
            raise RuntimeError(
                "OrbitPointCloudViewer requires the optional pyglet dependency"
            ) from exc

        window = pyglet.window.Window(
            self.width, self.height, caption="FlowSplat point cloud", resizable=True
        )
        self._window = window

        renderer = PointCloudRenderer(self.point_size)

        @window.event
        def on_draw():
            renderer.draw(
                window,
                self.camera_pose,
                self._points,
                self._colors,
                self._cloud_version,
            )

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self.camera.rotate(-dx * 0.01, -dy * 0.01)

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.SPACE and self.on_space is not None:
                self.on_space(self.camera_pose)

        pyglet.app.run()
