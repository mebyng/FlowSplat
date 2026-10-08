"""Pyglet rendering backend, imported only when a viewer window is opened."""

from __future__ import annotations

import torch
from pyglet.image import ImageData
from pyglet import gl
from pyglet.graphics import Batch
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

from .splats import viewport_intrinsics


class GaussianRenderer:
    """Render a CUDA scene and cache its display texture until the camera changes."""

    def __init__(self, scene):
        self.scene = scene
        self._key = None
        self._texture = None

    @torch.inference_mode()
    def draw(self, window, pose):
        width, height = max(window.width, 1), max(window.height, 1)
        # Limit rasterization cost while preserving the viewport aspect ratio.
        scale = min(1.0, 512 / max(width, height))
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        key = (width, height, tuple(pose.flatten().tolist()))
        if key != self._key:
            rgb = self.scene.render(pose, viewport_intrinsics(rw, rh), height=rh, width=rw)["rgb"][0]
            pixels = (rgb.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().contiguous().numpy().tobytes()
            if self._texture is not None:
                self._texture.delete()
            self._texture = ImageData(rw, rh, "RGB", pixels, pitch=-rw * 3).get_texture()
            self._key = key
        window.clear()
        gl.glDisable(gl.GL_DEPTH_TEST)
        window.projection = Mat4.orthogonal_projection(0, width, 0, height, -1, 1)
        window.view = Mat4()
        self._texture.blit(0, 0, width=width, height=height)

    def close(self):
        if self._texture is not None:
            self._texture.delete()
            self._texture = None


class PointCloudRenderer:
    """Own GPU buffers and render a versioned point cloud into a window."""

    def __init__(self, point_size: float):
        # Use Pyglet's WindowBlock for core OpenGL projection and view matrices.
        vertex_source = """#version 150 core
            in vec3 position;
            in vec3 color;
            out vec3 point_color;
            uniform WindowBlock {
                mat4 projection;
                mat4 view;
            } window;
            void main() {
                gl_Position = window.projection * window.view * vec4(position, 1.0);
                gl_PointSize = POINT_SIZE;
                point_color = color;
            }
        """.replace("POINT_SIZE", f"{point_size:.1f}")
        fragment_source = """#version 150 core
            in vec3 point_color;
            out vec4 final_color;
            void main() {
                final_color = vec4(point_color, 1.0);
            }
        """
        self._program = ShaderProgram(
            Shader(vertex_source, "vertex"), Shader(fragment_source, "fragment")
        )
        self._batch = Batch()
        self._vertex_list = None
        self._uploaded_version = -1
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def _upload(self, points: torch.Tensor, colors: torch.Tensor, version: int) -> None:
        if self._vertex_list is not None:
            self._vertex_list.delete()
            self._vertex_list = None
        if len(points):
            self._vertex_list = self._program.vertex_list(
                len(points),
                gl.GL_POINTS,
                batch=self._batch,
                position=("f", points.flatten().tolist()),
                color=("f", colors.clamp(0, 1).flatten().tolist()),
            )
        self._uploaded_version = version

    def draw(
        self,
        window,
        pose: torch.Tensor,
        points: torch.Tensor,
        colors: torch.Tensor,
        version: int,
    ) -> None:
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        window.projection = Mat4.perspective_projection(
            window.width / max(window.height, 1), 0.01, 100.0, fov=45.0
        )
        window.view = Mat4.look_at(
            Vec3(*pose[:3, 3].tolist()), Vec3(0, 0, 0), Vec3(*pose[:3, 1].tolist())
        )
        if self._uploaded_version != version:
            self._upload(points, colors, version)
        if self._vertex_list is not None:
            self._batch.draw()
