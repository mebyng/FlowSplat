"""Gaussian checkpoint loading and viewport camera intrinsics."""

import math
from pathlib import Path

import torch

from models import GaussianScene
from utils.checkpoints import resolve_checkpoint_path


def load_splats(run_name, logdir="runs", device="auto"):
    device = torch.device("cuda" if device == "auto" else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Gaussian viewing requires CUDA; use --device cuda")
    path = resolve_checkpoint_path(str(Path(logdir) / "gaussian"), run_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    scene = GaussianScene(num_gaussians=state["means"].shape[0])
    scene.load_state_dict(state)
    scene.background.fill_(1.0)
    return scene.to(device).eval().requires_grad_(False), path


def viewport_intrinsics(width, height):
    """Match the point renderer's 45-degree vertical field of view."""
    focal = height / (2 * math.tan(math.radians(45) / 2))
    return torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=torch.float32)
