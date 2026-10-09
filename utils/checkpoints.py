"""Checkpoint selection shared by training and the viewer."""

from pathlib import Path
import json
import torch
from models import AttentionAutoEncoder, FlowModel


def resolve_checkpoint_path(logdir: str, run_name: str | None):
    if not run_name:
        return None

    candidate = Path(run_name)
    if candidate.is_file():
        return str(candidate)

    if candidate.is_absolute():
        run_dir = candidate
    else:
        run_dir = Path(logdir) / run_name

    final_checkpoint = run_dir / "final" / "model_checkpoint.pth"
    if final_checkpoint.exists():
        return str(final_checkpoint)

    epoch_dirs = sorted(
        [
            path
            for path in run_dir.iterdir()
            if path.is_dir() and path.name.startswith("epoch_")
        ],
        key=lambda path: int(path.name.split("_")[-1]),
        reverse=True,
    )

    for epoch_dir in epoch_dirs:
        checkpoint_path = epoch_dir / "model_checkpoint.pth"
        if checkpoint_path.exists():
            return str(checkpoint_path)

    root_checkpoint = run_dir / "model_checkpoint.pth"
    if root_checkpoint.exists():
        return str(root_checkpoint)

    raise FileNotFoundError(
        f"Could not find a checkpoint for run '{run_name}' under {run_dir}"
    )


def load_encoding_model(run_name, logdir="runs", device="cuda"):
    import torch

    from models import SimpleAutoEncoder

    checkpoint_path = resolve_checkpoint_path(str(Path(logdir) / "encode"), run_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    autoencoder = SimpleAutoEncoder(
        latent_dim=int(state_dict["latent_dim"].item()),
        use_depth=bool(state_dict["use_depth"].item()),
    )
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(device).eval().requires_grad_(False)
    return autoencoder, checkpoint_path


def load_flow_teacher(run_name, logdir="runs", device="cuda"):
    """Load a flow teacher"""

    path = resolve_checkpoint_path(str(Path(logdir) / "generate"), run_name)
    checkpoint_path = Path(path)
    run = checkpoint_path.parent
    if run.name == "final" or run.name.startswith("epoch_"):
        run = run.parent
    metadata = json.loads((run / "metadata.json").read_text())
    use_depth = metadata.get("use_depth", False)
    if metadata.get("mode") != "generate" or metadata.get("use_encoding"):
        raise ValueError(
            "Gaussian supervision requires a raw RGB or RGB-D generate run"
        )
    if metadata.get("model_name") != "AttentionAutoEncoder":
        raise ValueError("Unsupported teacher architecture")
    channels = 4 if use_depth else 3
    network = AttentionAutoEncoder(mode="flow", channels=channels, emb_channels=128)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return FlowModel(network).to(device).eval().requires_grad_(False), path
