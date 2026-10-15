"""Load a named RGB-D run and predict views from a fixed source image."""

import json
from pathlib import Path

import torch

from models import AttentionAutoEncoder, FlowModel, RegressionModel
from utils.checkpoints import load_encoding_model, resolve_checkpoint_path


def resolve_run(run_name: str, logdir: str = "runs") -> tuple[Path, Path]:
    root = Path(logdir)
    candidates = [root / mode / run_name for mode in ("rotate", "generate")]
    candidates.append(root / run_name)
    runs = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
    if not runs:
        raise FileNotFoundError(f"Run {run_name!r} not found under {root}")
    if len(runs) != 1:
        raise ValueError(f"Ambiguous run name {run_name!r}: {runs}. Use --logdir to select its parent.")
    run = runs[0]
    return run, Path(resolve_checkpoint_path(str(run.parent), run.name))


class ViewPredictor:
    def __init__(self, run_name: str, logdir: str = "runs", device: str = "auto", steps: int = 10):
        if steps <= 0:
            raise ValueError("--steps must be positive")
        run, self.checkpoint = resolve_run(run_name, logdir)
        metadata = json.loads((run / "metadata.json").read_text())
        self.mode = metadata.get("mode")
        if self.mode not in {"rotate", "generate"}:
            raise ValueError("Viewer requires a rotate or generate run")
        use_encoding = metadata.get("use_encoding", False)
        if not metadata.get("use_depth") and not use_encoding:
            raise ValueError("Viewer requires a run trained with --use-depth or --use-encoding")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.steps = steps
        self.autoencoder = None
        self.encoding_checkpoint = None
        self.channels = 4
        if use_encoding:
            encoding_checkpoint = metadata.get("encoding_checkpoint")
            if not encoding_checkpoint:
                raise ValueError("Latent run metadata must record its encoding_checkpoint")
            self.autoencoder, self.encoding_checkpoint = load_encoding_model(
                encoding_checkpoint, logdir, self.device
            )
            self.channels = self.autoencoder.channels
        network = AttentionAutoEncoder(
            mode="flow" if self.mode == "generate" else "deterministic",
            channels=int(self.autoencoder.latent_dim.item()) if use_encoding else 4,
            emb_channels=128,
        )
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model = (FlowModel(network) if self.mode == "generate" else RegressionModel(network)).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, source, source_pose, target_pose, intrinsics):
        source = source.detach().clone().unsqueeze(0).to(self.device)
        source = source[:, :self.channels]
        image_shape = source.shape[1:]
        if self.autoencoder is not None:
            source = self.autoencoder.encode(source)
        cameras = {
            "source": source_pose.unsqueeze(0).to(self.device),
            "target": target_pose.unsqueeze(0).to(self.device),
            "intrinsics": intrinsics.unsqueeze(0).to(self.device),
        }
        noise = torch.randn_like(source) if self.mode == "generate" else None
        result = self.model.generate(source, noise, cameras, num_steps=self.steps)
        if self.autoencoder is not None:
            result = self.autoencoder.decode(result)
        result = result[0]
        if result.shape != image_shape:
            raise ValueError(f"Expected prediction of shape {image_shape}, got {result.shape}")
        return result.cpu()
