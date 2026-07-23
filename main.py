import argparse
import os
from datetime import datetime
from pathlib import Path

import torch

from data import ViewDataset
from models import (
    SimpleAutoEncoder,
    RegressionWrapper,
    RotationConditionedUNetRes,
    FlowWrapper,
)
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Diffusion Model to generate rotated scenes."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--scheduler-step", type=int, default=100, help="StepLR step size"
    )
    parser.add_argument(
        "--scheduler-gamma", type=float, default=0.5, help="StepLR gamma"
    )
    parser.add_argument(
        "--eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler"
    )
    parser.add_argument("--dataset-path", type=str, default="datasets/small_512/")
    parser.add_argument(
        "--mode",
        type=str,
        default="rotate",
        choices=["rotate", "generate", "encode"],
        help="Training mode: 'rotate' for learning rotations, 'generate' for learning to generate views directly (default: 'generate')",
    )
    parser.add_argument(
        "--savepoint",
        type=int,
        default=50,
        help="Save model and plots every N epochs (default: 10)",
    )
    parser.add_argument(
        "--logdir", type=str, default="runs", help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this training run (appended to logdir)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional name of a previous run whose checkpoint should be loaded before training",
    )
    return parser.parse_args()


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
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("epoch_")],
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


def main():
    args = parse_args()

    use_encoding = args.mode != "encode"
    dataset = ViewDataset(
        args.dataset_path, split="training", mode=args.mode, use_encoding=use_encoding
    )
    val_dataset = ViewDataset(
        args.dataset_path,
        split="validation",
        mode=args.mode,
        random_matching=False,
        use_encoding=use_encoding,
    )  # Use deterministic matching for validation
    if args.mode == "generate":
        out_channels = 64 if use_encoding else 3
        in_channels = out_channels + 72  # 72 for camera encoding
        resolution = 32 if use_encoding else 128
        model = FlowWrapper(
            RotationConditionedUNetRes(
                in_channels=in_channels, out_channels=out_channels
            )
        ).cuda()

        autoencoder = SimpleAutoEncoder().cuda().eval()
        autoencoder.load("weight_checkpoints/SimpleAutoEncoder_medium.pth")
    elif args.mode == "rotate":
        out_channels = 64 if use_encoding else 3
        in_channels = out_channels + 72  # 72 for camera encoding
        resolution = 32 if use_encoding else 128
        model = RegressionWrapper(
            RotationConditionedUNetRes(
                in_channels=in_channels, out_channels=out_channels
            ),
            mode=args.mode,
        ).cuda()

        autoencoder = SimpleAutoEncoder().cuda().eval()
        autoencoder.load("weight_checkpoints/SimpleAutoEncoder_medium.pth")
    elif args.mode == "encode":
        model = RegressionWrapper(SimpleAutoEncoder(), mode=args.mode).cuda()
        autoencoder = None
        resolution = 32
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Training configuration:")
    print(f"  mode: {args.mode}")
    print(f"  dataset: {args.dataset_path}")
    print(f"  training samples: {len(dataset)}")
    print(f"  batch size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning rate: {args.lr}")
    print(f"  scheduler: {args.scheduler}")
    print(f"  model parameters: {num_params:,} total, {num_trainable:,} trainable")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    if args.run_name:
        log_dir = os.path.join(args.logdir, args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.logdir, f"run_{timestamp}")

    print(f"TensorBoard logs will be written to: {log_dir}")

    if args.resume_from:
        checkpoint_path = resolve_checkpoint_path(args.logdir, args.resume_from)
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load(checkpoint_path)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        validation_dataset=val_dataset,
        autoencoder=autoencoder,
        mode=args.mode,
        batch_size=args.batch_size,
        scheduler=scheduler,
        savepoint=args.savepoint,
        log_dir=log_dir,
        resolution=resolution,
    )

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
