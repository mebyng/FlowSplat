import argparse
import os
from datetime import datetime

import torch

from data import ViewDataset
from models import SimpleAutoEncoder, RegressionWrapper
from models.UNetDummy import RotationConditionedUNetRes
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model to generate rotated scenes.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="Learning rate scheduler type")
    parser.add_argument("--scheduler-step", type=int, default=100, help="StepLR step size")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    parser.add_argument("--dataset-path", type=str, default="datasets/small_512/")
    parser.add_argument("--mode", type=str, default="rotate", choices=["rotate", "generate", "encode"], help="Training mode: 'rotate' for learning rotations, 'generate' for learning to generate views directly (default: 'generate')")
    parser.add_argument("--savepoint", type=int, default=100, help="Save model and plots every N epochs (default: 10)")
    parser.add_argument("--logdir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for this training run (appended to logdir)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ViewDataset(args.dataset_path, split="training", mode=args.mode, use_encoding=True)
    val_dataset = ViewDataset(args.dataset_path, split="training", mode=args.mode, random_matching=False, use_encoding=True)  # Use deterministic matching for validation
    if args.mode in ["rotate", "generate"]:
        model = RegressionWrapper(RotationConditionedUNetRes(in_channels=136, out_channels=64), mode=args.mode).cuda()
    elif args.mode == "encode":
        model = RegressionWrapper(SimpleAutoEncoder(), mode=args.mode).cuda()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    autoencoder = SimpleAutoEncoder().cuda()
    autoencoder.load("weight_checkpoints/SimpleAutoEncoder.pth")

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    if args.run_name:
        log_dir = os.path.join(args.logdir, args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.logdir, f"run_{timestamp}")
    
    print(f"TensorBoard logs will be written to: {log_dir}")

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
        results_dir="results",
        log_dir=log_dir,
        resolution=32,
    )

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
