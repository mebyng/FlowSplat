import argparse
import os
from datetime import datetime

import torch

from data import ViewDataset
from logger import SampleLogger
from models import (
    SimpleAutoEncoder,
    AutoEncoder,
    FlowModel,
    RegressionModel,
    AttentionAutoEncoder,
)
from trainer import Trainer
from utils.checkpoints import load_encoding_model, resolve_checkpoint_path
from utils.randomness import seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a Diffusion Model to generate rotated scenes."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    parser.add_argument("--seed", type=int, default=42, help="Training RNG seed.")
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=0,
        help="Independent seed for evaluation noise.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm; 0 disables clipping.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Linear LR warmup from 10%% to 100%%; 0 disables warmup.",
    )
    parser.add_argument("--dataset-path", type=str, default="datasets/small_512/")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["rotate", "generate", "encode"],
        help="Training mode: 'rotate' for learning rotations, 'generate' for learning to generate views directly (default: 'generate')",
    )
    parser.add_argument(
        "--rotation-encoding",
        type=str,
        default="6D",
        choices=["plucker", "matrix", "6D"],
        help="Type of rotation encoding to use (default: 'matrix')",
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
    parser.add_argument(
        "--use-encoding",
        type=str,
        metavar="RUN_NAME",
        default=None,
        help="Use precomputed latents and load the AE run from <logdir>/encode/RUN_NAME (final checkpoint preferred).",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Append ground-truth depth as a fourth RGB-D channel for rotation/flow training.",
    )
    args = parser.parse_args(argv)
    if args.use_encoding and args.mode == "encode":
        parser.error("--use-encoding is only supported for generate and rotate modes")
    if args.epochs <= 0 or not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("require epochs > 0 and 0 <= warmup-epochs <= epochs")
    if args.grad_clip < 0:
        parser.error("grad-clip must be nonnegative")
    if not 0 <= args.seed < 2**32 or not 0 <= args.validation_seed < 2**32:
        parser.error("seeds must be in [0, 2**32)")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    autoencoder = None
    encoding_checkpoint = None
    if args.use_encoding:
        autoencoder, encoding_checkpoint = load_encoding_model(
            args.use_encoding, args.logdir
        )
        print(f"Loading autoencoder checkpoint from: {encoding_checkpoint}")

    dataset = ViewDataset(
        args.dataset_path,
        split="training",
        mode=args.mode,
        use_encoding=bool(args.use_encoding),
        use_depth=args.use_depth,
    )
    val_dataset = ViewDataset(
        args.dataset_path,
        split="validation",
        mode=args.mode,
        random_matching=False,
        use_encoding=bool(args.use_encoding),
        use_depth=args.use_depth,
    )  # Use deterministic matching for validation
    if args.mode == "generate":
        channels = 4 if (args.use_encoding or args.use_depth) else 3
        resolution = 32 if args.use_encoding else 128
        model = FlowModel(
            AttentionAutoEncoder(
                mode="flow",
                in_channels=channels,
                out_channels=channels,
                emb_channels=128,
            ),
        ).cuda()

    elif args.mode == "rotate":
        channels = 4 if (args.use_encoding or args.use_depth) else 3
        resolution = 32 if args.use_encoding else 128
        model = RegressionModel(
            AttentionAutoEncoder(
                mode="deterministic",
                in_channels=channels,
                out_channels=channels,
                emb_channels=128,
            ),
        ).cuda()

    elif args.mode == "encode":
        model = AutoEncoder(
            SimpleAutoEncoder(in_channels=4 if args.use_depth else 3, latent_dim=4)
        ).cuda()
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
    print(f"  warmup epochs: {args.warmup_epochs}; gradient clip: {args.grad_clip}")
    print(f"  training seed: {args.seed}; validation seed: {args.validation_seed}")
    print(f"  model parameters: {num_params:,} total, {num_trainable:,} trainable")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.eta_min,
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    if args.run_name:
        log_dir = os.path.join(args.logdir, args.mode, args.run_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.logdir, args.mode, f"run_{timestamp}")

    print(f"TensorBoard logs will be written to: {log_dir}")

    metadata = vars(args).copy()
    metadata.update(
        {
            "model_name": type(model.model).__name__,
            "num_params": num_params,
            "num_trainable": num_trainable,
            "training_samples": len(dataset),
            "validation_samples": len(val_dataset),
            "log_dir": log_dir,
            "encoding_checkpoint": encoding_checkpoint,
        }
    )

    sample_logger = SampleLogger(
        train_dataset=dataset,
        validation_dataset=val_dataset,
        n_images=5,
        n_samples=2,
        log_path=log_dir,
        mode=args.mode,
        resolution=resolution,
        autoencoder=autoencoder,
        metadata=metadata,
        validation_seed=args.validation_seed,
    )

    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        logger=sample_logger,
        validation_dataset=val_dataset,
        autoencoder=autoencoder,
        mode=args.mode,
        rotation_encoding=args.rotation_encoding,
        batch_size=args.batch_size,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        savepoint=args.savepoint,
        resolution=resolution,
    )

    if args.resume_from:
        checkpoint_path = resolve_checkpoint_path(
            os.path.join(args.logdir, args.mode), args.resume_from
        )
        print(f"Loading checkpoint from: {checkpoint_path}")
        if args.resume_from == args.run_name:
            trainer.load_checkpoint(checkpoint_path, continued=True)
        else:
            trainer.load_checkpoint(checkpoint_path, lr=args.lr)

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
