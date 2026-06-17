import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import ViewDataset
from models import FlowMatching
from models.FlowModel import DebugRegression
from models.UNetDummy import RotationConditionedUNet
from models.UNet import UNet
from utils import compute_plucker
from torchvision.utils import save_image


def train(
    dataset: ViewDataset,
    model: FlowMatching,
    optimizer: torch.optim.Optimizer,
    validation_dataset: ViewDataset = None,
    mode: str = "generate",
    epochs: int = 10,
    batch_size: int = 32,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    savepoint: int = 10,
    results_dir: str = "results",
    log_dir: str = "runs",
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir=log_dir)

    criterion = torch.nn.MSELoss()
    base_random1 = torch.randn(1, 3, 128, 128).to(model.device())
    base_random2 = torch.randn(1, 3, 128, 128).to(model.device())

    epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        count = 0

        for x in loader:
            if mode == "rotate":
                input, target, target_extrinsics, intrinsics = x
                input, target, target_extrinsics, intrinsics = input.to(model.device()), target.to(model.device()), target_extrinsics.to(model.device()), intrinsics.to(model.device())
                plucker = compute_plucker(target_extrinsics, intrinsics, height=128, width=128)
            elif mode == "generate":
                target, extrinsics, intrinsics = x
                target, extrinsics, intrinsics = target.to(model.device()), extrinsics.to(model.device()), intrinsics.to(model.device())
                input = torch.randn_like(target) 
                plucker = compute_plucker(extrinsics, intrinsics, height=128, width=128) # (B, 6, H, W)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            

            loss = model.train_step(input, plucker, target, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.shape[0]
            count += target.shape[0]

        avg_loss = total_loss / max(count, 1)
        epoch_bar.set_postfix({"MSE": f"{avg_loss:.6f}"})
        
        # Log to TensorBoard
        writer.add_scalar("train/loss", avg_loss, epoch)
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", current_lr, epoch)

        # Save validation checkpoint every N epochs
        if validation_dataset is not None and epoch % savepoint == 0:
            for i in range(5):  # Save 5 random samples from validation set
                for j in range(2):
                    with torch.no_grad():
                        if mode == "rotate":
                            val_input, val_target, val_target_extrinsics, val_intrinsics = validation_dataset[i]
                            val_input = val_input.unsqueeze(0).to(model.device())
                            val_target = val_target.unsqueeze(0).to(model.device())
                            val_plucker = compute_plucker(val_target_extrinsics.unsqueeze(0).to(model.device()), val_intrinsics.unsqueeze(0).to(model.device()), height=128, width=128)
                        elif mode == "generate":
                            val_target, val_extrinsics, val_intrinsics = validation_dataset[i]
                            val_target = val_target.unsqueeze(0).to(model.device())
                            val_plucker = compute_plucker(val_extrinsics.unsqueeze(0).to(model.device()), val_intrinsics.unsqueeze(0).to(model.device()), height=128, width=128)
                            val_input = base_random1 if j == 0 else base_random2  # Use fixed random inputs for consistency
                        else:
                            raise ValueError(f"Invalid mode: {mode}")

                        val_pred = model.generate(val_input, val_plucker)

                        # Save images as PNG
                        epoch_dir = os.path.join(results_dir, f"epoch_{epoch:05d}")
                        os.makedirs(epoch_dir, exist_ok=True)

                        # Convert tensors to numpy and save
                        save_image(val_input, os.path.join(epoch_dir, f"{i}_{j}_input.png"))
                        save_image(val_target, os.path.join(epoch_dir, f"{i}_{j}_target.png"))
                        save_image(val_pred, os.path.join(epoch_dir, f"{i}_{j}_prediction.png"))

                        combined = torch.cat([val_input, val_target, val_pred], dim=-1)
                        
                        # Log images to TensorBoard
                        writer.add_images(f"validation/combined_{i}_{j}", combined, epoch)
                        # print(f"  Saved checkpoint to {epoch_dir}")

        if scheduler is not None:
            scheduler.step()
    
    writer.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model to generate rotated scenes.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="Learning rate scheduler type")
    parser.add_argument("--scheduler-step", type=int, default=100, help="StepLR step size")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    parser.add_argument("--dataset-path", type=str, default="datasets/small/")
    parser.add_argument("--mode", type=str, default="rotate", choices=["rotate", "generate"], help="Training mode: 'rotate' for learning rotations, 'generate' for learning to generate views directly (default: 'generate')")
    parser.add_argument("--savepoint", type=int, default=100, help="Save model and plots every N epochs (default: 10)")
    parser.add_argument("--logdir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for this training run (appended to logdir)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ViewDataset(args.dataset_path, split="training", mode=args.mode)
    val_dataset = ViewDataset(args.dataset_path, split="training", mode=args.mode, random_matching=False)  # Use deterministic matching for validation
    model = DebugRegression(RotationConditionedUNet(in_channels=75), mode=args.mode).cuda()

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

    train(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        validation_dataset=val_dataset,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler=scheduler,
        savepoint=args.savepoint,
        results_dir="results",
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
