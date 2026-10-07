from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data import ViewDataset
from logger import SampleLogger
from utils.randomness import rng_state, restore_rng_state


class Trainer:
    def __init__(
        self,
        dataset: ViewDataset,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: SampleLogger,
        validation_dataset: ViewDataset | None = None,
        autoencoder: torch.nn.Module | None = None,
        mode: str = "generate",
        rotation_encoding: str = "matrix",
        batch_size: int = 32,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        savepoint: int = 10,
        resolution: int = 128,
        grad_clip: float = 1.0,
        warmup_epochs: int = 0,
    ):
        if grad_clip < 0 or warmup_epochs < 0:
            raise ValueError("grad_clip and warmup_epochs must be nonnegative")
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.autoencoder = autoencoder
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.rotation_encoding = rotation_encoding
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.savepoint = savepoint
        self.resolution = resolution
        self.logger = logger
        self.start_epoch = 1

        self.train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = None
        if self.validation_dataset is not None:
            self.val_loader = DataLoader(
                self.validation_dataset, batch_size=self.batch_size, shuffle=False
            )

    def _prepare_batch(self, batch):
        """Move a batch to the model device and provide reconstruction targets for AE."""
        device = self.model.device()
        if self.mode == "encode":
            input = batch.to(device)
            return input, input, None

        input, target, input_pose, target_pose, intrinsics = batch
        input = input.to(device)
        target = target.to(device)
        input_pose = input_pose.to(device)
        target_pose = target_pose.to(device)
        intrinsics = intrinsics.to(device)
        cameras = {
            "source": input_pose,
            "target": target_pose,
            "intrinsics": intrinsics,
        }
        return input, target, cameras

    def train_step(self, batch):
        input, target, cameras = self._prepare_batch(batch)
        loss, loss_dict = self.model.train_step(input, target, cameras)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip or float("inf"),
            error_if_nonfinite=True,
        )
        loss_dict = dict(loss_dict, grad_norm=grad_norm.detach())
        self.optimizer.step()
        return loss_dict, target.shape[0]

    def val_step(self, batch):
        input, target, cameras = self._prepare_batch(batch)
        self.model.eval()
        with torch.no_grad():
            if self.mode == "generate":
                noise = (
                    self.logger.random_inputs[0]
                    .to(input.device)
                    .repeat(input.shape[0], 1, 1, 1)
                )
            else:
                noise = None
            if self.mode == "encode":
                prediction = self.model.model(input)
            else:
                prediction = self.model.generate(input, noise, cameras)
            loss, loss_dict = self.model.criterion(prediction, target)
        return loss_dict, target.shape[0]

    def train_epoch(self, epoch: int = 1):
        max_grad_norm = 0.0
        total_count = 0
        loss_sums = {}
        self.model.train()
        for batch_index, batch in enumerate(self.train_loader):
            if epoch <= self.warmup_epochs:
                warmup_steps = self.warmup_epochs * len(self.train_loader)
                step = (epoch - 1) * len(self.train_loader) + batch_index
                factor = 0.1 + 0.9 * step / max(warmup_steps - 1, 1)
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr * factor
            loss_dict, count = self.train_step(batch)
            max_grad_norm = max(max_grad_norm, loss_dict["grad_norm"].item())
            total_count += count
            for name, value in loss_dict.items():
                loss_sums[name] = (
                    loss_sums.get(name, 0.0) + value.detach().item() * count
                )

        scale = max(total_count, 1)
        metrics = {name: value / scale for name, value in loss_sums.items()}
        metrics["grad_norm_max"] = max_grad_norm
        return metrics

    def val_epoch(self):
        if self.val_loader is None:
            raise ValueError("Validation dataset is not provided for val_epoch")
        total_count = 0
        loss_sums = {}
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                loss_dict, count = self.val_step(batch)
                total_count += count
                for name, value in loss_dict.items():
                    loss_sums[name] = (
                        loss_sums.get(name, 0.0) + value.detach().item() * count
                    )

        scale = max(total_count, 1)
        return {name: value / scale for name, value in loss_sums.items()}

    def log_images(self, epoch: int, final: bool = False):
        datasets = [("train", self.dataset)]
        if self.validation_dataset is not None:
            datasets.append(("validation", self.validation_dataset))

        results = {}
        self.model.eval()
        for name, dataset in datasets:
            batch = self.logger.samples[name]
            # In rotate mode, there's one fixed target per input, so j is meaningless
            # In generate/encode mode, j represents different random samples
            n_loops = 1 if self.mode != "generate" else self.logger.n_samples
            results[name] = {
                "input": [],
                "target": [],
                "pred": [],
                "intermediates": [],
                "depth": [],
            }

            input, target, cameras = self._prepare_batch(batch)
            for j in range(n_loops):
                with torch.no_grad():
                    if self.mode == "generate":
                        # For generation, we need to sample different random inputs
                        noise = (
                            self.logger.random_inputs[j]
                            .to(input.device)
                            .repeat(input.shape[0], 1, 1, 1)
                        )
                    else:
                        noise = None

                    pred, intermediates = self.model.generation_log(
                        input, noise, cameras
                    )
                    intermediates = intermediates or []

                    if dataset.use_encoding:
                        if self.autoencoder is None:
                            raise ValueError(
                                "encoding is used but no autoencoder provided"
                            )
                        input = self.autoencoder.decode(input)
                        target = self.autoencoder.decode(target)
                        pred = self.autoencoder.decode(pred)
                        intermediates = [
                            self.autoencoder.decode(inter) for inter in intermediates
                        ]

                    results[name]["input"].append(input[:, :3])
                    results[name]["target"].append(target)
                    results[name]["pred"].append(pred)
                    results[name]["intermediates"].append(intermediates)
                    if dataset.use_depth and not dataset.use_encoding:
                        results[name]["depth"].append((target[:, 3:4], pred[:, 3:4]))

        self.logger.save(results, epoch, final=final)

    def save_checkpoint(
        self, epoch: int | None = None, final: bool = False, path: str | None = None
    ):
        save_path = (
            Path(path)
            if path is not None
            else self.logger.checkpoint_dir(epoch=epoch, final=final)
            / "model_checkpoint.pth"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "rng_state": rng_state(),
            "warmup_epochs": self.warmup_epochs,
            "base_lrs": self.base_lrs,
            "grad_clip": self.grad_clip,
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        return save_path

    def load_checkpoint(
        self, path: str, continued: bool = False, lr: float | None = None
    ):
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        scheduler_loaded = False
        fresh_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if (
                continued
                and self.scheduler is not None
                and "scheduler_state_dict" in checkpoint
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scheduler_loaded = True
        else:
            self.model.model.load_state_dict(checkpoint)

        if lr is not None and not scheduler_loaded:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        if not continued:
            for group, fresh_lr in zip(self.optimizer.param_groups, fresh_lrs):
                group["lr"] = fresh_lr if lr is None else lr
                group["initial_lr"] = group["lr"]
            self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        if continued:
            self.warmup_epochs = checkpoint.get("warmup_epochs", 0)
            self.base_lrs = checkpoint.get("base_lrs", self.base_lrs)
            self.grad_clip = checkpoint.get("grad_clip", self.grad_clip)
            if "rng_state" in checkpoint:
                restore_rng_state(checkpoint["rng_state"])
            # Extract epoch number from the checkpoint path if possible
            epoch_str = checkpoint_path.parent.name
            if checkpoint.get("epoch") is not None:
                self.start_epoch = checkpoint["epoch"] + 1
            elif epoch_str.startswith("epoch_"):
                self.start_epoch = int(epoch_str.split("_")[-1]) + 1
            else:
                self.start_epoch = 1

        return checkpoint

    def train(self, epochs):
        if self.start_epoch > epochs:
            raise ValueError("checkpoint has already reached the requested epoch count")
        epoch_bar = trange(self.start_epoch, epochs + 1, desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_metrics = self.train_epoch(epoch)
            avg_loss = train_metrics.get("total", 0.0)
            epoch_bar.set_postfix({"loss": f"{avg_loss:.6f}"})

            self.logger.log_metrics(train_metrics, epoch, subset="train")
            if self.validation_dataset is not None:
                val_metrics = self.val_epoch()
                self.logger.log_metrics(val_metrics, epoch, subset="validation")

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_metrics(
                {"learning_rate": current_lr}, epoch, subset="train"
            )

            if self.scheduler is not None and epoch > self.warmup_epochs:
                self.scheduler.step()

            if epoch % self.savepoint == 0:
                self.save_checkpoint(epoch=epoch)
                self.log_images(epoch, final=False)

        self.save_checkpoint(epoch=epoch, final=True)
        self.log_images(epoch, final=True)

        self.logger.close()
        return self.model
