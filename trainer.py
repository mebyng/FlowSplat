import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data import ViewDataset
from utils import compute_plucker
from logger import SampleLogger


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
    ):
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

        self.train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = None
        if self.validation_dataset is not None:
            self.val_loader = DataLoader(
                self.validation_dataset, batch_size=self.batch_size, shuffle=False
            )

    def _prepare_batch(self, batch):
        if self.mode == "rotate":
            input, target, target_extrinsics, intrinsics = batch
            input = input.to(self.model.device())
            target = target.to(self.model.device())
            target_extrinsics = target_extrinsics.to(self.model.device())
            intrinsics = intrinsics.to(self.model.device())
            if self.rotation_encoding == "matrix":
                rotation = target_extrinsics.reshape(-1, 16, 1, 1).repeat(
                    1, 1, self.resolution, self.resolution
                )
            elif self.rotation_encoding == "plucker":
                rotation = compute_plucker(
                    target_extrinsics,
                    intrinsics,
                    height=self.resolution,
                    width=self.resolution,
                )
        elif self.mode == "generate":
            input, target, target_extrinsics, intrinsics = batch
            input = input.to(self.model.device())
            input = torch.cat([input, torch.randn_like(input)], dim=1)
            target = target.to(self.model.device())
            target_extrinsics = target_extrinsics.to(self.model.device())
            intrinsics = intrinsics.to(self.model.device())
            if self.rotation_encoding == "matrix":
                rotation = target_extrinsics.reshape(-1, 16, 1, 1).repeat(
                    1, 1, self.resolution, self.resolution
                )
            elif self.rotation_encoding == "plucker":
                rotation = compute_plucker(
                    target_extrinsics,
                    intrinsics,
                    height=self.resolution,
                    width=self.resolution,
                )
        elif self.mode == "encode":
            input = batch.to(self.model.device())
            target = input
            rotation = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return input, rotation, target

    def train_step(self, batch):
        input, rotation, target = self._prepare_batch(batch)
        loss, loss_dict = self.model.train_step(input, rotation, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, target.shape[0]

    def val_step(self, batch):
        input, rotation, target = self._prepare_batch(batch)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.generate(input, rotation)
            loss, loss_dict = self.model.criterion(
                prediction, target
            )  # TODO does not work for AE
        return loss_dict, target.shape[0]

    def train_epoch(self):
        total_count = 0
        loss_sums = {}
        for batch in self.train_loader:
            loss_dict, count = self.train_step(batch)
            total_count += count
            for name, value in loss_dict.items():
                loss_sums[name] = (
                    loss_sums.get(name, 0.0) + value.detach().item() * count
                )

        scale = max(total_count, 1)
        return {name: value / scale for name, value in loss_sums.items()}

    def val_epoch(self):
        if self.val_loader is None:
            raise ValueError("Validation dataset is not provided for val_epoch")
        total_count = 0
        loss_sums = {}
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
        for name, dataset in datasets:
            indices = self.logger.sample_indices[name]
            # In rotate mode, there's one fixed target per input, so j is meaningless
            # In generate/encode mode, j represents different random samples
            n_loops = 1 if self.mode != "generate" else self.logger.n_samples
            results[name] = {"input": [], "target": [], "pred": []}

            batch = dataset.getitems(indices)
            input, rotation, target = self._prepare_batch(batch)
            for j in range(n_loops):
                with torch.no_grad():
                    if self.mode == "generate":
                        # For generation, we need to sample different random inputs
                        noise = (
                            self.logger.random_inputs[j]
                            .to(input.device)
                            .repeat(input.shape[0], 1, 1, 1)
                        )
                        input = torch.cat(
                            [
                                input[:, :3],
                                noise,
                            ],
                            dim=1,
                        )

                    pred = self.model.generate(input, rotation)

                    if dataset.use_encoding:
                        if self.autoencoder is None:
                            raise ValueError(
                                "encoding is used but no autoencoder provided"
                            )
                        input = self.autoencoder.decode(input)
                        target = self.autoencoder.decode(target)
                        pred = self.autoencoder.decode(pred)

                    results[name]["input"].append(input[:, :3])
                    results[name]["target"].append(target)
                    results[name]["pred"].append(pred)

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
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        return save_path

    def load_checkpoint(self, path: str, lr: float | None = None):
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        scheduler_loaded = False
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            self.model.model.load_state_dict(checkpoint)

        if lr is not None and not scheduler_loaded:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        return checkpoint

    def train(self, epochs):
        epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_metrics = self.train_epoch()
            avg_loss = train_metrics.get("total", 0.0)
            epoch_bar.set_postfix({"loss": f"{avg_loss:.6f}"})

            self.logger.log_metrics(train_metrics, epoch, subset="train")
            if self.validation_dataset is not None:
                val_metrics = self.val_epoch()
                self.logger.log_metrics(val_metrics, epoch, subset="validation")

            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics(
                    {"learning_rate": current_lr}, epoch, subset="train"
                )

            if epoch % self.savepoint == 0:
                self.save_checkpoint(epoch=epoch)
                self.log_images(epoch, final=False)

            if self.scheduler is not None:
                self.scheduler.step()
        self.save_checkpoint(epoch=epoch, final=True)
        self.log_images(epoch, final=True)

        self.logger.close()
        return self.model
