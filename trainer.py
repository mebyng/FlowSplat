import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        validation_dataset: ViewDataset | None = None,
        autoencoder: torch.nn.Module | None = None,
        mode: str = "generate",
        batch_size: int = 32,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        savepoint: int = 10,
        log_dir: str = "runs",
        resolution: int = 128,
        validation_n_images: int = 5,
        validation_n_samples: int = 2,
    ):
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.autoencoder = autoencoder
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.savepoint = savepoint
        self.log_dir = log_dir
        self.resolution = resolution

        self.train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = None
        self.logger = SampleLogger(
            train_dataset=self.dataset,
            validation_dataset=self.validation_dataset,
            n_images=validation_n_images,
            n_samples=validation_n_samples,
            log_path=self.log_dir,
            mode=self.mode,
            resolution=self.resolution,
            autoencoder=self.autoencoder,
        )
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
            plucker = compute_plucker(
                target_extrinsics,
                intrinsics,
                height=self.resolution,
                width=self.resolution,
            )
        elif self.mode == "generate":
            target, extrinsics, intrinsics = batch
            target = target.to(self.model.device())
            extrinsics = extrinsics.to(self.model.device())
            intrinsics = intrinsics.to(self.model.device())
            input = torch.randn_like(target)
            plucker = compute_plucker(
                extrinsics, intrinsics, height=self.resolution, width=self.resolution
            )
        elif self.mode == "encode":
            input = batch.to(self.model.device())
            target = input
            plucker = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return input, plucker, target

    def train_step(self, batch):
        input, plucker, target = self._prepare_batch(batch)
        loss, loss_dict = self.model.train_step(input, plucker, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, target.shape[0]

    def val_step(self, batch):
        input, plucker, target = self._prepare_batch(batch)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.generate(input, plucker)
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

    def train(self, epochs):
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir)

        epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_metrics = self.train_epoch()
            avg_loss = train_metrics.get("total", 0.0)
            epoch_bar.set_postfix({"loss": f"{avg_loss:.6f}"})
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch)

            # if self.validation_dataset is not None:
            #     val_metrics = self.val_epoch()
            #     for name, value in val_metrics.items():
            #         writer.add_scalar(f"val/{name}", value, epoch)

            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/learning_rate", current_lr, epoch)

            if epoch % self.savepoint == 0:
                self.logger.save(epoch, self.model)

            if self.scheduler is not None:
                self.scheduler.step()
        self.logger.save(epoch, self.model, final=True)

        writer.close()
        return self.model
