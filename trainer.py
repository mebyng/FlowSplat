import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import trange

from data import ViewDataset
from models import AutoEncoderLoss
from utils import compute_plucker


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
        results_dir: str = "results",
        log_dir: str = "runs",
        resolution: int = 128,
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
        self.results_dir = results_dir
        self.log_dir = log_dir
        self.resolution = resolution

        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = None
        if self.validation_dataset is not None:
            self.val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)

    def _build_criterion(self) -> torch.nn.Module:
        if self.mode in ["rotate", "generate"]:
            return torch.nn.MSELoss()
        elif self.mode == "encode":
            return AutoEncoderLoss()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _prepare_batch(self, batch):
        if self.mode == "rotate":
            input, target, target_extrinsics, intrinsics = batch
            input = input.to(self.model.device())
            target = target.to(self.model.device())
            target_extrinsics = target_extrinsics.to(self.model.device())
            intrinsics = intrinsics.to(self.model.device())
            plucker = compute_plucker(target_extrinsics, intrinsics, height=self.resolution, width=self.resolution)
        elif self.mode == "generate":
            target, extrinsics, intrinsics = batch
            target = target.to(self.model.device())
            extrinsics = extrinsics.to(self.model.device())
            intrinsics = intrinsics.to(self.model.device())
            input = torch.randn_like(target)
            plucker = compute_plucker(extrinsics, intrinsics, height=self.resolution, width=self.resolution)
        elif self.mode == "encode":
            input = batch.to(self.model.device())
            target = input
            plucker = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return input, plucker, target

    def train_step(self, batch, criterion: torch.nn.Module):
        input, plucker, target = self._prepare_batch(batch)
        loss = self.model.train_step(input, plucker, target, criterion)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), target.shape[0]

    def val_step(self, batch, criterion: torch.nn.Module):
        input, plucker, target = self._prepare_batch(batch)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.generate(input, plucker)
            loss = criterion(prediction, target)
        return loss.item(), target.shape[0]

    def train_epoch(self):
        total_loss = 0.0
        total_count = 0
        criterion = self._build_criterion()
        for batch in self.train_loader:
            loss_item, count = self.train_step(batch, criterion)
            total_loss += loss_item * count
            total_count += count
        return total_loss / max(total_count, 1)

    def val_epoch(self):
        if self.val_loader is None:
            raise ValueError("Validation dataset is not provided for val_epoch")
        total_loss = 0.0
        total_count = 0
        criterion = self._build_criterion()
        for batch in self.val_loader:
            loss_item, count = self.val_step(batch, criterion)
            total_loss += loss_item * count
            total_count += count
        return total_loss / max(total_count, 1)

    def _save_validation_samples(self, epoch: int):
        if self.validation_dataset is None:
            return
        epoch_dir = os.path.join(self.results_dir, f"epoch_{epoch:05d}")
        os.makedirs(epoch_dir, exist_ok=True)
        base_random1 = torch.randn(1, 3, self.resolution, self.resolution, device=self.model.device())
        base_random2 = torch.randn(1, 3, self.resolution, self.resolution, device=self.model.device())

        for i in range(5):
            for j in range(2):
                with torch.no_grad():
                    if self.mode == "rotate":
                        val_input, val_target, val_target_extrinsics, val_intrinsics = self.validation_dataset[i]
                        val_input = val_input.unsqueeze(0).to(self.model.device())
                        val_target = val_target.unsqueeze(0).to(self.model.device())
                        val_plucker = compute_plucker(
                            val_target_extrinsics.unsqueeze(0).to(self.model.device()),
                            val_intrinsics.unsqueeze(0).to(self.model.device()),
                            height=self.resolution,
                            width=self.resolution,
                        )
                    elif self.mode == "generate":
                        val_target, val_extrinsics, val_intrinsics = self.validation_dataset[i]
                        val_target = val_target.unsqueeze(0).to(self.model.device())
                        val_plucker = compute_plucker(
                            val_extrinsics.unsqueeze(0).to(self.model.device()),
                            val_intrinsics.unsqueeze(0).to(self.model.device()),
                            height=self.resolution,
                            width=self.resolution,
                        )
                        val_input = base_random1 if j == 0 else base_random2
                    elif self.mode == "encode":
                        val_input = self.validation_dataset[i].unsqueeze(0).to(self.model.device())
                        val_target = val_input
                        val_plucker = None
                    else:
                        raise ValueError(f"Invalid mode: {self.mode}")

                    val_pred = self.model.generate(val_input, val_plucker)

                    if self.validation_dataset.use_encoding:
                        if self.autoencoder is None:
                            raise ValueError("encoding is used but no autoencoder provided")
                        val_input = self.autoencoder.decode(val_input)
                        val_target = self.autoencoder.decode(val_target)
                        val_pred = self.autoencoder.decode(val_pred)

                    save_image(val_input, os.path.join(epoch_dir, f"{i}_{j}_input.png"))
                    save_image(val_target, os.path.join(epoch_dir, f"{i}_{j}_target.png"))
                    save_image(val_pred, os.path.join(epoch_dir, f"{i}_{j}_prediction.png"))

                    combined = torch.cat([val_input, val_target, val_pred], dim=-1)
                    writer = SummaryWriter(log_dir=self.log_dir)
                    writer.add_images(f"validation/combined_{i}_{j}", combined, epoch)
                    writer.close()

    def train(self, epochs):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir)

        criterion = self._build_criterion()
        epoch_bar = trange(1, epochs + 1, desc="Training", unit="epoch")
        for epoch in epoch_bar:
            avg_loss = self.train_epoch()
            epoch_bar.set_postfix({"MSE": f"{avg_loss:.6f}"})
            writer.add_scalar("train/loss", avg_loss, epoch)

            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/learning_rate", current_lr, epoch)

            if epoch % self.savepoint == 0:
                self._save_validation_samples(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

        writer.close()
        return self.model
