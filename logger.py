from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from data import ViewDataset
from utils import compute_plucker


class SampleLogger:
    def __init__(
        self,
        train_dataset: ViewDataset,
        validation_dataset: ViewDataset | None,
        n_images: int,
        n_samples: int,
        log_path: str,
        mode: str,
        resolution: int,
        autoencoder: torch.nn.Module | None = None,
    ):
        if n_images <= 0 or n_samples <= 0:
            raise ValueError("n_images and n_samples must be positive integers")

        self.datasets = [("train", train_dataset)]
        if validation_dataset is not None:
            self.datasets.append(("validation", validation_dataset))

        self.n_images = n_images
        self.n_samples = n_samples
        self.log_path = Path(log_path)
        self.mode = mode
        self.resolution = resolution
        self.autoencoder = autoencoder
        self.sample_indices = {
            name: self._select_sample_indices(dataset, name)
            for name, dataset in self.datasets
        }
        self.random_inputs = (
            self._create_random_inputs() if self.mode == "generate" else []
        )

    def _select_sample_indices(self, dataset: ViewDataset, name: str):
        dataset_length = len(dataset)
        if dataset_length == 0:
            raise ValueError(f"{name} dataset is empty")
        if dataset_length <= self.n_images:
            return list(range(dataset_length))

        generator = torch.Generator()
        generator.manual_seed(42)
        return torch.randperm(dataset_length, generator=generator)[
            : self.n_images
        ].tolist()

    def _create_random_inputs(self):
        channels = 64 if self.datasets[0][1].use_encoding else 3
        return [
            torch.randn(1, channels, self.resolution, self.resolution)
            for _ in range(self.n_samples)
        ]

    def save(self, epoch: int, model: torch.nn.Module, final: bool = False):
        save_dir = self.log_path / ("final" if final else f"epoch_{epoch:05d}")
        save_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(self.log_path))
        device = model.device()
        random_inputs = [tensor.to(device) for tensor in self.random_inputs]

        for name, dataset in self.datasets:
            subset_dir = save_dir / name
            subset_dir.mkdir(parents=True, exist_ok=True)
            indices = self.sample_indices[name]

            for i in indices:
                # In rotate mode, there's one fixed target per input, so j is meaningless
                # In generate/encode mode, j represents different random samples
                n_loops = 1 if self.mode != "generate" else self.n_samples

                for j in range(n_loops):
                    with torch.no_grad():
                        if self.mode == "rotate":
                            (
                                val_input,
                                val_target,
                                val_target_extrinsics,
                                val_intrinsics,
                            ) = dataset[i]
                            val_input = val_input.unsqueeze(0).to(device)
                            val_target = val_target.unsqueeze(0).to(device)
                            val_plucker = compute_plucker(
                                val_target_extrinsics.unsqueeze(0).to(device),
                                val_intrinsics.unsqueeze(0).to(device),
                                height=self.resolution,
                                width=self.resolution,
                            )
                        elif self.mode == "generate":
                            val_target, val_extrinsics, val_intrinsics = dataset[i]
                            val_target = val_target.unsqueeze(0).to(device)
                            val_plucker = compute_plucker(
                                val_extrinsics.unsqueeze(0).to(device),
                                val_intrinsics.unsqueeze(0).to(device),
                                height=self.resolution,
                                width=self.resolution,
                            )
                            val_input = random_inputs[j]
                        elif self.mode == "encode":
                            val_input = dataset[i].unsqueeze(0).to(device)
                            val_target = val_input
                            val_plucker = None
                        else:
                            raise ValueError(f"Invalid mode: {self.mode}")

                        val_pred = model.generate(val_input, val_plucker)

                        if dataset.use_encoding:
                            if self.autoencoder is None:
                                raise ValueError(
                                    "encoding is used but no autoencoder provided"
                                )
                            val_input = self.autoencoder.decode(val_input)
                            val_target = self.autoencoder.decode(val_target)
                            val_pred = self.autoencoder.decode(val_pred)

                        save_image(val_input, str(subset_dir / f"{i}_{j}_input.png"))
                        save_image(val_target, str(subset_dir / f"{i}_{j}_target.png"))
                        save_image(
                            val_pred, str(subset_dir / f"{i}_{j}_prediction.png")
                        )

                        combined = torch.cat([val_input, val_target, val_pred], dim=-1)
                        writer.add_images(f"{name}/combined_{i}_{j}", combined, epoch)

        writer.close()
