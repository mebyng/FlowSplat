from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from data import ViewDataset


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

    def save(self, results, epoch: int, final: bool = False):
        save_dir = self.log_path / ("final" if final else f"epoch_{epoch:05d}")
        save_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(self.log_path))
        for name, data in results.items():
            subset_dir = save_dir / name
            subset_dir.mkdir(parents=True, exist_ok=True)
            for j, (input, target, pred) in enumerate(
                zip(data["input"], data["target"], data["pred"])
            ):
                for i in range(input.shape[0]):
                    val_input = input[i].unsqueeze(0)
                    val_target = target[i].unsqueeze(0)
                    val_pred = pred[i].unsqueeze(0)

                    save_image(val_input, str(subset_dir / f"{i}_{j}_input.png"))
                    save_image(val_target, str(subset_dir / f"{i}_{j}_target.png"))
                    save_image(val_pred, str(subset_dir / f"{i}_{j}_prediction.png"))

                    combined = torch.cat(
                        [val_input[:, :3], val_target, val_pred], dim=-1
                    )
                    writer.add_images(
                        f"images_{name}/combined_{i}_{j}", combined, epoch
                    )

        writer.close()
