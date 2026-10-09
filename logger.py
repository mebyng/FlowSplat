from pathlib import Path
import json
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

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
        metadata: dict[str, Any] | None = None,
        validation_seed: int = 0,
    ):
        if n_images <= 0 or n_samples <= 0:
            raise ValueError("n_images and n_samples must be positive integers")

        self.datasets = [("train", train_dataset)]
        if validation_dataset is not None:
            self.datasets.append(("validation", validation_dataset))

        self.validation_seed = validation_seed
        self.n_images = n_images
        self.n_samples = n_samples
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.resolution = resolution
        self.autoencoder = autoencoder
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        if metadata:
            self.log_metadata(metadata)
        self.samples = {
            name: dataset.getitems(self._select_sample_indices(dataset, name))
            for name, dataset in self.datasets
        }
        if self.mode == "gaussian":
            self.writer.add_image(
                "images_train/input", self.samples["train"]["image"][0, :3].clamp(0, 1), 0
            )
        self.random_inputs = (
            self._create_random_inputs() if self.mode == "generate" else []
        )
        self.log_path.mkdir(parents=True, exist_ok=True)

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
        generator = torch.Generator().manual_seed(self.validation_seed)
        channels = (
            4
            if (self.datasets[0][1].use_encoding or self.datasets[0][1].use_depth)
            else 3
        )
        return [
            torch.randn(
                1, channels, self.resolution, self.resolution, generator=generator
            )
            for _ in range(self.n_samples)
        ]

    def log_metrics(self, metrics: dict[str, float], epoch: int, subset: str = "train"):
        for name, value in metrics.items():
            self.writer.add_scalar(f"{subset}/{name}", value, epoch)

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        metadata_json = json.dumps(metadata, indent=2, default=str)
        metadata_path = self.log_path / "metadata.json"
        metadata_path.write_text(metadata_json, encoding="utf-8")
        self.writer.add_text("run/metadata", metadata_json, 0)

    def checkpoint_dir(self, epoch: int | None = None, final: bool = False) -> Path:
        if final:
            return self.log_path / "final"
        if epoch is None:
            raise ValueError("epoch is required when final=False")
        return self.log_path / f"epoch_{epoch:05d}"

    @staticmethod
    def _depth_comparison(target: torch.Tensor, prediction: torch.Tensor):
        """Create a normalized target/predicted depth panel for TensorBoard."""
        target = target.clamp_min(0)
        prediction = prediction.clamp_min(0)
        scale = torch.maximum(target.amax(), prediction.amax()).clamp_min(1e-6)
        comparison = torch.cat([target / scale, prediction / scale], dim=-1)
        # TensorBoard's add_images expects RGB tensors in this project version.
        return comparison.repeat(1, 3, 1, 1)

    def save(self, results, epoch: int, final: bool = False):
        save_dir = self.checkpoint_dir(epoch=epoch, final=final)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, data in results.items():
            subset_dir = save_dir / name
            subset_dir.mkdir(parents=True, exist_ok=True)
            for j, pred in enumerate(data["pred"]):
                for i in range(pred.shape[0]):
                    val_pred = pred[i].unsqueeze(0)

                    if self.mode == "gaussian":
                        alpha = data["alpha"][j][i : i + 1].clamp(0, 1)
                        depth = data["depth"][j][i : i + 1].clamp_min(0)
                        depth = depth / depth.amax().clamp_min(1e-6)
                        combined = torch.cat(
                            [
                                val_pred[:, :3].clamp(0, 1),
                                depth.repeat(1, 3, 1, 1),
                                alpha.repeat(1, 3, 1, 1),
                            ],
                            dim=-1,
                        )
                    else:
                        val_input = data["input"][j][i : i + 1]
                        val_target = data["target"][j][i : i + 1]
                        combined = torch.cat(
                            [val_input[:, :3], val_target[:, :3], val_pred[:, :3]], dim=-1
                        )
                    self.writer.add_images(
                        f"images_{name}/combined_{i}_{j}", combined, epoch
                    )
                    if self.mode == "generate":
                        val_intermediates = [
                            inter[i : i + 1] for inter in data["intermediates"][j]
                        ]
                        # RGB-D flow states have a fourth depth channel. Depth is
                        # logged separately below, while this panel shows RGB only.
                        process = torch.cat(
                            [inter[:, :3] for inter in val_intermediates], dim=-1
                        )
                        self.writer.add_images(
                            f"generation_{name}/{i}_{j}", process, epoch
                        )

                    if self.mode != "gaussian" and j < len(data.get("depth", [])):
                        target_depth, predicted_depth = data["depth"][j]
                        depth_panel = self._depth_comparison(
                            target_depth[i : i + 1],
                            predicted_depth[i : i + 1],
                        )
                        self.writer.add_images(
                            f"depth_{name}/{i}_{j}", depth_panel, epoch
                        )

    def close(self):
        self.writer.close()
