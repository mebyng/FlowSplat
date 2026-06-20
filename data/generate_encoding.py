from pathlib import Path
import csv
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import ViewDataset
from models import SimpleAutoEncoder


def generate_encoding(dataset, autoencoder, batch_size=32, device=None, output_dirname="encodings", overwrite=False):
    """Run every dataset image through the autoencoder encoder and save latent encodings.

    Args:
        dataset: ViewDataset instance with loaded scene metadata.
        autoencoder: Autoencoder model with an `encode(x)` method.
        batch_size: Number of images to process in a single batch.
        device: Device to move tensors and model to. If None, infer from the model.
        output_dirname: Directory name under each scene containing encodings.
        overwrite: If True, overwrite existing encoding files.
    """
    if device is None:
        try:
            device = next(autoencoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    autoencoder.eval()

    for scene_dir, image_list in dataset.scene_info.items():
        encoding_dir = scene_dir / output_dirname
        encoding_dir.mkdir(exist_ok=True)

        csv_path = scene_dir / "camera_matrices.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if "encoded_filename" not in fieldnames:
            fieldnames = fieldnames + ["encoded_filename"]

        for image_info in image_list:
            image_path = image_info["image_path"]
            encoding_path = encoding_dir / f"{image_path.stem}.pt"
            if not overwrite and encoding_path.exists():
                continue
            image = dataset._load_image(image_path).unsqueeze(0).to(device)
            with torch.no_grad():
                encoding = autoencoder.encode(image)
            torch.save(encoding.cpu(), encoding_path)

        for row in rows:
            filename = row.get("filename")
            if filename:
                row["encoded_filename"] = str(Path(output_dirname) / f"{Path(filename).stem}.pt")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    dataset_path = "datasets/small_512"
    dataset = ViewDataset(dataset_path, split="validation", mode="encode")
    autoencoder = SimpleAutoEncoder().cuda()
    autoencoder.load("weight_checkpoints/SimpleAutoEncoder.pth")
    generate_encoding(dataset, autoencoder)