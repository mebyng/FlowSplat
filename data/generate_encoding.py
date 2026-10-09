import argparse
from pathlib import Path
import csv
import json
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import ViewDataset  # noqa: E402
from utils.checkpoints import load_encoding_model  # noqa: E402


def generate_encoding(
    dataset,
    autoencoder,
    batch_size=32,
    device=None,
    output_dirname="encodings",
    overwrite=False,
    *,
    run_name,
):
    """Run every dataset image through the autoencoder encoder and save latent encodings.

    Args:
        dataset: ViewDataset instance with loaded scene metadata.
        autoencoder: Autoencoder model with an `encode(x)` method.
        batch_size: Number of images to process in a single batch.
        device: Device to move tensors and model to. If None, infer from the model.
        output_dirname: Directory name under each scene containing encodings.
        overwrite: If True, overwrite existing encoding files.
        run_name: Autoencoder run used to generate the saved encodings.
    """
    if device is None:
        try:
            device = next(autoencoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    autoencoder.eval()

    scene_rows = {}
    for scene_dir in dataset.scene_info:
        csv_path = scene_dir / "camera_matrices.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if "encoded_filename" not in fieldnames:
            fieldnames = fieldnames + ["encoded_filename"]

        scene_rows[scene_dir] = (rows, fieldnames)

    for sample_idx in range(len(dataset)):
        scene_dir, img_idx = dataset.samples[sample_idx]
        sample = dataset[sample_idx]
        if not isinstance(sample, torch.Tensor):
            continue

        image_path = dataset.scene_info[scene_dir][img_idx][0]
        encoding_dir = scene_dir / output_dirname
        encoding_dir.mkdir(parents=True, exist_ok=True)
        encoding_path = encoding_dir / f"{image_path.stem}.pt"
        if not overwrite and encoding_path.exists():
            continue

        image = sample.unsqueeze(0).to(device)
        with torch.no_grad():
            encoding = autoencoder.encode(image)
        torch.save(encoding.cpu(), encoding_path)

    for scene_dir, (rows, fieldnames) in scene_rows.items():
        csv_path = scene_dir / "camera_matrices.csv"
        for row in rows:
            filename = row.get("filename")
            if filename:
                row["encoded_filename"] = str(
                    Path(output_dirname) / f"{Path(filename).stem}.pt"
                )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metadata_path = dataset.root.parent / "encoding_metadata.json"
    metadata_path.write_text(json.dumps({"run_name": run_name}, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute dataset latents from an autoencoder run."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--run-name",
        required=True,
        help="AE run under <logdir>/encode; final checkpoint preferred.",
    )
    parser.add_argument("--logdir", default="runs")
    parser.add_argument(
        "--split", choices=["training", "validation", "both"], default="both"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dirname", default="encodings")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.output_dirname)
    if directory.is_absolute() or ".." in directory.parts or not directory.parts:
        parser.error("--output-dirname must be a relative directory within each scene")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    autoencoder, checkpoint = load_encoding_model(args.run_name, args.logdir, device)
    print(f"Loaded autoencoder: {checkpoint}")
    channels = 4 if autoencoder.use_depth.item() else 3
    if channels not in (3, 4):
        raise ValueError(
            f"Expected an RGB or RGB-D autoencoder, got {channels} input channels"
        )
    splits = ("training", "validation") if args.split == "both" else (args.split,)
    for split in splits:
        dataset = ViewDataset(
            args.dataset_path,
            split=split,
            mode="encode",
            random_matching=False,
            use_depth=channels == 4,
        )
        print(f"Encoding {split}: {len(dataset)} images")
        generate_encoding(
            dataset,
            autoencoder,
            device=device,
            output_dirname=args.output_dirname,
            overwrite=args.overwrite,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
