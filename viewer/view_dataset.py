"""Launch the orbit viewer for one RGB view from a generated dataset.

Run with:
    python -m viewer.view_dataset DATASET_PATH INDEX

Pass --run-name NAME to load its final checkpoint or highest saved epoch. Press Space to predict
an RGB-D view from the original dataset image and append it to the cloud.
"""

from __future__ import annotations

import argparse

import torch

from data import ViewDataset
from viewer.camera_orbit import OrbitCamera
from viewer.orbit_viewer import OrbitPointCloudViewer
from viewer.point_cloud import depth_to_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View RGB-D points and generate additional views with Space."
    )
    parser.add_argument(
        "dataset_path",
        help="Dataset root containing training/ and validation/ directories.",
    )
    parser.add_argument("index", type=int, help="Zero-based index in the selected split.")
    parser.add_argument(
        "--split",
        choices=["training", "validation"],
        default="training",
        help="Dataset split to read (default: training).",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=2.0,
        help="Constant-depth fallback when ground-truth depth is unavailable.",
    )
    parser.add_argument(
        "--depth-source",
        choices=["auto", "ground-truth", "constant"],
        default="auto",
        help=(
            "Depth for the initial dataset image. 'auto' and 'ground-truth' use "
            "the saved depth channel; use 'constant' for --depth (default: auto)."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Viewer orbit radius; matches generate_dataset's camera radius by default.",
    )
    parser.add_argument("--run-name", help="RGB-D run whose final checkpoint or highest saved epoch is loaded.")
    parser.add_argument("--logdir", default="runs", help="Training runs directory.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda, etc.")
    parser.add_argument("--steps", type=int, default=10, help="Flow integration steps.")
    return parser.parse_args()


def make_initial_depth(image: torch.Tensor, value: float) -> torch.Tensor:
    """Return a temporary positive z-depth map for an RGB dataset image."""
    if value <= 0:
        raise ValueError("--depth must be positive")
    return torch.full(image.shape[1:], value, dtype=image.dtype, device=image.device)


def main() -> None:
    args = parse_args()
    if args.radius <= 0:
        raise ValueError("--radius must be positive")

    predictor = None
    if args.run_name:
        from viewer.predictor import ViewPredictor

        predictor = ViewPredictor(args.run_name, args.logdir, args.device, args.steps)
        print(f"Loaded checkpoint: {predictor.checkpoint}")

    use_ground_truth_depth = args.depth_source in {"auto", "ground-truth"}
    dataset = ViewDataset(
        args.dataset_path,
        split=args.split,
        mode="rotate",
        random_matching=False,
        use_depth=use_ground_truth_depth,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index {args.index} is outside the {args.split} split "
            f"(valid range: 0 to {len(dataset) - 1})"
        )

    input_image, _, source_pose, _, intrinsics = dataset[args.index]
    if use_ground_truth_depth:
        image = input_image[:3]
        depth = input_image[3]
        depth_message = "ground-truth depth from the dataset"
    else:
        image = input_image
        depth = make_initial_depth(image, args.depth)
        depth_message = f"temporary constant depth {args.depth}"
    points, colors = depth_to_point_cloud(image, depth, intrinsics, source_pose)

    # Preserve the selected image's orbit direction while enforcing the viewer's
    # fixed dataset-generation radius.
    source_orbit = OrbitCamera.from_pose(source_pose)
    viewer = OrbitPointCloudViewer(radius=args.radius)
    viewer.camera = OrbitCamera(
        radius=args.radius,
        azimuth=source_orbit.azimuth,
        elevation=source_orbit.elevation,
    )
    viewer.set_point_cloud(points, colors)

    if predictor is not None:
        # Keep this source and its pose fixed for every requested target view.
        original_rgbd = torch.cat([image, depth.unsqueeze(0)], dim=0).clone()
        original_pose = source_pose.clone()

        def generate_view(target_pose):
            print("Predicting view from the original image...", flush=True)
            try:
                prediction = predictor.predict(
                    original_rgbd, original_pose, target_pose, intrinsics
                )
                new_points, new_colors = depth_to_point_cloud(
                    prediction[:3], prediction[3], intrinsics, target_pose
                )
                valid = torch.isfinite(new_points).all(dim=1) & torch.isfinite(new_colors).all(dim=1)
                viewer.add_point_cloud(new_points[valid], new_colors[valid])
                print(f"Added {int(valid.sum()):,} points. Rotate and press Space for another view.")
            except (RuntimeError, ValueError) as exc:
                print(f"Could not generate view: {exc}")

        viewer.on_space = generate_view
        print("Drag to orbit; press Space to generate and add a view.")

    print(
        f"Viewing {args.split} sample {args.index}: {len(points):,} points. "
        f"Using {depth_message}."
    )
    viewer.run()


if __name__ == "__main__":
    main()
