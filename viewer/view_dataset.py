"""Launch the orbit viewer for one RGB view from a generated dataset.

Run with:
    python -m viewer.view_dataset DATASET_PATH INDEX

Pass --run-name NAME to load its final checkpoint or highest saved epoch. Press Space to predict
an RGB-D view from the original dataset image and append it to the cloud.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from data import ViewDataset
from viewer.camera_orbit import OrbitCamera
from viewer.orbit_viewer import OrbitPointCloudViewer
from viewer.point_cloud import depth_to_point_cloud


if TYPE_CHECKING:
    from viewer.predictor import ViewPredictor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View RGB-D points and generate additional views with Space."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        help="Dataset root containing training/ and validation/ directories.",
    )
    parser.add_argument(
        "index", type=int, nargs="?", default=0,
        help="Zero-based index in the selected split (default: 0)."
    )
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
    parser.add_argument(
        "--run-name",
        help="RGB-D run whose final checkpoint or highest saved epoch is loaded.",
    )
    parser.add_argument("--logdir", default="runs", help="Training runs directory.")
    parser.add_argument(
        "--gaussian-run",
        help="Gaussian run under <logdir>/gaussian, or checkpoint path; Tab toggles splats.",
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device: auto, cpu, cuda, etc."
    )
    parser.add_argument("--steps", type=int, default=10, help="Flow integration steps.")
    args = parser.parse_args(argv)
    if args.dataset_path is None:
        if not args.gaussian_run:
            parser.error("provide a dataset path or --gaussian-run")
        if args.run_name:
            parser.error("--run-name requires a dataset source image")
    return args


def make_initial_depth(image: torch.Tensor, value: float) -> torch.Tensor:
    """Return a temporary positive z-depth map for an RGB dataset image."""
    if value <= 0:
        raise ValueError("--depth must be positive")
    return torch.full(image.shape[1:], value, dtype=image.dtype, device=image.device)


def make_prediction_callback(
    viewer: OrbitPointCloudViewer,
    predictor: ViewPredictor,
    image: torch.Tensor,
    depth: torch.Tensor,
    source_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    fallback_depth: float = 2.0,
) -> Callable[[torch.Tensor], None]:
    """Capture the original RGB-D source and append each predicted target view."""
    # Keep this source and its pose fixed for every requested target view.
    original_rgbd = torch.cat([image, depth.unsqueeze(0)], dim=0).clone()
    original_pose = source_pose.clone()

    def generate_view(target_pose: torch.Tensor) -> None:
        print("Predicting view from the original image...", flush=True)
        try:
            prediction = predictor.predict(
                original_rgbd, original_pose, target_pose, intrinsics
            )
            new_points, new_colors = depth_to_point_cloud(
                prediction[:3],
                prediction[3] if prediction.shape[0] == 4 else make_initial_depth(prediction, fallback_depth),
                intrinsics, target_pose
            )
            valid = torch.isfinite(new_points).all(dim=1) & torch.isfinite(
                new_colors
            ).all(dim=1)
            viewer.add_point_cloud(new_points[valid], new_colors[valid])
            print(
                f"Added {int(valid.sum()):,} points. Rotate and press Space for another view."
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Could not generate view: {exc}")

    return generate_view


def main() -> None:
    args = parse_args()
    if args.radius <= 0:
        raise ValueError("--radius must be positive")

    gaussian_scene = None
    if args.gaussian_run:
        from viewer.splats import load_splats

        gaussian_scene, checkpoint = load_splats(args.gaussian_run, args.logdir, args.device)
        print(f"Loaded Gaussian checkpoint: {checkpoint}.")
    viewer = OrbitPointCloudViewer(radius=args.radius, gaussian_scene=gaussian_scene)
    if args.dataset_path is None:
        viewer.view_mode = "splats"
        viewer.set_point_cloud(gaussian_scene.means, gaussian_scene.colors)
        print("Drag to orbit; Tab switches between Gaussian centers and splats.")
        viewer.run()
        return

    predictor = None
    if args.run_name:
        from viewer.predictor import ViewPredictor

        predictor = ViewPredictor(args.run_name, args.logdir, args.device, args.steps)
        print(f"Loaded checkpoint: {predictor.checkpoint}")
        if predictor.encoding_checkpoint is not None:
            print(f"Loaded autoencoder checkpoint: {predictor.encoding_checkpoint}")
        if predictor.channels == 3:
            make_initial_depth(torch.empty(3, 1, 1), args.depth)
            print(f"Generated RGB views will use temporary constant depth {args.depth}.")

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
    if gaussian_scene is not None:
        print("Press Tab to toggle point cloud/splats.")
    viewer.camera = OrbitCamera(
        radius=args.radius,
        azimuth=source_orbit.azimuth,
        elevation=source_orbit.elevation,
    )
    viewer.set_point_cloud(points, colors)

    if predictor is not None:
        viewer.on_space = make_prediction_callback(
            viewer, predictor, image, depth, source_pose, intrinsics, args.depth
        )
        print("Drag to orbit; press Space to generate and add a view.")

    print(
        f"Viewing {args.split} sample {args.index}: {len(points):,} points. "
        f"Using {depth_message}."
    )
    viewer.run()


if __name__ == "__main__":
    main()
