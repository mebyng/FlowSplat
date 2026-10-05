import argparse

import numpy as np
from tqdm import tqdm


def create_dataset(
    num_scenes=10,
    num_views=100,
    resolution=128,
    split="training",
    output_dir="datasets",
    random_views=True,
    random_size=True,
    random_color=True,
    random_type=True,
    radius=2.0,
    num_objects=3,
    step_z=10,
    step_x=10,
):
    if __package__:
        from .create_scene import make_scene
        from .render_scene import render_views, render_views_random
    else:
        from create_scene import make_scene
        from render_scene import render_views, render_views_random

    for i in tqdm(range(num_scenes)):
        scene = make_scene(
            num_objects=num_objects,
            random_type=random_type,
            random_color=random_color,
            random_size=random_size,
        )
        if random_views:
            render_views_random(
                scene,
                num_views=num_views,
                radius=radius,
                resolution=resolution,
                output_dir=f"{output_dir}/{split}/scene_{i:03d}",
            )
        else:
            render_views(
                scene,
                step_z=step_z,
                step_x=step_x,
                radius=radius,
                resolution=resolution,
                output_dir=f"{output_dir}/{split}/scene_{i:03d}",
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-view datasets."
    )
    parser.add_argument("--output-dir", default="datasets/scenes_10000_10")
    parser.add_argument(
        "--training-scenes",
        type=int,
        default=10000,
        help="Training scene count; 0 skips this split.",
    )
    parser.add_argument(
        "--validation-scenes",
        type=int,
        default=10,
        help="Validation scene count; 0 skips this split.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=10,
        help="Random views per scene (minimum 2).",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional NumPy seed for scene and camera generation.",
    )
    parser.add_argument(
        "--random-size", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--random-color", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--random-type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mix boxes and spheres; disabled means spheres only.",
    )
    args = parser.parse_args(argv)
    if args.training_scenes < 0 or args.validation_scenes < 0:
        parser.error("scene counts must be nonnegative")
    if args.training_scenes + args.validation_scenes == 0:
        parser.error("at least one split must have scenes")
    if args.num_views < 2:
        parser.error("--num-views must be at least 2")
    if args.resolution <= 0 or args.num_objects <= 0:
        parser.error("--resolution and --num-objects must be positive")
    if not 0 < args.radius < float("inf"):
        parser.error("--radius must be positive and finite")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)
    options = vars(args).copy()
    training_scenes = options.pop("training_scenes")
    validation_scenes = options.pop("validation_scenes")
    options.pop("seed")
    for split, count in (
        ("training", training_scenes),
        ("validation", validation_scenes),
    ):
        if count:
            create_dataset(num_scenes=count, split=split, **options)


if __name__ == "__main__":
    main()
