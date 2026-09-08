from create_scene import make_scene
from render_scene import render_views, render_views_random
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
):
    for i in tqdm(range(num_scenes)):
        scene = make_scene(
            random_type=random_type, random_color=random_color, random_size=random_size
        )
        if random_views:
            render_views_random(
                scene,
                num_views=num_views,
                radius=2.0,
                resolution=resolution,
                output_dir=f"{output_dir}/{split}/scene_{i:03d}",
            )
        else:
            render_views(
                scene,
                step_z=10,
                step_x=10,
                radius=2.0,
                resolution=resolution,
                output_dir=f"{output_dir}/{split}/scene_{i:03d}",
            )


if __name__ == "__main__":
    create_dataset(
        num_scenes=100000,
        num_views=10,
        resolution=128,
        split="training",
        output_dir="datasets/scenes_100000_10",
        random_views=True,
        random_type=True,
        random_color=True,
        random_size=True,
    )
    create_dataset(
        num_scenes=10,
        num_views=10,
        resolution=128,
        split="validation",
        output_dir="datasets/scenes_100000_10",
        random_views=True,
        random_type=True,
        random_color=True,
        random_size=True,
    )
