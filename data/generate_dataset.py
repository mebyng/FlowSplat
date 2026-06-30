from create_scene import make_scene
from render_scene import render_views, render_views_random


def create_dataset(
    num_scenes=10,
    num_views=100,
    resolution=128,
    split="training",
    output_dir="datasets",
    random=True,
):
    for i in range(num_scenes):
        scene = make_scene()
        if random:
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
        num_scenes=10,
        num_views=100,
        resolution=512,
        split="training",
        output_dir="datasets/small_512",
        random=True,
    )
    create_dataset(
        num_scenes=1,
        num_views=5,
        resolution=512,
        split="validation",
        output_dir="datasets/small_512",
        random=True,
    )
