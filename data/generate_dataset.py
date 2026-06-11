from create_scene import make_scene
from render_scene import render_views, render_views_random


def create_dataset(num_scenes=10, split="training", output_dir="dataset", random=True):
    for i in range(num_scenes):
        scene = make_scene()
        if random:
            render_views_random(scene, num_views=100, radius=2.0, output_dir=f"{output_dir}/{split}/scene_{i:03d}")
        else:
            render_views(scene, step_z=10, step_x=10, radius=2.0, output_dir=f"{output_dir}/{split}/scene_{i:03d}")

if __name__ == "__main__":
    create_dataset(num_scenes=8, split="training", output_dir="dataset", random=True)
    create_dataset(num_scenes=2, split="validation", output_dir="dataset", random=True)
