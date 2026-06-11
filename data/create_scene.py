import trimesh
import pyrender
import numpy as np

valid_colors = [
    [200, 80, 80, 255],   # Red-ish
    [80, 200, 80, 255],   # Green-ish
    [80, 80, 200, 255],   # Blue-ish
    [200, 200, 80, 255],  # Yellow-ish
    [200, 80, 200, 255],  # Magenta-ish
    [80, 200, 200, 255],  # Cyan-ish
]


def generate_random_box():
    center = np.random.uniform(-0.5, 0.5, size=3)
    max_diagonal = (1.0 - np.linalg.norm(center)) * 2  # needs to be inside unit sphere
    extent = np.random.uniform(0.1, max_diagonal, size=3)
    diagonal = np.square(extent).sum() ** 0.5
    if diagonal > max_diagonal:
        scale = max_diagonal / diagonal
        extent *= scale
    box = trimesh.creation.box(extents=extent)
    box.apply_translation(center)
    box.visual.vertex_colors = valid_colors[np.random.randint(len(valid_colors))]
    return box


def generate_random_sphere():
    center = np.random.uniform(-0.5, 0.5, size=3)
    max_radius = 1.0 - np.linalg.norm(center)  # needs to be inside unit sphere
    radius = np.random.uniform(0.1, max_radius)
    sphere = trimesh.creation.icosphere(radius=radius)
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = valid_colors[np.random.randint(len(valid_colors))]
    return sphere



def make_scene(num_objects=3):
    
    scene = trimesh.Scene()
    for _ in range(num_objects):
        if np.random.rand() < 0.5:
            obj = generate_random_box()
        else:
            obj = generate_random_sphere()
        scene.add_geometry(obj)

    mesh = scene.dump(concatenate=True) 
    render_scene = pyrender.Scene()

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    render_scene.add(render_mesh)

    # simple light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    render_scene.add(light)

    return render_scene
