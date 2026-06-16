import os
import numpy as np
import csv
import pyrender
from PIL import Image


def rotation_matrix(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "z":
        return np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
    elif axis == "x":
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
    elif axis == "y":
        return np.array([
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
    raise ValueError(f"Unsupported axis: {axis}")


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward /= np.linalg.norm(forward)

    if abs(np.dot(forward, up)) > 0.9999:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def generate_camera_grid(step_z=10, step_x=10, radius=1.0):
    default_camera = np.array([0.0, 0.0, radius, 1.0], dtype=np.float32)
    default_up = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    grid = []
    for iz in range(step_z):
        z_angle = 2.0 * np.pi * iz / step_z
        Rz = rotation_matrix("z", z_angle)
        for ix, x_angle in enumerate(np.linspace(-np.pi / 2.0, np.pi / 2.0, step_x)):
            Rx = rotation_matrix("x", x_angle)
            cam_pos = Rz @ Rx @ default_camera
            cam_up = Rz @ Rx @ default_up
            grid.append((iz, ix, cam_pos[:3], cam_up[:3]))
    return grid


def compute_intrinsic_matrix(yfov, resolution):
    """Compute camera intrinsic matrix from field of view and resolution."""
    fy = resolution / (2.0 * np.tan(yfov / 2.0))
    fx = fy  # Assuming square pixels
    cx = resolution / 2.0
    cy = resolution / 2.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K


def render_views(scene, step_z=10, step_x=10, radius=2.0, resolution=128, output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # Compute and save intrinsic matrix
    K = compute_intrinsic_matrix(np.pi / 3.0, resolution)
    intrinsic_path = os.path.join(output_dir, "intrinsics.csv")
    with open(intrinsic_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in K:
            writer.writerow(row)

    camera_grid = generate_camera_grid(step_z=step_z, step_x=step_x, radius=radius)
    for iz, ix, cam_pos, cam_up in camera_grid:
        pose = look_at(cam_pos, np.zeros(3, dtype=np.float32), up=cam_up)
        cam_node = scene.add(camera, pose=pose)
        color, depth = renderer.render(scene)
        Image.fromarray(color).save(f"{output_dir}/render_z{iz:02d}_x{ix:02d}.png")
        scene.remove_node(cam_node)

    renderer.delete()


def generate_random_cameras(num_views=100, radius=1.0):
    cams = []
    for i in range(num_views):
        z = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r_xy = np.sqrt(max(0.0, 1.0 - z * z))
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        pos = np.array([x, y, z], dtype=np.float32) * radius
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cams.append((i, pos, up))
    return cams


def render_views_random(scene, num_views=100, radius=2.0, resolution=128, output_dir="datasets", csv_path=None):
    os.makedirs(output_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # Compute and save intrinsic matrix
    K = compute_intrinsic_matrix(np.pi / 3.0, resolution)
    intrinsic_path = os.path.join(output_dir, "intrinsics.csv")
    with open(intrinsic_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in K:
            writer.writerow(row)

    camera_list = generate_random_cameras(num_views=num_views, radius=radius)
    records = []
    for idx, cam_pos, cam_up in camera_list:
        pose = look_at(cam_pos, np.zeros(3, dtype=np.float32), up=cam_up)
        cam_node = scene.add(camera, pose=pose)
        color, depth = renderer.render(scene)
        fname = f"{output_dir}/render_rand_{idx:04d}.png"
        Image.fromarray(color).save(fname)
        scene.remove_node(cam_node)

        mat_flat = pose.reshape(-1).tolist()
        record = {f"m{j}": float(mat_flat[j]) for j in range(16)}
        record["filename"] = os.path.basename(fname)
        records.append(record)

    renderer.delete()

    if csv_path is None:
        csv_path = os.path.join(output_dir, "camera_matrices.csv")

    if records:
        fieldnames = ["filename"] + [f"m{j}" for j in range(16)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
