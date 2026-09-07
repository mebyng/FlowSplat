from scipy.spatial.transform import Rotation
import torch
import numpy as np


def rotation_matrix_to_quaternion(rotations):
    """Convert a batch of 3x3 rotation matrices to quaternions using scipy."""
    # scipy Rotation expects shape (N, 3, 3) in CPU numpy format.
    matrices = rotations.detach().cpu().numpy()
    quats = Rotation.from_matrix(matrices).as_quat()
    quats = torch.from_numpy(quats).to(device=rotations.device, dtype=rotations.dtype)
    return quats


def rotation_matrix_to_6d(rotations):
    """Convert a batch of 3x3 rotation matrices to 6D representation."""
    # Extract the first two columns of the rotation matrix
    return rotations[:, :, :2].reshape(rotations.shape[0], -1)  # (N, 6)


def compute_plucker(
    camera_matrix, intrinsics, height, width, fourier=True, num_frequencies=6
):
    """
    Transform camera rays to Plücker coordinates for each pixel.

    Plücker coordinates represent a line (ray) in 3D space as a 6D vector (d, m) where:
    - d: direction vector (normalized)
    - m: moment vector (point × direction)

    Args:
        camera_matrix: Camera extrinsic matrix, shape (4, 4) or (3, 4) or batch (N, 4, 4) or (N, 3, 4)
                      Format: [R | t] where R is 3x3 rotation, t is 3x1 translation
        intrinsics: Camera intrinsic matrix, shape (3, 3) or (N, 3, 3)
                   [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        height: Image height
        width: Image width
        fourier: If True, apply NeRF-style Fourier feature encoding to the Plücker coordinates.
        num_frequencies: Number of frequency bands for Fourier encoding.
        include_input: If True, keep the original Plücker coordinates alongside Fourier features.

    Returns:
        plucker_coords: Plücker coordinates or Fourier-encoded Plücker coordinates for each pixel,
                        shape (height, width, 6) or (N, height, width, 6) when fourier=False,
                        or shape (height, width, C) / (N, C, height, width) when fourier=True.
    """
    # Handle both single and batched inputs
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = torch.from_numpy(camera_matrix).float()
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float()

    is_batched = camera_matrix.dim() == 3
    if not is_batched:
        camera_matrix = camera_matrix.unsqueeze(0)
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)

    batch_size = camera_matrix.shape[0]
    device = camera_matrix.device

    # Extract rotation and translation
    R = camera_matrix[:, :3, :3]  # (N, 3, 3)
    t = camera_matrix[:, :3, 3:4]  # (N, 3, 1)

    # In this project, the pose matrix is stored as a world-to-camera basis transform, so
    # the camera center in world coordinates is the inverse rotation applied to the camera
    # translation: C = R^T @ t.
    camera_center = torch.matmul(R.transpose(-2, -1), t).squeeze(-1)  # (N, 3)

    # Extract camera intrinsics
    fx = intrinsics[:, 0, 0]  # (N,)
    fy = intrinsics[:, 1, 1]  # (N,)
    cx = intrinsics[:, 0, 2]  # (N,)
    cy = intrinsics[:, 1, 2]  # (N,)

    # Generate pixel-center coordinates in image space. Using pixel centers (0.5, 0.5, ...)
    # ensures the optical center maps to (cx, cy) exactly and the principal ray matches the
    # camera basis used by `look_at`.
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing="ij",
    )  # (height, width)

    plucker_coords = []

    for i in range(batch_size):
        # In this pose convention, the camera-space principal axis is +Z, and the world-space
        # ray is obtained by applying the inverse rotation to that camera-space vector.
        x_cam = (u - cx[i]) / fx[i]  # (height, width)
        y_cam = (cy[i] - v) / fy[
            i
        ]  # (height, width); flip row axis for image-space convention
        z_cam = torch.ones_like(x_cam)  # (height, width)

        ray_dir_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (height, width, 3)
        ray_dir_cam = ray_dir_cam / (
            torch.norm(ray_dir_cam, dim=-1, keepdim=True) + 1e-8
        )

        # Transform into world space using the inverse rotation basis of the pose matrix.
        ray_dir_world = torch.matmul(
            ray_dir_cam, R[i].transpose(0, 1)
        )  # (height, width, 3)

        # Camera center in world frame (broadcasted to match pixel grid)
        center_world = camera_center[i].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

        # Moment vector: m = C × d for each pixel
        moment = torch.cross(
            center_world.expand(height, width, -1), ray_dir_world, dim=-1
        )  # (height, width, 3)

        # Combine to form Plücker coordinates (d, m)
        plucker = torch.cat([ray_dir_world, moment], dim=-1)  # (height, width, 6)
        plucker_coords.append(plucker)

    plucker_coords = torch.stack(plucker_coords, dim=0).permute(
        0, 3, 1, 2
    )  # (N, 6, height, width)

    if fourier:
        # Convert to channels-last for encoding
        x = plucker_coords.permute(0, 2, 3, 1)  # (N, height, width, 6)
        freq_bands = 2.0 ** torch.arange(num_frequencies, device=device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-2) * freq_bands.view(
            *(1,) * (x.ndim - 1), num_frequencies, 1
        )  # (N, height, width, F, 6)
        x_expanded = x_expanded.view(
            *x.shape[:-1], num_frequencies * x.shape[-1]
        )  # (N, height, width, F*6)
        sin = torch.sin(x_expanded)
        cos = torch.cos(x_expanded)

        encoded = torch.cat([sin, cos], dim=-1)  # (N, height, width, F*12)
        plucker_coords = encoded.permute(0, 3, 1, 2)

    if not is_batched:
        plucker_coords = plucker_coords.squeeze(0)

    return plucker_coords


def default_align_cameras(cam1, cam2):
    if cam1.ndim == 3:
        rot = cam1[:, :3, :3].transpose(-2, -1)
        new_cam = torch.zeros(cam1.shape[0], 4, 4, device=cam1.device, dtype=cam1.dtype)
        new_cam[:, :3, :3] = torch.matmul(rot, cam2[:, :3, :3])
    else:
        rot = cam1[:3, :3].T
        new_cam = torch.zeros(4, 4, device=cam1.device, dtype=cam1.dtype)
        new_cam[:3, :3] = rot @ cam2[:3, :3]

    return new_cam
