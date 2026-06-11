from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import csv



class ViewDataset(Dataset):
    def __init__(self, root, split="training", transform=None, target_transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.samples = []
        self.scene_info = {}

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        scene_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not scene_dirs:
            raise ValueError(f"No scene directories found in {self.root}")

        for scene_dir in scene_dirs:
            image_list = []
            csv_path = scene_dir / "camera_matrices.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"camera_matrices.csv not found in {scene_dir}")
            idx = 0
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename")
                    if not filename:
                        continue
                    mat = np.zeros((4, 4), dtype=np.float32)
                    for j in range(16):
                        mat.flat[j] = float(row.get(f"m{j}", 0.0))
                    img_path = scene_dir / filename
                    if img_path.exists():
                        image_list.append((img_path, mat))
            if len(image_list) <= 1:
                raise ValueError(f"Not enough images listed in camera_matrices.csv for {scene_dir}")
            self.scene_info[scene_dir] = image_list
            for idx in range(len(image_list)):
                self.samples.append((scene_dir, idx))

        if not self.samples:
            raise ValueError(f"No valid image samples found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image

    def __getitem__(self, idx):
        scene_dir, img_idx = self.samples[idx]
        image_list = self.scene_info[scene_dir]

        input_path, input_pose = image_list[img_idx]

        # choose target depending on split
        if self.split == "training":
            while True:
                next_idx = np.random.randint(0, len(image_list))
                if next_idx != img_idx:
                    break
        else:
            next_idx = (img_idx + 1) % len(image_list)

        target_path, target_pose = image_list[next_idx]

        input_image = self._load_image(input_path)
        target_image = self._load_image(target_path)

        if self.target_transform is not None:
            target_image = self.target_transform(target_image)

        # compute rotation matrix transforming coords from input camera to target camera    
        Ra = input_pose[:3, :3]
        Rb = target_pose[:3, :3]
        R = Rb.T @ Ra

        R = torch.from_numpy(R.astype(np.float32))
        return input_image, target_image, R