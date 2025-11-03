import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npy(path: str) -> np.ndarray:

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Skeleton file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected array of shape (T, J, C), got {arr.shape} at {path}")
    return arr


def _temporal_resize(seq: np.ndarray, num_frames: int) -> np.ndarray:
   
    T, J, C = seq.shape
    if T == num_frames:
        return seq

    # Use linear spacing of indices
    idx = np.linspace(0, T - 1, num_frames)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, T - 1)
    return seq[idx]


class WLASLDataset(Dataset):
 

    def __init__(
        self,
        root: str,
        split_file: str,
        num_frames: int = 52,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.split_file = split_file
        self.num_frames = int(num_frames)
        self.normalize = bool(normalize)

        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples: List[Tuple[str, int]] = []
        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                video_id, label_str = parts[0], parts[1]
                label = int(label_str)
                pose_path = os.path.join(self.root, "poses", f"{video_id}.npy")
                self.samples.append((pose_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {self.split_file}. "
                f"Please check the format: each line 'video_id label'."
            )

        # Try to infer shape from first sample
        first_pose, _ = self.samples[0]
        arr = _load_npy(first_pose)
        self.num_joints = arr.shape[1]
        self.num_coords = arr.shape[2]

        print(
            f"[WLASLDataset] Loaded {len(self.samples)} samples from {self.split_file}. "
            f"Example shape: (T=?, J={self.num_joints}, C={self.num_coords}), "
            f"resampled to T={self.num_frames}."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pose_path, label = self.samples[idx]
        arr = _load_npy(pose_path)  # (T, J, C)
        arr = _temporal_resize(arr, self.num_frames)

        if self.normalize:
           
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
            arr = (arr - mean) * (1.0 / std)

  
        tensor = torch.from_numpy(arr).float()
        label = int(label)
        return tensor, label
