import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def normalize(x, x_min, x_max):
    """Map from [min, max] to [-1, 1]"""
    return 2 * (x - x_min) / (x_max - x_min) - 1


class UR5eDiffusionDataset(Dataset):
    def __init__(self, data_dir, chunk_size=32, num_episodes=None):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.episode_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
        self.episode_files.sort()

        stats_path = os.path.join(data_dir, "norm_stats.npz")
        stats = np.load(stats_path)
        self.qpos_min = torch.from_numpy(stats["qpos_min"]).float()
        self.qpos_max = torch.from_numpy(stats["qpos_max"]).float()
        self.action_min = torch.from_numpy(stats["action_min"]).float()
        self.action_max = torch.from_numpy(stats["action_max"]).float()

        # Slice the list if we only want a specific number of files
        if num_episodes is not None:
            self.episode_files = self.episode_files[:num_episodes]
            print(f"DEBUG MODE: Only loading {num_episodes} episode(s)...")

            if num_episodes == 1:
                print(f"file name: {self.episode_files[0]}")
        else:
            print(f"Loading all {len(self.episode_files)} episodes...")

        self.index_mapping = []

        for ep_file in self.episode_files:
            ep_path = os.path.join(data_dir, ep_file)
            with h5py.File(ep_path, "r") as f:
                ep_length = len(f["observations/qpos"])  # type: ignore
                valid_starts = ep_length - self.chunk_size
                for start_t in range(valid_starts):
                    self.index_mapping.append(
                        {"file_name": ep_path, "start_t": start_t}
                    )

        # --- NEW: Resize to 224x224 to save RAM/VRAM + Color Jitter ---
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.05),
            ]
        )

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        mapping = self.index_mapping[idx]
        ep_path = mapping["file_name"]
        start_t = mapping["start_t"]
        end_t = start_t + self.chunk_size

        with h5py.File(ep_path, "r") as f:
            # Get t-1 and t indices (duplicate t=0 if we are at the very start)
            idx_0 = max(0, start_t - 1)
            idx_1 = start_t

            scene_img_0 = np.moveaxis(f["observations/images/scene_cam"][idx_0], -1, 0)  # type: ignore
            scene_img_1 = np.moveaxis(f["observations/images/scene_cam"][idx_1], -1, 0)  # type: ignore

            wrist_img_0 = np.moveaxis(f["observations/images/gripper_cam"][idx_0], -1, 0)  # type: ignore
            wrist_img_1 = np.moveaxis(f["observations/images/gripper_cam"][idx_1], -1, 0)  # type: ignore

            # --- NEW: Extract BOTH the arm and the gripper ---
            arm_qpos_0 = f["observations/qpos"][idx_0]  # type: ignore
            arm_qpos_1 = f["observations/qpos"][idx_1]  # type: ignore

            action_chunk = f["actions"][start_t:end_t]  # type: ignore

        scene_tensor = (
            torch.from_numpy(np.stack([scene_img_0, scene_img_1])).float() / 255.0
        )
        wrist_tensor = (
            torch.from_numpy(np.stack([wrist_img_0, wrist_img_1])).float() / 255.0
        )
        qpos_tensor = torch.from_numpy(np.stack([arm_qpos_0, arm_qpos_1])).float()  # type: ignore
        action_tensor = torch.from_numpy(action_chunk).float()

        # Normalize qpos per-dimension to [-1, 1]
        qpos_tensor = normalize(qpos_tensor, self.qpos_min, self.qpos_max)

        # Normalize actions per-dimension to [-1, 1]
        action_tensor = normalize(action_tensor, self.action_min, self.action_max)

        # --- NEW: Apply the Resize transforms ---
        scene_tensor = self.image_transforms(scene_tensor)
        wrist_tensor = self.image_transforms(wrist_tensor)

        return {
            "condition": {
                "scene_cam": scene_tensor,
                "wrist_cam": wrist_tensor,
                "qpos": qpos_tensor,
            },
            "action_chunk": action_tensor,
        }
