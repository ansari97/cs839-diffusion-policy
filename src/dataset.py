import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UR5eDiffusionDataset(Dataset):
    def __init__(self, data_dir, chunk_size=16):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.episode_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
        self.episode_files.sort()

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
                transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.05),
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
            scene_img = f["observations/images/scene_cam"][start_t]  # type: ignore
            wrist_img = f["observations/images/gripper_cam"][start_t]  # type: ignore

            # --- NEW: Extract BOTH the arm and the gripper ---
            arm_qpos = f["observations/qpos"][start_t]  # type: ignore
            gripper_qpos = f["observations/gripper_qpos"][start_t]  # type: ignore

            # Glue them together into a single 7-number array!
            full_qpos = np.concatenate([arm_qpos, gripper_qpos])  # type: ignore

            action_chunk = f["actions"][start_t:end_t]  # type: ignore

            # OpenCV has HxWx3, Pytorch requires 3xWxH
            scene_img = np.moveaxis(scene_img, -1, 0)  # type: ignore
            wrist_img = np.moveaxis(wrist_img, -1, 0)  # type: ignore

        scene_tensor = torch.from_numpy(scene_img).float() / 255.0
        wrist_tensor = torch.from_numpy(wrist_img).float() / 255.0

        # Pass the newly combined 7-number state
        qpos_tensor = torch.from_numpy(full_qpos).float()
        action_tensor = torch.from_numpy(action_chunk).float()

        # --- NEW: Apply the Resize & Jitter transforms ---
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
