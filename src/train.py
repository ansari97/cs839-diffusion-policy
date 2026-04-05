import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# FIX 1 & 2: Strict imports for Diffusers
from diffusers.models.unets.unet_1d import UNet1DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from torchvision.models import resnet18, ResNet18_Weights
from dataset import UR5eDiffusionDataset

import time
from datetime import datetime


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # type: ignore # ignore the last layer

    def forward(self, img):
        return self.resnet(img)


def train():
    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(current_dir, "..", "data")

    # --- ADD THIS ---
    ckpt_dir = os.path.join(current_dir, "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading Dataset...")
    dataset = UR5eDiffusionDataset(data_dir=target_data_dir, chunk_size=16)

    # FIX: Drastically lowered batch size from 64 to 4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset loaded! Total valid chunks: {len(dataset)}")

    print("Building Neural Networks (This takes a second)...")
    vision_encoder = VisionEncoder().to(device)

    # THE FIX: Tell the U-Net to expect 64 steps instead of 16
    noise_pred_net = UNet1DModel(
        sample_size=64,
        in_channels=1095,
        out_channels=7,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(32, 64, 128),
    )
    noise_pred_net.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=100)

    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters()) + list(noise_pred_net.parameters()), lr=1e-4
    )
    loss_fn = nn.MSELoss()

    print("Starting training")

    num_epochs = 100
    for epoch in range(num_epochs):
        # FIX 5: Prevent unbound variable error

        start_time = time.time()
        start_clock = datetime.now().strftime("%H:%M:%S")
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Started at {start_clock}...")

        epoch_loss = 0.0

        for batch in dataloader:

            scene_img = batch["condition"]["scene_cam"].to(device)
            wrist_img = batch["condition"]["wrist_cam"].to(device)
            qpos = batch["condition"]["qpos"].to(device)

            # Extract actions -> Shape: [Batch, 16 (Sequence), 7 (Channels)]
            # Extract actions and fix dimensions
            clean_actions = batch["action_chunk"].to(device)
            clean_actions = clean_actions.transpose(1, 2)

            # THE FIX: Pad the 16-step sequence out to 64 steps
            # (0, 48) means add 0 elements to the left side, and 48 elements to the right
            clean_actions_padded = torch.nn.functional.pad(clean_actions, (0, 48))

            # Build the 1088-dimension Condition Vector
            scene_features = vision_encoder(scene_img)
            wrist_features = vision_encoder(wrist_img)
            qpos_padded = torch.nn.functional.pad(qpos, (0, 57))
            global_condition = torch.cat(
                [scene_features, wrist_features, qpos_padded], dim=1
            )

            # THE FIX: Broadcast the condition vector across all 64 timesteps
            # Shape goes from [Batch, 1088] -> [Batch, 1088, 64]
            global_condition = global_condition.unsqueeze(-1).repeat(1, 1, 64)

            # Generate noise for the full 64 steps
            noise = torch.randn(clean_actions_padded.shape, device=device)
            bsz = clean_actions_padded.shape[0]

            max_steps = noise_scheduler.config["num_train_timesteps"]
            timesteps = torch.randint(
                0, max_steps, (bsz,), device=device, dtype=torch.int32
            )

            # Add Noise to the padded actions
            noisy_actions = noise_scheduler.add_noise(clean_actions_padded, noise, timesteps)  # type: ignore

            # Glue noisy actions (7) and condition (1088) together
            # Final input shape: [Batch, 1095, 64]
            net_input = torch.cat([noisy_actions, global_condition], dim=1)

            # Predict the noise
            noise_pred = noise_pred_net(net_input, timesteps).sample

            # THE FIX: Calculate Loss ONLY on our valid 16 steps!
            # We completely ignore the 48 steps of padding we added.
            loss = loss_fn(noise_pred[:, :, :16], noise[:, :, :16])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()

        # --- NEW: Get end time and calculate duration ---
        end_time = time.time()
        end_clock = datetime.now().strftime("%H:%M:%S")
        duration = end_time - start_time

        print(
            f"[Epoch {epoch+1}/{num_epochs}] Finished at {end_clock} | Loss: {epoch_loss:.4f} | Took: {duration:.2f} seconds"
        )

        if (epoch + 1) % 5 == 0:
            vision_path = os.path.join(ckpt_dir, f"vision_encoder_ep{epoch+1}.pth")
            noise_path = os.path.join(ckpt_dir, f"noise_pred_net_ep{epoch+1}.pth")

            torch.save(vision_encoder.state_dict(), vision_path)
            torch.save(noise_pred_net.state_dict(), noise_path)
            print(f"--> Saved weights to {ckpt_dir}")
        # ------------------------------------------


if __name__ == "__main__":
    train()
