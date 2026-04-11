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


def replace_bn_with_gn(root_module):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    Crucial for end-to-end training with small batch sizes in Diffusion Policy!
    """
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # 8 groups is standard for ResNet18 in DP
            setattr(root_module, name, nn.GroupNorm(8, module.num_features))
        else:
            replace_bn_with_gn(module)
    return root_module


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. NOT pretrained (weights=None)
        self.resnet = resnet18(weights=None)

        # 2. Replace BatchNorm with GroupNorm
        self.resnet = replace_bn_with_gn(self.resnet)

        # 3. Replace the output layer
        self.resnet.fc = nn.Linear(512, 64)

    def forward(self, img):
        # NO MORE .eval()! We are fully training end-to-end.
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
    dataset = UR5eDiffusionDataset(
        data_dir=target_data_dir, chunk_size=32, num_episodes=1  # <--- CHANGE TO 32
    )

    # FIX: Drastically lowered batch size from 64 to 4
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"Dataset loaded! Total valid chunks: {len(dataset)}")

    print("Building Neural Networks (This takes a second)...")
    scene_encoder = VisionEncoder().to(device)
    wrist_encoder = VisionEncoder().to(device)

    # THE FIX: Tell the U-Net to expect 64 steps instead of 16
    # Set sample_size back to 16
    noise_pred_net = UNet1DModel(
        sample_size=32,
        in_channels=274,
        out_channels=6,
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
        # THE FIX: Expand the channels so it isn't an information bottleneck!
        block_out_channels=(256, 512),
    )
    noise_pred_net.to(device)

    all_parameters = (
        list(noise_pred_net.parameters())
        + list(scene_encoder.parameters())
        + list(wrist_encoder.parameters())
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, prediction_type="sample"  # <--- CHANGED FROM "epsilon"
    )

    optimizer = torch.optim.AdamW(all_parameters, lr=3e-4)
    loss_fn = nn.MSELoss()

    print("Starting training")

    START_EPOCH = 0  # The epoch you are loading from your previous run
    num_epochs = 500  # Your new target (200 + 300 epochs)
    RESUME_TRAINING = False

    if RESUME_TRAINING:
        print(f"Loading saved weights from Epoch {START_EPOCH}...")
        scene_path = os.path.join(ckpt_dir, f"scene_encoder_ep{START_EPOCH}.pth")
        wrist_path = os.path.join(ckpt_dir, f"wrist_encoder_ep{START_EPOCH}.pth")
        noise_path = os.path.join(ckpt_dir, f"unet_ep{START_EPOCH}.pth")

        # Load the saved dictionaries into the models
        scene_encoder.load_state_dict(
            torch.load(scene_path, weights_only=True, map_location=device)
        )
        wrist_encoder.load_state_dict(
            torch.load(wrist_path, weights_only=True, map_location=device)
        )
        noise_pred_net.load_state_dict(
            torch.load(noise_path, weights_only=True, map_location=device)
        )
        print("Successfully loaded checkpoints!")
    else:
        START_EPOCH = 0  # If we aren't resuming, start at 0

    for epoch in range(START_EPOCH, num_epochs):
        # FIX 5: Prevent unbound variable error

        start_time = time.time()
        start_clock = datetime.now().strftime("%H:%M:%S")
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Started at {start_clock}...")

        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:

            scene_img = batch["condition"]["scene_cam"].to(device)
            wrist_img = batch["condition"]["wrist_cam"].to(device)
            qpos = batch["condition"]["qpos"].to(device)

            # Extract actions -> Shape: [Batch, 16 (Sequence), 7 (Channels)]
            # Extract actions and fix dimensions
            clean_actions = batch["action_chunk"].to(device)
            clean_actions = clean_actions.transpose(1, 2)  # Shape: [Batch, 7, 16]

            scene_t_minus_1, scene_t = scene_img[:, 0], scene_img[:, 1]
            wrist_t_minus_1, wrist_t = wrist_img[:, 0], wrist_img[:, 1]
            qpos_t_minus_1, qpos_t = qpos[:, 0], qpos[:, 1]

            # 2. Encode independently as the paper specifies
            scene_feat_0 = scene_encoder(scene_t_minus_1)
            scene_feat_1 = scene_encoder(scene_t)
            wrist_feat_0 = wrist_encoder(wrist_t_minus_1)
            wrist_feat_1 = wrist_encoder(wrist_t)

            # 3. Build the massive 270-channel condition vector
            global_condition = torch.cat(
                [
                    scene_feat_0,
                    scene_feat_1,
                    wrist_feat_0,
                    wrist_feat_1,
                    qpos_t_minus_1,
                    qpos_t,
                ],
                dim=1,
            )

            # 4. Stretch to 32 steps
            global_condition = global_condition.unsqueeze(-1).repeat(1, 1, 32)

            # 4. Generate noise for exactly 16 steps
            noise = torch.randn(clean_actions.shape, device=device)
            bsz = clean_actions.shape[0]

            # 5. Fix dtype to torch.long (int64)
            max_steps = noise_scheduler.config["num_train_timesteps"]
            timesteps = torch.randint(
                0, max_steps, (bsz,), device=device, dtype=torch.long
            )

            # 6. Add noise
            noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)

            # 7. Concatenate (135 condition + 7 action = 142 channels)
            net_input = torch.cat([noisy_actions, global_condition], dim=1)

            # 8. Predict the CLEAN actions (x_0), not the noise!
            action_pred = noise_pred_net(net_input, timesteps).sample

            # 9. Calculate MSE against the clean_actions!
            loss = loss_fn(action_pred, clean_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches

        # --- NEW: Get end time and calculate duration ---
        end_time = time.time()
        end_clock = datetime.now().strftime("%H:%M:%S")
        duration = end_time - start_time

        print(
            f"[Epoch {epoch+1}/{num_epochs}] Finished at {end_clock} | Loss: {epoch_loss:.4f} | Took: {duration:.2f} seconds"
        )

        if (epoch + 1) % 5 == 0:
            scene_path = os.path.join(ckpt_dir, f"scene_encoder_ep{epoch+1}.pth")
            wrist_path = os.path.join(ckpt_dir, f"wrist_encoder_ep{epoch+1}.pth")
            noise_path = os.path.join(ckpt_dir, f"unet_ep{epoch+1}.pth")

            torch.save(scene_encoder.state_dict(), scene_path)
            torch.save(wrist_encoder.state_dict(), wrist_path)
            torch.save(noise_pred_net.state_dict(), noise_path)
            print(f"--> Saved weights to {ckpt_dir}")
        # ------------------------------------------


if __name__ == "__main__":
    train()
