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

from diffusers.training_utils import EMAModel
import copy

from diffusers.optimization import get_scheduler

# constants
action_chunk_size = 32
vis_encoder_out = 128
qpos_size = 6
action_size = 6


class SpatialSoftmax(nn.Module):
    """Keypoint-based spatial pooling. Output size = 2 * num_kp."""

    def __init__(self, in_channels, num_kp=64):
        super().__init__()
        self.num_kp = num_kp
        self.reduce = nn.Conv2d(in_channels, num_kp, kernel_size=1)

    def forward(self, feat):
        # feat: [B, in_channels, H, W]
        feat = self.reduce(feat)  # [B, num_kp, H, W]
        B, C, H, W = feat.shape

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=feat.device),
            torch.linspace(-1.0, 1.0, W, device=feat.device),
            indexing="ij",
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        feat = feat.reshape(B, C, H * W)
        softmax = torch.softmax(feat, dim=-1)

        exp_x = (softmax * pos_x).sum(dim=-1)  # [B, num_kp]
        exp_y = (softmax * pos_y).sum(dim=-1)  # [B, num_kp]
        return torch.cat([exp_x, exp_y], dim=-1)  # [B, 2*num_kp]


def replace_bn_with_gn(root_module):
    """Unchanged — still needed."""
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(root_module, name, nn.GroupNorm(8, module.num_features))
        else:
            replace_bn_with_gn(module)
    return root_module


class VisionEncoder(nn.Module):
    def __init__(self, num_kp=64):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone = replace_bn_with_gn(backbone)
        # Keep the ResNet trunk up through layer4; drop avgpool + fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.spatial_softmax = SpatialSoftmax(in_channels=512, num_kp=num_kp)
        self.out_dim = 2 * num_kp  # 128 for num_kp=64

    def forward(self, img):
        feat_map = self.backbone(img)  # [B, 512, 7, 7] for 224 input
        return self.spatial_softmax(feat_map)  # [B, 128]


def train(training_with, train_with_occlusions):

    print(f"Training: {training_with}, Occlusions: {train_with_occlusions}")

    print("Initializing device...")

    print(f"Action chunk size: {action_chunk_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(current_dir, "..", "data")

    # training_with = "with_obstacles"
    target_data_dir = os.path.join(target_data_dir, training_with)
    print(f"target_data_dir: {target_data_dir}")

    # --- ADD THIS ---
    ckpt_dir = os.path.join(current_dir, "..", "checkpoints")

    if train_with_occlusions:
        ckpt_dir = os.path.join(ckpt_dir, "occlusions")
    else:
        ckpt_dir = os.path.join(ckpt_dir, training_with)

    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading Dataset...")
    dataset = UR5eDiffusionDataset(
        data_dir=target_data_dir,
        chunk_size=action_chunk_size,
        num_episodes=None,
        train_with_occlusions=train_with_occlusions,
        occlusion_prob=0.3,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=24,  # or 24 if you want to push
        shuffle=True,
        num_workers=4,  # keep at 4 — data loading is the bottleneck
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"Dataset loaded! Total valid chunks: {len(dataset)}")

    print("Building Neural Networks (This takes a second)...")

    # Vision encoders
    scene_encoder = VisionEncoder().to(device)
    wrist_encoder = VisionEncoder().to(device)

    # NN for diffusion model
    noise_pred_net = UNet1DModel(
        sample_size=action_chunk_size,
        in_channels=vis_encoder_out * 4 + qpos_size * 2 + action_size,
        out_channels=action_size,
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
        # THE FIX: Expand the channels so it isn't an information bottleneck!
        block_out_channels=(256, 512),
    )
    noise_pred_net.to(device)  # type: ignore

    print(f"UNet in_channels: {noise_pred_net.config.in_channels}")  # type: ignore

    # Create EMA wrappers — 0.75 power is the DP paper default
    ema_scene_encoder = EMAModel(parameters=scene_encoder.parameters(), power=0.75)
    ema_wrist_encoder = EMAModel(parameters=wrist_encoder.parameters(), power=0.75)
    ema_noise_pred_net = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

    all_parameters = (
        list(noise_pred_net.parameters())
        + list(scene_encoder.parameters())
        + list(wrist_encoder.parameters())
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, prediction_type="sample"  # <--- CHANGED FROM "epsilon"
    )

    optimizer = torch.optim.AdamW(all_parameters, lr=1e-4)
    loss_fn = nn.MSELoss()

    START_EPOCH = 0  # The epoch you are loading from your previous run
    num_epochs = 200  # Your new target (200 + 300 epochs)
    RESUME_TRAINING = True

    num_training_steps = num_epochs * len(dataloader)  # type: ignore
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )

    print("Starting training")

    if RESUME_TRAINING:
        # Find the highest epoch checkpoint available
        ckpt_files = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("ckpt_ep") and f.endswith(".pth")
        ]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}!")

        epochs_found = [
            int(f.replace("ckpt_ep", "").replace(".pth", "")) for f in ckpt_files
        ]
        latest_epoch = max(epochs_found)
        resume_path = os.path.join(ckpt_dir, f"ckpt_ep{latest_epoch}.pth")

        print(f"Auto-resuming from epoch {latest_epoch}: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        scene_encoder.load_state_dict(ckpt["scene_encoder"])
        wrist_encoder.load_state_dict(ckpt["wrist_encoder"])
        noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        ema_scene_encoder.load_state_dict(ckpt["ema_scene_encoder"])
        ema_wrist_encoder.load_state_dict(ckpt["ema_wrist_encoder"])
        ema_noise_pred_net.load_state_dict(ckpt["ema_noise_pred_net"])
        START_EPOCH = ckpt["epoch"]
    else:
        START_EPOCH = 0  # If we aren't resuming, start at 0

    # for loop
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

            # Extract actions -> Shape: [Batch, action_chunk_size (Sequence), 7 (Channels)]
            # Extract actions and fix dimensions
            clean_actions = batch["action_chunk"].to(device)
            clean_actions = clean_actions.transpose(
                1, 2
            )  # Shape: [Batch, 7, action_chunk_size]

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

            # 4. Stretch to action_chunk_size steps
            global_condition = global_condition.unsqueeze(-1).repeat(
                1, 1, action_chunk_size
            )

            # 4. Generate noise for action_chunk_size steps
            noise = torch.randn(clean_actions.shape, device=device)
            bsz = clean_actions.shape[0]

            # 5. Fix dtype to torch.long (int64)
            max_steps = noise_scheduler.config["num_train_timesteps"]
            timesteps = torch.randint(
                0, max_steps, (bsz,), device=device, dtype=torch.long
            )

            # 6. Add noise
            noisy_actions = noise_scheduler.add_noise(clean_actions, noise, timesteps)  # type: ignore

            # 7. Concatenate
            net_input = torch.cat([noisy_actions, global_condition], dim=1)

            # 8. Predict the CLEAN actions (x_0), not the noise!
            action_pred = noise_pred_net(net_input, timesteps).sample

            # 9. Calculate MSE against the clean_actions!
            loss = loss_fn(action_pred, clean_actions)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            ema_scene_encoder.step(scene_encoder.parameters())
            ema_wrist_encoder.step(wrist_encoder.parameters())
            ema_noise_pred_net.step(noise_pred_net.parameters())

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches

        # --- NEW: Get end time and calculate duration ---
        end_time = time.time()
        end_clock = datetime.now().strftime("%H:%M:%S")
        duration = end_time - start_time

        print(
            f"[Epoch {epoch+1}/{num_epochs}] Finished at {end_clock} | Loss: {epoch_loss:.6f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | Took: {duration:.2f} seconds"
        )

        if (epoch + 1) % 25 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "scene_encoder": scene_encoder.state_dict(),
                "wrist_encoder": wrist_encoder.state_dict(),
                "noise_pred_net": noise_pred_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),  # if you added cosine
                "ema_scene_encoder": ema_scene_encoder.state_dict(),
                "ema_wrist_encoder": ema_wrist_encoder.state_dict(),
                "ema_noise_pred_net": ema_noise_pred_net.state_dict(),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_ep{epoch+1}.pth"))
            print(f"--> Saved full checkpoint to {ckpt_dir}")


if __name__ == "__main__":

    # enc = VisionEncoder(num_kp=64).cuda()
    # img = torch.randn(2, 3, 224, 224, device="cuda")
    # out = enc(img)
    # print(out.shape)  # should print torch.Size([2, 128])

    train_with_occlusions = False

    training_with = input("Train with obs? (y/n)")

    if training_with == "y" or training_with == "Y":
        training_with = "with_obstacles"

        train_with_occlusions_input = input("Train with occlusions? (y/n)")

        if train_with_occlusions_input == "y" or train_with_occlusions_input == "Y":
            train_with_occlusions = True

    else:
        training_with = "no_obstacles"

    train(training_with, train_with_occlusions)
