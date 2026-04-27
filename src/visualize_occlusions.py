import os
import matplotlib.pyplot as plt
from dataset import UR5eDiffusionDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data", "with_obstacles")

dataset = UR5eDiffusionDataset(
    data_dir=data_dir,
    chunk_size=32,
    num_episodes=None,
    train_with_occlusions=True,
    occlusion_prob=1.0,  # force it to ALWAYS apply so you can see the effect
)

# Grid: N samples x 2 frames (t-1, t)
N = 2
fig, axes = plt.subplots(N, 2, figsize=(6, 3 * N))
fig1, axes1 = plt.subplots(N, 2, figsize=(6, 3 * N))

for i in range(N):
    sample = dataset[i * 100]  # strided so we get different states
    scene = sample["condition"]["scene_cam"]  # [2, 3, 224, 224], values in [0, 1]
    wrist = sample["condition"]["wrist_cam"]  # [2, 3, 224, 224], values in [0, 1]

    for f in range(2):
        img = scene[f].permute(1, 2, 0).numpy()  # [H, W, C] for imshow
        axes[i, f].imshow(img)
        axes[i, f].set_title(f"sample {i}, frame t{'−1' if f == 0 else ''}")
        axes[i, f].axis("off")

    for f in range(2):
        img = wrist[f].permute(1, 2, 0).numpy()  # [H, W, C] for imshow
        axes1[i, f].imshow(img)
        axes1[i, f].set_title(f"sample {i}, frame t{'−1' if f == 0 else ''}")
        axes1[i, f].axis("off")


plt.tight_layout()
# plt.savefig("occlusion_preview.png", dpi=120)
plt.show()
# print("Saved occlusion_preview.png")
