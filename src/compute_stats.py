import os
import h5py
import numpy as np

# Point this at your training data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

for training_with in ["with_obstacles", "no_obstacles"]:
    target_dir = os.path.join(DATA_DIR, training_with)
    if not os.path.isdir(target_dir):
        print(f"Skipping {training_with}: directory does not exist")
        continue

    ep_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".hdf5")])

    if not ep_files:
        print(f"Skipping {training_with}: no .hdf5 files found in {target_dir}")
        continue

    print(f"Scanning {len(ep_files)} episodes in {training_with}...")

    all_qpos = []
    all_actions = []

    for ep_file in ep_files:
        with h5py.File(os.path.join(target_dir, ep_file), "r") as f:
            all_qpos.append(np.array(f["observations/qpos"]))
            all_actions.append(np.array(f["actions"]))

    qpos_arr = np.concatenate(all_qpos, axis=0)  # [total_steps, 6]
    action_arr = np.concatenate(all_actions, axis=0)  # [total_steps, 6]

    # Per-dimension stats
    qpos_min = qpos_arr.min(axis=0)
    qpos_max = qpos_arr.max(axis=0)
    action_min = action_arr.min(axis=0)
    action_max = action_arr.max(axis=0)

    # Add a small margin so we don't clip at exactly ±1
    margin = 0.02
    qpos_range = qpos_max - qpos_min
    action_range = action_max - action_min
    qpos_min -= margin * qpos_range
    qpos_max += margin * qpos_range
    action_min -= margin * action_range
    action_max += margin * action_range

    out_path = os.path.join(target_dir, "norm_stats.npz")
    np.savez(
        out_path,
        qpos_min=qpos_min,
        qpos_max=qpos_max,
        action_min=action_min,
        action_max=action_max,
    )
    print(f"Saved stats to {out_path}")
    print(f"  qpos_min:   {qpos_min}")
    print(f"  qpos_max:   {qpos_max}")
    print(f"  action_min: {action_min}")
    print(f"  action_max: {action_max}")
