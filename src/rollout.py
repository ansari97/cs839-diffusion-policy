import mujoco
import numpy as np
from scipy.interpolate import interp1d

import mujoco.viewer

from utils.utils import *

from scipy.spatial.transform import Rotation

import os

import cv2
import h5py

import torch
import torch.nn as nn

from torchvision import transforms

from train import (
    VisionEncoder,
    action_chunk_size,
    vis_encoder_out,
    qpos_size,
    action_size,
)

from diffusers.models.unets.unet_1d import UNet1DModel

from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from dataset import UR5eDiffusionDataset

import random


def tensor_to_bgr(t):
    # t shape: [1, 3, 224, 224], values in [0, 1]
    img = t.squeeze(0).cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class NoOpViewer:
    def sync(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


ROLLOUT_LIST = [
    ["n", "n", "1"],
    ["y", "n", "1"],
    ["y", "y", "1"],
    ["n", "n", "2"],
    ["y", "n", "2"],
    ["y", "y", "2"],
    ["n", "n", "3"],
    ["y", "n", "3"],
    ["y", "y", "3"],
]

ENABLE_VIEWER = False

for xml_scene, scene_with_occlusions_input, training_input in ROLLOUT_LIST:
    np.random.seed(42)

    rollout_episodes = 100

    occlusion_prob = 0.3
    scene_with_occlusions = False

    # directories
    # xml_scene = input("XML Scene with obs? (y/n)")

    if xml_scene == "y" or xml_scene == "Y":
        xml_scene = "with_obstacles"
    else:
        xml_scene = "no_obstacles"

    # scene_with_occlusions_input = input("Scene with occlusions? (y/n)")

    if scene_with_occlusions_input == "y" or scene_with_occlusions_input == "Y":
        scene_with_occlusions = True

    # training_input = input("Trained policy? (1 for P1, 2 for P2, and 3 for P3)")

    if training_input == "1":
        policy_ckpt_direc = "no_obstacles"
    elif training_input == "2":
        policy_ckpt_direc = "with_obstacles"
    elif training_input == "3":
        policy_ckpt_direc = "occlusions"
    else:
        raise SystemExit("Invalid input for trained policy")

    print(
        f"*** Test scene: {xml_scene} | Scene occlusions: {scene_with_occlusions} | Policy: {policy_ckpt_direc}"
    )

    # directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if xml_scene == "no_obstacles":
        XML_filename = "main_scene.xml"
    elif xml_scene == "with_obstacles":
        XML_filename = "main_scene_obs.xml"
    else:
        raise SystemExit("Invalid xml_scene")

    XML_PATH = PROJECT_ROOT + "/assets/" + XML_filename  # type: ignore

    # print(f"XML path: {XML_PATH}")

    if policy_ckpt_direc == "no_obstacles":
        policy_data_dir = os.path.join(PROJECT_ROOT, "data", "no_obstacles")
    else:  # "with_obstacles" or "occlusions"
        policy_data_dir = os.path.join(PROJECT_ROOT, "data", "with_obstacles")

    # print(f"data direc: {policy_data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load normalization stats (same file dataset.py uses during training)
    stats = np.load(os.path.join(policy_data_dir, "norm_stats.npz"))
    qpos_min_t = torch.from_numpy(stats["qpos_min"]).float().to(device)  # type: ignore
    qpos_max_t = torch.from_numpy(stats["qpos_max"]).float().to(device)  # type: ignore
    action_min = stats["action_min"]  # keep numpy for denormalization
    action_max = stats["action_max"]

    # Load the UR5e model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    viewer_cm = (
        mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        if ENABLE_VIEWER
        else NoOpViewer()
    )

    # set timestep
    model.opt.timestep = 0.01
    TIMESTEP = model.opt.timestep

    # renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Data save frequency
    SIM_HZ = int(1.0 / TIMESTEP)  # Should be 100
    DATA_HZ = 10
    STEPS_PER_RECORD = SIM_HZ // DATA_HZ  # Every 10 steps

    # Camera setup
    SCENE_CAM_NAME = "scene_camera"
    WRIST_CAM_NAME = "wrist_camera"

    # constants
    greenzone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_zone")
    greenzone_geom_id = model.geom("green_zone_cyl").id
    GREENZONE_CYL_HALF_HEIGHT = model.geom_size[greenzone_geom_id][1]

    site_id = model.site("gripper_center").id

    site_name = "success_zone"
    success_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
    mocap_id = model.body_mocapid[
        mocap_body_id
    ]  # Translates Body ID (23) to Mocap ID (0)

    # arm parameters
    arm_ndof = 6
    gripper_ndof = 6

    # reset model
    mujoco.mj_resetData(model, data)

    # initial position
    # # used for IK initial position guess
    arm_home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

    # arm actual initial position
    arm_init_qpos = np.array([0.314, -2.95, 1.35, -0.691, -1.45, 0])
    # np.array([0.7, -0.0, -2.43, -1.2, 1.5708, 3.1415])
    # np.array([0.7, -np.pi / 4, -2.43, -0.34, np.pi / 2, np.pi])

    # gripper open config
    gripper_open_qpos = np.array([0, 0, 0, 0, 0, 0])

    # load greenzone initial pos from hdf file
    # with h5py.File(ep_path, "r") as f:
    #     greenzone_cyl_init_pos = f["greenzone_cyl_init_pos"][:]
    #     greenzone_cyl_init_pos = np.array(greenzone_cyl_init_pos)
    # print(f"Loaded exact zone position from training data: {greenzone_cyl_init_pos[:3]}")

    primary_task_success_array = np.zeros(rollout_episodes)
    secondary_task_success_array = np.zeros_like(primary_task_success_array)
    tertiary_task_success_array = np.zeros_like(primary_task_success_array)

    time_to_completion_array = np.zeros_like(primary_task_success_array)

    # print("Loading Neural Networks...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(current_dir, "..", "checkpoints")
    ckpt_dir = os.path.join(ckpt_dir, policy_ckpt_direc)

    # print(f"ckpt direc: {ckpt_dir}")

    # Auto-find latest unified checkpoint
    ckpt_files = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("ckpt_ep") and f.endswith(".pth")
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    epochs_found = [
        int(f.replace("ckpt_ep", "").replace(".pth", "")) for f in ckpt_files
    ]
    highest_epoch = max(epochs_found)
    # print(f"Loading EMA weights from epoch {highest_epoch}...")

    ckpt = torch.load(
        os.path.join(ckpt_dir, f"ckpt_ep{highest_epoch}.pth"),
        map_location=device,
        weights_only=False,
    )

    # Build fresh model instances
    scene_encoder = VisionEncoder().to(device)
    wrist_encoder = VisionEncoder().to(device)
    noise_pred_net = UNet1DModel(
        sample_size=action_chunk_size,
        in_channels=vis_encoder_out * 4 + qpos_size * 2 + action_size,
        out_channels=action_size,
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
        block_out_channels=(256, 512),
    ).to(
        device
    )  # type: ignore

    # Apply EMA weights via EMAModel helper + copy_to
    ema_scene = EMAModel(parameters=scene_encoder.parameters(), power=0.75)
    ema_scene.load_state_dict(ckpt["ema_scene_encoder"])
    ema_scene.copy_to(scene_encoder.parameters())

    ema_wrist = EMAModel(parameters=wrist_encoder.parameters(), power=0.75)
    ema_wrist.load_state_dict(ckpt["ema_wrist_encoder"])
    ema_wrist.copy_to(wrist_encoder.parameters())

    ema_unet = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
    ema_unet.load_state_dict(ckpt["ema_noise_pred_net"])
    ema_unet.copy_to(noise_pred_net.parameters())

    raw_weight = ckpt["scene_encoder"]["backbone.0.weight"][0, 0, 0, 0].item()
    active_weight = scene_encoder.backbone[0].weight[0, 0, 0, 0].item()
    # print(f"Raw: {raw_weight:.6f} | EMA-loaded: {active_weight:.6f}")

    scene_encoder.eval()
    wrist_encoder.eval()
    noise_pred_net.eval()

    # 3. Setup Scheduler and Resize Tool
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="sample",
    )
    num_inference_steps = 16
    noise_scheduler.set_timesteps(num_inference_steps)

    resize_transform = transforms.Resize((224, 224), antialias=True)

    task_success_time_criterion = 2  # seconds

    for ep in range(rollout_episodes):

        # Init at episode start
        min_xy_dist = float("inf")

        # reset model
        mujoco.mj_resetData(model, data)

        # randomize greenzone position
        x = np.random.uniform(0.2, 0.5)
        y = np.random.uniform(0.4, 0.5)
        greenzone_cyl_init_pos = np.array([x, y, GREENZONE_CYL_HALF_HEIGHT])

        arm_goal_pos = greenzone_cyl_init_pos.copy()
        arm_goal_pos[2] += (
            GREENZONE_CYL_HALF_HEIGHT + 0.10
        )  # 10cm above the target zone

        # initial world config
        init_qpos = np.concatenate((arm_init_qpos, gripper_open_qpos))

        # we create a cube of 10cm and accept anything within that sphere
        cube_half_side = 5e-2

        # assign to initial position of the simulator
        # model.body_pos[greenzone_body_id] = greenzone_cyl_init_pos
        # model.site_pos[success_site_id] = arm_goal_pos
        # model.site_size[success_site_id] = [cube_half_side, cube_half_side, cube_half_side]
        data.mocap_pos[mocap_id] = greenzone_cyl_init_pos
        data.qpos = init_qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_forward(model, data)

        # data storage buffers
        obs_scene_imgs = []
        obs_wrist_imgs = []
        obs_qpos = []
        obs_gripper_qpos = []
        actions = []

        step_counter = 0

        # initialize matrices
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        print(f"Starting episode {ep}!")
        t = 0

        MAX_TIME = 20  # seconds

        prev_scene_tensor = None
        prev_wrist_tensor = None
        prev_qpos_tensor = None

        # greenzone_allowance_cube = np.random.uniform(0, cube_half_side, size=3)

        # arm_goal_pos += cube_half_side

        current_ee_pos = data.site(site_id).xpos.copy()
        primary_task_successful = False
        secondary_task_successful = False
        tertiary_task_successful = False
        task_success_time = 0

        with viewer_cm as viewer:
            viewer.sync()
            while t < MAX_TIME and not tertiary_task_successful:
                # loop until max time reached; improve later by looking at goal reached or failure or max steps

                # get current robot state 7 element
                arm_qpos = data.qpos[:arm_ndof].copy()
                arm_qpos = torch.from_numpy(arm_qpos).float().to(device)
                arm_qpos = 2 * (arm_qpos - qpos_min_t) / (qpos_max_t - qpos_min_t) - 1

                # get current camera images; at what freq?
                # Render Scene Camera
                renderer.update_scene(data, camera=SCENE_CAM_NAME)
                scene_img = renderer.render()  # Returns RGB numpy array

                # Render Gripper Camera
                renderer.update_scene(data, camera=WRIST_CAM_NAME)
                wrist_img = renderer.render()  # Returns RGB numpy array

                robot_qpos = arm_qpos.unsqueeze(
                    0
                )  # already a normalized tensor on device

                # transform images by resizing, passing through visual encoder, normalizing channels
                scene_tensor = (
                    torch.from_numpy(np.moveaxis(scene_img, -1, 0)).float() / 255.0
                )
                wrist_tensor = (
                    torch.from_numpy(np.moveaxis(wrist_img, -1, 0)).float() / 255.0
                )

                scene_tensor = scene_tensor.unsqueeze(0).to(device)
                wrist_tensor = wrist_tensor.unsqueeze(0).to(device)

                scene_tensor = resize_transform(scene_tensor)
                wrist_tensor = resize_transform(wrist_tensor)

                if prev_scene_tensor is None:
                    prev_scene_tensor = scene_tensor.clone()
                    prev_wrist_tensor = wrist_tensor.clone()
                    prev_qpos_tensor = robot_qpos.clone()

                if scene_with_occlusions and random.random() < occlusion_prob:  # type: ignore
                    scene_stacked_tensor = torch.cat(
                        [prev_scene_tensor, scene_tensor], dim=0
                    )
                    scene_tensor = UR5eDiffusionDataset.pixel_blackening(
                        scene_stacked_tensor
                    )
                    prev_scene_tensor, scene_tensor = (
                        scene_stacked_tensor[0:1],
                        scene_stacked_tensor[1:2],
                    )

                # pass scene and wrist tensors through the CNN
                with torch.no_grad():
                    scene_feat_0 = scene_encoder(prev_scene_tensor)
                    scene_feat_1 = scene_encoder(scene_tensor)
                    wrist_feat_0 = wrist_encoder(prev_wrist_tensor)
                    wrist_feat_1 = wrist_encoder(wrist_tensor)

                # concatenate to form global condition
                global_condition = torch.cat(
                    [
                        scene_feat_0,
                        scene_feat_1,
                        wrist_feat_0,
                        wrist_feat_1,
                        prev_qpos_tensor,  # type: ignore
                        robot_qpos,
                    ],
                    dim=1,
                )  # type: ignore
                # broadcast condition for action_chunk_size steps ahead
                global_condition = global_condition.unsqueeze(-1).repeat(
                    1, 1, action_chunk_size
                )

                # feed into nn
                # Start with pure random noise -> [1, 7, action_chunk_size]
                noisy_action = torch.randn(
                    (1, action_size, action_chunk_size), device=device
                )

                # The Denoising Loop

                for k in noise_scheduler.timesteps:
                    # 1. Glue noise and condition together
                    net_input = torch.cat([noisy_action, global_condition], dim=1)

                    with torch.no_grad():
                        # 2. Ask U-Net to guess the noise
                        predicted_clean_action = noise_pred_net(net_input, k).sample

                    # 3. Mathematically subtract a tiny bit of that noise
                    noisy_action = noise_scheduler.step(
                        model_output=predicted_clean_action, timestep=k, sample=noisy_action  # type: ignore
                    ).prev_sample  # type: ignore

                # At the end of the loop, noisy_action is perfectly clean!
                # Pull it off the GPU and flip the sequence back to [32, 7]
                clean_actions = noisy_action.squeeze(0).cpu().numpy()
                clean_actions = clean_actions.transpose(1, 0)

                # ACTION CHUNKING: We generated 32 steps, but let's only execute the first 8
                # before we stop and take a new picture!
                action_horizon_size = 8
                action_chunk = clean_actions[:action_horizon_size]
                # Denormalize from [-1, 1] back to real joint angles
                action_chunk = (action_chunk + 1) / 2 * (
                    action_max - action_min
                ) + action_min

                # current_actual_qpos = data.qpos[:arm_ndof].copy()
                # action_chunk[0] = current_actual_qpos

                time_chunk = np.linspace(
                    0, (action_horizon_size - 1) / DATA_HZ, action_horizon_size
                )

                dense_time_chunk = np.linspace(
                    0,
                    action_horizon_size / DATA_HZ - 1 / SIM_HZ,
                    action_horizon_size * DATA_HZ,
                )
                # print(action_chunk)
                # print(time_chunk)
                # input()
                # print(dense_time_chunk)
                # input()

                # 2. Build the interpolation function
                # axis=0 means we interpolate along the time dimension
                # kind='linear' draws straight lines between your 8 waypoints
                interpolator = interp1d(
                    time_chunk,
                    action_chunk,
                    axis=0,
                    kind="linear",
                    fill_value="extrapolate",  # type: ignore
                )
                dense_action_chunk = interpolator(dense_time_chunk)

                # print(dense_action_chunk)
                # print(dense_time_chunk)
                # input()

                for i, action in enumerate(dense_action_chunk):
                    # loop for next 16 timesteps
                    # print(f"time: {t}")
                    # print(f"ctrl: {data.ctrl}")
                    data.ctrl[:arm_ndof] = action
                    data.ctrl[arm_ndof:gripper_ndof] = 0

                    if i == len(dense_action_chunk) - STEPS_PER_RECORD:  # time 0.7

                        # Render Scene Camera
                        renderer.update_scene(data, camera=SCENE_CAM_NAME)
                        scene_img = renderer.render()  # Returns RGB numpy array

                        # Render Gripper Camera
                        renderer.update_scene(data, camera=WRIST_CAM_NAME)
                        wrist_img = renderer.render()  # Returns RGB numpy array

                        scene_tensor = (
                            torch.from_numpy(np.moveaxis(scene_img, -1, 0)).float()
                            / 255.0
                        )
                        wrist_tensor = (
                            torch.from_numpy(np.moveaxis(wrist_img, -1, 0)).float()
                            / 255.0
                        )

                        scene_tensor = scene_tensor.unsqueeze(0).to(device)
                        wrist_tensor = wrist_tensor.unsqueeze(0).to(device)

                        prev_scene_tensor = resize_transform(scene_tensor)
                        prev_wrist_tensor = resize_transform(wrist_tensor)

                        arm_qpos = data.qpos[:arm_ndof].copy()
                        arm_qpos = torch.from_numpy(arm_qpos).float().to(device)
                        arm_qpos = (
                            2 * (arm_qpos - qpos_min_t) / (qpos_max_t - qpos_min_t) - 1
                        )
                        prev_qpos_tensor = arm_qpos.unsqueeze(0)

                    current_ee_pos = data.site(site_id).xpos.copy()
                    # print(f"current ee x, y: {current_ee_pos[:2]}")
                    # print(f"goal x, y: {arm_goal_pos[:2]}")

                    xy_dist = np.linalg.norm(current_ee_pos[:2] - arm_goal_pos[:2])
                    min_xy_dist = min(min_xy_dist, xy_dist)

                    if any(abs(current_ee_pos[:2] - arm_goal_pos[:2]) > cube_half_side):
                        task_success_time = 0
                    else:
                        task_success_time += TIMESTEP

                    # print(task_success_time)

                    in_xy = all(
                        abs(current_ee_pos[:2] - arm_goal_pos[:2]) <= cube_half_side
                    )
                    in_3d = all(abs(current_ee_pos - arm_goal_pos) <= cube_half_side)

                    if in_xy:
                        primary_task_successful = True

                    if task_success_time >= 1:
                        secondary_task_successful = True

                    if in_3d:
                        tertiary_task_successful = True

                    # tertiary_task_successful =

                    mujoco.mj_step(model, data)
                    t += TIMESTEP

                    viewer.sync()

                    # live monitor
                    # cv2.imshow(
                    #     "Scene (policy view, prev)", tensor_to_bgr(prev_scene_tensor)
                    # )
                    # cv2.imshow("Scene (policy view, curr)", tensor_to_bgr(scene_tensor))
                    # cv2.imshow("Wrist (policy view, curr)", tensor_to_bgr(wrist_tensor))
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break
                    # input()

        # print(
        #     f"Episode {ep}: min XY dist = {min_xy_dist*100:.1f} cm | final XY dist = {xy_dist*100:.1f} cm | success = {primary_task_successful}"
        # )

        primary_task_success_array[ep] = primary_task_successful
        secondary_task_success_array[ep] = secondary_task_successful
        tertiary_task_success_array[ep] = tertiary_task_successful

        # if primary_task_successful:
        #     time_to_completion_array[ep] = t

        print(f"Episode: {ep}; Primary task success: {primary_task_successful}")
        print(f"Episode: {ep}; Secondary task success: {secondary_task_successful}")
        print(f"Episode: {ep}; Tertiary task success: {tertiary_task_successful}")
        print(f"Episode: {ep}; Time to completion: {t}")
        # viewer.close()

    renderer.close()

    print(
        f"primary task success rate: {np.sum(primary_task_success_array)}/{len(primary_task_success_array)} | {np.sum(primary_task_success_array)/len(primary_task_success_array)}"
    )

    print(
        f"secondary task success rate: {np.sum(secondary_task_success_array)}/{len(secondary_task_success_array)} | {np.sum(secondary_task_success_array)/len(secondary_task_success_array)}"
    )

    print(
        f"Tertiary task success rate: {np.sum(tertiary_task_success_array)}/{len(tertiary_task_success_array)} | {np.sum(tertiary_task_success_array)/len(tertiary_task_success_array)}"
    )

    # print(time_to_completion_array)

    # print(
    #     f"Average time to completion: {np.sum(time_to_completion_array)/np.sum(primary_task_success_array)}"
    # )

    ## saving

    from scipy.io import savemat
    from datetime import datetime

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_tag = (
        f"scene-{xml_scene}"
        f"_testOcc-{int(scene_with_occlusions)}"
        f"_policy-{policy_ckpt_direc}"
        f"_ep{highest_epoch}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    savemat(
        os.path.join(results_dir, f"{run_tag}.mat"),
        {
            "primary_success": primary_task_success_array,
            "secondary_success": secondary_task_success_array,
            "tertiary_success": tertiary_task_success_array,
            "time_to_completion": time_to_completion_array,
            "rollout_episodes": rollout_episodes,
            "checkpoint_epoch": highest_epoch,
            "cube_half_side": cube_half_side,
            "task_success_time_crit": task_success_time_criterion,
            "max_time": MAX_TIME,
            "trained_with_obstacles": policy_ckpt_direc,
            "tested_with_occlusions": int(scene_with_occlusions),
            "policy": training_input,
            "occlusion_prob": occlusion_prob,
        },
    )

    print(f"Saved results to {run_tag}.mat")
