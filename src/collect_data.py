import mujoco
import numpy as np

import mujoco.viewer

from utils.utils import *

from utils.utils_RRT import PathPlanning

from scipy.spatial.transform import Rotation

import os

import cv2
import h5py

import time

training_with = input("Train with obs? (y/n)")

if training_with == "y" or training_with == "Y":
    training_with = "with_obstacles"
else:
    training_with = "no_obstacles"

print(f"Training: {training_with}...")

# directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if training_with == "no_obstacles":
    XML_filename = "main_scene.xml"
elif training_with == "with_obstacles":
    XML_filename = "main_scene_obs.xml"
else:
    print("incorrect training_with selection")
    while 1:
        pass

XML_PATH = PROJECT_ROOT + "/assets/" + XML_filename  # type: ignore

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_DIR = os.path.join(DATA_DIR, training_with)

os.makedirs(DATA_DIR, exist_ok=True)  # Creates the folder if it doesn't exist

# Load the UR5e model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

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

# IDs
greenzone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_zone")
greenzone_geom_id = model.geom("green_zone_cyl").id
GREENZONE_CYL_HALF_HEIGHT = model.geom_size[greenzone_geom_id][1]

mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
mocap_id = model.body_mocapid[mocap_body_id]  # Translates Body ID (23) to Mocap ID (0)

# arm parameters
arm_ndof = 6
gripper_ndof = 6

# initial position
# # used for IK initial position guess
arm_home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

# arm actual initial position
arm_init_qpos = np.array([0.314, -2.95, 1.35, -0.691, -1.45, 0])

# gripper open config
gripper_open_qpos = np.array([0, 0, 0, 0, 0, 0])

total_episodes = 200
start_episode = 0
episode_iter = start_episode


while episode_iter < total_episodes:

    # reset model
    mujoco.mj_resetData(model, data)

    # reset data storage buffers
    obs_scene_imgs = []
    obs_wrist_imgs = []
    obs_qpos = []
    obs_gripper_qpos = []
    actions = []

    # randomize greenzone position
    x = np.random.uniform(0.2, 0.5)
    y = np.random.uniform(0.4, 0.5)
    greenzone_cyl_init_pos = np.array([x, y, GREENZONE_CYL_HALF_HEIGHT])

    print(f"episode: {episode_iter}")
    print(f"target zone init pos: {greenzone_cyl_init_pos}")

    # initial world config
    init_qpos = np.concatenate((arm_init_qpos, gripper_open_qpos))

    # assign to initial position of the simulator
    data.mocap_pos[mocap_id] = greenzone_cyl_init_pos
    data.qpos = init_qpos
    mujoco.mj_forward(model, data)

    # initialize matrices
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    ## inverse kinematics for the ball position
    # goal position
    arm_goal_pos = greenzone_cyl_init_pos.copy()
    arm_goal_pos[2] += GREENZONE_CYL_HALF_HEIGHT + 0.10  # 10cm above the target zone

    # we create a cube of 10cm and accept anything within that sphere
    cube_side = 0.05
    greenzone_allowance_cube = np.random.uniform(-cube_side, cube_side, size=3)

    arm_goal_pos += greenzone_allowance_cube

    # ee orientation
    arm_goal_rot_wrt_global = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ]
    )

    # IK
    arm_goal_qpos = UR5eIK(
        model,
        data,
        "gripper_center",
        arm_home_qpos,
        arm_goal_pos,
        arm_goal_rot_wrt_global,
    )

    if arm_goal_qpos is None:
        print("IK failed!")
        continue

    # for RRT
    joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    goal_qpos = np.concatenate((arm_goal_qpos, gripper_open_qpos))

    # print(init_qpos)
    # print(goal_qpos)

    init_to_goal_traj = PathPlanning(model, data, joints, init_qpos, goal_qpos, 0.1)

    if init_to_goal_traj is None:
        print("no path returned!")
        continue

    # pad the trajectory and add 16 mmore instances of same config so that planner stops at end position
    pad_steps = 16
    np.pad(init_to_goal_traj, pad_width=((0, pad_steps), (0, 0)), mode="edge")

    # print(init_to_goal_traj)
    print(f"Path waypoints: {init_to_goal_traj.shape}")  # type: ignore
    # input()

    # print(init_to_goal_traj.shape)

    print(f"Starting simulation for episode: {episode_iter}!")
    t = 0
    step_counter = 0

    with mujoco.viewer.launch_passive(  # type: ignore
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.sync()

        for traj in init_to_goal_traj:  # type: ignore
            target_arm_qpos = traj[:arm_ndof]
            target_gripper_qpos = 0

            if step_counter % STEPS_PER_RECORD == 0:
                # Render Scene Camera
                renderer.update_scene(data, camera=SCENE_CAM_NAME)
                scene_img = renderer.render()  # Returns RGB numpy array

                # Render Gripper Camera
                renderer.update_scene(data, camera=WRIST_CAM_NAME)
                wrist_img = renderer.render()  # Returns RGB numpy array

                # Get physical states
                current_qpos = data.qpos[:arm_ndof].copy()

                # Save Action (what we commanded at this step)
                current_action = target_arm_qpos.copy()

                # Append to buffers
                obs_scene_imgs.append(scene_img)
                obs_wrist_imgs.append(wrist_img)
                obs_qpos.append(current_qpos)
                actions.append(current_action)

                # # live monitor
                # cv2.imshow("Scene Camera", cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR))
                # cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break

            if np.random.rand() < 0.10:
                # Create a small noise vector (e.g., +/- 0.02 radians)
                noise = np.random.uniform(-0.02, 0.02, size=arm_ndof)
                noisy_qpos = target_arm_qpos + noise

                # Send the noisy command to the physics engine
                data.ctrl[:arm_ndof] = noisy_qpos
            else:
                # Send the perfect command
                data.ctrl[:arm_ndof] = target_arm_qpos

            # single action for the gripper
            data.ctrl[arm_ndof:gripper_ndof] = target_gripper_qpos

            mujoco.mj_step(model, data)
            viewer.sync()

            t += TIMESTEP
            step_counter += 1
            # time.sleep(0.01)

        # Cleanup OpenCV windows
        cv2.destroyAllWindows()

        print("Simulation complete!")
        save_sim = input("Save simulation?")
        # input()

        if save_sim == "n":
            continue

        HDF5_NAME = "episode_" + str(episode_iter) + ".hdf5"
        HDF5_PATH = os.path.join(DATA_DIR, HDF5_NAME)

        with h5py.File(HDF5_PATH, "w") as f:
            # Create the observations group
            training_with_ds = f.create_dataset(
                "training_with",
                data=training_with,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")

            # Save Images (Using uint8 compression to save massive amounts of disk space)
            img_grp.create_dataset(
                "scene_cam",
                data=np.array(obs_scene_imgs),
                dtype="uint8",
                compression="gzip",
            )
            img_grp.create_dataset(
                "gripper_cam",
                data=np.array(obs_wrist_imgs),
                dtype="uint8",
                compression="gzip",
            )

            # Save States (float32 is standard for policy inputs)
            obs_grp.create_dataset("qpos", data=np.array(obs_qpos), dtype="float32")

            # Save Actions
            f.create_dataset("actions", data=np.array(actions), dtype="float32")

            # Save ball init position and quaternion
            f.create_dataset("greenzone_cyl_init_pos", data=greenzone_cyl_init_pos)

    episode_iter += 1

    viewer.close()

renderer.close()
