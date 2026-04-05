import mujoco
import numpy as np

import mujoco.viewer

from utils.utils import *

from scipy.spatial.transform import Rotation

import os

import cv2
import h5py

# directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = PROJECT_ROOT + "/assets/main_scene.xml"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
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

# constants
ball_geom_id = model.geom("ball_geom").id
BALL_RADIUS = model.geom_size[ball_geom_id][0]

driver_joint_id = model.joint("right_driver_joint").id
driver_qpos_idx = model.jnt_qposadr[driver_joint_id]

# arm parameters
arm_ndof = 6


# initial position
# # used for IK initial position guess
arm_home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

# arm actual initial position
arm_init_qpos = np.array([0.314, -2.95, 1.35, -0.691, -1.45, 0])
# np.array([0.7, -0.0, -2.43, -1.2, 1.5708, 3.1415])
# np.array([0.7, -np.pi / 4, -2.43, -0.34, np.pi / 2, np.pi])

# gripper open config
gripper_open_qpos = np.array([0, 0, 0, 0, 0, 0])

total_episodes = 30

for episode_iter in range(total_episodes):

    # reset model
    mujoco.mj_resetData(model, data)

    # data storage buffers
    obs_scene_imgs = []
    obs_wrist_imgs = []
    obs_qpos = []
    obs_gripper_qpos = []
    actions = []

    step_counter = 0

    # ball initial position
    x = np.random.uniform(0.2, 0.75)
    y = np.random.uniform(0.2, 0.5)
    ball_init_pos = np.array([x, y, BALL_RADIUS])

    # print(f"ball rad: {BALL_RADIUS}")
    print(f"episode: {episode_iter}")
    print(f"ball init pos: {ball_init_pos}")

    # ball initial orientation
    scipy_quat = Rotation.random().as_quat()
    mujoco_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

    ball_init_qpos = np.concatenate((ball_init_pos, mujoco_quat))

    # initial world config
    init_qpos = np.concatenate((arm_init_qpos, gripper_open_qpos, ball_init_qpos))

    # assign to initial position of the simulator
    data.qpos = init_qpos
    mujoco.mj_forward(model, data)

    # initialize matrices
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # ----------------------------------------
    ## inverse kinematics for the ball position

    # goal position
    arm_goal_pos = ball_init_pos.copy()
    arm_goal_rot_wrt_global = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ]
    )

    # waypoints/goals
    arm_goal_waypoint_pos = ball_init_pos.copy()
    arm_goal_waypoint_pos[2] += 0.1

    # print(ball_init_pos)
    # print(arm_goal_waypoint_pos)

    # drop site
    drop_site_name = "bin_top_center"
    drop_site_id = model.site(drop_site_name).id
    # no need for mj_forward since drop site does not move
    arm_drop_pos = data.site(drop_site_id).xpos.copy()  # orientation is the same

    arm_drop_waypoint_pos = arm_drop_pos.copy()
    arm_drop_waypoint_pos[2] += 0.1

    # print(f"arm_drop_pos: {arm_drop_pos}")
    # print(arm_drop_waypoint_pos)

    # Do IK and get trajectories
    arm_goal_waypoint_qpos = UR5eIK(
        model,
        data,
        "gripper_center",
        arm_home_qpos,
        arm_goal_waypoint_pos,
        arm_goal_rot_wrt_global,
    )

    # input("IK complete for goal waypoint")

    arm_goal_qpos = UR5eIK(
        model,
        data,
        "gripper_center",
        arm_goal_waypoint_qpos,
        arm_goal_pos,
        arm_goal_rot_wrt_global,
    )

    # input("IK complete for goal")

    arm_drop_waypoint_qpos = UR5eIK(
        model,
        data,
        "gripper_center",
        arm_goal_qpos,
        arm_drop_waypoint_pos,
        arm_goal_rot_wrt_global,
    )

    # input("IK complete for drop waypoint")

    arm_drop_qpos = UR5eIK(
        model,
        data,
        "gripper_center",
        arm_drop_waypoint_qpos,
        arm_drop_pos,
        arm_goal_rot_wrt_global,
    )

    # input("IK complete for drop")

    ## Do RRT to ensure no collisions
    # create trajectories
    # initial position to goal waypoint
    init_to_goal_waypoint_tree = RRT_planning(
        model, data, arm_init_qpos, arm_goal_waypoint_qpos, 20, 50000, 0.2
    )

    init_to_goal_waypoint_traj = get_linear_trajectory(
        init_to_goal_waypoint_tree, dt=TIMESTEP, v_max=0.25
    )

    # goal waypoint to goal
    goal_waypoint_to_goal_tree = RRT_planning(
        model, data, arm_goal_waypoint_qpos, arm_goal_qpos, 20, 50000, 0.2
    )

    goal_waypoint_to_goal_traj = get_linear_trajectory(
        goal_waypoint_to_goal_tree, dt=TIMESTEP, v_max=0.25
    )

    # goal to drop waypoint
    goal_to_drop_waypoint_tree = RRT_planning(
        model, data, arm_goal_qpos, arm_drop_waypoint_qpos, 20, 50000, 0.2
    )

    goal_to_drop_waypoint_traj = get_linear_trajectory(
        goal_to_drop_waypoint_tree, dt=TIMESTEP, v_max=0.25
    )

    # drop waypoint to drop
    drop_waypoint_to_drop_tree = RRT_planning(
        model, data, arm_drop_waypoint_qpos, arm_drop_qpos, 20, 50000, 0.2
    )

    drop_waypoint_to_drop_traj = get_linear_trajectory(
        drop_waypoint_to_drop_tree, dt=TIMESTEP, v_max=0.25
    )

    # drop to drop waypoint
    drop_to_drop_waypoint_tree = RRT_planning(
        model, data, arm_drop_qpos, arm_drop_waypoint_qpos, 20, 50000, 0.2
    )

    drop_to_drop_waypoint_traj = get_linear_trajectory(
        drop_to_drop_waypoint_tree, dt=TIMESTEP, v_max=0.25
    )

    # drop waypoint to initial position
    drop_waypoint_to_init_tree = RRT_planning(
        model, data, arm_drop_waypoint_qpos, arm_init_qpos, 20, 50000, 0.2
    )

    drop_waypoint_to_init_traj = get_linear_trajectory(
        drop_waypoint_to_init_tree, dt=TIMESTEP, v_max=0.25
    )

    ## create a single list of paths and gripper commands
    master_traj = []

    for q_traj in init_to_goal_waypoint_traj[1]:
        master_traj.append((q_traj, 0))

    last_q_traj = master_traj[-1][0]

    # wait for some time
    for t in np.arange(0, 1, TIMESTEP):
        master_traj.append((last_q_traj, 0))

    for q_traj in goal_waypoint_to_goal_traj[1]:
        master_traj.append((q_traj, 0))

    last_q_traj = master_traj[-1][0]

    # wait for some time
    for t in np.arange(0, 1, TIMESTEP):
        master_traj.append((last_q_traj, 0))

    # pinch for 2 seconds
    for t in np.arange(0, 2, TIMESTEP):
        master_traj.append((last_q_traj, 1))

    for q_traj in goal_to_drop_waypoint_traj[1]:
        master_traj.append((q_traj, 1))

    for q_traj in drop_waypoint_to_drop_traj[1]:
        master_traj.append((q_traj, 1))

    last_q_traj = master_traj[-1][0]

    # wait for some time
    for t in np.arange(0, 2, TIMESTEP):
        master_traj.append((last_q_traj, 1))

    # drop pose for 2 seconds
    for t in np.arange(0, 2, TIMESTEP):
        master_traj.append((last_q_traj, 0))

        # no need to go back to home
        # for q_traj in drop_to_drop_waypoint_traj[1]:
        #     master_traj.append((q_traj, 0))

        # for q_traj in drop_waypoint_to_init_traj[1]:
        #     master_traj.append((q_traj, 0))

        # print(master_traj)

        # launch viewer
        # with mujoco.viewer.launch_passive(
        #     model, data, show_left_ui=False, show_right_ui=False
        # ) as viewer:

        #     # Set camera angle
        #     viewer.cam.azimuth = 90  # Azimuth angle (degrees)
        #     viewer.cam.elevation = -30  # Elevation angle (degrees)
        #     viewer.cam.distance = 4  # Distance from the lookat point (meters)
        #     viewer.cam.lookat[:] = [
        #         0.9,
        #         0.9,
        #         0.25,
        #     ]  # Point the camera is looking at [x, y, z]

        # 1. Get the integer ID of your custom camera from the XML
        # (Replace "your_camera_name" with whatever you named it in main_scene.xml)
        # camera_id = model.camera("scene_camera").id

        # # 2. Tell the viewer to switch to "Fixed" mode
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # # 3. Lock the viewer to your specific camera ID
        # viewer.cam.fixedcamid = camera_id

        # viewer.full_screen = True
        # model.vis.quality.offsamples = 8

        # # start simulation

    print("Starting simulation!")
    t = 0

    for i in range(len(master_traj)):
        target_qpos = master_traj[i][0]
        target_gripper = master_traj[i][1]

        data.ctrl[:arm_ndof] = target_qpos
        gripper_cmd(model, data, target_gripper)

        mujoco.mj_step(model, data)
        t += TIMESTEP

        # viewer.sync()

        if step_counter % STEPS_PER_RECORD == 0:
            # Render Scene Camera
            renderer.update_scene(data, camera=SCENE_CAM_NAME)
            scene_img = renderer.render()  # Returns RGB numpy array

            # Render Gripper Camera
            renderer.update_scene(data, camera=WRIST_CAM_NAME)
            wrist_img = renderer.render()  # Returns RGB numpy array

            # Get physical states
            current_qpos = data.qpos[:arm_ndof].copy()

            raw_gripper_pos = data.qpos[driver_qpos_idx]
            normalized_gripper_pos = raw_gripper_pos / 0.9
            current_gripper_qpos = np.array([normalized_gripper_pos])

            # Save Action (what we commanded at this step)
            current_action = np.concatenate([target_qpos, [target_gripper]])

            # Append to buffers
            obs_scene_imgs.append(scene_img)
            obs_wrist_imgs.append(wrist_img)
            obs_qpos.append(current_qpos)
            obs_gripper_qpos.append(current_gripper_qpos)
            actions.append(current_action)

            # live monitor
            cv2.imshow("Scene Camera", cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        step_counter += 1

    # Cleanup OpenCV windows
    cv2.destroyAllWindows()

    print("Simulation complete!")
    # input()

    HDF5_NAME = "episode_" + str(episode_iter) + ".hdf5"
    HDF5_PATH = os.path.join(DATA_DIR, HDF5_NAME)

    with h5py.File(HDF5_PATH, "w") as f:
        # Create the observations group
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
        obs_grp.create_dataset(
            "gripper_qpos", data=np.array(obs_gripper_qpos), dtype="float32"
        )

        # Save Actions
        f.create_dataset("actions", data=np.array(actions), dtype="float32")


renderer.close()
