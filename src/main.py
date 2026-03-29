import mujoco
import numpy as np

import mujoco.viewer

from utils.utils import *

from scipy.spatial.transform import Rotation

import os

# Load the UR5e model
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = dir_path + "/assets/main_scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

# set timestep
model.opt.timestep = 0.01
TIMESTEP = model.opt.timestep

# Camera setup
SCENE_CAM_NAME = "scene_camera"
WRIST_CAM_NAME = "gripper_camera"

# constants
ball_geom_id = model.geom("ball_geom").id
BALL_RADIUS = model.geom_size[ball_geom_id][0]

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

# ball initial position
x = np.random.uniform(0.3, 0.8)
y = np.random.uniform(0.1, 0.3)
ball_init_pos = np.array([x, y, BALL_RADIUS])

print(f"ball rad: {BALL_RADIUS}")
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

print(f"arm_drop_pos: {arm_drop_pos}")
print(arm_drop_waypoint_pos)


# Do IK and get trajectories
arm_goal_waypoint_qpos = UR5eIK(
    model,
    data,
    "gripper_center",
    arm_home_qpos,
    arm_goal_waypoint_pos,
    arm_goal_rot_wrt_global,
)

input("IK complete for goal waypoint")

arm_goal_qpos = UR5eIK(
    model,
    data,
    "gripper_center",
    arm_goal_waypoint_qpos,
    arm_goal_pos,
    arm_goal_rot_wrt_global,
)

input("IK complete for goal")

arm_drop_waypoint_qpos = UR5eIK(
    model,
    data,
    "gripper_center",
    arm_goal_qpos,
    arm_drop_waypoint_pos,
    arm_goal_rot_wrt_global,
)

input("IK complete for drop waypoint")

arm_drop_qpos = UR5eIK(
    model,
    data,
    "gripper_center",
    arm_drop_waypoint_qpos,
    arm_drop_pos,
    arm_goal_rot_wrt_global,
)

input("IK complete for drop")

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

for q_traj in drop_to_drop_waypoint_traj[1]:
    master_traj.append((q_traj, 0))

for q_traj in drop_waypoint_to_init_traj[1]:
    master_traj.append((q_traj, 0))


print(master_traj)

# launch viewer
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:

    # Set camera angle
    # viewer.cam.azimuth = 90  # Azimuth angle (degrees)
    # viewer.cam.elevation = -30  # Elevation angle (degrees)
    # viewer.cam.distance = 4  # Distance from the lookat point (meters)
    # viewer.cam.lookat[:] = [0.9, 0.9, 0.25]  # Point the camera is looking at [x, y, z]

    # viewer.full_screen = True
    # model.vis.quality.offsamples = 8

    # 1. Get the integer ID of your custom camera from the XML
    # (Replace "your_camera_name" with whatever you named it in main_scene.xml)
    camera_id = model.camera("scene_camera").id

    # 2. Tell the viewer to switch to "Fixed" mode
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # 3. Lock the viewer to your specific camera ID
    viewer.cam.fixedcamid = camera_id

    viewer.full_screen = True
    model.vis.quality.offsamples = 8

    data.qpos[:6] = arm_init_qpos
    mujoco.mj_forward(model, data)

    viewer.sync()
    input()

    data.qpos[:6] = arm_goal_qpos
    mujoco.mj_forward(model, data)

    viewer.sync()
    input()

    data.qpos[:6] = arm_drop_qpos
    mujoco.mj_forward(model, data)

    viewer.sync()
    input()

    # print(q_traj_goal.shape[0])

    input()

    # start simulation

    print("Starting simulation!")
    t = 0

    for i in range(len(master_traj)):
        data.ctrl[:arm_ndof] = master_traj[i][0]
        gripper_cmd(model, data, master_traj[i][1])
        mujoco.mj_step(model, data)
        t += TIMESTEP
        viewer.sync()

    print("Simulation complete!")
    input()
