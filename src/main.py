import mujoco
import numpy as np

import mujoco.viewer

from utils.utils import RRT_planning, Tree, get_linear_trajectory, UR5eIK

from scipy.spatial.transform import Rotation

import os

# Load the UR5e model
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = dir_path + "/assets/main_scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

# set timestep
model.opt.timestep = 0.01
TIMESTEP = model.opt.timestep

# constants
BALL_RADIUS = 0.04

# arm parameters
arm_ndof = 6

# initial position
arm_init_qpos = np.array([0.7, -np.pi / 4, -2.43, -0.346, np.pi / 2, np.pi])
gripper_init_qpos = np.array([0, 0, 0, 0, 0, 0])

x = np.random.uniform(0.2, 0.5)
y = np.random.uniform(0.2, 0.5)
ball_init_pos = np.array([x, y, BALL_RADIUS])

scipy_quat = Rotation.random().as_quat()
mujoco_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

ball_init_qpos = np.concatenate((ball_init_pos, mujoco_quat))

data.qpos = np.concatenate((arm_init_qpos, gripper_init_qpos, ball_init_qpos))

# initialize matrices
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# ----------------------------------------
# inverse kinematics for the ball position
arm_goal_pos = ball_init_pos
arm_goal_rot_wrt_global = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)

arm_goal_qpos = UR5eIK(
    model, data, "gripper_center", arm_init_qpos, arm_goal_pos, arm_goal_rot_wrt_global
)

input()

drop_site_name = "bin_top_center"
drop_site_id = model.site(drop_site_name).id
arm_drop_pos = site_pos = data.site(drop_site_id).xpos

arm_drop_qpos = UR5eIK(
    model, data, "gripper_center", arm_goal_qpos, arm_drop_pos, arm_goal_rot_wrt_global
)

# Do RRT to ensure no collisions
# get path using RRT; check dimensions of data inside the function
# tree_path = RRT_planning(model, data, arm_init_qpos, arm_goal_qpos, 20, 50000, 0.01)


with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:

    # Set camera angle
    viewer.cam.azimuth = 90  # Azimuth angle (degrees)
    viewer.cam.elevation = -30  # Elevation angle (degrees)
    viewer.cam.distance = 4  # Distance from the lookat point (meters)
    viewer.cam.lookat[:] = [0.9, 0.9, 0.25]  # Point the camera is looking at [x, y, z]

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
