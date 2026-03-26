import mujoco
import numpy as np

import mujoco.viewer

from scipy.spatial.transform import Rotation as R

import os

# Load the UR5e model
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = dir_path + "/assets/main_scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# set timestep
model.opt.timestep = 0.01
TIMESTEP = model.opt.timestep

# constants
BALL_RADIUS = 0.04

# initial position
manip_init_qpos = np.array([0.7, -np.pi / 4, -2.43, -0.346, np.pi / 2, np.pi])
gripper_init_qpos = np.array([0, 0, 0, 0, 0, 0])

x = np.random.uniform(0.5, 0.8)
y = np.random.uniform(0.5, 0.8)
ball_init_pos = np.array([x, y, BALL_RADIUS])

scipy_quat = R.random().as_quat()
mujoco_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])

ball_init_qpos = np.concatenate((ball_init_pos, mujoco_quat))

data.qpos = np.concatenate((manip_init_qpos, gripper_init_qpos, ball_init_qpos))


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

    viewer.sync()
    input()
