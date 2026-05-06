"""
Render a 3-pane MP4 from a recorded data-collection episode (.hdf5 produced by
collect_data.py).

Layout (1280 x 720):

    +----------+--------------------------+
    | scene cam|                          |
    +----------+   MuJoCo (3rd person)    |
    | wrist cam|                          |
    +----------+--------------------------+

The HDF5 stores camera frames + arm qpos at 10 Hz. The video plays back at
30 fps, with qpos linearly interpolated between samples so the MuJoCo pane
moves smoothly. Camera frames hold for 0.1 s each (their native capture rate),
which is honest to what was recorded.

Camera frames are 640 x 480 (4:3); they're letterboxed into the 300 x 300
camera panels so they don't get squished into squares. The panel geometry
matches make_rollout_video.py so demo and rollout videos look consistent
side-by-side on a slide.

Usage:
    python make_demo_video.py path/to/episode_42.hdf5
    python make_demo_video.py path/to/episode_42.hdf5 --max-seconds 6 --out demo.mp4
    python make_demo_video.py "C:\\path with spaces\\episode_42.hdf5"
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import mujoco

# ----------------------------------------------------------------------
# Layout / styling -- kept in lockstep with make_rollout_video.py
# ----------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1280, 720
PAD = 20
LABEL_H = 28
CAM_SIZE = 300

SCENE_X, SCENE_Y = PAD, PAD + LABEL_H
WRIST_X, WRIST_Y = PAD, SCENE_Y + CAM_SIZE + PAD + LABEL_H

MJ_X = CAM_SIZE + 2 * PAD
MJ_Y = PAD + LABEL_H
MJ_W = CANVAS_W - MJ_X - PAD
MJ_H = CANVAS_H - MJ_Y - PAD

BG_COLOR = (28, 28, 30)
LABEL_COLOR = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

VIDEO_FPS = 30
DATA_HZ = 10  # collect_data.py records at 10 Hz

# 3rd-person presentation camera (free camera; tweak to taste)
PRES_LOOKAT = (0.30, 0.30, 0.10)
PRES_DISTANCE = 1.6
PRES_AZIMUTH = 135.0
PRES_ELEVATION = -22.0


# ----------------------------------------------------------------------
def _put_label(canvas, text, x, y_baseline, color=LABEL_COLOR, scale=0.55, thick=1):
    cv2.putText(canvas, text, (x, y_baseline), FONT, scale, color, thick, cv2.LINE_AA)


def _letterbox(img_rgb, target_w, target_h, bg=BG_COLOR):
    """Resize img_rgb into a target_w x target_h panel preserving aspect.
    Returns a BGR image padded with `bg` on the unused sides.
    """
    h_in, w_in = img_rgb.shape[:2]
    in_aspect = w_in / h_in
    target_aspect = target_w / target_h
    if in_aspect > target_aspect:
        new_w = target_w
        new_h = int(round(target_w / in_aspect))
    else:
        new_h = target_h
        new_w = int(round(target_h * in_aspect))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = cv2.cvtColor(
        resized, cv2.COLOR_RGB2BGR
    )
    return canvas


def _read_str(dataset):
    """Robustly read an h5py scalar string dataset as a Python str."""
    val = dataset[()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        val = val.item()
        if isinstance(val, bytes):
            return val.decode("utf-8")
    return str(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", help="Path to a collect_data episode .hdf5")
    parser.add_argument("--out", default=None, help="Output .mp4 path")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Trim video to first N real seconds",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="Directory with main_scene.xml / main_scene_obs.xml "
        "(default: <project_root>/assets, inferred from hdf5 location)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load HDF5
    # ------------------------------------------------------------------
    with h5py.File(args.hdf5_path, "r") as f:
        scene_frames = f["observations/images/scene_cam"][:]  # (T, H, W, 3) uint8 RGB
        wrist_frames = f["observations/images/gripper_cam"][:]  # (T, H, W, 3) uint8 RGB
        qpos_seq = np.asarray(f["observations/qpos"][:], dtype=np.float64)  # (T, 6)
        greenzone_pos = np.asarray(
            f["greenzone_cyl_init_pos"][:], dtype=np.float64
        )  # (3,)
        training_with = _read_str(f["training_with"])

    T_data = qpos_seq.shape[0]
    duration_total = T_data / DATA_HZ
    print(
        f"Loaded HDF5: T_data = {T_data} samples @ {DATA_HZ} Hz "
        f"(duration {duration_total:.2f} s)"
    )
    print(f"  training_with = {training_with}")

    # ------------------------------------------------------------------
    # Resolve XML
    # ------------------------------------------------------------------
    if args.xml_dir is None:
        # collect_data layout: <PROJECT_ROOT>/data/<training_with>/<episode_X.hdf5>
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(args.hdf5_path)))
        )
        xml_dir = os.path.join(project_root, "assets")
    else:
        xml_dir = args.xml_dir

    xml_filename = (
        "main_scene_obs.xml" if training_with == "with_obstacles" else "main_scene.xml"
    )
    xml_path = os.path.join(xml_dir, xml_filename)
    print(f"Loading XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    # Bump the offscreen framebuffer to fit our 3rd-person pane.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), MJ_W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), MJ_H)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Place greenzone via mocap (matches collect_data setup)
    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
    mocap_id = model.body_mocapid[mocap_body_id]
    data.mocap_pos[mocap_id] = greenzone_pos

    # Free camera at presentation angle
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = PRES_LOOKAT
    cam.distance = PRES_DISTANCE
    cam.azimuth = PRES_AZIMUTH
    cam.elevation = PRES_ELEVATION

    renderer = mujoco.Renderer(model, height=MJ_H, width=MJ_W)

    # ------------------------------------------------------------------
    # Output writer + timing
    # ------------------------------------------------------------------
    out_path = args.out or os.path.splitext(args.hdf5_path)[0] + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (CANVAS_W, CANVAS_H))

    duration_s = (
        duration_total
        if args.max_seconds is None
        else min(duration_total, args.max_seconds)
    )
    n_video_frames = int(np.floor(duration_s * VIDEO_FPS))
    print(f"Rendering {n_video_frames} frames -> {out_path}")
    print(f"  duration: {n_video_frames / VIDEO_FPS:.2f} s @ {VIDEO_FPS} fps")

    # ------------------------------------------------------------------
    # Resample qpos from DATA_HZ to VIDEO_FPS via linear interpolation
    # ------------------------------------------------------------------
    arm_ndof = qpos_seq.shape[1]  # 6 for UR5e arm
    t_data_grid = np.arange(T_data) / DATA_HZ
    t_video_grid = np.arange(n_video_frames) / VIDEO_FPS
    qpos_interp = np.zeros((n_video_frames, arm_ndof))
    for j in range(arm_ndof):
        qpos_interp[:, j] = np.interp(t_video_grid, t_data_grid, qpos_seq[:, j])

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------
    for vf in range(n_video_frames):
        # Drive the sim from interpolated qpos -- no integration, just kinematics
        data.qpos[:arm_ndof] = qpos_interp[vf]
        mujoco.mj_forward(model, data)

        # 3rd-person view
        renderer.update_scene(data, camera=cam)
        mj_rgb = renderer.render()
        mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)

        # Step-function camera frames at DATA_HZ
        cam_idx = min(int(np.floor(t_video_grid[vf] * DATA_HZ)), T_data - 1)
        scene_disp = _letterbox(scene_frames[cam_idx], CAM_SIZE, CAM_SIZE)
        wrist_disp = _letterbox(wrist_frames[cam_idx], CAM_SIZE, CAM_SIZE)

        # Compose
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
        canvas[MJ_Y : MJ_Y + MJ_H, MJ_X : MJ_X + MJ_W] = mj_bgr
        canvas[SCENE_Y : SCENE_Y + CAM_SIZE, SCENE_X : SCENE_X + CAM_SIZE] = scene_disp
        canvas[WRIST_Y : WRIST_Y + CAM_SIZE, WRIST_X : WRIST_X + CAM_SIZE] = wrist_disp

        # Pane labels
        _put_label(canvas, "Scene camera", SCENE_X, SCENE_Y - 8)
        _put_label(canvas, "Wrist camera", WRIST_X, WRIST_Y - 8)
        _put_label(canvas, "MuJoCo viewer", MJ_X, MJ_Y - 8)

        writer.write(canvas)

    writer.release()
    renderer.close()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
