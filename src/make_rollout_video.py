"""
Render a 3-pane MP4 from a recorded rollout episode (.npz produced by rollout.py).

Layout (1280 x 720):

    +----------+--------------------------+
    | scene cam|                          |
    | (policy) |                          |
    +----------+   MuJoCo (3rd person)    |
    | wrist cam|                          |
    | (policy) |                          |
    +----------+--------------------------+

Playback runs at real wall-clock time (default 30 fps from a 100 Hz sim) so
the video shows the policy's behaviour as it would look on a real robot,
free of the live-planning latency.

Usage:
    python make_rollout_video.py path/to/video_data.npz
    python make_rollout_video.py path/to/video_data.npz --max-seconds 8 --out demo.mp4
"""

import os
import sys
import argparse
import numpy as np
import cv2
import mujoco


# ----------------------------------------------------------------------
# Layout / styling
# ----------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1280, 720
PAD = 20
LABEL_H = 28
CAM_SIZE = 300  # camera frames upscaled to this size for display

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
SIM_HZ = 100

# 3rd-person presentation camera (free camera; tweak to taste)
PRES_LOOKAT = (0.00, 0.40, 0.10)
PRES_DISTANCE = 1.6
PRES_AZIMUTH = 135.0
PRES_ELEVATION = -45.0


# ----------------------------------------------------------------------
def _put_label(canvas, text, x, y_baseline, color=LABEL_COLOR, scale=0.55, thick=1):
    cv2.putText(canvas, text, (x, y_baseline), FONT, scale, color, thick, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to recorded video data .npz")
    parser.add_argument("--out", default=None, help="Output .mp4 path")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Trim the video to the first N real seconds",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="Directory with main_scene.xml / main_scene_obs.xml "
        "(default: <project_root>/assets next to the npz)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load recording
    # ------------------------------------------------------------------
    rec = np.load(args.npz_path)
    qpos_seq = rec["qpos"]  # (T_sim, nq)
    scene_frames = rec["scene_frames"]  # (n_plan, 224, 224, 3) uint8 RGB
    wrist_frames = rec["wrist_frames"]  # (n_plan, 224, 224, 3) uint8 RGB
    plan_starts = rec["plan_starts"]  # (n_plan,) sim-step indices
    greenzone_pos = rec["greenzone_pos"]
    timestep = float(rec["timestep"])
    xml_scene = str(rec["xml_scene"])
    occ_flag = bool(int(rec["scene_with_occlusions"]))
    policy_dir = str(rec["policy_ckpt_direc"])

    n_sim_steps = qpos_seq.shape[0]
    if args.max_seconds is not None:
        n_sim_steps = min(n_sim_steps, int(round(args.max_seconds / timestep)))

    # ------------------------------------------------------------------
    # Resolve XML
    # ------------------------------------------------------------------
    if args.xml_dir is None:
        # rollout.py uses PROJECT_ROOT = parent-of-parent-of-script.
        # The npz lives in <PROJECT_ROOT>/video_data/, so PROJECT_ROOT = its parent.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.npz_path)))
        xml_dir = os.path.join(project_root, "assets")
    else:
        xml_dir = args.xml_dir

    xml_filename = (
        "main_scene_obs.xml" if xml_scene == "with_obstacles" else "main_scene.xml"
    )
    xml_path = os.path.join(xml_dir, xml_filename)
    print(f"Loading XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)

    # Bump the offscreen framebuffer to fit our 3rd-person pane.
    # Default offwidth/offheight in the XML is 640x480 which is too small.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), MJ_W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), MJ_H)

    data = mujoco.MjData(model)

    # Place greenzone via mocap (matches rollout setup)
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
    # Output writer
    # ------------------------------------------------------------------
    out_path = args.out or os.path.splitext(args.npz_path)[0] + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (CANVAS_W, CANVAS_H))

    sim_per_video = SIM_HZ / VIDEO_FPS
    n_video_frames = int(np.floor(n_sim_steps / sim_per_video))
    print(f"Rendering {n_video_frames} frames -> {out_path}")
    print(f"  sim steps   : {n_sim_steps}")
    print(f"  duration    : {n_video_frames / VIDEO_FPS:.2f} s @ {VIDEO_FPS} fps")
    print(f"  plan iters  : {len(plan_starts)}")

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------
    for vf in range(n_video_frames):
        sim_step = min(int(round(vf * sim_per_video)), n_sim_steps - 1)

        # Drive the sim from saved qpos -- no integration, just kinematics
        data.qpos[:] = qpos_seq[sim_step]
        mujoco.mj_forward(model, data)

        # 3rd-person view
        renderer.update_scene(data, camera=cam)
        mj_rgb = renderer.render()
        mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)

        # Find which planning iter is "current" at this sim step
        plan_idx = int(np.searchsorted(plan_starts, sim_step, side="right") - 1)
        plan_idx = max(0, min(plan_idx, len(scene_frames) - 1))

        scene_rgb = scene_frames[plan_idx]
        wrist_rgb = wrist_frames[plan_idx]

        scene_disp = cv2.cvtColor(
            cv2.resize(
                scene_rgb, (CAM_SIZE, CAM_SIZE), interpolation=cv2.INTER_LANCZOS4
            ),
            cv2.COLOR_RGB2BGR,
        )
        wrist_disp = cv2.cvtColor(
            cv2.resize(
                wrist_rgb, (CAM_SIZE, CAM_SIZE), interpolation=cv2.INTER_LANCZOS4
            ),
            cv2.COLOR_RGB2BGR,
        )

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
