import numpy as np
import mujoco
import mjpl


def PathPlanning(model, data, joints, init_qpos, goal_qpos, goal_biasing_probability):

    q_idx = mjpl.qpos_idx(model, joints)

    constraints = [
        mjpl.JointLimitConstraint(model),
        mjpl.CollisionConstraint(model),
    ]

    # Set up the planner.
    planner = mjpl.RRT(
        model,
        joints,
        constraints,
        seed=None,
        goal_biasing_probability=goal_biasing_probability,
    )

    data.qpos = init_qpos

    print("Planning...")

    try:
        waypoints = planner.plan_to_config(init_qpos, goal_qpos)
    except:
        print("Planning failed with errors")
        return None

    if not waypoints:
        print("Planning failed")
        return None

    print("Shortcutting...")
    shortcut_waypoints = mjpl.smooth_path(
        waypoints, constraints, eps=planner.epsilon, seed=None, sparse=True
    )

    dof = len(waypoints[0])
    traj_generator = mjpl.RuckigTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        max_jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    trajectory = mjpl.generate_constrained_trajectory(
        shortcut_waypoints, traj_generator, constraints
    )
    if trajectory is None:
        print("Trajectory generation failed.")
        return False

    print("RRT and trajectory generation complete!")

    # mujoco.mj_resetData(model, data)
    data.qpos = init_qpos

    return np.array(trajectory.positions)
