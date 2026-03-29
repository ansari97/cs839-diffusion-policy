### functions ###

import numpy as np
import mujoco
from math import ceil, floor

from scipy.spatial.transform import Rotation

arm_ndof = 6


# for RRT
def RRT_planning(
    model, data, init_qpos, goal_qpos, assign_goal_iter, max_iter, epsilon
):
    """
    hello
    """

    goal_found = False

    data.qpos[:arm_ndof] = init_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    if is_robot_collision(model, data):
        print("ERROR: Start configuration is in collision!")
        return None

    # Check if goal is in collision (Optional but recommended)
    data.qpos[:arm_ndof] = goal_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    if is_robot_collision(model, data):
        print(
            "ERROR: Goal configuration is in collision! RRT might fail to reach exact target."
        )
        return None

    # initialize empty tree and define parameters
    tree = Tree(init_qpos)

    # iter variable
    i = 0
    for i in np.arange(max_iter):

        # print(f">> iter {i}:")

        # create random config (qpos)
        lower_limit = -np.pi
        upper_limit = np.pi
        rand_qpos = np.random.uniform(low=lower_limit, high=upper_limit, size=arm_ndof)

        # bias towards goal
        if i % assign_goal_iter == 0:
            rand_qpos = goal_qpos

        # asign to robot
        # data.qpos = rand_qpos

        # check distance from tree points
        near_qpos, near_qpos_idx = tree.get_near_qpos(rand_qpos)

        # print(data.qpos)

        # check collision for rand_qpos or path
        diff_vector = rand_qpos - near_qpos
        vector_norm = np.linalg.norm(diff_vector)
        if vector_norm < 1e-6:
            continue
        unit_vector = diff_vector / vector_norm

        # divide path into N segments
        N = ceil(vector_norm / epsilon)

        break_occurred = False
        for n in range(N):
            new_qpos = near_qpos + (n + 1) * epsilon * unit_vector

            # rand_qpos
            if n == N - 1:
                new_qpos = rand_qpos

            # do calculations
            data.qpos[:arm_ndof] = new_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_collision(model, data)

            # if collision occurs
            if is_robot_collision(model, data):
                if n == 0:
                    break_occurred = True
                    break
                else:
                    new_qpos = new_qpos - epsilon * unit_vector
                    break
            else:
                pass

        # if collision not detected at first new_qpos along rand_qpos (epsilon), append tree
        if not break_occurred:
            tree.append_tree(new_qpos, near_qpos_idx)

        if np.allclose(new_qpos, goal_qpos, atol=1e-5):
            goal_found = True
            # print("goal found!")
            break

    if goal_found:
        # print(f"RRT complete in {i} iterations!")
        return tree.get_final_path()
    else:
        # print(f"RRT failed!")
        return None

    # print(tree.nodes)
    # print(tree.parent_idx)


class Tree:
    def __init__(self, init_node):
        self.nodes = [init_node]
        self.parent_idx = [0]

    def append_tree(self, node, parent_idx):
        self.nodes.append(node)
        self.parent_idx.append(parent_idx)

    def get_near_qpos(self, check_node: np.ndarray):
        diff_vector = check_node - np.vstack(self.nodes)
        diff_norm = np.linalg.norm(diff_vector, axis=1)
        min_norm_idx = np.argmin(diff_norm)

        return self.nodes[min_norm_idx], min_norm_idx

    def get_new_qpos(self, check_node: np.ndarray, epsilon):
        near_qpos, near_qpos_idx = self.get_near_qpos(check_node)

        diff_vector = check_node - near_qpos
        unit_vector = diff_vector / np.linalg.norm(diff_vector)

    def get_final_path(self):
        # Start at the most recently added node (the goal)
        curr_idx = len(self.nodes) - 1
        path = [self.nodes[curr_idx]]

        # Backtrack until we reach the root (index 0)
        while curr_idx != 0:
            parent_idx = self.parent_idx[curr_idx]
            path.append(self.nodes[parent_idx])
            curr_idx = parent_idx

        # reverse the path
        return path[::-1]


def get_linear_trajectory(path, v_max, dt=0.001):
    t = 0
    path = np.array(path)
    num_nodes, num_joints = path.shape

    q_traj, dq_traj, ddq_traj = (
        [],
        [],
        [],
    )

    for p in range(num_nodes - 1):
        q_start = path[p]
        q_end = path[p + 1]

        del_t = np.max(np.abs(q_end - q_start)) / v_max
        num_steps = floor(del_t / dt) + 1

        dq = (q_end - q_start) / (num_steps * dt)
        ddq = np.zeros(num_joints)

        for step in range(num_steps):
            q = q_start + dq * (step * dt)
            q_traj.append(q)
            dq_traj.append(dq)
            ddq_traj.append(ddq)
            t += dt

    return t, np.array(q_traj), np.array(dq_traj), np.array(ddq_traj)


def UR5eIK(
    model, data, site_name, arm_init_qpos, arm_goal_pos, arm_goal_rot_wrt_global
):

    # initialize matrices
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    q_new = arm_init_qpos
    site_id = model.site(site_name).id

    # while loop
    error = np.zeros(6)

    IK_iter = 0

    K = np.diag([0.3, 0.3, 0.3, 0.2, 0.2, 0.2])  # scaling matrix

    while np.linalg.norm(error) > 1e-6 or IK_iter == 0:
        mujoco.mj_forward(model, data)

        # get current position and orientation
        site_pos = data.site(site_id).xpos
        site_rot = data.site(site_id).xmat.reshape(3, 3)

        goal_rot_wrt_site = site_rot.T @ arm_goal_rot_wrt_global

        # change rotation to angle axis
        goal_rot_wrt_site = Rotation.from_matrix(goal_rot_wrt_site)
        goal_rot_wrt_site_vec = goal_rot_wrt_site.as_rotvec()

        # get position error in global frame
        error_pos = arm_goal_pos - site_pos

        # get rotation error in global frame
        error_rot = site_rot @ goal_rot_wrt_site_vec

        # combine error
        error = np.concatenate((error_pos, error_rot))

        # get jacobian at the gripper
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        arm_jacp = jacp[:, :arm_ndof]
        arm_jacr = jacr[:, :arm_ndof]

        arm_jac = np.vstack((arm_jacp, arm_jacr))
        # print(arm_jac)

        # get joint error
        # do SVD
        U, sigma, Vh = np.linalg.svd(arm_jac)
        sigma_ = sigma.copy()
        for i in range(arm_ndof):
            if sigma_[i] < 1e-3:
                sigma_[i] = 0
            else:
                sigma_[i] = 1 / sigma[i]

        # print(sigma_)

        J_inv = Vh.T @ np.diag(sigma_) @ U.T

        del_q = J_inv @ K @ error
        q_new = q_new + del_q
        data.qpos[:arm_ndof] = q_new

        # print(site_pos)
        # print(np.linalg.norm(error))
        # print(q_new)
        # input()

        IK_iter += 1

        if IK_iter > 5000:
            print("ERROR: couldn't converge :(")
            return None

    return q_new


def is_robot_collision(model, data):
    """
    Checks if the robot has collided with the environment,
    ignoring harmless background contacts like the ball on the table.
    """
    # Quick exit if there are absolutely no contacts
    if data.ncon == 0:
        return False

    for n in range(data.ncon):
        contact = data.contact[n]

        # Get the body IDs involved in the contact
        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]

        # Convert IDs to string names
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

        # Define pairs of bodies that are ALLOWED to touch
        contact_pair = {name1, name2}

        if contact_pair == {"target_ball", "work_table"}:
            continue  # This is just the ball resting on the table; ignore it!

        # Optional: You can add more allowed pairs here if your gripper
        # has self-collisions you want the RRT to ignore.

        # If we reach this line, an illegal collision happened!
        # print(f"RRT Collision detected between: {name1} and {name2}")
        return True

    return False


def gripper_cmd(model, data, cmd):
    if cmd == 0:
        data.ctrl[6] = 0
    if cmd == 1:
        data.ctrl[6] = 100
