import numpy as np

import rr_control as rr
from IK_utils import urInvKin
from control import urFwdKin
from rr_control import check_joint_limits, check_table_clearance, interp_cartesian_segment
from ur_interface import UrInterface


# Default tool0 -> pen-tip transform (update if you measure a different offset on hardware).
DEFAULT_T_TOOL0_TIP = np.eye(4)
DEFAULT_T_TOOL0_TIP[2, 3] = 0.15  # meters along +Z of tool0
TIP_FRAME_NAME = "pen_tip"


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def resolve_tool0_to_tip_transform(ur: UrInterface, q_ref: np.ndarray) -> np.ndarray:
    """Try to read tool0->tip from TF; fall back to a conservative default."""

    g_base_tool0 = urFwdKin(q_ref, rr.ROBOT_TYPE)
    try:
        g_base_tip = ur.get_current_transformation("base_link", TIP_FRAME_NAME)
        if g_base_tip is not None and not np.allclose(g_base_tip, np.eye(4)):
            return np.linalg.inv(g_base_tool0) @ g_base_tip
        print(f"TF frame '{TIP_FRAME_NAME}' unavailable; using default tip transform.")
    except Exception as exc:
        print(f"TF lookup for '{TIP_FRAME_NAME}' failed ({exc}); using default tip transform.")
    return DEFAULT_T_TOOL0_TIP.copy()


def select_best_solution(theta_6xN: np.ndarray, q_prev: np.ndarray, robot_type: str, T_tool0_tip: np.ndarray, W=None) -> np.ndarray:
    """Pick the IK column closest to the previous state while staying safe."""

    q_prev = np.asarray(q_prev, dtype=float).reshape(6)
    if theta_6xN.size == 0:
        raise RuntimeError("IK: no valid solutions from urInvKin.")

    if W is None:
        W = np.diag([1, 1, 1, 2, 2, 3])

    best_q = None
    best_cost = np.inf
    tip_tf = np.asarray(T_tool0_tip, dtype=float)

    for j in range(theta_6xN.shape[1]):
        q = np.asarray(theta_6xN[:, j], dtype=float).reshape(6)

        try:
            check_joint_limits(q)
        except Exception:
            continue

        try:
            g_tool0 = urFwdKin(q, robot_type)
            g_tip = g_tool0 @ tip_tf
            check_table_clearance(g_tip)
        except Exception:
            continue

        dq = wrap_to_pi(q - q_prev)
        cost = float(dq.T @ W @ dq)

        if cost < best_cost:
            best_cost = cost
            best_q = q

    if best_q is None:
        raise RuntimeError("IK: all candidate solutions violated safety constraints.")
    return best_q


def segment_to_joint_traj(g_start_tip: np.ndarray, g_end_tip: np.ndarray, q_seed: np.ndarray, robot_type: str, T_tool0_tip: np.ndarray, n_steps: int) -> np.ndarray:
    """Interpolate a Cartesian line in tip space and solve IK for each waypoint."""

    poses_tip = interp_cartesian_segment(g_start_tip, g_end_tip, n_steps)
    q_prev = np.array(q_seed, dtype=float).reshape(6)
    qs = []
    T_tip_to_tool0 = np.linalg.inv(T_tool0_tip)

    for g_tip_des in poses_tip[1:]:
        g_tool0_des = g_tip_des @ T_tip_to_tool0
        theta_6xN = urInvKin(g_tool0_des, robot_type)
        q_sol = select_best_solution(theta_6xN, q_prev, robot_type, T_tool0_tip)
        qs.append(q_sol)
        q_prev = q_sol

    # Return as shape (6, N) for downstream time-interval and move commands
    return np.array(qs).T


def make_time_intervals(ur: UrInterface, q_traj: np.ndarray, min_dt: float = 0.20):
    """Generate per-waypoint durations that obey the UR speed limit."""

    q_prev = ur.get_current_joints().astype(float)
    ts = []
    for i in range(q_traj.shape[1]):
        dq = np.max(np.abs(wrap_to_pi(q_traj[:, i] - q_prev)))
        dt = max(min_dt, float(dq) / max(ur.speed_limit, 1e-6))
        ts.append(dt)
        q_prev = q_traj[:, i]
    return ts


def exec_segment_ik(ur: UrInterface, q_curr: np.ndarray, g_curr_tip: np.ndarray, g_next_tip: np.ndarray, T_tool0_tip: np.ndarray, n_steps: int = 10):
    """Solve IK along a Cartesian segment and execute the resulting joint path."""

    ur.activate_pos_control()
    q_traj = segment_to_joint_traj(g_curr_tip, g_next_tip, q_curr, rr.ROBOT_TYPE, T_tool0_tip, n_steps)
    ts = make_time_intervals(ur, q_traj)
    ur.move_joints(q_traj, ts)
    q_new = ur.get_current_joints()
    g_tip_new = urFwdKin(q_new, rr.ROBOT_TYPE) @ T_tool0_tip
    return q_new, g_tip_new


def run_ik_mode(ur: UrInterface, home_q: np.ndarray):

    returned_home = False

    def go_home():
        nonlocal returned_home
        if not returned_home:
            rr.return_home(ur, home_q)
            returned_home = True

    try:
        # 1) teach + move to start
        q_start = rr.teach_pose(ur, "start")
        rr.move_to_configuration(ur, q_start)

        q_curr = ur.get_current_joints()
        T_tool0_tip = resolve_tool0_to_tip_transform(ur, q_curr)
        g_tip_des_start = urFwdKin(q_start, rr.ROBOT_TYPE) @ T_tool0_tip
        g_tip_curr = urFwdKin(q_curr, rr.ROBOT_TYPE) @ T_tool0_tip
        rr.publish_frame("start_pose", g_tip_curr)
        rr.log_pose_details("Start", g_tip_des_start, g_tip_curr)

        # 2) compute key poses (same as RR but in tip space)
        lift_vec = np.array([0.0, 0.0, rr.LIFT_HEIGHT])
        push_dir = rr.push_direction_from_pose(g_tip_curr)

        g_end1 = rr.cartesian_target(g_tip_curr, push_dir, rr.PUSH_DISTANCE)
        g_end1_up = rr.translate_pose(g_end1, lift_vec)
        g_front_above = rr.cartesian_target(g_end1_up, push_dir, rr.CUBE_LEN)
        g_contact2 = rr.translate_pose(g_front_above, -lift_vec)
        g_end2 = rr.cartesian_target(g_contact2, -push_dir, rr.PUSH_DISTANCE)

        # 3) execute segments (IK)
        # push1 (slow)
        rr.publish_frame("push1_end", g_end1)
        q_curr, g_tip_curr = exec_segment_ik(ur, q_curr, g_tip_curr, g_end1, T_tool0_tip, n_steps=12)

        # free-space (can be faster by temporarily increasing speed_limit like RR does)
        free_waypoints = [("end1_up", g_end1_up), ("front_above", g_front_above), ("contact2_pose", g_contact2)]
        for name, g_next in free_waypoints:
            rr.publish_frame(name, g_next)
            prev_limit = ur.speed_limit
            ur.speed_limit = rr.FREE_SPEED_LIMIT
            try:
                q_curr, g_tip_curr = exec_segment_ik(ur, q_curr, g_tip_curr, g_next, T_tool0_tip, n_steps=15)
            finally:
                ur.speed_limit = prev_limit

        # push2 (slow)
        rr.publish_frame("push2_end", g_end2)
        q_curr, g_tip_curr = exec_segment_ik(ur, q_curr, g_tip_curr, g_end2, T_tool0_tip, n_steps=12)

        rr.log_pose_details("Target", g_end2, g_tip_curr)
        print("IK push-and-place completed.")

    except Exception as e:
        print(f"IK mode aborted: {e}")
    finally:
        try:
            go_home()
        except Exception as e:
            print(f"Failed to return home: {e}")
