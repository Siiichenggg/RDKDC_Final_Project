import time
import numpy as np

import rr_control as rr
from IK_utils import urInvKin
from control import urFwdKin
from ur_interface import UrInterface


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


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
            rr.check_joint_limits(q)
        except Exception:
            continue

        try:
            g_tool0 = urFwdKin(q, robot_type)
            g_tip = g_tool0 @ tip_tf
            rr.check_table_clearance(g_tip)
        except Exception:
            continue

        dq = wrap_to_pi(q - q_prev)
        cost = float(dq.T @ W @ dq)
        # Penalize large configuration flips that might swing the elbow around.
        max_jump = float(np.max(np.abs(dq)))
        if max_jump > np.pi / 2:
            cost += 5.0 * max_jump

        if cost < best_cost:
            best_cost = cost
            best_q = q

    if best_q is None:
        raise RuntimeError("IK: all candidate solutions violated safety constraints.")
    return best_q


def segment_to_joint_traj(g_start_tip: np.ndarray, g_end_tip: np.ndarray, q_seed: np.ndarray, robot_type: str, T_tool0_tip: np.ndarray, n_steps: int) -> np.ndarray:
    """Interpolate a Cartesian line in tip space and solve IK for each waypoint."""

    poses_tip = rr.interp_cartesian_segment(g_start_tip, g_end_tip, n_steps)
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
        T_tool0_tip = rr.resolve_tool0_to_tip_transform(ur, q_curr, rr.USE_PEN_TIP)
        g_tool0_curr = urFwdKin(q_curr, rr.ROBOT_TYPE)
        g_tip_curr = rr.tip_from_tool0(g_tool0_curr, T_tool0_tip)
        rr.publish_frame("start_pose", g_tip_curr)
        rr.log_pose_details("Start", g_tip_curr, g_tip_curr)

        # Plan all waypoints in the control frame (tip if USE_PEN_TIP else tool0).
        push_dir = rr.push_direction_from_pose(g_tip_curr)
        plan = rr.generate_push_plan(g_tip_curr, push_dir)
        print(f"IK push direction: {push_dir}")

        for name, g_tip_target in plan:
            start_pose = np.array(g_tip_curr, copy=True)
            start_pos = g_tip_curr[:3, 3].copy()
            rr.check_table_clearance(g_tip_target)
            rr.publish_frame(f"{name}_des", g_tip_target)
            print(f"{name} start pose (control frame):\n{start_pose}")
            print(f"{name} target pose (control frame):\n{g_tip_target}")
            n_steps = rr.adaptive_interp_steps(g_tip_curr, g_tip_target)
            fast_segment = name in {"lift_after_push1", "side_lift", "retreat"}
            prev_limit = ur.speed_limit
            if fast_segment:
                ur.speed_limit = rr.FREE_SPEED_LIMIT
            start_time = time.time()
            try:
                q_curr, g_tip_curr = exec_segment_ik(
                    ur, q_curr, g_tip_curr, g_tip_target, T_tool0_tip, n_steps=n_steps
                )
            except Exception as exc:
                raise RuntimeError(
                    f"IK failed at segment '{name}' targeting position {g_tip_target[:3,3]}"
                ) from exc
            finally:
                ur.speed_limit = prev_limit
            duration = time.time() - start_time
            print(f"[IK] {name}: {n_steps} waypoints, {duration:.2f} s, pos {start_pos} -> {g_tip_target[:3,3]}")
            if name == "push2_end":
                rr.log_pose_details("Push2 end", g_tip_target, g_tip_curr)

        print("IK push-and-place completed.")

    except Exception as e:
        print(f"IK mode aborted: {e}")
    finally:
        try:
            go_home()
        except Exception as e:
            print(f"Failed to return home: {e}")


def main() -> None:
    """Standalone entry point for quickly running IK mode without the menu."""

    ur = UrInterface()
    home_q = rr.DEFAULT_HOME_Q.copy()
    run_ik_mode(ur, home_q)


if __name__ == "__main__":
    main()
