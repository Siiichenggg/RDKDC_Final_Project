import time
import numpy as np

import rr_control as rr
from IK_utils import urInvKin
from control import urFwdKin
from ur_interface import UrInterface


def unwrap_near_reference(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Shift each joint by +/- 2π to stay numerically close to q_ref."""
    q = np.asarray(q, dtype=float).reshape(6).copy()
    q_ref = np.asarray(q_ref, dtype=float).reshape(6)
    two_pi = 2.0 * np.pi

    out = q.copy()
    for i in range(6):
        low, high = rr.JOINT_LIMITS[i]
        k0 = int(np.round((q_ref[i] - q[i]) / two_pi))
        best = None
        best_abs = None
        for k in (k0 - 1, k0, k0 + 1):
            cand = q[i] + two_pi * k
            if cand < low - 1e-9 or cand > high + 1e-9:
                continue
            abs_diff = float(abs(cand - q_ref[i]))
            if best is None or abs_diff < best_abs:
                best = float(cand)
                best_abs = abs_diff
        if best is None:
            best = float(q[i])
        out[i] = best
    return out


def select_best_solution(
    theta_6xN: np.ndarray,
    q_prev: np.ndarray,
    robot_type: str,
    T_tool0_tip: np.ndarray,
    W=None,
) -> np.ndarray:
    """Pick the IK column closest to the previous state while staying safe."""
    q_prev = np.asarray(q_prev, dtype=float).reshape(6)
    if theta_6xN.size == 0:
        raise RuntimeError("IK: no valid solutions from urInvKin.")

    if W is None:
        W = np.diag([1, 1, 1, 2, 3, 5])

    best_q = None
    best_cost = np.inf
    tip_tf = np.asarray(T_tool0_tip, dtype=float)

    for j in range(theta_6xN.shape[1]):
        q_raw = np.asarray(theta_6xN[:, j], dtype=float).reshape(6)
        q = unwrap_near_reference(q_raw, q_prev)

        # Joint limits
        try:
            rr.check_joint_limits(q)
        except Exception:
            continue

        # Table clearance in CONTROL frame (tip if enabled, else tool0)
        try:
            g_tool0 = urFwdKin(q, robot_type)
            g_tip = g_tool0 @ tip_tf
            rr.check_table_clearance(g_tip if rr.USE_PEN_TIP else g_tool0)
        except Exception:
            continue

        dq = q - q_prev
        cost = float(dq.T @ W @ dq)

        # Penalize large jumps to reduce elbow-flips
        max_jump = float(np.max(np.abs(dq)))
        if max_jump > np.pi / 3:
            cost += 8.0 * max_jump

        # Bias away from wrist singularities (joint 5 ~ 0)
        wrist_angle = abs(q[4])
        if wrist_angle < 0.08:
            cost += 100.0 * (0.08 - wrist_angle)

        if cost < best_cost:
            best_cost = cost
            best_q = q

    if best_q is None:
        raise RuntimeError("IK: all candidate solutions violated safety constraints.")
    return best_q


def segment_to_joint_traj(
    g_start_tip: np.ndarray,
    g_end_tip: np.ndarray,
    q_seed: np.ndarray,
    robot_type: str,
    T_tool0_tip: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Interpolate a Cartesian line in tip space and solve IK for each waypoint."""
    poses_tip = rr.interp_cartesian_segment(g_start_tip, g_end_tip, n_steps)
    q_prev = np.array(q_seed, dtype=float).reshape(6)
    qs = [q_prev.copy()]

    T_tip_to_tool0 = np.linalg.inv(T_tool0_tip)

    for g_tip_des in poses_tip[1:]:
        # IK expects base->tool0
        g_tool0_des = g_tip_des @ T_tip_to_tool0
        theta_6xN = urInvKin(g_tool0_des, robot_type)
        q_sol = select_best_solution(theta_6xN, q_prev, robot_type, T_tool0_tip)
        qs.append(q_sol)
        q_prev = q_sol

    return np.array(qs).T  # shape (6, Nwaypoints)


def make_time_intervals(ur: UrInterface, q_traj: np.ndarray, min_dt: float = 0.20):
    """Generate per-waypoint durations that obey the UR speed limit."""
    q_prev = ur.get_current_joints().astype(float)
    ts = []
    for i in range(q_traj.shape[1]):
        dq = float(np.max(np.abs(q_traj[:, i] - q_prev)))
        dt = max(min_dt, float(dq) / max(ur.speed_limit, 1e-6))
        ts.append(dt)
        q_prev = q_traj[:, i]
    return ts


def exec_segment_ik(
    ur: UrInterface,
    q_curr: np.ndarray,
    g_curr_tip: np.ndarray,
    g_next_tip: np.ndarray,
    T_tool0_tip: np.ndarray,
    n_steps: int = 10,
):
    """Solve IK along a Cartesian segment and execute the resulting joint path."""
    ur.activate_pos_control()

    q_traj = segment_to_joint_traj(
        g_curr_tip, g_next_tip, q_curr, rr.ROBOT_TYPE, T_tool0_tip, n_steps
    )
    ts = make_time_intervals(ur, q_traj)
    total_time = float(np.sum(ts))
    print(f"[IK] Executing segment with {q_traj.shape[1]} waypoints, est {total_time:.2f}s")

    ur.move_joints(q_traj, ts)

    def wait_until_reached(
        q_goal: np.ndarray,
        timeout: float = None,
        tol: float = 0.02,
        stable_count: int = 3,
    ) -> np.ndarray:
        """Block until the robot settles at q_goal so segments don't overwrite each other."""
        if timeout is None:
            timeout = max(total_time + 2.0, 2.0 * total_time + 5.0)

        start_time = time.time()
        consecutive = 0
        while time.time() - start_time < timeout:
            q_meas = ur.get_current_joints().astype(float)
            if np.max(np.abs(rr.wrap_to_pi(q_meas - q_goal))) < tol:
                consecutive += 1
                if consecutive >= stable_count:
                    return q_meas
            else:
                consecutive = 0
            time.sleep(0.05)

        raise TimeoutError(f"Segment timed out after {timeout:.1f}s without reaching goal.")

    q_new = wait_until_reached(q_traj[:, -1])
    g_tool0_new = urFwdKin(q_new, rr.ROBOT_TYPE)
    g_tip_new = g_tool0_new @ T_tool0_tip
    return q_new, (g_tip_new if rr.USE_PEN_TIP else g_tool0_new)


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
        g_control_curr = (g_tool0_curr @ T_tool0_tip) if rr.USE_PEN_TIP else g_tool0_curr
        g_control_start = np.array(g_control_curr, copy=True)

        rr.publish_frame("start_pose", g_control_curr)
        rr.log_pose_details("Start", g_control_curr, g_control_curr)

        # Plan waypoints in the control frame (tip if USE_PEN_TIP else tool0).
        push_dir = rr.resolve_push_direction(g_control_curr)
        plan = rr.generate_push_plan(g_control_curr, push_dir)
        print(f"Plan segments: {[name for name, _ in plan]}")
        print(f"IK push direction: {push_dir}")

        for name, g_control_target in plan:
            rr.check_table_clearance(g_control_target)
            rr.publish_frame(f"{name}_des", g_control_target)

            n_steps = rr.adaptive_interp_steps(g_control_curr, g_control_target)

            fast_segment = name in {
                "lift_after_push",
                "cross_over",
                "retreat_lift",
                "return_above_origin",
                "back_over_start",
            }
            prev_limit = ur.speed_limit
            if fast_segment:
                ur.speed_limit = rr.FREE_SPEED_LIMIT

            start_time = time.time()
            try:
                q_curr, g_control_curr = exec_segment_ik(
                    ur,
                    q_curr,
                    g_control_curr,
                    g_control_target,
                    T_tool0_tip,
                    n_steps=n_steps,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"IK failed at segment '{name}' targeting position {g_control_target[:3,3]}: {exc}"
                ) from exc
            finally:
                ur.speed_limit = prev_limit

            duration = time.time() - start_time
            print(f"[IK] {name}: {n_steps} steps, {duration:.2f}s")

            if name == "push_2_end":
                rr.log_pose_details("Return push end", g_control_target, g_control_curr)

        q_final = ur.get_current_joints().astype(float)
        rr.verify_return_to_start(q_start, g_control_start, q_final, T_tool0_tip)
        print("IK push-and-place completed.")

    except Exception as e:
        print(f"IK mode aborted: {e}")
    finally:
        try:
            go_home()
        except Exception as e:
            print(f"Failed to return home: {e}")


def main() -> None:
    ur = UrInterface()
    home_q = rr.DEFAULT_HOME_Q.copy()
    run_ik_mode(ur, home_q)


if __name__ == "__main__":
    main()
