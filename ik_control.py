from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

import rr_control as rr
from control import urFwdKin
from IK_utils import urInvKin


class IKNoSolutionError(RuntimeError):
    """Raised when IK fails or no safe solution can be selected."""


def _wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""

    theta = np.asarray(theta, dtype=float)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _select_best_solution(theta_6xm: np.ndarray, q_prev: np.ndarray, robot_type: str) -> Tuple[np.ndarray, int]:
    """Pick the IK solution closest to q_prev that also passes safety checks."""

    theta_6xm = np.asarray(theta_6xm, dtype=float)
    q_prev = np.asarray(q_prev, dtype=float).flatten()
    if q_prev.size != 6:
        raise ValueError("q_prev must be a 6-element vector.")

    if theta_6xm.size == 0 or theta_6xm.shape[0] != 6:
        raise IKNoSolutionError("IK returned no candidate solutions.")

    best_cost = float("inf")
    best_q: Optional[np.ndarray] = None
    best_idx = -1

    for col in range(theta_6xm.shape[1]):
        q_raw = theta_6xm[:, col].flatten()
        if q_raw.size != 6 or not np.all(np.isfinite(q_raw)):
            continue

        dq = _wrap_to_pi(q_raw - q_prev)
        q_cand = q_prev + dq  # unwrap to remain closest to q_prev

        try:
            rr.check_joint_limits(q_cand)
            rr.check_table_clearance(urFwdKin(q_cand, robot_type))
        except Exception:
            continue

        cost = float(np.linalg.norm(dq))
        if cost < best_cost:
            best_cost = cost
            best_q = q_cand
            best_idx = col

    if best_q is None:
        raise IKNoSolutionError("All IK solutions failed safety checks (joint limits / table clearance).")

    return best_q, best_idx


def _interp_cartesian_segment_tool0(g_start: np.ndarray, g_end: np.ndarray, cart_step: float) -> List[np.ndarray]:
    """Interpolate a straight Cartesian segment (tool0) with fixed orientation."""

    p0 = g_start[:3, 3]
    p1 = g_end[:3, 3]
    dist = float(np.linalg.norm(p1 - p0))
    n_steps = max(2, int(np.ceil(dist / max(cart_step, 1e-6))) + 1)
    return rr.interp_cartesian_segment(g_start, g_end, n_steps)


def _compute_dt_for_joint_path(
    ur_speed_limit: float, q_path: Sequence[np.ndarray], base_dt: float, speed_margin: float
) -> float:
    """Compute a per-waypoint time step that respects joint-speed constraints."""

    speed_limit = float(max(ur_speed_limit * speed_margin, 1e-6))
    max_dq = 0.0
    for q_prev, q_next in zip(q_path[:-1], q_path[1:]):
        dq = np.max(np.abs(np.asarray(q_next) - np.asarray(q_prev)))
        if dq > max_dq:
            max_dq = float(dq)
    dt_needed = max_dq / speed_limit if max_dq > 0.0 else 0.0
    return float(max(base_dt, dt_needed))


def ik_follow_waypoints_tool0(
    ur,
    waypoints_tool0: Sequence[np.ndarray],
    q_start: np.ndarray,
    robot_type: str = rr.ROBOT_TYPE,
    base_dt: float = rr.RR_DT,
    cart_step: float = rr.RR_POS_TOL,
    speed_margin: float = rr.POS_SPEED_MARGIN,
    debug: bool = False,
) -> np.ndarray:
    """Follow a list of tool0 waypoints using IK (closest-solution selection)."""

    if len(waypoints_tool0) == 0:
        raise ValueError("waypoints_tool0 must be non-empty.")

    ur.activate_pos_control()
    q_prev = np.asarray(q_start, dtype=float).flatten()
    if q_prev.size != 6:
        raise ValueError("q_start must be a 6-element vector.")

    rr.check_joint_limits(q_prev)
    rr.check_table_clearance(urFwdKin(q_prev, robot_type))

    q_path: List[np.ndarray] = [q_prev.copy()]
    chosen: List[int] = []

    for k, g_des_tool0 in enumerate(waypoints_tool0):
        g_des_tool0 = np.asarray(g_des_tool0, dtype=float)
        if g_des_tool0.shape != (4, 4):
            raise ValueError("Each waypoint must be a 4x4 SE(3) matrix.")

        theta = urInvKin(g_des_tool0, robot_type)
        q_next, idx = _select_best_solution(theta, q_prev, robot_type)
        q_path.append(q_next)
        chosen.append(idx)
        q_prev = q_next

        if debug and (k == 0 or k == len(waypoints_tool0) - 1):
            print(f"[IK] waypoint {k+1}/{len(waypoints_tool0)} selected solution idx={idx}")

    dt_exec = _compute_dt_for_joint_path(ur.speed_limit, q_path, base_dt, speed_margin)
    if debug and dt_exec > base_dt + 1e-9:
        print(f"[IK] Time scaling: dt {base_dt:.3f} -> {dt_exec:.3f} to respect speed limit.")

    # Send the joint path as a trajectory (skip the first point: it's the current pose).
    q_goal = np.column_stack(q_path[1:])
    time_intervals = [dt_exec] * q_goal.shape[1]
    ur.move_joints(q_goal, time_intervals=time_intervals)
    time.sleep(sum(time_intervals))

    return ur.get_current_joints()


def ik_move_to_pose(
    ur,
    q_init: np.ndarray,
    g_des_tool0: np.ndarray,
    robot_type: str = rr.ROBOT_TYPE,
    base_dt: float = rr.RR_DT,
    cart_step: float = rr.RR_POS_TOL,
    speed_margin: float = rr.POS_SPEED_MARGIN,
    debug: bool = False,
) -> np.ndarray:
    """Plan a straight Cartesian segment to g_des_tool0 and execute it using IK."""

    print("Starting IK segment...")
    q_meas = ur.get_current_joints()
    q_start = np.asarray(q_meas if q_meas is not None else q_init, dtype=float).flatten()

    g_start = urFwdKin(q_start, robot_type)
    rr.check_table_clearance(g_start)
    rr.check_joint_limits(q_start)

    waypoints = _interp_cartesian_segment_tool0(g_start, g_des_tool0, cart_step=cart_step)
    # rr.interp_cartesian_segment includes the starting pose; skip it for IK solve.
    return ik_follow_waypoints_tool0(
        ur,
        waypoints_tool0=waypoints[1:],
        q_start=q_start,
        robot_type=robot_type,
        base_dt=base_dt,
        cart_step=cart_step,
        speed_margin=speed_margin,
        debug=debug,
    )


def run_ik_mode(ur, home_q: np.ndarray) -> None:
    """Execute the push-and-place sequence using IK tracking."""

    returned_home = False

    def go_home() -> None:
        nonlocal returned_home
        if not returned_home:
            try:
                rr.return_home(ur, home_q)
            finally:
                returned_home = True

    def safe_ik_move(q_from: np.ndarray, g_target: np.ndarray) -> np.ndarray:
        try:
            return ik_move_to_pose(
                ur,
                q_from,
                g_target,
                robot_type=rr.ROBOT_TYPE,
                base_dt=rr.RR_DT,
                cart_step=rr.RR_POS_TOL,
                speed_margin=rr.POS_SPEED_MARGIN,
                debug=False,
            )
        except rr.JointLimitError as exc:
            print(f"Joint limit reached: {exc}. Returning home.")
            go_home()
            raise

    try:
        q_start = rr.teach_pose(ur, "start")
        rr.move_to_configuration(ur, q_start)
        ur.activate_pos_control()

        g_start = urFwdKin(q_start, rr.ROBOT_TYPE)
        q_start_actual = ur.get_current_joints()
        g_start_actual = urFwdKin(q_start_actual, rr.ROBOT_TYPE)

        g_start_contact = rr.tip_from_tool0(g_start_actual)
        rr.publish_frame("start_pose", g_start_contact)
        rr.log_pose_details("Start", rr.tip_from_tool0(g_start), g_start_contact)

        lift_vec = np.array([0.0, 0.0, rr.LIFT_HEIGHT])
        push_dir_base = rr.compute_push_dir_base(rr.PUSH_DIR_INPUT, rr.PUSH_DIR_FRAME)
        print(f"[PUSH] frame={rr.PUSH_DIR_FRAME} input={rr.PUSH_DIR_INPUT}")
        print(f"[PUSH] push_dir_base(planar)={push_dir_base}")

        g_end1_contact = rr.cartesian_target(g_start_contact, push_dir_base, rr.PUSH_DISTANCE)
        g_end1 = rr.tool0_from_tip(g_end1_contact)
        rr.publish_frame("push1_end", g_end1_contact)
        q_curr = safe_ik_move(q_start_actual, g_end1)

        g_end1_up_contact = rr.translate_pose(g_end1_contact, lift_vec)
        g_front_above_contact = rr.cartesian_target(g_end1_up_contact, push_dir_base, rr.CUBE_LEN)
        g_contact2 = rr.translate_pose(g_front_above_contact, -lift_vec)
        rr.publish_frame("contact2_pose", g_contact2)

        free_waypoints = (
            (rr.tool0_from_tip(g_end1_up_contact), True),
            (rr.tool0_from_tip(g_front_above_contact), True),
            (rr.tool0_from_tip(g_contact2), False),
        )
        for waypoint, fast_speed in free_waypoints:
            prev_limit = ur.speed_limit
            if fast_speed:
                ur.speed_limit = rr.FREE_SPEED_LIMIT
            try:
                q_curr = safe_ik_move(q_curr, waypoint)
            finally:
                ur.speed_limit = prev_limit

        g_end2_contact = rr.cartesian_target(g_contact2, -push_dir_base, rr.PUSH_DISTANCE + 0.1)
        g_end2 = rr.tool0_from_tip(g_end2_contact)
        rr.publish_frame("push2_end", g_end2_contact)
        q_end_final = safe_ik_move(q_curr, g_end2)

        g_target_actual = urFwdKin(q_end_final, rr.ROBOT_TYPE)
        rr.log_pose_details("Target", g_end2_contact, rr.tip_from_tool0(g_target_actual))
        print("IK push-and-place completed.")
        print(rr.tip_from_tool0(g_start_actual)[:3, 3] - g_start_actual[:3, 3])

    except Exception as exc:
        print(f"IK mode aborted due to error: {exc}")
        try:
            ur.activate_pos_control()
        except Exception:
            pass

    finally:
        rr.tf_frame.shutdown()


__all__ = ["run_ik_mode"]
