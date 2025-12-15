from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

import rr_control as rr
from IK_utils import urInvKin
from tf_frame import tf_frame
from ur_interface import UrInterface
from control import urFwdKin


class IKError(RuntimeError):
    """Raised when no valid IK solution can be found for a waypoint."""


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _pick_ik_solution(
    q_solutions: np.ndarray,
    q_ref: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Pick the closest IK solution (column) to the reference joint vector."""

    if q_solutions.size == 0:
        raise IKError("No IK solutions returned for this pose.")

    q_ref = np.asarray(q_ref, dtype=float).reshape(6)
    if weights is None:
        weights = np.ones(6)
    weights = np.asarray(weights, dtype=float).reshape(6)

    costs = np.full(q_solutions.shape[1], np.inf, dtype=float)
    for idx in range(q_solutions.shape[1]):
        qk = q_solutions[:, idx]
        if not np.isfinite(qk).all():
            continue
        dq = _wrap_to_pi(qk - q_ref)
        costs[idx] = float(np.sum(weights * np.abs(dq)))

    best_idx = int(np.argmin(costs))
    if not np.isfinite(costs[best_idx]):
        raise IKError("No finite IK solution cost found for this pose.")

    # Make the selected solution continuous with respect to q_ref so the
    # trajectory controller does not command unnecessary ±2π jumps.
    q_best = q_solutions[:, best_idx].copy()
    q_best = q_ref + _wrap_to_pi(q_best - q_ref)
    return q_best, best_idx, costs


def _interp_cartesian_segment(g_start: np.ndarray, g_end: np.ndarray, n_steps: int) -> List[np.ndarray]:
    """Straight-line interpolation in translation with fixed orientation."""

    poses: List[np.ndarray] = []
    p0 = g_start[:3, 3]
    p1 = g_end[:3, 3]
    for s in np.linspace(0.0, 1.0, n_steps):
        g = np.array(g_start, copy=True)
        g[:3, 3] = (1.0 - s) * p0 + s * p1
        poses.append(g)
    return poses


def _time_intervals_for_traj(
    q_waypoints: np.ndarray,
    q_start: np.ndarray,
    speed_limit: float,
    min_dt: float,
) -> List[float]:
    """Generate per-waypoint timing to respect a joint-speed limit."""

    q_prev = np.asarray(q_start, dtype=float).reshape(6)
    speed_limit = float(speed_limit)
    if speed_limit <= 0.0:
        raise ValueError("speed_limit must be positive.")

    times: List[float] = []
    for i in range(q_waypoints.shape[1]):
        q_i = q_waypoints[:, i]
        dq = _wrap_to_pi(q_i - q_prev)
        dt = max(float(min_dt), float(np.max(np.abs(dq))) / speed_limit)
        times.append(dt)
        q_prev = q_i
    return times


def _steps_for_segment(g_start: np.ndarray, g_target: np.ndarray) -> int:
    """Choose waypoint count based on RR tolerances (mirrors RR params)."""

    dist = float(np.linalg.norm(g_target[:3, 3] - g_start[:3, 3]))
    step = float(max(1e-4, rr.RR_POS_TOL))
    n_steps = int(max(2, np.ceil(dist / step) + 1))
    return min(n_steps, 200)


def ik_move_to_pose(
    ur: UrInterface,
    q_seed: np.ndarray,
    g_target_tool0: np.ndarray,
    robot_type: str = rr.ROBOT_TYPE,
    weights: Optional[np.ndarray] = None,
    speed_margin: float = rr.POS_SPEED_MARGIN,
) -> np.ndarray:
    """Move from current pose to ``g_target_tool0`` using IK waypoints."""

    ur.activate_pos_control()
    q_curr = ur.get_current_joints().astype(float).copy()
    rr.check_joint_limits(q_curr)

    g_curr = urFwdKin(q_curr, robot_type)
    rr.check_table_clearance(g_curr)

    g_target_tool0 = np.asarray(g_target_tool0, dtype=float)
    if g_target_tool0.shape != (4, 4):
        raise ValueError("g_target_tool0 must be a 4x4 matrix.")
    rr.check_table_clearance(g_target_tool0)

    n_steps = _steps_for_segment(g_curr, g_target_tool0)
    waypoints_tool0 = _interp_cartesian_segment(g_curr, g_target_tool0, n_steps)

    # Match RR behavior: always anchor continuity to the measured joints.
    # q_seed is kept for API compatibility, but the measured q_curr is the
    # safest reference for selecting/unwraping IK solutions.
    _ = q_seed
    q_prev = q_curr.copy()
    q_goals: List[np.ndarray] = []
    for g_des in waypoints_tool0[1:]:
        q_solutions = urInvKin(g_des, robot_type=robot_type)
        q_i, _, _ = _pick_ik_solution(q_solutions, q_prev, weights=weights)
        rr.check_joint_limits(q_i)
        g_fk = urFwdKin(q_i, robot_type)
        rr.check_table_clearance(g_fk)
        q_goals.append(q_i)
        q_prev = q_i

    if not q_goals:
        return q_curr

    q_waypoints = np.stack(q_goals, axis=1)  # 6xN
    margin = max(0.1, min(float(speed_margin), 1.0))
    speed_limit = float(ur.speed_limit) * margin
    time_intervals = _time_intervals_for_traj(q_waypoints, q_curr, speed_limit=speed_limit, min_dt=rr.RR_DT)

    ur.move_joints(q_waypoints, time_intervals=list(time_intervals))
    time.sleep(float(np.sum(time_intervals)) + 0.2)
    return ur.get_current_joints()


def run_ik_mode(ur: UrInterface, home_q: np.ndarray) -> None:
    """Execute the push-and-place sequence with IK-based control (ROS environment)."""

    returned_home = False

    def go_home() -> None:
        nonlocal returned_home
        if not returned_home:
            try:
                rr.return_home(ur, home_q)
            finally:
                returned_home = True

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

        def safe_ik_move(q_from: np.ndarray, g_target_tool0: np.ndarray) -> np.ndarray:
            try:
                weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.6])
                return ik_move_to_pose(
                    ur,
                    q_seed=q_from,
                    g_target_tool0=g_target_tool0,
                    robot_type=rr.ROBOT_TYPE,
                    weights=weights,
                    speed_margin=rr.POS_SPEED_MARGIN,
                )
            except rr.JointLimitError as exc:
                print(f"Joint limit reached: {exc}. Returning home.")
                go_home()
                raise

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
        print("IK-based push-and-place completed.")
        print(rr.tip_from_tool0(g_start_actual)[:3, 3] - g_start_actual[:3, 3])

    except Exception as exc:
        print(f"IK mode aborted due to error: {exc}")
        try:
            ur.activate_pos_control()
        except Exception:
            pass
    finally:
        tf_frame.shutdown()


__all__ = ["run_ik_mode", "IKError"]
