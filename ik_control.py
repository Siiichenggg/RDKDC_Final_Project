from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from tf_transformations import quaternion_matrix

import rr_control as rr
from IK_utils import urInvKin
from tf_frame import tf_frame
from ur_interface import UrInterface


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
    return q_solutions[:, best_idx].copy(), best_idx, costs


def _lookup_transform_matrix(
    target_frame: str,
    source_frame: str,
    timeout_s: float = 2.0,
) -> np.ndarray:
    """Return g_target_source using TF (4x4)."""

    tf_buffer = tf_frame.get_tf_tree()
    if tf_buffer is None:
        raise RuntimeError("TF buffer unavailable.")

    transform = tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        Time(),
        timeout=Duration(seconds=float(timeout_s)),
    )

    q = transform.transform.rotation
    t = transform.transform.translation
    g = quaternion_matrix([q.x, q.y, q.z, q.w])  # (x, y, z, w)
    g[:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
    return g


def _get_g_base_tool0(timeout_s: float = 2.0) -> np.ndarray:
    """Robustly lookup base_link->tool0 transform."""

    candidates = ("tool0", "tool0_controller", "tcp", "tool0_tcp")
    last_exc: Optional[Exception] = None
    for source in candidates:
        try:
            return _lookup_transform_matrix("base_link", source, timeout_s=timeout_s)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to lookup base_link->tool0 TF (tried {candidates}): {last_exc}")


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


def _ik_solve_cartesian_waypoints(
    g_waypoints_tool0: Sequence[np.ndarray],
    q_seed: np.ndarray,
    robot_type: str,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve IK for a sequence of tool0 poses, choosing a continuous branch."""

    q_prev = np.asarray(q_seed, dtype=float).reshape(6)
    q_traj = np.zeros((6, len(g_waypoints_tool0)), dtype=float)

    for i, g_des in enumerate(g_waypoints_tool0):
        q_solutions = urInvKin(np.asarray(g_des, dtype=float), robot_type=robot_type)
        q_i, _, _ = _pick_ik_solution(q_solutions, q_prev, weights=weights)
        rr.check_joint_limits(q_i)
        rr.check_table_clearance(rr.tip_from_tool0(g_des))
        q_traj[:, i] = q_i
        q_prev = q_i

    return q_traj


def _time_intervals_for_traj(
    q_waypoints: np.ndarray,
    q_start: np.ndarray,
    speed_limit: float,
    min_dt: float = 0.08,
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


def _execute_joint_trajectory(
    ur: UrInterface,
    q_waypoints: np.ndarray,
    time_intervals: Sequence[float],
) -> None:
    """Send the trajectory and wait for completion."""

    ur.activate_pos_control()
    ur.move_joints(q_waypoints, time_intervals=list(time_intervals))
    time.sleep(float(np.sum(time_intervals)) + 0.2)


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
        q_curr = ur.get_current_joints()

        g_start_tool0 = _get_g_base_tool0()
        g_start_contact = rr.tip_from_tool0(g_start_tool0)
        rr.publish_frame("start_pose", g_start_contact)

        push_dir_base = rr.compute_push_dir_base(rr.PUSH_DIR_INPUT, rr.PUSH_DIR_FRAME)
        lift_vec = np.array([0.0, 0.0, rr.LIFT_HEIGHT])

        g_end1_contact = rr.cartesian_target(g_start_contact, push_dir_base, rr.PUSH_DISTANCE)
        rr.publish_frame("push1_end", g_end1_contact)

        segments_contact: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
        segments_contact.append(("push1", g_start_contact, g_end1_contact, 25))

        g_end1_up_contact = rr.translate_pose(g_end1_contact, lift_vec)
        g_front_above_contact = rr.cartesian_target(g_end1_up_contact, push_dir_base, rr.CUBE_LEN)
        g_contact2 = rr.translate_pose(g_front_above_contact, -lift_vec)
        rr.publish_frame("contact2_pose", g_contact2)

        segments_contact.append(("lift1", g_end1_contact, g_end1_up_contact, 20))
        segments_contact.append(("move_over", g_end1_up_contact, g_front_above_contact, 25))
        segments_contact.append(("down2", g_front_above_contact, g_contact2, 20))

        g_end2_contact = rr.cartesian_target(g_contact2, -push_dir_base, rr.PUSH_DISTANCE + 0.1)
        rr.publish_frame("push2_end", g_end2_contact)
        segments_contact.append(("push2", g_contact2, g_end2_contact, 35))

        weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.6])

        for label, g0_contact, g1_contact, n_steps in segments_contact:
            g0_tool0 = rr.tool0_from_tip(g0_contact)
            g1_tool0 = rr.tool0_from_tip(g1_contact)

            waypoints_tool0 = _interp_cartesian_segment(g0_tool0, g1_tool0, n_steps)
            q_waypoints = _ik_solve_cartesian_waypoints(waypoints_tool0, q_curr, rr.ROBOT_TYPE, weights=weights)

            speed_limit = ur.speed_limit * rr.POS_SPEED_MARGIN
            time_intervals = _time_intervals_for_traj(q_waypoints, q_curr, speed_limit=speed_limit, min_dt=0.08)
            print(f"[IK] Executing segment '{label}' with {q_waypoints.shape[1]} waypoints (T={sum(time_intervals):.2f}s)")
            _execute_joint_trajectory(ur, q_waypoints, time_intervals)

            q_curr = ur.get_current_joints()

        print("IK-based push-and-place completed.")

    except rr.JointLimitError as exc:
        print(f"IK mode aborted due to joint limit: {exc}")
        go_home()
    except IKError as exc:
        print(f"IK mode aborted due to IK failure: {exc}")
        go_home()
    except Exception as exc:
        print(f"IK mode aborted due to error: {exc}")
        go_home()
    finally:
        tf_frame.shutdown()


__all__ = ["run_ik_mode", "IKError"]
