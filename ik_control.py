from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from IK_utils import urInvKin
from control import urFwdKin
from rr_control import (
    JOINT_LIMITS,
    ROBOT_TYPE,
    TABLE_Z_MIN,
    move_to_configuration,
    teach_pose,
)
from ur_interface import UrInterface


# ---------------------------------------------------------------------------
# Tool geometry (tool0 -> pen tip)
# ---------------------------------------------------------------------------
# The pen holder offsets the tip along the tool0 +Z direction.  The value below
# (17.5 cm) matches the geometry used in the course handout and can be adjusted
# easily if a different end-effector is mounted.
T_TOOL0_TO_PEN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.175],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to the ``[-pi, pi]`` range element-wise."""

    return (angle + np.pi) % (2 * np.pi) - np.pi


def is_within_limits(q: np.ndarray, limits: np.ndarray) -> bool:
    """Check joint limit satisfaction."""

    for idx in range(6):
        lo, hi = limits[idx]
        if q[idx] < lo or q[idx] > hi:
            return False
    return True


def joint_limit_cost(q: np.ndarray, limits: np.ndarray, margin: float = 0.1) -> float:
    """Penalty that grows as a joint approaches its limits."""

    cost = 0.0
    for idx in range(6):
        lo, hi = limits[idx]
        range_half = (hi - lo) / 2.0
        mid = (hi + lo) / 2.0
        dist = range_half - abs(q[idx] - mid)
        if dist < margin:
            cost += (margin - dist) ** 2 * 10.0
    return cost


def discretize_cartesian_line(
    g_start: np.ndarray, g_end: np.ndarray, step_size: float = 0.003
) -> List[np.ndarray]:
    """Discretize a straight-line Cartesian path (fixed orientation)."""

    p0 = g_start[:3, 3]
    p1 = g_end[:3, 3]
    distance = float(np.linalg.norm(p1 - p0))
    n_steps = max(2, int(math.ceil(distance / step_size)) + 1)
    poses: List[np.ndarray] = []
    for s in np.linspace(0.0, 1.0, n_steps):
        g = np.array(g_start, copy=True)
        g[:3, 3] = (1 - s) * p0 + s * p1
        poses.append(g)
    return poses


def pen_tip_to_tool0(g_pen_tip: np.ndarray) -> np.ndarray:
    """Convert a pen-tip pose to the equivalent tool0 pose."""

    return g_pen_tip @ np.linalg.inv(T_TOOL0_TO_PEN)


def tool0_to_pen_tip(g_tool0: np.ndarray) -> np.ndarray:
    """Convert a tool0 pose to the attached pen-tip pose."""

    return g_tool0 @ T_TOOL0_TO_PEN


# ---------------------------------------------------------------------------
# IK selection logic
# ---------------------------------------------------------------------------

def pick_best_IK_solution(
    Q: np.ndarray,
    q_prev: Optional[np.ndarray],
    limits: np.ndarray = JOINT_LIMITS,
    target_pose: Optional[np.ndarray] = None,
    table_z_min: float = TABLE_Z_MIN,
) -> Tuple[np.ndarray, float]:
    """Filter and score the analytical IK solutions.

    Returns the chosen solution and its cost. Raises ``RuntimeError`` if no
    feasible solution remains after filtering.
    """

    if Q.size == 0:
        raise RuntimeError("No IK solutions provided.")

    valid_solutions: List[Tuple[np.ndarray, float]] = []
    for i in range(Q.shape[1]):
        q = wrap_to_pi(Q[:, i])
        if not is_within_limits(q, limits):
            continue
        if target_pose is not None and target_pose[2, 3] < table_z_min:
            continue

        if q_prev is None:
            continuity_cost = float(np.linalg.norm(q))
        else:
            diff = wrap_to_pi(q - q_prev)
            continuity_cost = float(np.sum(diff**2))
            elbow_flip_penalty = 0.0
            for j in (1, 2, 4):
                if abs(diff[j]) > np.deg2rad(90):
                    elbow_flip_penalty += 5.0
            continuity_cost += elbow_flip_penalty
        continuity_cost += joint_limit_cost(q, limits)
        valid_solutions.append((q, continuity_cost))

    if not valid_solutions:
        raise RuntimeError("No IK solutions satisfied constraints.")

    valid_solutions.sort(key=lambda item: item[1])
    return valid_solutions[0]


# ---------------------------------------------------------------------------
# IK trajectory generator
# ---------------------------------------------------------------------------

def ik_follow_SE3(
    g_list_tool0: Sequence[np.ndarray],
    q_seed: np.ndarray,
    robot_type: str = ROBOT_TYPE,
    limits: np.ndarray = JOINT_LIMITS,
    table_z_min: float = TABLE_Z_MIN,
) -> np.ndarray:
    """Map a list of ``base -> tool0`` poses to a continuous joint trajectory."""

    q_prev = wrap_to_pi(np.asarray(q_seed, dtype=float).flatten())
    traj: List[np.ndarray] = [q_prev]

    for g_des in g_list_tool0[1:]:
        Q = urInvKin(g_des, robot_type)
        q_best, _ = pick_best_IK_solution(Q, q_prev, limits, g_des, table_z_min)
        traj.append(q_best)
        q_prev = q_best
    return np.stack(traj)


# ---------------------------------------------------------------------------
# High-level IK mode
# ---------------------------------------------------------------------------

def execute_joint_trajectory(
    ur: UrInterface, q_traj: np.ndarray, segment_time: float = 0.25
) -> None:
    """Send a time-parameterized joint trajectory to the robot."""

    ur.activate_pos_control()
    n_points = q_traj.shape[0]
    ur.move_joints(q_traj.T, time_intervals=[segment_time] * n_points)


def run_ik_mode(ur: UrInterface, home_q: np.ndarray) -> None:
    """Teach start/goal poses, plan an IK path, and execute it."""

    returned_home = False

    def go_home() -> None:
        nonlocal returned_home
        if not returned_home:
            try:
                move_to_configuration(ur, home_q, min_segment_time=4.0)
            finally:
                returned_home = True

    try:
        q_start = teach_pose(ur, "start (IK)")
        move_to_configuration(ur, q_start)
        q_target = teach_pose(ur, "target (IK)")
        g_start = urFwdKin(q_start, ROBOT_TYPE)
        g_target = urFwdKin(q_target, ROBOT_TYPE)
        g_waypoints = discretize_cartesian_line(g_start, g_target, step_size=0.0025)
        q_traj = ik_follow_SE3(g_waypoints, q_start, robot_type=ROBOT_TYPE)
        execute_joint_trajectory(ur, q_traj)
        g_final = urFwdKin(q_traj[-1], ROBOT_TYPE)
        pos_err = np.linalg.norm(g_final[:3, 3] - g_target[:3, 3])
        rot_err = np.linalg.norm(g_final[:3, :3] - g_target[:3, :3])
        print(
            "IK mode completed. "
            f"Position error: {pos_err:.4f} m, rotation error Frobenius: {rot_err:.4f}"
        )
    except Exception as exc:
        print(f"IK mode aborted: {exc}")
    finally:
        try:
            go_home()
        except Exception as home_exc:
            print(f"Failed to return home after IK mode: {home_exc}")


__all__ = [
    "pick_best_IK_solution",
    "ik_follow_SE3",
    "pen_tip_to_tool0",
    "tool0_to_pen_tip",
    "T_TOOL0_TO_PEN",
    "run_ik_mode",
]
