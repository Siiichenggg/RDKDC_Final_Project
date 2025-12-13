from __future__ import annotations
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

from RDKDC_Final_Project.Python.control import position_jacobian, urFwdKin
from RDKDC_Final_Project.Python.tf_frame import tf_frame
from RDKDC_Final_Project.Python.ur_interface import UrInterface


class JointLimitError(RuntimeError):
    """Raised when a joint exceeds the configured safety limits."""

    pass
# ---------------------------------------------------------------------------
# Global configuration (adjust in one place when moving to hardware)
# ---------------------------------------------------------------------------
ROBOT_TYPE = "ur5e"
USE_VEL_CONTROL = False  # Toggle RR mode between velocity control (True) and position steps (False)
PUSH_DIRECTION_BASE = np.array([0, 1.0, 0.0])  # nominal push direction in the base frame
PUSH_DISTANCE = 0.03  # 3 cm contact push distance
CUBE_LEN = 0.13  # cube edge length in meters
LIFT_HEIGHT = 0.08  # clearance height for free-space motion
RR_DT = 0.08
RR_KP = 0.6
RR_ORIENT_KP = 0.4
RR_POS_TOL = 5e-3
MAX_RR_ITERS = 600
RR_SPEED_MARGIN = 0.5  # scale commands to remain comfortably within the joint-speed limit
POS_SPEED_MARGIN = 0.6  # position-control moves stay inside the joint-speed limit
FREE_SPEED_LIMIT = 0.25  # faster joint-speed fraction for free-space RR moves (capped at hardware limit)
TABLE_Z_MIN = 0.02  # keep tool above the table plane by at least 2 cm
JOINT_LIMITS = np.deg2rad(
    np.array(
        [
            [-360, 360],
            [-360, 360],
            [-360, 360],
            [-360, 360],
            [-360, 360],
            [-360, 360],
        ]
    )
)
SIMULATION_MODE = True  # Set to False on the real robot to enable Freedrive.
SIM_START_Q = np.array([0.0, -1.3, 1.4, -1.4, -1.6, 0.0])
ENABLE_TF_FRAMES = True
DEFAULT_HOME_Q = np.array([0.0, -np.pi / 2, 0, -np.pi / 2, 0, 0.0])


tf_handles: dict[str, tf_frame] = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def publish_frame(name: str, g: np.ndarray) -> None:
    """Send an SE(3) pose to RViz via tf_frame for visualization."""

    if not ENABLE_TF_FRAMES:
        return
    if name not in tf_handles:
        tf_handles[name] = tf_frame("base_link", name, g)
    else:
        tf_handles[name].move_frame("base_link", g)


def teach_pose(ur: UrInterface, label: str) -> np.ndarray:
    """Implement the teach workflow described in the project PDF."""

    if not SIMULATION_MODE:
        print(f"\n--- Teach pose: {label} ---")
        print("Switching to Freedrive. Move the arm with the pendant, then press ENTER.")
        ur.switch_to_pendant_control()
        input("Press ENTER once the robot is at the desired pose...")
        q = ur.get_current_joints()
        ur.switch_to_ros_control()
        return q

    print(f"[SIM] Using taught pose '{label}' from the current RViz state.")
    if SIM_START_Q is not None and label.lower() == "start":
        return SIM_START_Q.copy()
    return ur.get_current_joints()


def check_joint_limits(q: np.ndarray) -> None:
    """Raise if any joint exits the conservative limits."""

    for idx in range(6):
        low, high = JOINT_LIMITS[idx]
        if not (low <= q[idx] <= high):
            raise JointLimitError(f"Joint {idx} exceeded limits: {q[idx]:.3f} rad")


def check_table_clearance(g: np.ndarray) -> None:
    """Keep the tool above the table plane."""

    if g[2, 3] < TABLE_Z_MIN:
        raise RuntimeError(
            f"Tool height {g[2,3]:.3f} m violates clearance constraint ({TABLE_Z_MIN:.3f} m)."
        )


def cartesian_target(g_src: np.ndarray, direction: np.ndarray, distance: float) -> np.ndarray:
    """Translate ``g_src`` along ``direction`` by ``distance`` while preserving orientation."""

    direction = direction / np.linalg.norm(direction)
    g_target = np.array(g_src, copy=True)
    g_target[:3, 3] = g_src[:3, 3] + distance * direction
    return g_target


def translate_pose(g_src: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate ``g_src`` by ``offset`` (in meters) while keeping the same rotation."""

    g_target = np.array(g_src, copy=True)
    g_target[:3, 3] = g_src[:3, 3] + offset
    return g_target


def push_direction_from_pose(_: np.ndarray) -> np.ndarray:
    """Return the normalized push direction expressed in the base frame."""

    direction = np.asarray(PUSH_DIRECTION_BASE, dtype=float).copy()
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise RuntimeError("PUSH_DIRECTION_BASE must be non-zero.")
    return direction / norm


def log_pose_details(label: str, g_des: np.ndarray, g_actual: np.ndarray) -> None:
    """Print desired vs. actual rotation and translation for reporting."""

    R_d = g_des[:3, :3]
    r_d = g_des[:3, 3]
    R = g_actual[:3, :3]
    r = g_actual[:3, 3]
    print(f"{label} desired R_d:\n{R_d}")
    print(f"{label} actual R:\n{R}")
    print(f"{label} desired r_d: {r_d}")
    print(f"{label} actual r: {r}")


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Map a rotation matrix to its so(3) vector via the matrix logarithm."""

    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


def rr_move_to_pose(
    ur: UrInterface,
    q_init: np.ndarray,
    g_des: np.ndarray,
    robot_type: str = ROBOT_TYPE,
) -> np.ndarray:
    """Resolved-rate servoing loop that drives the tool position to ``g_des``."""

    print("Starting resolved-rate segment...")
    q = q_init.astype(float).copy()
    use_vel_ctrl = USE_VEL_CONTROL
    if use_vel_ctrl:
        ur.activate_vel_control()
    else:
        ur.activate_pos_control()
    dt = RR_DT
    speed_limit = ur.speed_limit * RR_SPEED_MARGIN

    for _ in range(MAX_RR_ITERS):
        q_meas = ur.get_current_joints()
        q = q_meas.astype(float).copy()

        g = urFwdKin(q, robot_type)
        check_table_clearance(g)
        check_joint_limits(q)

        p = g[:3, 3]
        p_des = g_des[:3, 3]
        error = p_des - p
        if np.linalg.norm(error) < RR_POS_TOL:
            print("Resolved-rate target reached.")
            break

        v = RR_KP * error
        J_pos = position_jacobian(q, robot_type)
        qdot = np.linalg.pinv(J_pos, rcond=1e-3) @ v

        max_speed = np.max(np.abs(qdot))
        if max_speed > speed_limit:
            qdot = qdot / max_speed * speed_limit

        if use_vel_ctrl:
            ur.move_joints_vel(qdot)
            time.sleep(dt)
            q = q + qdot * dt
        else:
            q_next = q + qdot * dt
            ur.move_joints(q_next, time_intervals=[dt])
            time.sleep(dt)
            q = q_next
    else:
        print("Warning: RR loop hit max iterations without converging.")

    if use_vel_ctrl:
        ur.move_joints_vel(np.zeros(6))
    final_q = ur.get_current_joints()
    return final_q


def interp_cartesian_segment(g_start: np.ndarray, g_end: np.ndarray, n_steps: int) -> List[np.ndarray]:
    """Generate SE(3) poses along a straight segment with fixed orientation."""

    poses: List[np.ndarray] = []
    p0 = g_start[:3, 3]
    p1 = g_end[:3, 3]
    for s in np.linspace(0.0, 1.0, n_steps):
        g = np.array(g_start, copy=True)
        g[:3, 3] = (1 - s) * p0 + s * p1
        poses.append(g)
    return poses




def move_to_configuration(
    ur: UrInterface, q_target: np.ndarray, min_segment_time: float = 3.0, speed_margin: Optional[float] = None
) -> None:
    """Move the arm to ``q_target`` while respecting the configured joint-speed limit."""

    q_target = np.asarray(q_target, dtype=float)
    ur.activate_pos_control()
    q_curr = ur.get_current_joints()
    diff = np.abs(q_target - q_curr)
    move_time = min_segment_time
    margin = POS_SPEED_MARGIN if speed_margin is None else max(0.1, min(speed_margin, 1.0))
    speed_limit = ur.speed_limit * margin
    if speed_limit > 0.0 and np.max(diff) > 0.0:
        min_time = float(np.max(diff)) / speed_limit
        if min_time > move_time:
            move_time = min_time
    ur.move_joints(q_target, time_intervals=[move_time])
    time.sleep(move_time)


def return_home(ur: UrInterface, home_q: np.ndarray) -> None:
    """Send the robot back to the taught home configuration."""

    print("Returning to home configuration...")
    move_to_configuration(ur, home_q, min_segment_time=4.0)


def run_rr_mode(ur: UrInterface, home_q: np.ndarray) -> None:
    """Execute the push-and-place sequence with resolved-rate control."""

    returned_home = False

    def go_home() -> None:
        nonlocal returned_home
        if not returned_home:
            try:
                return_home(ur, home_q)
            finally:
                returned_home = True

    def safe_rr_move(q_from: np.ndarray, g_target: np.ndarray) -> np.ndarray:
        try:
            return rr_move_to_pose(ur, q_from, g_target, ROBOT_TYPE)
        except JointLimitError as exc:
            print(f"Joint limit reached: {exc}. Returning home.")
            go_home()
            raise

    try:
        q_start = teach_pose(ur, "start")
        move_to_configuration(ur, q_start)
        if USE_VEL_CONTROL:
            ur.activate_vel_control()
        else:
            ur.activate_pos_control()
        g_start = urFwdKin(q_start, ROBOT_TYPE)
        q_start_actual = ur.get_current_joints()
        g_start_actual = urFwdKin(q_start_actual, ROBOT_TYPE)
        publish_frame("start_pose", g_start_actual)
        log_pose_details("Start", g_start, g_start_actual)

        lift_vec = np.array([0.0, 0.0, LIFT_HEIGHT])
        push_dir = push_direction_from_pose(g_start_actual)
        g_end1 = cartesian_target(g_start_actual, push_dir, PUSH_DISTANCE)
        publish_frame("push1_end", g_end1)
        q_curr = safe_rr_move(q_start_actual, g_end1)

        g_end1_up = translate_pose(g_end1, lift_vec)
        g_front_above = cartesian_target(g_end1_up, push_dir, CUBE_LEN)
        g_contact2 = translate_pose(g_front_above, -lift_vec)
        publish_frame("contact2_pose", g_contact2)

        free_waypoints = (
            (g_end1_up, True),
            (g_front_above, True),
            (g_contact2, False),
        )
        for waypoint, fast_speed in free_waypoints:
            prev_limit = ur.speed_limit
            if fast_speed:
                ur.speed_limit = FREE_SPEED_LIMIT
            try:
                q_curr = safe_rr_move(q_curr, waypoint)
            finally:
                ur.speed_limit = prev_limit

        g_end2 = cartesian_target(g_contact2, -push_dir, PUSH_DISTANCE)
        publish_frame("push2_end", g_end2)
        q_end_final = safe_rr_move(q_curr, g_end2)
        g_target_actual = urFwdKin(q_end_final, ROBOT_TYPE)
        log_pose_details("Target", g_end2, g_target_actual)
        print("Resolved-rate push-and-place completed.")
    except Exception as exc:
        print(f"RR mode aborted due to error: {exc}")
        try:
            if USE_VEL_CONTROL:
                ur.move_joints_vel(np.zeros(6))
            else:
                ur.activate_pos_control()
        except Exception:
            pass
    finally:
        try:
            go_home()
        except Exception as home_exc:
            print(f"Failed to return home: {home_exc}")


__all__ = [
    "run_rr_mode",
    "return_home",
    "DEFAULT_HOME_Q",
    "teach_pose",
    "move_to_configuration",
]
