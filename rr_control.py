"""Resolved-rate control module for UR5/UR5e robot arm.

This module implements resolved-rate (RR) control for the UR robot arm,
including push-and-place task execution with Cartesian space control.
It provides functions for teaching poses, moving to configurations, and
executing complex manipulation tasks using velocity or position control.
"""

from __future__ import annotations
import time
from typing import List, Optional, Sequence

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time

from control import normalize, position_jacobian, urFwdKin
from tf_frame import tf_frame
from ur_interface import UrInterface


class JointLimitError(RuntimeError):
    """Raised when a joint exceeds the configured safety limits."""


# ---------------------------------------------------------------------------
# Global configuration (adjust in one place when moving to hardware)
# ---------------------------------------------------------------------------
ROBOT_TYPE = "ur5e"
# Toggle RR mode between velocity control (True) and position steps (False)
USE_VEL_CONTROL = False
# Push direction frame: "base" or "world"
PUSH_DIR_FRAME = "world"
# Direction expressed in PUSH_DIR_FRAME
PUSH_DIR_INPUT = np.array([1.0, 0.0, 0.0])
# 3 cm contact push distance
PUSH_DISTANCE = 0.03
# Cube edge length in meters
CUBE_LEN = 0.25
# Clearance height for free-space motion
LIFT_HEIGHT = 0.10
# Resolved-rate control parameters
RR_DT = 0.08
RR_KP = 0.6
RR_ORIENT_KP = 0.4
RR_POS_TOL = 5e-3
MAX_RR_ITERS = 600
# Scale commands to remain comfortably within the joint-speed limit
RR_SPEED_MARGIN = 0.5
# Position-control moves stay inside the joint-speed limit
POS_SPEED_MARGIN = 0.6
# Faster joint-speed fraction for free-space RR moves (capped at hardware limit)
FREE_SPEED_LIMIT = 0.25
# Keep tool above the table plane by at least 2 cm
TABLE_Z_MIN = 0.14
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
# Set to False on the real robot to enable Freedrive
SIMULATION_MODE = False
SIM_START_Q = np.array([0.0, -1.3, 1.4, -1, -0.5, 1.0])
ENABLE_TF_FRAMES = True
DEFAULT_HOME_Q = np.array([0.0, -np.pi / 2, 0, -np.pi / 2, 0, 0.0])
# Whether to use pen_tip frame for teaching instead of tool0
USE_PEN_TIP = True
# Transform from tool0 to pen_tip frame
TOOL0_TO_PEN_TIP = np.array([[1, 0, 0, -0.049],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0.12228],
                              [0, 0, 0, 1]])
TF_HANDLES: dict[str, tf_frame] = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def publish_frame(name: str, g: np.ndarray) -> None:
    """Send an SE(3) pose to RViz via tf_frame for visualization."""

    if not ENABLE_TF_FRAMES:
        return
    if name not in TF_HANDLES:
        TF_HANDLES[name] = tf_frame("base_link", name, g)
    else:
        TF_HANDLES[name].move_frame("base_link", g)


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
    """Raise if any joint exceeds the conservative limits."""

    for idx in range(6):
        low, high = JOINT_LIMITS[idx]
        if not (low <= q[idx] <= high):
            raise JointLimitError(f"Joint {idx} exceeded limits: {q[idx]:.3f} rad")


def tip_from_tool0(g_tool0: np.ndarray) -> np.ndarray:
    """Compute the pen_tip frame from the tool0 frame."""

    if not USE_PEN_TIP:
        return g_tool0
    return g_tool0 @ TOOL0_TO_PEN_TIP


def tool0_from_tip(g_tip: np.ndarray) -> np.ndarray:
    """Compute the tool0 frame from the pen_tip frame."""

    if not USE_PEN_TIP:
        return g_tip
    return g_tip @ np.linalg.inv(TOOL0_TO_PEN_TIP)


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


def _quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Map a quaternion (x, y, z, w) to a rotation matrix."""

    x, y, z, w = quaternion
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def get_R_base_world(timeout: float = 1.0) -> np.ndarray:
    """Return the rotation from world to base_link, fallback to I on failure."""

    try:
        tf_buffer = tf_frame.get_tf_tree()
        if tf_buffer is None:
            raise RuntimeError("TF buffer unavailable.")
        transform = tf_buffer.lookup_transform(
            "base_link",
            "world",
            Time(),
            timeout=Duration(seconds=timeout),
        )
        quat = transform.transform.rotation
        return _quaternion_to_rotation_matrix((quat.x, quat.y, quat.z, quat.w))
    except Exception as exc:
        print(f"Warning: Failed to lookup world->base_link rotation ({exc}); using identity.")
        return np.eye(3)


def compute_push_dir_base(push_dir_input: np.ndarray, frame: str) -> np.ndarray:
    """Compute the planar push direction expressed in the base frame."""

    direction = np.asarray(push_dir_input, dtype=float).flatten()
    if direction.size != 3:
        raise ValueError("push_dir_input must be a 3-element vector.")
    frame_key = frame.lower()
    if frame_key == "base":
        v = normalize(direction)
    elif frame_key == "world":
        R_base_world = get_R_base_world()
        v = normalize(R_base_world @ direction)
    else:
        raise ValueError("PUSH_DIR_FRAME must be either 'base' or 'world'.")

    v[2] = 0.0
    planar_norm = np.linalg.norm(v)
    if planar_norm < 1e-6:
        raise ValueError(
            "Push direction has near-zero planar component after projection; choose a non-vertical direction."
        )
    return v / planar_norm


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
    """Resolved-rate servoing loop that drives the tool position to g_des."""

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
    ur: UrInterface,
    q_target: np.ndarray,
    min_segment_time: float = 3.0,
    speed_margin: Optional[float] = None
) -> None:
    """Move the arm to q_target while respecting the configured joint-speed limit."""

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


def compute_d_R3(r_actual: np.ndarray, r_desired: np.ndarray) -> float:
    """Compute position error d_R3 in millimeters.

    Args:
        r_actual: Actual position (3D vector)
        r_desired: Desired position (3D vector)

    Returns:
        Position error in millimeters: ||r - r_d|| × 1000
    """
    return np.linalg.norm(r_actual - r_desired) * 1000.0


def compute_d_SO3(R_actual: np.ndarray, R_desired: np.ndarray) -> float:
    """Compute orientation error d_SO3 (unitless).

    Args:
        R_actual: Actual rotation matrix (3x3)
        R_desired: Desired rotation matrix (3x3)

    Returns:
        Orientation error: sqrt(tr((R - R_d)(R - R_d)^T))
    """
    R_diff = R_actual - R_desired
    return np.sqrt(np.trace(R_diff @ R_diff.T))


def print_detailed_errors(
    g_start_desired: np.ndarray,
    g_start_actual: np.ndarray,
    g_end_desired: np.ndarray,
    g_end_actual: np.ndarray,
    control_strategy: str = "RR"
) -> None:
    """Print detailed error analysis for the push-and-place task.

    Args:
        g_start_desired: Desired start pose (4x4 homogeneous transform)
        g_start_actual: Actual start pose (4x4 homogeneous transform)
        g_end_desired: Desired end pose (4x4 homogeneous transform)
        g_end_actual: Actual end pose (4x4 homogeneous transform)
        control_strategy: Control strategy name (e.g., "RR ROS", "IK ROS", "RR RTDE")
    """

    print("\n" + "=" * 80)
    print(f"ERROR ANALYSIS - {control_strategy}")
    print("=" * 80)

    # Extract position and rotation for start
    r_start_des = g_start_desired[:3, 3]
    R_start_des = g_start_desired[:3, :3]
    r_start_act = g_start_actual[:3, 3]
    R_start_act = g_start_actual[:3, :3]

    # Extract position and rotation for target
    r_end_des = g_end_desired[:3, 3]
    R_end_des = g_end_desired[:3, :3]
    r_end_act = g_end_actual[:3, 3]
    R_end_act = g_end_actual[:3, :3]

    # Compute errors for Start
    d_R3_start = compute_d_R3(r_start_act, r_start_des)
    d_SO3_start = compute_d_SO3(R_start_act, R_start_des)

    # Compute errors for Target
    d_R3_target = compute_d_R3(r_end_act, r_end_des)
    d_SO3_target = compute_d_SO3(R_end_act, R_end_des)

    # Print results in table format
    print("\n" + "-" * 80)
    print(f"{'Location':<15} {'d_R3 (mm)':<20} {'d_SO3 (unitless)':<20}")
    print("-" * 80)
    print(f"{'Start':<15} {d_R3_start:<20.6f} {d_SO3_start:<20.6f}")
    print(f"{'Target':<15} {d_R3_target:<20.6f} {d_SO3_target:<20.6f}")
    print("-" * 80)

    # Additional detailed information
    print("\n--- Detailed Information ---")
    print(f"\nStart:")
    print(f"  Desired position (m): [{r_start_des[0]:.6f}, {r_start_des[1]:.6f}, {r_start_des[2]:.6f}]")
    print(f"  Actual position  (m): [{r_start_act[0]:.6f}, {r_start_act[1]:.6f}, {r_start_act[2]:.6f}]")
    print(f"  Position error (mm): {d_R3_start:.6f}")
    print(f"  Orientation error:   {d_SO3_start:.6f}")

    print(f"\nTarget:")
    print(f"  Desired position (m): [{r_end_des[0]:.6f}, {r_end_des[1]:.6f}, {r_end_des[2]:.6f}]")
    print(f"  Actual position  (m): [{r_end_act[0]:.6f}, {r_end_act[1]:.6f}, {r_end_act[2]:.6f}]")
    print(f"  Position error (mm): {d_R3_target:.6f}")
    print(f"  Orientation error:   {d_SO3_target:.6f}")

    print("=" * 80 + "\n")


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

        g_start_contact = tip_from_tool0(g_start_actual)
        publish_frame("start_pose", g_start_contact)
        log_pose_details("Start", tip_from_tool0(g_start), g_start_contact)
        lift_vec = np.array([0.0, 0.0, LIFT_HEIGHT])
        push_dir_base = compute_push_dir_base(PUSH_DIR_INPUT, PUSH_DIR_FRAME)
        print(f"[PUSH] frame={PUSH_DIR_FRAME} input={PUSH_DIR_INPUT}")
        print(f"[PUSH] push_dir_base(planar)={push_dir_base}")
        g_end1_contact = cartesian_target(g_start_contact, push_dir_base, PUSH_DISTANCE)
        g_end1 = tool0_from_tip(g_end1_contact)
        publish_frame("push1_end", g_end1_contact)
        q_curr = safe_rr_move(q_start_actual, g_end1)

        g_end1_up_contact = translate_pose(g_end1_contact, lift_vec)
        g_front_above_contact = cartesian_target(g_end1_up_contact, push_dir_base, CUBE_LEN)
        g_contact2 = translate_pose(g_front_above_contact, -lift_vec)
        publish_frame("contact2_pose", g_contact2)

        free_waypoints = (
            (tool0_from_tip(g_end1_up_contact), True),
            (tool0_from_tip(g_front_above_contact), True),
            (tool0_from_tip(g_contact2), False),
        )
        for waypoint, fast_speed in free_waypoints:
            prev_limit = ur.speed_limit
            if fast_speed:
                ur.speed_limit = FREE_SPEED_LIMIT
            try:
                q_curr = safe_rr_move(q_curr, waypoint)
            finally:
                ur.speed_limit = prev_limit

        g_end2_contact = cartesian_target(g_contact2, -push_dir_base, PUSH_DISTANCE + 0.1)
        g_end2 = tool0_from_tip(g_end2_contact)
        publish_frame("push2_end", g_end2_contact)
        q_end_final = safe_rr_move(q_curr, g_end2)
        g_target_actual = urFwdKin(q_end_final, ROBOT_TYPE)
        g_end_actual = tip_from_tool0(g_target_actual)

        log_pose_details("Target", g_end2_contact, g_end_actual)
        print("Resolved-rate push-and-place completed.")

        # Print detailed error analysis
        # For Start: compare taught start pose (from q_start FK) with actual start pose (from current joints FK)
        # For Target: compare desired end pose with actual end pose
        g_start_desired = tip_from_tool0(g_start)  # Desired start pose from taught joint angles
        print_detailed_errors(
            g_start_desired,  # Desired start pose (from taught q_start)
            g_start_contact,  # Actual start pose (from measured q_start_actual)
            g_end2_contact,   # Desired target pose
            g_end_actual,     # Actual target pose
            control_strategy="RR"
        )

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
        tf_frame.shutdown()


__all__ = [
    "run_rr_mode",
    "return_home",
    "DEFAULT_HOME_Q",
    "teach_pose",
    "move_to_configuration",
    "tip_from_tool0",
    "tool0_from_tip",
    "print_detailed_errors",
    "compute_d_R3",
    "compute_d_SO3",
]
