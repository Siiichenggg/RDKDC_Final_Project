from __future__ import annotations
import time
from typing import List, Optional

import numpy as np

from control import urFwdKin
from tf_frame import tf_frame
from ur_interface import UrInterface


class JointLimitError(RuntimeError):
    """Raised when a joint exceeds the configured safety limits."""
    pass


# ---------------------------------------------------------------------------
# Global configuration (adjust in one place when moving to hardware)
# ---------------------------------------------------------------------------
ROBOT_TYPE = "ur5e"
USE_VEL_CONTROL = False  # Toggle RR mode between velocity control (True) and position steps (False)

# Nominal desired push direction in base frame (projected to table plane).
# If the axis derived from the start pose points the opposite direction, we flip it to align with this.
PUSH_DIRECTION_BASE = np.array([0.0, 1.0, 0.0])

PUSH_DISTANCE = 0.03  # 3 cm contact push distance
CUBE_LEN = 0.13       # cube edge length in meters
SIDE_CLEARANCE = 0.02 # extra room when moving to the opposite side
LIFT_HEIGHT = 0.08    # clearance height for free-space motion

RR_DT = 0.08
RR_KP = 0.6
RR_ORIENT_KP = 0.4
RR_POS_TOL = 5e-3
MAX_RR_ITERS = 600

RR_SPEED_MARGIN = 0.5      # scale commands to remain comfortably within the joint-speed limit
POS_SPEED_MARGIN = 0.6     # position-control moves stay inside the joint-speed limit
FREE_SPEED_LIMIT = 0.25    # faster joint-speed fraction for free-space RR moves (capped at hardware limit)

TABLE_Z_MIN = 0.02  # keep control frame above the table plane by at least 2 cm

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
DEFAULT_HOME_Q = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

USE_PEN_TIP = False  # Set True if using a pen-tip offset instead of tool0
TIP_FRAME_NAME = "pen_tip"

# Default tool0 -> pen-tip transform; translation only so orientation stays with tool0.
TOOL0_TO_PEN_TIP = np.eye(4)
TOOL0_TO_PEN_TIP[:3, 3] = np.array([-0.049, 0.0, 0.12228])  # Adjust after measuring the real pen mount


tf_handles: dict[str, tf_frame] = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles elementwise to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def tool0_from_tip(g_tip: np.ndarray, T_tool0_tip: np.ndarray) -> np.ndarray:
    """Convert a desired tip pose to the equivalent tool0 pose."""
    return g_tip @ np.linalg.inv(T_tool0_tip)


def tip_from_tool0(g_tool0: np.ndarray, T_tool0_tip: np.ndarray) -> np.ndarray:
    """Convert a measured tool0 pose to the pen-tip pose."""
    return g_tool0 @ T_tool0_tip


def resolve_tool0_to_tip_transform(ur: UrInterface, q_ref: np.ndarray, use_pen_tip: bool = USE_PEN_TIP) -> np.ndarray:
    """Try to read tool0->tip from TF; fall back to the conservative default offset."""
    if not use_pen_tip:
        return np.eye(4)

    g_base_tool0 = urFwdKin(q_ref, ROBOT_TYPE)
    try:
        g_base_tip = ur.get_current_transformation("base_link", TIP_FRAME_NAME)
        if g_base_tip is not None and not np.allclose(g_base_tip, np.eye(4)):
            return np.linalg.inv(g_base_tool0) @ g_base_tip
        print(f"TF frame '{TIP_FRAME_NAME}' unavailable; using default tip transform.")
    except Exception as exc:
        print(f"TF lookup for '{TIP_FRAME_NAME}' failed ({exc}); using default tip transform.")
    return TOOL0_TO_PEN_TIP.copy()


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


def check_table_clearance(g_control: np.ndarray) -> None:
    """Keep the control frame (tool0 or pen tip) above the table plane."""
    if g_control[2, 3] < TABLE_Z_MIN:
        raise RuntimeError(
            f"Tool height {g_control[2,3]:.3f} m violates clearance constraint ({TABLE_Z_MIN:.3f} m)."
        )


def cartesian_target(g_src: np.ndarray, direction: np.ndarray, distance: float) -> np.ndarray:
    """Translate ``g_src`` along ``direction`` by ``distance`` while preserving orientation."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    g_target = np.array(g_src, copy=True)
    g_target[:3, 3] = g_src[:3, 3] + distance * direction
    return g_target


def translate_pose(g_src: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate ``g_src`` by ``offset`` (in meters) while keeping the same rotation."""
    g_target = np.array(g_src, copy=True)
    g_target[:3, 3] = g_src[:3, 3] + np.asarray(offset, dtype=float)
    return g_target


def push_direction_from_pose(g_start_control: np.ndarray) -> np.ndarray:
    """
    Derive a horizontal push direction from the taught start pose.

    Steps:
    1) Prefer start-frame y-axis (lateral), fallback to x-axis.
    2) Project onto table plane (zero z) and normalize.
    3) Flip sign if needed to align with configured PUSH_DIRECTION_BASE (also projected).
       This prevents "computed axis is backwards" issues without user interaction.
    """
    R = np.asarray(g_start_control[:3, :3], dtype=float)

    # Nominal base direction projected to plane (for sign disambiguation)
    base_dir = np.asarray(PUSH_DIRECTION_BASE, dtype=float).copy()
    base_dir[2] = 0.0
    if np.linalg.norm(base_dir) > 1e-9:
        base_dir = base_dir / np.linalg.norm(base_dir)
    else:
        base_dir = None

    chosen = None
    for idx in (1, 0):  # prefer y then x
        v = R[:, idx].copy()
        v[2] = 0.0
        n = np.linalg.norm(v)
        if n > 1e-6:
            v = v / n
            chosen = (idx, v)
            break

    if chosen is None:
        # Fallback: just use configured base axis
        if base_dir is None:
            raise RuntimeError("Configured PUSH_DIRECTION_BASE must be non-zero.")
        print(f"Push direction fell back to configured base axis: {base_dir}")
        return base_dir

    axis_idx, v = chosen
    axis_name = "y" if axis_idx == 1 else "x"

    flipped = False
    if base_dir is not None:
        if float(np.dot(v, base_dir)) < 0.0:
            v = -v
            flipped = True

    msg = f"Push direction uses start-frame {axis_name}-axis (projected)"
    if base_dir is not None:
        msg += f", aligned-to-base={base_dir}, flipped={flipped}"
    msg += f": {v}"
    print(msg)
    return v


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


def geometric_jacobian(q: np.ndarray, robot_type: str = ROBOT_TYPE, eps: float = 1e-4) -> np.ndarray:
    """Numerically compute the 6x6 spatial Jacobian at tool0."""
    q = np.asarray(q, dtype=float).flatten()
    g0 = urFwdKin(q, robot_type)
    p0 = g0[:3, 3]
    R0 = g0[:3, :3]
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        g_eps = urFwdKin(q + dq, robot_type)
        p_eps = g_eps[:3, 3]
        R_eps = g_eps[:3, :3]
        dp = (p_eps - p0) / eps
        dR = R_eps @ R0.T
        w = rotation_log(dR) / eps
        J[:, i] = np.hstack((w, dp))
    return J


def damped_pseudoinverse(J: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Compute a damped pseudoinverse to stay robust near singularities."""
    JT = J.T
    JJt = J @ JT
    n = JJt.shape[0]
    return JT @ np.linalg.inv(JJt + (damping ** 2) * np.eye(n))


def rr_move_to_pose(
    ur: UrInterface,
    q_init: np.ndarray,
    g_des_tool0: np.ndarray,
    robot_type: str = ROBOT_TYPE,
    T_tool0_tip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resolved-rate servoing loop that drives the tool0 pose to ``g_des_tool0``.

    Safety clearance is checked in the control frame:
    - if T_tool0_tip is provided (pen tip mode), check tip pose
    - else check tool0 pose
    """
    print("Starting resolved-rate segment...")
    q = q_init.astype(float).copy()
    use_vel_ctrl = USE_VEL_CONTROL
    if use_vel_ctrl:
        ur.activate_vel_control()
    else:
        ur.activate_pos_control()

    dt = RR_DT
    speed_limit = ur.speed_limit * RR_SPEED_MARGIN
    tip_tf = np.eye(4) if T_tool0_tip is None else np.asarray(T_tool0_tip, dtype=float)

    for _ in range(MAX_RR_ITERS):
        q_meas = ur.get_current_joints()
        q = q_meas.astype(float).copy()

        g_tool0 = urFwdKin(q, robot_type)
        g_control = g_tool0 @ tip_tf  # tool0 if tip_tf=I, tip if tip_tf is set
        check_table_clearance(g_control)
        check_joint_limits(q)

        R = g_tool0[:3, :3]
        p = g_tool0[:3, 3]
        R_des = g_des_tool0[:3, :3]
        p_des = g_des_tool0[:3, 3]
        pos_err = p_des - p
        rot_err = rotation_log(R_des @ R.T)

        if np.linalg.norm(pos_err) < RR_POS_TOL and np.linalg.norm(rot_err) < 5e-3:
            print("Resolved-rate target reached.")
            break

        twist = np.hstack((RR_ORIENT_KP * rot_err, RR_KP * pos_err))
        J = geometric_jacobian(q, robot_type)
        J_pinv = damped_pseudoinverse(J, damping=1e-3)
        qdot = J_pinv @ twist

        max_speed = np.max(np.abs(qdot))
        if max_speed > speed_limit and max_speed > 1e-9:
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


def adaptive_interp_steps(g_start: np.ndarray, g_end: np.ndarray, min_steps: int = 30, max_steps: int = 100) -> int:
    """Choose a waypoint count based on distance while respecting bounds."""
    dist = float(np.linalg.norm(g_end[:3, 3] - g_start[:3, 3]))
    steps = int(np.ceil(dist / 0.005))  # ~5 mm spacing
    steps = max(min_steps, steps)
    steps = min(max_steps, steps)
    return steps


def move_to_configuration(
    ur: UrInterface,
    q_target: np.ndarray,
    min_segment_time: float = 3.0,
    speed_margin: Optional[float] = None,
) -> None:
    """Move the arm to ``q_target`` while respecting the configured joint-speed limit."""
    q_target = np.asarray(q_target, dtype=float)
    ur.activate_pos_control()

    q_curr = ur.get_current_joints()
    diff = np.abs(wrap_to_pi(q_target - q_curr))
    move_time = float(min_segment_time)

    margin = POS_SPEED_MARGIN if speed_margin is None else max(0.1, min(float(speed_margin), 1.0))
    speed_limit = ur.speed_limit * margin

    if speed_limit > 0.0 and np.max(diff) > 0.0:
        min_time = float(np.max(diff)) / max(speed_limit, 1e-9)
        if min_time > move_time:
            move_time = min_time

    ur.move_joints(q_target, time_intervals=[move_time])
    time.sleep(move_time)


def return_home(ur: UrInterface, home_q: np.ndarray) -> None:
    """Send the robot back to the taught home configuration."""
    print("Returning to home configuration...")
    move_to_configuration(ur, home_q, min_segment_time=4.0)


def generate_push_plan(g_start_control: np.ndarray, push_dir: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Waypoints in the control frame (tool0 or tip):
    push 3cm -> lift -> move around cube to opposite side -> drop -> push back 3cm -> retreat lift.

    FIXED:
    - Lateral clearance is now (CUBE_LEN + 2*SIDE_CLEARANCE), not inflated by push distances.
    """
    push_dir = np.asarray(push_dir, dtype=float)
    n = np.linalg.norm(push_dir)
    if n < 1e-9:
        raise RuntimeError("push_dir must be non-zero for planning.")
    push_dir = push_dir / n
    opp_dir = -push_dir

    lift_vec = np.array([0.0, 0.0, LIFT_HEIGHT], dtype=float)

    g_push_1_end = cartesian_target(g_start_control, push_dir, PUSH_DISTANCE)
    g_lift_after_1 = translate_pose(g_push_1_end, lift_vec)

    # Move to the opposite side above the cube.
    lateral_clearance = CUBE_LEN + 2.0 * SIDE_CLEARANCE
    g_swing_to_opp = cartesian_target(g_lift_after_1, opp_dir, lateral_clearance)

    g_opp_contact = translate_pose(g_swing_to_opp, -lift_vec)
    g_push_2_end = cartesian_target(g_opp_contact, opp_dir, PUSH_DISTANCE)
    g_retreat = translate_pose(g_push_2_end, lift_vec)

    return [
        ("push_left_end", g_push_1_end),
        ("lift_after_left", g_lift_after_1),
        ("swing_to_right", g_swing_to_opp),
        ("right_contact", g_opp_contact),
        ("push_right_end", g_push_2_end),
        ("retreat", g_retreat),
    ]


def rr_follow_cartesian_segment(
    ur: UrInterface,
    q_start: np.ndarray,
    g_start_tool0: np.ndarray,
    g_end_tool0: np.ndarray,
    n_steps: int,
    T_tool0_tip: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track a straight-line Cartesian segment with RR control (tool0 targets)."""
    poses = interp_cartesian_segment(g_start_tool0, g_end_tool0, n_steps)
    q_curr = q_start
    g_tool0_curr = g_start_tool0
    for g_next_tool0 in poses[1:]:
        # Safety check in control frame
        if T_tool0_tip is not None:
            check_table_clearance(g_next_tool0 @ T_tool0_tip)
        else:
            check_table_clearance(g_next_tool0)

        q_curr = rr_move_to_pose(ur, q_curr, g_next_tool0, ROBOT_TYPE, T_tool0_tip=T_tool0_tip)
        g_tool0_curr = urFwdKin(q_curr, ROBOT_TYPE)
    return q_curr, g_tool0_curr


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

    try:
        q_start = teach_pose(ur, "start")
        move_to_configuration(ur, q_start)
        ur.activate_pos_control()

        q_start_actual = ur.get_current_joints()
        T_tool0_tip = resolve_tool0_to_tip_transform(ur, q_start_actual, USE_PEN_TIP)
        g_tool0_start = urFwdKin(q_start_actual, ROBOT_TYPE)
        g_start_control = tip_from_tool0(g_tool0_start, T_tool0_tip)

        publish_frame("start_pose", g_start_control)
        log_pose_details("Start", g_start_control, g_start_control)

        push_dir = push_direction_from_pose(g_start_control)
        plan = generate_push_plan(g_start_control, push_dir)
        print(f"Plan segments: {[name for name, _ in plan]}")

        q_curr = q_start_actual
        g_tool0_curr = g_tool0_start
        g_control_curr = g_start_control

        push2_actual = None

        for name, g_control_target in plan:
            check_table_clearance(g_control_target)
            publish_frame(f"{name}_des", g_control_target)

            g_tool0_target = tool0_from_tip(g_control_target, T_tool0_tip)
            n_steps = adaptive_interp_steps(g_control_curr, g_control_target)

            fast_segment = name in {"lift_after_left", "swing_to_right", "retreat"}
            prev_limit = ur.speed_limit
            if fast_segment:
                ur.speed_limit = FREE_SPEED_LIMIT

            start_time = time.time()
            try:
                q_curr, g_tool0_curr = rr_follow_cartesian_segment(
                    ur,
                    q_curr,
                    tool0_from_tip(g_control_curr, T_tool0_tip),
                    g_tool0_target,
                    n_steps,
                    T_tool0_tip=T_tool0_tip if USE_PEN_TIP else None,
                )
            finally:
                ur.speed_limit = prev_limit

            g_control_curr = tip_from_tool0(g_tool0_curr, T_tool0_tip)
            duration = time.time() - start_time
            print(f"[RR] {name}: {n_steps} waypoints, {duration:.2f}s")

            if name == "push_right_end":
                push2_actual = g_control_curr

        if push2_actual is not None:
            log_pose_details("Return push end", plan[-2][1], push2_actual)

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
    "push_direction_from_pose",
    "publish_frame",
    "cartesian_target",
    "translate_pose",
    "generate_push_plan",
    "check_table_clearance",
    "check_joint_limits",
    "tool0_from_tip",
    "tip_from_tool0",
    "resolve_tool0_to_tip_transform",
    "USE_PEN_TIP",
    "TOOL0_TO_PEN_TIP",
    "TIP_FRAME_NAME",
    "adaptive_interp_steps",
    "interp_cartesian_segment",
    "wrap_to_pi",
    "ROBOT_TYPE",
    "FREE_SPEED_LIMIT",
]
