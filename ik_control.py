from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple
from collections import Counter

import numpy as np

import rr_control as rr
from control import urFwdKin
from IK_utils import urInvKin


class IKNoSolutionError(RuntimeError):
    """Raised when IK fails or no safe solution can be selected."""


def _clearance_pose_from_tool0(g_tool0: np.ndarray) -> np.ndarray:
    """Return the pose to use for clearance checks (tip if enabled, else tool0)."""

    return rr.tip_from_tool0(g_tool0) if rr.USE_PEN_TIP else g_tool0


def _wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""

    theta = np.asarray(theta, dtype=float)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _select_best_solution(
    theta_6xm: np.ndarray,
    q_prev: np.ndarray,
    robot_type: str,
    debug: bool = False,
    weights: Optional[Sequence[float]] = None,
    wrist_singularity_margin: float = 0.12,
    clearance_fallback_margin: float = 0.005,
) -> Tuple[np.ndarray, int]:
    """Pick the IK solution closest to q_prev that also passes safety checks.

    The cost prioritizes small wrapped joint changes (weighted L1) and
    penalizes solutions that drive wrist pitch (joint 5) near singularity.
    If all candidates violate table clearance by a tiny margin, fall back
    to the least-cost one that stays within `clearance_fallback_margin`.
    """

    theta_6xm = np.asarray(theta_6xm, dtype=float)
    q_prev = np.asarray(q_prev, dtype=float).flatten()
    if q_prev.size != 6:
        raise ValueError("q_prev must be a 6-element vector.")

    if theta_6xm.size == 0 or theta_6xm.shape[0] != 6:
        raise IKNoSolutionError("IK returned no candidate solutions.")

    if weights is None:
        weights_arr = np.ones(6, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float).flatten()
        if weights_arr.size != 6 or np.any(weights_arr < 0):
            raise ValueError("weights must be a 6-element nonnegative vector.")

    best_cost = float("inf")
    best_max_step = float("inf")
    best_q: Optional[np.ndarray] = None
    best_idx = -1
    rejected_count = 0
    all_costs = []
    rejected_info = []

    for col in range(theta_6xm.shape[1]):
        q_raw = theta_6xm[:, col].flatten()
        if q_raw.size != 6 or not np.all(np.isfinite(q_raw)):
            all_costs.append((col, float("inf"), "invalid"))
            continue

        dq = _wrap_to_pi(q_raw - q_prev)
        q_cand = q_prev + dq  # unwrap to remain closest to q_prev (2π-invariant)

        weighted_step = weights_arr * np.abs(dq)
        l1_cost = float(np.sum(weighted_step))
        max_step = float(np.max(np.abs(dq)))

        # Penalize wrist-pitch singularity (joint 5 close to 0 where sin(theta5) ~ 0)
        wrist_penalty = 0.0
        sin_t5 = abs(np.sin(q_cand[4]))
        if sin_t5 < wrist_singularity_margin:
            wrist_penalty = (wrist_singularity_margin - sin_t5) / max(wrist_singularity_margin, 1e-6)

        cost = l1_cost + 0.2 * max_step + wrist_penalty
        all_costs.append((col, cost, "pending"))

        g_tool0 = urFwdKin(q_cand, robot_type)
        g_clear = _clearance_pose_from_tool0(g_tool0)

        try:
            rr.check_joint_limits(q_cand)
            rr.check_table_clearance(g_clear)
        except Exception as e:
            rejected_count += 1
            all_costs[-1] = (col, float("inf"), f"rejected({type(e).__name__})")
            rejected_info.append(
                {
                    "col": col,
                    "cost": cost,
                    "max_step": max_step,
                    "reason": type(e).__name__,
                    "clearance": float(g_clear[2, 3]) if g_clear is not None else None,
                    "clearance_frame": "tip" if rr.USE_PEN_TIP else "tool0",
                    "q": q_cand,
                }
            )
            continue

        all_costs[-1] = (col, cost, "valid")
        if cost < best_cost or (abs(cost - best_cost) < 1e-9 and max_step < best_max_step):
            best_cost = cost
            best_max_step = max_step
            best_q = q_cand
            best_idx = col

    if debug:
        print(f"[IK_SELECT] Total solutions: {theta_6xm.shape[1]}, Rejected: {rejected_count}, Valid: {theta_6xm.shape[1] - rejected_count}")
        sorted_costs = sorted(all_costs, key=lambda x: x[1])
        for idx, cost, status in sorted_costs[:3]:
            print(f"  Solution {idx}: cost={cost:.4f}, status={status}")
        if best_q is not None:
            print(f"  Selected: solution {best_idx} with cost={best_cost:.4f} rad")
            max_joint_move = np.max(np.abs(_wrap_to_pi(best_q - q_prev)))
            print(f"  Max single-joint displacement: {np.rad2deg(max_joint_move):.2f}°")

    if best_q is None:
        # Fallback: allow tiny clearance violations if joint limits are OK.
        clearance_thresh = rr.TABLE_Z_MIN - clearance_fallback_margin
        clearance_candidates = [
            info for info in rejected_info
            if info["reason"] != "JointLimitError"
            and info.get("clearance") is not None
            and info["clearance"] >= clearance_thresh
        ]

        if clearance_candidates:
            clearance_candidates.sort(key=lambda x: (x["cost"], x["max_step"]))
            pick = clearance_candidates[0]
            best_q = pick["q"]
            best_idx = pick["col"]
            if debug:
                print(
                    f"[IK_SELECT] No strictly safe solution. "
                    f"Falling back to idx={best_idx} (clearance {pick['clearance']:.3f} m)"
                )
            return best_q, best_idx

        reason_summary = Counter([info["reason"] for info in rejected_info])
        summary_str = ", ".join(f"{k}:{v}" for k, v in reason_summary.items()) if reason_summary else "none"
        raise IKNoSolutionError(
            f"All IK solutions failed safety checks (joint limits / table clearance). Reasons: {summary_str}"
        )

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
    max_retries: int = 3,
    position_tolerance: float = 0.05,
) -> np.ndarray:
    """Follow a list of tool0 waypoints using IK with closed-loop feedback.

    Args:
        ur: Robot interface
        waypoints_tool0: List of SE(3) poses to follow
        q_start: Initial joint configuration
        robot_type: Robot model type
        base_dt: Base time step for motion
        cart_step: Cartesian step size
        speed_margin: Speed scaling factor
        debug: Enable debug output
        max_retries: Maximum retry attempts per waypoint if position error exceeds tolerance
        position_tolerance: Joint space position tolerance (radians)

    Returns:
        Final joint configuration
    """

    if len(waypoints_tool0) == 0:
        raise ValueError("waypoints_tool0 must be non-empty.")

    ur.activate_pos_control()
    q_current = np.asarray(q_start, dtype=float).flatten()
    if q_current.size != 6:
        raise ValueError("q_start must be a 6-element vector.")

    rr.check_joint_limits(q_current)
    rr.check_table_clearance(_clearance_pose_from_tool0(urFwdKin(q_current, robot_type)))

    # Execute waypoints one by one with closed-loop feedback
    for k, g_des_tool0 in enumerate(waypoints_tool0):
        g_des_tool0 = np.asarray(g_des_tool0, dtype=float)
        if g_des_tool0.shape != (4, 4):
            raise ValueError("Each waypoint must be a 4x4 SE(3) matrix.")

        retry_count = 0
        reached = False

        while retry_count <= max_retries and not reached:
            # Always compute IK from the current actual position
            if retry_count > 0:
                q_current = ur.get_current_joints()

            # Compute IK solution
            theta = urInvKin(g_des_tool0, robot_type)

            # Enable detailed debug for first waypoint to diagnose "large rotation" issue
            debug_ik_select = debug and retry_count == 0 and k == 0
            q_next, idx = _select_best_solution(theta, q_current, robot_type, debug=debug_ik_select)

            if debug and retry_count == 0 and (k == 0 or k == len(waypoints_tool0) - 1):
                print(f"[IK] waypoint {k+1}/{len(waypoints_tool0)} selected solution idx={idx}")

            # Compute time step for this segment
            dt_exec = _compute_dt_for_joint_path(ur.speed_limit, [q_current, q_next], base_dt, speed_margin)

            # Execute this waypoint
            ur.move_joints(q_next, time_intervals=[dt_exec])
            time.sleep(dt_exec)

            # Read actual position after execution
            q_actual = ur.get_current_joints()

            # Compute joint space error
            joint_error = np.linalg.norm(q_actual - q_next)

            # Check if waypoint is reached
            if joint_error < position_tolerance:
                reached = True
                if debug and retry_count > 0:
                    print(f"[IK] waypoint {k+1}/{len(waypoints_tool0)} reached after {retry_count} retries")
            else:
                retry_count += 1
                if debug or retry_count > max_retries:
                    status = "retrying" if retry_count <= max_retries else "continuing"
                    print(f"[IK] waypoint {k+1}/{len(waypoints_tool0)} error {joint_error:.4f} rad (tolerance {position_tolerance:.4f}), {status}...")

            # Update current position for next iteration
            q_current = q_actual

        if not reached and debug:
            print(f"[IK] Warning: waypoint {k+1}/{len(waypoints_tool0)} not fully reached after {max_retries} retries, proceeding to next waypoint")

        if debug and (k == 0 or k == len(waypoints_tool0) - 1 or retry_count > 0):
            joint_error = np.linalg.norm(q_current - q_next)
            print(f"[IK] waypoint {k+1}/{len(waypoints_tool0)} final error={joint_error:.4f} rad")

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
    max_retries: int = 3,
    position_tolerance: float = 0.05,
) -> np.ndarray:
    """Plan a straight Cartesian segment to g_des_tool0 and execute it using IK with closed-loop feedback."""

    print("Starting IK segment...")
    q_meas = ur.get_current_joints()
    q_start = np.asarray(q_meas if q_meas is not None else q_init, dtype=float).flatten()

    g_start = urFwdKin(q_start, robot_type)
    rr.check_table_clearance(_clearance_pose_from_tool0(g_start))
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
        max_retries=max_retries,
        position_tolerance=position_tolerance,
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

    def safe_ik_move(q_from: np.ndarray, g_target: np.ndarray, is_contact: bool = False) -> np.ndarray:
        """Execute IK move with appropriate settings for contact vs free-space motion.

        Args:
            q_from: Starting joint configuration
            g_target: Target pose
            is_contact: True for contact motion (pushing), False for free-space motion
        """
        try:
            # Use more retries and tighter tolerance for contact motion (pushing)
            max_retries = 5 if is_contact else 2
            position_tolerance = 0.08 if is_contact else 0.05

            return ik_move_to_pose(
                ur,
                q_from,
                g_target,
                robot_type=rr.ROBOT_TYPE,
                base_dt=rr.RR_DT,
                cart_step=rr.RR_POS_TOL,
                speed_margin=rr.POS_SPEED_MARGIN,
                debug=True,
                max_retries=max_retries,
                position_tolerance=position_tolerance,
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
        print("[PUSH] Starting first push (contact motion)...")
        print(f"[DEBUG] Current joint angles (deg): {np.rad2deg(q_start_actual)}")
        q_curr = safe_ik_move(q_start_actual, g_end1, is_contact=True)

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
        print("[PUSH] Starting second push (contact motion)...")
        q_end_final = safe_ik_move(q_curr, g_end2, is_contact=True)

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
