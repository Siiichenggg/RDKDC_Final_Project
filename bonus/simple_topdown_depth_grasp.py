#!/usr/bin/env python3
"""
Super simple top-down grasp demo (no ROS):

- You manually place the gripper straight above the target (XY already aligned).
- The script reads RealSense depth at a pixel (default: image center).
- It moves the robot straight down (base Z) to a computed grasp height, then back up.

Requirements:
- `ur-rtde` (imports `rtde_control` / `rtde_receive`)
- `pyrealsense2` (RealSense SDK)
"""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


def _import_rtde():
    try:
        from ur_rtde import rtde_control, rtde_receive  # type: ignore[import-not-found]
        return rtde_control, rtde_receive
    except ModuleNotFoundError:
        import rtde_control  # type: ignore[import-not-found]
        import rtde_receive  # type: ignore[import-not-found]

        return rtde_control, rtde_receive


def _import_rs():
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: `pyrealsense2` (install Intel librealsense / pyrealsense2 first)."
        ) from exc
    return rs


def _median_depth_m(depth_u16: np.ndarray, depth_scale: float, u: int, v: int, half: int) -> Optional[float]:
    h, w = depth_u16.shape[:2]
    u0 = max(0, u - half)
    u1 = min(w, u + half + 1)
    v0 = max(0, v - half)
    v1 = min(h, v + half + 1)

    patch = depth_u16[v0:v1, u0:u1].astype(np.float32)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return float(np.median(patch) * depth_scale)


@dataclass(frozen=True)
class GraspParams:
    camera_height_above_tcp_m: float
    grasp_offset_m: float
    pregrasp_clearance_m: float
    max_down_m: float


@dataclass(frozen=True)
class GripperParams:
    speed: int
    force: int
    settle_sec: float


def _compute_down_distance(depth_m: float, params: GraspParams) -> float:
    """
    depth_m: distance from camera origin to object along camera Z (meters).

    Assumptions (simple!):
    - Camera optical axis is roughly aligned with robot base -Z.
    - Tool is pointing straight down (TCP Z aligned with base -Z).
    - Only move in base Z (keep XY + orientation unchanged).
    """
    down = depth_m - params.camera_height_above_tcp_m - params.grasp_offset_m
    return float(down)


def _read_center_depth(
    pipeline,
    align,
    depth_scale: float,
    u: int,
    v: int,
    patch_half: int,
    max_patch_half: int,
    samples: int,
    settle_sec: float,
) -> Tuple[float, Tuple[int, int]]:
    time.sleep(settle_sec)

    vals = []
    for _ in range(samples):
        frames = pipeline.wait_for_frames()
        if align is not None:
            frames = align.process(frames)
        depth = frames.get_depth_frame()
        if not depth:
            continue

        depth_u16 = np.asanyarray(depth.get_data())
        d = _median_depth_m(depth_u16, depth_scale, u, v, patch_half)
        if d is None:
            # Fallback: expand patch until we find any non-zero depth.
            for half in range(patch_half + 1, max_patch_half + 1):
                d = _median_depth_m(depth_u16, depth_scale, u, v, half)
                if d is not None:
                    break
        if d is not None and np.isfinite(d) and d > 0:
            vals.append(d)

    if not vals:
        raise RuntimeError(
            "No valid depth samples at the chosen pixel. Try `--click` to re-pick, increase `--max-patch`, "
            "or avoid reflective/black surfaces."
        )

    return float(np.median(np.array(vals, dtype=np.float32))), (u, v)


def _pick_pixel_with_depth(
    pipeline,
    align,
    depth_scale: float,
    patch_half: int,
    max_patch_half: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    window = "pick target (click; Enter accept; q/Esc cancel)"
    state: dict = {"uv": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["uv"] = (int(x), int(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color:
                continue

            bgr = np.asanyarray(color.get_data())
            uv = state["uv"]
            depth_m: Optional[float] = None
            if depth and uv is not None:
                depth_u16 = np.asanyarray(depth.get_data())
                depth_m = _median_depth_m(depth_u16, depth_scale, uv[0], uv[1], patch_half)
                if depth_m is None:
                    for half in range(patch_half + 1, max_patch_half + 1):
                        depth_m = _median_depth_m(depth_u16, depth_scale, uv[0], uv[1], half)
                        if depth_m is not None:
                            break

            if uv is not None:
                ok = depth_m is not None and np.isfinite(depth_m) and depth_m > 0
                color_bgr = (0, 200, 0) if ok else (0, 0, 255)
                cv2.drawMarker(bgr, uv, color_bgr, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                cv2.putText(
                    bgr,
                    f"u,v={uv[0]},{uv[1]}  depth={depth_m:.3f}m" if ok else f"u,v={uv[0]},{uv[1]}  depth=INVALID",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_bgr,
                    2,
                )
            cv2.imshow(window, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10) and uv is not None and depth_m is not None and np.isfinite(depth_m) and depth_m > 0:
                return uv
            if key in (27, ord("q")):
                raise RuntimeError("Canceled pixel picking.")
    finally:
        cv2.destroyWindow(window)


class Robotiq2FGripper:
    """
    Minimal Robotiq 2F gripper control via URScript (requires Robotiq URCap on the teach pendant).
    """

    def __init__(self, robot_ip: str, port: int = 30002, params: Optional[GripperParams] = None):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.params = params or GripperParams(speed=50, force=50, settle_sec=0.6)

    def _send_script(self, script: str) -> None:
        data = script.encode("utf-8")
        with socket.create_connection((self.robot_ip, self.port), timeout=2.0) as sock:
            sock.sendall(data)

    def _wrap(self, body: str) -> str:
        return "def codex_gripper():\n" + body + "\nend\n"

    def activate(self) -> None:
        self._send_script(self._wrap("  rq_activate()\n  sleep(0.5)"))

    def open(self) -> None:
        body = (
            f"  rq_set_speed({int(self.params.speed)})\n"
            f"  rq_set_force({int(self.params.force)})\n"
            "  rq_open()\n"
            f"  sleep({float(self.params.settle_sec)})"
        )
        self._send_script(self._wrap(body))

    def close(self) -> None:
        body = (
            f"  rq_set_speed({int(self.params.speed)})\n"
            f"  rq_set_force({int(self.params.force)})\n"
            "  rq_close()\n"
            f"  sleep({float(self.params.settle_sec)})"
        )
        self._send_script(self._wrap(body))


@dataclass(frozen=True)
class ToolIOPattern:
    do0: int
    do1: int


class ToolIOGripper:
    """
    Generic gripper control via UR tool digital outputs.

    Many grippers use tool DO0/DO1 as open/close signals. You must know your wiring/pattern.
    Patterns use -1 to mean "leave unchanged".
    """

    def __init__(self, robot_ip: str, open_pattern: ToolIOPattern, close_pattern: ToolIOPattern, port: int = 30002):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.open_pattern = open_pattern
        self.close_pattern = close_pattern

    def _send_script(self, script: str) -> None:
        data = script.encode("utf-8")
        with socket.create_connection((self.robot_ip, self.port), timeout=2.0) as sock:
            sock.sendall(data)

    def _wrap(self, body: str) -> str:
        return "def codex_toolio():\n" + body + "\nend\n"

    def _pattern_body(self, pat: ToolIOPattern, settle_sec: float) -> str:
        lines = []
        if pat.do0 in (0, 1):
            lines.append(f"  set_tool_digital_out(0, {str(bool(pat.do0)).lower()})")
        if pat.do1 in (0, 1):
            lines.append(f"  set_tool_digital_out(1, {str(bool(pat.do1)).lower()})")
        lines.append(f"  sleep({float(settle_sec)})")
        return "\n".join(lines)

    def open(self, settle_sec: float = 0.4) -> None:
        self._send_script(self._wrap(self._pattern_body(self.open_pattern, settle_sec)))

    def close(self, settle_sec: float = 0.4) -> None:
        self._send_script(self._wrap(self._pattern_body(self.close_pattern, settle_sec)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple top-down depth grasp (RealSense + UR RTDE).")
    ap.add_argument("--ip", default="172.22.22.2", help="UR robot IP.")
    ap.add_argument("--speed", type=float, default=0.10, help="Linear speed for moveL (m/s).")
    ap.add_argument("--accel", type=float, default=0.15, help="Linear acceleration for moveL (m/s^2).")
    ap.add_argument("--execute", action="store_true", help="Actually move the robot (default: ask for confirmation).")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt (use with --execute).")

    ap.add_argument(
        "--gripper",
        choices=["none", "robotiq2f", "toolio"],
        default="none",
        help="Optional gripper control (requires appropriate hardware/URCap).",
    )
    ap.add_argument("--gripper-speed", type=int, default=50, help="Robotiq speed (0-255).")
    ap.add_argument("--gripper-force", type=int, default=50, help="Robotiq force (0-255).")
    ap.add_argument("--gripper-settle", type=float, default=0.6, help="Seconds to wait after open/close.")
    ap.add_argument("--toolio-open-do0", type=int, default=-1, help="ToolIO open pattern DO0: 0/1, or -1 unchanged.")
    ap.add_argument("--toolio-open-do1", type=int, default=-1, help="ToolIO open pattern DO1: 0/1, or -1 unchanged.")
    ap.add_argument("--toolio-close-do0", type=int, default=-1, help="ToolIO close pattern DO0: 0/1, or -1 unchanged.")
    ap.add_argument("--toolio-close-do1", type=int, default=-1, help="ToolIO close pattern DO1: 0/1, or -1 unchanged.")
    ap.add_argument(
        "--open-at-pregrasp",
        action="store_true",
        help="Open gripper at the pregrasp pose (recommended).",
    )
    ap.add_argument(
        "--close-at-grasp",
        action="store_true",
        help="Close gripper at the grasp pose (optional).",
    )

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--click", action="store_true", help="Pick (u,v) by clicking on the color image.")
    ap.add_argument("--u", type=int, default=None, help="Pixel u (default: image center).")
    ap.add_argument("--v", type=int, default=None, help="Pixel v (default: image center).")
    ap.add_argument("--patch", type=int, default=7, help="Patch size (odd), median depth in patch.")
    ap.add_argument(
        "--max-patch",
        type=int,
        default=31,
        help="Max patch size (odd) used as fallback when depth is invalid in the small patch.",
    )
    ap.add_argument("--samples", type=int, default=15, help="Number of frames to sample depth from.")
    ap.add_argument("--settle", type=float, default=0.4, help="Seconds to wait before sampling depth.")

    ap.add_argument(
        "--camera-height-above-tcp",
        type=float,
        default=0.12,
        help="Meters from TCP (gripper tip) up to camera origin along +Z (tune!).",
    )
    ap.add_argument(
        "--grasp-offset",
        type=float,
        default=0.015,
        help="Extra meters to stop above the object (avoid smashing).",
    )
    ap.add_argument(
        "--pregrasp",
        type=float,
        default=0.08,
        help="Meters above grasp height for approach/retreat.",
    )
    ap.add_argument("--dwell", type=float, default=0.3, help="Seconds to pause at grasp height.")
    ap.add_argument("--max-down", type=float, default=0.35, help="Safety: max allowed downward travel (m).")
    ap.add_argument(
        "--expected-down",
        type=float,
        default=None,
        help="If you know the robot is ~this many meters above the object, prints a suggested `--camera-height-above-tcp`.",
    )
    args = ap.parse_args()

    if args.patch <= 0 or args.patch % 2 != 1:
        raise SystemExit("--patch must be an odd positive integer (e.g. 7, 9, 11).")
    if args.max_patch <= 0 or args.max_patch % 2 != 1:
        raise SystemExit("--max-patch must be an odd positive integer (e.g. 31, 51).")
    if args.max_patch < args.patch:
        raise SystemExit("--max-patch must be >= --patch.")

    u = args.u if args.u is not None else (args.width // 2)
    v = args.v if args.v is not None else (args.height // 2)
    patch_half = args.patch // 2
    max_patch_half = args.max_patch // 2
    if not args.click and not (0 <= u < args.width and 0 <= v < args.height):
        raise SystemExit(f"Pixel (u,v)=({u},{v}) out of bounds for {args.width}x{args.height}.")

    rs = _import_rs()
    rtde_control, rtde_receive = _import_rtde()

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(cfg)

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        align = rs.align(rs.stream.color)

        while True:
            if args.click:
                u, v = _pick_pixel_with_depth(
                    pipeline=pipeline,
                    align=align,
                    depth_scale=depth_scale,
                    patch_half=patch_half,
                    max_patch_half=max_patch_half,
                    width=args.width,
                    height=args.height,
                )

            try:
                depth_m, (uu, vv) = _read_center_depth(
                    pipeline=pipeline,
                    align=align,
                    depth_scale=depth_scale,
                    u=u,
                    v=v,
                    patch_half=patch_half,
                    max_patch_half=max_patch_half,
                    samples=args.samples,
                    settle_sec=args.settle,
                )
            except RuntimeError as exc:
                print(f"unreachable: invalid depth ({exc})")
                print("unreachable: depth is invalid; continuing (move camera / re-pick pixel / adjust lighting).")
                continue

            params = GraspParams(
                camera_height_above_tcp_m=float(args.camera_height_above_tcp),
                grasp_offset_m=float(args.grasp_offset),
                pregrasp_clearance_m=float(args.pregrasp),
                max_down_m=float(args.max_down),
            )
            down = _compute_down_distance(depth_m, params)

            print(f"Depth@({uu},{vv})={depth_m:.4f} m, computed down={down:.4f} m")
            if args.expected_down is not None:
                expected_down = float(args.expected_down)
                suggested_h = depth_m - expected_down - params.grasp_offset_m
                print(
                    "Suggested `--camera-height-above-tcp` (based on --expected-down): "
                    f"{suggested_h:.4f} m  [= depth - expected_down - grasp_offset]"
                )

            if down <= 0:
                print("unreachable: computed down <= 0")
                print("unreachable: computed down <= 0; continuing (wrong pixel or bad height parameters).")
                continue

            max_down_eps = 1e-3
            if down > params.max_down_m + max_down_eps:
                print(
                    f"unreachable: computed down too large ({down:.3f} m > --max-down {params.max_down_m:.3f})"
                )
                print(
                    "unreachable: down is too large; continuing (move closer / re-pick pixel / increase --camera-height-above-tcp)."
                )
                continue

            break

        print(f"Connecting to robot {args.ip} ...")
        rc = rtde_control.RTDEControlInterface(args.ip)
        rr = rtde_receive.RTDEReceiveInterface(args.ip)
        try:
            tcp = rr.getActualTCPPose()
            x, y, z, rx, ry, rz = [float(v) for v in tcp]
            z_grasp = z - down
            pre = [x, y, z_grasp + params.pregrasp_clearance_m, rx, ry, rz]
            grasp = [x, y, z_grasp, rx, ry, rz]

            print("Current TCP:", [round(v, 4) for v in tcp])
            print("Pregrasp:", [round(v, 4) for v in pre])
            print("Grasp:", [round(v, 4) for v in grasp])

            if not args.execute:
                answer = input("Execute robot motion now? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Canceled (no motion). Use --execute to skip this prompt.")
                    return 0
            elif not args.yes:
                answer = input("Confirm executing robot motion? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Canceled (no motion).")
                    return 0

            print("Executing: moveL(pre) -> moveL(grasp) -> wait -> moveL(pre)")
            gripper = None
            if args.gripper == "robotiq2f":
                gripper = Robotiq2FGripper(
                    robot_ip=str(args.ip),
                    params=GripperParams(
                        speed=int(args.gripper_speed),
                        force=int(args.gripper_force),
                        settle_sec=float(args.gripper_settle),
                    ),
                )
                try:
                    gripper.activate()
                except Exception:
                    pass
            elif args.gripper == "toolio":
                gripper = ToolIOGripper(
                    robot_ip=str(args.ip),
                    open_pattern=ToolIOPattern(int(args.toolio_open_do0), int(args.toolio_open_do1)),
                    close_pattern=ToolIOPattern(int(args.toolio_close_do0), int(args.toolio_close_do1)),
                )

            rc.moveL(pre, speed=float(args.speed), acceleration=float(args.accel))
            if gripper is not None and args.open_at_pregrasp:
                gripper.open()
            rc.moveL(grasp, speed=max(0.03, float(args.speed) * 0.5), acceleration=max(0.05, float(args.accel) * 0.5))
            if gripper is not None and args.close_at_grasp:
                gripper.close()
            time.sleep(float(args.dwell))
            rc.moveL(pre, speed=float(args.speed), acceleration=float(args.accel))
            return 0
        finally:
            try:
                rc.stopScript()
            except Exception:
                pass
            for obj in (locals().get("rc"), locals().get("rr")):
                try:
                    if obj is not None and hasattr(obj, "disconnect"):
                        obj.disconnect()
                except Exception:
                    pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
