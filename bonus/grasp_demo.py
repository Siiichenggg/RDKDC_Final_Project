import math
import socket
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ------------------------------- Utility ------------------------------------ #


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    """UR pose [x,y,z,rx,ry,rz] to 4x4 homogeneous matrix."""
    rvec = np.asarray(pose[3:6])
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pose[:3])
    return T


def matrix_to_pose(T: np.ndarray) -> List[float]:
    """4x4 matrix to UR pose [x,y,z,rx,ry,rz]."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    pose = list(T[:3, 3].reshape(-1)) + list(rvec.reshape(-1))
    return pose


def create_topdown_orientation(yaw: float = 0.0) -> np.ndarray:
    """Return 3x1 Rodrigues vector for z-axis pointing down with yaw around Z."""
    Rz, _ = cv2.Rodrigues(np.array([0, 0, yaw], dtype=float))
    Rx_pi, _ = cv2.Rodrigues(np.array([math.pi, 0, 0], dtype=float))
    R = Rz @ Rx_pi
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def ensure_workspace(p: np.ndarray, bounds: dict) -> bool:
    x_ok = bounds["x"][0] <= p[0] <= bounds["x"][1]
    y_ok = bounds["y"][0] <= p[1] <= bounds["y"][1]
    z_ok = bounds["z"][0] <= p[2] <= bounds["z"][1]
    return x_ok and y_ok and z_ok


# ----------------------------- RealSense ------------------------------------ #


class RealSenseCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = None
        self.depth_scale = None
        self.start()

    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to acquire frames")

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        color_image = np.asanyarray(color_frame.get_data())
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
        }
        return color_image, depth_image, intrinsics

    def __del__(self):
        self.stop()


# --------------------------- Sponge Detection ------------------------------- #


def detect_square_sponge(
    bgr: np.ndarray,
    hsv_lower: List[int],
    hsv_upper: List[int],
    min_area: float = 400.0,
) -> Tuple[bool, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if 0.75 <= ratio <= 1.25:
            if best is None or area > cv2.contourArea(best):
                best = approx

    if best is None:
        return False, None, None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return False, None, None
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return True, (u, v), best


def pixel_to_cam(
    u: int,
    v: int,
    depth: np.ndarray,
    intr: dict,
    contour: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    Z = depth[v, u]
    if Z <= 0 and contour is not None:
        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        valid = depth[(mask == 255) & (depth > 0)]
        if valid.size == 0:
            return None
        Z = np.median(valid)

    if Z <= 0:
        return None

    X = (u - intr["cx"]) * Z / intr["fx"]
    Y = (v - intr["cy"]) * Z / intr["fy"]
    return np.array([X, Y, Z, 1.0], dtype=float)


# --------------------------- Robot + Gripper -------------------------------- #


class RobotController:
    def __init__(self, robot_ip: str, speed: float, accel: float):
        from ur_rtde import rtde_control, rtde_receive

        self.speed = speed
        self.accel = accel
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    def movej(self, q: List[float], speed: Optional[float] = None, accel: Optional[float] = None) -> None:
        spd = speed if speed is not None else self.speed
        acc = accel if accel is not None else self.accel
        self.rtde_c.moveJ(q, spd, acc)

    def movel(self, pose: List[float], speed: Optional[float] = None, accel: Optional[float] = None) -> None:
        spd = speed if speed is not None else self.speed
        acc = accel if accel is not None else self.accel
        self.rtde_c.moveL(pose, spd, acc)

    def get_base_T_tcp(self) -> np.ndarray:
        pose = self.rtde_r.getActualTCPPose()
        return pose_to_matrix(pose)

    def go_home(self, q_home: List[float]) -> None:
        self.movej(q_home)

    def stop(self) -> None:
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass

    def __del__(self):
        self.stop()


class Robotiq2FGripper:
    def __init__(self, robot_ip: str, force: int = 50, speed: int = 50):
        self.robot_ip = robot_ip
        self.force = force
        self.speed = speed
        self.port = 30002

    def _send_script(self, script: str) -> None:
        try:
            with socket.create_connection((self.robot_ip, self.port), timeout=2.0) as sock:
                sock.sendall(script.encode("utf-8"))
        except Exception as exc:
            print(f"Gripper script failed: {exc}")

    def activate(self) -> None:
        script = "rq_activate()\n"
        self._send_script(script)
        time.sleep(2.0)

    def open(self) -> None:
        script = (
            "rq_set_force({})\n"
            "rq_set_speed({})\n"
            "rq_open()\n"
        ).format(self.force, self.speed)
        self._send_script(script)

    def close(self, force: Optional[int] = None, speed: Optional[int] = None) -> None:
        f = force if force is not None else self.force
        s = speed if speed is not None else self.speed
        script = (
            "rq_set_force({})\n"
            "rq_set_speed({})\n"
            "rq_close()\n"
        ).format(f, s)
        self._send_script(script)


# --------------------------- Grasp Planning --------------------------------- #


def execute_grasp(
    robot: RobotController,
    gripper: Robotiq2FGripper,
    p_base: np.ndarray,
    table_z: float,
    sponge_height: float,
    pregrasp_offset: float,
    workspace: dict,
    q_home: List[float],
    grasp_speed: float,
    approach_speed: float,
) -> None:
    if not ensure_workspace(p_base, workspace):
        print("Grasp aborted: target outside workspace.")
        return

    grasp_z = table_z + sponge_height * 0.5
    pre_z = grasp_z + pregrasp_offset
    rvec = create_topdown_orientation(0.0)
    pre_pose = [p_base[0], p_base[1], pre_z, rvec[0], rvec[1], rvec[2]]
    grasp_pose = [p_base[0], p_base[1], grasp_z, rvec[0], rvec[1], rvec[2]]

    print("Starting grasp sequence.")
    robot.go_home(q_home)
    gripper.open()
    robot.movel(pre_pose, speed=approach_speed)
    robot.movel(grasp_pose, speed=grasp_speed)
    gripper.close(force=20, speed=20)
    robot.movel(pre_pose, speed=approach_speed)  # lift back up
    robot.go_home(q_home)  # park
    print("Grasp sequence finished.")


# ------------------------------- Main --------------------------------------- #


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_extrinsics(path: str) -> np.ndarray:
    try:
        data = load_yaml(path)
        T = np.array(data["T_tcp_cam"], dtype=float)
        if T.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 4x4")
        return T
    except Exception as exc:
        print(f"Failed to load {path}: {exc}. Using identity.")
        return np.eye(4)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (T @ p.reshape(4, 1)).reshape(4)


def draw_overlay(
    image: np.ndarray,
    detection_ok: bool,
    center: Optional[Tuple[int, int]],
    contour: Optional[np.ndarray],
    p_cam: Optional[np.ndarray],
    p_base: Optional[np.ndarray],
) -> np.ndarray:
    canvas = image.copy()
    if detection_ok and contour is not None and center is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
        cv2.circle(canvas, center, 5, (0, 0, 255), -1)
    if p_cam is not None:
        txt = f"Cam XYZ: {p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f} m"
        cv2.putText(canvas, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if p_base is not None:
        txt = f"Base XYZ: {p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f} m"
        cv2.putText(canvas, txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "Keys: g=grasp o=open p=close q=quit",
        (10, canvas.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )
    return canvas


def main():
    cfg = load_yaml("config.yaml")
    camera = RealSenseCamera(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"])
    robot = RobotController(cfg["robot"]["ip"], cfg["robot"]["speed"], cfg["robot"]["accel"])
    gripper = Robotiq2FGripper(cfg["robot"]["ip"])
    gripper.activate()

    T_tcp_cam = load_extrinsics("T_tcp_cam.yaml")
    while True:
        try:
            rgb, depth, intrinsics = camera.get_rgbd()
        except Exception as exc:
            print(f"Camera error: {exc}")
            continue

        ok, center, contour = detect_square_sponge(
            rgb,
            cfg["vision"]["hsv_lower"],
            cfg["vision"]["hsv_upper"],
            cfg["vision"]["min_area"],
        )
        p_cam = None
        p_base = None
        if ok and center is not None:
            p_cam = pixel_to_cam(center[0], center[1], depth, intrinsics, contour)
            if p_cam is not None:
                T_base_tcp = robot.get_base_T_tcp()
                T_base_cam = T_base_tcp @ T_tcp_cam
                p_base = transform_point(T_base_cam, p_cam)

        overlay = draw_overlay(rgb, ok, center, contour, p_cam, p_base)
        cv2.imshow("grasp_demo", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("o"):
            gripper.open()
        if key == ord("p"):
            gripper.close(force=20, speed=20)
        if key == ord("g"):
            if p_base is None:
                print("No valid target for grasp.")
                continue
            execute_grasp(
                robot,
                gripper,
                p_base,
                cfg["task"]["table_z"],
                cfg["task"]["sponge_height"],
                cfg["task"]["pregrasp_offset"],
                cfg["task"]["workspace"],
                cfg["robot"]["home_joints"],
                cfg["robot"]["grasp_speed"],
                cfg["robot"]["approach_speed"],
            )

    cv2.destroyAllWindows()
    camera.stop()
    robot.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
