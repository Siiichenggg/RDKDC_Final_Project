# UR5e + Robotiq 2F + RealSense D435 Top-Down Grasp Demo

最小可跑版本：传统视觉 + 几何，无需 ROS/深度学习。硬件：UR5e，Robotiq 2F（URCap 已安装），RealSense D435 眼在手配置，桌面高度已知。

## 依赖
- Ubuntu，Python 3.8+
- `pip install numpy opencv-python pyyaml pyrealsense2 scipy ur-rtde`
- 机器人与相机网络可达；RealSense 固件与 librealsense 安装完成。

## 配置
1. 按需修改 `config.yaml`：
   - `robot.ip`：UR 控制器 IP。
   - `home_joints`：安全 Home 位姿（弧度）。
   - `vision.hsv_lower/upper`：海绵 HSV 阈值。
   - `task.table_z`/`workspace`：桌面高度与工作空间边界。
   - `task.place_pose`：放置位姿（Top-Down）。
2. `T_tcp_cam.yaml`：相机到 TCP 外参，默认单位阵；若已标定可覆盖。

## 运行抓取 Demo
1. 确认机器人周围安全，桌面无障碍。
2. `python grasp_demo.py`
3. 窗口按键：
   - `g`：执行一次抓取（检测 → 3D 回投 → 坐标变换 → Top-Down 抓取）
   - `o`：夹爪打开
   - `p`：夹爪关闭
   - `q`：退出

## 控制流程
- RealSense 获取对齐 RGB-D 与内参（米）。
- HSV + 轮廓近似检测最大正方形海绵，取中心像素。
- 深度中值回投为相机系 3D 点。
- `T_base_cam = T_base_tcp @ T_tcp_cam` → 转到基坐标。
- 生成 Top-Down 抓取位姿（预抓取 + 抓取 + 放置）。
- RTDE MoveL/MoveJ 轨迹；Robotiq URScript 开/闭合（软力）。

## 安全提醒
- 确保 `workspace`、`table_z` 与 `home_joints` 经现场验证。
- 深度无效或超界会自动放弃本次抓取。
- 使用前在仿真或低速模式验证轨迹。
