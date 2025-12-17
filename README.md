# UR5/UR5e Robot Control Project

A comprehensive control system for Universal Robots UR5/UR5e manipulators using ROS2 and RTDE interfaces.

## Features

- **Forward/Inverse Kinematics** — DH-parameter based analytical solutions
- **Resolved-Rate Control** — Damped least-squares Jacobian-based motion
- **Push-and-Place Task** — 5-phase object manipulation demo
- **Depth-Guided Grasping** — RealSense camera top-down grasp (bonus)

## Project Structure

```
├── ROS/                        # ROS2 interface
│   ├── ur_interface.m          # MATLAB ROS2 interface class
│   ├── urFwdKin.m              # Forward kinematics
│   ├── urInvKin.m              # Inverse kinematics (8 solutions)
│   ├── five_phase_push_manipulation.m
│   └── rr_control.py           # Resolved-rate controller
│
├── RTDE/                       # RTDE direct communication
│   ├── ur_rtde_interface.m     # MATLAB RTDE interface class
│   ├── rtde_urFwdKin.m
│   ├── rtde_urBodyJacobian.m
│   ├── rtde_rr_move_to_pose.m  # RR control motion
│   ├── rtde_push_and_place_task.m
│   └── Matlab/helper_function/ # Math utilities
│
└── bonus/
    └── simple_topdown_depth_grasp.py
```

## Requirements

**ROS2 Interface:**
- MATLAB R2022a+ with ROS Toolbox
- ROS2 Humble/Iron

**RTDE Interface:**
- MATLAB R2022a+ with Robotics System Toolbox
- `urRTDEClient` toolbox

**Bonus (Grasping):**
```bash
pip install ur-rtde pyrealsense2 opencv-python numpy
```

## Usage

### ROS2 Interface

```matlab
robot = ur_interface();
robot.move_joints(robot.home, 5);

% Run push manipulation task
error_cm = five_phase_push_manipulation(robot, 'ur5e');
```

### RTDE Interface

```matlab
% Simulation
ur = ur_rtde_interface("sim", "192.168.1.100");

% Real robot
ur = ur_rtde_interface("real");

% Read joints
q = ur.get_current_joints();

% Move to target
ur.move_joints(q_target, 3);
```

### Resolved-Rate Control

```matlab
cfg.robotType = 'ur5e';
cfg.dt = 0.1;
cfg.K = 1.5;
cfg.posTol = 0.002;      % 2mm
cfg.rotTol = 0.02;       % rad
cfg.dampLambda = 0.05;
cfg.maxSteps = 500;

result = rtde_rr_move_to_pose(ur, g_desired, cfg, "segment_name");
```

### Depth-Guided Grasping

```bash
python simple_topdown_depth_grasp.py --ip 192.168.1.100 --click --execute
```

## Push-and-Place Task

5-phase manipulation sequence:

```
Phase 1: Push forward (3cm)
Phase 2: Lift up (9cm)
Phase 3: Translate laterally (16cm)
Phase 4: Lower to contact
Phase 5: Push back (3cm)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PUSH_DISTANCE_M` | 0.03 | Push distance (m) |
| `LIFT_HEIGHT_M` | 0.09 | Lift clearance (m) |
| `LATERAL_OFFSET_M` | 0.16 | Lateral travel (m) |
| `MOTION_TIME_SEC` | 3.0 | Motion duration (s) |

## Safety

- Test in simulation before running on real hardware
- Keep workspace clear of obstacles
- Ensure emergency stop is accessible
- Verify joint limits before execution

## Contributors

- Sicheng Lu
- Jiafeng Gu
- Yuxin Xie
- Zhuoqun (Ray) Zhang (2021)
- Jakub Piwowarczyk (2023)
- Jiacheng Li (2025)
