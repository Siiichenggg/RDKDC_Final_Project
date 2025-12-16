# 误差计算功能说明

## 概述

已在 `ur_interface.m` 和 `five_phase_push_manipulation.m` 中添加了标准化的位姿误差计算功能。

---

## 误差指标

### 1. d_R3 (mm) - 位置误差
```
d_R3 = ||r - r_d|| × 1000
```
- 计算实际位置与期望位置之间的欧几里得距离
- 单位：毫米（mm）

### 2. d_SO3 (unitless) - 旋转误差
```
d_SO3 = sqrt(trace((R - R_d)(R - R_d)^T))
```
- 计算实际旋转矩阵与期望旋转矩阵之间的差异
- 单位：无量纲

---

## 修改的文件

### 1. ur_interface.m

添加了新方法 `calculate_pose_error`：

**位置**：[ur_interface.m:440-477](ur_interface.m#L440-L477)

**使用方法**：
```matlab
[d_R3, d_SO3] = robot_interface.calculate_pose_error(...
    location,     % "Start" 或 "Target"
    r_actual,     % 实际位置向量 (3×1)
    R_actual,     % 实际旋转矩阵 (3×3)
    r_desired,    % 期望位置向量 (3×1)
    R_desired);   % 期望旋转矩阵 (3×3)
```

### 2. five_phase_push_manipulation.m

#### A. Start 时刻误差计算
**位置**：[five_phase_push_manipulation.m:47-59](five_phase_push_manipulation.m#L47-L59)

在 teach 完起始位姿后立即计算，检查机器人是否真的在参考起始位置。

#### B. Target 时刻误差计算
**位置**：[five_phase_push_manipulation.m:108-120](five_phase_push_manipulation.m#L108-L120)

在完成5阶段推动操作后计算，检查最终位姿与目标的偏差。

#### C. 综合误差总结
**位置**：[five_phase_push_manipulation.m:148-158](five_phase_push_manipulation.m#L148-L158)

在脚本结束时显示 Start 和 Target 两个时刻的误差汇总。

---

## 输出格式

当运行 `main_robot_control_script.m` 时，你将看到：

### Start 时刻输出
```
========================================
Pose Error Analysis - Start
========================================
Location:        Start
d_R3 (mm):       0.1234
d_SO3:           0.0056
========================================
```

### Target 时刻输出
```
========================================
Pose Error Analysis - Target
========================================
Location:        Target
d_R3 (mm):       2.3456
d_SO3:           0.0234
========================================
```

### 最终总结
```
========================================
COMPLETE ERROR SUMMARY
========================================
START Location:
  d_R3 (mm):  0.1234
  d_SO3:      0.0056

TARGET Location:
  d_R3 (mm):  2.3456
  d_SO3:      0.0234
========================================
```

---

## 如何运行

1. 确保机器人已连接并且 ROS2 节点正在运行
2. 运行主脚本：
   ```matlab
   main_robot_control_script
   ```
3. 按照提示将机器人移动到起始位置
4. 脚本将自动记录 Start 误差
5. 执行5阶段推动操作
6. 脚本将自动记录 Target 误差
7. 查看最终的误差总结

---

## 技术细节

### 数据源
- **实际位姿**：从 ROS2 TF 树获取 (`get_current_transformation('base', 'tool0')`)
- **期望位姿**：
  - Start: 从 teach 模式记录的参考位姿
  - Target: 从逆运动学计算的最终目标位姿

### 坐标系
- 基座坐标系：`base`
- 工具坐标系：`tool0`
- 所有位姿都在基座坐标系中表示

---

## 注意事项

1. 确保在 teach 起始位置时机器人已经稳定
2. 旋转误差 d_SO3 的有效范围取决于旋转矩阵的差异
3. 位置误差以毫米为单位，便于直观理解精度
4. 保留了原有的误差分析格式（"Legacy Format"）以保持向后兼容性

---

## 示例脚本

参考 [error_calculation_example.m](error_calculation_example.m) 了解如何在其他策略中使用此功能。
