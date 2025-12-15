# IK Organized - 整合的逆运动学文件

这个文件夹包含了整合后的IK相关代码，功能与原文件夹相同，但结构更加清晰简洁。

## 文件结构

### 核心文件 (整合后的新文件)

1. **ik_utils.m** - 工具函数类
   - `ROTX(alpha)` - X轴旋转矩阵
   - `ROTZ(theta)` - Z轴旋转矩阵
   - `urFwdKin(q, type)` - 正向运动学
   - `selectClosestIK(Q, q_cur)` - 选择最接近的IK解
   - `wrapToPi(q)` - 角度包裹到[-π, π]
   - `translatePose(g_in, offset)` - 平移位姿

2. **ik_demos.m** - 演示任务类
   - `move_toolY_3cm(ur, type)` - 沿工具Y轴移动3cm
   - `move_push_back(ur, type)` - 完整的推-提-移-降-推回序列

3. **ik_main.m** - 主程序入口
   - 支持多种演示模式
   - 自动连接仿真或真实机器人

### 保持不变的文件

- **DH.m** - DH变换函数 (未修改)
- **urInvKin.m** - 逆运动学求解 (未修改)
- **test_urInvKin.m** - 逆运动学测试 (未修改)

### 接口文件

- **ur_interface.m** - ROS接口 (复制)
- **ur_rtde_interface.m** - RTDE接口 (复制)

## 使用方法

### 基本用法

```matlab
% 默认：仿真模式，简单演示
ik_main()

% 真实机器人，简单演示
ik_main("real")

% 仿真模式，推回演示
ik_main("sim", "push_back")

% 仿真模式，3cm推动演示
ik_main("sim", "toolY_3cm")
```

### 演示类型说明

1. **"simple"** (默认)
   - 推forward 3cm → 提升15cm → 右移10cm → 下降 → 推back 3cm

2. **"push_back"**
   - 完整的推-提-移-降-推回序列
   - 包含侧向移动和多阶段运动

3. **"toolY_3cm"**
   - 简单的沿工具Y轴推动3cm

### 直接调用演示函数

```matlab
% 使用ur_interface或ur_rtde_interface
ur = ur_rtde_interface("sim");
type = 'ur5e';

% 调用演示函数
ik_demos.move_toolY_3cm(ur, type);
% 或
ik_demos.move_push_back(ur, type);
```

### 使用工具函数

```matlab
% 正向运动学
q = [0; -pi/2; 0; -pi/2; 0; 0];
g = ik_utils.urFwdKin(q, 'ur5e');

% 选择最接近的IK解
Q = urInvKin(g_desired, 'ur5e');
[q_best, idx] = ik_utils.selectClosestIK(Q, q_current);

% 角度包裹
q_wrapped = ik_utils.wrapToPi(q);
```

## 整合说明

### 原文件对应关系

**整合到 ik_utils.m:**
- ROTX.m
- ROTZ.m
- selectClosestIK.m
- urFwdKin.m
- 添加了wrapToPi和translatePose辅助函数

**整合到 ik_demos.m:**
- ik_move_toolY_3cm.m
- ik_move_push_back.m
- 包含内部辅助函数ik_go

**整合到 ik_main.m:**
- ik_main_simple.m
- ik_test.m
- 添加了演示类型选择功能

## 优势

1. **更清晰的结构** - 相关功能组织在一起
2. **更少的文件** - 从13个文件减少到8个文件
3. **更好的可维护性** - 使用类静态方法组织代码
4. **保持兼容性** - 核心功能不变，DH、urInvKin等保持原样
5. **更易使用** - 统一的入口点和清晰的接口

## 注意事项

- 确保MATLAB路径包含此文件夹
- 需要urInvKin.m和DH.m才能正常工作
- 根据使用的机器人选择正确的接口(ur_interface或ur_rtde_interface)
