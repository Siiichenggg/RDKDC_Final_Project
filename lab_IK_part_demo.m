% lab_IK_part_demo
% A minimal "IK part" implementation for the project/lab:
% - Build a Cartesian straight-line path in SE(3)
% - Use urInvKin.m to map each waypoint to joint space
% - Select a continuous solution (closest to previous joint angles)
%
% This file does NOT modify any of the provided course files.

clear; clc;

type = 'ur5e'; % 'ur5' or 'ur5e'

% Sample start/target frames from the project manual (base_link -> tool0).
g_st1 = [ 0, -1,  0, 0.25; ...
         -1,  0,  0, 0.60; ...
          0,  0, -1, 0.22; ...
          0,  0,  0, 1.00 ];

g_st2 = [ 0, -1,  0, 0.40; ...
         -1,  0,  0, 0.45; ...
          0,  0, -1, 0.22; ...
          0,  0,  0, 1.00 ];

num_steps = 40;
q_seed = zeros(6, 1);

% If you are using a pen-gripper and your g_* are base_link->pen_tip,
% set g_tool0_tip accordingly and pass it via 'Tool0ToTip'.
g_tool0_tip = eye(4);

[q_traj, g_traj_tool0] = ur_ik_cartesian_traj( ...
    g_st1, g_st2, type, ...
    'NumSteps', num_steps, ...
    'QSeed', q_seed, ...
    'Tool0ToTip', g_tool0_tip);

fprintf('Computed IK trajectory with %d waypoints.\n', size(q_traj, 2));

% Optional: execute on RTDE interface (MATLAB RTDE).
% ur = ur_rtde_interface("sim", "0.0.0.0"); % or ur_rtde_interface("real")
% ur.activate_pos_control();
% ur.move_joints(q_traj, 0.2);
