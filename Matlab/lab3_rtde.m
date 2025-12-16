clear; clc; close all;

robotType = "ur5";    % or "ur5e"

ur = ur_rtde_interface();

q_init = [0; -pi/4;  pi/2; -pi/2;  pi/4; 0];
q_goal = [0; -pi/3;  pi/3; -pi/2;  pi/3; 0];

ur.move_joints(q_init, 5.0); pause(6);

g_desired = urFwdKin(q_goal, robotType);
K = 1.0;

fprintf('RTDE: normal RR control run...\n');
finalerr = urRRcontrol(g_desired, K, ur, robotType);
fprintf('RTDE: final error (normal) = %.3f cm\n', finalerr);

q_sing_init = [0; -pi/2;  0.01; 0; 0; 0];
q_sing_goal = [0; -pi/2; -0.01; 0; 0; 0];

ur.move_joints(q_sing_init, 5.0); pause(6);

g_sing = urFwdKin(q_sing_goal, robotType);

fprintf('RTDE: near singularity run (expect -1)...\n');
finalerr_sing = urRRcontrol(g_sing, K, ur, robotType);
fprintf('RTDE: return near singularity = %.1f\n', finalerr_sing);