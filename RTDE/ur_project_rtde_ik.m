% ur_project_rtde_ik.m
% Final Project (RDKDC EN.530.646) - MATLAB RTDE Environment
% Main script: teach start pose then run IK-based push-and-place.
% This version uses inverse kinematics instead of RR control.

clear; clc;

%% =========================
%  User Configuration
%  =========================
cfg = struct();

% --- Connection ---
% mode: "sim" or "real"
cfg.mode = "real";
% For "sim", you may pass an explicit IP. Leave empty to use interface default.
cfg.sim_ip = "";

% --- Robot model used by FK/IK helpers ---
cfg.robotType = "ur5e"; % "ur5" or "ur5e"

% --- IK and motion parameters ---
cfg.timeToReturnToStart = 3.0;       % [s] time for each movement
cfg.rotWeight = 10.0;                % IK weight on orientation (higher = maintain orientation better)

% --- Safety / limits ---
cfg.zMin = 0.01;            % [m] minimum allowed tool z (table safety)
cfg.jointLimits = repmat([-2*pi, 2*pi], 6, 1); % conservative default

% --- Task parameters (push-and-place) ---
cfg.pushDist = 0.03;                 % [m] push distance (manual says "about 3 cm")
cfg.pushDirBase = [1; 0; 0];         % unit vector in base frame (edit if needed)
cfg.cubeSide = 0.13;                 % [m] foam cube edge length

%% =========================
%  Path setup (avoid ur_rtde_interface shadowing)
%  =========================
rtde_path_setup();

%% =========================
%  Connect (required: ur = ur_rtde_interface(mode, ip); )
%  =========================
if cfg.mode == "sim" && strlength(cfg.sim_ip) > 0
    ur = ur_rtde_interface(cfg.mode, cfg.sim_ip);
else
    ur = ur_rtde_interface(cfg.mode);
end

cleanupObj = onCleanup(@() delete(ur));

%% =========================
%  Teach Start Pose (must not be hard-coded)
%  =========================
[q_start, g_start] = rtde_teach_start_pose(ur, cfg.robotType);

fprintf("\nRecorded start joint config (rad):\n");
disp(q_start);

%% Manual requires running via return button in RTDE script
input("Press ENTER to execute IK-based push-and-place...", "s");

%% =========================
%  Execute Task (IK + move_joints)
%  =========================
taskResult = rtde_push_and_place_task_ik(ur, q_start, g_start, cfg);

fprintf("\nTask finished. Summary:\n");
disp(taskResult);
