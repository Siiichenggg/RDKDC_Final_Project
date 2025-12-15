% ur_project_rtde.m
% Final Project (RDKDC EN.530.646) - MATLAB RTDE Environment
% Main script: teach start pose then run RR (position-step) push-and-place.

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

% --- Robot model used by FK/Jacobian helpers ---
cfg.robotType = "ur5e"; % "ur5" or "ur5e"

% --- RR controller (position-step / discrete integration) ---
cfg.dt = 0.10;              % [s] RR update period AND move_joints time interval
cfg.K = 1.0;                % RR gain (dimensionless)
cfg.posTol = 0.005;         % [m] translation tolerance
cfg.rotTol = 5*pi/180;      % [rad] rotation tolerance
cfg.maxSteps = 600;         % max RR iterations per segment
cfg.dampLambda = 0.03;      % DLS damping (helps near singularities)
cfg.rotWeight = 3.0;        % dimensionless weight on orientation error (keeps world orientation fixed)

% --- Safety / limits ---
cfg.dqMax = 0.03;           % [rad/step] max joint increment magnitude (scaled)
cfg.zMin = 0.01;            % [m] minimum allowed tool z (table safety)
cfg.minSigma = 1e-3;        % minimum singular value threshold
cfg.maxCond = 1e8;          % maximum Jacobian condition number before warning
cfg.jointLimits = repmat([-2*pi, 2*pi], 6, 1); % conservative default

% --- Logging / diagnostics ---
cfg.logEvery = 20;          % print every N RR iterations
cfg.qMeasEps = 2e-4;        % [rad] "q_meas not changing" threshold (inf-norm)
cfg.freezeWarnIters = 40;   % consecutive iterations before warning

% --- Task parameters (push-and-place) ---
cfg.pushDist = 0.03;                 % [m] push distance (manual says "about 3 cm")
cfg.pushDirBase = [1; 0; 0];         % unit vector in base frame (edit if needed)
cfg.cubeSide = 0.13;                 % [m] foam cube edge length (user/lab setup dependent)
cfg.clearance = 0.03;                % [m] extra clearance for re-approach
cfg.liftHeight = 0.05;               % [m] lift height for repositioning (safety)
cfg.backApproachExtra = cfg.cubeSide + cfg.clearance; % [m] reach "other side" of cube
cfg.timeToReturnToStart = 3.0;       % [s] time for returning to taught start

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
cfg.R_ref = g_start(1:3, 1:3); % keep world orientation equal to taught start orientation

fprintf("\nRecorded start joint config (rad):\n");
disp(q_start);

%% Manual requires running RR via return button in RTDE script
input("Press ENTER to execute RR (push-and-place)...", "s");

%% =========================
%  Execute Task (RR + position-step)
%  =========================
taskResult = rtde_push_and_place_task(ur, q_start, g_start, cfg);

fprintf("\nTask finished. Summary:\n");
disp(taskResult);
