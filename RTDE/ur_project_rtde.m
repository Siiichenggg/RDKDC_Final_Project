% ur_project_rtde.m
% Main RTDE entry script for RDKDC Final Project (RR control only).
% How to run:
%   - Place this folder next to ur_rtde_interface.m (parent directory).
%   - Set MODE="sim" to talk to URSim, or MODE="real" for the physical arm.
%   - Requires Robotics System Toolbox (loadrobot('universalUR5e')) or will
%     fall back to a simple DH model. Helper files used: rr_move_to_pose.m,
%     ur5e_fkine.m, ur5e_geometric_jacobian.m.
%   - Tune task constants below; all parameters live in the top block.

clear; clc;

%% ---------------- User-editable constants ----------------
MODE = "real";                 % "real" or "sim"
HOME_Q = [0; -pi/2; pi/2; -pi/2; -pi/2; 0]; % safe home posture
PUSH_DIST = 0.03;              % push distance in meters (3 cm)
DT = 0.05;                     % RR step time (s)
MAX_STEPS = 800;               % max iterations per RR segment
POS_TOL = 4e-3;                % 4 mm
ROT_TOL = 2e-2;                % rad
Kp_pos = 2.5;
Kp_rot = 2.0;
Z_MIN = 0.05;                  % table collision threshold (m)
JOINT_LIMITS = repmat([-2*pi, 2*pi], 6, 1); % joint limits
LIFT_HEIGHT = 0.08;            % lift above contact height (m)
CUBE_LEN = 0.13;               % cube side length (m)
SIDE_CLEARANCE = 0.02;         % clearance when moving around cube (m)
PLOT_ERRORS = true;            % plot error curves at the end
%% ---------------------------------------------------------

% Add current folder to path for helpers, and parent for ur_rtde_interface
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fileparts(scriptDir));

try
    % 1) Connect
    fprintf('Connecting to UR5e in %s mode...\n', MODE);
    ur = ur_rtde_interface(MODE);
    if isprop(ur, 'isConnected') && ~ur.isConnected
        error('RTDE connection failed.');
    end
    fprintf('Connection OK. Speed limit enforced at %.3f rad/s.\n', ur.speed_limit);

    % 2) Move to home
    q_curr = ur.get_current_joints();
    time_home = max(6.0, max(abs(HOME_Q - q_curr)) / ur.speed_limit + 0.5);
    fprintf('Moving to home (%.2f s)...\n', time_home);
    ur.move_joints(HOME_Q, time_home);

    % 3) Teach start pose
    disp('--- Teach start (1) ---');
    disp('Manually guide the arm to the start pose (contact with cube).');
    input('Press ENTER to record start pose...','s');
    g_start = ur.get_current_transformation();
    q_start = ur.get_current_joints();
    fprintf('Start pose recorded.\n');
    disp(g_start);

    % 4) Wait for user to trigger RR
    input('Press ENTER to start RR motion from taught start...','s');

    % Common RR parameters
    rrParams = struct( ...
        'dt', DT, ...
        'maxSteps', MAX_STEPS, ...
        'posTol', POS_TOL, ...
        'rotTol', ROT_TOL, ...
        'kp_pos', Kp_pos, ...
        'kp_rot', Kp_rot, ...
        'speed_limit', ur.speed_limit, ...
        'fk_fun', @ur5e_fkine, ...
        'jac_fun', @ur5e_geometric_jacobian ...
    );

    % Push direction in base frame (right-to-left along -Y in UR base)
    push_dir_base = [0; -1; 0];
    push_dir_base = push_dir_base / norm(push_dir_base);

    % 5) First push: start -> end
    p_start = g_start(1:3,4);
    R_start = g_start(1:3,1:3);
    p_end = p_start + PUSH_DIST * push_dir_base;
    g_end = [R_start, p_end; 0 0 0 1];

    [succ1, q_after_first, log1] = rr_move_to_pose( ...
        ur, q_start, g_end, rrParams, "push forward 3cm", JOINT_LIMITS, Z_MIN);
    g_end_actual = ur.get_current_transformation();
    fprintf('Reached end of first push (measured):\n');
    disp(g_end_actual);
    if ~succ1
        error('First push did not converge. Aborting.');
    end

    % 6) Compute target (2) using end pose
    z_contact = g_start(3,4);
    z_lift = z_contact + LIFT_HEIGHT;
    approach_shift = CUBE_LEN + SIDE_CLEARANCE + PUSH_DIST;

    % Lift straight up from end
    g_lift_from_end = g_end_actual;
    g_lift_from_end(3,4) = z_lift;

    % Move over the cube to the opposite side at lift height
    p_pre = g_end_actual(1:3,4) + approach_shift * push_dir_base;
    g_pre_lift = [R_start, p_pre; 0 0 0 1];
    g_pre_lift(3,4) = z_lift;

    % Descend to contact height on far side
    g_pre_contact = g_pre_lift;
    g_pre_contact(3,4) = z_contact;

    % Final desired pose after pushing back
    p_final = g_pre_contact(1:3,4) - PUSH_DIST * push_dir_base;
    g_final = [R_start, p_final; 0 0 0 1];

    % 7) Execute second sequence
    fprintf('--- Second segment: reposition for reverse push ---\n');
    [succ2a, q_lift, log2a] = rr_move_to_pose( ...
        ur, q_after_first, g_lift_from_end, rrParams, "lift after push 1", JOINT_LIMITS, Z_MIN);
    [succ2b, q_over, log2b] = rr_move_to_pose( ...
        ur, q_lift, g_pre_lift, rrParams, "move above far side", JOINT_LIMITS, Z_MIN);
    [succ2c, q_contact, log2c] = rr_move_to_pose( ...
        ur, q_over, g_pre_contact, rrParams, "descend to contact", JOINT_LIMITS, Z_MIN);
    [succ2d, q_final, log2d] = rr_move_to_pose( ...
        ur, q_contact, g_final, rrParams, "push back 3cm", JOINT_LIMITS, Z_MIN);

    g_final_meas = ur.get_current_transformation();
    fprintf('Reached final target (measured):\n');
    disp(g_final_meas);
    if ~(succ2a && succ2b && succ2c && succ2d)
        warning('Second push sequence hit max iterations; review trajectory before running on hardware.');
    end

    % 8) Return home
    q_curr = ur.get_current_joints();
    time_home_back = max(6.0, max(abs(HOME_Q - q_curr)) / ur.speed_limit + 0.5);
    fprintf('Returning home (%.2f s)...\n', time_home_back);
    ur.move_joints(HOME_Q, time_home_back);
    disp('Back at home. Task complete.');

    % 9) Optional plots
    if PLOT_ERRORS
        plot_error_logs(log1, log2a, log2b, log2c, log2d);
    end

catch ME
    warning('Aborting due to error: %s', ME.message);
    rethrow(ME);
end

%% --- Helper for plotting ---
function plot_error_logs(varargin)
figure; hold on;
colors = lines(numel(varargin));
idx = 1;
for i = 1:numel(varargin)
    logSeg = varargin{i};
    if isempty(logSeg) || ~isfield(logSeg, 'posErr')
        continue;
    end
    steps = 1:length(logSeg.posErr);
    plot(steps, logSeg.posErr, 'Color', colors(idx,:), 'DisplayName', sprintf('Segment %d pos', i));
    plot(steps, logSeg.rotErr, '--', 'Color', colors(idx,:), 'DisplayName', sprintf('Segment %d rot', i));
    idx = idx + 1;
end
xlabel('Iteration'); ylabel('Error');
title('RR tracking errors');
legend('show');
grid on;
end
