% UR5e Push-and-Place Final Project (RTDE)
% Requirements:
% - Must use Resolved-Rate (RR) control via position interface (move_joints)
% - Must NOT modify ur_rtde_interface.m
% - Start pose must be taught by the user (no hard-code)

main();
return;

function main()
    clc;

    % Ensure this folder is on path (for ur_rtde_interface.m)
    thisDir = fileparts(mfilename('fullpath'));
    addpath(thisDir);

    % ---------------- User-configurable parameters ----------------
    mode = "real";                % "real" (UR5e) or "sim" (URSim)
    simIP = "0.0.0.0";            % used only when mode == "sim"

    dt = 0.10;                    % RR integration time step (s)
    pushDist = 0.03;              % 3 cm (m)
    liftHeight = 0.08;            % lift after pushing (m)
    otherSideOffset = 0.06;       % move to the other side (m), adjust to your block size

    % Push direction expressed in the same base frame as ur.get_current_transformation()
    % (default: base +Y)
    pushDir = [0; 1; 0];
    pushDir = pushDir / max(1e-12, norm(pushDir));

    % RR controller gains (units: 1/s)
    Kp_pos = 1.5;
    Kp_ori = 1.0;

    % Limits & stopping conditions
    qdotLimit = 1.2;              % rad/s (per-joint clamp)
    posTol = 0.002;               % m
    oriTol = 3 * pi/180;          % rad (approx)
    maxTimePerSegment = 25;       % s

    % Jacobian damping (for near-singularity robustness)
    lambdaMin = 1e-4;
    lambdaMax = 5e-2;
    condSoft = 250;
    condHard = 2000;
    % --------------------------------------------------------------

    ur = [];
    try
        % Initialize
        if mode == "sim"
            ur = ur_rtde_interface(mode, simIP);
        else
            ur = ur_rtde_interface(mode);
        end
        ur.activate_pos_control();

        fprintf('\n=== UR5e Push-and-Place (RTDE) ===\n');
        fprintf('RR dt = %.3f s, pushDist = %.3f m\n', dt, pushDist);

        % ---------------- Step 1: Teaching start ----------------
        fprintf('\n[Step 1] Teaching start pose (no hard-code)\n');
        fprintf('Please use the teach pendant to move the robot to the START pose:\n');
        fprintf('- End-effector should be on the LEFT side of the block (ready to push)\n');
        fprintf('- Make sure the TCP is at the pushing height and aligned safely\n');
        input('When ready, press ENTER to record g_start and q_start... ', 's');

        q_start = ur.get_current_joints();
        q_start = q_start(:);
        g_start = ur.get_current_transformation();
        fprintf('Recorded start.\n');

        % ---------------- Step 2: Compute target ----------------
        fprintf('\n[Step 2] Compute push target pose (+%.0f mm along pushDir)\n', pushDist*1000);
        g_target = g_start;
        g_target(1:3,4) = g_start(1:3,4) + pushDist * pushDir;

        % ---------------- Step 3: Execute pushing ----------------
        fprintf('\n[Step 3] Push block forward %.0f mm using RR control\n', pushDist*1000);
        rr_move_to_pose(ur, g_target, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
            maxTimePerSegment, lambdaMin, lambdaMax, condSoft, condHard);

        g_push_actual = ur.get_current_transformation();
        [dSO3_fwd, dR3_fwd] = pose_error_metrics(g_push_actual, g_target);
        fprintf('\n[Error Report] Forward push: actual vs theoretical target\n');
        fprintf('d_SO(3) = %.6f\n', dSO3_fwd);
        fprintf('d_R^3   = %.6f m\n', dR3_fwd);

        % -------- Lift, move to other side, push back (place) ----
        fprintf('\n[Step 3b] Lift end-effector (%.0f mm)\n', liftHeight*1000);
        g_lift = g_push_actual;
        g_lift(3,4) = g_lift(3,4) + liftHeight;
        rr_move_to_pose(ur, g_lift, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
            maxTimePerSegment, lambdaMin, lambdaMax, condSoft, condHard);

        fprintf('\n[Step 3c] Move to the other side (offset %.0f mm)\n', otherSideOffset*1000);
        g_other_above = g_lift;
        g_other_above(1:3,4) = g_other_above(1:3,4) + otherSideOffset * pushDir;
        rr_move_to_pose(ur, g_other_above, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
            maxTimePerSegment, lambdaMin, lambdaMax, condSoft, condHard);

        fprintf('\n[Step 3d] Lower to pushing height\n');
        g_other = g_other_above;
        g_other(3,4) = g_other(3,4) - liftHeight;
        rr_move_to_pose(ur, g_other, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
            maxTimePerSegment, lambdaMin, lambdaMax, condSoft, condHard);

        fprintf('\n[Step 3e] Push back %.0f mm (towards start)\n', pushDist*1000);
        g_back_target = g_other;
        g_back_target(1:3,4) = g_back_target(1:3,4) - pushDist * pushDir;
        rr_move_to_pose(ur, g_back_target, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
            maxTimePerSegment, lambdaMin, lambdaMax, condSoft, condHard);

        g_back_actual = ur.get_current_transformation();
        [dSO3_back, dR3_back] = pose_error_metrics(g_back_actual, g_back_target);
        fprintf('\n[Error Report] Push-back: actual vs theoretical target\n');
        fprintf('d_SO(3) = %.6f\n', dSO3_back);
        fprintf('d_R^3   = %.6f m\n', dR3_back);

        fprintf('\nDone.\n');
    catch ME
        fprintf(2, '\n[ur_project_rtde] ERROR: %s\n', ME.message);
        for k = 1:numel(ME.stack)
            fprintf(2, '  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end

    cleanup_ur(ur);
end

% =========================== RR CONTROL ===========================
function rr_move_to_pose(ur, g_des, dt, Kp_pos, Kp_ori, qdotLimit, posTol, oriTol, ...
    maxTime, lambdaMin, lambdaMax, condSoft, condHard)

    tStart = tic;
    while true
        if toc(tStart) > maxTime
            error('RR timeout (%.1f s) before reaching target.', maxTime);
        end

        q = ur.get_current_joints();
        q = q(:);
        g = ur.get_current_transformation();

        r = g(1:3,4);
        R = g(1:3,1:3);
        rd = g_des(1:3,4);
        Rd = g_des(1:3,1:3);

        % Task-space error (base frame of get_current_transformation)
        e_p = rd - r;
        R_err = Rd * R.';                 % rotation from current to desired
        e_w = 0.5 * vee3(R_err - R_err.'); % small-angle orientation error vector

        if norm(e_p) < posTol && norm(e_w) < oriTol
            return;
        end

        v = Kp_pos * e_p;
        w = Kp_ori * e_w;
        V = [v; w]; % 6x1 spatial twist in the same frame as get_current_transformation

        J = get_jacobian(q);

        % Damped least-squares "inverse" near singularities
        c = cond(J);
        if c < condSoft
            qdot = J \ V;
        else
            alpha = min(1, max(0, (c - condSoft) / max(1, (condHard - condSoft))));
            lambda = (1 - alpha) * lambdaMin + alpha * lambdaMax;
            qdot = (J.'*J + (lambda^2)*eye(6)) \ (J.' * V);
        end

        qdot = clamp_vec(qdot, -qdotLimit, qdotLimit);
        q_next = (q + qdot(:) * dt);
        q_next = q_next(:);

        stepTic = tic;
        ur.move_joints(q_next, dt);
        % Keep approximate update rate (move_joints may be blocking or non-blocking)
        elapsed = toc(stepTic);
        if elapsed < dt
            pause(dt - elapsed);
        end
    end
end

% =========================== JACOBIAN =============================
function J = get_jacobian(q)
    % UR5e (UR5 family) standard DH parameters (meters)
    % a(i), alpha(i), d(i) with joint variable q(i)
    a = [0, -0.42500, -0.39225, 0, 0, 0];
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823];
    alpha = [pi/2, 0, 0, pi/2, -pi/2, 0];

    T = eye(4);
    p = zeros(3, 7);
    z = zeros(3, 7);
    p(:,1) = T(1:3,4);
    z(:,1) = T(1:3,3);

    for i = 1:6
        T = T * dh(a(i), alpha(i), d(i), q(i));
        p(:,i+1) = T(1:3,4);
        z(:,i+1) = T(1:3,3);
    end

    p_e = p(:,7);
    Jv = zeros(3,6);
    Jw = zeros(3,6);
    for i = 1:6
        z_prev = z(:,i);
        p_prev = p(:,i);
        Jv(:,i) = cross(z_prev, (p_e - p_prev));
        Jw(:,i) = z_prev;
    end
    J_dh = [Jv; Jw];

    % Match the frame convention used by ur.get_current_transformation():
    % g_measured = Offset * g_true, Offset = RotX(pi)
    R0 = rot_x(pi);
    Ad = [R0 zeros(3); zeros(3) R0];
    J = Ad * J_dh;
end

function A = dh(a, alpha, d, theta)
    ct = cos(theta); st = sin(theta);
    ca = cos(alpha); sa = sin(alpha);
    A = [ ct, -st*ca,  st*sa, a*ct;
          st,  ct*ca, -ct*sa, a*st;
           0,     sa,     ca,    d;
           0,      0,      0,    1];
end

% =========================== METRICS ==============================
function [dSO3, dR3] = pose_error_metrics(g, gd)
    R = g(1:3,1:3);
    Rd = gd(1:3,1:3);
    r = g(1:3,4);
    rd = gd(1:3,4);

    % Manual-specified formulas (DO NOT change):
    dSO3 = sqrt(trace((R - Rd) * (R - Rd).'));
    dR3 = norm(r - rd);
end

% =========================== HELPERS ==============================
function v = vee3(S)
    v = [S(3,2); S(1,3); S(2,1)];
end

function R = rot_x(theta)
    R = [1 0 0;
         0 cos(theta) -sin(theta);
         0 sin(theta)  cos(theta)];
end

function x = clamp_vec(x, lo, hi)
    x = min(max(x, lo), hi);
end

function cleanup_ur(ur)
    if isempty(ur)
        return;
    end
    try
        delete(ur);
    catch
        % ignore cleanup errors
    end
end
