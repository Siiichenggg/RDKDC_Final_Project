function ik_main_simple(mode)
%IK_MAIN_SIMPLE Minimal IK demo: push 3cm, up 15cm, right 10cm, down, push back.
%   ik_main_simple()              % defaults to "sim"
%   ik_main_simple("real")        % connect to real robot (check IP in ur_rtde_interface)
%
% Uses urInvKin.m and the closest-solution selector shown in the reference
% PDF: choose the IK column with the smallest wrapped joint difference to
% the current configuration. No retries or safety extras to keep logic
% simple and transparent.

    if nargin < 1
        mode = "sim";
    end

    % Parameters (meters)
    push_dist = 0.03;
    lift_dist = 0.15;
    right_dist = 0.10;
    dt_segment = 1.0; % seconds per segment (adjust as needed)
    type = 'ur5e';

    % Connect
    ur = ur_rtde_interface(mode);
    cleanupObj = onCleanup(@() delete(ur));

    % Teach start
    fprintf("\n--- Teach pose: start ---\n");
    ur.switch_to_pendant_control();
    input("Move the arm to the start pose, then press ENTER...");
    q_start = ur.get_current_joints();
    ur.switch_to_ros_control();
    ur.activate_pos_control();

    q_curr = q_start(:);
    g_curr = ur.get_current_transformation(); % current SE(3) from RTDE

    % Sequence of Cartesian offsets in base frame
    steps = { ...
        "push_fwd",  [ push_dist;      0;          0], "Push forward 3 cm"; ...
        "lift_up",   [ 0;              0;  lift_dist], "Lift up 15 cm"; ...
        "move_right",[ 0;      right_dist;          0], "Move right 10 cm"; ...
        "lower",     [ 0;              0; -lift_dist], "Lower back down"; ...
        "push_back", [-push_dist;      0;          0], "Push back 3 cm"; ...
    };

    for k = 1:size(steps, 1)
        label = steps{k, 1};
        offset = steps{k, 2};
        desc = steps{k, 3};

        g_goal = translate_pose(g_curr, offset);
        Q = urInvKin(g_goal, type); % 6 x N
        [q_next, idx] = select_closest_ik(Q, q_curr);

        fprintf("Phase %d (%s) -> IK sol #%d\n", k, desc, idx);
        ur.move_joints(q_next, dt_segment);

        q_curr = ur.get_current_joints();
        g_curr = ur.get_current_transformation(); % refresh from hardware/sim
    end

    fprintf("Sequence complete.\n");
end

function g_out = translate_pose(g_in, offset)
%TRANSLATE_POSE Translate homogeneous transform by offset (base frame).
    g_out = g_in;
    g_out(1:3, 4) = g_in(1:3, 4) + offset;
end

function [q_star, idx] = select_closest_ik(Q, q_cur)
%SELECT_CLOSEST_IK Choose IK column closest to current joints (wrapped).
%   Q: 6xN IK candidates, q_cur: 6x1 current joints
    if isempty(Q)
        error("select_closest_ik:NoSolution", "No IK candidates returned.");
    end

    [~, m] = size(Q);
    cost = inf(1, m);
    for j = 1:m
        qj = Q(:, j);
        if any(~isfinite(qj))
            continue;
        end
        dq = wrap_to_pi(qj - q_cur);
        cost(j) = norm(dq);
    end
    [cmin, idx] = min(cost);
    if isinf(cmin)
        error("select_closest_ik:NoValid", "All IK candidates invalid.");
    end
    q_star = Q(:, idx);
end

function q = wrap_to_pi(q)
%WRAP_TO_PI Wrap angles to (-pi, pi].
    q = mod(q + pi, 2*pi) - pi;
end
