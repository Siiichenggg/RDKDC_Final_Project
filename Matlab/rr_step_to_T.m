function [done, q_next] = rr_step_to_T(ur, T_des, dt, posTol, ~)
%RR_STEP_TO_T One RR step toward desired transform (base->tool0).
% Output q_next is 6x1.

    q = ur.get_current_joints();            % 6x1 (force column)
    q = q(:);
    T = ur.get_current_transformation();    % 4x4

    % --- position error (position-only control) ---
    p  = T(1:3,4);
    pd = T_des(1:3,4);

    % position error
    ep = pd - p;

    % Check convergence (position only, like Python version)
    done = (norm(ep) < posTol);

    % --- RR law (position-only control, like Python version) ---
    Kp_pos = 0.3;   % softer gain, align with slower Python tuning
    v = Kp_pos * ep;   % 3x1 velocity command

    % --- Position Jacobian (3x6) ---
    J_full = get_tool0_jacobian(ur, q);    % 6x6
    J_pos = J_full(4:6, :);   % Extract linear velocity part (rows 4-6)

    qdot = pinv(J_pos, 1e-3) * v;

    % Debug: check if qdot makes sense
    % (Uncomment for debugging)
    % fprintf('  ep = [%.4f, %.4f, %.4f], norm = %.4f\n', ep, norm(ep));
    % fprintf('  qdot max = %.4f\n', max(abs(qdot)));

    % --- safety: joint speed limit (use ur.speed_limit with margin) ---
    speed_margin = 0.3;  % mirror Python RR_SPEED_MARGIN
    eff_speed_limit = ur.speed_limit * speed_margin;
    qdot = max(min(qdot, eff_speed_limit), -eff_speed_limit);

    q_next = q + qdot * dt;
    q_next = q_next(:);                     % keep as 6x1 (no implicit expansion)

    % --- safety: joint limits (from rigidBodyTree) ---
    q_next = clamp_to_joint_limits(ur.robotModel, q_next);

    % if clamping jumps to a limit, re-enforce per-step speed bound
    dq = q_next - q;
    max_step = eff_speed_limit * dt;
    scale = max(abs(dq)) / max_step;
    if scale > 1
        dq = dq / scale;
        q_next = q + dq;
    end
end


function J = get_tool0_jacobian(ur, q)
    rbt = ur.robotModel;
    % assume last body is tool0-like
    ee = rbt.BodyNames{end};
    J = geometricJacobian(rbt, q(:).', ee);  % use row vector config
end


function q2 = clamp_to_joint_limits(rbt, q)
    q2 = q;
    lim = rbt.homeConfiguration; %#ok<NASGU>
    % generic way: read limits from each joint if present
    bodies = rbt.Bodies;
    idx = 1;
    for i = 1:numel(bodies)
        j = bodies{i}.Joint;
        if strcmp(j.Type,'revolute') || strcmp(j.Type,'prismatic')
            if idx <= numel(q2) && all(isfinite(j.PositionLimits))
                q2(idx) = min(max(q2(idx), j.PositionLimits(1)), j.PositionLimits(2));
            end
            idx = idx + 1;
        end
    end
end
