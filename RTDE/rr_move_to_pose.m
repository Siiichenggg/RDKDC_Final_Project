function [success, q_final, logData] = rr_move_to_pose(ur, q_init, g_des, params, label, jointLimits, zMin)
%RR_MOVE_TO_POSE Run resolved-rate control to reach a desired SE(3) pose.
%   [success, q_final, logData] = rr_move_to_pose(ur, q_init, g_des, params, label, jointLimits, zMin)
%
%   Inputs:
%     ur           - ur_rtde_interface object (position control only)
%     q_init       - 6x1 column vector of starting joints
%     g_des        - 4x4 desired homogeneous transform (base -> tool)
%     params       - struct with fields:
%                       dt, maxSteps, posTol, rotTol,
%                       kp_pos, kp_rot, speed_limit, fk_fun, jac_fun
%     label        - string for console logging
%     jointLimits  - 6x2 matrix [min max] for each joint
%     zMin         - scalar minimum allowed tool z-height (table collision check)
%
%   Outputs:
%     success      - true if tolerance met before maxSteps
%     q_final      - last commanded joint configuration
%     logData      - struct of error norms and heights (for plotting/debug)
%
%   This helper only performs small joint-space steps, calling ur.move_joints
%   each iteration. Velocity control is intentionally not used.

if nargin < 6
    error('rr_move_to_pose requires ur, q_init, g_des, params, label, jointLimits, zMin');
end

success = false;
q_curr = q_init;

logData.posErr = zeros(params.maxSteps, 1);
logData.rotErr = zeros(params.maxSteps, 1);
logData.dqMax = zeros(params.maxSteps, 1);
logData.zHeight = zeros(params.maxSteps, 1);

fprintf('--- RR segment: %s ---\n', label);

bSize = 1;
if isfield(params, 'batch_size') && params.batch_size > 0
    bSize = round(params.batch_size);
    bSize = max(1, bSize);
end

% Weighted damped least-squares (helps keep world orientation fixed)
w_pos = 1.0;
if isfield(params, 'w_pos') && ~isempty(params.w_pos)
    w_pos = params.w_pos;
end
w_rot = 1.0;
if isfield(params, 'w_rot') && ~isempty(params.w_rot)
    w_rot = params.w_rot;
end
dls_lambda = 0.0;
if isfield(params, 'dls_lambda') && ~isempty(params.dls_lambda)
    dls_lambda = params.dls_lambda;
end
v_max = inf;
if isfield(params, 'v_max') && ~isempty(params.v_max)
    v_max = params.v_max;
end
w_max = inf;
if isfield(params, 'w_max') && ~isempty(params.w_max)
    w_max = params.w_max;
end

q_batch = [];
dt_batch = [];

for k = 1:params.maxSteps
    q_curr = ur.get_current_joints(); % keep Jacobian consistent with actual state
    g_curr = ur.get_current_transformation();
    [e_p, e_R] = compute_pose_error(g_des, g_curr);
    pos_err_norm = norm(e_p);
    rot_err_norm = norm(e_R);

    logData.posErr(k) = pos_err_norm;
    logData.rotErr(k) = rot_err_norm;
    logData.zHeight(k) = g_curr(3,4);

    if pos_err_norm < params.posTol && rot_err_norm < params.rotTol
        success = true;
        fprintf('Reached %s: ||pos_err||=%.4f m, ||rot_err||=%.4f rad (step %d)\n', ...
            label, pos_err_norm, rot_err_norm, k);
        break;
    end

    % RR twist command in spatial frame
    v_cmd = params.kp_pos * e_p;
    w_cmd = params.kp_rot * e_R;
    if isfinite(v_max)
        v_norm = norm(v_cmd);
        if v_norm > v_max
            v_cmd = (v_max / v_norm) * v_cmd;
        end
    end
    if isfinite(w_max)
        w_norm = norm(w_cmd);
        if w_norm > w_max
            w_cmd = (w_max / w_norm) * w_cmd;
        end
    end
    V = [v_cmd; w_cmd];

    % Spatial Jacobian at current configuration (expressed in base frame)
    J = params.jac_fun(q_curr);

    % Solve dq with weighted damped least squares:
    %   dq = argmin ||W(J*dq - V)||^2 + lambda^2||dq||^2
    W = diag([w_pos; w_pos; w_pos; w_rot; w_rot; w_rot]);
    Jw = W * J;
    Vw = W * V;
    if dls_lambda > 0
        dq = (Jw' * ((Jw * Jw' + (dls_lambda^2) * eye(6)) \ Vw)) * params.dt;
    else
        dq = pinv(Jw) * Vw * params.dt;
    end
    [dq_scaled, dt_cmd] = clamp_velocity(dq, params.dt, params.speed_limit);
    dq = dq_scaled;

    dq_rate = dq / dt_cmd;
    logData.dqMax(k) = max(abs(dq_rate));

    q_next = q_curr + dq;

    % Safety checks before sending the waypoint
    safety_checks(q_next, jointLimits, zMin, params.fk_fun);

    % Buffer waypoints to reduce start/stop jerk on hardware
    q_batch = [q_batch, q_next]; %#ok<AGROW>
    dt_batch = [dt_batch, dt_cmd]; %#ok<AGROW>
    if size(q_batch, 2) >= bSize || k == params.maxSteps
        ur.move_joints(q_batch, dt_batch);
        q_batch = [];
        dt_batch = [];
        % Sync current joints after sending a batch
        q_curr = ur.get_current_joints();
    end

    if mod(k, 20) == 0
        fprintf('step %d/%d | ||pos_err||=%.4f | ||rot_err||=%.4f | dq_max=%.4f | z=%.3f\n', ...
            k, params.maxSteps, pos_err_norm, rot_err_norm, logData.dqMax(k), logData.zHeight(k));
    end
end

logData.posErr = logData.posErr(1:k);
logData.rotErr = logData.rotErr(1:k);
logData.dqMax = logData.dqMax(1:k);
logData.zHeight = logData.zHeight(1:k);
q_final = q_curr;

if ~success
    warning('RR segment "%s" stopped after %d steps without hitting tolerance.', label, params.maxSteps);
end

end

function [e_p, e_R] = compute_pose_error(g_des, g_curr)
% Compute spatial position and orientation error between desired/current SE(3).
R_d = g_des(1:3,1:3);
R = g_curr(1:3,1:3);
p_d = g_des(1:3,4);
p = g_curr(1:3,4);

e_p = p_d - p;
R_err = R_d * R';
e_R = so3ToVec(MatrixLog3(R_err));
end

function [dq_scaled, dt_cmd] = clamp_velocity(dq, dt, vel_limit)
% Scale dq if needed to satisfy joint rate limit.
dq_rate = dq / dt;
maxRate = max(abs(dq_rate));
if maxRate <= vel_limit
    dq_scaled = dq;
    dt_cmd = dt;
    return;
end

scale = (vel_limit * 0.98) / maxRate; % small margin
dq_scaled = dq * scale;
dt_cmd = dt; % keep timing; motion just uses smaller step
end

function safety_checks(q, jointLimits, zMin, fk_fun)
% Check joint limits and simple table collision via tool z-height.
if any(q < jointLimits(:,1) - 1e-6) || any(q > jointLimits(:,2) + 1e-6)
    error('Joint limit violated. Aborting motion.');
end

g_pred = fk_fun(q);
if g_pred(3,4) < zMin
    error('Tool z-height %.3f below table threshold %.3f. Aborting motion.', g_pred(3,4), zMin);
end
end

function so3mat = MatrixLog3(R)
% Robust matrix log for SO(3) -> so(3).
acosinput = (trace(R) - 1) / 2;
acosinput = min(max(acosinput, -1), 1);
theta = acos(acosinput);
if abs(theta) < 1e-6
    so3mat = zeros(3,3);
else
    so3mat = theta / (2*sin(theta)) * (R - R');
end
end

function v = so3ToVec(so3mat)
% Vee operator for so(3).
v = [so3mat(3,2); so3mat(1,3); so3mat(2,1)];
end
