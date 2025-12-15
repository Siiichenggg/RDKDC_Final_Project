function result = rtde_rr_move_to_pose(ur, g_desired, cfg, segmentName)
%RTDE_RR_MOVE_TO_POSE Resolved-Rate (RR) control with position-step commands.
%
% Discrete integration loop:
%   q_{k+1} = q_k + dq
%   dq      = -K*dt * J_b(q_k)^+ * xi
% where xi is the body twist error extracted from:
%   exp(hat(xi)) = g_desired^{-1} * g(q_k)
%
% Commands are sent ONLY through ur.move_joints(q_next, dt) (position interface).

if nargin < 4
    segmentName = "segment";
end

dt = cfg.dt;
if ~(isscalar(dt) && dt > 0)
    error('cfg.dt must be a positive scalar.');
end

pushDir = cfg.pushDirBase;
if ~(isnumeric(pushDir) && numel(pushDir) == 3)
    error('cfg.pushDirBase must be a 3x1 vector.');
end

q_prev = [];
freezeCount = 0;
warnedFreeze = false;

result = struct();
result.segment = segmentName;
result.converged = false;
result.steps = 0;
result.posErr = NaN;
result.rotErr = NaN;
result.sigmaMin = NaN;
result.condJ = NaN;
result.maxAbsDq = NaN;
result.q_final = [];
result.msg = "";

for k = 1:cfg.maxSteps
    q = ur.get_current_joints();

    if ~isempty(q_prev)
        if norm(q - q_prev, inf) < cfg.qMeasEps
            freezeCount = freezeCount + 1;
        else
            freezeCount = 0;
            warnedFreeze = false;
        end
    end

    g_current = urFwdKin(q, cfg.robotType);
    if g_current(3,4) < cfg.zMin
        result.msg = sprintf("ABORT: tool z=%.4f < zMin=%.4f (table safety).", g_current(3,4), cfg.zMin);
        result.q_final = q;
        result.steps = k;
        return;
    end

    g_error = FINV(g_desired) * g_current;
    xi = getXi(g_error); % [v; w] body error (un-normalized)

    posErr = norm(xi(1:3));
    rotErr = norm(xi(4:6));
    result.posErr = posErr;
    result.rotErr = rotErr;

    if posErr < cfg.posTol && rotErr < cfg.rotTol
        result.converged = true;
        result.msg = "OK";
        result.q_final = q;
        result.steps = k;
        return;
    end

    Jb = urBodyJacobian(q, cfg.robotType);
    s = svd(Jb);
    sigmaMin = s(end);
    result.sigmaMin = sigmaMin;
    if sigmaMin < cfg.minSigma
        result.msg = sprintf("ABORT: singularity risk (sigma_min=%.3e < %.3e).", sigmaMin, cfg.minSigma);
        result.q_final = q;
        result.steps = k;
        return;
    end

    condJ = cond(Jb);
    result.condJ = condJ;
    if condJ > cfg.maxCond && mod(k, cfg.logEvery) == 0
        fprintf("[%s] WARN: Jacobian ill-conditioned: cond=%.2e\n", segmentName, condJ);
    end

    % Damped least squares pseudo-inverse: J^+ = J' * (J*J' + λ^2 I)^{-1}
    A = (Jb * Jb.') + (cfg.dampLambda^2) * eye(6);
    dq = -(cfg.K * dt) * (Jb.' * (A \ xi));

    % Step limiting (scale uniformly)
    maxAbsDq = max(abs(dq));
    result.maxAbsDq = maxAbsDq;
    if maxAbsDq > cfg.dqMax
        dq = dq * (cfg.dqMax / maxAbsDq);
        maxAbsDq = cfg.dqMax;
        result.maxAbsDq = maxAbsDq;
    end

    q_next = q + dq;

    % Joint limits clamp (conservative)
    for i = 1:6
        q_next(i) = min(max(q_next(i), cfg.jointLimits(i,1)), cfg.jointLimits(i,2));
    end

    if mod(k, cfg.logEvery) == 0 || k == 1
        fprintf("[%s] k=%d posErr=%.4f[m] rotErr=%.3f[rad] |dq|max=%.4f[rad] sigmaMin=%.2e\n", ...
            segmentName, k, posErr, rotErr, maxAbsDq, sigmaMin);

        if freezeCount >= cfg.freezeWarnIters && ~warnedFreeze
            warnedFreeze = true;
            fprintf("[%s] WARN: q_meas not changing for %d iters (possible not in remote/servo, paused sim, or protective stop).\n", ...
                segmentName, freezeCount);
        end
    end

    ur.move_joints(q_next, dt);
    pause(dt);

    q_prev = q;
    result.steps = k;
end

result.msg = "ABORT: maxSteps reached without convergence.";
result.q_final = ur.get_current_joints();
end

