function out = rtde_push_and_place_task_ik(ur, q_start, g_start, cfg)
%RTDE_PUSH_AND_PLACE_TASK_IK Execute push-and-place using IK + move_joints commands.
%
% This version uses inverse kinematics and direct joint commands instead of RR control.
% Segments:
%   A: Push forward 3cm
%   B: Return to start position
%   C: Lift 10cm above start
%   D: Translate horizontally to position(2) above (19cm forward, 10cm up)
%   E: Lower to position(2)
%   F: Push back 6cm

out = struct();
out.ok = false;
out.segments = struct([]);
out.msg = "";

% Normalize direction
pushDir = cfg.pushDirBase(:);
if norm(pushDir) < 1e-9
    error('cfg.pushDirBase must be non-zero.');
end
pushDir = pushDir / norm(pushDir);

% Keep world orientation fixed to the taught start orientation
R_ref = g_start(1:3, 1:3);

% Setup IK solver using MATLAB Robotics Toolbox
ik = inverseKinematics('RigidBodyTree', ur.robotModel);
ik.SolverParameters.AllowRandomRestart = false;
weights = [1 1 1 cfg.rotWeight cfg.rotWeight cfg.rotWeight]; % [x y z rx ry rz]

% Helper function to solve IK and move
    function [success, q_result] = solveIKAndMove(targetPose, initialGuess, moveTime, segmentName)
        % Convert SE(3) matrix to position + orientation
        targetPos = targetPose(1:3, 4)';
        targetRot = rotm2quat(targetPose(1:3, 1:3));

        % Solve IK
        [q_result, solInfo] = ik('tool0', targetPos, targetRot, weights, initialGuess);

        if ~strcmp(solInfo.Status, 'success')
            fprintf("[%s] WARN: IK failed with status: %s\n", segmentName, solInfo.Status);
            success = false;
            return;
        end

        % Check joint limits
        for i = 1:6
            if q_result(i) < cfg.jointLimits(i,1) || q_result(i) > cfg.jointLimits(i,2)
                fprintf("[%s] WARN: Joint %d out of limits\n", segmentName, i);
                success = false;
                return;
            end
        end

        % Execute motion
        fprintf("\n=== Executing %s ===\n", segmentName);
        ur.move_joints(q_result, moveTime);
        pause(moveTime + 0.1); % Wait for motion to complete

        % Verify position
        q_actual = ur.get_current_joints();
        g_actual = rtde_urFwdKin(q_actual, cfg.robotType);
        posErr = norm(g_actual(1:3,4) - targetPose(1:3,4));
        fprintf("[%s] Position error: %.4f m\n", segmentName, posErr);

        success = true;
    end

% Return to start position first
fprintf("\n=== Returning to start position ===\n");
ur.move_joints(q_start, cfg.timeToReturnToStart);
pause(cfg.timeToReturnToStart + 0.1);

%% Segment A: Push forward 3cm
fprintf("\n=== Segment A: Push forward 3cm ===\n");
p_push_end = g_start(1:3,4) + cfg.pushDist * pushDir;
g_push_end = rtde_make_pose(R_ref, p_push_end);

[success, q_A] = solveIKAndMove(g_push_end, q_start, cfg.timeToReturnToStart, "A_push_forward");
if ~success
    out.msg = "Failed in Segment A: IK failed";
    return;
end

segA = struct('segment', "A_push_forward", 'q_final', q_A, 'msg', "OK");
out.segments = segA;

%% Segment B: Return to start position
fprintf("\n=== Segment B: Return to start position ===\n");
[success, q_B] = solveIKAndMove(g_start, q_A, cfg.timeToReturnToStart, "B_return_to_start");
if ~success
    out.msg = "Failed in Segment B: IK failed";
    return;
end

segB = struct('segment', "B_return_to_start", 'q_final', q_B, 'msg', "OK");
out.segments(end+1,1) = segB;

%% Segment C: Lift 10cm above start
fprintf("\n=== Segment C: Lift 10cm ===\n");
p_start_above = g_start(1:3,4);
p_start_above(3) = p_start_above(3) + 0.10; % 10cm up
g_start_above = rtde_make_pose(R_ref, p_start_above);

[success, q_C] = solveIKAndMove(g_start_above, q_B, cfg.timeToReturnToStart, "C_lift");
if ~success
    out.msg = "Failed in Segment C: IK failed";
    return;
end

segC = struct('segment', "C_lift", 'q_final', q_C, 'msg', "OK");
out.segments(end+1,1) = segC;

%% Segment D: Translate horizontally to position(2) above (19cm forward, 10cm up)
fprintf("\n=== Segment D: Translate to position(2) above ===\n");
p_target2_above = g_start(1:3,4) + (cfg.cubeSide + 2 * cfg.pushDist) * pushDir;
p_target2_above(3) = g_start(3,4) + 0.10; % Same 10cm height
g_target2_above = rtde_make_pose(R_ref, p_target2_above);

[success, q_D] = solveIKAndMove(g_target2_above, q_C, cfg.timeToReturnToStart, "D_translate");
if ~success
    out.msg = "Failed in Segment D: IK failed";
    return;
end

segD = struct('segment', "D_translate", 'q_final', q_D, 'msg', "OK");
out.segments(end+1,1) = segD;

%% Segment E: Lower to position(2)
fprintf("\n=== Segment E: Lower to position(2) ===\n");
p_target2 = g_start(1:3,4) + (cfg.cubeSide + 2 * cfg.pushDist) * pushDir;
p_target2(3) = g_start(3,4); % Same height as start
g_target2 = rtde_make_pose(R_ref, p_target2);

[success, q_E] = solveIKAndMove(g_target2, q_D, cfg.timeToReturnToStart, "E_lower");
if ~success
    out.msg = "Failed in Segment E: IK failed";
    return;
end

segE = struct('segment', "E_lower", 'q_final', q_E, 'msg', "OK");
out.segments(end+1,1) = segE;

%% Segment F: Push back 6cm
fprintf("\n=== Segment F: Push back 6cm ===\n");
p_push_back_end = p_target2 - 2 * cfg.pushDist * pushDir; % 6cm back
g_push_back_end = rtde_make_pose(R_ref, p_push_back_end);

[success, q_F] = solveIKAndMove(g_push_back_end, q_E, cfg.timeToReturnToStart, "F_push_back");
if ~success
    out.msg = "Failed in Segment F: IK failed";
    return;
end

segF = struct('segment', "F_push_back", 'q_final', q_F, 'msg', "OK");
out.segments(end+1,1) = segF;

%% Success
out.ok = true;
out.msg = "OK - All segments completed";
fprintf("\n=== Task completed successfully ===\n");

end
