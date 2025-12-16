function out = rtde_push_and_place_task(ur, q_start, g_start, cfg)
%RTDE_PUSH_AND_PLACE_TASK Execute push-and-place using RR + position-step commands.

out = struct();
out.ok = false;
out.segments = struct([]);
out.msg = "";
out.errors = struct([]);

% Normalize direction
pushDir = cfg.pushDirBase(:);
if norm(pushDir) < 1e-9
    error('cfg.pushDirBase must be non-zero.');
end
pushDir = pushDir / norm(pushDir);

% Keep world orientation fixed to the taught start orientation
R_ref = g_start(1:3, 1:3);

% Return to taught start pose deterministically
ur.move_joints(q_start, cfg.timeToReturnToStart);
pause(cfg.timeToReturnToStart);

% ========== START ERROR MEASUREMENT ==========
fprintf("\n=== Measuring Start Pose Error ===\n");
q_start_actual = ur.get_current_joints();
g_start_actual = rtde_urFwdKin(q_start_actual, cfg.robotType);
[d_R3_start, d_SO3_start] = rtde_compute_pose_error(g_start_actual, g_start);
fprintf("Start Error - d_R3: %.4f mm, d_SO3: %.6f\n", d_R3_start, d_SO3_start);

% Record start error
error_start = struct();
error_start.location = "Start";
error_start.d_R3_mm = d_R3_start;
error_start.d_SO3 = d_SO3_start;
out.errors = error_start;

% Segment A: Push cube ~3cm (start -> end)
p_push_end = g_start(1:3,4) + cfg.pushDist * pushDir;
g_push_end = rtde_make_pose(R_ref, p_push_end);

segA = rtde_rr_move_to_pose(ur, g_push_end, cfg, "push_forward");
out.segments = segA;
if ~segA.converged
    out.msg = "Failed in push_forward: " + segA.msg;
    return;
end

q_end = segA.q_final;
g_end = rtde_urFwdKin(q_end, cfg.robotType);

% Segment B: Return to taught start position (before push)
fprintf("\n=== Returning to taught start position (before push) ===\n");
segB = rtde_rr_move_to_pose(ur, g_start, cfg, "return_to_start");
out.segments(end+1,1) = segB;
if ~segB.converged
    out.msg = "Failed in return_to_start: " + segB.msg;
    return;
end

% Segment C: Lift to position (1) above (10cm) - safe intermediate point
p_start_above = g_start(1:3,4);
p_start_above(3) = g_start(3,4) + 0.10; % 10cm above position (1)
g_start_above = rtde_make_pose(R_ref, p_start_above);

fprintf("\n=== Lifting to position (1) above (safe intermediate point) ===\n");
segC = rtde_rr_move_to_pose(ur, g_start_above, cfg, "lift_to_start_above");
out.segments(end+1,1) = segC;
if ~segC.converged
    out.msg = "Failed in lift_to_start_above: " + segC.msg;
    return;
end

% Segment D: Horizontally translate to position (2) above with 6cm safety compensation
% Position (2) = g_start + 13cm (cube side) + 6cm (safety compensation) = g_start + 19cm
p_target2_above = g_start(1:3,4) + (cfg.cubeSide + 2 * cfg.pushDist) * pushDir;
p_target2_above(3) = g_start(3,4) + 0.10; % 10cm above, same height
g_target2_above = rtde_make_pose(R_ref, p_target2_above);

fprintf("\n=== Translating horizontally to position (2) above (with 6cm compensation) ===\n");
segD = rtde_rr_move_to_pose(ur, g_target2_above, cfg, "translate_to_target2_above");
out.segments(end+1,1) = segD;
if ~segD.converged
    out.msg = "Failed in translate_to_target2_above: " + segD.msg;
    return;
end

% Segment E: Lower to position (2) at contact height
p_target2 = g_start(1:3,4) + (cfg.cubeSide + 2 * cfg.pushDist) * pushDir;
p_target2(3) = g_start(3,4); % Same height as the start position
g_target2 = rtde_make_pose(R_ref, p_target2);

fprintf("\n=== Lowering to position (2) ===\n");
segE = rtde_rr_move_to_pose(ur, g_target2, cfg, "lower_to_position2");
out.segments(end+1,1) = segE;
if ~segE.converged
    out.msg = "Failed in lower_to_position2: " + segE.msg;
    return;
end

% Segment F: Push back 6cm (which pushes the cube back 3cm due to 3cm compensation)
% Manual requirement: push the cube back near the original location
q_contact2 = segE.q_final;
g_contact2 = rtde_urFwdKin(q_contact2, cfg.robotType);

p_push_back_end = g_contact2(1:3,4) - 2 * cfg.pushDist * pushDir; % 6cm = 2 * 3cm
g_push_back_end = rtde_make_pose(R_ref, p_push_back_end);

fprintf("\n=== Pushing back 6cm (cube moves 3cm back) ===\n");
segF = rtde_rr_move_to_pose(ur, g_push_back_end, cfg, "push_back");
out.segments(end+1,1) = segF;
if ~segF.converged
    out.msg = "Failed in push_back: " + segF.msg;
    return;
end

% ========== TARGET ERROR MEASUREMENT ==========
fprintf("\n=== Measuring Target Pose Error ===\n");
q_target_actual = segF.q_final;
g_target_actual = rtde_urFwdKin(q_target_actual, cfg.robotType);
[d_R3_target, d_SO3_target] = rtde_compute_pose_error(g_target_actual, g_push_back_end);
fprintf("Target Error - d_R3: %.4f mm, d_SO3: %.6f\n", d_R3_target, d_SO3_target);

% Record target error
error_target = struct();
error_target.location = "Target";
error_target.d_R3_mm = d_R3_target;
error_target.d_SO3 = d_SO3_target;
out.errors(end+1,1) = error_target;

out.ok = true;
out.msg = "OK";
end
