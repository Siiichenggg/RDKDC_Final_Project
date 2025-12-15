function out = rtde_push_and_place_task(ur, q_start, g_start, cfg)
%RTDE_PUSH_AND_PLACE_TASK Execute push-and-place using RR + position-step commands.

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

% Return to taught start pose deterministically
ur.move_joints(q_start, cfg.timeToReturnToStart);
pause(cfg.timeToReturnToStart);

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

% Segment C: Return to UR5e default home position
fprintf("\n=== Returning to UR5e home position ===\n");
q_home = [0, -pi/2, 0, -pi/2, 0, 0]'; % UR5e default home position
ur.move_joints(q_home, cfg.timeToReturnToStart);
pause(cfg.timeToReturnToStart);
segC = struct('segment',"return_to_home",'converged',true,'steps',0,'posErr',0,'rotErr',0,'sigmaMin',NaN,'condJ',NaN,'maxAbsDq',0,'q_final',q_home,'msg',"Returned to UR5e home position");
out.segments(end+1,1) = segC;

% Segment D: Compute target position (2) for push-back
% Position (2) is: start position (1) + cube side length (13cm) along push direction
% This positions the robot on the other side of the cube
p_target2 = g_start(1:3,4) + cfg.cubeSide * pushDir;
p_target2(3) = g_start(3,4); % Same height as the start position
g_target2 = rtde_make_pose(R_ref, p_target2);

segD = rtde_rr_move_to_pose(ur, g_target2, cfg, "move_to_other_side");
out.segments(end+1,1) = segD;
if ~segD.converged
    out.msg = "Failed in move_to_other_side: " + segD.msg;
    return;
end

% Segment E: Push back ~3cm (target2 -> back)
% Manual requirement: push the cube back near the original location
q_contact2 = segD.q_final;
g_contact2 = rtde_urFwdKin(q_contact2, cfg.robotType);

p_push_back_end = g_contact2(1:3,4) - cfg.pushDist * pushDir;
g_push_back_end = rtde_make_pose(R_ref, p_push_back_end);

segE = rtde_rr_move_to_pose(ur, g_push_back_end, cfg, "push_back");
out.segments(end+1,1) = segE;
if ~segE.converged
    out.msg = "Failed in push_back: " + segE.msg;
    return;
end

out.ok = true;
out.msg = "OK";
end
