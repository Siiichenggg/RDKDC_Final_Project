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

% Segment B: Return to default "home" joint configuration (no绕行平移段)
q_home = [];
if isprop(ur, 'home')
    q_home = ur.home;
end
if isempty(q_home) || ~isnumeric(q_home) || numel(q_home) ~= 6
    out.msg = "ABORT: cannot resolve ur.home (default initial joint configuration).";
    return;
end
q_home = reshape(q_home, [6, 1]);
ur.move_joints(q_home, cfg.timeToHome);
pause(cfg.timeToHome);
segB = struct('segment',"return_home",'converged',true,'steps',0,'posErr',0,'rotErr',0,'sigmaMin',NaN,'condJ',NaN,'maxAbsDq',0,'q_final',q_home,'msg',"OK");
out.segments(end+1,1) = segB;

% Segment C: From home, go above target (2) computed from end, then descend to contact height
pause(max(0, cfg.timeHomeToTarget2));

% Target (2) computed from the end location (end) of the push_forward trajectory
p_target2_xy = g_end(1:3,4) + cfg.backApproachExtra * pushDir;
z_contact = g_end(3,4);
z_above = z_contact + max(0, cfg.liftHeight);

g_target2_above = rtde_make_pose(R_ref, [p_target2_xy(1); p_target2_xy(2); z_above]);
segC = rtde_rr_move_to_pose(ur, g_target2_above, cfg, "go_above_target2");
out.segments(end+1,1) = segC;
if ~segC.converged
    out.msg = "Failed in go_above_target2: " + segC.msg;
    return;
end

g_target2_contact = rtde_make_pose(R_ref, [p_target2_xy(1); p_target2_xy(2); z_contact]);
segD = rtde_rr_move_to_pose(ur, g_target2_contact, cfg, "descend_to_target2");
out.segments(end+1,1) = segD;
if ~segD.converged
    out.msg = "Failed in descend_to_target2: " + segD.msg;
    return;
end

% Segment E: Push back by pushDist (target2 -> back)
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

% Optional: end at home (begin/end at a home configuration you decide)
ur.move_joints(q_home, cfg.timeToHome);
pause(cfg.timeToHome);
segF = struct('segment',"end_home",'converged',true,'steps',0,'posErr',0,'rotErr',0,'sigmaMin',NaN,'condJ',NaN,'maxAbsDq',0,'q_final',q_home,'msg',"OK");
out.segments(end+1,1) = segF;

out.ok = true;
out.msg = "OK";
end
