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

% Return to taught start pose deterministically
ur.move_joints(q_start, cfg.timeToReturnToStart);
pause(cfg.timeToReturnToStart);

% Segment A: Push cube ~3cm (start -> end)
g_push_end = g_start;
g_push_end(1:3,4) = g_start(1:3,4) + cfg.pushDist * pushDir;

segA = rtde_rr_move_to_pose(ur, g_push_end, cfg, "push_forward");
out.segments = segA;
if ~segA.converged
    out.msg = "Failed in push_forward: " + segA.msg;
    return;
end

q_end = segA.q_final;
g_end = rtde_urFwdKin(q_end, cfg.robotType);

% Segment B: Lift (safety) before re-approach to other side
if cfg.liftHeight > 0
    g_lift = g_end;
    g_lift(3,4) = g_end(3,4) + cfg.liftHeight;
    segB = rtde_rr_move_to_pose(ur, g_lift, cfg, "lift");
    out.segments(end+1,1) = segB;
    if ~segB.converged
        out.msg = "Failed in lift: " + segB.msg;
        return;
    end
else
    segB = struct('segment',"lift",'converged',true,'steps',0,'posErr',0,'rotErr',0,'sigmaMin',NaN,'condJ',NaN,'maxAbsDq',0,'q_final',q_end,'msg',"SKIP");
    out.segments(end+1,1) = segB;
end

% Segment C: Move to "target (2)" computed from (end) to other side
% (computed from last location end, as required)
q_afterLift = out.segments(end).q_final;
g_afterLift = rtde_urFwdKin(q_afterLift, cfg.robotType);

g_target2 = g_afterLift;
g_target2(1:3,4) = g_afterLift(1:3,4) + cfg.backApproachExtra * pushDir;

segC = rtde_rr_move_to_pose(ur, g_target2, cfg, "move_to_other_side");
out.segments(end+1,1) = segC;
if ~segC.converged
    out.msg = "Failed in move_to_other_side: " + segC.msg;
    return;
end

% Segment D: Lower back near surface (optional, to match taught contact height)
q_afterMove = segC.q_final;
g_afterMove = rtde_urFwdKin(q_afterMove, cfg.robotType);

g_lower = g_afterMove;
g_lower(3,4) = g_end(3,4);
segD = rtde_rr_move_to_pose(ur, g_lower, cfg, "lower_to_surface");
out.segments(end+1,1) = segD;
if ~segD.converged
    out.msg = "Failed in lower_to_surface: " + segD.msg;
    return;
end

% Segment E: Push back ~3cm (target2 -> back)
q_contact2 = segD.q_final;
g_contact2 = rtde_urFwdKin(q_contact2, cfg.robotType);

g_push_back_end = g_contact2;
g_push_back_end(1:3,4) = g_contact2(1:3,4) - cfg.pushDist * pushDir;

segE = rtde_rr_move_to_pose(ur, g_push_back_end, cfg, "push_back");
out.segments(end+1,1) = segE;
if ~segE.converged
    out.msg = "Failed in push_back: " + segE.msg;
    return;
end

out.ok = true;
out.msg = "OK";
end
