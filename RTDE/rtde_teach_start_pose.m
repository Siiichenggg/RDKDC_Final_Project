function [q_start, g_start] = rtde_teach_start_pose(ur, robotType)
%RTDE_TEACH_START_POSE Teach (record) the start pose from the pendant.
%
% Returns:
%   q_start: 6x1 joint angles [rad]
%   g_start: 4x4 SE(3) transform from base to tool0 (from urFwdKin)

fprintf("\n=== Teach Start Pose ===\n");
fprintf("1) Use the pendant to enter Freedrive (if needed).\n");
fprintf("2) Manually move UR5e to the START pose near the cube/board.\n");
input("3) Press ENTER to record the current joint configuration...", "s");

q_start = ur.get_current_joints();
g_start = urFwdKin(q_start, robotType);
end

