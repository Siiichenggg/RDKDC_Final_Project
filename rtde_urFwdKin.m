function gst = rtde_urFwdKin(q, type)
%RTDE_URFWDKIN Forward kinematics (no relative addpath side effects).
%
% q:    6x1 joint angles [rad]
% type: "ur5" or "ur5e"
% gst:  4x4 homogeneous transform (base -> tool0)

[xi, gst0] = urGetScrews(type); % xi is [w; v]

gst = EXPCF(xi(:,1) * q(1)) * EXPCF(xi(:,2) * q(2)) * EXPCF(xi(:,3) * q(3)) * ...
      EXPCF(xi(:,4) * q(4)) * EXPCF(xi(:,5) * q(5)) * EXPCF(xi(:,6) * q(6)) * gst0;
end

