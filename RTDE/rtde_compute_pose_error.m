function [d_R3_mm, d_SO3] = rtde_compute_pose_error(g_actual, g_desired)
%RTDE_COMPUTE_POSE_ERROR Compute pose error metrics.
%
% Inputs:
%   g_actual  - 4x4 actual pose (homogeneous transform)
%   g_desired - 4x4 desired pose (homogeneous transform)
%
% Outputs:
%   d_R3_mm - Position error in millimeters
%   d_SO3   - Rotation error (unitless)
%
% Formulas:
%   d_R3 = ||r - r_d|| * 1000   (convert meters to millimeters)
%   d_SO3 = sqrt(trace((R - R_d)(R - R_d)^T))

% Extract position and rotation from actual pose
r = g_actual(1:3, 4);
R = g_actual(1:3, 1:3);

% Extract position and rotation from desired pose
r_d = g_desired(1:3, 4);
R_d = g_desired(1:3, 1:3);

% Compute position error in millimeters
d_R3_mm = 1000 * norm(r - r_d);

% Compute rotation error
R_diff = R - R_d;
d_SO3 = sqrt(trace(R_diff * R_diff'));

end
