function g = ur5e_fkine(q)
%UR5E_FKINE Forward kinematics for UR5e (base -> tool0) with offset.
%   Uses Robotics System Toolbox model if available, otherwise falls back
%   to a simple DH-based model. The returned frame matches the orientation
%   convention used by ur_rtde_interface (extra Rz(pi) offset).
q = reshape(q, [6, 1]);
[robot, R_offset] = load_ur5e_model();

if ~isempty(robot)
    g_raw = getTransform(robot, q, 'tool0');
else
    g_raw = fkine_fallback(q);
end

offset = eye(4);
offset(1:3,1:3) = R_offset;
g = offset * g_raw;
end

function J = ur5e_geometric_jacobian(q)
%UR5E_GEOMETRIC_JACOBIAN Spatial Jacobian (expressed in base frame).
q = reshape(q, [6, 1]);
[robot, R_offset] = load_ur5e_model();

if ~isempty(robot)
    J_raw = geometricJacobian(robot, q, 'tool0');
else
    J_raw = jacobian_fallback(q);
end

adj = blkdiag(R_offset, R_offset);
J = adj * J_raw;
end

function [robot, R_offset] = load_ur5e_model()
%LOAD_UR5E_MODEL Return persistent rigidBodyTree if available.
persistent robotModel ROff;
if isempty(ROff)
    ROff = rotz(pi); % match ur_rtde_interface offset (flip x/y)
end

if isempty(robotModel)
    try
        robotModel = loadrobot("universalUR5e", "DataFormat", "column", "Gravity", [0 0 0]);
    catch
        robotModel = [];
        warning('Robotics System Toolbox model not found. Falling back to DH kinematics.');
    end
end

robot = robotModel;
R_offset = ROff;
end

function g = fkine_fallback(q)
%FKINE_FALLBACK Simple standard DH forward kinematics for UR5e.
d1 = 0.1625;
a2 = -0.425;
a3 = -0.3922;
d4 = 0.1333;
d5 = 0.0997;
d6 = 0.0996;

dh = [0,     pi/2, d1, q(1);
      a2,    0,    0,  q(2);
      a3,    0,    0,  q(3);
      0,     pi/2, d4, q(4);
      0,    -pi/2, d5, q(5);
      0,     0,    d6, q(6)];

T = eye(4);
for i = 1:6
    a = dh(i,1); alpha = dh(i,2); d = dh(i,3); theta = dh(i,4);
    T = T * dh_transform(a, alpha, d, theta);
end
g = T;
end

function J = jacobian_fallback(q)
%JACOBIAN_FALLBACK Geometric Jacobian via standard DH (base frame).
d1 = 0.1625;
a2 = -0.425;
a3 = -0.3922;
d4 = 0.1333;
d5 = 0.0997;
d6 = 0.0996;

dh = [0,     pi/2, d1, q(1);
      a2,    0,    0,  q(2);
      a3,    0,    0,  q(3);
      0,     pi/2, d4, q(4);
      0,    -pi/2, d5, q(5);
      0,     0,    d6, q(6)];

T = eye(4);
z = zeros(3,6);
p = zeros(3,6);
for i = 1:6
    a = dh(i,1); alpha = dh(i,2); d = dh(i,3); theta = dh(i,4);
    T = T * dh_transform(a, alpha, d, theta);
    z(:,i) = T(1:3,3);
    p(:,i) = T(1:3,4);
end
pe = p(:,6);

J = zeros(6,6);
for i = 1:6
    J(1:3,i) = cross(z(:,i), pe - p(:,i));
    J(4:6,i) = z(:,i);
end
end

function T = dh_transform(a, alpha, d, theta)
%DH_TRANSFORM Standard DH homogeneous transform.
ct = cos(theta); st = sin(theta);
ca = cos(alpha); sa = sin(alpha);

T = [ct, -st*ca, st*sa, a*ct;
     st, ct*ca, -ct*sa, a*st;
     0,  sa,    ca,     d;
     0,  0,     0,      1];
end

function R = rotz(theta)
%ROTZ Rotation about z-axis.
ct = cos(theta); st = sin(theta);
R = [ct, -st, 0; st, ct, 0; 0, 0, 1];
end
