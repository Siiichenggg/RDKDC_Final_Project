function Jb = rtde_urBodyJacobian(q, type)
%RTDE_URBODYJACOBIAN Body Jacobian in [v; w] ordering (no relative addpath side effects).

% urGetScrews returns screw axes as [w; v]
[xi_wv, gst0] = urGetScrews(type);

% Convert to [v; w] ordering for adjoint() (and consistency with getXi()).
xi_vw = [xi_wv(4:6, :); xi_wv(1:3, :)];

% Compute spatial Jacobian in [v; w] ordering
Js = zeros(6, 6);
g = eye(4);
Js(:, 1) = xi_vw(:, 1);
for i = 2:6
    % EXPCF expects [w; v], so swap back when calling it
    g = g * EXPCF([xi_vw(4:6, i-1); xi_vw(1:3, i-1)] * q(i-1));
    Js(:, i) = adjoint(g) * xi_vw(:, i);
end

% Forward kinematics to current configuration
g = g * EXPCF([xi_vw(4:6, 6); xi_vw(1:3, 6)] * q(6));
gst = g * gst0;

% Body Jacobian: J_b = Ad_{g^{-1}} J_s
Jb = adjoint(FINV(gst)) * Js;
end

