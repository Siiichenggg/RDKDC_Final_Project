function jst_b = urBodyJacobian(q, type)
    % urBodyJacobian: Computes the Body Jacobian for a UR robot (UR5 or UR5e)
    %
    % Inputs:
    %   q    - 6x1 vector of joint angles [θ1, θ2, ..., θ6]'
    %   type - 'ur5' or 'ur5e', specifies the robot type
    %
    % Output:
    %   jst_b - 6x6 Body Jacobian matrix
    %
    % Each column J^b_i is calculated as:
    %   J^b_i = Ad_{e^{-[ξ_6]θ_6} ⋯ e^{-[ξ_{i}]θ_{i}}} * ξ_i

    % Add helper function path (contains adjoint, EXPCF, etc.)
    addpath("./helper_function/")

    % Get screw axes (returned as [ω; v]) and home configuration
    [xi_wv, gst0] = urGetScrews(type);

    % Convert screws to the [v; ω] ordering used by getXi/adjoint
    xi_vw = [xi_wv(4:6, :); xi_wv(1:3, :)];

    % Compute spatial Jacobian J_s first
    Js = zeros(6, 6);
    g = eye(4);
    Js(:, 1) = xi_vw(:, 1);
    for i = 2:6
        % EXPCF expects [ω; v], so swap back when calling it
        g = g * EXPCF([xi_vw(4:6, i-1); xi_vw(1:3, i-1)] * q(i-1));
        Js(:, i) = adjoint(g) * xi_vw(:, i);
    end

    % Forward kinematics to current configuration
    g = g * EXPCF([xi_vw(4:6, 6); xi_vw(1:3, 6)] * q(6));
    gst = g * gst0;

    % Body Jacobian: J_b = Ad_{g^{-1}} J_s
    jst_b = adjoint(FINV(gst)) * Js;
end
