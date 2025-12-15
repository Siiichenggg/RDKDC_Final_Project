function gst = urFwdKin(q, type)
% urFwdKin - Forward Kinematics for UR5/UR5e
% Input: q - 6x1 joint angles (radians)
%        type - string 'ur5' or 'ur5e' (default 'ur5e')
% Output: gst - 4x4 Homogeneous Transformation Matrix (Base -> Tool0)

    if nargin < 2
        type = 'ur5e';
    end

    % DH Parameters (Aligned with urInvKin.m)
    % [a, alpha, d, theta_offset]
    % Note: alpha is fixed in the loop usually, but we can store it here.
    % Using the structure from urInvKin:
    
    % Default UR5 params
    d = [0.089159; 0; 0; 0.10915; 0.09465; 0.0823];
    a = [0; -0.425; -0.39225; 0; 0; 0];
    alpha = [pi/2; 0; 0; pi/2; -pi/2; 0];
    
    % UR5e params overrides
    if strcmp(type, 'ur5e')
        d(1) = 0.1625;
        d(4) = 0.1333;
        d(5) = 0.0997;
        d(6) = 0.0996;
        a(2) = -0.425;
        a(3) = -0.3922;
    end
    
    % Theta offsets (Inferred from urInvKin.m logic)
    % urInvKin subtracts pi from theta1 result, implying the FK model has +pi offset on joint 1
    offset = [pi; 0; 0; 0; 0; 0];
    
    gst = eye(4);
    
    for i = 1:6
        % Call the professor's DH function
        % DH(a, alpha, d, theta)
        A_i = DH(a(i), alpha(i), d(i), q(i) + offset(i));
        gst = gst * A_i;
    end
end

