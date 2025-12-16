function [xi, gst0] = urGetScrews(type)
    % Get screw axes and home configuration for UR robot
    %
    % Input:
    %   type: string ('ur5' or 'ur5e')
    %
    % Output:
    %   xi: 6x6 matrix, each column is a screw axis [w; v]
    %   gst0: 4x4 home configuration matrix
    %
    if strcmp(type, 'ur5')
        % UR5 dimensions (in mm)
        L0 = 0.0892; 
        L1 = 0.425;
        L2 = 0.392;
        L3 = 0.1093;
        L4 = 0.09475;
        L5 = 0.0825;
    elseif strcmp(type, 'ur5e')
        % UR5e dimensions (in mm)
        L0 = 0.1625;
        L1 = 0.425;
        L2 = 0.392;
        L3 = 0.1333;
        L4 = 0.0997;
        L5 = 0.0996;
    else
        error('Type must be either "ur5" or "ur5e"');
    end

    % Home configuration gst(0)
    gst0 = [1,  0,  0,  0;
            0,  1,  0, L3+L5;
            0,  0,  1,  L0+L1+L2+L4;
            0,  0,  0,  1];

    % Screw axes in base frame (spatial twist coordinates)
    % Format: ξ = [ω; v] where ω is unit rotation axis, v = -ω × q

    % Joint 1: rotation about z-axis at origin [0,0,0]
    w1 = [0; 0; 1];
    q1 = [0; 0; 0];
    v1 = -cross(w1, q1);
    xi1 = [w1; v1];

    % Joint 2: rotation about y-axis at [0, 0, L0]
    w2 = [0; 1; 0];
    q2 = [0; 0; L0];
    v2 = -cross(w2, q2);
    xi2 = [w2; v2];

    % Joint 3: rotation about y-axis at [L1, 0, L0]
    w3 = [0; 1; 0];
    q3 = [0; 0; L0+L1];
    v3 = -cross(w3, q3);
    xi3 = [w3; v3];

    % Joint 4: rotation about y-axis at [L1+L2, 0, L0]
    w4 = [0; 1; 0];
    q4 = [0; 0; L0+L1+L2];
    v4 = -cross(w4, q4);
    xi4 = [w4; v4];

    % Joint 5: rotation about z-axis at [L1+L2+L3, 0, L0]
    w5 = [0; 0; 1];
    q5 = [0; L3; L0+L1+L2];
    v5 = -cross(w5, q5);
    xi5 = [w5; v5];

    % Joint 6: rotation about y-axis at [L1+L2+L3, 0, L0-L4]
    w6 = [0; 1; 0];
    q6 = [0; L3+L5; L0+L1+L2+L4];
    v6 = -cross(w6, q6);
    xi6 = [w6; v6];

    % Pack all screw axes into a 6x6 matrix
    xi = [xi1, xi2, xi3, xi4, xi5, xi6];
end
