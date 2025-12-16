function rotation_matrix = ROTX(angle_rad)
    % ========================================
    % ROTX: Generate rotation matrix about X-axis
    %
    % Arguments:
    %   angle_rad - rotation angle in radians
    %
    % Returns:
    %   rotation_matrix - 3x3 SO(3) rotation matrix
    % ========================================

    % Compute trigonometric values
    cos_val = cos(angle_rad);
    sin_val = sin(angle_rad);

    % Construct rotation matrix using standard formula
    rotation_matrix = [
        1,       0,        0;
        0, cos_val, -sin_val;
        0, sin_val,  cos_val
    ];
end

