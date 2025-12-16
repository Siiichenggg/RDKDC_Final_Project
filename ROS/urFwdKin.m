function transformation_matrix = urFwdKin(joint_angles, robot_type)
    % ========================================
    % urFwdKin - Compute Forward Kinematics for UR5/UR5e robots
    %
    % Arguments:
    %   joint_angles - 6x1 vector of joint angles in radians
    %   robot_type   - string specifying robot model ('ur5' or 'ur5e')
    %                  defaults to 'ur5e' if not provided
    %
    % Returns:
    %   transformation_matrix - 4x4 homogeneous transformation matrix
    %                          representing base-to-tool0 pose
    % ========================================

    % Set default robot type if not specified
    if nargin < 2
        robot_type = 'ur5e';
    end

    % Initialize DH parameters structure
    dh_params = initialize_dh_parameters(robot_type);

    % Extract DH parameters
    link_lengths = dh_params.a;
    twist_angles = dh_params.alpha;
    link_offsets = dh_params.d;
    joint_offsets = dh_params.theta_offset;

    % Initialize transformation as identity matrix
    transformation_matrix = eye(4);

    % Chain all link transformations
    num_joints = 6;
    for joint_idx = 1:num_joints
        % Compute effective joint angle with offset
        effective_angle = joint_angles(joint_idx) + joint_offsets(joint_idx);

        % Get transformation for current link using DH convention
        link_transform = DH(link_lengths(joint_idx), ...
                           twist_angles(joint_idx), ...
                           link_offsets(joint_idx), ...
                           effective_angle);

        % Multiply transformations
        transformation_matrix = transformation_matrix * link_transform;
    end
end


function dh_params = initialize_dh_parameters(robot_type)
    % ========================================
    % Initialize DH parameters based on robot type
    %
    % Arguments:
    %   robot_type - string 'ur5' or 'ur5e'
    %
    % Returns:
    %   dh_params - structure containing DH parameters
    % ========================================

    % Initialize with UR5 standard parameters
    dh_params.d = [0.089159; 0; 0; 0.10915; 0.09465; 0.0823];
    dh_params.a = [0; -0.425; -0.39225; 0; 0; 0];
    dh_params.alpha = [pi/2; 0; 0; pi/2; -pi/2; 0];

    % Override with UR5e specific parameters if needed
    if strcmp(robot_type, 'ur5e')
        dh_params.d(1) = 0.1625;
        dh_params.d(4) = 0.1333;
        dh_params.d(5) = 0.0997;
        dh_params.d(6) = 0.0996;
        dh_params.a(2) = -0.425;
        dh_params.a(3) = -0.3922;
    end

    % Joint angle offsets (aligned with urInvKin convention)
    dh_params.theta_offset = [pi; 0; 0; 0; 0; 0];
end

