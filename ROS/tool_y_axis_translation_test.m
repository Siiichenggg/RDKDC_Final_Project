function position_error_cm = tool_y_axis_translation_test(robot_interface, robot_model)
    % ========================================
    % tool_y_axis_translation_test - Move robot 3cm along tool Y-axis
    %
    % Arguments:
    %   robot_interface - UR robot interface object
    %   robot_model     - string specifying robot type ('ur5' or 'ur5e')
    %
    % Returns:
    %   position_error_cm - final positioning error in centimeters
    % ========================================

    % ========== Configuration Parameters ==========
    DISPLACEMENT_METERS = 0.03;     % Translation distance (3 cm)
    MOTION_DURATION_SEC = 3.0;      % Time allowed for movement
    WAIT_BUFFER_SEC = 0.5;          % Additional settling time

    % ========== Step 1: Initialize Robot to Home Position ==========
    robot_interface.move_joints(robot_interface.home, 10);

    % ========== Step 2: Capture Starting Configuration ==========
    robot_interface.switch_to_pendant_control();
    disp('>> Manual positioning required. Move robot to start pose, then press any key.');
    pause;

    current_joints = robot_interface.get_current_joints();
    robot_interface.switch_to_ros_control();

    start_pose = urFwdKin(current_joints, robot_model);
    disp('>> Starting pose has been recorded successfully.');

    % ========== Step 3: Extract Tool Y-Axis Direction ==========
    % Tool Y-axis is the second column of rotation matrix
    tool_y_direction = start_pose(1:3, 2);

    % ========== Step 4: Construct Target Pose ==========
    target_pose = start_pose;
    % Maintain orientation (copy rotation matrix)
    target_pose(1:3, 1:3) = start_pose(1:3, 1:3);
    % Translate position along tool Y-axis
    target_pose(1:3, 4) = start_pose(1:3, 4) + DISPLACEMENT_METERS * tool_y_direction;

    % ========== Step 5: Solve Inverse Kinematics ==========
    ik_solutions = urInvKin(target_pose, robot_model);  % Returns 6x8 matrix

    [selected_joints, solution_idx] = selectClosestIK(ik_solutions, current_joints);

    % Validate IK solution
    if isempty(selected_joints)
        error('ERROR: No feasible IK solution exists for the target pose.');
    end

    fprintf('>> Selected IK solution index: %d\n', solution_idx);

    % ========== Step 6: Execute Motion ==========
    robot_interface.move_joints(selected_joints, MOTION_DURATION_SEC);
    pause(MOTION_DURATION_SEC + WAIT_BUFFER_SEC);

    % ========== Step 7: Evaluate Positioning Accuracy ==========
    actual_pose = urFwdKin(robot_interface.get_current_joints(), robot_model);

    % Compute position errors
    position_diff = actual_pose(1:3, 4) - target_pose(1:3, 4);
    position_error_x = position_diff(1) * 100;  % cm
    position_error_y = position_diff(2) * 100;  % cm
    position_error_z = position_diff(3) * 100;  % cm
    position_error_total = norm(position_diff) * 100;  % cm

    % Compute orientation error
    rotation_error_matrix = actual_pose(1:3, 1:3)' * target_pose(1:3, 1:3);
    angle_error_rad = acos((trace(rotation_error_matrix) - 1) / 2);
    angle_error_deg = rad2deg(angle_error_rad);

    % Display detailed error analysis
    fprintf('\n========================================\n');
    fprintf('DETAILED ERROR ANALYSIS\n');
    fprintf('========================================\n');
    fprintf('Position Errors:\n');
    fprintf('  X-axis: %+.4f cm\n', position_error_x);
    fprintf('  Y-axis: %+.4f cm\n', position_error_y);
    fprintf('  Z-axis: %+.4f cm\n', position_error_z);
    fprintf('  Total:  %.4f cm\n', position_error_total);
    fprintf('\nOrientation Error:\n');
    fprintf('  Angular: %.4f degrees (%.4f rad)\n', angle_error_deg, angle_error_rad);
    fprintf('========================================\n');

    position_error_cm = position_error_total;
end
