function position_error_cm = five_phase_push_manipulation(robot_interface, robot_model)
    % ========================================
    % five_phase_push_manipulation - Execute push-back manipulation sequence
    %
    % This function performs a 5-phase manipulation:
    %   1. Push forward along planar tool-Y direction
    %   2. Lift vertically
    %   3. Translate horizontally to opposite side
    %   4. Lower to contact height
    %   5. Push back along planar tool-Y direction
    %
    % Arguments:
    %   robot_interface - UR robot interface object
    %   robot_model     - string specifying robot type ('ur5' or 'ur5e')
    %
    % Returns:
    %   position_error_cm - final positioning error in centimeters
    % ========================================

    % ========== Motion Parameters ==========
    PUSH_DISTANCE_M = 0.03;      % Horizontal push distance
    LIFT_HEIGHT_M = 0.09;        % Vertical lift clearance
    LATERAL_OFFSET_M = 0.16;     % Distance to traverse object
    MOTION_TIME_SEC = 3.0;       % Duration per motion segment

    % ========== Initialization ==========
    robot_interface.move_joints(robot_interface.home, 10);
    pause(10);

    % ========== Capture Starting Configuration ==========
    robot_interface.switch_to_pendant_control();
    disp('========================================');
    disp('MANUAL SETUP: Position robot at initial contact pose');
    disp('Press any key when ready to continue...');
    disp('========================================');
    pause;

    initial_joints = robot_interface.get_current_joints();
    robot_interface.switch_to_ros_control();
    disp('>> Control switched to ROS mode.');

    initial_pose = urFwdKin(initial_joints, robot_model);
    reference_orientation = initial_pose(1:3, 1:3);
    reference_position = initial_pose(1:3, 4);
    disp('>> Initial pose recorded successfully.');

    % ========== START ERROR CALCULATION ==========
    % Get actual start pose from robot's TF
    T_start_actual = robot_interface.get_current_transformation('base', 'tool0');
    r_start_actual = T_start_actual(1:3, 4);
    R_start_actual = T_start_actual(1:3, 1:3);

    % Desired start pose is the taught pose (reference)
    r_start_desired = reference_position;
    R_start_desired = reference_orientation;

    % Calculate and display Start errors
    [d_R3_start, d_SO3_start] = robot_interface.calculate_pose_error(...
        "Start", r_start_actual, R_start_actual, r_start_desired, R_start_desired);

    % ========== Compute Planar Push Direction ==========
    planar_direction = compute_planar_tool_y(initial_pose);

    % ========== Execute 5-Phase Manipulation Sequence ==========

    % --- Phase 1: Push Forward ---
    pose_after_push = construct_push_forward_pose(...
        reference_orientation, reference_position, ...
        planar_direction, PUSH_DISTANCE_M);

    disp('[Phase 1/5] Pushing forward...');
    execute_motion_with_ik(robot_interface, robot_model, pose_after_push, MOTION_TIME_SEC);

    % --- Phase 2: Lift Up ---
    pose_lifted = construct_lift_pose(...
        reference_orientation, pose_after_push, LIFT_HEIGHT_M);

    disp('[Phase 2/5] Lifting vertically...');
    execute_motion_with_ik(robot_interface, robot_model, pose_lifted, MOTION_TIME_SEC);

    % --- Phase 3: Move to Opposite Side ---
    pose_opposite_above = construct_lateral_move_pose(...
        reference_orientation, pose_lifted, ...
        planar_direction, LATERAL_OFFSET_M);

    disp('[Phase 3/5] Translating to opposite side...');
    execute_motion_with_ik(robot_interface, robot_model, pose_opposite_above, MOTION_TIME_SEC);

    % --- Phase 4: Lower to Contact ---
    pose_opposite_contact = construct_lower_pose(...
        reference_orientation, pose_opposite_above, ...
        reference_position, LIFT_HEIGHT_M);

    disp('[Phase 4/5] Lowering to contact...');
    execute_motion_with_ik(robot_interface, robot_model, pose_opposite_contact, MOTION_TIME_SEC);

    % --- Phase 5: Push Back ---
    pose_final = construct_push_back_pose(...
        reference_orientation, pose_opposite_contact, ...
        planar_direction, PUSH_DISTANCE_M, reference_position);

    disp('[Phase 5/5] Pushing back...');
    execute_motion_with_ik(robot_interface, robot_model, pose_final, MOTION_TIME_SEC);

    % ========== Evaluate Final Positioning Error ==========
    actual_pose = urFwdKin(robot_interface.get_current_joints(), robot_model);

    % ========== TARGET ERROR CALCULATION ==========
    % Get actual target pose from robot's TF
    T_target_actual = robot_interface.get_current_transformation('base', 'tool0');
    r_target_actual = T_target_actual(1:3, 4);
    R_target_actual = T_target_actual(1:3, 1:3);

    % Desired target pose is the final planned pose
    r_target_desired = pose_final(1:3, 4);
    R_target_desired = pose_final(1:3, 1:3);

    % Calculate and display Target errors
    [d_R3_target, d_SO3_target] = robot_interface.calculate_pose_error(...
        "Target", r_target_actual, R_target_actual, r_target_desired, R_target_desired);

    % ========== Legacy Error Calculation (for backward compatibility) ==========
    % Compute position errors
    position_diff = actual_pose(1:3, 4) - pose_final(1:3, 4);
    position_error_x = position_diff(1) * 100;  % cm
    position_error_y = position_diff(2) * 100;  % cm
    position_error_z = position_diff(3) * 100;  % cm
    position_error_total = norm(position_diff) * 100;  % cm

    % Compute orientation error
    rotation_error_matrix = actual_pose(1:3, 1:3)' * pose_final(1:3, 1:3);
    angle_error_rad = acos((trace(rotation_error_matrix) - 1) / 2);
    angle_error_deg = rad2deg(angle_error_rad);

    % Display detailed error analysis
    fprintf('\n========================================\n');
    fprintf('DETAILED ERROR ANALYSIS (Legacy Format)\n');
    fprintf('========================================\n');
    fprintf('Position Errors:\n');
    fprintf('  X-axis: %+.4f cm\n', position_error_x);
    fprintf('  Y-axis: %+.4f cm\n', position_error_y);
    fprintf('  Z-axis: %+.4f cm\n', position_error_z);
    fprintf('  Total:  %.4f cm\n', position_error_total);
    fprintf('\nOrientation Error:\n');
    fprintf('  Angular: %.4f degrees (%.4f rad)\n', angle_error_deg, angle_error_rad);
    fprintf('========================================\n');

    % ========== SUMMARY OF ALL ERRORS ==========
    fprintf('\n========================================\n');
    fprintf('COMPLETE ERROR SUMMARY\n');
    fprintf('========================================\n');
    fprintf('START Location:\n');
    fprintf('  d_R3 (mm):  %.4f\n', d_R3_start);
    fprintf('  d_SO3:      %.4f\n', d_SO3_start);
    fprintf('\nTARGET Location:\n');
    fprintf('  d_R3 (mm):  %.4f\n', d_R3_target);
    fprintf('  d_SO3:      %.4f\n', d_SO3_target);
    fprintf('========================================\n');

    position_error_cm = position_error_total;
end


% ========== Helper Functions ==========

function planar_dir = compute_planar_tool_y(pose)
    % Extract tool Y-axis (2nd column of rotation matrix)
    tool_y_axis = pose(1:3, 2);

    % Project onto horizontal plane (zero out Z component)
    planar_dir = tool_y_axis;
    planar_dir(3) = 0;

    % Validate and normalize
    magnitude = norm(planar_dir);
    if magnitude < 1e-6
        error('ERROR: Tool Y-axis has insufficient horizontal component. Reorient tool.');
    end
    planar_dir = planar_dir / magnitude;
end


function pose = construct_push_forward_pose(R, p0, direction, distance)
    pose = eye(4);
    pose(1:3, 1:3) = R;
    pose(1:3, 4) = p0 + distance * direction;
    pose(3, 4) = p0(3);  % Maintain height
end


function pose = construct_lift_pose(R, previous_pose, lift_height)
    pose = previous_pose;
    pose(1:3, 1:3) = R;
    pose(1:3, 4) = previous_pose(1:3, 4) + [0; 0; lift_height];
end


function pose = construct_lateral_move_pose(R, previous_pose, direction, offset)
    pose = previous_pose;
    pose(1:3, 1:3) = R;
    pose(1:3, 4) = previous_pose(1:3, 4) + offset * direction;
end


function pose = construct_lower_pose(R, above_pose, reference_pos, lift_height)
    pose = above_pose;
    pose(1:3, 1:3) = R;
    pose(1:3, 4) = above_pose(1:3, 4) - [0; 0; lift_height];
    pose(3, 4) = reference_pos(3);  % Lock to original contact height
end


function pose = construct_push_back_pose(R, contact_pose, direction, distance, reference_pos)
    pose = contact_pose;
    pose(1:3, 1:3) = R;
    pose(1:3, 4) = contact_pose(1:3, 4) - distance * direction;
    pose(3, 4) = reference_pos(3);  % Maintain contact height
end


function execute_motion_with_ik(robot, model, target_pose, duration)
    % Get current configuration
    current_config = robot.get_current_joints();

    % Solve inverse kinematics
    ik_candidates = urInvKin(target_pose, model);
    [optimal_config, sol_idx] = selectClosestIK(ik_candidates, current_config);

    % Validate solution
    if isempty(optimal_config)
        error('ERROR: No valid IK solution exists for waypoint.');
    end

    fprintf('  >> Using IK solution #%d\n', sol_idx);

    % Execute motion
    robot.move_joints(optimal_config, duration);
    pause(duration + 0.2);
end
