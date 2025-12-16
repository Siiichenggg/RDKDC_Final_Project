% ========================================
% Error Calculation Example
%
% This example shows how to calculate and display pose errors
% at Start and Target locations using the ur_interface class
% ========================================

%% Initialize robot interface
ur = ur_interface();

%% A. Start 时刻 - After teaching start pose
% 1. Teach the start pose (use freedrive or manual control)
fprintf('Please move robot to START position using freedrive mode...\n');
ur.enable_freedrive();
% Wait for user to position the robot and confirm
input('Press Enter when robot is at START position...');
ur.switch_to_ros_control();
pause(1.0);

% 2. Record the desired start pose (this becomes your reference)
T_start_desired = ur.get_current_transformation('base', 'tool0');
r_d_start = T_start_desired(1:3, 4);  % Desired position (3x1)
R_d_start = T_start_desired(1:3, 1:3); % Desired rotation (3x3)

% 3. Get actual pose (should be very close to desired at this moment)
T_start_actual = ur.get_current_transformation('base', 'tool0');
r_start = T_start_actual(1:3, 4);
R_start = T_start_actual(1:3, 1:3);

% 4. Calculate and display Start error
[d_R3_start, d_SO3_start] = ur.calculate_pose_error(...
    "Start", r_start, R_start, r_d_start, R_d_start);

%% Execute your control strategy (e.g., pushing, reaching, etc.)
fprintf('\nExecuting control strategy...\n');

% Example: Define target pose for your strategy
% This could be from trajectory planning, pushing strategy, etc.
r_d_target = r_d_start + [0.1; 0; 0];  % Example: move 10cm in x direction
R_d_target = R_d_start;  % Keep same orientation

% TODO: Replace with your actual control strategy
% For example:
% - Trajectory generation
% - Impedance control
% - Pushing control
% - etc.

% Simulate motion (replace with actual control)
fprintf('Moving to target position...\n');
% Your control code here...
pause(2.0);  % Placeholder for actual motion

%% B. Target 时刻 - After completing the strategy
% 1. Get actual final pose
T_end_actual = ur.get_current_transformation('base', 'tool0');
r_end = T_end_actual(1:3, 4);
R_end = T_end_actual(1:3, 1:3);

% 2. Calculate and display Target error
[d_R3_target, d_SO3_target] = ur.calculate_pose_error(...
    "Target", r_end, R_end, r_d_target, R_d_target);

%% Summary
fprintf('\n========================================\n');
fprintf('Error Summary\n');
fprintf('========================================\n');
fprintf('Start Location:\n');
fprintf('  d_R3 (mm):  %.4f\n', d_R3_start);
fprintf('  d_SO3:      %.4f\n', d_SO3_start);
fprintf('\nTarget Location:\n');
fprintf('  d_R3 (mm):  %.4f\n', d_R3_target);
fprintf('  d_SO3:      %.4f\n', d_SO3_target);
fprintf('========================================\n');
