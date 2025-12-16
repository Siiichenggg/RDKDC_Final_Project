% ========================================
% Main Robot Control Script - UR5/UR5e Manipulation Demonstrations
%
% This script provides a framework for testing various robotic
% manipulation tasks using inverse kinematics control.
%
% Available Demonstrations:
%   1. Tool Y-axis translation (3cm linear motion)
%   2. Five-phase push manipulation sequence
% ========================================

%% Initialize Robot Interface
fprintf('========================================\n');
fprintf('Initializing UR Robot Interface...\n');
fprintf('========================================\n');

robot_controller = ur_interface();
robot_model_type = "ur5e";

fprintf('Robot interface initialized successfully.\n\n');

%% Select and Execute Task
fprintf('========================================\n');
fprintf('Available Manipulation Tasks:\n');
fprintf('  1. Tool Y-Axis Translation Test (3cm)\n');
fprintf('  2. Five-Phase Push Manipulation Sequence\n');
fprintf('========================================\n');

% Uncomment the desired task:

% TASK 1: Tool Y-axis translation test
% final_error = tool_y_axis_translation_test(robot_controller, robot_model_type);

% TASK 2: Five-phase push manipulation (DEFAULT)
final_error = five_phase_push_manipulation(robot_controller, robot_model_type);

%% Display Summary
fprintf('\n========================================\n');
fprintf('TASK COMPLETION SUMMARY\n');
fprintf('========================================\n');
fprintf('Final Total Positioning Error: %.4f cm\n', final_error);
fprintf('========================================\n');
