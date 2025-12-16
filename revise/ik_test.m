% ========================================
% Main Test Script for UR Robot Inverse Kinematics
%
% This script demonstrates robot manipulation tasks using
% inverse kinematics and the UR interface.
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
fprintf('Available Tasks:\n');
fprintf('  1. Simple 3cm translation along tool Y-axis\n');
fprintf('  2. Complete push-back manipulation sequence\n');
fprintf('========================================\n');

% Uncomment the desired task:

% TASK 1: Simple translation test
% final_error = ik_move_toolY_3cm(robot_controller, robot_model_type);

% TASK 2: Push-back manipulation (DEFAULT)
final_error = ik_move_push_back(robot_controller, robot_model_type);

%% Display Results
fprintf('\n========================================\n');
fprintf('Task completed with final error: %.3f cm\n', final_error);
fprintf('========================================\n');
