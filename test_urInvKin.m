% test_urInvKin allows students to test the accuracy of their urFwdKin.m functions
% Author: Jakub Piwowarczyk, 08/27/2023
% Update: Jiacheng Li,  07/12/2025

% initialize an array to store the errors
errors_5 = zeros(6,1);
errors_5e = zeros(6,1);

% select the number of tests to ensure many configurations work
n_tests = 1000;
tic
for i = 1:n_tests
    % generate a random set of joint angles between -pi and pi
    q = (rand(6,1) * 2*pi) - pi;
    
    % calculate the tool0 pose using your forward kinematics
    gBaseTool = urFwdKin(q, 'ur5');
    gBaseTool_e = urFwdKin(q, 'ur5e');
    
    % calculate the set of solution using urInvKin
    q_sol = urInvKin(gBaseTool, 'ur5');
    q_sol_e = urInvKin(gBaseTool_e, 'ur5e');
    
    % find the closest matching kinematic configuration
    [min_error, min_error_i] = min(vecnorm(q - q_sol,1));
    
    % calculate the errors
    errors_5 = errors_5 + abs(q_sol(:,min_error_i) - q);

    % repeat for 5e
    [min_error, min_error_i] = min(vecnorm(q - q_sol_e,1));
    errors_5e = errors_5e + abs(q_sol(:,min_error_i) - q);
end
toc

% print the errors
fprintf("UR5 : The Average Joint Errors are: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",errors_5/n_tests)
fprintf("UR5e: The Average Joint Errors are: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",errors_5e/n_tests)
