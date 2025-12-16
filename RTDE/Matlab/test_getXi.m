%% Test script for getXi function
% This script tests the getXi function with a given transformation matrix
% and verifies that we can reconstruct the original matrix

clear; clc;
addpath("helper_function/")

% Test matrix
g_example = [0,-1,0,1; 1,0,0,2; 0,0,1,3; 0,0,0,1];

% Extract rotation and translation
R = g_example(1:3,1:3);
p = g_example(1:3,4);

% Check rotation angle
theta = acos((trace(R)-1)/2);
fprintf('Rotation angle theta: %.15f radians (%.4f degrees)\n', theta, theta*180/pi);

% Extract twist
xi = getXi(g_example);
fprintf('\nExtracted twist xi:\n');
disp(xi);
fprintf('v (linear part): [%.4f, %.4f, %.4f]''\n', xi(1), xi(2), xi(3));
fprintf('omega (angular part): [%.4f, %.4f, %.4f]''\n', xi(4), xi(5), xi(6));

% Verify by reconstructing g from xi
% Construct the 4x4 twist matrix xi_hat
xi_hat = [SKEW3(xi(4:6)), xi(1:3); 0, 0, 0, 0];

fprintf('\nTwist matrix xi_hat:\n');
disp(xi_hat);

g_reconstructed = expm(xi_hat);

fprintf('Original g:\n');
disp(g_example);

fprintf('Reconstructed g from xi:\n');
disp(g_reconstructed);

fprintf('Difference (should be close to zero):\n');
diff_matrix = g_example - g_reconstructed;
disp(diff_matrix);

fprintf('Max absolute error: %.15e\n', max(abs(diff_matrix), [], 'all'));
