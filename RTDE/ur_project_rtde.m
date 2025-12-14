clear all; close all; clc;

% Add paths for helper functions
addpath('./Matlab/');
addpath('./Matlab/helper_function/');

%% ======================== CONFIGURATION ========================
ROBOT_TYPE = 'ur5e';           % Robot type
PUSH_DISTANCE = 0.16;          % Push distance in meters (16 cm)
BOX_SIZE = 0.13;               % Box edge length in meters (13 cm)
LIFT_HEIGHT = 0.05;            % Height to lift end-effector (5 cm)
CONTROL_GAIN = 1.0;            % RR control gain
TIME_STEP = 0.1;               % Control loop time step (seconds)
POS_THRESHOLD = 0.01;          % Position convergence threshold (1 cm)
ROT_THRESHOLD = 5 * pi/180;    % Rotation convergence threshold (5 degrees)
MAX_ITERATIONS = 500;          % Maximum iterations for each movement
SINGULARITY_THRESHOLD = 0.01;  % Minimum singular value threshold

% Push direction in base frame (choose X or Y axis)
% For this example, we push along +Y direction
PUSH_DIRECTION = [0; 1; 0];    % Unit vector along Y-axis

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║      UR5e Push-and-Place Project (RTDE Environment)       ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% ======================== INITIALIZATION ========================
fprintf('[STEP 1] Initializing robot interface...\n');

% Create robot interface (use 'sim' for simulation or 'physical' for real robot)
% Uncomment the appropriate line below:
ur = ur_rtde_interface("sim");  % For simulation
% ur = ur_rtde_interface("physical");  % For physical robot

fprintf('✓ Robot interface initialized successfully.\n\n');

%% ===================== STEP 1: TEACHING PHASE =====================
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TEACHING PHASE                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('Please manually move the robot to the starting position\n');
fprintf('(left side of the box, end-effector touching the box).\n\n');
fprintf('Press ENTER when ready to record the starting position...\n');
pause;

% Record starting configuration
q_start = ur.get_current_joints();
g_start = ur.get_current_transformation();

fprintf('✓ Starting position recorded:\n');
fprintf('  Joint angles (rad): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', q_start);
fprintf('  Position (m): [%.4f, %.4f, %.4f]\n', g_start(1:3, 4));
fprintf('\n');

%% ================ STEP 2: COMPUTE TARGET POSITIONS ================
fprintf('[STEP 2] Computing target positions...\n\n');

% Target 1: Push forward 16 cm along push direction
p_start = g_start(1:3, 4);
p_target1 = p_start + PUSH_DISTANCE * PUSH_DIRECTION;
R_start = g_start(1:3, 1:3);

g_target1 = [R_start, p_target1; 0 0 0 1];

fprintf('✓ Target 1 (after push forward):\n');
fprintf('  Position (m): [%.4f, %.4f, %.4f]\n', p_target1);
fprintf('  Push direction: [%.1f, %.1f, %.1f]\n', PUSH_DIRECTION);
fprintf('  Push distance: %.2f cm\n\n', PUSH_DISTANCE * 100);

%% ============= STEP 3: PUSH BOX FORWARD (16 cm) =============
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║              TASK 1: PUSH BOX FORWARD 16 cm                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Execute RR control to push box forward
fprintf('Starting Resolved-Rate control...\n');
final_error1 = executeRRControl(ur, g_target1, ROBOT_TYPE, CONTROL_GAIN, ...
                                TIME_STEP, POS_THRESHOLD, ROT_THRESHOLD, ...
                                MAX_ITERATIONS, SINGULARITY_THRESHOLD);

if final_error1 < 0
    error('❌ Task 1 failed! See error messages above.');
end

fprintf('✓ Task 1 completed. Final position error: %.2f cm\n\n', final_error1);
pause(0.5);

%% ============= STEP 4: LIFT END-EFFECTOR =============
fprintf('[STEP 3] Lifting end-effector...\n');

q_current = ur.get_current_joints();
g_current = ur.get_current_transformation();

% Create lifted position (move up along Z-axis)
p_lifted = g_current(1:3, 4) + [0; 0; LIFT_HEIGHT];
R_current = g_current(1:3, 1:3);
g_lifted = [R_current, p_lifted; 0 0 0 1];

% Execute lift motion
final_error_lift = executeRRControl(ur, g_lifted, ROBOT_TYPE, CONTROL_GAIN, ...
                                    TIME_STEP, POS_THRESHOLD, ROT_THRESHOLD, ...
                                    MAX_ITERATIONS, SINGULARITY_THRESHOLD);

fprintf('✓ End-effector lifted by %.1f cm\n\n', LIFT_HEIGHT * 100);
pause(0.5);

%% ============= STEP 5: MOVE TO OPPOSITE SIDE =============
fprintf('[STEP 4] Moving to opposite side of box...\n');

% Calculate approach position on opposite side
% Move to opposite side: go back in push direction by (PUSH_DISTANCE + BOX_SIZE)
% and maintain the lifted height
p_opposite = p_lifted - (PUSH_DISTANCE + BOX_SIZE) * PUSH_DIRECTION;
g_opposite = [R_current, p_opposite; 0 0 0 1];

% Execute move to opposite side
final_error_move = executeRRControl(ur, g_opposite, ROBOT_TYPE, CONTROL_GAIN, ...
                                    TIME_STEP, POS_THRESHOLD, ROT_THRESHOLD, ...
                                    MAX_ITERATIONS, SINGULARITY_THRESHOLD);

fprintf('✓ Moved to opposite side of box\n\n');
pause(0.5);

%% ============= STEP 6: LOWER TO BOX HEIGHT =============
fprintf('[STEP 5] Lowering to box contact height...\n');

% Lower back down
p_lowered = g_opposite(1:3, 4) - [0; 0; LIFT_HEIGHT];
g_lowered = [R_current, p_lowered; 0 0 0 1];

% Execute lowering motion
final_error_lower = executeRRControl(ur, g_lowered, ROBOT_TYPE, CONTROL_GAIN, ...
                                     TIME_STEP, POS_THRESHOLD, ROT_THRESHOLD, ...
                                     MAX_ITERATIONS, SINGULARITY_THRESHOLD);

fprintf('✓ End-effector lowered to box contact height\n\n');
pause(0.5);

%% ============= STEP 7: PUSH BOX BACK =============
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║           TASK 2: PUSH BOX BACK TO NEAR ORIGIN             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Calculate push-back target (push forward to move box back)
% Push same distance as before to return box near original position
p_pushback = p_lowered + PUSH_DISTANCE * PUSH_DIRECTION;
g_pushback = [R_current, p_pushback; 0 0 0 1];

% Execute push back
fprintf('Starting push-back motion...\n');
final_error2 = executeRRControl(ur, g_pushback, ROBOT_TYPE, CONTROL_GAIN, ...
                                TIME_STEP, POS_THRESHOLD, ROT_THRESHOLD, ...
                                MAX_ITERATIONS, SINGULARITY_THRESHOLD);

if final_error2 < 0
    error('❌ Task 2 failed! See error messages above.');
end

fprintf('✓ Task 2 completed. Final position error: %.2f cm\n\n', final_error2);

%% =================== FINAL ERROR REPORTING ===================
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                    FINAL ERROR REPORT                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Get final actual pose
g_final_actual = ur.get_current_transformation();

% Compute errors using specified formulas
% Desired final pose is g_pushback
[pos_error, ori_error] = computePoseError(g_final_actual, g_pushback);

fprintf('Target pose (g_target):\n');
fprintf('  Position (m): [%.4f, %.4f, %.4f]\n', g_pushback(1:3, 4));
fprintf('  Rotation matrix:\n');
fprintf('    [%.4f, %.4f, %.4f]\n', g_pushback(1, 1:3));
fprintf('    [%.4f, %.4f, %.4f]\n', g_pushback(2, 1:3));
fprintf('    [%.4f, %.4f, %.4f]\n\n', g_pushback(3, 1:3));

fprintf('Actual pose (g_actual):\n');
fprintf('  Position (m): [%.4f, %.4f, %.4f]\n', g_final_actual(1:3, 4));
fprintf('  Rotation matrix:\n');
fprintf('    [%.4f, %.4f, %.4f]\n', g_final_actual(1, 1:3));
fprintf('    [%.4f, %.4f, %.4f]\n', g_final_actual(2, 1:3));
fprintf('    [%.4f, %.4f, %.4f]\n\n', g_final_actual(3, 1:3));

fprintf('────────────────────────────────────────────────────────────\n');
fprintf('ERROR METRICS (using specified formulas):\n');
fprintf('────────────────────────────────────────────────────────────\n');
fprintf('  Position Error (d_ℝ³)  : %.6f m (%.4f cm)\n', pos_error, pos_error * 100);
fprintf('  Orientation Error (d_SO(3)): %.6f rad (%.4f deg)\n', ori_error, ori_error * 180/pi);
fprintf('────────────────────────────────────────────────────────────\n');
fprintf('\n');

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║              TASK COMPLETED SUCCESSFULLY! ✓                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% ======================= CLEANUP =======================
fprintf('[CLEANUP] Closing robot connection...\n');
delete(ur);
fprintf('✓ Cleanup complete.\n\n');


%% ===================== HELPER FUNCTIONS =====================

function final_error = executeRRControl(ur, g_desired, robot_type, K, ...
                                        T_step, pos_thresh, rot_thresh, ...
                                        max_iter, sing_thresh)
    % Executes Resolved-Rate control to move robot to desired pose
    %
    % Inputs:
    %   ur: ur_rtde_interface object
    %   g_desired: 4x4 desired end-effector pose
    %   robot_type: 'ur5' or 'ur5e'
    %   K: control gain
    %   T_step: time step (seconds)
    %   pos_thresh: position convergence threshold (m)
    %   rot_thresh: rotation convergence threshold (rad)
    %   max_iter: maximum iterations
    %   sing_thresh: singularity threshold
    %
    % Output:
    %   final_error: final position error in cm (-1 if failed)

    iteration = 0;
    converged = false;

    while ~converged && iteration < max_iter
        iteration = iteration + 1;

        % Get current joint configuration
        q_current = ur.get_current_joints();

        % Compute current end-effector pose
        g_current = urFwdKin(q_current, robot_type);

        % Compute error transformation: g_error = g_desired^{-1} * g_current
        g_error = FINV(g_desired) * g_current;

        % Extract twist from error
        xi_error = getXi(g_error);

        % Extract position and orientation components
        v_error = xi_error(1:3);      % Linear velocity (m)
        omega_error = xi_error(4:6);  % Angular velocity (rad)

        % Compute error magnitudes
        pos_error = norm(v_error);
        rot_error = norm(omega_error);

        % Display progress every 10 iterations
        if mod(iteration, 10) == 1
            fprintf('  Iter %3d: pos_err = %.4f cm, rot_err = %.2f deg\n', ...
                    iteration, pos_error*100, rot_error*180/pi);
        end

        % Check convergence
        if pos_error < pos_thresh && rot_error < rot_thresh
            converged = true;
            final_error = pos_error * 100;  % Convert to cm
            fprintf('  ✓ Converged at iteration %d\n', iteration);
            fprintf('    Final errors: pos = %.4f cm, rot = %.2f deg\n', ...
                    final_error, rot_error*180/pi);
            break;
        end

        % Compute Body Jacobian
        J_b = urBodyJacobian(q_current, robot_type);

        % Check for singularity
        sigma = svd(J_b);
        sigma_min = min(sigma);

        if sigma_min < sing_thresh
            fprintf('  ❌ ABORT: Singularity detected! σ_min = %.6f\n', sigma_min);
            final_error = -1;
            return;
        end

        % Check condition number
        cond_num = cond(J_b);
        if cond_num > 1e6
            fprintf('  ❌ ABORT: Ill-conditioned Jacobian! cond = %.2e\n', cond_num);
            final_error = -1;
            return;
        end

        % Compute joint velocity: qdot = -K * J_b^{-1} * xi_error
        % (negative sign because we want to reduce the error)
        qdot = -K * (J_b \ xi_error);

        % Integration: q_next = q_current + qdot * dt
        q_next = q_current + qdot * T_step;

        % Send position command to robot (simulating velocity control)
        ur.move_joints(q_next, T_step);

        % Wait for motion to complete
        pause(T_step);
    end

    % Check if we timed out
    if ~converged
        fprintf('  ❌ ABORT: Maximum iterations (%d) reached\n', max_iter);
        fprintf('    Final errors: pos = %.4f cm, rot = %.2f deg\n', ...
                pos_error*100, rot_error*180/pi);
        final_error = -1;
    end
end


function [pos_error, ori_error] = computePoseError(g_actual, g_desired)
    % Computes pose error using specified formulas from project handbook
    %
    % Inputs:
    %   g_actual: 4x4 actual transformation matrix
    %   g_desired: 4x4 desired transformation matrix
    %
    % Outputs:
    %   pos_error: position error d_ℝ³ = ||r - r_d|| (meters)
    %   ori_error: orientation error d_SO(3) = sqrt(tr((R - R_d)(R - R_d)^T))

    % Extract position vectors
    r_actual = g_actual(1:3, 4);
    r_desired = g_desired(1:3, 4);

    % Extract rotation matrices
    R_actual = g_actual(1:3, 1:3);
    R_desired = g_desired(1:3, 1:3);

    % Position error: d_ℝ³ = ||r - r_d||
    pos_error = norm(r_actual - r_desired);

    % Orientation error: d_SO(3) = sqrt(tr((R - R_d)(R - R_d)^T))
    R_diff = R_actual - R_desired;
    ori_error = sqrt(trace(R_diff * R_diff'));
end
