function finalerr = urRRcontrol(gdesired, K, ur, type)
    % Inputs:
    %   gdesired: 4x4 desired end-effector pose (homogeneous transform)
    %   K: controller gain (scalar)
    %   ur: ur_rtde_interface object
    %   type: 'ur5' or 'ur5e' (string)
    %
    % Output:
    %   finalerr: -1 if failure (singularity or timeout)
    %             final positional error in cm if success
    %
    % Control law: q_{k+1} = q_k - K * T_step * inv(J^b_st(q_k)) * xi_k
    % where xi_k is extracted from: exp(hat(xi_k)) = g_desired^{-1} * g_st(q_k)

    % Add path to helper functions
    addpath("./helper_function/")

    T_step = 0.1;              % Time step in seconds 
    max_iterations = 1000;      % Maximum number of iterations

    % Convergence thresholds
    pos_threshold = 0.05;      % 5 cm in meters
    rot_threshold = 15 * pi/180;  % 15 degrees in radians

    % Singularity threshold
    singular_threshold = 0.01; % Minimum singular value threshold

    % Initialize
    iteration = 0;
    converged = false;

    fprintf('Starting Resolved-Rate Control...\n');
    fprintf('Position threshold: %.2f cm\n', pos_threshold * 100);
    fprintf('Rotation threshold: %.2f degrees\n', rot_threshold * 180/pi);

    % Main control loop
    while ~converged && iteration < max_iterations
        iteration = iteration + 1;

        % Get current joint angles
        q_current = ur.get_current_joints();

        % Compute current end-effector pose
        g_current = urFwdKin(q_current, type);

        % Compute error transformation: g_error = g_desired^{-1} * g_current
        g_error = FINV(gdesired) * g_current;

        % Extract twist from error (un-normalized)
        xi_k = getXi(g_error);

        % Extract linear and angular velocity components
        v_k = xi_k(1:3);  % Linear velocity (m)
        omega_k = xi_k(4:6);  % Angular velocity (rad)

        % Compute error norms
        pos_error = norm(v_k);
        rot_error = norm(omega_k);

        % Display progress
        fprintf('Iter %d: pos_err = %.4f cm, rot_err = %.2f deg\n', ...
                iteration, pos_error*100, rot_error*180/pi);

        % Check convergence
        if pos_error < pos_threshold && rot_error < rot_threshold
            converged = true;
            finalerr = pos_error * 100;  % Convert to cm
            fprintf('Converged! Final position error: %.4f cm\n', finalerr);
            break;
        end

        % Compute Body Jacobian
        J_b = urBodyJacobian(q_current, type);

        % Check for singularity
        mu_sigma = manipulability(J_b, 'sigmamin');
        if mu_sigma < singular_threshold
            fprintf('ABORT: Singularity detected! sigma_min = %.6f\n', mu_sigma);
            finalerr = -1;
            return;
        end

        % Check condition number for numerical stability
        cond_num = cond(J_b);
        if cond_num > 1e6
            fprintf('ABORT: Jacobian is ill-conditioned! cond = %.2e\n', cond_num);
            finalerr = -1;
            return;
        end

        % Compute joint velocity update
        % delta_q = -(K * T_step) * inv(J_b) * xi_k;  %the form, lower efficiency
        delta_q = -(K * T_step) * inv(J_b) * xi_k;g

        % Update joint angles
        q_next = q_current + delta_q;

        % Send command to robot
        ur.move_joints(q_next, T_step);

        % Small pause to allow robot to move
        pause(T_step);
    end

    % Check if we timed out
    if ~converged
        fprintf('ABORT: Maximum iterations reached without convergence\n');
        fprintf('Final errors: pos = %.4f cm, rot = %.2f deg\n', ...
                pos_error*100, rot_error*180/pi);
        finalerr = -1;
    end
end
