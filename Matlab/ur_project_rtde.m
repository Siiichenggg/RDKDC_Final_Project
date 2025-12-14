% ur_project_rtde.m
% RTDE + RR only. Press Enter to run RR (per spec).

clear; clc;

% ========== Configuration ==========
mode = "sim";             % "sim" or "real"
ip   = "0.0.0.0";         % sim default; real IP for hardware

ur = ur_rtde_interface(mode, ip);

% --- Control parameters ---
dt = 0.15;                % RR step time (s) - larger steps = faster convergence
posTol = 5e-3;            % position tolerance: 5mm (relaxed)
angTol = 2e-2;            % angular tolerance
maxSteps = 300;           % reduced - should reach target faster with larger steps

% --- Task parameters ---
push_dist = 0.3;         % 3cm push distance (project requirement)

% Push direction in BASE frame (like Python version)
% Change this to control push direction:
push_direction = [0; 1; 0];   % [X; Y; Z] in base frame
                               % [1;0;0] = +X (forward)
                               % [0;1;0] = +Y (left)
                               % [0;0;1] = +Z (up)

% Normalize the direction
push_direction = push_direction / norm(push_direction);

% ========== Move to Home ==========
fprintf('\n========== Initializing ==========\n');
try
    fprintf('Moving to home...\n');
    ur.move_joints(ur.home, 3.0);
    pause(0.5);
catch ME
    fprintf('Warning: Could not move home: %s\n', ME.message);
end

% ========== Teach START Position ==========
fprintf('\n========== TEACH MODE ==========\n');
fprintf('Manually move robot to START position, then press any key...\n');
pause;

T_start = ur.get_current_transformation();  % 4x4 base->tool0
q_start = ur.get_current_joints();

fprintf('\n✓ START recorded:\n');
fprintf('  Position: [%.4f, %.4f, %.4f] m\n', T_start(1:3,4));
fprintf('  Joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] rad\n', q_start);

% ========== Execute Push-and-Place ==========
fprintf('\nPress Enter to execute push-and-place...\n');
input('', 's');

% ------------------------------
% Segment A: First push (3cm)
% ------------------------------
fprintf('\n--- Segment A: Pushing 3cm ---\n');

% Compute push vector in base frame
push_vec = push_direction * push_dist;
fprintf('Push direction (base): [%.3f, %.3f, %.3f]\n', push_direction);

% Target position = start position + push vector
p_start = T_start(1:3, 4);
p_endpush = p_start + push_vec;

% Keep same orientation, new position
T_endpush = T_start;
T_endpush(1:3, 4) = p_endpush;

fprintf('Target: [%.4f, %.4f, %.4f] m\n', p_endpush);

% Execute first push
rr_move_to_T(ur, T_endpush, dt, posTol, angTol, maxSteps);
fprintf('✓ First push done.\n');

pause(0.5);

% ------------------------------
% Segment B: Push back (3cm opposite)
% ------------------------------
fprintf('\n--- Segment B: Pushing back ---\n');

% Get actual position after first push
T_after = ur.get_current_transformation();
p_after = T_after(1:3, 4);

fprintf('Position after push: [%.4f, %.4f, %.4f] m\n', p_after);

% Push back (opposite direction)
p_target = p_after - push_vec;

% Keep same orientation
T_target = T_after;
T_target(1:3, 4) = p_target;

fprintf('Target: [%.4f, %.4f, %.4f] m\n', p_target);

% Execute second push
rr_move_to_T(ur, T_target, dt, posTol, angTol, maxSteps);
fprintf('✓ Second push done.\n');

% ========== Done ==========
fprintf('\n========== COMPLETE ==========\n');
final_T = ur.get_current_transformation();
fprintf('Final position: [%.4f, %.4f, %.4f] m\n', final_T(1:3,4));

clear ur;


% ===== Helper function =====
function rr_move_to_T(ur, T_des, dt, posTol, angTol, maxSteps)
    % Initial position for debugging
    T_init = ur.get_current_transformation();
    p_init = T_init(1:3, 4);
    p_des = T_des(1:3, 4);
    total_dist = norm(p_des - p_init);
    fprintf('  Initial distance to target: %.4f m\n', total_dist);

    for k = 1:maxSteps
        try
            [done, q_next] = rr_step_to_T(ur, T_des, dt, posTol, angTol);

            % Send joint command
            % Note: move_joints is blocking and waits for completion
            ur.move_joints(q_next, dt);

            % Small pause to let RTDE stabilize
            pause(0.02);

            if done
                fprintf('  Reached target in %d steps\n', k);
                break;
            end

            % Progress indicator every 50 steps with distance check
            if mod(k, 50) == 0
                T_curr = ur.get_current_transformation();
                p_curr = T_curr(1:3, 4);
                dist_remaining = norm(p_des - p_curr);
                fprintf('  Step %d/%d... (%.4f m remaining)\n', k, maxSteps, dist_remaining);
            end

        catch ME
            fprintf('  Error at step %d: %s\n', k, ME.message);
            % Brief pause after error, then continue
            pause(0.3);
        end
    end

    if ~done
        fprintf('  Warning: Max steps reached\n');
        T_final = ur.get_current_transformation();
        p_final = T_final(1:3, 4);
        fprintf('  Final error: %.4f m\n', norm(p_des - p_final));
    end
end
