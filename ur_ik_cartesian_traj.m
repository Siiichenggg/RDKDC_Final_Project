function [q_traj, g_traj_tool0] = ur_ik_cartesian_traj(g_start, g_goal, type, varargin)
%UR_IK_CARTESIAN_TRAJ IK-based joint trajectory for a Cartesian line path.
%   q_traj = ur_ik_cartesian_traj(g_start, g_goal, type) creates a joint-space
%   trajectory whose end-effector pose follows a straight line in Cartesian
%   space (in SE(3)), using urInvKin.m at each waypoint and selecting a
%   continuous solution.
%
%   Inputs:
%     - g_start, g_goal: 4x4 desired transforms (default: base_link->tool0)
%     - type: 'ur5' or 'ur5e'
%
%   Optional name-value args:
%     - 'NumSteps' (>=2): number of waypoints (default 50)
%     - 'QSeed' (6x1): seed for the first waypoint (default zeros)
%     - 'Weights' (6x1): joint weights for solution selection (default ones)
%     - 'Tool0ToTip' (4x4): constant transform g_tool0_tip. If g_start/g_goal
%       are base_link->tip, set this and the function will convert to
%       base_link->tool0 via g_base_tool0 = g_base_tip / g_tool0_tip.
%
%   Outputs:
%     - q_traj: 6xNumSteps joint trajectory (each column is a waypoint)
%     - g_traj_tool0: 4x4xNumSteps desired tool0 transforms along the path

    validateattributes(g_start, {'numeric'}, {'size', [4, 4]});
    validateattributes(g_goal, {'numeric'}, {'size', [4, 4]});

    parser = inputParser();
    parser.FunctionName = 'ur_ik_cartesian_traj';
    parser.addParameter('NumSteps', 50, @(n) isnumeric(n) && isscalar(n) && n >= 2 && mod(n, 1) == 0);
    parser.addParameter('QSeed', zeros(6, 1), @(q) isnumeric(q) && isequal(size(q), [6, 1]));
    parser.addParameter('Weights', ones(6, 1), @(w) isnumeric(w) && isequal(size(w), [6, 1]) && all(w >= 0));
    parser.addParameter('Tool0ToTip', eye(4), @(g) isnumeric(g) && isequal(size(g), [4, 4]));
    parser.parse(varargin{:});

    num_steps = parser.Results.NumSteps;
    q_prev = parser.Results.QSeed;
    weights = parser.Results.Weights;
    g_tool0_tip = parser.Results.Tool0ToTip;

    g_start_tool0 = g_start / g_tool0_tip;
    g_goal_tool0 = g_goal / g_tool0_tip;

    q_traj = zeros(6, num_steps);
    g_traj_tool0 = zeros(4, 4, num_steps);

    for i = 1:num_steps
        s = (i - 1) / (num_steps - 1);
        g_des = interpolate_se3(g_start_tool0, g_goal_tool0, s);
        g_traj_tool0(:, :, i) = g_des;

        q_solutions = urInvKin(g_des, type); % 6x8
        [q_i, ~] = ur_pick_ik_solution(q_solutions, q_prev, 'Weights', weights);

        q_traj(:, i) = q_i;
        q_prev = q_i;
    end
end

function g = interpolate_se3(g0, g1, s)
    R0 = g0(1:3, 1:3);
    p0 = g0(1:3, 4);
    R1 = g1(1:3, 1:3);
    p1 = g1(1:3, 4);

    p = (1 - s) * p0 + s * p1;
    R = slerp_rotm(R0, R1, s);

    g = eye(4);
    g(1:3, 1:3) = R;
    g(1:3, 4) = p;
end

function R = slerp_rotm(R0, R1, s)
    Rrel = R0.' * R1;
    c = (trace(Rrel) - 1) / 2;
    c = max(min(c, 1), -1);
    angle = acos(c);

    if angle < 1e-9
        R = R0;
        return;
    end

    axis = [Rrel(3, 2) - Rrel(2, 3); ...
            Rrel(1, 3) - Rrel(3, 1); ...
            Rrel(2, 1) - Rrel(1, 2)] / (2 * sin(angle));

    axis = axis / norm(axis);
    K = [    0, -axis(3),  axis(2); ...
          axis(3),     0, -axis(1); ...
         -axis(2), axis(1),     0];

    a = s * angle;
    Rinc = eye(3) + sin(a) * K + (1 - cos(a)) * (K * K);
    R = R0 * Rinc;
end

