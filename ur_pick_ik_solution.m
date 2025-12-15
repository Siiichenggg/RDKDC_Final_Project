function [q_best, best_idx, costs] = ur_pick_ik_solution(q_solutions, q_ref, varargin)
%UR_PICK_IK_SOLUTION Pick a practical IK solution from urInvKin output.
%   [q_best, best_idx] = ur_pick_ik_solution(q_solutions, q_ref)
%   selects the column of q_solutions (6xN) that is closest to q_ref (6x1),
%   using wrapped joint differences. This is intended for choosing among the
%   8 solutions returned by urInvKin.m.
%
%   Optional name-value args:
%     - 'Weights' (6x1): nonnegative joint weights (default ones(6,1)).
%
%   Outputs:
%     - q_best (6x1): selected joint vector
%     - best_idx (1x1): selected column index
%     - costs (1xN): cost for each candidate (Inf for invalid columns)

    validateattributes(q_solutions, {'numeric'}, {'nrows', 6, '2d'});
    if isempty(q_ref)
        q_ref = zeros(6, 1);
    end
    validateattributes(q_ref, {'numeric'}, {'size', [6, 1]});

    parser = inputParser();
    parser.FunctionName = 'ur_pick_ik_solution';
    parser.addParameter('Weights', ones(6, 1), @(w) isnumeric(w) && isequal(size(w), [6, 1]) && all(w >= 0));
    parser.parse(varargin{:});
    weights = parser.Results.Weights;

    num_solutions = size(q_solutions, 2);
    costs = inf(1, num_solutions);

    for k = 1:num_solutions
        qk = q_solutions(:, k);
        if ~isreal(qk) || any(~isfinite(qk))
            continue;
        end
        dq = ur_wrap_to_pi(qk - q_ref);
        costs(k) = sum(weights .* abs(dq));
    end

    [~, best_idx] = min(costs);
    if ~isfinite(costs(best_idx))
        error('ur_pick_ik_solution:NoValidSolution', 'No valid IK solution to choose from.');
    end

    q_best = q_solutions(:, best_idx);
end

