function [optimal_solution, solution_index] = selectClosestIK(IK_candidates, current_joint_config)
    % ========================================
    % selectClosestIK: Choose the IK solution closest to current configuration
    %
    % Arguments:
    %   IK_candidates       - 6xN matrix of candidate joint configurations
    %   current_joint_config - 6x1 current joint angles (radians)
    %
    % Returns:
    %   optimal_solution  - 6x1 best joint configuration ([] if none valid)
    %   solution_index    - index of chosen solution (-1 if none valid)
    % ========================================

    num_candidates = size(IK_candidates, 2);
    distance_metric = inf(1, num_candidates);

    % Evaluate each candidate solution
    for candidate_idx = 1:num_candidates
        candidate_config = IK_candidates(:, candidate_idx);

        % Skip invalid configurations (NaN or Inf values)
        if any(~isfinite(candidate_config))
            continue;
        end

        % Compute angular distance with wrapping to [-pi, pi]
        angular_diff = wrapToPi(candidate_config - current_joint_config);
        distance_metric(candidate_idx) = norm(angular_diff);
    end

    % Select solution with minimum distance
    [min_distance, solution_index] = min(distance_metric);

    if isinf(min_distance)
        % No valid solution found
        optimal_solution = [];
        solution_index = -1;
    else
        % Return the optimal configuration
        optimal_solution = IK_candidates(:, solution_index);
    end
end
