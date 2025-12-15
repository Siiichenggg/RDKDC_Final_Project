function [q_star, idx] = findOptimalJoints(Q, q_cur)
% Q: 6x8 candidates, q_cur: 6x1
% returns q_star: 6x1

m = size(Q,2);
cost = inf(1,m);

for j = 1:m
    qj = Q(:,j);
    if any(~isfinite(qj)), continue; end

    % wrap angle differences to [-pi, pi] for fair comparison
    dq = wrapToPi(qj - q_cur);
    cost(j) = norm(dq);
end

[cmin, idx] = min(cost);
if isinf(cmin)
    q_star = [];
    idx = -1;
else
    q_star = Q(:,idx);
end
end
