function xi = getXi(g)
    % Extracts the (un-normalized) twist xi from a homogeneous transformation g
    % such that g = expm(hat(xi))

    R = g(1:3,1:3);    % Rotation part
    p = g(1:3,4);      % Translation part

    % Compute angle and axis from rotation matrix
    theta = acos((trace(R)-1)/2);

    if abs(theta) < 1e-5
        % Pure translation (or very small rotation)
        omega = [0; 0; 0];
        v = p;
    else
        % Skew-symmetric matrix of the unit omega
        omega_hat_unit = (R - R') / (2*sin(theta));
        omega_unit = [omega_hat_unit(3,2); omega_hat_unit(1,3); omega_hat_unit(2,1)];

        % Un-normalized omega (scaled by theta)
        omega = theta * omega_unit;
        omega_hat = theta * omega_hat_unit;

        % Compute the V matrix and its inverse
        omega_hat_sq = omega_hat * omega_hat;
        I = eye(3);
        V = I + (1 - cos(theta))/theta^2 * omega_hat + ...
            (theta - sin(theta))/theta^3 * omega_hat_sq;

        v = V \ p;  % Solve V * v = p
    end

    xi = [v; omega];
end
