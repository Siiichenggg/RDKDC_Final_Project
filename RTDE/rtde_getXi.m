function xi = rtde_getXi(g)
%RTDE_GETXI Extract the (un-normalized) twist xi from an SE(3) transform g.
% Returns xi = [v; w] such that g = expm(hat(xi)).

R = g(1:3, 1:3);
p = g(1:3, 4);

theta = acos((trace(R) - 1) / 2);

if abs(theta) < 1e-5
    omega = zeros(3, 1);
    v = p;
else
    omega_hat_unit = (R - R.') / (2 * sin(theta));
    omega_unit = [omega_hat_unit(3,2); omega_hat_unit(1,3); omega_hat_unit(2,1)];

    omega = theta * omega_unit;
    omega_hat = theta * omega_hat_unit;

    omega_hat_sq = omega_hat * omega_hat;
    I = eye(3);
    V = I + (1 - cos(theta)) / theta^2 * omega_hat + ...
        (theta - sin(theta)) / theta^3 * omega_hat_sq;

    v = V \ p;
end

xi = [v; omega];
end

