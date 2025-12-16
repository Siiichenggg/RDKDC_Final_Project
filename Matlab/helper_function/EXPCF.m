function T = EXPCF(x)
x = double(x(:));
eps = 1e-8;
w  = x(1:3);
v_in = x(4:6);
theta = norm(w);
I = eye(3);

% ---- Case 1: pure / near-pure translation ----
if theta < eps
    R = I;
    p = v_in;   % no rotation 
else
    % ---- General case: MLS (2.36) ----
    n = w / theta;              % unit axis
    v = v_in / theta;           % ξ = [n; v]θ 
    N = SKEW3(n);               % [n]^
    R = EXPCR(w);               % e^{[n] θ}
    p = (I - R) * (N * v) + (n * n.') * v * theta;
end

T = eye(4);
T(1:3,1:3) = R;
T(1:3,4)   = p;
end
