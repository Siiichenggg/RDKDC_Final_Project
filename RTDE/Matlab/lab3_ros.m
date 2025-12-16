% Lab3 MATLAB ROS tests

clear; clc; close all;

% User options
robotType = "ur5";        % or "ur5e"
useRobot  = false;         % set false if you just want to test numerics

if useRobot
    
    ur = ur_interface();
else
    ur = [];  % placeholder so script still runs
end

q_test = [0; -pi/4; pi/2; -pi/2; pi/4; 0];

% (a) Test urFwdKin
disp('(a) Testing urFwdKin');
gst = urFwdKin(q_test, robotType);
disp('gst(q_test) =');
disp(gst);

if useRobot
    % Put a frame at this pose in RViz
    % tf_frame(name of parent, name of child, initial transform)
    fwdKinToolFrame = tf_frame('base_link', 'fwdKinToolFrame', eye(4));
    fwdKinToolFrame.move_frame('base_link', gst);

    % Move simulated UR to q_test
    ur.move_joints(q_test, 5.0);
    disp('Robot moved to q_test. Check the frame in RViz and take screenshots.');
end

% (b) Test urBodyJacobian with central-difference approximation
disp('(b) Testing urBodyJacobian vs finite-difference Jacobian');

eps = 1e-6;
J  = urBodyJacobian(q_test, robotType);
J_approx = zeros(6,6);

g0 = urFwdKin(q_test, robotType);
g0_inv = inv(g0);

for i = 1:6
    e_i = zeros(6,1); 
    e_i(i) = eps;

    g_plus  = urFwdKin(q_test + e_i, robotType);
    g_minus = urFwdKin(q_test - e_i, robotType);

    % central difference on homogeneous transforms
    dg_dqi = (g_plus - g_minus) / (2*eps);

    % Xi_hat ≈ g^{-1} * dg/dqi
    Xi_hat = g0_inv * dg_dqi;

    % twist-ify: extract skew-symmetric rotational part
    Rdot = Xi_hat(1:3,1:3);
    v    = Xi_hat(1:3,4);
    w_hat = 0.5 * (Rdot - Rdot.');
    w = [ w_hat(3,2); w_hat(1,3); w_hat(2,1) ];

    xi_i = [v; w];       % 6x1 twist coordinates
    J_approx(:,i) = xi_i;
end

err_J = norm(J_approx - J, 'fro');
fprintf('||J_approx - J||_F = %.3e\n', err_J);

% (c) Test manipulability measures near singularity theta3 = 0
disp('(c) Testing manipulability() near singularity (theta3 = 0)');

nPts = 101;
theta3_vals = linspace(-pi/4, pi/4, nPts);

mu_sigma   = zeros(1,nPts);
mu_detjac  = zeros(1,nPts);
mu_invcond = zeros(1,nPts);

% singularity: theta3 = 0; keep other joints at some non-singular pose
q_base = [0; -pi/2; 0; 0; 0; 0];

for k = 1:nPts
    q = q_base;
    q(3) = theta3_vals(k);

    Jk = urBodyJacobian(q, robotType);
    mu_sigma(k)   = manipulability(Jk, "sigmamin");
    mu_detjac(k)  = manipulability(Jk, "detjac");
    mu_invcond(k) = manipulability(Jk, "invcond");
end

figure;
plot(theta3_vals, mu_sigma, ...
     theta3_vals, mu_detjac, ...
     theta3_vals, mu_invcond);
xlabel('\theta_3 (rad)');
ylabel('Manipulability');
legend('sigmamin', 'detjac', 'invcond', 'Location', 'Best');
title('Manipulability near singularity \theta_3 = 0');

% (d) Test getXi using expm(se3hat(xi)) ≈ g
disp('(d) Testing getXi() and reconstruction with expm(se3hat(xi))');

R1 = myRotz(pi/6) * myRotx(pi/8);
R2 = myRoty(-pi/5);

g1 = [R1, [0.30; 0.10; 0.25];
      0 0 0 1];

g2 = [R2, [-0.10; 0.25; 0.40];
      0 0 0 1];

g_list = {g1, g2};

for k = 1:numel(g_list)
    g = g_list{k};
    xi = getXi(g);             % 6x1 twist (unscaled)
    g_recon = expm(se3hat(xi));
    err_g = norm(g - g_recon, 'fro');
    fprintf('getXi test %d: ||g - expm(xi^)||_F = %.3e\n', k, err_g);
end

% (e) Test urRRcontrol
disp('(e) Testing urRRcontrol() ');

if ~useRobot
    warning('Set useRobot = true to test RR control with ROS.');
else
    % 1) "Normal" test: move from q_init to q_goal
    q_init = [0; -pi/4;  pi/2; -pi/2;  pi/4; 0];
    q_goal = [0; -pi/3;  pi/3; -pi/2;  pi/3; 0];

    ur.move_joints(q_init, 5.0);
    pause(6);

    g_desired = urFwdKin(q_goal, robotType);
    K = 1.0;

    fprintf('Running RR control to regular goal...\n');
    finalerr = urRRcontrol(g_desired, K, ur, robotType);
    fprintf('Resolved-rate controller final position error (cm) = %.3f\n', finalerr);

    % 2) Near-singular initial condition (theta3 ≈ 0)
    q_sing_init = [0; -pi/2;  0.01; 0; 0; 0];
    q_sing_goal = [0; -pi/2; -0.01; 0; 0; 0];

    ur.move_joints(q_sing_init, 5.0);
    pause(6);

    g_sing = urFwdKin(q_sing_goal, robotType);
    fprintf('Running RR control near singularity; expect ABORT...\n');
    finalerr_sing = urRRcontrol(g_sing, K, ur, robotType);
    fprintf('Return value near singularity = %.1f (should be -1 on abort)\n', finalerr_sing);
end

disp('All lab3\_ros.m tests finished.');

% Local helper: se3 hat operator
function Xi_hat = se3hat(xi)
% xi = [v; w], v,w in R^3
v = xi(1:3);
w = xi(4:6);

w_hat = [   0   -w(3)  w(2);
          w(3)    0   -w(1);
         -w(2)  w(1)    0 ];

Xi_hat = [w_hat, v;
          0 0 0 0];
end

function R = myRotx(theta)
c = cos(theta); s = sin(theta);
R = [1  0  0;
     0  c -s;
     0  s  c];
end

function R = myRoty(theta)
c = cos(theta); s = sin(theta);
R = [ c  0  s;
      0  1  0;
     -s  0  c];
end

function R = myRotz(theta)
c = cos(theta); s = sin(theta);
R = [ c -s  0;
      s  c  0;
      0  0  1];
end
