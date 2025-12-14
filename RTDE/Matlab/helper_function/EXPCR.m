function R = EXPCR(x)
theta = norm(x);
I = eye(3);
X = SKEW3(x);
R = I + (sin(theta)/theta)*X + ((1 - cos(theta))/(theta^2))*(X*X);
end
