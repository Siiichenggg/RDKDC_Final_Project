function T = DH(a, alpha, d, theta)
%DH Standard Denavit-Hartenberg homogeneous transform.
%   T = DH(a, alpha, d, theta) returns the 4x4 transform from frame i-1 to i
%   using standard D-H parameters.
%
%   This helper is used by urInvKin.m (provided in the course materials).

    ca = cos(alpha);
    sa = sin(alpha);
    ct = cos(theta);
    st = sin(theta);

    T = [ ct, -st*ca,  st*sa, a*ct; ...
          st,  ct*ca, -ct*sa, a*st; ...
           0,     sa,     ca,    d; ...
           0,      0,      0,    1 ];
end

