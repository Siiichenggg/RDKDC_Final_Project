function ang = ur_wrap_to_pi(ang)
%UR_WRAP_TO_PI Wrap angles to [-pi, pi).
%   ang = ur_wrap_to_pi(ang) wraps each element of ang into [-pi, pi).

    ang = mod(ang + pi, 2*pi) - pi;
end

