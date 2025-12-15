function R = ROTZ(theta)
% ROTZ Rotate around Z axis
% R = ROTZ(theta) returns a 3x3 rotation matrix
    c = cos(theta);
    s = sin(theta);
    R = [c -s 0;
         s c 0;
         0 0 1];
end

