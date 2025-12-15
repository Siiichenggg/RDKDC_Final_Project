function R = RotMatrixZ(theta)
% RotMatrixZ Rotate around Z axis
% R = RotMatrixZ(theta) returns a 3x3 rotation matrix
    c = cos(theta);
    s = sin(theta);
    R = [c -s 0;
         s c 0;
         0 0 1];
end

