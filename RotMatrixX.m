function R = RotMatrixX(alpha)
% RotMatrixX Rotate around X axis
% R = RotMatrixX(alpha) returns a 3x3 rotation matrix
    c = cos(alpha);
    s = sin(alpha);
    R = [1 0 0;
         0 c -s;
         0 s c];
end

