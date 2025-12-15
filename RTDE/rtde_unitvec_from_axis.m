function v = rtde_unitvec_from_axis(axisSpec)
%RTDE_UNITVEC_FROM_AXIS Map "+X/-X/+Y/-Y/+Z/-Z" to a 3x1 unit vector.

if isstring(axisSpec) || ischar(axisSpec)
    s = upper(string(axisSpec));
else
    error('axisSpec must be a string like "+X" or "-Z".');
end

switch s
    case "+X"
        v = [1; 0; 0];
    case "-X"
        v = [-1; 0; 0];
    case "+Y"
        v = [0; 1; 0];
    case "-Y"
        v = [0; -1; 0];
    case "+Z"
        v = [0; 0; 1];
    case "-Z"
        v = [0; 0; -1];
    otherwise
        error('Invalid axisSpec "%s". Use one of: +X,-X,+Y,-Y,+Z,-Z.', s);
end
end

