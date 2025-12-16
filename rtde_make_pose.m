function g = rtde_make_pose(R, p)
%RTDE_MAKE_POSE Build an SE(3) transform from rotation R and position p.

validateattributes(R, {'numeric'}, {'size', [3,3]});
validateattributes(p, {'numeric'}, {'size', [3,1]});

g = eye(4);
g(1:3,1:3) = R;
g(1:3,4) = p;
end

