function Ti = FINV(T)
    R  = T(1:3,1:3);
    p  = T(1:3,4);
    Rt = R.';                    % R^{-1} = R^T
    Ti = eye(4);
    Ti(1:3,1:3) = Rt;
    Ti(1:3,4)   = -Rt * p;
end
