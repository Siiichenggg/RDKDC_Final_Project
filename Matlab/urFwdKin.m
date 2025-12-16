
function gst=urFwdKin(q,type)
    % q: [θ1, θ2, θ3, θ4, θ5, θ6]'
    % type: string ('ur5' or 'ur5e')
    % gst: 4x4 homogeneous (end effector pose)
    
    % Add path to helper functionsd
    addpath("./helper_function/")
    % Get screw axis and home configuration
    [xi, gst0] = urGetScrews(type);
    
    
    % Product of exponentials formula: gst(θ) = e^{[ξ1]θ1} ... e^{[ξ6]θ6} gst(0)
    gst = EXPCF(xi(:,1) * q(1)) * EXPCF(xi(:,2) * q(2)) * EXPCF(xi(:,3) * q(3)) * ...
          EXPCF(xi(:,4) * q(4)) * EXPCF(xi(:,5) * q(5)) * EXPCF(xi(:,6) * q(6)) * gst0;
end