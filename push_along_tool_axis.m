function finalerr_cm = push_along_tool_axis(ur, type)
    push_dist = 0.03;   % 3 cm
    Tmove = 3.0;        % move time (s)
    ur.move_joints(ur.home,10);
    % 1) read current joints and FK
    ur.switch_to_pendant_control();
    disp('move to start position then press any key to continue.')
    pause;
    q_cur = ur.get_current_joints();
    ur.switch_to_ros_control();
    g_start = robotForwardKinematics(q_cur, type);
    disp('start location recorded')

    % 2) tool-y direction in base frame (2nd column of R)
    y_tool_in_base = g_start(1:3,2);

    % 3) build desired pose: same orientation, shifted position
    g_des = g_start;
    g_des(1:3,1:3) = g_start(1:3,1:3);                     % lock orientation
    g_des(1:3,4)   = g_start(1:3,4) + push_dist*y_tool_in_base;

    % 4) IK solve (6x8) and select closest solution
    Q = urInvKin(g_des, type);           % 6x8
    [q_star, idx] = findOptimalJoints(Q, q_cur);
    if isempty(q_star)
        error('No valid IK solution found.');
    end
    fprintf('Selected IK solution #%d\n', idx);

    % 5) execute
    ur.move_joints(q_star, Tmove);
    pause(Tmove + 0.5);

    % 6) compute final error (cm)
    g_act = robotForwardKinematics(ur.get_current_joints(), type);
    finalerr_cm = norm(g_act(1:3,4) - g_des(1:3,4)) * 100;
    fprintf('Final position error = %.3f cm\n', finalerr_cm);
end
