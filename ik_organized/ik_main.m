function ik_main(mode, demo_type)
    %IK_MAIN Main entry point for IK demonstrations
    %   ik_main()                      % defaults to "sim" mode, "simple" demo
    %   ik_main("real")                % connect to real robot, "simple" demo
    %   ik_main("sim", "push_back")    % simulation mode, push-back demo
    %   ik_main("sim", "toolY_3cm")    % simulation mode, 3cm push demo
    %
    % Demo types:
    %   "simple"     - Push forward 3cm, lift 15cm, move right 10cm, lower, push back 3cm
    %   "push_back"  - Full push-back sequence with lift and side movement
    %   "toolY_3cm"  - Simple 3cm push along tool Y-axis

    if nargin < 1
        mode = "sim";
    end
    if nargin < 2
        demo_type = "simple";
    end

    % Robot type
    type = 'ur5e';

    % Connect to robot
    if mode == "sim"
        fprintf("Connecting to simulation\n");
        ur = ur_rtde_interface(mode);
    else
        fprintf("Connecting to physical UR5e\n");
        ur = ur_interface();
    end
    cleanupObj = onCleanup(@() delete(ur));

    % Run selected demo
    switch demo_type
        case "simple"
            run_simple_demo(ur, type);
        case "push_back"
            ik_demos.move_push_back(ur, type);
        case "toolY_3cm"
            ik_demos.move_toolY_3cm(ur, type);
        otherwise
            error('Unknown demo type: %s. Use "simple", "push_back", or "toolY_3cm"', demo_type);
    end

    fprintf("Demo complete.\n");
end

function run_simple_demo(ur, type)
    %RUN_SIMPLE_DEMO Execute simple IK sequence: push-lift-right-lower-push_back
    % This is the simplified version from ik_main_simple.m

    % Parameters (meters)
    push_dist = 0.03;
    lift_dist = 0.15;
    right_dist = 0.10;
    dt_segment = 1.0;  % seconds per segment

    % Teach start pose
    fprintf("\n--- Teach pose: start ---\n");
    ur.switch_to_pendant_control();
    input("Move the arm to the start pose, then press ENTER...");
    q_start = ur.get_current_joints();
    ur.switch_to_ros_control();
    ur.activate_pos_control();

    q_curr = q_start(:);
    g_curr = ur.get_current_transformation();  % current SE(3) from RTDE

    % Sequence of Cartesian offsets in base frame
    steps = { ...
        "push_fwd",   [push_dist;      0;          0], "Push forward 3 cm"; ...
        "lift_up",    [0;              0;  lift_dist], "Lift up 15 cm"; ...
        "move_right", [0;      right_dist;          0], "Move right 10 cm"; ...
        "lower",      [0;              0; -lift_dist], "Lower back down"; ...
        "push_back",  [-push_dist;     0;          0], "Push back 3 cm"; ...
    };

    for k = 1:size(steps, 1)
        label = steps{k, 1};
        offset = steps{k, 2};
        desc = steps{k, 3};

        g_goal = ik_utils.translatePose(g_curr, offset);
        Q = urInvKin(g_goal, type);  % 6 x N
        [q_next, idx] = ik_utils.selectClosestIK(Q, q_curr);

        fprintf("Phase %d (%s) -> IK sol #%d\n", k, desc, idx);
        ur.move_joints(q_next, dt_segment);

        q_curr = ur.get_current_joints();
        g_curr = ur.get_current_transformation();  % refresh from hardware/sim
    end

    fprintf("Sequence complete.\n");
end
