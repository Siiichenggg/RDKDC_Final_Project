classdef ik_demos
    %IK_DEMOS Demonstration tasks for inverse kinematics
    %   Contains various motion tasks: push, lift, move, etc.

    methods (Static)

        function finalerr_cm = move_toolY_3cm(ur, type)
            % move_toolY_3cm - Move 3cm along tool Y-axis direction
            % Input: ur - robot interface object
            %        type - robot type ('ur5' or 'ur5e')
            % Output: finalerr_cm - final position error in cm

            push_dist = 0.03;   % 3 cm
            Tmove = 3.0;        % move time (s)
            ur.move_joints(ur.home, 10);

            % 1) read current joints and FK
            ur.switch_to_pendant_control();
            disp('Move to start position then press any key to continue.')
            pause;
            q_cur = ur.get_current_joints();
            ur.switch_to_ros_control();
            g_start = ik_utils.urFwdKin(q_cur, type);
            disp('Start location recorded')

            % 2) tool-y direction in base frame (2nd column of R)
            y_tool_in_base = g_start(1:3, 2);

            % 3) build desired pose: same orientation, shifted position
            g_des = g_start;
            g_des(1:3, 1:3) = g_start(1:3, 1:3);                      % lock orientation
            g_des(1:3, 4) = g_start(1:3, 4) + push_dist * y_tool_in_base;

            % 4) IK solve (6x8) and select closest solution
            Q = urInvKin(g_des, type);           % 6x8
            [q_star, idx] = ik_utils.selectClosestIK(Q, q_cur);
            if isempty(q_star)
                error('No valid IK solution found.');
            end
            fprintf('Selected IK solution #%d\n', idx);

            % 5) execute
            ur.move_joints(q_star, Tmove);
            pause(Tmove + 0.5);

            % 6) compute final error (cm)
            g_act = ik_utils.urFwdKin(ur.get_current_joints(), type);
            finalerr_cm = norm(g_act(1:3, 4) - g_des(1:3, 4)) * 100;
            fprintf('Final position error = %.3f cm\n', finalerr_cm);
        end

        function finalerr_cm = move_push_back(ur, type)
            % move_push_back - Complete push-lift-move-lower-push back sequence
            % Input: ur - robot interface object
            %        type - robot type ('ur5' or 'ur5e')
            % Output: finalerr_cm - final position error in cm

            push_dist = 0.03;   % push distance (m)
            Tmove = 3.0;        % move time per segment (s)
            lift_z = 0.09;      % lift height (m)
            side_gap = 0.16;    % go to other side in XY (m)

            ur.move_joints(ur.home, 10);
            pause(10);

            % -------------------- Teach start pose --------------------
            ur.switch_to_pendant_control();
            disp('Move to START contact pose, then press any key to continue.');
            pause;
            q_start = ur.get_current_joints();
            ur.switch_to_ros_control();
            disp('Switched to ROS control.');

            g_start = ik_utils.urFwdKin(q_start, type);
            R0 = g_start(1:3, 1:3);
            p0 = g_start(1:3, 4);
            disp('Start location recorded.');

            % -------------------- Define planar push direction: tool-y --------------------
            y = g_start(1:3, 2);   % tool-y in base
            y(3) = 0;              % project to table plane
            ny = norm(y);
            if ny < 1e-6
                error('Projected tool-y is too small. Rotate tool so tool-y has XY component.');
            end
            y = y / ny;

            % ==================== Phase 1: Push forward ====================
            g_push = g_start;
            g_push(1:3, 1:3) = R0;                   % lock orientation
            g_push(1:3, 4) = p0 + push_dist * y;     % move along +y
            g_push(3, 4) = p0(3);                    % lock height

            disp('Phase 1: push forward...');
            ik_demos.ik_go(ur, type, g_push, Tmove);

            % ==================== Phase 2: Lift up ====================
            g_lift = g_push;
            g_lift(1:3, 1:3) = R0;
            g_lift(1:3, 4) = g_push(1:3, 4) + [0; 0; lift_z];

            disp('Phase 2: lift up...');
            ik_demos.ik_go(ur, type, g_lift, Tmove);

            % ==================== Phase 3: Move to other side (above) ====================
            g_other_above = g_lift;
            g_other_above(1:3, 1:3) = R0;
            g_other_above(1:3, 4) = g_lift(1:3, 4) + side_gap * y;

            disp('Phase 3: move to other side (above)...');
            ik_demos.ik_go(ur, type, g_other_above, Tmove);

            % ==================== Phase 4: Lower down to contact height ====================
            g_other_contact = g_other_above;
            g_other_contact(1:3, 1:3) = R0;
            g_other_contact(1:3, 4) = g_other_above(1:3, 4) - [0; 0; lift_z];
            g_other_contact(3, 4) = p0(3);           % lock height back to contact

            disp('Phase 4: lower down...');
            ik_demos.ik_go(ur, type, g_other_contact, Tmove);

            % ==================== Phase 5: Push back ====================
            g_back = g_other_contact;
            g_back(1:3, 1:3) = R0;
            g_back(1:3, 4) = g_other_contact(1:3, 4) - push_dist * y;
            g_back(3, 4) = p0(3);

            disp('Phase 5: push back...');
            ik_demos.ik_go(ur, type, g_back, Tmove);

            % -------------------- Final error (cm) wrt "back" target --------------------
            g_act = ik_utils.urFwdKin(ur.get_current_joints(), type);
            finalerr_cm = norm(g_act(1:3, 4) - g_back(1:3, 4)) * 100;
            fprintf('Final position error (back target) = %.3f cm\n', finalerr_cm);
        end

    end

    methods (Static, Access = private)
        % Helper function for IK motion
        function ik_go(ur, type, g_des, Tmove)
            % ik_go - Execute IK motion to desired pose
            % Input: ur - robot interface
            %        type - robot type
            %        g_des - desired SE(3) pose
            %        Tmove - move time

            q_cur = ur.get_current_joints();
            Q = urInvKin(g_des, type);                 % 6x8
            [q_star, idx] = ik_utils.selectClosestIK(Q, q_cur);
            if isempty(q_star)
                error('No valid IK solution for this waypoint.');
            end
            fprintf('  -> IK sol #%d\n', idx);
            ur.move_joints(q_star, Tmove);
            pause(Tmove + 0.2);
        end
    end
end
