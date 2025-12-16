% ========================================
% ur_interface - MATLAB ROS2 Interface for Universal Robots
%
% This class provides a comprehensive API for controlling UR robots
% (UR5/UR5e) through ROS2 topics and services.
%
% Original Author: Mengze Xu (07/24/2017)
% Revised by: Zhuoqun (Ray) Zhang (08/04/2021)
% Revised by: Jakub Piwowarczyk (08/27/2023)
% ========================================

classdef ur_interface < handle
    % ========================================
    % Universal Robot ROS2 Interface Class
    %
    % Provides methods for:
    %   - Joint position and velocity control
    %   - Pendant control switching
    %   - Transformation queries
    %   - Control mode management
    % ========================================

    % ========== Immutable Configuration ==========
    properties (SetAccess = immutable)
        speed_limit = 0.25;  % Maximum speed factor (0-1 range)
        home = [0 -pi 0 -pi 0 0]' / 2;  % Home configuration (radians)
        joint_names = {
            'shoulder_pan_joint', ...
            'shoulder_lift_joint', ...
            'elbow_joint', ...
            'wrist_1_joint', ...
            'wrist_2_joint', ...
            'wrist_3_joint'
        };
    end
    
    % ========== Public Read-Only State ==========
    properties (SetAccess = protected)
        tftree  % ROS2 TF tree for coordinate transformations
    end

    % ========== Private Internal State ==========
    properties (SetAccess = private, Hidden = true)
        % ROS2 node handle
        node_handle

        % Subscriber handles
        current_joint_states_sub

        % Publisher handles
        trajectory_goal_pub
        urscript_pub
        velocity_goal_pub

        % Service client handles
        pendant_control_client
        ros_control_client
        swtich_to_pos_ctrl_client
        swtich_to_vel_ctrl_client
        get_curr_ctrl_mode_client

        % Current control mode ("Position" or "Velocity")
        control_mode
    end
    
    % ========== Public Methods ==========
    methods
        % ========================================
        % Constructor - Initialize ROS2 interface
        % ========================================
        function self = ur_interface()
            % Create ROS2 node
            self.node_handle = ros2node("/ur_interface");

            % Initialize subscriber for joint states
            self.current_joint_states_sub = ros2subscriber(...
                self.node_handle, ...
                '/joint_states', ...
                'sensor_msgs/JointState', ...
                'Depth', 10);

            % Initialize publishers
            self.trajectory_goal_pub = ros2publisher(...
                self.node_handle, ...
                'rdkdc/joint_pos_msg', ...
                'trajectory_msgs/JointTrajectory');

            self.velocity_goal_pub = ros2publisher(...
                self.node_handle, ...
                'rdkdc/joint_vel_msg', ...
                'std_msgs/Float64MultiArray');

            self.urscript_pub = ros2publisher(...
                self.node_handle, ...
                '/urscript_interface/script_command', ...
                'std_msgs/String');

            % Initialize service clients for control switching
            self.pendant_control_client = ros2svcclient(...
                self.node_handle, ...
                '/io_and_status_controller/hand_back_control', ...
                'std_srvs/Trigger');

            self.ros_control_client = ros2svcclient(...
                self.node_handle, ...
                '/io_and_status_controller/resend_robot_program', ...
                'std_srvs/Trigger');

            self.swtich_to_pos_ctrl_client = ros2svcclient(...
                self.node_handle, ...
                "rdkdc/swtich_to_pos_ctrl", ...
                'std_srvs/Trigger');

            self.swtich_to_vel_ctrl_client = ros2svcclient(...
                self.node_handle, ...
                "rdkdc/swtich_to_vel_ctrl", ...
                'std_srvs/Trigger');

            self.get_curr_ctrl_mode_client = ros2svcclient(...
                self.node_handle, ...
                "rdkdc/get_curr_ctrl_mode", ...
                'std_srvs/Trigger');

            % Initialize TF tree
            self.tftree = ros2tf(self.node_handle);

            % Set initial control mode
            self.control_mode = "Position";
            pause(0.5);

            % Query actual control mode from robot
            self.get_current_control_mode;
        end

        % ========================================
        % Query current control mode from robot
        % ========================================
        function current_mode = get_current_control_mode(self)
            try
                waitForServer(self.get_curr_ctrl_mode_client, "Timeout", 1);
            catch
                error("ERROR: UR ROS2 node is not running or unreachable");
            end

            request = ros2message(self.get_curr_ctrl_mode_client);
            response = call(self.get_curr_ctrl_mode_client, request, "Timeout", 5);
            self.control_mode = response.message;
            current_mode = self.control_mode;
        end

        % ========================================
        % Retrieve current joint angles
        % ========================================
        function joint_config = get_current_joints(self)
            joint_state_msg = receive(self.current_joint_states_sub);
            joint_config = zeros(6, 1);

            % Map received joint states to standard ordering
            for received_idx = 1:6
                for standard_idx = 1:6
                    if strcmp(joint_state_msg.name{received_idx}, self.joint_names{standard_idx})
                        joint_config(standard_idx) = joint_state_msg.position(received_idx);
                    end
                end
            end
        end
        
        % ========================================
        % Query transformation between coordinate frames
        % ========================================
        function transform_matrix = get_current_transformation(self, target_frame, source_frame)
            transform_msg = getTransform(self.tftree, target_frame, source_frame);

            % Extract translation vector
            translation = [
                transform_msg.transform.translation.x, ...
                transform_msg.transform.translation.y, ...
                transform_msg.transform.translation.z
            ];

            % Extract rotation as quaternion and convert to rotation matrix
            quaternion = [
                transform_msg.transform.rotation.w, ...
                transform_msg.transform.rotation.x, ...
                transform_msg.transform.rotation.y, ...
                transform_msg.transform.rotation.z
            ];
            rotation_matrix = quat2rotm(quaternion);

            % Construct 4x4 homogeneous transformation matrix
            transform_matrix = [rotation_matrix, translation'; 0, 0, 0, 1];
        end
        
        % ========================================
        % Execute joint-space motion command
        %
        % Arguments:
        %   joint_goal    - 6xN matrix of target joint configurations (radians)
        %   time_interval - scalar or 1xN vector of motion durations (seconds)
        % ========================================
        function move_joints(self, joint_goal, time_interval)
            % Verify control mode
            if self.control_mode ~= "Position"
                if self.get_current_control_mode ~= "Position"
                    error('ERROR: Robot not in Position Mode. Switch mode before commanding.');
                end
            end

            % Validate input dimensions
            validateattributes(joint_goal, {'numeric'}, {'nrows', 6, '2d'});
            validateattributes(time_interval, {'numeric'}, {'nonnegative', 'nonzero'});

            num_waypoints = size(joint_goal, 2);
            if ~isscalar(time_interval) && length(time_interval) ~= num_waypoints
                error("ERROR: time_interval must be scalar or match number of waypoints");
            end

            % Create trajectory message
            trajectory_goal = ros2message('trajectory_msgs/JointTrajectory');

            % Populate joint names
            trajectory_goal.joint_names = self.joint_names;

            % Set timestamp to current time
            trajectory_goal.header.stamp = ros2time(self.node_handle, 'now');

            % Compute waypoint velocities
            joint_velocities = zeros(size(joint_goal));
            current_position = self.get_current_joints();

            % First waypoint velocity (from current to first target)
            dt_first = time_interval(1);
            joint_velocities(:, 1) = (joint_goal(:, 1) - current_position) / dt_first;

            % Subsequent waypoint velocities
            if num_waypoints >= 2
                for waypoint_idx = 2:num_waypoints
                    if isscalar(time_interval)
                        dt = time_interval;
                    else
                        dt = time_interval(waypoint_idx);
                    end
                    joint_velocities(:, waypoint_idx) = ...
                        (joint_goal(:, waypoint_idx) - joint_goal(:, waypoint_idx - 1)) / dt;
                end
            end

            % Enforce speed limit
            max_velocity = max(max(abs(joint_velocities)));
            if max_velocity > self.speed_limit
                error('ERROR: Commanded velocity (%.3f) exceeds limit (%.3f). Increase time_interval.', ...
                      max_velocity, self.speed_limit);
            end
            
            % Construct trajectory waypoints
            for waypoint_idx = 1:num_waypoints
                trajectory_point = ros2message('trajectory_msgs/JointTrajectoryPoint');

                % Set positions
                trajectory_point.positions = joint_goal(:, waypoint_idx);

                % Set velocities (average with next waypoint, or zero at end)
                if waypoint_idx < num_waypoints
                    trajectory_point.velocities = ...
                        (joint_velocities(:, waypoint_idx + 1) + joint_velocities(:, waypoint_idx)) / 2;
                else
                    trajectory_point.velocities = zeros(6, 1);
                end

                % Set accelerations (zero for smooth motion)
                trajectory_point.accelerations = zeros(6, 1);

                % No effort specification
                trajectory_point.effort = [];

                % Set time from start
                if isscalar(time_interval)
                    trajectory_point.time_from_start = ros2duration(time_interval * waypoint_idx);
                else
                    trajectory_point.time_from_start = ros2duration(sum(time_interval(1:waypoint_idx)));
                end

                trajectory_goal.points(waypoint_idx) = trajectory_point;
            end

            % Publish trajectory command
            send(self.trajectory_goal_pub, trajectory_goal);
        end

        % ========================================
        % Command joint velocities
        %
        % Arguments:
        %   velocity_command - 6x1 vector of joint velocities (rad/s)
        % ========================================
        function move_joints_vel(self, velocity_command)
            % Verify control mode
            if self.control_mode ~= "Velocity"
                if self.get_current_control_mode ~= "Velocity"
                    error('ERROR: Robot not in Velocity Mode. Switch mode before commanding.');
                end
            end

            % Validate input
            validateattributes(velocity_command, {'numeric'}, {'size', [6, 1]});

            % Enforce speed limit
            max_vel = max(abs(velocity_command));
            if max_vel > self.speed_limit
                error('ERROR: Commanded velocity (%.3f) exceeds limit (%.3f).', ...
                      max_vel, self.speed_limit);
            end

            % Create and publish velocity command
            velocity_msg = ros2message('std_msgs/Float64MultiArray');
            velocity_msg.data = velocity_command;
            send(self.velocity_goal_pub, velocity_msg);
        end

        % ========================================
        % Switch robot control to teach pendant
        %
        % Allows manual positioning via pendant or freedrive mode
        % ========================================
        function success = switch_to_pendant_control(self)
            request = ros2message(self.pendant_control_client);
            waitForServer(self.pendant_control_client, "Timeout", 1);
            response = call(self.pendant_control_client, request, "Timeout", 1);
            success = response.success;
        end

        % ========================================
        % Switch robot control back to ROS2
        %
        % Resumes programmatic control after pendant usage
        % ========================================
        function success = switch_to_ros_control(self)
            request = ros2message(self.ros_control_client);
            waitForServer(self.ros_control_client, "Timeout", 1);
            response = call(self.ros_control_client, request, "Timeout", 1);
            success = response.success;
        end

        % ========================================
        % Enable freedrive mode via URScript
        %
        % Freedrive remains active until user confirms exit on pendant.
        % Must call switch_to_ros_control() to resume programmatic control.
        % ========================================
        function enable_freedrive(self)
            urscript_msg = ros2message('std_msgs/String');

            % Construct URScript program
            script_lines = {
                'def my_prog():', ...
                'freedrive_mode()', ...
                'while True: my_variable=request_boolean_from_primary_client("Would you like to end FreeDrive?") if my_variable: end_freedrive_mode() break end end', ...
                'end'
            };
            urscript_msg.data = strjoin(script_lines, newline);

            send(self.urscript_pub, urscript_msg);
        end

        % ========================================
        % Activate position control mode
        %
        % Switches controller to enable move_joints() commands
        % ========================================
        function success = activate_pos_control(self)
            % Check if already in position mode
            if self.get_current_control_mode == "Position"
                success = true;
                return;
            end

            % Stop robot motion before switching
            self.move_joints_vel(zeros(6, 1));
            pause(0.5);

            % Request controller switch
            request = ros2message(self.swtich_to_pos_ctrl_client);
            try
                waitForServer(self.swtich_to_pos_ctrl_client, "Timeout", 1);
            catch
                error("ERROR: Velocity control functionality not available");
            end

            response = call(self.swtich_to_pos_ctrl_client, request, "Timeout", 5);
            success = response.success;

            if success
                self.control_mode = "Position";
            end
        end

        % ========================================
        % Activate velocity control mode
        %
        % Switches controller to enable move_joints_vel() commands
        % ========================================
        function success = activate_vel_control(self)
            % Check if already in velocity mode
            if self.get_current_control_mode == "Velocity"
                success = true;
                return;
            end

            % Wait until robot stops moving
            MOTION_THRESHOLD = 0.0005;  % rad/s threshold
            SAMPLE_INTERVAL = 0.1;      % seconds

            is_moving = true;
            while is_moving
                position_1 = self.get_current_joints;
                pause(SAMPLE_INTERVAL);
                position_2 = self.get_current_joints;

                max_joint_change = max(abs(position_2 - position_1));
                is_moving = (max_joint_change > MOTION_THRESHOLD);
            end
            pause(0.5);

            % Request controller switch
            request = ros2message(self.swtich_to_vel_ctrl_client);
            try
                waitForServer(self.swtich_to_vel_ctrl_client, "Timeout", 1);
            catch
                error("ERROR: Velocity control functionality not available");
            end

            response = call(self.swtich_to_vel_ctrl_client, request, "Timeout", 5);
            success = response.success;

            if success
                self.control_mode = "Velocity";
            end
        end

        % ========================================
        % Calculate and display pose error metrics
        %
        % Arguments:
        %   location     - String: "Start" or "Target"
        %   r_actual     - 3x1 vector: actual position (meters)
        %   R_actual     - 3x3 matrix: actual rotation matrix
        %   r_desired    - 3x1 vector: desired position (meters)
        %   R_desired    - 3x3 matrix: desired rotation matrix
        %
        % Outputs:
        %   d_R3         - Position error in mm
        %   d_SO3        - Rotation error (unitless)
        % ========================================
        function [d_R3, d_SO3] = calculate_pose_error(~, location, r_actual, R_actual, r_desired, R_desired)
            % Validate inputs
            validateattributes(r_actual, {'numeric'}, {'size', [3, 1]});
            validateattributes(R_actual, {'numeric'}, {'size', [3, 3]});
            validateattributes(r_desired, {'numeric'}, {'size', [3, 1]});
            validateattributes(R_desired, {'numeric'}, {'size', [3, 3]});

            % Calculate position error in R^3 (convert meters to millimeters)
            d_R3 = norm(r_actual - r_desired) * 1000;

            % Calculate rotation error in SO(3)
            % d_SO3 = sqrt(trace((R - R_d)(R - R_d)^T))
            R_diff = R_actual - R_desired;
            d_SO3 = sqrt(trace(R_diff * R_diff'));

            % Display results
            fprintf('\n========================================\n');
            fprintf('Pose Error Analysis - %s\n', location);
            fprintf('========================================\n');
            fprintf('Location:        %s\n', location);
            fprintf('d_R3 (mm):       %.4f\n', d_R3);
            fprintf('d_SO3:           %.4f\n', d_SO3);
            fprintf('========================================\n\n');
        end

    end  % methods

end  % classdef