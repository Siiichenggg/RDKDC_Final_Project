% API-compatible replacement for ur_interface, using urRTDEClient.
classdef ur_rtde_interface < handle
    
    properties (SetAccess = immutable)
        speed_limit = 0.25
        home = [0 -pi 0 -pi 0 0]'/2;
        joint_names = {'shoulder_pan_joint', 'shoulder_lift_joint', ...
                       'elbow_joint', 'wrist_1_joint', ...
                       'wrist_2_joint', 'wrist_3_joint'};
    end
    
    properties (SetAccess = private)
        robotIP
        isConnected = false
    end
    
    properties (Access = private)
        ur              % The urRTDEClient object
        robotModel      % RigidBodyTree model
        
        % Control & Timer
        updateTimer                 % Timer for velocity commands
        targetVelocity = [];        % (6x1) Target velocity
        control_mode = "Position";  % Internal state
        
        % Watchdog Timer
        lastVelCmdTime  % Stores the time of the last velocity command
        watchdogTimeout = 1.0; % Deadman switch
    end

    methods
        function self = ur_rtde_interface(mode, varargin)
            if ~exist('urRTDEClient', 'class')
                error('urRTDEClient class not found. Please install the toolbox.');
            end

            if mode == "sim"
                fprintf("Connecting to simulation\n")
                if nargin>1
                    self.robotIP = varargin{1};
                else
                    self.robotIP = '0.0.0.0';
                    fprintf("Using default IP 0.0.0.0\n");
                    fprintf("Please make sure URSim is running\n");
                end
            else
                fprintf("Connecting to physical UR5e\n")
                self.robotIP = '172.22.22.2';
            end
            
            try
                self.ur = urRTDEClient(self.robotIP);
                self.robotModel = self.ur.RigidBodyTree;
                self.isConnected = true;
                disp('Connection successful.');

                % Setup and start the background timer
                % self.lastVelCmdTime = tic; % Initialize watchdog timer
                % self.setupTimer();
                % start(self.updateTimer);

            catch ME
                disp('Initialization failed:');
                disp(ME.message);
                delete(self);
                error('Failed to create ur_rtde_interface object.');
            end
        end

        function delete(self)
            disp('Cleaning up ur_rtde_interface...');
            
            if ~isempty(self.updateTimer) && isvalid(self.updateTimer)
                stop(self.updateTimer);
                delete(self.updateTimer);
                disp('Update timer stopped.');
            end
            
            if ~isempty(self.ur)
                if self.control_mode=="Velocity"
                    sendSpeedJCommands(self.ur, zeros(6,1));
                end
                clear self.ur;
                disp('Robot connection closed.');
            end

            
            self.isConnected = false;
        end

        function joint_angles = get_current_joints(self)
            joint_angles = readJointConfiguration(self.ur).';
        end
        
        function R = rpy2M(self, z, y, x)
            Rz = [cos(z) -sin(z) 0; sin(z) cos(z) 0; 0 0 1];
            Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)];
            Rx = [1 0 0; 0 cos(x) -sin(x); 0 sin(x) cos(x)];
            R = Rz*Ry*Rx;
        end

        function g = get_current_transformation(self)
            pose = readCartesianPose(self.ur);
            z = pose(1);
            y = pose(2);
            x = pose(3);
            R = self.rpy2M(z, y, x);
            offset = eye(4);
            offset(1:3, 1:3) = self.rpy2M(pi, 0, 0);

            g = offset*[R pose(4:6)'; zeros(1,3) 1];
            % orientation and position values 
            % (represented as [theta(z) theta(y) theta(x) x y z]) 
            % in radians and meters respectively
        end
        
        function mode = get_current_control_mode(self)
            mode = self.control_mode;
            return
        end

        function move_joints(self, joint_goal, time_interval)
            if (self.control_mode ~= "Position")
                if (self.get_current_control_mode ~= "Position")
                    error('Not in Position Mode, please switch to Position Mode');
                end
            end       

            validateattributes(joint_goal,{'numeric'},{'nrows',6,'2d'})
            validateattributes(time_interval,{'numeric'},{'nonnegative','nonzero'})
            if (~isequal(size(joint_goal,2),length(time_interval)) && ~isscalar(time_interval))
                error("time_interval must either be the same size as joint_goal or a scalar")
            end
            
            [~, num_waypoints] = size(joint_goal);
            if isscalar(time_interval)
                time_interval = repmat(time_interval, 1, num_waypoints);
            end
            if length(time_interval) ~= num_waypoints
                error('time_interval must be scalar or match waypoint count.');
            end
            
            q_current = self.get_current_joints;
            joint_v = zeros(6, num_waypoints);
            joint_v(:,1) = (joint_goal(:,1) - q_current) ./ time_interval(1);
            for i = 2:num_waypoints
                joint_v(:,i) = (joint_goal(:,i) - joint_goal(:,i-1)) ./ time_interval(i);
            end

            if max(abs(joint_v), [], 'all') > self.speed_limit
                error('Velocity over speed limit, please increase time_interval');
            end
            
            % Send trajectory
            positions = [q_current, joint_goal];
            timestamps = [0, cumsum(time_interval)];
                        
            followJointWaypoints(self.ur, positions, 'WaypointTimes', timestamps);
        end
        
        % move in joint vel space
        % goal should be 6*1 vector
        function move_joints_vel(self, joint_vel_goal)
            if (self.control_mode ~= "Velocity")
                if (self.get_current_control_mode ~= "Velocity")
                    error('Not in Velocity Mode, please switch to Velocity Mode');
                end
            end

            error("Velocity control via RTDE is probihited")

            % check input
            validateattributes(joint_vel_goal, {'numeric'}, {'size', [6,1]});
            joint_vel_goal = reshape(joint_vel_goal, [1 ,6]);
            % check the speed limit
            if max(abs(joint_vel_goal)) > self.speed_limit
                error('Velocity over speed limit, please decrease command');
            end
            
            self.targetVelocity = joint_vel_goal;
            sendSpeedJCommands(self.ur, self.targetVelocity);
            disp("Sending cmd");
            self.lastVelCmdTime = tic; % Reset the watchdog timer
        end

        % for compatibility
        function pendant_control_resp = switch_to_pendant_control(self)
            % delete(self);
            pendant_control_resp = true;
        end

        function ros_control_resp = switch_to_ros_control(self)
            %
            ros_control_resp = true;
        end
        
        % mimic mode switching
        function resp = activate_pos_control(self)
            if (self.get_current_control_mode == "Position")
                resp = true;
                return;
            end

            self.move_joints_vel(zeros(6, 1));
            pause(0.5)

            resp = true;
            self.control_mode = "Position";
        end

        function resp = activate_vel_control(self)
            resp = false;
            error("Velocity control via RTDE is probihited")
            return 
            % if (self.get_current_control_mode == "Velocity")
            %     resp = true;
            %     return;
            % end
            % 
            % 
            % robot_in_motion = true;
            % 
            % while robot_in_motion
            %     jp_1 = self.get_current_joints;
            %     pause(0.1)
            %     jp_2 = self.get_current_joints;
            %     robot_in_motion = max(abs(jp_2-jp_1)) > 0.0005;
            % end
            % pause(0.5)
            % 
            % resp = true;
            % self.control_mode = "Velocity";
        end
    end

    % --- Private Helper Methods ---
    % methods (Access = private)
    %     % --- Setup Background Timer ---
    %     function setupTimer(self)
    %         self.updateTimer = timer(...
    %             'ExecutionMode', 'fixedRate', ... 
    %             'Period', 0.1, ... % 10 Hz
    %             'BusyMode', 'drop', ... 
    %             'TimerFcn', @self.onTimerTick, ...
    %             'ErrorFcn', @self.onTimerError);
    %     end
    % 
    %     % --- Timer Callback (Deadman) ---
    %     function onTimerTick(self, ~, ~)
    %         if ~isvalid(self) || ~self.isConnected
    %             return;
    %         end
    % 
    %         if self.control_mode == "Velocity"
    %             if toc(self.lastVelCmdTime) > self.watchdogTimeout
    %                 sendSpeedJCommands(self.ur, zeros(1,6));
    %                 self.targetVelocity = []; 
    %                 %disp("Triggered Safety");
    %             end
    %         end
    %     end
    % 
    %     function onTimerError(self, ~, event)
    %         fprintf('Timer error: %s\n', event.Data.message);
    %         % delete(self); 
    %     end
    % end

end