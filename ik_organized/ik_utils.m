classdef ik_utils
    %IK_UTILS Utility functions for inverse kinematics operations
    %   Contains rotation matrices, forward kinematics, IK solution selection

    methods (Static)

        function R = ROTX(alpha)
            % ROTX Rotate around X axis
            % R = ROTX(alpha) returns a 3x3 rotation matrix
            c = cos(alpha);
            s = sin(alpha);
            R = [1 0 0;
                 0 c -s;
                 0 s c];
        end

        function R = ROTZ(theta)
            % ROTZ Rotate around Z axis
            % R = ROTZ(theta) returns a 3x3 rotation matrix
            c = cos(theta);
            s = sin(theta);
            R = [c -s 0;
                 s c 0;
                 0 0 1];
        end

        function gst = urFwdKin(q, type)
            % urFwdKin - Forward Kinematics for UR5/UR5e
            % Input: q - 6x1 joint angles (radians)
            %        type - string 'ur5' or 'ur5e' (default 'ur5e')
            % Output: gst - 4x4 Homogeneous Transformation Matrix (Base -> Tool0)

            if nargin < 2
                type = 'ur5e';
            end

            % DH Parameters (Aligned with urInvKin.m)
            % Default UR5 params
            d = [0.089159; 0; 0; 0.10915; 0.09465; 0.0823];
            a = [0; -0.425; -0.39225; 0; 0; 0];
            alpha = [pi/2; 0; 0; pi/2; -pi/2; 0];

            % UR5e params overrides
            if strcmp(type, 'ur5e')
                d(1) = 0.1625;
                d(4) = 0.1333;
                d(5) = 0.0997;
                d(6) = 0.0996;
                a(2) = -0.425;
                a(3) = -0.3922;
            end

            % Theta offsets
            offset = [pi; 0; 0; 0; 0; 0];

            gst = eye(4);

            for i = 1:6
                % Call the DH function
                A_i = DH(a(i), alpha(i), d(i), q(i) + offset(i));
                gst = gst * A_i;
            end
        end

        function [q_star, idx] = selectClosestIK(Q, q_cur)
            % selectClosestIK - Select IK solution closest to current configuration
            % Input: Q - 6x8 IK solution candidates
            %        q_cur - 6x1 current joint configuration
            % Output: q_star - 6x1 selected solution
            %         idx - index of selected solution

            m = size(Q, 2);
            cost = inf(1, m);

            for j = 1:m
                qj = Q(:, j);
                if any(~isfinite(qj)), continue; end

                % wrap angle differences to [-pi, pi] for fair comparison
                dq = ik_utils.wrapToPi(qj - q_cur);
                cost(j) = norm(dq);
            end

            [cmin, idx] = min(cost);
            if isinf(cmin)
                q_star = [];
                idx = -1;
            else
                q_star = Q(:, idx);
            end
        end

        function q = wrapToPi(q)
            % wrapToPi - Wrap angles to (-pi, pi]
            % Input: q - angle or array of angles
            % Output: q - wrapped angles
            q = mod(q + pi, 2*pi) - pi;
        end

        function g_out = translatePose(g_in, offset)
            % translatePose - Translate homogeneous transform by offset (base frame)
            % Input: g_in - 4x4 homogeneous transformation matrix
            %        offset - 3x1 translation offset in base frame
            % Output: g_out - 4x4 translated transformation matrix
            g_out = g_in;
            g_out(1:3, 4) = g_in(1:3, 4) + offset;
        end

    end
end
