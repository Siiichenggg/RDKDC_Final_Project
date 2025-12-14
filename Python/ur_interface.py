# Copyright (c) 2025 Annie Huang, Jiacheng Li. All rights reserved.

import time
import numpy as np
import threading
import rclpy
import atexit
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String, Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix

class UrInterface(Node):
    """
    Class ur_interface provides Python API to ur_modern_driver which controls the ur robots through ROS2
    """

    _ros_inited     = False
    _executor       = None
    _spin_thread    = None
    _init_flag      = False

    def __init__(self):
        """Constructor"""
        if not UrInterface._ros_inited:
            if not rclpy.ok():
                rclpy.init()
                UrInterface._init_flag = True
            UrInterface._executor = SingleThreadedExecutor()
            UrInterface._ros_inited = True
            atexit.register(UrInterface._shutdown)

        super().__init__('ur_interface')
        
        # Settings that are not supposed to change after constructor
        self.speed_limit = 0.25  # fraction of maximum speed
        self.home = np.array([0, -np.pi, 0, -np.pi, 0, 0]) / 2 # joint states in home position [rad]
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Operating state
        self.current_joint_states = None
        self.control_mode = 'Position'

        # Initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 1)

        # Publishers
        self.trajectory_pub = self.create_publisher(JointTrajectory, 'rdkdc/joint_pos_msg', 10)
        self.velocity_pub = self.create_publisher(Float64MultiArray, 'rdkdc/joint_vel_msg', 10)
        self.urscript_pub = self.create_publisher(String, '/urscript_interface/script_command', 10)

        # Service clients
        self.pendant_control_client = self.create_client(Trigger, '/io_and_status_controller/hand_back_control')
        self.ros_control_client = self.create_client(Trigger, '/io_and_status_controller/resend_robot_program')
        # Correct service names (typos prevented successful lookups and blocked motion)
        self.switch_to_pos_ctrl_client = self.create_client(Trigger, 'rdkdc/switch_to_pos_ctrl')
        self.switch_to_vel_ctrl_client = self.create_client(Trigger, 'rdkdc/switch_to_vel_ctrl')
        self.get_curr_ctrl_mode_client = self.create_client(Trigger, 'rdkdc/get_curr_ctrl_mode')

        UrInterface._executor.add_node(self)
        if UrInterface._spin_thread is None:
            UrInterface._spin_thread = threading.Thread(
                target=UrInterface._executor.spin,
                daemon=True,
                name='URSpinThread'
            )
            UrInterface._spin_thread.start()

        # while (self.trajectory_pub.get_subscription_count() == 0) or (UrInterface._executor.spin_once(timeout_sec=0.1)):
        #     UrInterface._executor.spin_once(timeout_sec=0.1)

        # Heartbeat warm-up
        self._dds_wakeup = self.create_publisher(Empty, 'dds_wakeup', 1)
        for _ in range(10):
            UrInterface._executor.spin_once(timeout_sec=0.05)
            self._dds_wakeup.publish(Empty())

        self.get_current_control_mode()

    @classmethod
    def _shutdown(cls):
        """Cleanly shut down rclpy on process exit."""
        if cls._ros_inited:
            if cls._init_flag:
                rclpy.shutdown()

            if cls._executor is not None:
                cls._executor.shutdown()
            if cls._spin_thread is not None:
                cls._spin_thread.join(timeout=1.0)

            cls._ros_inited = False
            cls._executor = None
            cls._spin_thread = None

    def _joint_state_callback(self, msg: JointState):
        """Callback for joint states"""
        # print("joint state callback received")
        self.current_joint_states = msg

    # exception handling requires update
    def get_current_control_mode(self) -> str:
        """Get current control mode"""
        if not self.get_curr_ctrl_mode_client.wait_for_service(timeout_sec=1.5):
            self.get_logger().error('UR5 control-mode service not available')
            raise RuntimeError('UR5 Node Not Running')

        req = Trigger.Request()
        future = self.get_curr_ctrl_mode_client.call_async(req)
        UrInterface._executor.spin_until_future_complete(future, timeout_sec=5.0)
        if future.result() is not None:
            mode = future.result().message
            self.control_mode = mode
            return mode
        else:
            # This will be triggered on a timeout
            raise RuntimeError(f'Service call to get control mode failed: {future.exception()}')

    def get_current_joints(self) -> np.ndarray:
        """Update current joint states and return array of current joint angles"""
        while self.current_joint_states is None:
            time.sleep(0.01)

        joint_angles = np.zeros(6)
        for i, name in enumerate(self.joint_names):
            if name in self.current_joint_states.name:
                idx = self.current_joint_states.name.index(name)
                joint_angles[i] = self.current_joint_states.position[idx]
        return joint_angles

    def get_current_transformation(self, target_frame: str, source_frame: str) -> np.ndarray:
        """Get current transformation from target frame to source frame"""
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().error(f'Transform lookup failed: {e}')
            return np.eye(4)

        # Build transformation matrix
        qw = trans.transform.rotation.w
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        T = quaternion_matrix([qw, qx, qy, qz])
        T[:3, 3] = [
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ]
        return T

    def move_joints(self, joint_goal: np.ndarray, time_intervals) -> None:
        """
        Move in joint space
        
        Parameters:
        joint_goal: 6xN numpy array of joint goals
        time_interval: scalar or array of time intervals
        """
        # Ensure position mode
        if self.control_mode != 'Position':
            if self.get_current_control_mode() != 'Position':
                raise RuntimeError('Not in Position Mode')
        # print('Check1')
        goal = np.array(joint_goal)
        if goal.ndim == 1:
            goal = goal.reshape((6, 1))
        if np.isscalar(time_intervals):
            time_intervals = [time_intervals] * goal.shape[1]

        # time intercal size check updated required.

        # Compute velocities
        vels = np.zeros_like(goal)
        vels[:, 0] = (goal[:, 0] - self.get_current_joints()) / time_intervals[0]

        for i in range(1, goal.shape[1]):
            dt = time_intervals[i] if not np.isscalar(time_intervals) else time_intervals[0]
            vels[:, i] = (goal[:, i] - goal[:, i-1]) / dt

        # Check speed limit
        if np.max(np.abs(vels)) > self.speed_limit:
            raise Exception('Velocity over speed limit, please increase time_interval')
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        traj.header.stamp = self.get_clock().now().to_msg()

        # print('Check2')

        cum_time = 0.0
        for i in range(goal.shape[1]):
            pt = JointTrajectoryPoint()
            
            pt.positions = list(goal[:, i])
            pt.accelerations = [0.0]*6

            # average velocity to next point or zero at end
            if i < goal.shape[1] - 1:
                pt.velocities = list(0.5 * (vels[:, i] + vels[:, i+1]))
            else:
                pt.velocities = [0.0]*6

            cum_time += time_intervals[i]

            pt.time_from_start = Duration(seconds=cum_time).to_msg()
            traj.points.append(pt)
        
        # print('Check3')
        # print(traj)
        self.trajectory_pub.publish(traj)

    def move_joints_vel(self, joint_vel_goal: np.ndarray) -> None:
        """
        Move in joint velocity space
        
        Parameters:
        joint_vel_goal: 6x1 numpy array of joint velocities
        """
        # require update and tests
        # Ensure velocity mode
        if self.control_mode != 'Velocity':
            if self.get_current_control_mode() != 'Velocity':
                raise RuntimeError('Not in Velocity Mode')

        if np.max(np.abs(joint_vel_goal)) > self.speed_limit:
            raise RuntimeError('Requested velocity exceeds limit')

        # Prepare message
        msg = Float64MultiArray()
        msg.data = list(joint_vel_goal)
        # Publish velocity command
        self.velocity_pub.publish(msg)

    def switch_to_pendant_control(self) -> bool:
        """Switch to pendant control"""
        if not self.pendant_control_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('Pendant-control service unavailable')

        req = Trigger.Request()
        future = self.pendant_control_client.call_async(req)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        UrInterface._executor.spin_until_future_complete(future, timeout_sec=1.0)

        if future.result() is not None:
            return future.result().success
        return False

    def switch_to_ros_control(self) -> bool:
        """Switch back to ROS control"""
        if not self.ros_control_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('ROS-control service unavailable')
        
        req = Trigger.Request()
        future = self.ros_control_client.call_async(req)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        UrInterface._executor.spin_until_future_complete(future, timeout_sec=1.0)

        if future.result() is not None:
            return future.result().success
        return False

    def enable_freedrive(self) -> None:
        """Enable freedrive mode"""
        script = (
            'def my_prog():\n'
            '  freedrive_mode()\n'
            '  while True:\n'
            '    end = request_boolean_from_primary_client("Would you like to end FreeDrive?")\n'
            '    if end:\n'
            '      end_freedrive_mode()\n'
            '      break\n'
            '  end\n'
            'end'
        )
        msg = String()
        msg.data = script
        self.urscript_pub.publish(msg)

    def activate_pos_control(self) -> bool:
        """Activate position control mode"""
        if self.get_current_control_mode() == 'Position':
            return True
        
        # Stop robot before switching controllers
        self.move_joints_vel(np.zeros(6))
        time.sleep(0.5)

        # Send request to switch controllers
        if not self.switch_to_pos_ctrl_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('Position-control service unavailable')

        req = Trigger.Request()
        future = self.switch_to_pos_ctrl_client.call_async(req)

        UrInterface._executor.spin_until_future_complete(future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.control_mode = "Position"
            return True
        return False

    def activate_vel_control(self) -> bool:
        """Activate velocity control mode"""
        if self.get_current_control_mode() == 'Velocity':
            return True

        # wait until motion stops
        while True:
            pos1 = self.get_current_joints()
            time.sleep(0.1)
            pos2 = self.get_current_joints()
            if pos1 is not None and pos2 is not None and np.max(np.abs(pos2 - pos1)) < 5e-4:
                break

        time.sleep(0.5)

        if not self.switch_to_vel_ctrl_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('Velocity-control service unavailable')
        
        req = Trigger.Request()
        future = self.switch_to_vel_ctrl_client.call_async(req)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        UrInterface._executor.spin_until_future_complete(future, timeout_sec=5.0)

        if future.result() is not None and future.result().success:
            self.control_mode = "Velocity"
            return True
        return False 
