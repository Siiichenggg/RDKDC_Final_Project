# Copyright (c) 2025 Annie Huang, Jiacheng Li. All rights reserved.

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy
import atexit
import threading


class tf_frame:
    """
    Class to create and maintain a frame in tf 
    communicate with ros through topic matlab_frame
    """
    
    _node_handle = None
    _tf_broadcaster = None
    _tf_buffer = None
    _tf_listener = None
    _executor     = None
    _spin_thread  = None
    _init_flag    = False

    def __init__(self, base_frame_name, frame_name, g):
        """
        Constructor
        
        Parameters:
        base_frame_name: reference frame name
        frame_name: frame name
        g: 4x4 transformation matrix
        """
        self.frame_name = frame_name
        self.base_frame_name = base_frame_name
        self.pose = g
        
        # Initialize ROS2 components if not already done
        self._initialize_ros()
        
        # Move frame to initial position
        self.move_frame(base_frame_name, g)
    
    @classmethod
    def _initialize_ros(cls):
        """Initialize ROS2 node and components"""
        if cls._node_handle is None:
            if not rclpy.ok():
                rclpy.init()
                cls._init_flag = True
            
            cls._node_handle = Node('tf_interface')

            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            cls._tf_broadcaster = cls._node_handle.create_publisher(
                TransformStamped,
                'rdkdc/tf_msg',
                qos
            )
            cls._tf_buffer = Buffer()
            cls._tf_listener = TransformListener(cls._tf_buffer, cls._node_handle)

            cls._executor = SingleThreadedExecutor()
            cls._executor.add_node(cls._node_handle)
            cls._spin_thread = threading.Thread(
                target=cls._executor.spin,
                daemon=True,
                name='TFFrameSpinner'
            )
            cls._spin_thread.start()

    @classmethod
    def get_node_handle(cls):
        """Get the ROS2 node handle"""
        if cls._node_handle is None:
            cls._initialize_ros()
        return cls._node_handle
    
    @classmethod
    def get_tf_broadcaster(cls):
        """Get the TF broadcaster"""
        if cls._tf_broadcaster is None:
            cls._initialize_ros()
        return cls._tf_broadcaster
    
    @classmethod
    def get_tf_tree(cls):
        """Get the TF buffer"""
        if cls._tf_buffer is None:
            cls._initialize_ros()
        return cls._tf_buffer
    
    def move_frame(self, ref_frame_name, g):
        """
        Move the frame by g relative to ref_frame
        
        Parameters:
        ref_frame_name: reference frame name
        g: 4x4 transformation matrix
        """
        msg = TransformStamped()
        msg.child_frame_id = self.frame_name
        msg.header.frame_id = ref_frame_name
        msg.header.stamp = self.get_node_handle().get_clock().now().to_msg()
        
        # Extract rotation and translation from transformation matrix
        rotation_matrix = g[:3, :3]
        translation = g[:3, 3]
        
        # Convert rotation matrix to quaternion
        quaternion = quaternion_from_matrix(g)
        
        msg.transform.translation.x = float(translation[0])
        msg.transform.translation.y = float(translation[1])
        msg.transform.translation.z = float(translation[2])
        msg.transform.rotation.x = float(quaternion[0])
        msg.transform.rotation.y = float(quaternion[1])
        msg.transform.rotation.z = float(quaternion[2])
        msg.transform.rotation.w = float(quaternion[3])

        self.get_tf_broadcaster().publish(msg)

    def read_frame(self, ref_frame_name):
        """
        Read frame transformation relative to reference frame
        
        Parameters:
        ref_frame_name: reference frame name
        
        Returns:
        g: 4x4 transformation matrix
        """
        try:
            now = self.get_node_handle().get_clock().now()
            default_timeout = rclpy.duration.Duration(seconds=5.0)
            transform = self.get_tf_tree().lookup_transform(
                ref_frame_name, self.frame_name, now, timeout=default_timeout)

            # Extract translation
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Extract quaternion and convert to rotation matrix
            quaternion = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            rotation_matrix = quaternion_matrix(quaternion)[:3, :3]
            
            # Construct 4x4 transformation matrix
            g = np.eye(4)
            g[:3, :3] = rotation_matrix
            g[:3, 3] = translation
            
            return g
            
        except Exception as e:
            print(f"Failed to lookup transform: {e}")
            return None
    
    def disappear(self):
        """
        Delete the frame in RVIZ, can be recovered by move_frame
        """
        msg = TransformStamped()
        msg.header.frame_id = 'Delete'
        msg.child_frame_id = self.frame_name
        self.get_tf_broadcaster().publish(msg)

    @classmethod
    def shutdown(cls):
        """Tear down the single node and TF resources."""
        if cls._node_handle:
            if cls._init_flag:
                rclpy.shutdown()
            if cls._executor is not None:
                cls._executor.shutdown()
            if cls._spin_thread is not None:
                cls._spin_thread.join(timeout=1.0)

            cls._node_handle = None
            cls._tf_broadcaster = None
            cls._tf_buffer = None
            cls._tf_listener = None
            cls._executor      = None
            cls._spin_thread   = None

atexit.register(tf_frame.shutdown)