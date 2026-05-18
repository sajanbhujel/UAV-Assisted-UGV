#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStateVizPublisher(Node):

    def __init__(self):
        super().__init__('joint_state_viz_publisher')

      
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

    
        self.subscription = self.create_subscription(
            JointState,
            '/encoder/joint_states', 
            self.encoder_callback,
            10
        )
        
        self.get_logger().info("Joint State Viz Publisher started.")
        self.get_logger().info("Subscribing to /encoder/joint_states")
        self.get_logger().info("Publishing to /joint_states")
    
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0


        self.timer = self.create_timer(0.02, self.publish_joint_state)

    def encoder_callback(self, msg):
        """
        Listen to the encoder data and store the latest positions.
        """
        try:

            left_index = msg.name.index('left_wheel_joint')
            right_index = msg.name.index('right_wheel_joint')
            
            # Store the latest positions
            self.left_wheel_pos = msg.position[left_index]
            self.right_wheel_pos = msg.position[right_index]
            
        except ValueError as e:
            self.get_logger().warn(f"Joint name not found in /encoder/joint_states message: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in encoder_callback: {e}")

    def publish_joint_state(self):
        """
        Called by the timer to publish the complete JointState message
        for visualization.
        """
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'front_left_wheel_joint',   # Name from your URDF
            'front_right_wheel_joint'   # Name from your URDF
        ]

        joint_msg.position = [
            self.left_wheel_pos, 
            self.right_wheel_pos 
        ]
        
    
        joint_msg.velocity = []
        joint_msg.effort = []

        self.publisher_.publish(joint_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = JointStateVizPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

