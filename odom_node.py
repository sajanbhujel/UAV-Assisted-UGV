#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import math

class OdometryPublisher(Node):

    def __init__(self):
        super().__init__('odometry_publisher')
        
        self.declare_parameter('wheel_radius', 0.040)    
        self.declare_parameter('wheel_separation', 0.126) 
        
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_separation = self.get_parameter('wheel_separation').get_parameter_value().double_value
        
        self.get_logger().info(f"Odometry using wheel_radius: {self.wheel_radius}m")
        self.get_logger().info(f"Odometry using wheel_separation: {self.wheel_separation}m")

        # --- State Variables ---
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_left_rad = None
        self.last_right_rad = None
        self.last_time = None


        self.subscription = self.create_subscription(
            JointState,
            'encoder/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.odom_publisher = self.create_publisher(Odometry, '/odom/raw', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('Odometry node started, subscribing to /encoder/joint_states...')

    def joint_state_callback(self, msg):

        try:

            current_left_rad = msg.position[0]
            current_right_rad = msg.position[1]
            current_time = Time.from_msg(msg.header.stamp)

      
            if self.last_time is None:
                self.last_left_rad = current_left_rad
                self.last_right_rad = current_right_rad
                self.last_time = current_time
                self.get_logger().info('First JointState received, initializing odometry state.')
                return

  
            dt = (current_time - self.last_time).nanoseconds / 1e9 


            if dt == 0:
                self.get_logger().warn("dt is zero, skipping odometry update.")
                return


            delta_left_rad = current_left_rad - self.last_left_rad
            delta_right_rad = current_right_rad - self.last_right_rad


            dist_left_m = delta_left_rad * self.wheel_radius
            dist_right_m = delta_right_rad * self.wheel_radius

            delta_distance = (dist_right_m + dist_left_m) / 2.0
            delta_theta = (dist_right_m - dist_left_m) / self.wheel_separation


            delta_x = delta_distance * math.cos(self.theta + delta_theta / 2.0)
            delta_y = delta_distance * math.sin(self.theta + delta_theta / 2.0)
            
            self.x += delta_x
            self.y += delta_y
            self.theta += delta_theta

            linear_velocity = delta_distance / dt
            angular_velocity = delta_theta / dt

            odom_msg = Odometry()
            odom_msg.header.stamp = current_time.to_msg()
            odom_msg.header.frame_id = "odom_raw"  # The odometry frame
            odom_msg.child_frame_id = "base_link"   # The robot's base frame

            odom_msg.pose.pose.position.x = self.x
            odom_msg.pose.pose.position.y = self.y
            odom_msg.pose.pose.position.z = 0.0


            q = quaternion_from_euler(0, 0, self.theta)
            odom_msg.pose.pose.orientation.x = q[0]
            odom_msg.pose.pose.orientation.y = q[1]
            odom_msg.pose.pose.orientation.z = q[2]
            odom_msg.pose.pose.orientation.w = q[3]

 
            odom_msg.twist.twist.linear.x = linear_velocity
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.angular.z = angular_velocity



            self.odom_publisher.publish(odom_msg)

            t = TransformStamped()
            t.header.stamp = current_time.to_msg()
            t.header.frame_id = "odom_raw"
            t.child_frame_id = "base_link"

            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)


            self.last_left_rad = current_left_rad
            self.last_right_rad = current_right_rad
            self.last_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error in odometry callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    odom_publisher = None
    try:
        odom_publisher = OdometryPublisher()
        rclpy.spin(odom_publisher)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if odom_publisher:
            odom_publisher.get_logger().fatal(f"Unhandled exception: {e}")
    finally:
        if odom_publisher:
            odom_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
