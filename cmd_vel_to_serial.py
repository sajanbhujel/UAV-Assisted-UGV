#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial 
import sys
import math

class CmdVelToSerial(Node):

    def __init__(self):
        super().__init__('cmd_vel_to_serial')
        
        # --- Parameters ---
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_separation', 0.126) 

        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        self.wheel_separation = self.get_parameter('wheel_separation').get_parameter_value().double_value

        self.get_logger().info(f"Using wheel_separation: {self.wheel_separation}m")

        # --- Serial Port Setup ---
        self.serial_port_obj = None
        try:
            self.get_logger().info(f"Connecting to serial port {serial_port} at {baud_rate}...")
            self.serial_port_obj = serial.Serial(serial_port, baud_rate, timeout=1)
            self.get_logger().info("Serial connection successful (Write-Only Node).")
        except Exception as e:
            self.get_logger().fatal(f"Failed to open serial port {serial_port}: {e}")


    
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.get_logger().info('Subscribed to /cmd_vel topic.')

      
        self.timer = self.create_timer(0.2, self.check_timeout)
        self.last_cmd_vel_time = self.get_clock().now().nanoseconds

    def cmd_vel_callback(self, msg):
        """Receives Twist messages and sends motor commands via serial."""
        self.last_cmd_vel_time = self.get_clock().now().nanoseconds
        
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # --- Differential Drive Kinematics ---
        left_wheel_speed_ms = linear_x - (angular_z * self.wheel_separation / 2.0)
        right_wheel_speed_ms = linear_x + (angular_z * self.wheel_separation / 2.0)
        
        # --- Normalization ---
        # Assumes max speed is approx 1.0 m/s.
        left_normalized = max(-1.0, min(1.0, left_wheel_speed_ms))
        right_normalized = max(-1.0, min(1.0, right_wheel_speed_ms))

        self.send_motor_command(left_normalized, right_normalized)

    def check_timeout(self):
        """Stops the motors if no new command has been received recently."""
        time_since_last_cmd = (self.get_clock().now().nanoseconds - self.last_cmd_vel_time) / 1e9
        
 
        if time_since_last_cmd > 0.5:
            self.get_logger().warn("Command timeout. Stopping motors.", throttle_duration_sec=2.0)
      

    def send_motor_command(self, left_norm, right_norm):
        """Helper to format and write the motor command string to serial."""
        if self.serial_port_obj is None or not self.serial_port_obj.is_open:
        
            if not self.reconnect_serial():
                self.get_logger().warn("Serial port not open. Cannot send motor command.", throttle_duration_sec=5.0)
                return
        right_norm = right_norm * 0.992
        cmd = f"L:{left_norm:.3f},R:{right_norm:.3f}\n"
        
        try:
            self.serial_port_obj.write(cmd.encode('utf-8'))
            self.get_logger().info(f"publishing:{cmd}")
        except Exception as e:
            self.get_logger().warn(f"Failed to write to serial port: {e}")

    def reconnect_serial(self):
        """Attempts to (re)open the serial port."""
        if self.serial_port_obj and self.serial_port_obj.is_open:
            return True # Already open
        
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        
        try:
            self.get_logger().info(f"Attempting to (re)connect to serial port {serial_port}...")
            self.serial_port_obj = serial.Serial(serial_port, baud_rate, timeout=1)
            self.get_logger().info("Serial connection successful.")
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to (re)connect to serial port: {e}", throttle_duration_sec=5.0)
            return False

    def destroy_node(self):
        """Safely stops motors on shutdown."""
        self.get_logger().info("Shutting down node. Sending stop command.")
        self.send_motor_command(0.0, 0.0)
        if self.serial_port_obj and self.serial_port_obj.is_open:
            self.serial_port_obj.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToSerial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()