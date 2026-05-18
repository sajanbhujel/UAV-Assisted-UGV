import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ScanCorrectorNode(Node):
    def __init__(self):
        super().__init__('scan_corrector_node')
        
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.pub = self.create_publisher(
            LaserScan,
            '/scan_corrected',
            10
        )
        
        self.get_logger().info("Scan Corrector Running. Publishing to /scan_corrected")

    def scan_callback(self, msg):
      
        new_msg = LaserScan()
        new_msg.header = msg.header
        new_msg.angle_min = msg.angle_min + math.pi # Add 180 deg
        new_msg.angle_max = msg.angle_max + math.pi # Add 180 deg
        new_msg.angle_increment = msg.angle_increment
        new_msg.time_increment = msg.time_increment
        new_msg.scan_time = msg.scan_time
        new_msg.range_min = msg.range_min
        new_msg.range_max = msg.range_max
        new_msg.ranges = msg.ranges
        new_msg.intensities = msg.intensities
        
     
        self.pub.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanCorrectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()