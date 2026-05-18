#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
import time
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion

# --- CONTROLLER PARAMS ---
LOOKAHEAD_DIST = 0.3   # Meters
MAX_SPEED = 0.3        # m/s
MIN_SPEED = 0.22       # m/s
CURVATURE_GAIN = 1.0   
GOAL_TOLERANCE = 0.05 

# --- STOP-AND-GO PARAMS ---
MOVE_DISTANCE_LIMIT = 0.6  
WAIT_DURATION = 4.0       

class RegulatedPurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')
        
        # Subscribers
        self.sub_path = self.create_subscription(Path, '/path', self.path_cb, 10)
        self.sub_odom = self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.odom_cb, 10)
        
        # Publishers
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_lookahead = self.create_publisher(Marker, '/lookahead_marker', 10)
        self.pub_arc = self.create_publisher(Path, '/predicted_arc', 10)
        
        self.path = []
        self.robot_pose = None
        
       
        self.state = "MOVING" # Options: "MOVING", "WAITING", "DONE"
        self.start_move_pose = None # To track distance traveled
        self.wait_start_time = None # To track wait duration
        
        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz
        self.get_logger().info(f"Pure Pursuit Ready (Step Mode: {MOVE_DISTANCE_LIMIT}m move / {WAIT_DURATION}s wait)")

    def path_cb(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"Path received with {len(self.path)} points")
        # Reset state on new path
        self.state = "MOVING"
        self.start_move_pose = None 

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.robot_pose = (pos.x, pos.y, yaw)

    def control_loop(self):
        if not self.path or self.robot_pose is None: return
        
      
        if self.start_move_pose is None:
            self.start_move_pose = self.robot_pose

        # --- STATE MACHINE ---
        if self.state == "WAITING":
            self.stop_robot()
            
            # Check if wait time is over
            elapsed = time.time() - self.wait_start_time
            if elapsed >= WAIT_DURATION:
                self.get_logger().info("EKF settled. Resuming movement.")
                self.state = "MOVING"
                self.start_move_pose = self.robot_pose # Reset distance tracker
            return # Skip the rest of the loop while waiting

        elif self.state == "DONE":
            self.stop_robot()
            return

        # --- MOVING STATE LOGIC ---
        rx, ry, ryaw = self.robot_pose
        
      
        dist_traveled = math.sqrt((rx - self.start_move_pose[0])**2 + (ry - self.start_move_pose[1])**2)
        
        if dist_traveled >= MOVE_DISTANCE_LIMIT:
            self.get_logger().info(f"Moved {dist_traveled:.2f}m. Pausing for EKF update...")
            self.state = "WAITING"
            self.wait_start_time = time.time()
            self.stop_robot()
            return

  
        target = self.get_lookahead_point(rx, ry)
        
        if target is None:
            self.state = "DONE"
            self.stop_robot()
            return

        tx, ty = target
        self.publish_lookahead_marker(tx, ty)

        dx = tx - rx
        dy = ty - ry
        target_x_robot = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        target_y_robot = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        
        dist = math.sqrt(dx**2 + dy**2)
     
        curvature = (2.0 * target_y_robot) / (dist**2) if dist > 0.01 else 0.0
        
    
        v = MAX_SPEED / (1.0 + CURVATURE_GAIN * abs(curvature))
        v = max(v, MIN_SPEED)
        
        # 6. Goal Check
        end_dist = math.sqrt((self.path[-1][0]-rx)**2 + (self.path[-1][1]-ry)**2)
        
        if end_dist < LOOKAHEAD_DIST:
            if end_dist < GOAL_TOLERANCE:
                self.get_logger().info("Goal Reached!")
                self.path = [] 
                self.state = "DONE"
                self.stop_robot()
                return
            else:
                v = MIN_SPEED 

        w = v * curvature

        if v > 0:
            self.publish_predicted_arc(rx, ry, ryaw, v, w)

        # Publish Command
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub_vel.publish(cmd)
        print(f"[{self.state}] Dist: {dist_traveled:.2f}/{MOVE_DISTANCE_LIMIT}m | Cmd: v={v:.2f}, w={w:.2f}")

    def get_lookahead_point(self, rx, ry):
        closest_idx = -1
        min_dist = float('inf')
        
        for i, (px, py) in enumerate(self.path):
            d = math.sqrt((px-rx)**2 + (py-ry)**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        for i in range(closest_idx, len(self.path)):
            px, py = self.path[i]
            d = math.sqrt((px-rx)**2 + (py-ry)**2)
            if d >= LOOKAHEAD_DIST:
                return (px, py)
        
        if closest_idx >= len(self.path) - 10:
            return self.path[-1]
            
        return None

    def publish_lookahead_marker(self, tx, ty):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = tx
        marker.pose.position.y = ty
        marker.pose.position.z = 0.1
        marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0
        self.pub_lookahead.publish(marker)

    def publish_predicted_arc(self, rx, ry, ryaw, v, w):
        arc_msg = Path()
        arc_msg.header.frame_id = "map"
        arc_msg.header.stamp = self.get_clock().now().to_msg()
        
        dt = 0.05
        sim_time = 1.5 
        steps = int(sim_time / dt)
        
        sim_x, sim_y, sim_yaw = rx, ry, ryaw
        
        for _ in range(steps):
            pose = PoseStamped()
            pose.pose.position.x = sim_x
            pose.pose.position.y = sim_y
            arc_msg.poses.append(pose)
            
            sim_x += v * math.cos(sim_yaw) * dt
            sim_y += v * math.sin(sim_yaw) * dt
            sim_yaw += w * dt

        self.pub_arc.publish(arc_msg)

    def stop_robot(self):
        self.pub_vel.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = RegulatedPurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()