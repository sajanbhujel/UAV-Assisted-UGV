#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
import math
import heapq
import os
from scipy import interpolate
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import euler_from_quaternion


MAP_IMAGE = "/home/slam/ros2_ws/src/EduBot/EduBot/scaled_occupancy_map.png"
MAP_YAML = "/home/slam/ros2_ws/src/EduBot/EduBot/scaled_occupancy_map.yaml"
ROBOT_RADIUS = 0.15     
HEADING_PROJ_DIST = 0.3 
SMOOTHING_FACTOR = 1.0  

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        self.declare_parameter('use_smoothing', True)
        
   
        self.load_map_from_yaml(MAP_YAML)
        

        self.path_pub = self.create_publisher(Path, '/path', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', qos)
        
 
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
    
        self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.pose_callback, 10)
        
        self.current_pose = None
        self.timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info("A* Planner Ready. Waiting for EKF Pose and Goal...")

    def load_map_from_yaml(self, yaml_file):
        if not os.path.exists(yaml_file):
            self.get_logger().error(f"Map YAML '{yaml_file}' not found!")
            return

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        self.resolution = data['resolution']
        self.origin = data['origin'] # [x, y, yaw]
        
        img_path = data['image']
        if not os.path.exists(img_path):
            img_path = MAP_IMAGE 
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error("Failed to load map image!")
            return

      
        self.map_img = cv2.flip(img, 0) 
        self.h, self.w = self.map_img.shape
        
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = 'map'
        self.map_msg.info.resolution = self.resolution
        self.map_msg.info.width = self.w
        self.map_msg.info.height = self.h
        self.map_msg.info.origin.position.x = float(self.origin[0])
        self.map_msg.info.origin.position.y = float(self.origin[1])
        
        flat_data = self.map_img.flatten()
        grid_data = np.zeros_like(flat_data, dtype=np.int8)
        grid_data[flat_data > 128] = 100 # Walls
        grid_data[flat_data <= 128] = 0  # Free
        self.map_msg.data = grid_data.tolist()

        k_size = int(math.ceil(ROBOT_RADIUS / self.resolution))
        if k_size % 2 == 0: k_size += 1
        self.inflated_map = cv2.dilate(self.map_img, np.ones((k_size, k_size), np.uint8), iterations=1)

    def publish_map(self):
        if hasattr(self, 'map_msg'):
            self.map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.map_msg)

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_yaw(self, orientation):
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        return euler_from_quaternion(q)[2]

    def world_to_grid(self, wx, wy):
        ox, oy = self.origin[0], self.origin[1]
        gx = int((wx - ox) / self.resolution)
        gy = int((wy - oy) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        ox, oy = self.origin[0], self.origin[1]
        wx = (gx * self.resolution) + ox
        wy = (gy * self.resolution) + oy
        return wx, wy

    def goal_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("Waiting for /ekf_pose...")
            return

     
        start_world = (self.current_pose.position.x, self.current_pose.position.y)
        start_yaw = self.get_yaw(self.current_pose.orientation)
        
        goal_world = (msg.pose.position.x, msg.pose.position.y)
        goal_yaw = self.get_yaw(msg.pose.orientation)
        
        self.get_logger().info(f"Planning: {start_world} -> {goal_world}")


        proj_start_world = (
            start_world[0] + HEADING_PROJ_DIST * math.cos(start_yaw),
            start_world[1] + HEADING_PROJ_DIST * math.sin(start_yaw)
        )
        
        # Project Goal BACKWARD
        proj_goal_world = (
            goal_world[0] - HEADING_PROJ_DIST * math.cos(goal_yaw),
            goal_world[1] - HEADING_PROJ_DIST * math.sin(goal_yaw)
        )

        # 3. Convert to Grid
        sx, sy = self.world_to_grid(*start_world)
        gx, gy = self.world_to_grid(*goal_world)
        psx, psy = self.world_to_grid(*proj_start_world)
        pgx, pgy = self.world_to_grid(*proj_goal_world)


        if self.is_collision(psx, psy): 
            self.get_logger().warn("Start Heading blocked! Using direct start.")
            psx, psy = sx, sy
        if self.is_collision(pgx, pgy): 
            self.get_logger().warn("Goal Heading blocked! Using direct goal.")
            pgx, pgy = gx, gy

 
        center_path_px = self.run_astar((psx, psy), (pgx, pgy))
        
        if center_path_px:
       
            full_path_px = [(sx, sy)] + center_path_px + [(gx, gy)]
            
     
            use_smoothing = self.get_parameter('use_smoothing').get_parameter_value().bool_value
            
            if use_smoothing:
         
                pruned_center = self.prune_path(center_path_px)
                full_pruned = [(sx, sy)] + pruned_center + [(gx, gy)]
                final_path_m = self.smooth_path(full_pruned)
                self.get_logger().info("Published Smoothed Path")
            else:
                final_path_m = [self.grid_to_world(x, y) for x, y in full_path_px]
                self.get_logger().info("Published Raw A* Path")

            self.publish_path_msg(final_path_m)
        else:
            self.get_logger().warn("No path found!")

    def is_collision(self, x, y):
        if not (0 <= x < self.w and 0 <= y < self.h): return True
        return self.inflated_map[y, x] > 128

    def run_astar(self, start, goal):
        if self.is_collision(*start) or self.is_collision(*goal): return None
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        motions = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy, cost in motions:
                nx, ny = current[0]+dx, current[1]+dy
                if not self.is_collision(nx, ny):
                    new_g = g_score[current] + cost
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = math.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        heapq.heappush(open_set, (new_g + h, (nx, ny)))
                        came_from[(nx, ny)] = current
        return None

    def prune_path(self, path):
        if len(path) < 3: return path
        pruned = [path[0]]
        curr = 0
        while curr < len(path)-1:
            for i in range(len(path)-1, curr, -1):
                if self.check_line_of_sight(path[curr], path[i]):
                    pruned.append(path[i])
                    curr = i
                    break
            else:
                curr += 1
                pruned.append(path[curr])
        return pruned

    def check_line_of_sight(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        points = np.linspace(0, 1, max(abs(x2-x1), abs(y2-y1)) + 1)
        for t in points:
            x = int(x1 + t*(x2-x1))
            y = int(y1 + t*(y2-y1))
            if self.is_collision(x, y): return False
        return True

    def smooth_path(self, path_px):
        path_m = [self.grid_to_world(x, y) for x, y in path_px]
        if len(path_m) < 3: return path_m
        try:
            ux = [p[0] for p in path_m]
            uy = [p[1] for p in path_m]
            ux_uniq, uy_uniq = [], []
            seen = set()
            for x, y in zip(ux, uy):
                if (x, y) not in seen:
                    ux_uniq.append(x); uy_uniq.append(y); seen.add((x,y))
            if len(ux_uniq) < 3: return path_m
            tck, u = interpolate.splprep([ux_uniq, uy_uniq], s=SMOOTHING_FACTOR, k=2)
            u_new = np.linspace(0, 1, len(path_m) * 10)
            xn, yn = interpolate.splev(u_new, tck)
            return list(zip(xn, yn))
        except:
            return path_m

    def publish_path_msg(self, path_m):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path_m:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()