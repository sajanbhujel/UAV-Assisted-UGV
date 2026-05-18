import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion
from visualization_msgs.msg import Marker
import numpy as np
import math
import time
from scipy.linalg import block_diag
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from threading import Lock
import os
import csv



MAP_FILE_PATH = 'global_map_ransac.csv'
INITIAL_THRESHOLD = 0.1   
INLIER_THRESHOLD = 0.008   
MIN_INLIERS = 40 
# 2. Robot Physical Parameters
WHEEL_BASE_B = 0.126  # Distance between wheels (meters)
WHEEL_RADIUS = 0.040  # Wheel radius (meters)

KL = 0.04698  
KR = 0.04783 

R_NOISE = np.array([
    [5.29682992e-03, -3.31730780e-03],
    [-3.31730780e-03, 7.35328825e-03]   
])
GATE_THRESHOLD = 5  

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EKFLocalizationNode(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')

        self.lock = Lock()

   
        self.last_left_rad = None
        self.last_right_rad = None

        self.global_map = self.load_map_from_csv(MAP_FILE_PATH)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.cb_group = ReentrantCallbackGroup()
        self.scan_skip_count = 0

        self.x = np.zeros((3, 1)) 
        
   
        self.P = np.diag([0.125, 0.125, 1.25]) 

      
        self.last_odom_time = self.get_clock().now()


        self.joint_sub = self.create_subscription(
            JointState,
            'encoder/joint_states',
            self.encoder_callback,
            10,
            callback_group=self.cb_group
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan_corrected',
            self.scan_callback,
            10,
            callback_group=self.cb_group
        )

    
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/ekf_pose',
            10
        )

        self.line_viz_pub = self.create_publisher(
            Marker,
            '/ekf/observed_lines',
            10
        )

        self.get_logger().info("EKF Node Started. Waiting for data...")

    def load_map_from_csv(self, file_path):
        """Reads [theta, rho] lines from CSV."""
        if not os.path.exists(file_path):
            self.get_logger().error(f"Map file not found at: {file_path}")
            self.get_logger().error("Please generate map first or check path.")
            return []

        loaded_map = []
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Skip header ['theta', 'rho']
                for row in reader:
                    if len(row) >= 2:
                        theta = float(row[0])
                        rho = float(row[1])
                        loaded_map.append([theta, rho]) # [alpha, r]
            
            self.get_logger().info(f"Loaded {len(loaded_map)} lines from {file_path}")
            return loaded_map
        except Exception as e:
            self.get_logger().error(f"Failed to read map CSV: {e}")
            return []
    # =========================================================
    # PART 1: PREDICTION STEP (Motion Update)
    # =========================================================
    def encoder_callback(self, msg):

        current_left_rad = msg.position[0]
        current_right_rad = msg.position[1]
        
  
        if self.last_left_rad is None:
            self.last_left_rad = current_left_rad
            self.last_right_rad = current_right_rad
            return

 
        delta_l_rad = current_left_rad - self.last_left_rad
        delta_r_rad = current_right_rad - self.last_right_rad
        

        self.last_left_rad = current_left_rad
        self.last_right_rad = current_right_rad

 
        ds_l = delta_l_rad * WHEEL_RADIUS
        ds_r = delta_r_rad * WHEEL_RADIUS
        
     
        ds = (ds_r + ds_l) / 2.0
        d_theta = (ds_r - ds_l) / WHEEL_BASE_B

        with self.lock:
            theta = self.x[2, 0]
            theta_mid = theta + d_theta / 2.0

            # State Projection
            self.x[0, 0] += ds * math.cos(theta_mid)
            self.x[1, 0] += ds * math.sin(theta_mid)
            self.x[2, 0] = normalize_angle(self.x[2, 0] + d_theta)

            # Jacobians
            Fx = np.eye(3)
            Fx[0, 2] = -ds * math.sin(theta_mid)
            Fx[1, 2] =  ds * math.cos(theta_mid)

            term1 = 0.5 * math.cos(theta_mid) - (ds / (2 * WHEEL_BASE_B)) * math.sin(theta_mid)
            term2 = 0.5 * math.cos(theta_mid) + (ds / (2 * WHEEL_BASE_B)) * math.sin(theta_mid)
            term3 = 0.5 * math.sin(theta_mid) + (ds / (2 * WHEEL_BASE_B)) * math.cos(theta_mid)
            term4 = 0.5 * math.sin(theta_mid) - (ds / (2 * WHEEL_BASE_B)) * math.cos(theta_mid)
            
            Fu = np.array([
                [term1, term2],
                [term3, term4],
                [-1/WHEEL_BASE_B, 1/WHEEL_BASE_B]
            ])

            # Motion Noise
            Qt = np.array([
                [KR * abs(ds_r), 0],
                [0, KL * abs(ds_l)]
            ])

            # Covariance Projection
            self.P = (Fx @ self.P @ Fx.T) + (Fu @ Qt @ Fu.T)

            # Publish Pose
            self.publish_pose(msg.header.stamp)


    # =========================================================
    # PART 2: CORRECTION STEP (Measurement Update)
    # =========================================================
    def scan_callback(self, msg):

        self.scan_skip_count += 1
        if self.scan_skip_count % 5 != 0:
            return
        observed_lines = self.extract_lines_from_scan(msg)
        
        if observed_lines:
            self.publish_observed_lines(observed_lines)

        if not observed_lines:
            return

        # 2. Matching (Data Association)
        matches = self.match_lines(observed_lines)

        print(f"Found {len(observed_lines)} line and {len(matches)} valid matches.")  # --- IGNORE ---
        if not matches:
            return

        # 3. Stack Matrices for Update
        v_list = []
        H_list = []
        R_list = []

        for m in matches:
            v_list.append(m['innovation'])
            H_list.append(m['jacobian'])
            R_list.append(R_NOISE)

    
        v_composite = np.vstack(v_list)
  
        H_composite = np.vstack(H_list)
        
        R_composite = block_diag(*R_list)


        S = (H_composite @ self.P @ H_composite.T) + R_composite

 
        try:
            K = self.P @ H_composite.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular matrix in EKF update. Skipping.")
            return

        # State Update (Eq 5.84)
        correction = K @ v_composite
        self.x = self.x + correction
        self.x[2, 0] = normalize_angle(self.x[2, 0])

        # Covariance Update (Eq 5.85)
        I = np.eye(3)
        self.P = (I - K @ H_composite) @ self.P

        self.publish_pose(self.get_clock().now().to_msg())

    def publish_observed_lines(self, lines):

        marker = Marker()
        marker.header.frame_id = "base_link" # RANSAC lines are relative to robot
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ransac_lines"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03 # Line width

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        viz_length = 5.0 
        for theta, rho in lines:
            c = math.cos(theta)
            s = math.sin(theta)
            
            # Point closest to origin
            x0 = rho * c
            y0 = rho * s
  
            dir_x = -s
            dir_y = c
            
            p1 = Point()
            p1.x = x0 + dir_x * viz_length
            p1.y = y0 + dir_y * viz_length
            p1.z = 0.0
            
            p2 = Point()
            p2.x = x0 - dir_x * viz_length
            p2.y = y0 - dir_y * viz_length
            p2.z = 0.0
            
            marker.points.append(p1)
            marker.points.append(p2)
            
        self.line_viz_pub.publish(marker)

    # =========================================================
    # HELPER: Line Matching (Mahalanobis)
    # =========================================================
    def match_lines(self, observed_lines):

        if not self.global_map:
            return []

        with self.lock:
            x_t, y_t, theta_t = self.x.flatten()
            P_t = self.P.copy()

        
        valid_matches = []

        for i, z_i in enumerate(observed_lines):
            z_alpha_i, z_r_i = z_i
            
            min_mahalanobis = float('inf')
            best_match_entry = None
            
      
            for j, m_j in enumerate(self.global_map):
                w_alpha_j, w_r_j = m_j # World frame alpha and r
                
                pred_alpha = w_alpha_j - theta_t
                
                # h_2: Predicted r
                # r - (x * cos(alpha) + y * sin(alpha))
                pred_r = w_r_j - (x_t * np.cos(w_alpha_j) + y_t * np.sin(w_alpha_j))
                
                z_hat_j = np.array([
                    [pred_alpha],
                    [pred_r]
                ])
                
                
                H_j = np.array([
                    [0, 0, -1],
                    [-np.cos(w_alpha_j), -np.sin(w_alpha_j), 0]
                ])
                

                
                z_i_vec = np.array([[z_alpha_i], [z_r_i]])
                v_ij = z_i_vec - z_hat_j
                
                # CRITICAL: Normalize the angle error
                v_ij[0, 0] = normalize_angle(v_ij[0, 0])
                

                
                Sigma_IN = (H_j @ P_t @ H_j.T) + R_NOISE
                

                
                try:
                    inv_Sigma = np.linalg.inv(Sigma_IN)
                    d_squared = (v_ij.T @ inv_Sigma @ v_ij).item()
                    
     
                    if d_squared < min_mahalanobis:
                        min_mahalanobis = d_squared
                        # Store all data needed for the Correction Step
                        best_match_entry = {
                            'observation_index': i,
                            'map_index': j,
                            'innovation': v_ij,         # v_t in Eq 5.84
                            'jacobian': H_j,            # H_t in Eq 5.86
                            'innovation_cov': Sigma_IN, # Sigma_IN in Eq 5.86
                            'mahalanobis': d_squared
                        }
                        
                except np.linalg.LinAlgError:
                    print(f"Singular Matrix for map line {j}. Skipping.")
                    continue


            if best_match_entry is not None and min_mahalanobis <= GATE_THRESHOLD:
                valid_matches.append(best_match_entry)

        return valid_matches

    # =========================================================
    # HELPER: Line Extraction Stub
    # =========================================================
    def extract_lines_from_scan(self, scan_msg):


        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        ranges = np.array(scan_msg.ranges)
        
  
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
            
        # Filter invalid ranges (inf, nan, too close, too far)
        valid_indices = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]
        
        if len(valid_ranges) < MIN_INLIERS:
            return []

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))
        
        segments = []
        current_segment_start_index = 0
        for i in range(len(points) - 1):
            dist = np.linalg.norm(points[i+1] - points[i])
            if dist > INITIAL_THRESHOLD:
                segment = points[current_segment_start_index : i+1]
                if len(segment) >= MIN_INLIERS:
                    segments.append(segment)
                current_segment_start_index = i + 1
        last_segment = points[current_segment_start_index:]
        if len(last_segment) >= MIN_INLIERS:
            segments.append(last_segment)

        all_found_lines = []
        for segment in segments:
            line = self.ransac_find_all_lines(
                segment,
                distance_threshold=INLIER_THRESHOLD,
                min_inliers=MIN_INLIERS
            )
            if line:
                all_found_lines.extend(line)

        polar_lines = []
        for line in all_found_lines:  
            p1, p2 = line
            # line_length = np.linalg.norm(p2 - p1)
            rho, theta = self.cartesian_line_to_polar(p1, p2)
            polar_lines.append((theta, rho))


        return polar_lines 
    
    def ransac_find_all_lines(self, points, distance_threshold=0.1, min_inliers=10, max_iterations=100):
        found_lines = []
        remaining_points = points.copy()

 
        while len(remaining_points) >= min_inliers:
            best_line = None
            best_inlier_count = 0
            best_inlier_indices = None

            for _ in range(max_iterations):
                sample_indices = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[sample_indices]
                
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = -A * p1[0] - B * p1[1]
                norm = np.sqrt(A**2 + B**2)
                if norm == 0: continue

                distances = np.abs(A * remaining_points[:, 0] + B * remaining_points[:, 1] + C) / norm
                inlier_indices = np.where(distances < distance_threshold)[0]
                
                if len(inlier_indices) > best_inlier_count and len(inlier_indices) >= min_inliers:
                    best_inlier_count = len(inlier_indices)
                    best_inlier_indices = inlier_indices
            
            if best_inlier_indices is not None:
                # Refit the line using all the best inliers found
                best_inlier_points = remaining_points[best_inlier_indices]
                centroid = np.mean(best_inlier_points, axis=0)
                _, _, Vh = np.linalg.svd(best_inlier_points - centroid)
                direction_vector = Vh[0, :]
                
                projections = best_inlier_points.dot(direction_vector)
                start_point = best_inlier_points[np.argmin(projections)]
                end_point = best_inlier_points[np.argmax(projections)]
                best_line = (start_point, end_point)
                found_lines.append(best_line)
                # Remove the inliers from the remaining points
                remaining_points = np.delete(remaining_points, best_inlier_indices, axis=0)
            else:
                break
               
        return found_lines
    
    def cartesian_line_to_polar(self, p1, p2):
        # Line equation: Ax + By + C = 0
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = -A * p1[0] - B * p1[1]
        norm = np.sqrt(A**2 + B**2)

        if norm == 0:
            return 0, 0 # Handle degenerate case


        if C <= 0:

            rho = -C / norm
            theta = np.arctan2(B, A) 
        else:

            rho = C / norm
            theta = np.arctan2(-B, -A) 

        return rho, theta
    # =========================================================
    # HELPER: Publishing
    # =========================================================
    def publish_pose(self, stamp_msg):
        # --- 1. Publish the Topic (/ekf_pose) ---
        msg = PoseWithCovarianceStamped()
        
        # Copy timestamp manually to avoid assertion errors
        msg.header.stamp.sec = stamp_msg.sec
        msg.header.stamp.nanosec = stamp_msg.nanosec
        msg.header.frame_id = "map" 

        msg.pose.pose.position.x = self.x[0, 0]
        msg.pose.pose.position.y = self.x[1, 0]
        
        # Yaw to Quaternion
        cy = math.cos(self.x[2, 0] * 0.5)
        sy = math.sin(self.x[2, 0] * 0.5)
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy

        # Fill Covariance
        cov = np.zeros(36)
        cov[0] = self.P[0, 0]; cov[1] = self.P[0, 1]; cov[5] = self.P[0, 2]
        cov[7] = self.P[1, 1]; cov[11] = self.P[1, 2]; cov[35] = self.P[2, 2]
        msg.pose.covariance = cov.tolist()

        self.pose_pub.publish(msg)


        t = TransformStamped()

        # Copy timestamp
        t.header.stamp.sec = stamp_msg.sec
        t.header.stamp.nanosec = stamp_msg.nanosec
        
        # Define the Frame Relationship
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link' 

        # Translation
        t.transform.translation.x = self.x[0, 0]
        t.transform.translation.y = self.x[1, 0]
        t.transform.translation.z = 0.0

        # Rotation
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy


        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()