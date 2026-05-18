import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import csv

# --- CONFIGURATION ---
IMAGE_PATH = "env.jpg"
REAL_HEIGHT_METERS = 2.36

# RANSAC Settings
RANSAC_ITERATIONS = 1000
DISTANCE_THRESH = 3
MIN_INLIERS = 75
MAX_WALLS = 100
GRID_SIZE = 77  
RAY_STEP_DEG = 2
EXTEND_PIXEL = 30

# Connectivity Check
GAP_THRESHOLD = 50.0  

MAP_RESOLUTION_M = 0.01  # 1 pixel = 1 cm
map_filename = "scaled_occupancy_map.png"


CAMERA_MATRIX = np.array([
    [1.42812771e+03, 0.00000000e+00, 6.49218265e+02],
    [0.00000000e+00, 1.42411817e+03, 3.24391559e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Distortion: [k1, k2, p1, p2, k3]
DIST_COEFFS = np.array([
    8.27263406e-02, -9.17308816e-01, -3.95049708e-03,
    -3.05640551e-04, 1.41474775e+00
])

def show_step(title, image):
 
    print(f"Showing: {title} (Press SPACE/ENTER to continue...)")
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fit_line_pca(points):
    """ Fits the best line using PCA (handling vertical lines). """
    cx, cy = np.mean(points, axis=0)
    centered = points - [cx, cy]
    u, s, vh = np.linalg.svd(centered)
    vx, vy = vh[0]
    norm = math.sqrt(vx ** 2 + vy ** 2)
    return vx / norm, vy / norm, cx, cy


def find_lines_from_edges(edges):

    y_idxs, x_idxs = np.where(edges > 0)
    all_points = np.column_stack((x_idxs, y_idxs))

    output_lines = []

    print(f"Starting Orthogonal RANSAC on {len(all_points)} points...")

    wall_idx = 0
    while len(all_points) > MIN_INLIERS and wall_idx < MAX_WALLS:

        best_inliers = []

        # --- RANSAC LOOP ---
        for _ in range(RANSAC_ITERATIONS):
            if len(all_points) < 2: break
            idx = np.random.choice(len(all_points), 2, replace=False)
            p1, p2 = all_points[idx]

            if np.linalg.norm(p1 - p2) < 10: continue

            # Line Model
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            nx, ny = -vy, vx
            norm = math.sqrt(nx ** 2 + ny ** 2)
            if norm == 0: continue
            nx, ny = nx / norm, ny / norm
            c = - (nx * p1[0] + ny * p1[1])

            # Find Inliers
            distances = np.abs(nx * all_points[:, 0] + ny * all_points[:, 1] + c)
            current_inliers = all_points[distances < DISTANCE_THRESH]

            if len(current_inliers) > len(best_inliers):
                best_inliers = current_inliers

        if len(best_inliers) < MIN_INLIERS:
            break

        # --- SEGMENTATION LOGIC ---

        # 1. Project to 1D
        vx, vy, cx, cy = fit_line_pca(best_inliers)
        dx = best_inliers[:, 0] - cx
        dy = best_inliers[:, 1] - cy
        projections = dx * vx + dy * vy

        # 2. Sort points by position along the line
        sort_idxs = np.argsort(projections)
        best_inliers = best_inliers[sort_idxs]
        projections = projections[sort_idxs]

        # 3. Split into segments based on Gaps
        segments = []
        current_segment_start = 0

        for i in range(len(projections) - 1):
            dist = projections[i + 1] - projections[i]
            if dist > GAP_THRESHOLD:
                segments.append(best_inliers[current_segment_start: i + 1])
                current_segment_start = i + 1

        segments.append(best_inliers[current_segment_start:])

        valid_segments = [seg for seg in segments if len(seg) > MIN_INLIERS]

        # 5. Check if we found ANY valid segments
        if not valid_segments:
            print(f"  -> Rejected candidate. All {len(segments)} segments were too small.")


            nx, ny = -vy, vx
            c = - (nx * cx + ny * cy)
            dists = np.abs(nx * all_points[:, 0] + ny * all_points[:, 1] + c)
            all_points = all_points[dists > DISTANCE_THRESH]
            continue

        print(f"  -> Found {len(valid_segments)} valid wall segments.")


        global_nx, global_ny = -vy, vx
        global_c = - (global_nx * cx + global_ny * cy)

        # Process ALL valid segments
        for best_inliers in valid_segments:

            vx, vy, cx, cy = fit_line_pca(best_inliers)

            dx = best_inliers[:, 0] - cx
            dy = best_inliers[:, 1] - cy
            projections = dx * vx + dy * vy

            # --- EXTEND LINE ---
            min_proj = np.min(projections)
            max_proj = np.max(projections)

            mid_proj = (min_proj + max_proj) / 2
            half_length = (max_proj - min_proj) / 2

            # Using your EXTEND_PIXEL logic
            scaled_half_length = half_length + EXTEND_PIXEL

            new_min_proj = mid_proj - scaled_half_length
            new_max_proj = mid_proj + scaled_half_length

            p1_x = cx + new_min_proj * vx
            p1_y = cy + new_min_proj * vy
            p2_x = cx + new_max_proj * vx
            p2_y = cy + new_max_proj * vy

            output_lines.append([int(p1_x), int(p1_y), int(p2_x), int(p2_y)])
            wall_idx += 1


        distances = np.abs(global_nx * all_points[:, 0] + global_ny * all_points[:, 1] + global_c)
        all_points = all_points[distances > (DISTANCE_THRESH * 2.0)]

        wall_idx += 1

    print(f"Done. Found {len(output_lines)} clean walls.")
    return output_lines

def get_line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def ray_intersect_line(ray_origin, ray_dir, line):

    rx, ry = ray_origin
    rdx, rdy = ray_dir
    x1, y1, x2, y2 = line


    ax, ay = x1, y1
    sx, sy = x2 - x1, y2 - y1

    # Cross product 2D
    cross = rdx * sy - rdy * sx
    if abs(cross) < 1e-6: return None, None  # Parallel

    t = ((ax - rx) * sy - (ay - ry) * sx) / cross
    u = ((ax - rx) * rdy - (ay - ry) * rdx) / cross

    if t >= 0 and 0 <= u <= 1:
        # Intersection is valid
        ix = rx + t * rdx
        iy = ry + t * rdy
        return t, (ix, iy)
    return None, None

def get_segment_intersection(line1, line2):

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0: return None  # Parallel

    ua_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    ub_num = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)

   
    ua = ua_num / denom
    ub = ub_num / denom

   
    epsilon = 1e-5
    if -epsilon <= ua <= 1 + epsilon and -epsilon <= ub <= 1 + epsilon:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)

    return None

def dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def normalize_angle(angle):
    """ Normalizes angle to range [-pi, pi] """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def plot_with_interactive_origin(img, lines, scale_factor):
    print("Please select the ORIGIN (0,0) point on the popup window...")

    # --- STEP 1: SELECT ORIGIN ---
    fig_select = plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("CLICK the Robot's Origin (0,0), then press ENTER", fontsize=14, color='red')
    plt.axis('off')

    # Get 1 click
    pts = plt.ginput(n=1, timeout=-1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    plt.close(fig_select)

    if not pts:
        print("No point selected! Defaulting to image center.")
        origin_x, origin_y = img.shape[1] // 2, img.shape[0] // 2
    else:
        origin_x, origin_y = pts[0]
        print(f"Origin Set: Pixel({origin_x:.1f}, {origin_y:.1f})")

    # --- STEP 2: TRANSFORM & PLOT ---
    plt.figure(figsize=(12, 12))

  
    h, w = img.shape[:2]

    extent_left = -origin_x * scale_factor
    extent_right = (w - origin_x) * scale_factor
    extent_bottom = (origin_y - h) * scale_factor
    extent_top = origin_y * scale_factor

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
               extent=[extent_left, extent_right, extent_bottom, extent_top],
               alpha=0.6)

    print("\n--- Cartesian Coordinates (Meters) & Polar Conversion ---")

    csv_data = []

    for i, line in enumerate(lines):
        px1, py1, px2, py2 = line

        # --- 1. VISUALIZATION COORDINATES (Standard Graph: Y is UP) ---
        vis_mx1 = (px1 - origin_x) * scale_factor
        vis_my1 = (origin_y - py1) * scale_factor
        vis_mx2 = (px2 - origin_x) * scale_factor
        vis_my2 = (origin_y - py2) * scale_factor

        # Plot on screen (Visual check)
        plt.plot([vis_mx1, vis_mx2], [vis_my1, vis_my2], color='blue', linewidth=3, solid_capstyle='round')
        plt.plot([vis_mx1, vis_mx2], [vis_my1, vis_my2], 'ro', markersize=4)

        length = math.sqrt((vis_mx2 - vis_mx1) ** 2 + (vis_my2 - vis_my1) ** 2)
        mid_x, mid_y = (vis_mx1 + vis_mx2) / 2, (vis_my1 + vis_my2) / 2
        plt.text(mid_x, mid_y, f"{length:.2f}m", color='white', fontsize=9, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(facecolor='blue', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        # --- 2. ROBOT COORDINATES (TRANSFORMED) ---
        FLIP_X = True
        FLIP_Y = True  

        rx1 = -vis_mx1 if FLIP_X else vis_mx1
        ry1 = -vis_my1 if FLIP_Y else vis_my1
        rx2 = -vis_mx2 if FLIP_X else vis_mx2
        ry2 = -vis_my2 if FLIP_Y else vis_my2

        # --- 3. CALCULATE POLAR (rho, theta) ---
        # Line Eq: Ax + By + C = 0
        A = ry1 - ry2
        B = rx2 - rx1
        C = rx1 * ry2 - rx2 * ry1

        norm = math.sqrt(A * A + B * B)

        if norm != 0:
            signed_dist = -C / norm
            nx = A / norm
            ny = B / norm

            # Ensure rho is positive
            if signed_dist < 0:
                rho = -signed_dist
                nx = -nx
                ny = -ny
            else:
                rho = signed_dist

            # Theta calculation
            theta_rad = math.atan2(ny, nx)

            # Normalize to [-pi, pi]
            theta_rad = normalize_angle(theta_rad)
            # theta_deg = math.degrees(theta_rad)

            csv_data.append([theta_rad, rho])
            print(f"Line {i + 1}: rho={rho:.3f}m, theta={theta_rad:.4f} rad")

    # --- STEP 4: SAVE TO CSV ---
    csv_filename = "map_lines_robot_frame.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["theta", "rho"])  # Header
        writer.writerows(csv_data)

    print(f"\nSuccessfully saved {len(csv_data)} lines to '{csv_filename}'")
    print("NOTE: Y-axis was flipped to match robot coordinates.")

    # --- STEP 5: FINAL PLOT ---
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')
    plt.xlabel("X Distance (meters)", fontsize=25)
    plt.ylabel("Y Distance (meters)", fontsize=25)
    plt.title(f"Visual Map (Saved as Robot Frame)", fontsize=25)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    return (origin_x, origin_y)


def generate_occupancy_grid(lines, scale_factor, origin_px, resolution_m=0.01, padding_px=20, flip_x=True, flip_y=True):

    print(f"\nGenerating Scaled Map (Res: {resolution_m * 100:.1f} cm/pixel)...")

    ox, oy = origin_px
    scaled_lines_robot_frame = []

    all_x = []
    all_y = []

    for l in lines:
        px1, py1, px2, py2 = l

        # Standard Cartesian (Relative to Origin)
        mx1 = (px1 - ox) * scale_factor
        my1 = (oy - py1) * scale_factor
        mx2 = (px2 - ox) * scale_factor
        my2 = (oy - py2) * scale_factor

        # Apply Flips (Match your plot logic)
        rx1 = -mx1 if flip_x else mx1
        ry1 = -my1 if flip_y else my1
        rx2 = -mx2 if flip_x else mx2
        ry2 = -my2 if flip_y else my2

        scaled_lines_robot_frame.append([rx1, ry1, rx2, ry2])
        all_x.extend([rx1, rx2])
        all_y.extend([ry1, ry2])

    if not all_x: return np.zeros((100, 100), dtype=np.uint8), 0, 0, 0

    # 2. Determine Bounding Box in Robot Frame
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width_m = max_x - min_x
    height_m = max_y - min_y

    print(f"Map Bounds (Robot Frame): X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")

    # 3. Create Grid Image
    grid_w = int(math.ceil(width_m / resolution_m)) + (padding_px * 2)
    grid_h = int(math.ceil(height_m / resolution_m)) + (padding_px * 2)

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)



    for l in scaled_lines_robot_frame:
        rx1, ry1, rx2, ry2 = l

        ix1 = int((rx1 - min_x) / resolution_m) + padding_px
        iy1 = int((max_y - ry1) / resolution_m) + padding_px

        ix2 = int((rx2 - min_x) / resolution_m) + padding_px
        iy2 = int((max_y - ry2) / resolution_m) + padding_px

        cv2.line(grid, (ix1, iy1), (ix2, iy2), 255, thickness=2)


    return grid, min_x, max_y


def save_map_metadata(filename, image_shape, resolution, min_x, max_y, padding_px=20):
    """
    Calculates the YAML Origin for the flipped map.
    The YAML origin is the World Coordinate of the image's Bottom-Left pixel (0, H-1).
    """
    h_grid, w_grid = image_shape


    origin_x = min_x - (padding_px * resolution)

    origin_y = max_y - (h_grid - padding_px) * resolution

    metadata = {
        'image': filename,
        'resolution': resolution,
        'origin': [float(origin_x), float(origin_y), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }

    yaml_filename = filename.replace('.png', '.yaml').replace('.jpg', '.yaml')
    with open(yaml_filename, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=None)

    print(f"\n[Metadata] Saved {yaml_filename}")
    print(f"[Metadata] Origin (Bottom-Left): [{origin_x:.3f}, {origin_y:.3f}]")

def main():
    np.random.seed(42)  # Fix seed
    img = cv2.imread(IMAGE_PATH)
    if img is None: print("Image error"); return
    
    # --- APPLY CALIBRATION (UNDISTORT) ---
    print("Undistorting image with calibration data...")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w,h), 1, (w,h))
    img = cv2.undistort(img, CAMERA_MATRIX, DIST_COEFFS, None, newcameramtx)
    
    h, w = img.shape[:2]

    # 1. Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    show_step("Step 1: Canny Edges", edges)

    # 2. Get Lines (Refactored Function)
    found_lines = find_lines_from_edges(edges)

    # 3. Visualization
    vis_img = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for i, line in enumerate(found_lines):
        x1, y1, x2, y2 = line
        color = colors[i % len(colors)]
        cv2.line(vis_img, (x1, y1), (x2, y2), color, 3)
 

    show_step("Step 2: Ransac Lines", vis_img)

  
    print("Generating Grid...")
    safe_cells = [] 
    vis_grid = img.copy()

    # Iterate grid
    for y in range(0, h, GRID_SIZE):
        for x in range(0, w, GRID_SIZE):
            # Define ROI
            roi = edges[y:y + GRID_SIZE, x:x + GRID_SIZE]

        
            if cv2.countNonZero(roi) == 0:
                # Completely black (Safe)
                cx = x + GRID_SIZE // 2
                cy = y + GRID_SIZE // 2

             
                if cx < w and cy < h:
                    safe_cells.append((cx, cy))
                    # Draw Green Square
                    cv2.rectangle(vis_grid, (x, y), (x + GRID_SIZE, y + GRID_SIZE), (0, 255, 0), 1)
            else:
                # Has edge (Unsafe) - Draw Red X
                cv2.line(vis_grid, (x, y), (x + GRID_SIZE, y + GRID_SIZE), (0, 0, 255), 1)
                cv2.line(vis_grid, (x + GRID_SIZE, y), (x, y + GRID_SIZE), (0, 0, 255), 1)

    show_step("Step 4: Grid (Green=Safe, Red=Edge)", vis_grid)


    print(f"Raycasting from {len(safe_cells)} safe cells...")

    final_lines_indices = set()  
    vis_rays = img.copy()

    # For every safe cell
    for i, center in enumerate(safe_cells):
        if i % 10 == 0: print(f"Processing cell {i}/{len(safe_cells)}...", end='\r')
        cx, cy = center

       
        for deg in range(0, 360, RAY_STEP_DEG):
            rad = math.radians(deg)
            ray_dir = (math.cos(rad), math.sin(rad))

            closest_dist = float('inf')
            closest_line_idx = -1
            hit_point = None

     
            for idx, m_line in enumerate(found_lines):
                dist, point = ray_intersect_line(center, ray_dir, m_line)

                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    closest_line_idx = idx
                    hit_point = point

        
            if closest_line_idx != -1:
                # Ray Angle
                ray_angle = deg

          
                lx1, ly1, lx2, ly2 = found_lines[closest_line_idx]
                line_angle = get_line_angle(lx1, ly1, lx2, ly2)

            
                diff = abs(ray_angle - line_angle) % 180

                deviation = abs(diff - 90)

                if deviation <= 1:  # 88 to 92 degrees
                    final_lines_indices.add(closest_line_idx)

                
                    cv2.line(vis_rays, center, (int(hit_point[0]), int(hit_point[1])), (0, 255, 255), 1)

    print("\nRaycasting complete.")
    show_step("Step 5: Valid Rays (Yellow)", vis_rays)

    # --- STEP 6: FINAL OUTPUT ---
    vis_final = np.zeros_like(img)

  
    valid_count = 0
    for idx in final_lines_indices:
        l = found_lines[idx]
        cv2.line(vis_final, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 2)
        valid_count += 1

    print(f"Final Map: Kept {valid_count} walls out of {len(found_lines)} candidates.")
    show_step("Step 6: Final Filtered Map", vis_final)



    final_lines_list = [list(found_lines[i]) for i in final_lines_indices]

    print(f"Snapping crossing corners for {len(final_lines_list)} lines...")


    for _ in range(3):
        points_changed = False

        for i in range(len(final_lines_list)):
            for j in range(i + 1, len(final_lines_list)):
                l1 = final_lines_list[i]
                l2 = final_lines_list[j]

                # 1. Get segment intersection
                pt = get_segment_intersection(l1, l2)
                if pt is None: continue

                ix, iy = pt
                new_ix, new_iy = int(ix), int(iy)

              
                d1_start = dist_sq((ix, iy), (l1[0], l1[1]))
                d1_end = dist_sq((ix, iy), (l1[2], l1[3]))

                # Target index 0 for start (x1,y1), 2 for end (x2,y2)
                l1_target = 0 if d1_start < d1_end else 2

                
                d2_start = dist_sq((ix, iy), (l2[0], l2[1]))
                d2_end = dist_sq((ix, iy), (l2[2], l2[3]))

                l2_target = 0 if d2_start < d2_end else 2

                if final_lines_list[i][l1_target] != new_ix or final_lines_list[i][l1_target + 1] != new_iy:
                    final_lines_list[i][l1_target] = new_ix
                    final_lines_list[i][l1_target + 1] = new_iy
                    points_changed = True

                if final_lines_list[j][l2_target] != new_ix or final_lines_list[j][l2_target + 1] != new_iy:
                    final_lines_list[j][l2_target] = new_ix
                    final_lines_list[j][l2_target + 1] = new_iy
                    points_changed = True

        if not points_changed:
            break

    # --- VISUALIZATION ---
    vis_snapped = img.copy()

    for l in final_lines_list:
        cv2.line(vis_snapped, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 2)
        # Draw corners to verify
        cv2.circle(vis_snapped, (l[0], l[1]), 3, (0, 0, 255), -1)
        cv2.circle(vis_snapped, (l[2], l[3]), 3, (0, 0, 255), -1)

    show_step("Step 7: Corner Snapped Map", vis_snapped)


    KNOWN_WALL_LENGTH_M = 2.112


    sorted_lines = sorted(final_lines_list, key=lambda l: (l[1] + l[3]) / 2)
    ref_line = sorted_lines[0]  # The top-most line


    x1_ref, y1_ref, x2_ref, y2_ref = ref_line
    ref_px_dist = math.sqrt((x2_ref - x1_ref) ** 2 + (y2_ref - y1_ref) ** 2)

    if ref_px_dist == 0:
        print("Error: Reference line has 0 length!")
        scale_factor = 1.0
    else:
        scale_factor = KNOWN_WALL_LENGTH_M / ref_px_dist

    print(f"\n--- CALIBRATION ---")
    print(f"Reference Wall (Top): {ref_px_dist:.1f} pixels")
    print(f"Known Length: {KNOWN_WALL_LENGTH_M} meters")
    print(f"Calculated Scale: {scale_factor:.6f} meters/pixel")

    selected_origin = plot_with_interactive_origin(vis_snapped, final_lines_list, scale_factor)

    scaled_map, min_x_val, max_y_val = generate_occupancy_grid(
        final_lines_list,
        scale_factor,
        selected_origin,  # Pass the origin here!
        resolution_m=MAP_RESOLUTION_M,
        flip_x=True,
        flip_y=True
    )
    show_step(f"Scaled Occupancy Map ({MAP_RESOLUTION_M * 100}cm/px)", scaled_map)
    cv2.imwrite(map_filename, scaled_map)

    save_map_metadata(
        map_filename,
        scaled_map.shape,
        MAP_RESOLUTION_M,
        min_x_val,
        max_y_val
    )

if __name__ == "__main__":
    main()