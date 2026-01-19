import numpy as np
import cv2

# --- CONFIGURATION ---
# Actual ball diameter in meters (e.g., Volleyball is ~0.21m)
BALL_REAL_DIAMETER = 0.21 
NET_HEIGHT = 2.43
# Define your real-world coordinates for your labels (X, Y, Z)
# Z = 0 is the ground. Let's assume the net height is 2.43m.
world_points_map = {
    "BE_LEFT":       [0.0, 0.0, 0.0   ],
    "BA_LEFT":       [0.0, 6.0, 0.0   ],
    "M_LEFT":        [0.0, 9.0, 0.0   ],
    "TA_LEFT":       [0.0, 12.0, 0.0  ],
    "TE_LEFT":       [0.0, 18.0, 0.0  ],
    "NET_B_LEFT":    [0.0, 9.0, NET_HEIGHT-1],
    "NET_T_LEFT":    [0.0, 9.0, NET_HEIGHT], 
    "ANT_T_LEFT":    [0.0, 9.0, NET_HEIGHT+0.8],
    "BE_RIGHT":      [9.0, 0.0, 0.0   ],
    "BA_RIGHT":      [9.0, 6.0, 0.0   ],
    "M_RIGHT":       [9.0, 9.0, 0.0   ],
    "TA_RIGHT":      [9.0, 12.0, 0.0  ],
    "TE_RIGHT":      [9.0, 18.0, 0.0  ],
    "NET_B_RIGHT": [9.0, 9.0, NET_HEIGHT-1],
    "NET_T_RIGHT":   [9.0, 9.0, NET_HEIGHT],
    "ANT_T_RIGHT": [9.0, 9.0, NET_HEIGHT+0.8],
}
# M_RIGHT,942.481,359.948
# M_LEFT,410.769,359.948
# TE_LEFT,504.773,204.253
# TE_RIGHT,851.414,207.191
# NET_TOP_LEFT,391.674,226.285
# NET_TOP_RIGHT,958.638,227.754
# Data from your CSV (name, img_x, img_y)
collected_data = [
    ("M_RIGHT", 942,360),
    ("M_LEFT", 411,360),
    ("TE_LEFT", 505,204),
    ("TE_RIGHT", 851.414,210),
    ("NET_T_LEFT", 392,226),
    ("NET_T_RIGHT",960.638,226)
]

collected_data = [
    ("BA_RIGHT",1040.129,532.658),
    ("BA_LEFT",229.935,469.545),
    ("M_LEFT",414.871,451.932),
    ("M_RIGHT",1125.258,488.626),
    ("NET_T_RIGHT",1110.581,236.174),
    ("NET_T_LEFT",410.468,265.529),
    ("TA_RIGHT",1175.161,460.739)

]

import numpy as np
import cv2

def calibrate_camera(data, world_map, img_w=1920, img_h=1080):
    image_pts = []
    object_pts = []
    
    # Only use points that exist in BOTH your CSV data and your world_map
    for name, ix, iy in data:
        if name in world_map:
            image_pts.append([ix, iy])
            object_pts.append(world_map[name])
        else:
            print(f"Warning: Point '{name}' in data but not defined in world_map. Skipping.")
            
    if len(image_pts) < 4:
        raise ValueError(f"Need at least 4 points to calibrate. Found {len(image_pts)}.")

    image_pts = np.array(image_pts, dtype=np.float32)
    object_pts = np.array(object_pts, dtype=np.float32)

    # Intrinsics: Use a focal length guess. 
    # For many wide-angle cameras, focal_length ≈ image_width
    focal_length = img_w-50
    center = (img_w / 2, img_h / 2)
    
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) 

    # solvePnP finds the camera position/rotation relative to the court
    success, rvec, tvec = cv2.solvePnP(
        object_pts, 
        image_pts, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("Calibration failed!")
        return None, None, None

    return camera_matrix, rvec, tvec

#changed to calculate net position
def get_ball_height(ball_x, ball_y, ball_w_px, camera_matrix, rvec, tvec):
    """
    Calculates the 3D position and height of the ball.
    Uses the pixel width to determine distance (Z depth from camera).
    """
    f_x = camera_matrix[0, 0]
    
    # 1. Calculate distance from camera using the apparent width
    # distance = (focal_length * real_width) / width_in_pixels
    distance_from_cam = (f_x * BALL_REAL_DIAMETER) / ball_w_px
    
    # 2. Get Normalized Image Coordinates
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x_norm = (ball_x - cx) / f_x
    y_norm = (ball_y - cy) / camera_matrix[1, 1]
    
    # 3. Ball position in Camera Space
    P_cam = np.array([x_norm * distance_from_cam, 
                      y_norm * distance_from_cam, 
                      distance_from_cam])
    
    # 4. Transform Camera Space to World Space
    R, _ = cv2.Rodrigues(rvec)
    # P_world = R_inv * (P_cam - T)
    P_world = np.dot(R.T, (P_cam - tvec.flatten()))
    
    return P_world # Return the Z-coordinate (Height)

def verify_calibration(camera_matrix, rvec, tvec, world_map):
    # Convert world points to a numpy array
    names = list(world_map.keys())
    pts_3d = np.array([world_map[n] for n in names], dtype=np.float32)
    
    # Project 3D points to 2D image plane
    img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, np.zeros((4,1)))
    
    for i, name in enumerate(names):
        print(f"Point {name}: World {world_map[name]} -> Image {img_pts[i].ravel()}")
    return img_pts

def get_net_homography(camera_matrix, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    
    # Normally for ground (Z=0), we take columns 0 and 1 of R.
    # For the Net (Y=9), the plane is defined by X and Z.
    # So we take column 0 (X) and column 2 (Z).
    
    # We also need to account for the fact that Y is not 0, it's 9.
    # The translation becomes: t_effective = (R * [0, 9, 0]^T) + tvec
    y_offset_world = np.array([0, 9.0, 0], dtype=np.float32).reshape(3, 1)
    t_effective = np.dot(R, y_offset_world) + tvec
    
    # Combine Column 0 (X), Column 2 (Z), and the effective translation
    rt_vertical = np.column_stack((R[:, 0], R[:, 2], t_effective))
    
    # H = K * [r1 r3 t_eff]
    h_net = np.dot(camera_matrix, rt_vertical)
    
    return np.linalg.inv(h_net)

def image_to_net_coords(u, v, h_net_inv):
    """Convert pixel (u, v) to world (X, Z) on the net plane."""
    pixel_pt = np.array([u, v, 1.0], dtype=np.float32)
    world_pt = np.dot(h_net_inv, pixel_pt)
    
    world_pt /= world_pt[2] # Normalize
    return world_pt[0], world_pt[1] # Returns X and Z (Height)


def get_ground_homography(camera_matrix, rvec, tvec):
    """
    Creates a Homography matrix that maps image pixels (u, v) 
    directly to ground coordinates (X, Y, 0).
    """
    R, _ = cv2.Rodrigues(rvec)
    
    # The transformation matrix from World to Camera is [R | t]
    # We only care about the mapping to the Z=0 plane
    # So we take columns 1, 2, and 4 of the projection matrix
    rt_extrinsic = np.column_stack((R[:, 0], R[:, 1], tvec))
    
    # H = K * [r1 r2 t]
    homography = np.dot(camera_matrix, rt_extrinsic)
    
    # We want the inverse to go from Image -> World
    h_inv = np.linalg.inv(homography)
    return h_inv

def image_to_world_ground(u, v, h_inv):
    """Convert pixel (u, v) to world (X, Y) assuming it's on the ground."""
    pixel_pt = np.array([u, v, 1.0], dtype=np.float32)
    world_pt = np.dot(h_inv, pixel_pt)
    
    # Normalize by the third coordinate (scale factor)
    world_pt /= world_pt[2]
    return world_pt[0], world_pt[1]
    
# --- EXECUTION ---
cam_mtx, rvec, tvec = calibrate_camera(collected_data, world_points_map, img_w=1365, img_h=767)

if cam_mtx is not None:
    # 1. Project the Net Top Left to see if it matches the image
    net_tl_world = np.array([world_points_map["NET_T_LEFT"]], dtype=np.float32)
    projected_pt, _ = cv2.projectPoints(net_tl_world, rvec, tvec, cam_mtx, np.zeros((4,1)))
    print(f"Actual Image Pt: [410.468, 265.529]")
    print(f"Projected 3D Net Pt: {projected_pt.ravel()}")
    print(f"Error is{[410.468, 265.529] - projected_pt.ravel()} ")

    # 2. Get the full 3D position of the "detected" net center
    # Note: net_w here acts as the 'real diameter' scale
    ball_x = (1110.581 + 410.468) / 2
    ball_y = (236.174 + 265.529) / 2
    ball_w_px = 19
    h_inv = get_ground_homography(cam_mtx,rvec,tvec)
    nh_inv = get_net_homography(cam_mtx,rvec,tvec)
    world_p = image_to_world_ground(1125.258,488.626,h_inv)
    world_p2 = image_to_net_coords(1125.258,488.626,nh_inv)
    # Using your function (ensure BALL_REAL_DIAMETER is set to net width, e.g., 9.0m)
    # We temporarily override the diameter constant for the net calculation
    P_world = get_ball_height(ball_x, ball_y, ball_w_px, cam_mtx, rvec, tvec)
    print(f"the Ground Point is at: {world_p}")
    print(f"the Net Point is at: {world_p2}")
    print(f"--- 3D World Position of Net Center ---")
    print(f"X: {P_world[0]:.2f}m, Y: {P_world[1]:.2f}m, Z (Height): {P_world[2]:.2f}m")


def draw_homography_overlay(image, cam_mtx, rvec, tvec):
    overlay = image.copy()
    
    # 1. FLOOR OVERLAY (Green)
    # Define the 4 corners of the court floor in World Space (X, Y, Z=0)
    floor_corners = np.array([
        [0, 0, 0], [9, 0, 0], [9, 18, 0], [0, 18, 0]
    ], dtype=np.float32)
    
    # Project floor to image
    floor_img_pts, _ = cv2.projectPoints(floor_corners, rvec, tvec, cam_mtx, np.zeros((4,1)))
    floor_img_pts = floor_img_pts.reshape(-1, 2).astype(np.int32)
    
    # Draw floor polygon
    cv2.fillPoly(overlay, [floor_img_pts], (0, 255, 0)) # Green floor

    # 2. NET OVERLAY (Blue)
    # Define the 4 corners of the net plane in World Space (X, Y=9, Z)
    net_corners = np.array([
        [0, 9, 0], [9, 9, 0], [9, 9, NET_HEIGHT], [0, 9, NET_HEIGHT]
    ], dtype=np.float32)
    
    # Project net to image
    net_img_pts, _ = cv2.projectPoints(net_corners, rvec, tvec, cam_mtx, np.zeros((4,1)))
    net_img_pts = net_img_pts.reshape(-1, 2).astype(np.int32)
    
    # Draw net polygon
    cv2.fillPoly(overlay, [net_img_pts], (255, 0, 0)) # Blue net

    # 3. Blend overlay with original image
    alpha = 0.3  # Transparency factor
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# --- IN YOUR EXECUTION BLOCK ---
# Load your image (replace with your actual image path)
img = cv2.imread("tests/image2.png") 
if img is not None:
    # Resize image to match the img_w/img_h used in calibration if necessary
    img = cv2.resize(img, (1365, 767))
    
    # Generate the visualization
    result_img = draw_homography_overlay(img, cam_mtx, rvec, tvec)
    
    # Add dots for your collected_data points for verification
    for name, ix, iy in collected_data:
        cv2.circle(result_img, (int(ix), int(iy)), 5, (0, 0, 255), -1)
        cv2.putText(result_img, name, (int(ix)+10, int(iy)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Calibration Overlay", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()