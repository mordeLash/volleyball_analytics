
import numpy as np
import cv2

# Standard Volleyball Dimensions (Meters)
# Origin (0,0,0) at center of court on floor.
# X: along center line (Siderline to Sideline is 9m) -> -4.5 to 4.5
# Y: along depth (Endline to Endline is 18m) -> -9 to 9
# Z: Up

# We define the order of points matching the user prompt:
# 10 court points: 1 point for each 2 line connection.
# The user prompt describes: "the court will be a 10 point segment placmentment, 1 point for each 2 line connection on the court"
# This likely refers to the corners and T-junctions.
# A standard court has:
# 4 corners (FAR-LEFT, FAR-RIGHT, NEAR-LEFT, NEAR-RIGHT)
# 2 center line ends (LEFT-CENTER, RIGHT-CENTER)
# 4 attack line ends (FAR-ATTACK-L, FAR-ATTACK-R, NEAR-ATTACK-L, NEAR-ATTACK-R)
# Total 10 points. 

# Let's verify the order. The prompt doesn't specify order, so we define it.
# We will define a logical order for the user to follow:
# 1. Far Left Corner
# 2. Far Right Corner
# 3. Near Right Corner
# 4. Near Left Corner
# 5. Far Attack Line Left
# 6. Far Attack Line Right
# 7. Center Line Left (Net Post)
# 8. Center Line Right (Net Post)
# 9. Near Attack Line Right
# 10. Near Attack Line Left
#
# Net Points (6): "2 for the bottom 2 for the top and two for the antena"
# We assume standard net.
# 11. Net Top Left
# 12. Net Top Right
# 13. Net Bottom Left
# 14. Net Bottom Right
# 15. Antenna Top Left? Or just Antenna tips? "two for the antena". usually top.
# Let's assume Top of Antenna on Left and Right side. Or maybe specific markers.
# Let's stick to the 4 main net corners + 2 antenna tips used as extra reference.
# Actually, let's use:
# 11. Net Pole Bottom Left
# 12. Net Pole Bottom Right
# 13. Net Top Band Left
# 14. Net Top Band Right
# 15. Antenna Top Left
# 16. Antenna Top Right

COURT_WIDTH = 9.0
COURT_LENGTH = 18.0
ATTACK_LINE_DIST = 3.0 # From center line
NET_HEIGHT = 2.43 # Men's standard
ANTENNA_HEIGHT = 0.8 + NET_HEIGHT # Antenna is 1.8m long, extends 0.8m above net? usually 80cm above.

# Coordinates (X, Y, Z)
# Origin (0,0,0) at Corners[3] -> Near-Left Corner (matches "Bottom Left" in UI)
# X: Right (+), Y: Forward/Depth (+), Z: Up (+)

WORLD_POINTS = np.array([
    [0.0, 18.0, 0.0],  # 1. Far Left Corner (FL)
    [9.0, 18.0, 0.0],  # 2. Far Right Corner (FR) - Width 9m
    [9.0, 0.0, 0.0],   # 3. Near Right Corner (BR)
    [0.0, 0.0, 0.0],   # 4. Near Left Corner (BL) - ORIGIN
    
    [0.0, 12.0, 0.0],  # 5. Far Attack Line Left (Center(9) + 3)
    [9.0, 12.0, 0.0],  # 6. Far Attack Line Right
    
    [0.0, 9.0, 0.0],   # 7. Center Line Left (Net Post position on line)
    [9.0, 9.0, 0.0],   # 8. Center Line Right
    
    [9.0, 6.0, 0.0],   # 9. Near Attack Line Right (Center(9) - 3)
    [0.0, 6.0, 0.0],   # 10. Near Attack Line Left
    
    [0.0, 9.0, 2.43],  # 11. Net Top Left
    [9.0, 9.0, 2.43],  # 12. Net Top Right
    [0.0, 9.0, 1.43],  # 13. Net Bottom Left (Assuming 1m band width)
    [9.0, 9.0, 1.43],  # 14. Net Bottom Right

    [0.0, 9.0, 3.23],  # 15. Antenna Top Left (2.43 + 0.8)
    [9.0, 9.0, 3.23]   # 16. Antenna Top Right
], dtype=np.float32)

POINT_LABELS = [
    "Far-Left Corner", "Far-Right Corner", "Near-Right Corner", "Near-Left Corner (Origin)",
    "Far-Attack-Left", "Far-Attack-Right", "Center-Left (Floor)", "Center-Right (Floor)",
    "Near-Attack-Right", "Near-Attack-Left",
    "Net-Top-Left", "Net-Top-Right", "Net-Bottom-Left", "Net-Bottom-Right",
    "Antenna-Top-Left", "Antenna-Top-Right"
]

VOLLEYBALL_DIAMETER = 0.21 # Meters

def compute_camera_matrix(image_points_map, image_width=1920, image_height=1080):
    """
    Computes Projection Matrix P from 2D-3D correspondences.
    
    Args:
        image_points_map: Dictionary {point_id (int): (u, v)} or List of 16 (u,v)
        image_width, image_height: Dimensions.
    """
    # Handle list input (legacy/full)
    if isinstance(image_points_map, list):
         if len(image_points_map) != 16:
             raise ValueError(f"Expected 16 points, got {len(image_points_map)}")
         img_pts = np.array(image_points_map, dtype=np.float32)
         obj_pts = WORLD_POINTS
    elif isinstance(image_points_map, dict):
         # Filter WORLD_POINTS based on keys
         sorted_keys = sorted(image_points_map.keys())
         if len(sorted_keys) < 6:
             raise ValueError(f"Need at least 6 points for calibration, got {len(sorted_keys)}")
             
         img_pts = np.array([image_points_map[k] for k in sorted_keys], dtype=np.float32)
         # Map 1-based IDs (user facing?) to 0-based index? 
         # The POINT_LABELS are 0-indexed in the array, but user UI might use 1-16 or 0-15.
         # Let's assume the dictionary keys CORRESPOND to the index in WORLD_POINTS.
         obj_pts = WORLD_POINTS[sorted_keys]
    else:
        raise ValueError("Invalid input for image_points")

    # 1. Calculate P using DLT/CalibrateCamera
    try:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj_pts], [img_pts], (image_width, image_height), None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
        rvec = rvecs[0]
        tvec = tvecs[0]
    except:
        # Fallback to solvePnP if calibration fails
        K_guess = np.array([[image_width, 0, image_width/2], [0, image_width, image_height/2], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(5)
        ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_guess, dist)
        K = K_guess


    R, _ = cv2.Rodrigues(rvec)
    
    # Compute P = K [R|t]
    P = K @ np.hstack((R, tvec))
    
    return {
        "K": K,
        "dist": dist,
        "R": R,
        "t": tvec,
        "P": P
    }

def estimate_3d_position(bbox, calibration):
    """
    Estimates 3D position (X, Y, Z) of the ball.
    bbox: [cx, cy, w, h] (or similar)
    calibration: Dictionary with K, R, t
    """
    cx, cy, w, h = bbox
    K = calibration['K']
    R = calibration['R']
    t = np.array(calibration['t']).reshape(3, 1)
    
    # 1. Estimate depth (Z_cam) from size
    # apparent_size = (f * real_size) / Z_cam
    # Z_cam = (f * real_size) / apparent_size
    
    fx = K[0, 0]
    fy = K[1, 1]
    f_avg = (fx + fy) / 2.0
    
    apparent_size = max(w, h)
    if apparent_size == 0: return [0,0,0]

    Z_cam = (f_avg * VOLLEYBALL_DIAMETER) / apparent_size
    
    # 2. Back-project to 3D in Camera Frame
    # Pixel (u, v) -> Normalized (x_n, y_n)
    # x_n = (u - cx) / fx, y_n = (v - cy) / fy
    # X_cam = x_n * Z_cam, Y_cam = y_n * Z_cam
    
    u, v = cx, cy
    c_x = K[0, 2]
    c_y = K[1, 2]
    
    X_cam = ((u - c_x) / fx) * Z_cam
    Y_cam = ((v - c_y) / fy) * Z_cam
    
    P_cam = np.array([[X_cam], [Y_cam], [Z_cam]])
    
    # 3. Transform to World Frame
    # P_cam = R * P_world + t
    # P_world = R_inv * (P_cam - t)
    
    R_inv = np.linalg.inv(R)
    P_world = R_inv @ (P_cam - t)
    
    return P_world.flatten() # [X, Y, Z]
