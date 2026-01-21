
import numpy as np
import cv2
from src.calibration.geometry import estimate_3d_position, compute_camera_matrix, VOLLEYBALL_DIAMETER, WORLD_POINTS

def test_geometry_logic():
    print("--- Testing Geometry Logic ---")
    
    # 1. Create a Synthetic Camera
    # Let's verify if we can recover a known 3D point.
    width, height = 1920, 1080
    f = 1000
    K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]], dtype=np.float32)
    
    # Camera at (0, -20, 10) looking at (0, 0, 0)
    # T = [0, -20, 10]
    # R: Look down 30 degrees? Simple: Look along +Y? No, look at origin.
    # Let's say camera is high up at Z=10, Y=-20 (Way back).
    # Up vector (0, 0, 1) -> Camera Y down?
    # Simple LookAt matrix.
    
    # Define a test point in 3D (A ball at center of net, top)
    # P_ball = [0, 0, 2.43 + 0.5] -> 2.93m high
    P_ball = np.array([0.0, 0.0, 3.0]) 
    
    # Project to 2D
    # We need a valid Projection Matrix P to test `estimate_3d_position` because it needs calibration data
    # Let's use the implementation's own compute function if possible? 
    # No, we need to mock the stored calibration data structure.
    
    # Let's simulate P = K [I | 0] for simplicity (Camera at origin looking down Z)
    # Camera at (0,0,0) looking at Z+ 
    # Obj at (0, 0, 10).
    R_ident = np.eye(3)
    t_zero = np.zeros((3, 1))
    
    # Construct calibration dict
    calib = {
        'K': K,
        'R': R_ident,
        't': t_zero
    }
    
    # Ball at (1, 1, 10) meters relative to camera
    P_cam = np.array([1.0, 1.0, 10.0])
    
    # Projection: u = fx * X/Z + cx
    u = f * (P_cam[0]/P_cam[2]) + width/2
    v = f * (P_cam[1]/P_cam[2]) + height/2
    
    # Ball Size Projection
    # Diameter D = 0.21
    # Size in pixels s = (f * D) / Z
    s = (f * VOLLEYBALL_DIAMETER) / P_cam[2]
    w_pix = s
    h_pix = s
    
    print(f"True Position: {P_cam}")
    print(f"Projected Detection: u={u}, v={v}, w={w_pix}, h={h_pix}")
    
    # 2. Reconstruct
    bbox = [u, v, w_pix, h_pix]
    P_est = estimate_3d_position(bbox, calib)
    
    print(f"Estimated Position: {P_est}")
    
    error = np.linalg.norm(P_cam - P_est)
    print(f"Error: {error:.4f} meters")
    
    assert error < 0.1, "Error too large!"
    print("Test PASSED")

if __name__ == "__main__":
    test_geometry_logic()
