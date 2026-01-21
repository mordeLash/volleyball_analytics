import pandas as pd
import numpy as np
from src.calibration.geometry import estimate_3d_position

def calibrate_to_relative_space(input_csv, output_csv, calibration_data=None):
    """
    this function should calculate and extract the direction that means the ball is going along the court
    and the direction that means the ball is going up
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        return
        
    # Apply Calibration if available
    if calibration_data:
        print("Applying 3D Calibration...")
        
        cols = ['x_world', 'y_world', 'z_world']
        world_coords = []
        
        for _, row in df.iterrows():
            bbox = (row['cx'], row['cy'], row['w'], row['h'])
            pt3d = estimate_3d_position(bbox, calibration_data) # Returns [x, y, z]
            world_coords.append(pt3d)
            
        world_arr = np.array(world_coords)
        df['world_x'] = world_arr[:, 0]
        df['world_y'] = world_arr[:, 1]
        df['world_z'] = world_arr[:, 2]
    else:
        # Fill with NaN if no calibration
        df['world_x'] = np.nan
        df['world_y'] = np.nan
        df['world_z'] = np.nan

    # 3. Final Output (Strictly cx and cy as requested)
    base_cols = ['frame', 'track_id', 'cx', 'cy','w','h']
    extra_cols = ['world_x', 'world_y', 'world_z']
    
    # Check for track_id in case it was dropped in a previous step
    available_cols = [c for c in base_cols if c in df.columns]
    
    # Combine
    final_cols = available_cols + extra_cols
    
    df_out = df[final_cols].sort_values('frame')
    df_out.to_csv(output_csv, index=False)
    