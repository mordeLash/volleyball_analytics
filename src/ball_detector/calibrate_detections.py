import pandas as pd
import numpy as np

def calibrate_to_relative_space(input_csv, output_csv):
    """
    this function should calculate and extract the direction that means the ball is going along the court
    and the direction that means the ball is going up
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        return
    # 3. Final Output (Strictly cx and cy as requested)
    cols_to_keep = ['frame', 'track_id', 'cx', 'cy','w','h']
    
    # Check for track_id in case it was dropped in a previous step
    available_cols = [c for c in cols_to_keep if c in df.columns]
    
    df_out = df[available_cols].sort_values('frame')
    df_out.to_csv(output_csv, index=False)
    