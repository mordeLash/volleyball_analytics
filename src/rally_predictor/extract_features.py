# src/rally_predictor/extract_features.py

import pandas as pd
import numpy as np

def interpolate_small_gaps(series, max_gap):
    """
    Interpolates missing values (NaNs) only if the gap size is within a specific limit.

    In sports tracking, short disappearances (e.g., ball passing behind a player's head) 
    are safe to interpolate linearly. Long gaps are left as NaNs to prevent the model 
    from assuming a straight-line path across the entire court.

    Args:
        series (pd.Series): The data series containing NaNs (e.g., cx or cy coordinates).
        max_gap (int): The maximum number of consecutive NaNs to fill.

    Returns:
        pd.Series: The series with small gaps filled and large gaps preserved.
    """
    # Find groups of NaNs
    isnull = series.isnull()
    
    # Create a unique ID for each contiguous block of NaNs by checking where the 'null-ness' changes
    gap_ids = (isnull != isnull.shift()).cumsum()
    
    # Map each gap ID to the total count of NaNs in that specific group
    gap_sizes = gap_ids.map(gap_ids[isnull].value_counts())
    
    # Perform linear interpolation, then revert values if the original gap was too large
    return series.interpolate(method='linear').where(gap_sizes <= max_gap, series)

def extract_features(input_csv, output_csv, window_size=60, start_frame=0, interpolation_size=5):
    """
    Transforms raw tracking coordinates into physics-based features for rally classification.

    The function creates a fixed-frequency timeline, fills small tracking gaps, 
    calculates instantaneous kinematics (velocity, acceleration, angles), and 
    generates rolling window statistics to provide temporal context.

    Args:
        input_csv (str): Path to the cleaned tracking CSV.
        output_csv (str): Path to save the engineered features CSV.
        window_size (int): Size of the rolling window in frames (60 frames = ~2 sec).
        start_frame (int): The frame index to start the timeline from.
        interpolation_size (int): Max consecutive frames to fill via linear interpolation.

    Returns:
        None: Saves the feature-rich DataFrame to output_csv.
    """
    df = pd.read_csv(input_csv)
    
    # 1. TIMELINE EXPANSION
    # Ensure every frame exists in the DataFrame, even if no detection occurred.
    # This is critical for accurate velocity (pixels per frame) math.
    df = df.sort_values('frame')
    full_range = pd.DataFrame({'frame': range(start_frame, int(df['frame'].max()) + 1)})
    df = pd.merge(full_range, df, on='frame', how='left')
    
    # Fill tiny gaps (flickers) while keeping long disappearances empty
    df['cx'] = interpolate_small_gaps(df['cx'], max_gap=interpolation_size)
    df['cy'] = interpolate_small_gaps(df['cy'], max_gap=interpolation_size)
    df['conf'] = df['conf'].fillna(0)
    
    # 2. BASIC PHYSICS (Vectorized)
    # Velocity is the first derivative of position
    df['vx'] = df['cx'].diff().fillna(0)
    df['vy'] = df['cy'].diff().fillna(0)
    df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Noise Reduction: Clip unrealistic jumps caused by ID-switching or tracking errors
    df['velocity'] = df['velocity'].clip(upper=300)
    
    # Acceleration is the first derivative of velocity
    df['acceleration'] = df['velocity'].diff().fillna(0)
    
    # Angular changes: Used to detect sharp changes in direction (sets, spikes, floor bounces)
    df['angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    df['angle_diff'] = df['angle'].diff().abs() % 360
    # Ensure we get the shortest angular distance (e.g., 350 deg to 10 deg is 20, not 340)
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff']).fillna(0)

    # 3. RALLY-SPECIFIC ENGINEERED FEATURES
    # y_dominance: Rallies involve high arcs (vertical). Downtime involves horizontal walking.
    df['y_dominance'] = np.abs(df['vy']) / (np.abs(df['vx']) + 1e-5)
    
    # accel_jitter: Detects "shaky" or erratic movement typical of manual handling or noise.
    df['accel_jitter'] = df['acceleration'].diff().abs().fillna(0)
    
    # energy: Squares velocity to emphasize high-speed spikes/serves.
    df['energy'] = df['velocity']**2
    
    # is_low: Heuristic to detect if the ball is on the floor (usually downtime).
    if not df['cy'].dropna().empty:
        floor_threshold = df['cy'].quantile(0.7)
        df['is_low'] = (df['cy'] > floor_threshold).astype(int)
    else:
        df['is_low'] = 0

    # 4. ROLLING WINDOW FEATURES (The 'Context')
    # Using center=True allows the model to "see" a bit of the past and future 
    # when determining if the current frame is part of a rally.
    rolling = df.rolling(window=window_size, center=True, min_periods=1)
    df_features = pd.DataFrame()

    df_features['mean_x'] = rolling['cx'].mean()
    df_features['mean_y'] = rolling['cy'].mean()
    df_features['std_x'] = rolling['cx'].std().fillna(0)
    df_features['std_y'] = rolling['cy'].std().fillna(0)
    df_features['mean_velocity'] = rolling['velocity'].mean()
    df_features['max_velocity'] = rolling['velocity'].max()
    df_features['std_velocity'] = rolling['velocity'].std().fillna(0)
    df_features['mean_acceleration'] = rolling['acceleration'].mean()
    df_features['mean_angle_change'] = rolling['angle_diff'].mean()
    df_features['mean_y_dominance'] = rolling['y_dominance'].mean()
    df_features['mean_accel_jitter'] = rolling['accel_jitter'].mean()
    df_features['mean_energy'] = rolling['energy'].mean()
    
    # y_range_window: Measures the vertical "spread." Rallies have huge ranges;
    # someone holding a ball has a very narrow vertical range.
    df_features['y_range_window'] = rolling['cy'].max() - rolling['cy'].min()
    
    # Visibility/Confidence consistency: Rallies often have lower/varied confidence
    # due to motion blur, unlike static downtime.
    df_features['mean_vis'] = rolling['conf'].mean()
    df_features['vis_std'] = rolling['conf'].std().fillna(0)
    df_features['is_low_window'] = rolling['is_low'].mean()
    
    df_features.to_csv(output_csv, index=False)


def extract_features_v2(input_csv,output_csv, window_size=45, start_frame=0, interpolation_size=5):
    """
    Transforms raw tracking coordinates into physics-based features.
    Updated to provide TWO windows: 
    - 'prev_': Statistics of frames BEFORE the current frame.
    - 'next_': Statistics of frames AFTER the current frame.
    """
    df = pd.read_csv(input_csv)
    
    # 1. TIMELINE EXPANSION (Same as before)
    df = df.sort_values('frame')
    full_range = pd.DataFrame({'frame': range(start_frame, int(df['frame'].max()) + 1)})
    df = pd.merge(full_range, df, on='frame', how='left')
    
    # Fill tiny gaps
    df['cx'] = df['cx'].interpolate(limit=interpolation_size) # Placeholder for your interpolate function
    df['cy'] = df['cy'].interpolate(limit=interpolation_size)
    
    # 2. BASIC PHYSICS (Vectorized)
    df['vx'] = df['cx'].diff().fillna(0)
    df['vy'] = df['cy'].diff().fillna(0)
    df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2).clip(upper=300)
    df['acceleration'] = df['velocity'].diff().fillna(0)
    df['energy'] = df['velocity']**2

    # Extra
    
    
    # 3. DEFINE ROLLING OBJECTS
    # 'Past' window: looks at window_size frames ending at current frame
    # We use closed='left' to ensure the CURRENT frame isn't included in the past stats
    past_rolling = df.rolling(window=window_size, min_periods=1, closed='left')
    
    # 'Future' window: We shift the dataframe backwards and then roll
    # This effectively looks at window_size frames starting AFTER the current frame
    future_rolling = df.shift(-window_size).rolling(window=window_size, min_periods=1)

    # 4. APPLY ROLLING STATS
    # List of features to calculate for both windows
    features_to_roll = ['cy','cx', 'velocity', 'acceleration', 'energy']
    df_features = pd.DataFrame()
    for feat in features_to_roll:
        # Past features
        df_features[f'prev_mean_{feat}'] = past_rolling[feat].mean()
        df_features[f'prev_max_{feat}'] = past_rolling[feat].max()
        df_features[f'prev_std_{feat}'] = past_rolling[feat].std().fillna(0)
        
        # Future features
        df_features[f'next_mean_{feat}'] = future_rolling[feat].mean()
        df_features[f'next_max_{feat}'] = future_rolling[feat].max()
        df_features[f'next_std_{feat}'] = future_rolling[feat].std().fillna(0)


    

    # Clean up NaNs created by windows at the very start/end of the video
    df_features = df_features.fillna(0)
    
    df_features.to_csv(output_csv, index=False)
    

def extract_features_v3(input_csv, output_csv, window_size=45, start_frame=0, interpolation_size=5):
    """
    Transforms raw tracking coordinates into physics-based features, 
    normalized by ball dimensions (w, h).
    """
    df = pd.read_csv(input_csv)
    
    # 1. TIMELINE EXPANSION
    df = df.sort_values('frame')
    full_range = pd.DataFrame({'frame': range(start_frame, int(df['frame'].max()) + 1)})
    df = pd.merge(full_range, df, on='frame', how='left')
    
    # Fill gaps for coordinates and dimensions
    for col in ['cx', 'cy', 'w', 'h']:
        df[col] = df[col].interpolate(limit=interpolation_size)

    # 2. BASIC PHYSICS (Pixel-based)
    df['vx'] = df['cx'].diff().fillna(0)
    df['vy'] = df['cy'].diff().fillna(0)
    
    # 3. DIMENSION-BASED CALCULATIONS
    # We use the average of width and height as a "unit" of scale
    df['ball_scale'] = (df['w'] + df['h']) / 2
    df['ball_scale'].rolling(window=3).mean()
    
    # Avoid division by zero if the ball is lost for a long period
    df['ball_scale'] = df['ball_scale'].replace(0, np.nan).ffill().fillna(1)

    df['vz_norm'] = df['ball_scale'].diff().fillna(0) / df['ball_scale']

    # Normalized Velocity: How many "ball sizes" did it move?
    v_xy_norm = np.sqrt(df['vx']**2 + df['vy']**2) / df['ball_scale']
    df['v_norm'] = np.sqrt(v_xy_norm**2 + df['vz_norm']**2)
    
    # Normalized Acceleration
    df['a_norm'] = df['v_norm'].diff().fillna(0)
    
    # Normalized Energy (Kinetic energy is proportional to v^2)
    df['e_norm'] = df['v_norm']**2
    


    # 4. DEFINE ROLLING OBJECTS
    # 'Past' window: looks at window_size frames before current
    past_rolling = df.rolling(window=window_size, min_periods=1, closed='left')
    
    # 'Future' window: looks at window_size frames after current
    future_rolling = df.shift(-window_size).rolling(window=window_size, min_periods=1)

    rolling = df.rolling(window=window_size, center=True, min_periods=1)

    # 5. APPLY ROLLING STATS
    # We prioritize normalized features for the ML model
    features_to_roll = ['v_norm','a_norm','e_norm','cy','cx']
    df_features = pd.DataFrame() 
    
    for feat in features_to_roll:
        # Past features
        df_features[f'prev_mean_{feat}'] = past_rolling[feat].mean()
        df_features[f'prev_max_{feat}'] = past_rolling[feat].max()
        df_features[f'prev_std_{feat}'] = past_rolling[feat].std().fillna(0)
        
        # Future features
        df_features[f'next_mean_{feat}'] = future_rolling[feat].mean()
        df_features[f'next_max_{feat}'] = future_rolling[feat].max()
        df_features[f'next_std_{feat}'] = future_rolling[feat].std().fillna(0)

        # now features
        df_features[f'mean_{feat}'] = rolling[feat].mean()
        df_features[f'max_{feat}'] = rolling[feat].max()
        df_features[f'std_{feat}'] = rolling[feat].std().fillna(0)   

    # Clean up
    df_features = df_features.fillna(0)
    df_features.to_csv(output_csv, index=False)
    return df_features


def extract_features_v4(input_csv, output_csv, window_size=45, start_frame=0, interpolation_size=5):
    """
    Transforms raw tracking coordinates into physics-based features, 
    normalized by ball dimensions (w, h).
    """
    df = pd.read_csv(input_csv)
    
    # 1. TIMELINE EXPANSION
    df = df.sort_values('frame')
    full_range = pd.DataFrame({'frame': range(start_frame, int(df['frame'].max()) + 1)})
    df = pd.merge(full_range, df, on='frame', how='left')
    
    # Fill gaps for coordinates and dimensions
    for col in ['world_x', 'world_y', 'world_z',]:
        df[col] = df[col].interpolate(limit=interpolation_size)

    
    df['world_vx'] = df['world_x'].diff().fillna(0)
    df['world_vy'] = df['world_y'].diff().fillna(0)
    df['world_vz'] = df['world_z'].diff().fillna(0)
    df['world_ex'] = df['world_vx']**2
    df['world_ey'] = df['world_vy']**2
    df['world_ez'] = df['world_vz']**2
    


    # 4. DEFINE ROLLING OBJECTS
    # 'Past' window: looks at window_size frames before current
    past_rolling = df.rolling(window=window_size, min_periods=1, closed='left')
    
    # 'Future' window: looks at window_size frames after current
    future_rolling = df.shift(-window_size).rolling(window=window_size, min_periods=1)

    rolling = df.rolling(window=window_size, center=True, min_periods=1)

    # 5. APPLY ROLLING STATS
    # We prioritize normalized features for the ML model
    features_to_roll = ['world_z','world_y','world_x',
                        'world_vz','world_vy','world_vx',
                        'world_ez','world_ey','world_ex',
                        ]
    df_features = pd.DataFrame() 
    
    for feat in features_to_roll:
        # Past features
        df_features[f'prev_mean_{feat}'] = past_rolling[feat].mean()
        df_features[f'prev_max_{feat}'] = past_rolling[feat].max()
        df_features[f'prev_min_{feat}'] = past_rolling[feat].min()
        df_features[f'prev_std_{feat}'] = past_rolling[feat].std().fillna(0)
        
        # Future features
        df_features[f'next_mean_{feat}'] = future_rolling[feat].mean()
        df_features[f'next_max_{feat}'] = future_rolling[feat].max()
        df_features[f'next_min_{feat}'] = future_rolling[feat].min()
        df_features[f'next_std_{feat}'] = future_rolling[feat].std().fillna(0)

        # now features
        df_features[f'mean_{feat}'] = rolling[feat].mean()
        df_features[f'max_{feat}'] = rolling[feat].max()
        df_features[f'min_{feat}'] = rolling[feat].min()
        df_features[f'std_{feat}'] = rolling[feat].std().fillna(0)   

    # Clean up
    df_features = df_features.fillna(0)
    df_features.to_csv(output_csv, index=False)
    return df_features


def extract_features_v5(
    input_csv, 
    output_csv, 
    window_size=45, 
    start_frame=0, 
    interpolation_size=5
):
    """
    Improved feature extractor for predicting:
    - rally vs downtime in volleyball
    Uses physics + motion + rolling context.
    """

    df = pd.read_csv(input_csv)

    # ---------- 1) TIMELINE EXPANSION ----------
    df = df.sort_values('frame')
    full_range = pd.DataFrame({
        'frame': range(start_frame, int(df['frame'].max()) + 1)
    })
    df = pd.merge(full_range, df, on='frame', how='left')

    # Interpolate missing positions
    for col in ['world_x', 'world_y', 'world_z']:
        df[col] = df[col].interpolate(limit=interpolation_size)

    # ---------- 2) BASIC MOTION FEATURES ----------
    df['world_vx'] = df['world_x'].diff().fillna(0)
    df['world_vy'] = df['world_y'].diff().fillna(0)
    df['world_vz'] = df['world_z'].diff().fillna(0)

    # Speed & acceleration (VERY IMPORTANT)
    df['speed'] = np.sqrt(
        df['world_vx']**2 +
        df['world_vy']**2 +
        df['world_vz']**2
    )

    df['accel_x'] = df['world_vx'].diff().fillna(0)
    df['accel_y'] = df['world_vy'].diff().fillna(0)
    df['accel_z'] = df['world_vz'].diff().fillna(0)

    df['accel_mag'] = np.sqrt(
        df['accel_x']**2 +
        df['accel_y']**2 +
        df['accel_z']**2
    )



    # Proxy for kinetic energy
    df['kinetic_energy'] = df['speed']**2

    # ---------- 3) ROLLING WINDOWS ----------
    past_rolling = df.rolling(window=window_size, min_periods=1, closed='left')
    future_rolling = df.shift(-window_size).rolling(window=window_size, min_periods=1)
    center_rolling = df.rolling(window=window_size, center=True, min_periods=1)

    # ---------- 4) FEATURES TO ROLL ----------
    features_to_roll = [
        'world_x','world_y','world_z',
        'speed',
        'accel_mag',
        'kinetic_energy',
    ]

    df_features = pd.DataFrame()

    for feat in features_to_roll:

        # --- PAST ---
        df_features[f'prev_mean_{feat}'] = past_rolling[feat].mean()
        df_features[f'prev_max_{feat}'] = past_rolling[feat].max()
        df_features[f'prev_min_{feat}'] = past_rolling[feat].min()
        # df_features[f'prev_min_max_{feat}'] = past_rolling[feat].max()-past_rolling[feat].min()
        df_features[f'prev_std_{feat}'] = past_rolling[feat].std().fillna(0)

        # --- FUTURE ---
        df_features[f'next_mean_{feat}'] = future_rolling[feat].mean()
        df_features[f'next_max_{feat}'] = future_rolling[feat].max()
        # df_features[f'next_min_max_{feat}'] = future_rolling[feat].max()-future_rolling[feat].min()
        df_features[f'next_min_{feat}'] = future_rolling[feat].min()
        df_features[f'next_std_{feat}'] = future_rolling[feat].std().fillna(0)

        # --- CENTERED (NOW) ---
        df_features[f'mean_{feat}'] = center_rolling[feat].mean()
        df_features[f'max_{feat}'] = center_rolling[feat].max()
        df_features[f'min_{feat}'] = center_rolling[feat].min()
        # df_features[f'min_max_{feat}'] = center_rolling[feat].max()-center_rolling[feat].min()
        df_features[f'std_{feat}'] = center_rolling[feat].std().fillna(0)

    df_features['z_range_window'] = center_rolling['world_z'].max() - center_rolling['world_z'].min()
    df_features = df_features.fillna(0)
    df_features.to_csv(output_csv, index=False)
    return df_features

def extract_features_v6(
    input_csv, 
    output_csv, 
    win_short=45,   
    win_long=90,    
    start_frame=0, 
    interpolation_size=10
):
    """
    Improved feature extractor for predicting:
    - rally vs downtime in volleyball
    Uses dual-window (short/long) physics + motion rolling context.
    """

    df = pd.read_csv(input_csv)

    # ---------- 1) TIMELINE EXPANSION ----------
    df = df.sort_values('frame')
    full_range = pd.DataFrame({
        'frame': range(start_frame, int(df['frame'].max()) + 1)
    })
    df = pd.merge(full_range, df, on='frame', how='left')

    # Interpolate missing positions
    for col in ['world_x', 'world_y', 'world_z']:
        df[col] = df[col].interpolate(limit=interpolation_size)

    # ---------- 2) BASIC MOTION FEATURES ----------
    df['world_vx'] = df['world_x'].diff().fillna(0)
    df['world_vy'] = df['world_y'].diff().fillna(0)
    df['world_vz'] = df['world_z'].diff().fillna(0)

    # Speed & acceleration
    df['speed'] = np.sqrt(df['world_vx']**2 + df['world_vy']**2 + df['world_vz']**2)

    df['accel_x'] = df['world_vx'].diff().fillna(0)
    df['accel_y'] = df['world_vy'].diff().fillna(0)
    df['accel_z'] = df['world_vz'].diff().fillna(0)

    df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['kinetic_energy'] = df['speed']**2

    # ---------- 3) MULTI-WINDOW ROLLING ----------
    # Define rolling objects for both window sizes
    windows = {
        's': win_short,
        'l': win_long
    }
    
    df_features = pd.DataFrame()

    features_to_roll = [
        'world_x', 'world_y', 'world_z',
        'speed',
        'accel_mag',
        'kinetic_energy',
    ]

    for label, size in windows.items():
        # Define rolling groups for this specific window size
        past_rolling = df.rolling(window=size, min_periods=1, closed='left')
        future_rolling = df.shift(-size).rolling(window=size, min_periods=1)
        center_rolling = df.rolling(window=size, center=True, min_periods=1)

        for feat in features_to_roll:
            # --- PAST (History) ---
            df_features[f'{label}_prev_mean_{feat}'] = past_rolling[feat].mean()
            df_features[f'{label}_prev_max_{feat}'] = past_rolling[feat].max()
            df_features[f'{label}_prev_std_{feat}'] = past_rolling[feat].std().fillna(0)

            # --- FUTURE (Lookahead) ---
            df_features[f'{label}_next_mean_{feat}'] = future_rolling[feat].mean()
            df_features[f'{label}_next_max_{feat}'] = future_rolling[feat].max()
            df_features[f'{label}_next_std_{feat}'] = future_rolling[feat].std().fillna(0)

            # --- CENTERED (Current Context) ---
            df_features[f'{label}_mean_{feat}'] = center_rolling[feat].mean()
            df_features[f'{label}_max_{feat}'] = center_rolling[feat].max()
            df_features[f'{label}_std_{feat}'] = center_rolling[feat].std().fillna(0)

        # Specialized Feature: Z-Range (Important for identifying ball flight height)
        df_features[f'{label}_x_range'] = center_rolling['world_x'].max() - center_rolling['world_x'].min()
        df_features[f'{label}_y_range'] = center_rolling['world_y'].max() - center_rolling['world_y'].min()
        df_features[f'{label}_z_range'] = center_rolling['world_z'].max() - center_rolling['world_z'].min()

    # ---------- 4) CLEANUP & SAVE ----------
    df_features = df_features.fillna(0)
    df_features.to_csv(output_csv, index=False)
    
    return df_features

def extract_features_v7(
    input_csv, 
    output_csv, 
    win_short=45,   
    win_long=90,    
    start_frame=0, 
    interpolation_size=10
):
    """
    Advanced Feature Extractor V7:
    - Adds "Context-Aware" features (Net proximity, Court Bounds).
    - Adds "Physics-Consistency" features (Gravity deviation, Jerk, Vertical Dominance).
    """

    df = pd.read_csv(input_csv)

    # ---------- 1) TIMELINE EXPANSION ----------
    df = df.sort_values('frame')
    full_range = pd.DataFrame({
        'frame': range(start_frame, int(df['frame'].max()) + 1)
    })
    df = pd.merge(full_range, df, on='frame', how='left')

    # Interpolate missing positions
    for col in ['world_x', 'world_y', 'world_z']:
        df[col] = df[col].interpolate(limit=interpolation_size)

    # ---------- 2) BASIC KINEMATICS ----------
    for axis in ['x', 'y', 'z']:
        # Velocity
        df[f'world_v{axis}'] = df[f'world_{axis}'].diff().fillna(0)
        # Acceleration
        df[f'accel_{axis}'] = df[f'world_v{axis}'].diff().fillna(0)
        # Jerk (Change in acceleration - smoothness)
        df[f'jerk_{axis}'] = df[f'accel_{axis}'].diff().fillna(0)

    # Magnitudes
    df['speed'] = np.sqrt(df['world_vx']**2 + df['world_vy']**2 + df['world_vz']**2)
    df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['jerk_mag'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2 + df['jerk_z']**2)
    
    df['kinetic_energy'] = df['speed']**2

    # ---------- 3) CONTEXT & ADVANCED PHYSICS ----------
    
    # A. Court Context (Using new Origin: BL=(0,0), Center=(4.5, 9), Net Y=9, Net Z=2.43)
    df['dist_to_center'] = np.sqrt((df['world_x'] - 4.5)**2 + (df['world_y'] - 9.0)**2)
    df['dist_to_net_y'] = (df['world_y'] - 9.0).abs() # Distance to net plane
    df['height_above_net'] = df['world_z'] - 2.43
    
    # Boolean: Is inside logical court bounds? (Allowing slight buffer)
    # Court is 9x18. Origin at 0,0.
    df['is_in_court'] = ((df['world_x'] >= -1) & (df['world_x'] <= 10) & 
                         (df['world_y'] >= -1) & (df['world_y'] <= 19)).astype(float)

    # B. Trajectory Shape
    # Vertical Dominance: Ratio of Z-speed to total speed. 
    # High for sets/digs. Low for walking/rolling.
    df['vert_dominance'] = df['world_vz'].abs() / (df['speed'] + 1e-6)

    # Gravity Deviation (Heuristic)
    # In ideal freefall, accel_z is constant -g. 
    # Sudden changes in accel_z imply contact (hit/bounce). 
    # Low magnitude accel_z close to 0 implies holding/carrying.
    # We'll just track raw accel_z stability in rolling windows mostly.
    
    # ---------- 4) MULTI-WINDOW ROLLING ----------
    windows = {
        's': win_short,
        'l': win_long
    }
    
    # Collection dictionary to avoid DataFrame fragmentation
    features_dict = {}

    features_to_roll = [
        # Base
        'world_x', 'world_y', 'world_z',
        'speed', 'accel_mag', 'jerk_mag', 'kinetic_energy',
        
        # New Context
        'dist_to_center', 'dist_to_net_y', 'height_above_net',
        'vert_dominance', 
        
        # Components (useful for detailed variance)
        'world_vz', 'accel_z' 
    ]

    for label, size in windows.items():
        past_rolling = df.rolling(window=size, min_periods=1, closed='left')
        future_rolling = df.shift(-size).rolling(window=size, min_periods=1)
        center_rolling = df.rolling(window=size, center=True, min_periods=1)

        for feat in features_to_roll:
            # Stats including std (critical for distinguishing smooth flight vs chaotic noise)
            
            # MEAN
            features_dict[f'{label}_prev_mean_{feat}'] = past_rolling[feat].mean()
            features_dict[f'{label}_next_mean_{feat}'] = future_rolling[feat].mean()
            features_dict[f'{label}_mean_{feat}'] = center_rolling[feat].mean()
            
            # MAX
            features_dict[f'{label}_max_{feat}'] = center_rolling[feat].max()
            
            # STD (Stability)
            features_dict[f'{label}_prev_std_{feat}'] = past_rolling[feat].std().fillna(0)
            features_dict[f'{label}_next_std_{feat}'] = future_rolling[feat].std().fillna(0)
            features_dict[f'{label}_std_{feat}'] = center_rolling[feat].std().fillna(0)

        # Z-Range (Spread)
        features_dict[f'{label}_z_range'] = center_rolling['world_z'].max() - center_rolling['world_z'].min()
        
        # In-Court Ratio (How much of the window was in court?)
        features_dict[f'{label}_in_court_ratio'] = center_rolling['is_in_court'].mean()

    # ---------- 5) CLEANUP & SAVE ----------
    df_features = pd.DataFrame(features_dict)
    df_features = df_features.fillna(0)
    df_features.to_csv(output_csv, index=False)
    
    return df_features