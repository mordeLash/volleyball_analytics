import pandas as pd
import numpy as np

def interpolate_small_gaps(series, max_gap):
    # Find groups of NaNs
    isnull = series.isnull()
    # Create a unique ID for each contiguous block of NaNs
    gap_ids = (isnull != isnull.shift()).cumsum()
    # Count the size of each gap
    gap_sizes = gap_ids.map(gap_ids[isnull].value_counts())
    
    # Only interpolate where the gap size is <= our limit
    return series.interpolate(method='linear').where(gap_sizes <= max_gap, series)

def extract_features(input_csv, output_csv, window_size=60, start_frame=0,interpolation_size=5):
    """
    Extracts features specifically tuned to classify 'Rally' vs 'Downtime'.
    Window_size defaults to 60 (approx 2 seconds at 30fps) to capture state.
    """
    df = pd.read_csv(input_csv)
    
    # 1. Timeline Expansion & Cleaning
    df = df.sort_values('frame')
    full_range = pd.DataFrame({'frame': range(start_frame, int(df['frame'].max()) + 1)})
    df = pd.merge(full_range, df, on='frame', how='left')
    
    # Fill gaps: limit to 3-5 frames to avoid creating "fake" physics for long disappearances
    df['cx'] = interpolate_small_gaps(df['cx'], max_gap=interpolation_size)
    df['cy'] = interpolate_small_gaps(df['cy'], max_gap=interpolation_size)

    df['conf'] = df['conf'].fillna(0)
    
    # 2. Basic Physics (Vectorized)
    df['vx'] = df['cx'].diff().fillna(0)
    df['vy'] = df['cy'].diff().fillna(0)
    df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Clip outliers (ball moving > 300px/frame is usually a tracking jump)
    df['velocity'] = df['velocity'].clip(upper=300)
    df['acceleration'] = df['velocity'].diff().fillna(0)
    
    # Angular changes
    df['angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    df['angle_diff'] = df['angle'].diff().abs() % 360
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff']).fillna(0)

    # 3. Rally-Specific Engineered Features
    # Ratio of vertical to horizontal movement (High in rallies)
    df['y_dominance'] = np.abs(df['vy']) / (np.abs(df['vx']) + 1e-5)
    
    # Acceleration Jitter (Downtime often has erratic 'hand-held' movements)
    df['accel_jitter'] = df['acceleration'].diff().abs().fillna(0)
    
    # Kinetic Energy Proxy
    df['energy'] = df['velocity']**2
    
    # Is the ball in the bottom 20% of the screen? (Likely downtime)
    if not df['cy'].dropna().empty:
        floor_threshold = df['cy'].quantile(0.7)
        df['is_low'] = (df['cy'] > floor_threshold).astype(int)
    else:
        df['is_low'] = 0

    # 4. Rolling Window Features (The 'Context')
    rolling = df.rolling(window=window_size, center=True, min_periods=1)
    
    df['mean_x'] = rolling['cx'].mean()
    df['mean_y'] = rolling['cy'].mean()
    df['std_x'] = rolling['cx'].std().fillna(0)
    df['std_y'] = rolling['cy'].std().fillna(0)
    df['mean_velocity'] = rolling['velocity'].mean()
    df['max_velocity'] = rolling['velocity'].max()
    df['std_velocity'] = rolling['velocity'].std().fillna(0)
    df['mean_acceleration'] = rolling['acceleration'].mean()
    df['mean_angle_change'] = rolling['angle_diff'].mean()
    df['mean_y_dominance'] = rolling['y_dominance'].mean()
    df['mean_accel_jitter'] = rolling['accel_jitter'].mean()
    df['mean_energy'] = rolling['energy'].mean()
    # Vertical range is a huge indicator for rallies (high arcs)
    df['y_range_window'] = rolling['cy'].max() - rolling['cy'].min()
    
    # Visibility/Confidence consistency
    df['mean_vis'] = rolling['conf'].mean()
    df['vis_std'] = rolling['conf'].std().fillna(0)
    
    df['is_low_window'] = rolling['is_low'].mean()
    
    df.to_csv(output_csv, index=False)
    