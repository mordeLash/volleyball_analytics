# src/ball_detector/clean_tracking_data.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def clean_noise(tracking_csv, output_csv):
    """
    Filters noisy tracking data and resolves temporal overlaps to produce a clean single-object trajectory.

    The function performs a four-stage cleaning process:
    1. Statistics Calculation: Aggregates track duration, confidence, and Euclidean displacement.
    2. Hard Filtering: Removes tracks that are static (displacement < 20px), very short 
       (flickers), or consistently low confidence.
    3. Overlap Resolution: For frames with multiple valid detections, selects the track 
       with the highest 'quality_score' (Duration * Confidence * Displacement).
    4. Post-Processing: Removes fragmented tracks that fall below a minimum length 
       threshold after deduplication.

    Args:
        tracking_csv (str): Path to the input CSV file containing raw tracking data. 
            Expected columns: ['track_id', 'frame', 'conf', 'cx', 'cy'].
        output_csv (str): Path where the cleaned CSV will be saved.

    Returns:
        None: Saves the resulting DataFrame to output_csv.
    """
    df = pd.read_csv(tracking_csv)
    
    # 1. Calculate Statistics with Vectorized Physics
    stats = df.groupby('track_id').agg(
        duration=('frame', 'count'),
        start_f=('frame', 'min'),
        end_f=('frame', 'max'),
        m_conf=('conf', 'mean'),
        max_conf=('conf', 'max'),
        dist_x=('cx', lambda x: x.max() - x.min()),
        dist_y=('cy', lambda x: x.max() - x.min())
    )

    # Calculate Euclidean displacement
    stats['displacement'] = np.sqrt(stats['dist_x']**2 + stats['dist_y']**2)
    
    # Quality Score: Rewards movement and duration
    stats['quality_score'] = stats['duration'] * stats['m_conf'] * (stats['displacement'] + 1)

    # 2. Hard Noise Filtering
    is_noise = (
        (stats['displacement'] < 500) | # Static objects
        (stats['duration'] < 20) |      # Flickers
        ((stats['duration'] < 30) & (stats['max_conf'] < 0.6)) | # Short low-confidence tracks
        (stats['m_conf'] < 0.35) # Overall low-confidence tracks
    )
    valid_ids = stats[~is_noise].index 
    df = df[df['track_id'].isin(valid_ids)].copy() 

    # 3. Resolve Overlaps by Quality Score
    df = df.merge(stats[['quality_score']], on='track_id', how='left')
    df = df.sort_values(by=['frame', 'quality_score'], ascending=[True, False])
    
    # Keep only the best detection per frame
    df_cleaned = df.drop_duplicates(subset=['frame'], keep='first')

    # 4. Post-Process: Remove fragmented tracks left over after deduplication
    final_counts = df_cleaned['track_id'].value_counts()
    valid_final_tracks = final_counts[final_counts >= 5].index
    df_cleaned = df_cleaned[df_cleaned['track_id'].isin(valid_final_tracks)]

    # Final Sort
    df_cleaned = df_cleaned.sort_values('frame').drop(columns=['quality_score'])
    
    df_cleaned.to_csv(output_csv, index=False)


def clean_noise_v2(tracking_csv, output_csv):
    """
    Phase 1: Uses K-Means clustering to separate tracks into two groups: 
    Signal (Ball) and Noise (Flickers/Stationary).
    """
    df = pd.read_csv(tracking_csv)
    if df.empty: return

    # 1. Feature Extraction per Track
    stats = df.groupby('track_id').agg(
        duration=('frame', 'count'),
        m_conf=('conf', 'mean'),
        min_y=('cy', 'min'), max_y=('cy', 'max'),
        min_x=('cx', 'min'), max_x=('cx', 'max'),
        avg_speed=('speed_px_frame', 'mean')
    )
    
    stats['displacement'] = np.sqrt((stats['max_x'] - stats['min_x'])**2 + 
                                    (stats['max_y'] - stats['min_y'])**2)

    # 2. Prepare Data for Clustering
    # We cluster based on Duration, Displacement, and Speed
    features = ['duration', 'displacement', 'avg_speed']
    X = stats[features].values
    
    # Scale features so 'duration' (frames) doesn't outweigh 'displacement' (pixels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. K-Means Clustering (k=2: One for Ball, One for Noise)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=28)
    stats['cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Identify which cluster is the "Ball"
    # The ball cluster will have the HIGHER average displacement/duration
    cluster_0_mean = stats[stats['cluster'] == 0]['displacement'].mean()
    cluster_1_mean = stats[stats['cluster'] == 1]['displacement'].mean()
    
    ball_cluster_id = 0 if cluster_0_mean > cluster_1_mean else 1
    
    valid_ids = stats[stats['cluster'] == ball_cluster_id].index
    
    # 5. Filter and Deduplicate
    df_filtered = df[df['track_id'].isin(valid_ids)].copy()
    
    # In case multiple 'ball' tracks overlap in one frame, keep the highest confidence
    df_filtered = df_filtered.sort_values(by=['frame', 'conf'], ascending=[True, False])
    df_filtered = df_filtered.drop_duplicates(subset=['frame'], keep='first')

    df_filtered.to_csv(output_csv, index=False)
    
    n_noise = len(stats) - len(valid_ids)
    print(f"Clustering Complete: Found {len(valid_ids)} 'Ball' tracks and {n_noise} 'Noise' tracks.")





def clean_noise_v3(tracking_csv, output_csv):
    df = pd.read_csv(tracking_csv)
    if len(df) < 10: # Minimum threshold to even consider processing
        df.to_csv(output_csv, index=False)
        return

    # 1. Advanced Feature Extraction
    # Calculate path length per track_id (sum of Euclidean distances)
    df['dx'] = df.groupby('track_id')['cx'].diff().fillna(0)
    df['dy'] = df.groupby('track_id')['cy'].diff().fillna(0)
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)

    stats = df.groupby('track_id').agg(
        duration=('frame', 'count'),
        path_length=('dist', 'sum'),
        avg_conf=('conf', 'mean'),
        avg_speed=('speed_px_frame', 'mean'),
        max_speed=('speed_px_frame', 'max')
    ).reset_index()

    if len(stats) < 2:
        df.to_csv(output_csv, index=False)
        return

    # 2. Scaling and Clustering
    features = ['duration', 'path_length', 'avg_speed', 'avg_conf']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stats[features])

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=28)
    stats['cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Intelligent Cluster Selection
    # We rank clusters by a "Ball Score" (Higher is better)
    cluster_metrics = stats.groupby('cluster')[features].mean()
    # Normalize metrics within the two clusters to decide which is the ball
    ball_score = (cluster_metrics['path_length'] * 0.5 + 
                  cluster_metrics['duration'] * 0.4 + 
                  cluster_metrics['avg_conf'] * 0.1)
    
    ball_cluster_id = ball_score.idxmax()
    
    # 4. Filter
    valid_ids = stats[stats['cluster'] == ball_cluster_id]['track_id']
    df_filtered = df[df['track_id'].isin(valid_ids)].copy()

    # Deduplicate: Keep highest confidence per frame
    df_filtered = df_filtered.sort_values(['frame', 'conf'], ascending=[True, False])
    df_filtered = df_filtered.drop_duplicates(subset=['frame'], keep='first')

    df_filtered.drop(columns=['dx', 'dy', 'dist'], inplace=True)
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Kept {len(valid_ids)} tracks. Removed {len(stats) - len(valid_ids)} noise tracks.")