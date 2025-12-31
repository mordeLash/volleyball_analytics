import pandas as pd
import numpy as np

def clean_noise(tracking_csv, output_csv):
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
    
    # Improved Quality Score: Rewards movement and duration
    stats['quality_score'] = stats['duration'] * stats['m_conf'] * (stats['displacement'] + 1)

    # 2. Hard Noise Filtering (Refined thresholds)
    # Using 'query' for readability
    is_noise = (
        (stats['displacement'] < 20) | # Static objects
        (stats['duration'] < 5) |      # Flickers
        ((stats['duration'] < 15) & (stats['max_conf'] < 0.6)) |
        (stats['m_conf'] < 0.35)
    )
    valid_ids = stats[~is_noise].index
    df = df[df['track_id'].isin(valid_ids)].copy()

    # 3. Resolve Overlaps (The "Winner-Takes-All" Frame Strategy)
    # This replaces the nested 'interrupter' loop. If two tracks exist in the 
    # same frame, the one with the higher global quality score wins.
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
    
    print(f"Cleanup complete. Tracks reduced from {stats.index.nunique()} to {df_cleaned['track_id'].nunique()}")
    df_cleaned.to_csv(output_csv, index=False)