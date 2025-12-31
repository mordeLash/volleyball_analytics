import pandas as pd
import numpy as np

def analyze_rally_stats(predictions, fps=30):
    # Convert to Series for easier manipulation
    s = pd.Series(predictions)
    
    # Identify groups of consecutive identical values
    # (e.g., [0,0,1,1,1,0] becomes [1,1,2,2,2,3])
    groups = (s != s.shift()).cumsum()
    
    # Create a DataFrame of segments
    segments = s.groupby(groups).agg(
        label='first',
        frame_count='count'
    )
    
    # Calculate duration in seconds
    segments['duration_sec'] = segments['frame_count'] / fps
    
    # Filter for Rallies and Downtime
    rallies = segments[segments['label'] == 1]['duration_sec']
    downtime = segments[segments['label'] == 0]['duration_sec']
    
    stats = {
        "Total Rallies": len(rallies),
        "Shortest Rally (s)": rallies.min() if not rallies.empty else 0,
        "Longest Rally (s)": rallies.max() if not rallies.empty else 0,
        "Average Rally (s)": rallies.mean() if not rallies.empty else 0,
        "Shortest Downtime (s)": downtime.min() if not downtime.empty else 0,
        "Total Playtime (s)": rallies.sum()
    }
    
    return stats, segments


def smooth_predictions(predictions, window_size=5, min_segment_len=60):
    if len(predictions) == 0:
        return predictions

    # 1. Apply rolling mode filter
    # center=True ensures the window is balanced around the current frame
    series = pd.Series(predictions)
    smoothed = series.rolling(window=window_size, center=True).apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[len(x)//2]
    ).fillna(series) # Fill NaNs at edges with original values
    
    smoothed_list = smoothed.tolist()

    # 2. Merge small segments
    # We loop until no more small segments are found to handle 
    # cases where merging creates a new small segment
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(smoothed_list):
            start = i
            val = smoothed_list[i]
            
            # Find the end of the current segment
            while i < len(smoothed_list) and smoothed_list[i] == val:
                i += 1
            segment_len = i - start
            
            # If segment is too short, merge it
            if segment_len < min_segment_len:
                # Determine the replacement value
                # Prefer the neighbor that exists; if both exist, 
                # you could pick the longer one or just the previous one.
                prev_val = smoothed_list[start - 1] if start > 0 else None
                next_val = smoothed_list[i] if i < len(smoothed_list) else None
                
                new_val = prev_val if prev_val is not None else next_val
                
                if new_val is not None and new_val != val:
                    smoothed_list[start:i] = [new_val] * segment_len
                    changed = True
                    
    return smoothed_list
