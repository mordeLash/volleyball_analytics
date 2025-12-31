import pandas as pd
import numpy as np

def track_with_physics_predictive(input_csv, output_csv, min_radius=50, max_radius=250, max_gap=15, velocity_margin=0.5):
    df = pd.read_csv(input_csv).sort_values('frame')
    df['cx'] = (df['x1'] + df['x2']) / 2
    df['cy'] = (df['y1'] + df['y2']) / 2
    
    df['track_id'] = -1
    df['vx'] = 0.0
    df['vy'] = 0.0
    df['speed_px_frame'] = 0.0
    df['direction_deg'] = 0.0
    
    next_id = 0
    active_tracks = {} 
    
    for f in sorted(df['frame'].unique()):
        frame_indices = df[df['frame'] == f].index.tolist()
        assigned_indices = []
        
        sorted_tids = sorted(active_tracks.keys(), key=lambda x: active_tracks[x]['last_f'], reverse=True)
        
        for tid in sorted_tids:
            last_info = active_tracks[tid]
            
            # --- FIX 3: SKIP STATIC TRACKS ---
            # If this track is already flagged as static, we don't try to 
            # "pull" new detections into it anymore.
            if last_info.get('is_static', False):
                continue
                
            dt = f - last_info['last_f']
            if dt > max_gap:
                del active_tracks[tid]
                continue
            
            # 1. PREDICT position
            curr_vx, curr_vy = last_info.get('vx', 0), last_info.get('vy', 0)
            curr_speed = last_info.get('speed', 0)
            
            pred_x = last_info['last_pos'][0] + (curr_vx * dt)
            pred_y = last_info['last_pos'][1] + (curr_vy * dt)
            
            # 2. DYNAMIC SEARCH RANGE
            dynamic_threshold = min(min_radius + (3*curr_speed * dt * velocity_margin), max_radius)
            
            best_dist, best_idx = dynamic_threshold, -1
            for idx in frame_indices:
                if idx in assigned_indices: continue
                
                dist = np.sqrt((pred_x - df.loc[idx, 'cx'])**2 + (pred_y - df.loc[idx, 'cy'])**2)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            
            # 3. Update track
            if best_idx != -1:
                new_pos = (df.loc[best_idx, 'cx'], df.loc[best_idx, 'cy'])
                
                # --- STATIONARY LOGIC ---
                # Calculate distance from the literal last recorded position
                displacement = np.sqrt((new_pos[0] - last_info['last_pos'][0])**2 + 
                                       (new_pos[1] - last_info['last_pos'][1])**2)
                
                # If movement is negligible (e.g., < 3 pixels), increment counter
                if displacement < 3:
                    stationary_count = last_info.get('stationary_count', 0) + 1
                else:
                    stationary_count = 0 # Reset if it actually moves
                
                # Flag as static if threshold reached
                is_static = stationary_count >= 30
                
                # Calculate physics
                dx, dy = new_pos[0] - last_info['last_pos'][0], new_pos[1] - last_info['last_pos'][1]
                vx, vy = dx / dt, dy / dt
                speed = np.sqrt(vx**2 + vy**2)
                
                df.at[best_idx, 'track_id'] = tid
                df.at[best_idx, 'vx'], df.at[best_idx, 'vy'] = vx, vy
                df.at[best_idx, 'speed_px_frame'] = speed
                
                active_tracks[tid] = {
                    'last_f': f, 
                    'last_pos': new_pos,
                    'vx': vx, 'vy': vy, 'speed': speed,
                    'stationary_count': stationary_count,
                    'is_static': is_static
                }
                assigned_indices.append(best_idx)
        
        # New track initialization
        for idx in frame_indices:
            if idx not in assigned_indices:
                df.at[idx, 'track_id'] = next_id
                active_tracks[next_id] = {
                    'last_f': f, 
                    'last_pos': (df.loc[idx, 'cx'], df.loc[idx, 'cy']),
                    'vx': 0.0, 'vy': 0.0, 'speed': 0.0,
                    'stationary_count': 0,
                    'is_static': False
                }
                next_id += 1
                
    df.to_csv(output_csv, index=False)




def track_with_physics_predictive_old(csv_input, output_csv, min_radius=50, max_radius=250, max_gap=15, velocity_margin=0.5):
    """
    Args:
        min_radius: The minimum search area (pixels) even if the ball is stationary.
        max_radius: The absolute maximum distance to search (safety cap).
        max_gap: Max frames to keep a track alive without a detection.
        velocity_margin: How much "extra" space to give based on current speed. 
                         0.5 means search radius grows by 50% of the distance traveled per frame.
    """
    df = pd.read_csv(csv_input).sort_values('frame')
    df['cx'] = (df['x1'] + df['x2']) / 2
    df['cy'] = (df['y1'] + df['y2']) / 2
    
    df['track_id'] = -1
    df['vx'] = 0.0
    df['vy'] = 0.0
    df['speed_px_frame'] = 0.0
    df['direction_deg'] = 0.0
    
    next_id = 0
    active_tracks = {} 
    
    for f in sorted(df['frame'].unique()):
        frame_indices = df[df['frame'] == f].index.tolist()
        assigned_indices = []
        
        sorted_tids = sorted(active_tracks.keys(), key=lambda x: active_tracks[x]['last_f'], reverse=True)
        
        for tid in sorted_tids:
            last_info = active_tracks[tid]
            dt = f - last_info['last_f']
            
            if dt > max_gap:
                del active_tracks[tid]
                continue
            
            # 1. PREDICT position
            curr_vx = last_info.get('vx', 0)
            curr_vy = last_info.get('vy', 0)
            curr_speed = last_info.get('speed', 0)
            
            pred_x = last_info['last_pos'][0] + (curr_vx * dt)
            pred_y = last_info['last_pos'][1] + (curr_vy * dt)
            
            # 2. DYNAMIC SEARCH RANGE
            # The radius grows based on speed and time elapsed (dt)
            # Formula: Base Radius + (Potential deviation based on speed)
            dynamic_threshold = min_radius + (3*curr_speed * dt * velocity_margin)
            dynamic_threshold = min(dynamic_threshold, max_radius) # Cap it
            
            if(dynamic_threshold > max_radius):
                print("consider increasing max_radius because of frame ", f)
            best_dist, best_idx = dynamic_threshold, -1
            
            for idx in frame_indices:
                if idx in assigned_indices: continue
                
                dist = np.sqrt((pred_x - df.loc[idx, 'cx'])**2 + 
                               (pred_y - df.loc[idx, 'cy'])**2)
                
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            
            # 3. Update track
            if best_idx != -1:
                dx = df.loc[best_idx, 'cx'] - last_info['last_pos'][0]
                dy = df.loc[best_idx, 'cy'] - last_info['last_pos'][1]
                
                vx, vy = dx / dt, dy / dt
                speed = np.sqrt(vx**2 + vy**2)
                angle = np.degrees(np.arctan2(dy, dx))
                
                df.at[best_idx, 'track_id'] = tid
                df.at[best_idx, 'vx'] = vx
                df.at[best_idx, 'vy'] = vy
                df.at[best_idx, 'speed_px_frame'] = speed
                df.at[best_idx, 'direction_deg'] = angle
                
                active_tracks[tid] = {
                    'last_f': f, 
                    'last_pos': (df.loc[best_idx, 'cx'], df.loc[best_idx, 'cy']),
                    'vx': vx,
                    'vy': vy,
                    'speed': speed
                }
                assigned_indices.append(best_idx)
        
        for idx in frame_indices:
            if idx not in assigned_indices:
                df.at[idx, 'track_id'] = next_id
                active_tracks[next_id] = {
                    'last_f': f, 
                    'last_pos': (df.loc[idx, 'cx'], df.loc[idx, 'cy']),
                    'vx': 0.0,
                    'vy': 0.0,
                    'speed': 0.0
                }
                next_id += 1
    df.to_csv(output_csv, index=False)
                
