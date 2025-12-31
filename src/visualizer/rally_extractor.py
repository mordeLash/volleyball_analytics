import pandas as pd
import subprocess
import os

def get_video_info(video_path):
    """Gets FPS and total duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')
    
    # Calculate FPS from fraction (e.g., "30000/1001")
    fps_fraction = lines[0].split('/')
    fps = float(fps_fraction[0]) / float(fps_fraction[1]) if len(fps_fraction) > 1 else float(fps_fraction[0])
    duration = float(lines[1])
    return fps, duration

def cut_rallies(video_path, csv_path, output_path, buffer_sec=1.0):
    print("Reading CSV and video metadata...")
    
    # 1. Load Data & Clean Columns
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Remove hidden spaces
    
    # Identify the correct column
    target_col = 'rf_preds2'
    if target_col not in df.columns:
        print(f"Warning: '{target_col}' not found. Using '{df.columns[0]}' instead.")
        target_col = df.columns[0]

    fps, total_duration = get_video_info(video_path)

    # 2. Identify Rally Blocks
    df['is_rally'] = df[target_col] == 1
    # Create a unique ID for each continuous block of 0s or 1s
    df['block'] = (df['is_rally'] != df['is_rally'].shift()).cumsum()
    
    # Filter for rallies only
    df_rallies = df[df['is_rally']].copy()
    
    if df_rallies.empty:
        print("No rallies detected in the CSV.")
        return

    # FIX: Correct way to get min/max of index for each block
    df_rallies['frame_idx'] = df_rallies.index
    rallies = df_rallies.groupby('block')['frame_idx'].agg(['min', 'max'])
    
    # 3. Convert to Time Intervals with Buffer
    intervals = []
    for _, row in rallies.iterrows():
        start = max(0, (row['min'] / fps) - buffer_sec)
        end = min(total_duration, (row['max'] / fps) + buffer_sec)
        intervals.append((start, end))

    # 4. Merge Overlapping Intervals
    intervals.sort()
    merged = []
    if intervals:
        curr_start, curr_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

    # 5. Build FFmpeg Filter Script
    print(f"Detected {len(merged)} rally segments. Processing video...")
    
    filter_content = ""
    concat_inputs = ""
    for i, (start, end) in enumerate(merged):
        # Format times to 3 decimal places
        s, e = f"{start:.3f}", f"{end:.3f}"
        filter_content += f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}];\n"
        filter_content += f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}];\n"
        concat_inputs += f"[v{i}][a{i}]"
    
    filter_content += f"{concat_inputs}concat=n={len(merged)}:v=1:a=1[outv][outa]"

    # Write filter to a temporary file (prevents "Command line too long" error)
    filter_file = "ffmpeg_filter.txt"
    with open(filter_file, "w") as f:
        f.write(filter_content)

    # 6. Execute FFmpeg
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path,
        '-filter_complex_script', filter_file,
        '-map', '[outv]', '-map', '[outa]',
        '-c:v', 'libx264', '-crf', '21', '-preset', 'veryfast',
        '-c:a', 'aac', '-b:a', '128k', 
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccess! Saved to {output_path}")
    finally:
        if os.path.exists(filter_file):
            os.remove(filter_file)

if __name__ == "__main__":
    # Your specific paths
    VIDEO = r"C:\Users\morde\Desktop\projects\volleyball cv\data\input_videos\game7.mp4"
    CSV = r"rally-detector\s_predictions_game7_full.csv"
    OUTPUT = "game7_full_cut.mp4"
    
    cut_rallies(VIDEO, CSV, OUTPUT)