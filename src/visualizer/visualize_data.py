import os
import json
import shutil
import cv2
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.utils.manipulate_video import get_video_properties
from src.utils.utils import get_bin_path
def run_processing(segments, video_path, output_path, lookup, mode, fps, width, height):
    """
    The main rendering engine. Supports two distinct paths:
    1. 'cut': Blazing fast lossless-like cutting with audio using FFmpeg filters.
    2. Other modes: OpenCV drawing + FFmpeg pipe to render data/trajectories.

    Args:
        segments (list): List of (start_frame, end_frame) tuples to include.
        video_path (str): Input video path.
        output_path (str): Where to save the result.
        lookup (dict): Dictionary mapping frames to tracking data.
        mode (str): Visualization style ('cut', 'data', 'trajectory', 'both').
        fps, width, height (int/float): Video dimensions and speed.
    """
    ffmpeg_bin = get_bin_path("ffmpeg")
    if mode == "cut":
        # --- PATH A: PURE FFMPEG (Stream Manipulation) ---
        # Extracts and concatenates segments directly. This keeps the audio 
        # perfectly synced and avoids re-encoding overhead where possible.
        filter_script = ""
        for i, (f0, f1) in enumerate(segments):
            t0, dur = f0 / fps, (f1 - f0 + 1) / fps
            # Video segment
            filter_script += f"[0:v]trim=start={t0}:duration={dur},setpts=PTS-STARTPTS[v{i}]; "
            # Audio segment + Resampling to prevent sync drift
            filter_script += f"[0:a]atrim=start={t0}:duration={dur},asetpts=PTS-STARTPTS,aresample=async=1[a{i}]; "

        concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(len(segments))])
        filter_script += f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]"
        
        cmd = [
            ffmpeg_bin, "-y", "-i", video_path,
            "-filter_complex", filter_script,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", output_path
        ]
        subprocess.run(cmd, check=True,stdout=subprocess.DEVNULL, 
    stderr=subprocess.DEVNULL)
        return

    # --- PATH B: VISUALIZATION (OpenCV + FFmpeg Pipe) ---
    # We open a pipe to FFmpeg. We feed it raw frames from OpenCV via stdin,
    # and FFmpeg merges those frames with the original audio segments.
    # Calculate total frames for audio segments to ensure duration matches video
    filter_parts = []
    for i, (f0, f1) in enumerate(segments):
        t0 = f0 / fps
        # Using duration based on frames ensures the audio segment 
        # is exactly as long as the frames you are pushing
        dur = (f1 - f0 + 1) / fps 
        filter_parts.append(
            f"[1:a]atrim=start={t0}:duration={dur},asetpts=PTS-STARTPTS,aresample=async=1[a{i}];"
        )

    audio_concat = "".join([f"[a{i}]" for i in range(len(segments))])
    filter_script = "".join(filter_parts) + f"{audio_concat}concat=n={len(segments)}:v=0:a=1[outa]"

    ffmpeg_cmd = [
        ffmpeg_bin, '-y', '-loglevel', 'error',
        # Input 0: Raw frames pipe
        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', 
        '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',  
        # Input 1: Source video
        '-i', video_path, 
        '-filter_complex', filter_script,
        '-map', '0:v',          # Take video from pipe
        '-map', '[outa]',       # Take audio from filter
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', 
        '-pix_fmt', 'yuv420p', 
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',            # CRITICAL: Cuts the file if one stream ends before the other
        output_path
    ]
    
    cap = cv2.VideoCapture(video_path)
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for f0, f1 in tqdm(segments, desc=f"Visualizing: {mode}"):
            # Seek to the start of the rally segment
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != f0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            
            trail = [] # Stores points for the trajectory 'tail'
            for current_frame_idx in range(f0, f1 + 1):
                ret, frame = cap.read()
                if not ret: break

                d = lookup.get(current_frame_idx)
                if d:
                    # Draw Bounding Box and ID/Speed info
                    if mode in ("data", "both"):
                        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID:{d.get('track_id','N/A')} S:{d.get('conf',0):.1f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    trail.append((int(d["cx"]), int(d["cy"])))
                else:
                    trail.append(None)

                # Keep a 30-frame rolling window for the trajectory visual
                if len(trail) > 30: trail.pop(0)
                if mode in ("trajectory", "both") and len(trail) > 1:
                    pts = np.array([p for p in trail if p is not None], np.int32).reshape((-1, 1, 2))
                    if len(pts) > 1:
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 2, cv2.LINE_AA)

                # Push the modified frame to FFmpeg's stdin
                process.stdin.write(frame.tobytes())
    finally:
        cap.release()
        process.stdin.close()
        process.wait()

# -----------------------------
# PUBLIC API
# -----------------------------
def visualize(
    video_path,
    output_path,
    tracking_csv,
    predictions_csv=None,
    overlay_mode="both",
    buffer_sec=1.5,
):
    """
    The public interface to generate highlight reels from tracking and rally data.

    It calculates segment boundaries (adding buffer time to rallies) and calls the 
    processing engine to generate the final MP4 files.

    Args:
        video_path (str): Original video file.
        output_path (str): Desired output filename.
        tracking_csv (str): CSV with ball coordinates and speeds.
        predictions_csv (str, optional): CSV with 'Rally' vs 'Downtime' labels.
        overlay_mode (str): 'both', 'data', 'trajectory', 'cut', or 'all'.
        buffer_sec (float): Extra time to include before/after a rally for context.
    """
    fps, total_frames, width, height = get_video_properties(video_path)

    # 1. Prepare Tracking Data
    df = pd.read_csv(tracking_csv)
    if "cx" not in df:
        df["cx"] = (df["x1"] + df["x2"]) / 2
        df["cy"] = (df["y1"] + df["y2"]) / 2
    
    if "speed_px_frame" not in df and "cx" in df:
        df['speed_px_frame'] = np.sqrt(df['cx'].diff()**2 + df['cy'].diff()**2).fillna(0)

    # Convert to dictionary for O(1) frame lookup during video loop
    lookup = df.set_index("frame").to_dict("index")

    # 2. Segment Generation Logic
    segments = []
    if predictions_csv and os.path.exists(predictions_csv):
        p = pd.read_csv(predictions_csv)
        p["frame"] = p.index.astype(int)
        # Group consecutive predictions (State segments)
        p["grp"] = (p["label"] != p["label"].shift()).cumsum()
        rallies = p[p["label"] == 1].groupby("grp")["frame"].agg(["min", "max"])
        
        buf_frames = int(buffer_sec * fps)
        edge_buf_frames = int(3.0 * fps) # Extra buffer for the very start/end of the match
        
        rally_list = list(rallies.itertuples())
        for idx, r in enumerate(rally_list):
            start_b = edge_buf_frames if idx == 0 else buf_frames
            end_b = edge_buf_frames if idx == len(rally_list) - 1 else buf_frames
            
            segments.append((
                max(0, int(r.min - start_b)),
                min(total_frames - 1, int(r.max + end_b)),
            ))
    else:
        # If no predictions provided, process the entire video as one segment
        segments = [(0, total_frames - 1)]

    # 3. Final Execution
    if overlay_mode == "all":
        # Generate two versions: one with boxes/trajectories and one clean-cut version
        viz_out = output_path.replace(".mp4", "_visualized.mp4")
        run_processing(segments, video_path, viz_out, lookup, "both", fps, width, height)
        
        cut_out = output_path.replace(".mp4", "_cuts_only.mp4")
        run_processing(segments, video_path, cut_out, lookup, "cut", fps, width, height)
    else:
        # Generate the single requested version
        cut_out = output_path.replace(".mp4", "_cuts_only.mp4")
        run_processing(segments, video_path, cut_out, lookup, overlay_mode, fps, width, height)

    # Cleanup temporary directories
    if os.path.exists("tmp"): shutil.rmtree("tmp")