import os
import json
import shutil
import cv2
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# HELPERS
# -----------------------------
def get_video_properties(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,width,height,duration",
        "-of", "json",
        video_path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    s = json.loads(p.stdout)["streams"][0]

    num, den = map(int, s["r_frame_rate"].split("/"))
    fps = num / den
    width = int(s["width"])
    height = int(s["height"])

    if "nb_frames" in s:
        total_frames = int(s["nb_frames"])
    else:
        total_frames = int(float(s["duration"]) * fps)

    return fps, total_frames, width, height

# -----------------------------
# CORE ENGINE
# -----------------------------
def run_processing(segments, video_path, output_path, lookup, mode, fps, width, height):
    """Internal engine with updated visualization for track_id, speed, and confidence."""
    os.makedirs("tmp", exist_ok=True)
    final_clips = []

    for i, (f0, f1) in enumerate(tqdm(segments, desc=f"Mode: {mode}")):
        t0 = f0 / fps
        dur = (f1 - f0 + 1) / fps
        temp_output = f"tmp/{mode}_seg_{i:04d}.mp4"

        if mode == "none":
            # PATH A: CUT ONLY (Frame-perfect seek with audio)
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "quiet", "-i", video_path,
                "-ss", str(t0), "-t", str(dur),
                "-c:v", "libx264", "-crf", "18", "-c:a", "aac", 
                temp_output
            ], check=True, capture_output=True)
        else:
            # PATH B: VISUALIZATION (No Audio)
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-loglevel', 'quiet',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-', 
                '-an', 
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
                '-crf', '23', temp_output
            ]
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            
            trail = []
            for current_frame_idx in range(f0, f1 + 1):
                ret, frame = cap.read()
                if not ret: break

                d = lookup.get(current_frame_idx)
                if d:
                    cx, cy = int(d["cx"]), int(d["cy"])
                    
                    if mode in ("data", "both"):
                        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
                        # Draw Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # --- Metadata Display (Including Confidence) ---
                        tid = d.get("track_id", "N/A")
                        speed = d.get("speed_px_frame", 0.0)
                        conf = d.get("conf", 0.0) # Added Confidence
                        
                        # Added C: for confidence to the label
                        label = f"ID:{tid} S:{speed:.1f} C:{conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    trail.append((cx, cy))
                else:
                    trail.append(None)

                if len(trail) > 30: trail.pop(0)

                if mode in ("trajectory", "both"):
                    for j in range(1, len(trail)):
                        if trail[j] is not None and trail[j - 1] is not None:
                            if np.linalg.norm(np.subtract(trail[j], trail[j - 1])) < 120:
                                cv2.line(frame, trail[j - 1], trail[j], (255, 255, 0), 2)

                process.stdin.write(frame.tobytes())

            cap.release()
            process.stdin.close()
            process.wait()

        final_clips.append(temp_output)

    # --- BUG FIX: Handle Single or Multiple Clips ---
    if len(final_clips) == 1:
        # If only one clip, just move it to the final destination
        shutil.move(final_clips[0], output_path)
    elif len(final_clips) > 1:
        # Concat multiple clips
        list_path = f"tmp/list_{mode}.txt"
        with open(list_path, "w") as f:
            for c in final_clips:
                f.write(f"file '{Path(c).resolve().as_posix()}'\n")
        
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "quiet", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)
        
        if os.path.exists(list_path): os.remove(list_path)

    # Cleanup any remaining temp segments
    for c in final_clips: 
        if os.path.exists(c): os.remove(c)

# -----------------------------
# PUBLIC API
# -----------------------------
def visualize(
    video_path,
    output_path,
    tracking_csv,
    predictions_csv=None,
    overlay_mode="both",
    buffer_sec=1.0,
):
    fps, total_frames, width, height = get_video_properties(video_path)

    df = pd.read_csv(tracking_csv)
    if "cx" not in df:
        df["cx"] = (df["x1"] + df["x2"]) / 2
        df["cy"] = (df["y1"] + df["y2"]) / 2
    
    if "speed_px_frame" not in df and "cx" in df:
        df['speed_px_frame'] = np.sqrt(df['cx'].diff()**2 + df['cy'].diff()**2).fillna(0)

    lookup = df.set_index("frame").to_dict("index")

    segments = []
    if predictions_csv and os.path.exists(predictions_csv):
        p = pd.read_csv(predictions_csv)
        p["frame"] = p.index.astype(int)
        p["grp"] = (p["label"] != p["label"].shift()).cumsum()
        rallies = p[p["label"] == 1].groupby("grp")["frame"].agg(["min", "max"])
        
        buf_frames = int(buffer_sec * fps)
        edge_buf_frames = int(3.0 * fps)
        
        rally_list = list(rallies.itertuples())
        for idx, r in enumerate(rally_list):
            start_b = edge_buf_frames if idx == 0 else buf_frames
            end_b = edge_buf_frames if idx == len(rally_list) - 1 else buf_frames
            
            segments.append((
                max(0, int(r.min - start_b)),
                min(total_frames - 1, int(r.max + end_b)),
            ))
    else:
        segments = [(0, total_frames - 1)]

    if overlay_mode == "all":
        viz_out = output_path.replace(".mp4", "_visualized.mp4")
        run_processing(segments, video_path, viz_out, lookup, "both", fps, width, height)
        
        cut_out = output_path.replace(".mp4", "_cuts_only.mp4")
        run_processing(segments, video_path, cut_out, lookup, "none", fps, width, height)
    else:
        run_processing(segments, video_path, output_path, lookup, overlay_mode, fps, width, height)

    if os.path.exists("tmp"): shutil.rmtree("tmp")