import json
import subprocess
import cv2

from .utils import get_bin_path
def get_video_properties(video_path):
    """
    Extracts core metadata from a video file using ffprobe.

    Args:
        video_path (str): Path to the source video.

    Returns:
        tuple: (fps, total_frames, width, height)
    """
    ffprobe_bin = get_bin_path("ffprobe")

    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,width,height,duration",
        "-of", "json",
        video_path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    s = json.loads(p.stdout)["streams"][0]

    # Handle frame rate fractions (e.g., '30000/1001')
    num, den = map(int, s["r_frame_rate"].split("/"))
    fps = num / den
    width = int(s["width"])
    height = int(s["height"])

    # Fallback for total frames if nb_frames is missing from metadata
    if "nb_frames" in s:
        total_frames = int(s["nb_frames"])
    else:
        total_frames = int(float(s["duration"]) * fps)

    return fps, total_frames, width, height


def ensure_30fps(input_path, output_path, log_func):
    """Checks FPS and converts to 30 if higher. Requires re-encoding."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    ffmpeg_bin = get_bin_path("ffmpeg")
    if fps > 30.5:
        log_func(f"Video FPS is {fps:.2f}. Downsampling to 30 FPS (Re-encoding)...")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-filter:v", "fps=fps=30,setpts=N/FRAME_RATE/TB",  # Reset timestamps!
            "-c:a", "copy",
            "-crf", "23",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    elif fps < 29.5:
        log_func(f"WARNING: Video FPS is low ({fps:.2f}). Tracking accuracy may decrease.")
    
    return input_path

def trim_video(input_path, output_path, start_time, end_time, log_func):
    """Cuts a section of the video using FFmpeg with frame accuracy."""
    if not start_time and not end_time:
        return input_path
    
    log_func(f"Trimming video: {start_time or '00:00:00'} to {end_time or 'End'}")
    ffmpeg_bin = get_bin_path("ffmpeg")
    cmd = [ffmpeg_bin, '-y']
    
    # Put -ss AFTER -i for frame-accurate seeking (slower but precise)
    cmd.extend(['-i', input_path])
    
    if start_time:
        cmd.extend(['-ss', start_time])
    if end_time:
        cmd.extend(['-to', end_time])
    
    # Re-encode for frame accuracy
    cmd.extend([
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ])
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path