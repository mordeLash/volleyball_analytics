import cv2
import pandas as pd
import os

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\morde\Desktop\projects\volleyball cv\data\input_videos\game5_set1.mp4"
PER_FRAME_OUTPUT_FILE = './output/training_data/game5_rally_labels_per_frame.csv'
PLAYBACK_SPEED = 2 

def label_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if actual_fps == 0: actual_fps = 30.0
    
    # --- INTERNAL TRACKING ---
    # We only need to store the event markers (where the state changed)
    event_markers = [] 
    current_state = "Downtime"
    
    delay = max(1, int((1000 / actual_fps) / PLAYBACK_SPEED))
    paused = False
    
    print(f"Controls:")
    print(" [2] Rally | [1] Downtime | [Space] Pause")
    print(" [Z] Undo Last | [Left Arrow] Rewind 5s | [Q] Save & Quit")

    curr_frame = 0

    while cap.isOpened():
        if not paused:
            # Skip frames based on playback speed
            for _ in range(PLAYBACK_SPEED):
                ret, frame = cap.read()
                if not ret: break
            if not ret: break
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        timestamp = curr_frame / actual_fps
        display_frame = frame.copy()
        state_color = (0, 255, 0) if current_state == "Rally" else (0, 0, 255)
        
        cv2.putText(display_frame, f"State: {current_state}", (50, 50), 2, 1, state_color, 2)
        cv2.putText(display_frame, f"Frame: {curr_frame} / {total_frames}", (50, 100), 2, 1, (255, 255, 255), 2)
        
        cv2.imshow('Volleyball Labeler', display_frame)
        key = cv2.waitKeyEx(delay)

        # Record event markers: (frame_index, label_name)
        if key == ord('2'):
            current_state = "Rally"
            event_markers.append((curr_frame, "Rally"))
        elif key == ord('1'):
            current_state = "Downtime"
            event_markers.append((curr_frame, "Downtime"))
        elif key == ord('z') or key == ord('Z'):
            if event_markers:
                event_markers.pop()
                current_state = event_markers[-1][1] if event_markers else "Downtime"
        elif key == 2424832: # Left Arrow
            curr_frame = max(0, curr_frame - int(actual_fps * 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            ret, frame = cap.read() 
        elif key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if not event_markers and current_state == "Downtime":
        print("No labels to save.")
        return

    # --- GENERATE PER-FRAME LABELS ---
    print("Processing per-frame labels...")
    
    # Initialize list with default state
    full_labels = ["Downtime"] * total_frames
    
    # Sort markers by frame to ensure correct filling
    event_markers.sort()

    # Fill the labels based on markers
    for i in range(len(event_markers)):
        start_f, label = event_markers[i]
        # Everything from this marker to the end of the video is 'label'
        # (Subsequent markers will overwrite their portions)
        full_labels[start_f:] = [label] * (total_frames - start_f)

    # --- SAVE TO CSV ---
    os.makedirs(os.path.dirname(PER_FRAME_OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame({'label': full_labels})
    df.to_csv(PER_FRAME_OUTPUT_FILE, index_label='frame')
    
    print(f"Successfully saved {total_frames} frame labels to {PER_FRAME_OUTPUT_FILE}")

if __name__ == "__main__":
    label_video(VIDEO_PATH)