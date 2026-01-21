import cv2
import pandas as pd
import numpy as np
import os
import time
import argparse

class RallyAnnotator:
    def __init__(self, video_path, output_csv, predictions_path=None):
        self.video_path = video_path
        self.output_csv = output_csv
        self.predictions_path = predictions_path
        
        # Video Init
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Resize Config (720p target)
        self.disp_h = 720
        self.scale = self.disp_h / self.height
        self.disp_w = int(self.width * self.scale)
        
        # State
        self.current_frame = 0
        self.playback_speed = 0 # 0=Paused, 1=1x, 2=2x
        self.labels = np.zeros(self.total_frames, dtype=np.int8) # 0=Downtime, 1=Rally
        self.recording_label = None # If not None, we overwrite as we play
        
        # Load Labels (or predictions)
        self.load_labels()
        
        # Window
        self.window_name = "Rally Annotator v2"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.disp_w, self.disp_h)
        cv2.createTrackbar("Frame", self.window_name, 0, self.total_frames, self.on_trackbar)
        
    def load_labels(self):
        # Priority 1: Load existing work (Output CSV)
        if os.path.exists(self.output_csv):
            print(f"Loading existing annotations from {self.output_csv}...")
            try:
                df = pd.read_csv(self.output_csv)
                # Map Strings to Ints
                mapping = {'Rally': 1, 'Downtime': 0}
                # Handle potential column names
                col = 'label' if 'label' in df.columns else df.columns[1]
                vals = df[col].map(mapping).fillna(0).astype(np.int8).values
                
                # Copy to array
                limit = min(len(vals), self.total_frames)
                self.labels[:limit] = vals[:limit]
                return
            except Exception as e:
                print(f"Load failed: {e}")
        
        # Priority 2: Load Predictions (if provided and no existing work)
        if self.predictions_path and os.path.exists(self.predictions_path):
            print(f"Loading predictions from {self.predictions_path}...")
            try:
                df = pd.read_csv(self.predictions_path)
                # Predictions are usually 0/1 in 'label' column
                if 'label' in df.columns:
                    vals = df['label'].fillna(0).astype(np.int8).values
                    limit = min(len(vals), self.total_frames)
                    self.labels[:limit] = vals[:limit]
                    print("Predictions loaded as initial labels.")
                else:
                    print("Error: 'label' column not found in predictions.")
            except Exception as e:
                print(f"Failed to load predictions: {e}")

    def save(self):
        print(f"Saving to {self.output_csv}...")
        str_labels = ["Rally" if x == 1 else "Downtime" for x in self.labels]
        df = pd.DataFrame({
            'frame': range(self.total_frames),
            'label': str_labels
        })
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print("Saved.")

    def on_trackbar(self, val):
        # Only seek if meaningful change to avoid self-trigger
        if abs(self.current_frame - val) > 5:
            self.current_frame = val
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    def jump_event(self, direction):
        # direction: +1 (Next), -1 (Prev)
        curr = int(self.current_frame)
        current_lab = self.labels[curr]
        
        search_range = range(curr + direction, self.total_frames) if direction > 0 else range(curr + direction, -1, -1)
        
        for i in search_range:
            if self.labels[i] != current_lab:
                # Found change (e.g. Rally -> Downtime)
                self.current_frame = i
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                return

    def extend_section(self):
        # "Extend Section" by 10 frames
        # Logic: Find the END of the current contiguous block of labels from current position
        # And extend it 10 frames forward.
        
        curr = int(self.current_frame)
        current_lab = self.labels[curr]
        
        # Find end
        end_idx = curr
        for i in range(curr, self.total_frames):
            if self.labels[i] != current_lab:
                break
            end_idx = i
            
        # Extend from end_idx
        target_end = min(self.total_frames, end_idx + 10 + 1) # +10 frames
        self.labels[end_idx : target_end] = current_lab
        print(f"Extended {current_lab} to frame {target_end}")
        
        # Jump to new end to show user
        self.current_frame = min(target_end - 1, self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def force_refresh_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.last_display_frame = cv2.resize(frame, (self.disp_w, self.disp_h))
            
    def draw_timeline(self, img):
        h, w = img.shape[:2]
        bar_h = 30
        y = h - bar_h
        
        # Resample labels
        idxs = np.linspace(0, self.total_frames-1, w).astype(int)
        labs = self.labels[idxs]
        
        strip = np.zeros((1, w, 3), dtype=np.uint8)
        strip[0, labs==1] = [0, 255, 0]
        strip[0, labs==0] = [0, 0, 255]
        
        bar = cv2.resize(strip, (w, bar_h), interpolation=cv2.INTER_NEAREST)
        img[y:h, 0:w] = bar
        
        # Cursor
        x = int(self.current_frame / self.total_frames * w)
        cv2.line(img, (x, y), (x, h), (255, 255, 255), 2)

    def run_final(self):
        print("--- CONTROLS ---")
        print("[Tab]    Frequency: Pause -> 1x -> 2x")
        print("[Space]  Pause/Play")
        print("[1]      DOWNTIME (Auto-Play)")
        print("[2]      RALLY (Auto-Play)")
        print("[E]      Extend Section (+10 frames)")
        print("[N / P]  Next / Prev Event")
        print("[Arrows] Seek")
        print("[S]      Save")
        print("[Q]      Quit")
        
        # Initial Frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.last_display_frame = cv2.resize(frame, (self.disp_w, self.disp_h))
        else:
            self.last_display_frame = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

        while True:
            t0 = time.time()
            
            # PLAYBACK
            if self.playback_speed > 0:
                # ... skip logic ...
                step = int(self.playback_speed)
                for _ in range(step-1):
                    self.cap.grab()
                    self.current_frame += 1
                
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame += 1
                    # Record
                    if self.recording_label is not None:
                         start = max(0, int(self.current_frame) - step)
                         self.labels[start : int(self.current_frame)] = self.recording_label
                    
                    self.last_display_frame = cv2.resize(frame, (self.disp_w, self.disp_h))
                else:
                    self.playback_speed = 0

            # RENDER
            display = self.last_display_frame.copy()
            curr = int(self.current_frame)
            
            # Text
            lbl = self.labels[min(curr, self.total_frames-1)]
            txt = "RALLY" if lbl == 1 else "DOWNTIME"
            col = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.putText(display, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            
            spd_str = "PAUSED" if self.playback_speed == 0 else f"{self.playback_speed}x"
            rec_str = "[REC]" if self.recording_label is not None else ""
            cv2.putText(display, f"{spd_str} {rec_str}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            self.draw_timeline(display)
            cv2.imshow(self.window_name, display)
            cv2.setTrackbarPos("Frame", self.window_name, curr)
            
            # INPUT
            delay = 33 if self.playback_speed == 0 else int(1000/self.fps)
            # Adjust for processing
            dt = time.time() - t0
            wait = max(1, delay - int(dt*1000))
            
            key = cv2.waitKey(wait) & 0xFF
            
            if key == ord('q'):
                self.save()
                break
            elif key == ord('s'):
                self.save()
            elif key == 9: # Tab
                self.playback_speed = (self.playback_speed + 1) % 3
            elif key == ord(' '):
                self.playback_speed = 1 if self.playback_speed == 0 else 0
            
            elif key == ord('1'):
                self.recording_label = 0
                self.labels[curr] = 0
                if self.playback_speed == 0: self.playback_speed = 1
                
            elif key == ord('2'):
                self.recording_label = 1
                self.labels[curr] = 1
                if self.playback_speed == 0: self.playback_speed = 1

            elif key == ord('e'):
                self.extend_section()
                self.force_refresh_frame()
                
            elif key == ord('n'): self.jump_event(1); self.force_refresh_frame()
            elif key == ord('p'): self.jump_event(-1); self.force_refresh_frame()
            
            # Arrows
            elif key == 2424832 or key == 65361: # Left
                 # ... seek logic ...
                 step_seek = 1 if self.playback_speed == 0 else 30
                 self.current_frame = max(0, self.current_frame - step_seek)
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                 self.force_refresh_frame()
            elif key == 2555904 or key == 65363: # Right
                 step_seek = 1 if self.playback_speed == 0 else 30
                 self.current_frame = min(self.total_frames-1, self.current_frame + step_seek)
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                 self.force_refresh_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volleyball Rally Annotator")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output labels CSV file")
    parser.add_argument("--predictions", type=str, default=None, help="Optional: Path to PREDICTIONS csv to load as starting point")
    
    args = parser.parse_args()
    
    # Defaults for dev
    DEFAULT_VIDEO = r"C:\Users\morde\Desktop\projects\volleyball cv\data\input_videos\game7_set1.mp4"
    DEFAULT_CSV = './output/training_data/game7_set1_rally_labels_per_frame.csv'
    DEFAULT_PREDICTIONS_CSV = r"C:\Users\morde\.gemini\antigravity\cloned\volleyball_analytics\output\training_data\game7_set1_predictions.csv"

    vid = args.video if args.video else DEFAULT_VIDEO
    out = args.output if args.output else DEFAULT_CSV
    predictions = args.predictions if args.predictions else DEFAULT_PREDICTIONS_CSV
    
    ann = RallyAnnotator(vid, out, predictions_path=predictions)
    ann.run_final()