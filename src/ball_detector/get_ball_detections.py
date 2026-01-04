import cv2
import time
import csv
from tqdm import tqdm
from ultralytics import YOLO
import os
import pandas as pd # Optional, but recommended


def get_ball_detections(model_path=None, video_path=None, output_csv=None, device=None):
    # Ensure the path points to the folder containing the .xml and .bin files
    model_path = model_path
    model = YOLO(model_path, task="detect")

    # 2. Setup Video Input and CSV Output
    video_path = video_path
    output_csv = output_csv

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: frame_id, class, confidence, x_min, y_min, x_max, y_max
        writer.writerow(['frame', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

        frame_count = 0
        print(f"Starting inference on {total_frames} frames using CPU...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 3. Run Inference
            start_time = time.time()
            # but good for clarity. stream=True is more memory efficient.
            results = model.predict(
                source=frame, 
                device=device, 
                conf=0.1, 
                iou=0, 
                imgsz=640, 
                verbose=False,
                stream=False # Set to False here to handle single frame objects easily
            )
            
            end_time = time.time()

            # 4. Calculate and Print FPS
            inference_time = end_time - start_time
            fps = 1 / inference_time
            #print(f"Frame {frame_count}/{total_frames} | FPS: {fps:.2f}", end="\r")

            # 5. Process Results and Save to CSV
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract coordinates and info
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Write row to CSV
                    writer.writerow([frame_count, cls, f"{conf:.4f}", int(x1), int(y1), int(x2), int(y2)])

            frame_count += 1

        cap.release()
        print(f"\nProcessing complete. Results saved to {output_csv}")

def get_ball_detections_fast(model_path, video_path, output_csv, device='cpu'): # Use '0' for GPU
    model = YOLO(model_path, task="detect")

    # Get total frames for progress bar
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Generator for inference
    results = model.predict(
        source=video_path,
        device=device,
        conf=0.1,
        iou=0,
        imgsz=640,
        stream=True, 
        verbose=False
    )

    print(f"Starting inference on {total_frames} frames...")
    
    # Open the file ONCE at the start
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            for frame_count, result in enumerate(results):
                boxes = result.boxes
                if len(boxes) > 0:
                    # Move all data to CPU in one single batch transfer
                    # This is much faster than individual calls
                    data = boxes.data.cpu().numpy() 
                    # data format: [x1, y1, x2, y2, conf, class]

                    rows = []
                    for row in data:
                        rows.append([
                            frame_count, 
                            int(row[5]),          # class
                            f"{row[4]:.4f}",      # conf
                            int(row[0]), int(row[1]), int(row[2]), int(row[3]) # coords
                        ])
                    
                    # Write rows in bulk while the file is already open
                    writer.writerows(rows)

                pbar.update(1)

    print(f"\nProcessing complete. Results saved to {output_csv}")


if __name__ == "__main__":
    pass