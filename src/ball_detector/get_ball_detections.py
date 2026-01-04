import cv2
import time
import csv
from tqdm import tqdm
from ultralytics import YOLO
import os

def get_ball_detections(model_path=None, video_path=None, output_csv=None, device=None):
    # 1. Load the OpenVINO model
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
            
            # device="cpu" is redundant if the model is already OpenVINO, 
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

def get_ball_detections_fast(model_path, video_path, output_csv, device='cpu'):
    model = YOLO(model_path, task="detect")

    # Get total frames for the progress bar
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
    
    # Initialize CSV with header
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

    # Wrap the results generator with tqdm for a live progress bar
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        for frame_count, result in enumerate(results):
            boxes = result.boxes
            if len(boxes) > 0:
                # Optimized bulk extraction
                coords = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                rows = []
                for i in range(len(coords)):
                    rows.append([
                        frame_count, clss[i], f"{confs[i]:.4f}", 
                        coords[i][0], coords[i][1], coords[i][2], coords[i][3]
                    ])
                
                # Append to CSV
                with open(output_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)

            pbar.update(1)

    print(f"\nProcessing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    pass