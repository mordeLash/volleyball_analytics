import cv2
import csv
from ultralytics import YOLO
from tqdm import tqdm

def get_ball_detections(model_path, video_path, output_csv, device='cpu'):
    """
    Runs YOLO inference on a video and saves detections to a CSV file.

    This function utilizes a generator-based approach to handle large video files 
    efficiently without overloading memory. It captures bounding box coordinates, 
    class IDs, and confidence scores for every detection.

    Args:
        model_path (str): Path to the exported YOLO model weights (e.g., .pt or .engine).
        video_path (str): Path to the input volleyball video file.
        output_csv (str): Path where the detection results will be saved.
        device (str): Device to run inference on ('cpu', '0', 'cuda', etc.). 
            Defaults to 'cpu'.

    Returns:
        None: Results are written directly to the specified CSV file.
    """
    # Initialize YOLO model for detection task
    model = YOLO(model_path, task="detect")

    # Get total frames to initialize the tqdm progress bar
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Generator for inference: stream=True reduces memory usage by yielding 
    # results one frame at a time rather than loading the whole video.
    results = model.predict(
        source=video_path,
        device=device,
        conf=0.1,    # Low threshold to capture all potential ball movements
        iou=0,       # only tracking one class (the ball), so IOU threshold is less relevant
        imgsz=640,
        stream=True, 
        verbose=False
    )

    print(f"Starting inference on {total_frames} frames...")
    
    # Open CSV in write mode and define the header
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

        # Progress bar context
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            for frame_count, result in enumerate(results):
                boxes = result.boxes
                if len(boxes) > 0:
                    # Move the entire tensor to CPU and convert to 
                    # numpy in one go to avoid repeated synchronization overhead.
                    data = boxes.data.cpu().numpy() 
                    # Row format: [x1, y1, x2, y2, conf, class]

                    rows = []
                    for row in data:
                        rows.append([
                            frame_count, 
                            int(row[5]),          # class_id
                            f"{row[4]:.4f}",      # confidence (4 decimal places)
                            int(row[0]), int(row[1]), int(row[2]), int(row[3]) # x1, y1, x2, y2
                        ])
                    
                    # Write all detections for the current frame at once
                    writer.writerows(rows)

                pbar.update(1)

    print(f"\nProcessing complete. Results saved to {output_csv}")