from ultralytics import YOLO

# Load your custom or pretrained PyTorch model
model = YOLO("./models/ball_detection/vbn11.pt")  # or "yolo11n.pt"

# Export to OpenVINO format
model.export(
    format="openvino", 
    half=True,         # This is the key for FP16
    imgsz=640,         # Match your training image size
    augment=False
)