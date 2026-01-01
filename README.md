# Automated Volleyball Match Analysis 

This project provides an automated computer vision pipeline for analyzing volleyball match footage. It tracks ball movement to distinguish between active rallies and downtime, facilitating automated highlight extraction and performance analytics.

| ![cut clip1](./assets/track_clip1.gif) | ![track clip2](./assets/track_clip2.gif) |

---

## 🛠️ Installation & Setup

This project requires **Python 3.12+**.

### Dependencies

The following libraries are required for the pipeline:

* **Computer Vision:** `ultralytics` (YOLOv11), `opencv-python`
* **Data Analysis:** `numpy`, `pandas`, `scikit-learn`
* **Visualization:** `seaborn`, `tqdm`
* **Utility:** `pathlib`

---

## 📑 Pipeline Stages

### Stage 1: Detect Volleyball in Video

* **Train YOLOv11 Detection Model:** The model was trained using an iterative active learning pipeline involving **SAM3** (Segment Anything Model) to generate ground-truth data from "hard frames."
* **Build Framework for Video Inference:** The framework runs detection on every frame, saving bounding box coordinates to a CSV.
* **Optimization:** The model was converted to **OpenVINO** to speed up inference on local PCs.

### Stage 2: Convert Detections into Tracks

* **Track Assignment:** Each detection is assigned a unique Track ID.
* **Physics-based Prediction:** Future frame assignments are based on physics-based location predictions.
* **Static Object Filtering:** An `is_static` flag prevents moving tracks from incorrectly merging with background objects like sideline balls.

### Stage 3: Clean Tracks

* **Noise Removal:** Low-movement objects and "flickering" detections caused by limbs are filtered out.
* **Continuity:** Prioritizes track continuity over raw confidence scores to maintain detection during high-velocity or blurry frames.

### Stage 4: Feature Extraction

* **Interpolation:** Small gaps in tracks are interpolated for continuity.
* **Movement Metrics:** Calculates **velocity**, **acceleration**, and **y-axis dominance** within a sliding window.

### Stage 5: Predict Rally or Downtime

This stage utilizes a machine learning classifier to categorize match states.

* **Train a Random Forest Model:** A `RandomForestClassifier` from Scikit-learn was trained on labeled data from three different matches.
* **Evaluation:** The model is assessed based on its F1 score and the relative importance of specific features.

| ![Predictor F1 Score](./assets/rally_predictor_v4_f1_score.png) | ![Feature Importance](./assets/rally_predictor_v4_feature_importance.png) |

---

### Stage 6: Clean Predictions

* **Temporal Smoothing:** A rolling window is applied to the Random Forest output to remove jitter.
* **Analytics Extraction:** Extracts final stats such as the total number of rallies and average rally duration.

### Stage 7: Visualization

* **Option 1 (Highlights):** Automatically trims the video based on predictions with a 1-second buffer.
* **Option 2 (Data Overlay):** Visualizes bounding boxes, ball "trails," and real-time velocity (pixels per frame).

---

## 💻 Usage

The pipeline is executed via `main.py`. You can run the full process or start from intermediate CSV files.

```bash
# Full pipeline execution
python main.py --video_path "path/to/video.mp4"

# Start from specific intermediate data
python main.py --input_detections "detections.csv"
python main.py --input_tracks "tracks.csv"

# Stop at a specific stage
python main.py --video_path "match.mp4" --stop_at cleaning

```

---

## 🗺️ Roadmap & Future Improvements

* **Handling Variations:** Support for non-30 FPS video models.
* **Calibration:** Add a calibration phase based on ball detections.
* **Feature Refinement:** Implement dual-window extraction (short and long) to better detect serves and high tosses during downtime.
* **Expanded Data:** Add training frames with more diverse ball colors and different camera angles (mid-court, heavy left side).