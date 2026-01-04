import pandas as pd
import os
import argparse
import shutil
import sys
import yaml
# 1. Import the utility function
from src.utils import get_resource_path

from src.ball_detector.get_ball_detections import get_ball_detections, get_ball_detections_fast
from src.ball_detector.track_ball_detections import track_with_physics_predictive
from src.ball_detector.clean_tracking_data import clean_noise
from src.rally_predictor.extract_features import extract_features
from src.rally_predictor.rf_predictor import predict_rallies
from src.rally_predictor.predictions_handler import analyze_rally_stats, smooth_predictions
from src.visualizer.visualize_data import visualize

def main():
    STAGES = ["detection", "tracking", "cleaning", "features", "prediction", "visualization"]

    parser = argparse.ArgumentParser(description="Volleyball Rally Analytics Pipeline")

    # --- Entry Points ---
    parser.add_argument("--video_path", type=str, help="Path to input video (required for detection or visualization)")
    parser.add_argument("--input_detections", type=str, help="Path to existing detections.csv to start from Tracking")
    parser.add_argument("--input_tracks", type=str, help="Path to existing tracks.csv to start from Cleaning")
    parser.add_argument("--input_clean", type=str, help="Path to existing cleaned.csv to start from Features/Prediction")

    # --- Config ---
    # 2. UPDATED: Wrap the default detection model path
    parser.add_argument("--detection_model", type=str, 
                        default=get_resource_path("models/ball_detection/vbn11_openvino_model_1"))
    parser.add_argument("--rf_model", type=str, default="v3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stop_at", type=str, choices=STAGES, default="visualization")
    parser.add_argument("--visualize_early", action="store_true")
    parser.add_argument("--visualize_type", type=str, default="none")
    parser.add_argument("--keep_all", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()

    # --- Directory Setup (DO NOT wrap these; they are for user output) ---
    table_data_dir = os.path.join(args.output_dir, "table_data")
    temp_dir = os.path.join(args.output_dir, "temp")
    video_out_dir = os.path.join(args.output_dir, "videos")
    for d in [table_data_dir, temp_dir, video_out_dir]: os.makedirs(d, exist_ok=True)

    # --- Base Name Logic ---
    input_source = args.video_path or args.input_detections or args.input_tracks or args.input_clean
    if not input_source:
        print("Error: You must provide at least one input (--video_path, --input_detections, etc.)")
        sys.exit(1)
    
    base_name = os.path.basename(input_source)
    
    # --- Internal Path Definitions ---
    fn_detections = args.input_detections or os.path.join(temp_dir, f"{base_name}_detections.csv")
    fn_tracks = args.input_tracks or os.path.join(temp_dir, f"{base_name}_tracks.csv")
    fn_clean = args.input_clean or os.path.join(temp_dir, f"{base_name}_cleaned.csv")
    fn_features = os.path.join(temp_dir, f"{base_name}_features.csv")
    fn_predictions = os.path.join(temp_dir, f"{base_name}_predictions.csv")
    final_video_output = os.path.join(video_out_dir, f"{base_name}_{args.stop_at}.mp4")

    # --- Load RF Model Config ---
    # 3. UPDATED: Wrap the Config YAML path
    rf_model_config_path = get_resource_path(os.path.join("models", "rally_prediction", "config.yaml"))
    
    with open(rf_model_config_path, 'r') as f:
        rf_configs = yaml.safe_load(f)["models"]

    if args.rf_model in rf_configs:
        rf_model_info = rf_configs[args.rf_model]
        
        # 4. UPDATED: Wrap the specific RF model path
        default_rel_path = f"models/rally_prediction/{args.rf_model}.pkl"
        rf_path = get_resource_path(rf_model_info.get("path", default_rel_path))
        
        feature_params = rf_model_info.get("feature_extraction", {})

    # ... [Rest of the logic remains the same] ...

    # --- Helper: Should we skip a step? ---
    def is_already_provided(file_path, stage_name):
        if file_path and os.path.exists(file_path) and not file_path.startswith(temp_dir):
            print(f"--- Skipping {stage_name}: Input provided at {file_path} ---")
            return True
        return False

    def finalize_run():
        if args.keep_all:
            for f in [fn_detections, fn_tracks, fn_clean, fn_predictions, fn_features]:
                if os.path.exists(f) and f.startswith(temp_dir):
                    shutil.move(f, os.path.join(table_data_dir, os.path.basename(f)))
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        print("Done!")

    def should_stop(current_stage, tracking_file=None, predictions_file=None):
        if args.stop_at == current_stage:
            if args.visualize_early and args.video_path:
                print(f"--- Running Early Visualization for {current_stage} ---")
                visualize(video_path=args.video_path, output_path=final_video_output,
                          tracking_csv=tracking_file, predictions_csv=predictions_file, overlay_mode="both")
            print(f"\n--- Stopping early after {current_stage} stage ---")
            finalize_run()
            sys.exit(0)

    # 1. Detection
    if not is_already_provided(args.input_detections, "detection") and \
       not is_already_provided(args.input_tracks, "detection") and \
       not is_already_provided(args.input_clean, "detection"):
        if not args.video_path: 
            print("Error: Video path required for detection."); sys.exit(1)
        print(f"--- Running Detection ---")
        get_ball_detections_fast(model_path=args.detection_model, video_path=args.video_path, output_csv=fn_detections, device=args.device)
    should_stop("detection", tracking_file=fn_detections)

    # 2. Tracking
    if not is_already_provided(args.input_tracks, "tracking") and \
       not is_already_provided(args.input_clean, "tracking"):
        print("--- Running Physics-based Tracking ---")
        track_with_physics_predictive(input_csv=fn_detections, output_csv=fn_tracks)
    should_stop("tracking", tracking_file=fn_tracks)

    # 3. Cleaning
    if not is_already_provided(args.input_clean, "cleaning"):
        print("--- Cleaning Tracking Data ---")
        clean_noise(tracking_csv=fn_tracks, output_csv=fn_clean)
    should_stop("cleaning", tracking_file=fn_clean)

    # 4. Features
    print("--- Extracting Features ---")
    extract_features(input_csv=fn_clean, output_csv=fn_features,window_size=feature_params.get("window_size",60),
                     interpolation_size=feature_params.get("interpolation_size",5))
    should_stop("features", tracking_file=fn_features)

    # 5. Prediction
    print("--- Predicting Rallies ---")
    predictions = predict_rallies(rf_path=rf_path, df_path=fn_features)
    predictions = smooth_predictions(predictions)
    pd.DataFrame({'label': predictions}).to_csv(fn_predictions, index=False)
    
    clean_stats, _ = analyze_rally_stats(predictions)
    print(f"Analysis Complete: {clean_stats}")
    should_stop("prediction", tracking_file=fn_clean, predictions_file=fn_predictions)

    # 6. Full Visualization
    if args.video_path:
        print(f"--- Exporting Final Rallies to {video_out_dir} ---")
        visualize(video_path=args.video_path, output_path=final_video_output,
                  tracking_csv=fn_clean, predictions_csv=fn_predictions, overlay_mode=args.visualize_type)
    else:
        print("--- Skipping Visualization: No video_path provided ---")
    
    finalize_run()

if __name__ == "__main__":
    main()