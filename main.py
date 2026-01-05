import pandas as pd
import os
import argparse
import shutil
import sys
import yaml

# Utility for relative-to-absolute path resolution
from src.utils import get_resource_path

# Pipeline stage imports
from src.ball_detector.get_ball_detections import get_ball_detections
from src.ball_detector.track_ball_detections import track_with_physics_predictive
from src.ball_detector.clean_tracking_data import clean_noise
from src.rally_predictor.extract_features import extract_features
from src.rally_predictor.rf_predictor import predict_rallies
from src.rally_predictor.predictions_handler import analyze_rally_stats, smooth_predictions
from src.visualizer.visualize_data import visualize

def main():
    """
    Main entry point for the Volleyball Rally Analytics CLI.

    This script parses command-line arguments to execute the full video analysis 
    pipeline or specific subsets of it. It manages data flow between stages using 
    temporary CSV files and handles directory setup for final outputs.

    Functionality:
    - Modular Execution: Start at any stage using --input_detections, --input_tracks, etc.
    - Early Exit: Use --stop_at to terminate after a specific stage for debugging.
    - Resource Management: Automatically resolves paths for AI models and configs.
    - Post-processing: Cleans up temporary files and organizes final results.
    """
    STAGES = ["detection", "tracking", "cleaning", "features", "prediction", "visualization"]

    parser = argparse.ArgumentParser(description="Volleyball Rally Analytics Pipeline")

    # --- INPUT ARGUMENTS ---
    # These allow the user to provide a video or a CSV from a previous run
    parser.add_argument("--video_path", type=str, help="Path to input video.")
    parser.add_argument("--input_detections", type=str, help="Skip detection; start from an existing detections CSV.")
    parser.add_argument("--input_tracks", type=str, help="Skip tracking; start from an existing tracks CSV.")
    parser.add_argument("--input_clean", type=str, help="Skip cleaning; start from an existing cleaned CSV.")

    # --- CONFIGURATION ARGUMENTS ---
    # Default paths are wrapped in get_resource_path for portability
    parser.add_argument("--detection_model", type=str, 
                        default=get_resource_path("models/ball_detection/vbn11_openvino_model_1"),
                        help="Path to the ball detection model.")
    parser.add_argument("--rf_model", type=str, default="v3", help="Version of the RF model (v1-v4).")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, intel:gpu).")
    parser.add_argument("--stop_at", type=str, choices=STAGES, default="visualization", 
                        help="Stage after which to terminate execution.")
    parser.add_argument("--visualize_early", action="store_true", help="Render video even if stopping early.")
    parser.add_argument("--visualize_type", type=str, default="none", help="Visual overlay style.")
    parser.add_argument("--keep_all", action="store_true", help="Move intermediate CSVs to output directory.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Root directory for outputs.")

    args = parser.parse_args()

    # --- WORKSPACE INITIALIZATION ---
    table_data_dir = os.path.join(args.output_dir, "table_data")
    temp_dir = os.path.join(args.output_dir, "temp")
    video_out_dir = os.path.join(args.output_dir, "videos")
    for d in [table_data_dir, temp_dir, video_out_dir]: 
        os.makedirs(d, exist_ok=True)

    # Determine base filename for output generation
    input_source = args.video_path or args.input_detections or args.input_tracks or args.input_clean
    if not input_source:
        print("Error: You must provide at least one input (--video_path, --input_detections, etc.)")
        sys.exit(1)
    
    base_name = os.path.basename(input_source)
    
    # Internal CSV naming conventions
    fn_detections = args.input_detections or os.path.join(temp_dir, f"{base_name}_detections.csv")
    fn_tracks = args.input_tracks or os.path.join(temp_dir, f"{base_name}_tracks.csv")
    fn_clean = args.input_clean or os.path.join(temp_dir, f"{base_name}_cleaned.csv")
    fn_features = os.path.join(temp_dir, f"{base_name}_features.csv")
    fn_predictions = os.path.join(temp_dir, f"{base_name}_predictions.csv")
    final_video_output = os.path.join(video_out_dir, f"{base_name}_{args.stop_at}.mp4")

    # --- MODEL CONFIGURATION ---
    rf_model_config_path = get_resource_path(os.path.join("models", "rally_prediction", "config.yaml"))
    with open(rf_model_config_path, 'r') as f:
        rf_configs = yaml.safe_load(f)["models"]

    if args.rf_model in rf_configs:
        rf_model_info = rf_configs[args.rf_model]
        # Resolve specific pkl file path
        default_rel_path = f"models/rally_prediction/{args.rf_model}.pkl"
        rf_path = get_resource_path(rf_model_info.get("path", default_rel_path))
        feature_params = rf_model_info.get("feature_extraction", {})

    # --- PIPELINE UTILITIES ---
    def is_already_provided(file_path, stage_name):
        """Checks if a stage can be skipped based on user-provided input files."""
        if file_path and os.path.exists(file_path) and not file_path.startswith(temp_dir):
            print(f"--- Skipping {stage_name}: Input provided at {file_path} ---")
            return True
        return False

    def finalize_run():
        """Handles file cleanup and moves persistent data to output folders."""
        if args.keep_all:
            for f in [fn_detections, fn_tracks, fn_clean, fn_predictions, fn_features]:
                if os.path.exists(f) and f.startswith(temp_dir):
                    shutil.move(f, os.path.join(table_data_dir, os.path.basename(f)))
        if os.path.exists(temp_dir): 
            shutil.rmtree(temp_dir)
        print("Done!")

    def should_stop(current_stage, tracking_file=None, predictions_file=None):
        """Checks if the user requested to terminate the pipeline at the current stage."""
        if args.stop_at == current_stage:
            if args.visualize_early and args.video_path:
                print(f"--- Running Early Visualization for {current_stage} ---")
                visualize(video_path=args.video_path, output_path=final_video_output,
                          tracking_csv=tracking_file, predictions_csv=predictions_file, overlay_mode="both")
            print(f"\n--- Stopping early after {current_stage} stage ---")
            finalize_run()
            sys.exit(0)

    # --- PIPELINE EXECUTION ---

    # 1. Detection: YOLO model identifies ball in frames
    if not is_already_provided(args.input_detections, "detection") and \
       not is_already_provided(args.input_tracks, "detection") and \
       not is_already_provided(args.input_clean, "detection"):
        if not args.video_path: 
            print("Error: Video path required for detection."); sys.exit(1)
        print(f"--- Running Detection ---")
        get_ball_detections(model_path=args.detection_model, video_path=args.video_path, 
                            output_csv=fn_detections, device=args.device)
    should_stop("detection", tracking_file=fn_detections)

    # 2. Tracking: Connects detections into temporal paths
    if not is_already_provided(args.input_tracks, "tracking") and \
       not is_already_provided(args.input_clean, "tracking"):
        print("--- Running Physics-based Tracking ---")
        track_with_physics_predictive(input_csv=fn_detections, output_csv=fn_tracks)
    should_stop("tracking", tracking_file=fn_tracks)

    # 3. Cleaning: Filters out static noise and resolve overlaps
    if not is_already_provided(args.input_clean, "cleaning"):
        print("--- Cleaning Tracking Data ---")
        clean_noise(tracking_csv=fn_tracks, output_csv=fn_clean)
    should_stop("cleaning", tracking_file=fn_clean)

    # 4. Features: Calculate kinematics for RF classification
    print("--- Extracting Features ---")
    extract_features(input_csv=fn_clean, output_csv=fn_features,
                     window_size=feature_params.get("window_size", 60),
                     interpolation_size=feature_params.get("interpolation_size", 5))
    should_stop("features", tracking_file=fn_features)

    # 5. Prediction: RF model classifies Rally vs. Downtime
    print("--- Predicting Rallies ---")
    predictions = predict_rallies(rf_path=rf_path, df_path=fn_features)
    predictions = smooth_predictions(predictions)
    pd.DataFrame({'label': predictions}).to_csv(fn_predictions, index=False)
    
    clean_stats, _ = analyze_rally_stats(predictions)
    print(f"Analysis Complete: {clean_stats}")
    should_stop("prediction", tracking_file=fn_clean, predictions_file=fn_predictions)

    # 6. Full Visualization: Render final MP4 with overlays
    if args.video_path:
        print(f"--- Exporting Final Rallies to {video_out_dir} ---")
        visualize(video_path=args.video_path, output_path=final_video_output,
                  tracking_csv=fn_clean, predictions_csv=fn_predictions, 
                  overlay_mode=args.visualize_type)
    else:
        print("--- Skipping Visualization: No video_path provided ---")
    
    finalize_run()

if __name__ == "__main__":
    main()