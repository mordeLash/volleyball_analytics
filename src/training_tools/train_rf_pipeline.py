from tqdm import tqdm
import os
import pickle
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

from src.rally_predictor.rf_predictor import train_random_forest
from src.ball_detector.track_ball_detections import track_with_physics_predictive_v2 as track_with_physics_predictive
from src.ball_detector.clean_tracking_data import clean_noise_v2 as clean_noise
from src.ball_detector.calibrate_detections import calibrate_to_relative_space
from src.rally_predictor.extract_features import extract_features_v7 as extract_features
from src.gui.calibration_dialog import CalibrationDialog
from src.calibration.geometry import compute_camera_matrix

def main():
    """
    Main execution script to train the Rally Prediction model.

    This script acts as the configuration layer for the training process. It 
    specifies the location of the final model artifact, the collection of 
    engineered feature files, and their corresponding manual ground-truth labels.

    Hyperparameters:
        num_est (300): We use 300 trees to ensure a stable majority vote 
            across diverse game scenarios.
        num_min_samples (50): By requiring 50 samples per leaf, we force the 
            model to learn general trends of a 'rally' rather than memorizing 
            specific frames, reducing overfitting.
    """
    # 1. PATH CONFIGURATION
    # The output filename for our trained Random Forest model (version 5)
    rf_model_path = "./models/rally_prediction/rally_predictor_rf_v6.pkl"
    
    # 2. DATASET DEFINITION
    # We include data from three different games to provide the model with 
    # a diverse set of examples (different serving styles, camera angles, etc.)
# Define your game names
    data_names = ["game1_set1", 
                 "game5_set1",
                 "game7_set1", 
                 "game9_c"
    ]

    # Define the standard file suffixes you need for every game
    file_suffixes = [
        "detections.csv",
        "tracks.csv",
        "cleaned.csv",
        "calibrated.csv",
        "features.csv"
    ]

    # Automatically generate the list of lists
    base_path = "./output/training_data"

    data_csvs = [
        [f"{base_path}/{name}_{suffix}" for suffix in file_suffixes]
        for name in data_names
    ]
    feature_csvs = []
    for name, data_csv in tqdm(zip(data_names, data_csvs), total=len(data_names)):
        track_with_physics_predictive(data_csv[0],data_csv[1])
        clean_noise(data_csv[1],data_csv[2])
        
        # Load Calibration
        calib_file = f"{base_path}/calibrations/{name}_calibration.pkl"
        calib_data = None
        if os.path.exists(calib_file):
            with open(calib_file, 'rb') as f:
                calib_data = pickle.load(f)
        else:
             print(f"Warning: Calibration missing for {name}")
             # Interactive Fallback
             root = ctk.CTk()
             root.withdraw()
             
             print(f"Please select the video file for {name}...")
             video_path = filedialog.askopenfilename(
                 title=f"Select Video for {name}",
                 filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
             )
             
             if video_path:
                 # Launch Calibration Dialog
                 dialog = CalibrationDialog(root, video_path)
                 # We need to show root slightly or just wait? 
                 # Dialog is Toplevel, so it needs root to exist.
                 # wait_window blocks until dialog is destroyed.
                 root.wait_window(dialog)
                 
                 if dialog.output_points:
                     try:
                         print("Computing Camera Matrix...")
                         calib_data = compute_camera_matrix(dialog.output_points)
                         
                         # Ensure dir exists
                         os.makedirs(os.path.dirname(calib_file), exist_ok=True)
                         
                         with open(calib_file, 'wb') as f:
                             pickle.dump(calib_data, f)
                         print(f"Calibration saved to {calib_file}")
                     except Exception as e:
                         print(f"Calibration Failed: {e}")
             
             root.destroy()

        calibrate_to_relative_space(data_csv[2],data_csv[3], calibration_data=calib_data)
        extract_features(data_csv[3],data_csv[4],interpolation_size=10)
        feature_csvs.append(data_csv[4])
    

    # These must match the order of the feature_csvs list
    label_csvs = [
       "./output/training_data/game1_rally_labels_per_frame.csv",
       "./output/training_data/game5_rally_labels_per_frame.csv",
       "./output/training_data/game7_set1_rally_labels_per_frame.csv",
       "./output/training_data/game9_rally_labels_per_frame.csv",
    ]

    # 3. MODEL TRAINING
    # Launch the multi-game training pipeline
    train_random_forest(
        rf_model_path,
        feature_csvs,
        label_csvs,
        num_est=300,            # Increased from default for better stability
        num_min_samples=70      # Increased for stronger regularization
    )

if __name__ == "__main__":
    main()