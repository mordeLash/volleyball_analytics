# src/rally_predictor/rf_predictor.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error, accuracy_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from .predictions_handler import smooth_predictions, get_error_streaks

# List of continuous features used as input for the model
CONTS = ['mean_x', 'mean_y', 'std_x', 'std_y', 'mean_velocity', 'max_velocity', 'std_velocity',
         'mean_acceleration', 'mean_angle_change', 'mean_y_dominance', 'mean_accel_jitter', 
         'mean_energy', 'y_range_window', 'mean_vis', 'vis_std', 'is_low_window']

CONTS_V2 = [
    # Past Window Statistics (The "Approach")
    'prev_mean_cy', 'prev_std_cy', 'prev_max_cy',
    'prev_mean_cx', 'prev_std_cx', 'prev_max_cx',
    'prev_mean_velocity', 'prev_std_velocity', 'prev_max_velocity',
    'prev_mean_acceleration', 'prev_mean_energy', 'prev_max_energy',
    'prev_mean_conf', 'prev_std_conf', 'prev_max_conf',
    'prev_y_range',
    
    # Future Window Statistics (The "Result")
    'next_mean_cy', 'next_std_cy', 'next_max_cy',
    'next_mean_cx', 'next_std_cx', 'next_max_cx',
    'next_mean_velocity', 'next_std_velocity', 'next_max_velocity',
    'next_mean_acceleration', 'next_mean_energy', 'next_max_energy',
    'next_mean_conf', 'next_std_conf', 'next_max_conf',
    'next_y_range',
]
dep = "label"

def train_random_forest(rf_path, df_paths, labels_paths, num_est=200, num_min_samples=30):
    combined_dfs = []
    dep = "label"
    
    # 1. DATA AGGREGATION
    for df_p, labels_p in zip(df_paths, labels_paths):
        curr_features = pd.read_csv(df_p)
        curr_labels = pd.read_csv(labels_p)
        curr_features[dep] = curr_labels[dep]
        combined_dfs.append(curr_features)

    features_df = pd.concat(combined_dfs, ignore_index=True)
    features_df = features_df.fillna(0)

    # --- AUTOMATIC FEATURE SELECTION ---
    # We take all columns EXCEPT the label and any non-numeric metadata
    # This automatically creates your 'CONTS' list
    feature_cols = [c for c in features_df.columns if c != dep and not c.startswith('unnamed')]
    print(f"Automatically detected {len(feature_cols)} features for training.")

    # 2. PRE-PROCESSING
    features_df[dep] = features_df[dep].astype('category')
    class_names = features_df[dep].cat.categories
    
    # Split before encoding to maintain chronological order for streak validation
    trn_df, val_df = train_test_split(features_df, test_size=0.25, shuffle=False)
    
    trn_y = trn_df[dep].cat.codes
    val_y = val_df[dep].cat.codes
    trn_xs = trn_df[feature_cols]
    val_xs = val_df[feature_cols]

    # 3. TRAINING
    rf = RandomForestClassifier(n_estimators=num_est, min_samples_leaf=num_min_samples, n_jobs=-1)
    
    rf.fit(trn_xs, trn_y)
    
    # 4. EVALUATION
    preds = rf.predict(val_xs)
    s_preds = smooth_predictions(preds)
    
    # 4.5 DETAILED STREAK ANALYSIS
    for label, current_preds in [("RAW", preds), ("SMOOTHED", s_preds)]:
        streaks = get_error_streaks(val_y, current_preds)
        print(f"\n--- Sequence Error Analysis ({label}) ---")
        print(f"Longest False Rally (FP):    {streaks['False Rally']} frames ({streaks['False Rally']/30:.2f}s)")
        print(f"Longest False Downtime (FN): {streaks['False Downtime']} frames ({streaks['False Downtime']/30:.2f}s)")
        print(f"Accuracy: {accuracy_score(val_y, current_preds):.4f}")

    # 5. VISUALIZATION
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ConfusionMatrixDisplay.from_predictions(val_y, preds, display_labels=class_names, ax=ax[0], cmap='Blues')
    ax[0].set_title("Confusion Matrix (Raw)")

    ConfusionMatrixDisplay.from_predictions(val_y, s_preds, display_labels=class_names, ax=ax[1], cmap='Greens')
    ax[1].set_title("Confusion Matrix (Smooth)")

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    importances.tail(20).plot(kind='barh', ax=ax[2])
    ax[2].set_title("Top 20 Features")
    
    plt.tight_layout()
    plt.show()
    
    joblib.dump(rf, rf_path)
    print(f"Model and feature list saved to {rf_path}")

def predict_rallies(rf_path, df_path):
    """
    Loads a trained Random Forest model and predicts labels for new tracking data.

    Args:
        rf_path (str): Path to the saved .joblib model file.
        df_path (str): Path to the CSV containing features extracted from a new video.

    Returns:
        numpy.ndarray: Array of predicted class codes (e.g., [0, 1, 1, 0...]) 
            representing the state of each frame.
    """
    # Load the serialized model from disk
    rf_model = joblib.load(rf_path)
    
    # Read the features extracted from the target video
    features_df = pd.read_csv(df_path)
    features_df = features_df.fillna(0)
    feature_cols = [c for c in features_df.columns if c != dep and not c.startswith('unnamed')]
    # Ensure the feature columns match the training order defined in CONTS
    predictions = rf_model.predict(features_df[feature_cols])
    
    return predictions