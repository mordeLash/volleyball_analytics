import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

CONTS = ['mean_x', 'mean_y', 'std_x', 'std_y', 'mean_velocity', 'max_velocity', 'std_velocity',
             'mean_acceleration', 'mean_angle_change', 'mean_y_dominance', 'mean_accel_jitter', 'mean_energy', 'y_range_window', 'mean_vis', 'vis_std', 'is_low_window']
dep = "label"

def train_random_forest(rf_path, df_paths, labels_paths, num_est=200, num_min_samples=30):
    """
    df_paths: List of strings (paths to feature CSVs)
    labels_paths: List of strings (paths to label CSVs)
    """
    
    combined_dfs = []
    
    # Iterate through pairs of feature and label files
    for df_p, labels_p in zip(df_paths, labels_paths):
        curr_features = pd.read_csv(df_p)
        curr_labels = pd.read_csv(labels_p)
        
        # Merge labels into the features dataframe
        curr_features['label'] = curr_labels['label']
        
        # Append the actual data directly
        combined_dfs.append(curr_features)

    # Concatenate all dataframes into one
    features_df = pd.concat(combined_dfs, ignore_index=True)
    features_df = features_df.fillna(0)

    # Split data 
    # Note: If your data is no longer time-series dependent, 
    # you might consider shuffle=True for a more representative split.
    trn_df, val_df = train_test_split(features_df, test_size=0.25, shuffle=False)
    
    trn_df = trn_df.copy()
    val_df = val_df.copy()

    # Convert labels to codes
    # Use the full dataset to ensure all categories are captured in the mapping
    features_df[dep] = features_df[dep].astype('category')
    class_names = features_df[dep].cat.categories
    
    trn_df[dep] = pd.Categorical(trn_df[dep], categories=class_names).codes
    val_df[dep] = pd.Categorical(val_df[dep], categories=class_names).codes
    
    def xs_y(df):
        xs = df[CONTS].copy()
        return xs, df[dep] if dep in df else None

    trn_xs, trn_y = xs_y(trn_df)
    val_xs, val_y = xs_y(val_df)

    # Initialize and Train
    rf = RandomForestClassifier(n_estimators=num_est, min_samples_leaf=num_min_samples, n_jobs=-1)
    rf.fit(trn_xs, trn_y)
    
    # Evaluation
    preds = rf.predict(val_xs)
    print(f"\n--- Model Evaluation ---")
    print(f"Validation MAE: {mean_absolute_error(val_y, preds):.4f}")
    print(f"Validation Accuracy: {accuracy_score(val_y, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(val_y, preds, target_names=[str(c) for c in class_names]))
    
    # --- Visualization: Confusion Matrix & Feature Importance ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    cm = confusion_matrix(val_y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title("Confusion Matrix")

    importances = pd.Series(rf.feature_importances_, index=CONTS).sort_values(ascending=True)
    importances.tail(20).plot(kind='barh', ax=ax[1])
    ax[1].set_title("Top Feature Importances")
    
    plt.tight_layout()
    plt.show()
    
    # Save the model
    joblib.dump(rf, rf_path)
    print(f"Model saved to {rf_path}")

def predict_rallies(rf_path, df_path):
    rf_model = joblib.load(rf_path)
    features_df = pd.read_csv(df_path)
    features_df = features_df.fillna(0)
    
    # Ensure we only use the features the model was trained on
    predictions = rf_model.predict(features_df[CONTS])
    return predictions