from src.rally_predictor.rf_predictor import train_random_forest

def main():
    rf_model_path = "./models/rally_prediction/rally_predictor_rf_v4.pkl"
    feature_csvs = [
        "./output/training_data/game1_features.csv",
        "./output/training_data/game5_features.csv",
        "./output/training_data/game9_features.csv",
    ]
    label_csvs = [
        "./output/training_data/game1_rally_labels_per_frame.csv",
        "./output/training_data/game5_rally_labels_per_frame.csv",
        "./output/training_data/game9_rally_labels_per_frame.csv",
    ]

    train_random_forest(
        rf_model_path,
        feature_csvs,
        label_csvs,
        num_est=300,
        num_min_samples=50
    )
if __name__ == "__main__":
    main()