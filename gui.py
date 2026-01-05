import os
import sys
import threading
import yaml
import shutil
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Import utility for path handling (crucial for PyInstaller/Assets)
from src.utils import get_resource_path 

# Import pipeline stage functions
from src.ball_detector.get_ball_detections import get_ball_detections
from src.ball_detector.track_ball_detections import track_with_physics_predictive
from src.ball_detector.clean_tracking_data import clean_noise
from src.rally_predictor.extract_features import extract_features
from src.rally_predictor.rf_predictor import predict_rallies
from src.rally_predictor.predictions_handler import analyze_rally_stats, smooth_predictions
from src.visualizer.visualize_data import visualize

class VolleyballAnalyticsGUI(ctk.CTk):
    """
    A graphical user interface for the Volleyball Rally Analytics pipeline.

    This class manages the lifecycle of a video processing job, including file selection,
    parameter configuration (model version, hardware device), and real-time logging 
    within the app. It executes the processing pipeline in a background thread to 
    prevent UI freezing.

    Attributes:
        video_path (ctk.StringVar): Path to the source video file.
        output_dir (ctk.StringVar): Target directory for results.
        device (ctk.StringVar): Inference hardware ('cpu', 'cuda', etc.).
        rf_model_ver (ctk.StringVar): Version of the Random Forest model to use.
        stop_at (ctk.StringVar): Choice to terminate the pipeline early for debugging.
    """
    def __init__(self):
        super().__init__()
        self.title("Volleyball Rally Analytics")
        self.geometry("850x750")
        ctk.set_appearance_mode("dark")
        
        # Default output path set to the user's Videos folder
        default_out = os.path.join(os.path.expanduser("~"), "Videos", "volleyball")
        
        # State Variables
        self.video_path = ctk.StringVar()
        self.output_dir = ctk.StringVar(value=default_out)
        self.device = ctk.StringVar(value="cpu")
        self.rf_model_ver = ctk.StringVar(value="v3")
        self.stop_at = ctk.StringVar(value="visualization")
        self.viz_type = ctk.StringVar(value="both")
        self.keep_all = ctk.BooleanVar(value=True)
        
        self.setup_ui()

    def setup_ui(self):
        """Constructs the visual components of the application."""
        self.label = ctk.CTkLabel(self, text="Volleyball Rally Analytics", font=("Roboto", 24, "bold"))
        self.label.pack(pady=20)

        # --- File Selection Section ---
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(pady=10, padx=20, fill="x")

        # Input Video Selection
        ctk.CTkLabel(file_frame, text="Input Video:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.video_entry = ctk.CTkEntry(file_frame, textvariable=self.video_path, width=450)
        self.video_entry.grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkButton(file_frame, text="Browse Video", command=self.browse_video, width=120).grid(row=0, column=2, padx=10)

        # Output Folder Selection
        ctk.CTkLabel(file_frame, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.output_entry = ctk.CTkEntry(file_frame, textvariable=self.output_dir, width=450)
        self.output_entry.grid(row=1, column=1, padx=10, pady=5)
        ctk.CTkButton(file_frame, text="Browse Folder", command=self.browse_output, width=120).grid(row=1, column=2, padx=10)

        # --- Configuration Options Section ---
        opt_frame = ctk.CTkFrame(self)
        opt_frame.pack(pady=10, padx=20, fill="x")
        
        # Model and Device selection
        ctk.CTkLabel(opt_frame, text="RF Model:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rf_dropdown = ctk.CTkComboBox(opt_frame, values=["v1", "v2", "v3", "v4"], variable=self.rf_model_ver)
        self.rf_dropdown.grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(opt_frame, text="Device:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.device_dropdown = ctk.CTkComboBox(opt_frame, values=["cpu", "cuda", "intel:gpu"], variable=self.device)
        self.device_dropdown.grid(row=0, column=3, padx=10, pady=10)

        # Execution logic (Stop stage and Visualization mode)
        ctk.CTkLabel(opt_frame, text="Stop At:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.stop_dropdown = ctk.CTkComboBox(opt_frame, values=["detection", "tracking", "cleaning", "features", "prediction", "visualization"], variable=self.stop_at)
        self.stop_dropdown.grid(row=1, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(opt_frame, text="Viz Type:").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        self.viz_dropdown = ctk.CTkComboBox(opt_frame, values=["none", "both", "tracking", "predictions","all"], variable=self.viz_type)
        self.viz_dropdown.grid(row=1, column=3, padx=10, pady=10)
        
        self.keep_check = ctk.CTkCheckBox(opt_frame, text="Keep Intermediate Files (.csv)", variable=self.keep_all)
        self.keep_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # --- Logging Box ---
        self.log_text = ctk.CTkTextbox(self, height=250, width=750)
        self.log_text.pack(pady=20, padx=20)
        self.log_text.configure(state="disabled")

        # --- Execution Button ---
        self.run_button = ctk.CTkButton(self, text="Start Pipeline", command=self.start_pipeline_thread, fg_color="#1f538d", hover_color="#14375e", height=40, font=("Roboto", 16, "bold"))
        self.run_button.pack(pady=10)

    def browse_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if filename:
            self.video_path.set(filename)

    def browse_output(self):
        """Method to pick the base directory for outputs"""
        foldername = filedialog.askdirectory()
        if foldername:
            self.output_dir.set(foldername)
    
    def log(self, message):
        """Thread-safe logging to the UI Textbox."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_pipeline_thread(self):
        """Spawns a background thread to run the AI pipeline."""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return
        # Disable button to prevent double-execution
        self.run_button.configure(state="disabled", text="Processing...")
        thread = threading.Thread(target=self.run_pipeline)
        thread.daemon = True # Thread closes if app closes
        thread.start()

    def run_pipeline(self):
        """
        The main execution loop that sequentially calls each pipeline stage.
        Handles errors gracefully and ensures cleanup via finalize().
        """
        try:
            # 1. SETUP WORKSPACE
            out_dir = self.output_dir.get()
            table_data_dir = os.path.join(out_dir, "table_data")
            temp_dir = os.path.join(out_dir, "temp")
            video_out_dir = os.path.join(out_dir, "videos")
            for d in [table_data_dir, temp_dir, video_out_dir]: 
                os.makedirs(d, exist_ok=True)

            video_path = self.video_path.get()
            base_name = os.path.basename(video_path)
            
            # Temporary CSV path definitions
            fn_detections = os.path.join(temp_dir, f"{base_name}_detections.csv")
            fn_tracks = os.path.join(temp_dir, f"{base_name}_tracks.csv")
            fn_clean = os.path.join(temp_dir, f"{base_name}_cleaned.csv")
            fn_features = os.path.join(temp_dir, f"{base_name}_features.csv")
            fn_predictions = os.path.join(temp_dir, f"{base_name}_predictions.csv")
            final_video_output = os.path.join(video_out_dir, f"{base_name}_{self.stop_at.get()}.mp4")

            # 2. LOAD MODEL CONFIGS
            # get_resource_path is vital for finding these files when bundled in an EXE
            rf_model_config_path = get_resource_path(os.path.join("models", "rally_prediction", "config.yaml"))
            with open(rf_model_config_path, 'r') as f:
                rf_configs = yaml.safe_load(f)["models"]
            
            rf_info = rf_configs.get(self.rf_model_ver.get(), {})
            rf_path = get_resource_path(rf_info.get("path", f"models/rally_prediction/rally_predictor_rf_{self.rf_model_ver.get()}.pkl"))
            feature_params = rf_info.get("feature_extraction", {})

            # --- STAGE 1: DETECTION ---
            self.log(f"--- Running Detection on {self.device.get()} ---")
            detection_model_path = get_resource_path("models/ball_detection/vbn11_openvino_model_1")
            get_ball_detections(model_path=detection_model_path, 
                                     video_path=video_path, output_csv=fn_detections, device=self.device.get())
            if self.stop_at.get() == "detection": return self.finalize(temp_dir, table_data_dir)

            # --- STAGE 2: TRACKING ---
            self.log("--- Running Physics-based Tracking ---")
            track_with_physics_predictive(input_csv=fn_detections, output_csv=fn_tracks)
            if self.stop_at.get() == "tracking": return self.finalize(temp_dir, table_data_dir)

            # --- STAGE 3: CLEANING ---
            self.log("--- Cleaning Tracking Data ---")
            clean_noise(tracking_csv=fn_tracks, output_csv=fn_clean)
            if self.stop_at.get() == "cleaning": return self.finalize(temp_dir, table_data_dir)

            # --- STAGE 4: FEATURES ---
            self.log("--- Extracting Features ---")
            extract_features(input_csv=fn_clean, output_csv=fn_features, 
                             window_size=feature_params.get("window_size", 60),
                             interpolation_size=feature_params.get("interpolation_size", 5))
            if self.stop_at.get() == "features": return self.finalize(temp_dir, table_data_dir)

            # --- STAGE 5: PREDICTION ---
            self.log("--- Predicting Rallies ---")
            predictions = predict_rallies(rf_path=rf_path, df_path=fn_features)
            predictions = smooth_predictions(predictions)
            pd.DataFrame({'label': predictions}).to_csv(fn_predictions, index=False)
            
            clean_stats, _ = analyze_rally_stats(predictions)
            self.log(f"Stats: {clean_stats}")
            if self.stop_at.get() == "prediction": return self.finalize(temp_dir, table_data_dir)

            # --- STAGE 6: VISUALIZATION ---
            if self.viz_type.get() != "none":
                self.log(f"--- Exporting Final Video (Mode: {self.viz_type.get()}) ---")
                visualize(video_path=video_path, output_path=final_video_output,
                          tracking_csv=fn_clean, predictions_csv=fn_predictions, 
                          overlay_mode=self.viz_type.get())

            self.finalize(temp_dir, table_data_dir)
            self.log("Pipeline Finished Successfully!")
            messagebox.showinfo("Success", f"Pipeline complete!\nVideo saved to: {final_video_output}")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Pipeline Error", str(e))
        finally:
            self.run_button.configure(state="normal", text="Start Pipeline")

    def finalize(self, temp_dir, table_data_dir):
        """Moves files to permanent storage or deletes them based on user preference."""
        if self.keep_all.get() and os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                shutil.move(os.path.join(temp_dir, f), os.path.join(table_data_dir, f))
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    app = VolleyballAnalyticsGUI()
    app.mainloop()