# src/gui/components.py

import customtkinter as ctk
from tkinter import filedialog

class FilePicker(ctk.CTkFrame):
    """A reusable row for picking a file or directory."""
    def __init__(self, master, label_text, variable, mode="file", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.variable = variable
        self.mode = mode

        self.label = ctk.CTkLabel(self, text=label_text, width=120, anchor="w")
        self.label.pack(side="left", padx=(10, 5))

        self.entry = ctk.CTkEntry(self, textvariable=self.variable)
        self.entry.pack(side="left", padx=5, fill="x", expand=True)

        btn_text = "Browse" if mode == "file" else "Folder"
        self.button = ctk.CTkButton(self, text=btn_text, width=100, command=self.browse)
        self.button.pack(side="left", padx=(5, 10))

    def browse(self):
        if self.mode == "file":
            path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        else:
            path = filedialog.askdirectory()
        
        if path:
            self.variable.set(path)

class AdvancedOptionsFrame(ctk.CTkFrame):
    def __init__(self, master, variables, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure((1, 3), weight=1)

        # --- Row 0: RF Model & Device ---
        ctk.CTkLabel(self, text="RF Model:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(self, values=["v1", "v2", "v3", "v4"], variable=variables['rf_model_ver']).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self, text="Device:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(self, values=["cpu", "cuda", "intel:gpu"], variable=variables['device']).grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        # --- Row 1: Start At & Stop At ---
        stages = ["detection", "tracking", "cleaning", "features", "prediction", "visualization"]
        ctk.CTkLabel(self, text="Start At:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(self, values=stages, variable=variables['start_at']).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self, text="Stop At:").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(self, values=stages, variable=variables['stop_at']).grid(row=1, column=3, padx=10, pady=10, sticky="ew")

        # --- Row 2: Intermediate CSV Input (Optional) ---
        # This button triggers a file dialog for the specific starting CSV
        ctk.CTkLabel(self, text="Custom Input CSV:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.csv_entry = ctk.CTkEntry(self, textvariable=variables['input_csv_path'], placeholder_text="Required if skipping detection")
        self.csv_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(self, text="Browse CSV", width=80, command=self.browse_csv).grid(row=2, column=3, padx=10)

        # --- Row 3: Trim Settings ---
        ctk.CTkLabel(self, text="Trim Start:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(self, textvariable=variables['start_time'], placeholder_text="00:00:10").grid(row=3, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(self, text="Trim End:").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(self, textvariable=variables['end_time'], placeholder_text="00:00:45").grid(row=3, column=3, padx=10, pady=10, sticky="ew")

        # --- Row 4: Keep Files & Viz Early ---
        ctk.CTkCheckBox(self, text="Keep Intermediate Files", variable=variables['keep_all']).grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self, text="Visualize Early (on Stop)", variable=variables['viz_early']).grid(row=4, column=2, columnspan=2, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(self, text="Visualize Type", variable=variables['viz_type']).grid(row=4, column=2, columnspan=2, padx=10, pady=10, sticky="w")

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path: self.csv_entry.delete(0, 'end'); self.csv_entry.insert(0, path)