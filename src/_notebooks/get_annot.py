import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import tkinter as tk
import csv
import os

# Configuration
IMAGE_PATH = r"C:\Users\morde\Desktop\volleyball\raw footage\netball_imgs\net_ball1_frame_1450.jpg"# Replace with your image file
OUTPUT_FILE = 'camera_calibration_points.csv'

# Predefined calibration labels
LABELS = [
    "BE_LEFT", "BE_RIGHT",
    "BA_LEFT", "BA_RIGHT",
    "M_LEFT", "M_RIGHT",
    "TA_LEFT", "TA_RIGHT",
    "TE_LEFT", "TE_RIGHT",
    "NET_B_LEFT", "NET_B_RIGHT",
    "NET_T_LEFT", "NET_T_RIGHT",
    "ANT_TOP LEFT", "ANT_T_RIGHT"
]

class LabelSelector(tk.Toplevel):
    def __init__(self, parent, labels):
        super().__init__(parent)
        self.title("Select Calibration Point")
        self.result = None
        
        frame = tk.Frame(self)
        frame.pack(padx=15, pady=15)
        
        # Listbox for selection
        self.listbox = tk.Listbox(frame, height=len(labels), width=35, font=("Arial", 10))
        for label in labels:
            self.listbox.insert(tk.END, label)
        self.listbox.pack(side=tk.LEFT)
        
        self.listbox.bind('<Double-1>', lambda x: self.on_select())
        
        btn = tk.Button(self, text="Confirm Label", command=self.on_select)
        btn.pack(pady=10)

        self.grab_set()
        self.wait_window()

    def on_select(self):
        selection = self.listbox.curselection()
        if selection:
            self.result = self.listbox.get(selection[0])
        self.destroy()

class ImageAnnotator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = [] # List of tuples (name, x, y)
        self.last_clicked_coords = None
        
        # Plot Setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        try:
            self.img = plt.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
            
        self.ax.imshow(self.img)
        self.ax.set_title("Calibration Tool\n1. Zoom/Pan | 2. Click Point | 3. Press 'n' to Assign Label")

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        
        plt.show()
        self.save_to_csv()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.last_clicked_coords = (event.xdata, event.ydata)
        
        # Temporary marker for current focus
        if hasattr(self, 'temp_point'):
            self.temp_point[0].remove()
        self.temp_point = self.ax.plot(event.xdata, event.ydata, 'yx', markersize=12, markeredgewidth=2)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'n' and self.last_clicked_coords:
            self.prompt_selection()

    def prompt_selection(self):
        root = tk.Tk()
        root.withdraw()
        dialog = LabelSelector(root, LABELS)
        choice = dialog.result
        root.destroy()

        if choice:
            x, y = self.last_clicked_coords
            self.points.append((choice, round(x, 3), round(y, 3)))
            
            # Permanent visual marker
            self.ax.plot(x, y, 'ro', markersize=6)
            self.ax.text(x, y, f' {choice}', color='black', fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            self.fig.canvas.draw()

    def save_to_csv(self):
        if not self.points:
            print("No points selected. Nothing saved.")
            return

        file_exists = os.path.isfile(OUTPUT_FILE)
        
        with open(OUTPUT_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header only if the file is new
            if not file_exists:
                writer.writerow(['name', 'x', 'y'])
            
            for pt in self.points:
                writer.writerow(pt)
        
        print(f"Calibration data saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        ImageAnnotator(IMAGE_PATH)
    else:
        print(f"Error: Image '{IMAGE_PATH}' not found in the current directory.")