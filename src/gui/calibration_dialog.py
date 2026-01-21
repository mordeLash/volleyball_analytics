
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from src.calibration.geometry import WORLD_POINTS

class DraggablePoint:
    def __init__(self, dialog, item_id, logical_x, logical_y, short_label, coord_label, color="red", radius=6):
        self.dialog = dialog
        self.canvas = dialog.canvas
        self.id = item_id 
        self.lx = logical_x 
        self.ly = logical_y 
        self.short_label = short_label
        self.coord_label = coord_label
        self.color = color
        self.radius = radius
        self.active = True
        
        # Default show short label
        self.canvas_id = self.canvas.create_oval(0,0,0,0, fill=color, outline="white", width=2, tags="point")
        self.text_id = self.canvas.create_text(0,0, text=short_label, fill="yellow", font=("Arial", 9, "bold"), tags="point_label")
        
        self.canvas.tag_bind(self.canvas_id, "<Button-1>", self.on_click)
        self.canvas.tag_bind(self.canvas_id, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.canvas_id, "<Button-3>", self.on_right_click)

    def set_label_mode(self, show_coords):
        text = self.coord_label if show_coords else self.short_label
        self.canvas.itemconfig(self.text_id, text=text)

    def update_screen_position(self):
        sx, sy = self.dialog.to_screen(self.lx, self.ly)
        r = self.radius
        self.canvas.coords(self.canvas_id, sx - r, sy - r, sx + r, sy + r)
        self.canvas.coords(self.text_id, sx, sy - 15)
        color = self.color if self.active else "gray"
        self.canvas.itemconfig(self.canvas_id, fill=color)

    # ... (Handlers remain same)
    def on_click(self, event):
        self.start_sx = event.x
        self.start_sy = event.y

    def on_drag(self, event):
        dsx = event.x - self.start_sx
        dsy = event.y - self.start_sy
        dlx = dsx / self.dialog.zoom
        dly = dsy / self.dialog.zoom
        self.lx += dlx
        self.ly += dly
        self.start_sx = event.x
        self.start_sy = event.y
        self.dialog.redraw()

    def on_right_click(self, event):
        self.active = not self.active
        self.dialog.redraw()

class CalibrationDialog(ctk.CTkToplevel):
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.parent = parent
        self.title("Calibration - Interactive Setup")
        self.geometry("1200x900")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.video_path = video_path
        self.output_points = None
        
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.show_coords = False # State for toggle
        
        self.load_image()
        self.setup_ui()
        self.init_points()
        
        self.fit_to_window()
        self.redraw()
        
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<Shift-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Shift-B1-Motion>", self.do_pan)

    # ... (load_image, fit_to_window, transforms remain same)
    def load_image(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.destroy()
            return
        
        self.img_h, self.img_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(frame_rgb)
        self.tk_image = None

    def fit_to_window(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1: cw, ch = 800, 600
        scale_w = cw / self.img_w
        scale_h = ch / self.img_h
        self.zoom = min(scale_w, scale_h) * 0.9
        vw = self.img_w * self.zoom
        vh = self.img_h * self.zoom
        self.pan_x = (cw - vw) / 2
        self.pan_y = (ch - vh) / 2
        self.redraw()

    def to_screen(self, lx, ly):
        return (lx * self.zoom + self.pan_x, ly * self.zoom + self.pan_y)

    def to_logical(self, sx, sy):
        return ((sx - self.pan_x) / self.zoom, (sy - self.pan_y) / self.zoom)

    def start_pan(self, event):
        self._pan_start = (event.x, event.y)

    def do_pan(self, event):
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self._pan_start = (event.x, event.y)
        self.redraw()

    def on_zoom(self, event):
        scale = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        mx, my = event.x, event.y
        lx, ly = self.to_logical(mx, my)
        self.zoom *= scale
        self.pan_x = mx - lx * self.zoom
        self.pan_y = my - ly * self.zoom
        self.redraw()

    def setup_ui(self):
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        frm = ctk.CTkFrame(self)
        frm.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        help_text = "Drag Points | Scroll to Zoom | Middle-Click to Pan | Right-Click to Toggle"
        ctk.CTkLabel(frm, text=help_text).pack(side="left", padx=10)
        
        ctk.CTkSwitch(frm, text="Show Coordinates", command=self.toggle_labels).pack(side="left", padx=20)
        
        ctk.CTkButton(frm, text="Done", command=self.on_done, fg_color="green").pack(side="right", padx=10)
        ctk.CTkButton(frm, text="Reset View", command=self.fit_to_window, fg_color="#444").pack(side="right", padx=5)
        ctk.CTkButton(frm, text="Reset Points", command=self.reset_all, fg_color="red").pack(side="right", padx=5)

    def toggle_labels(self):
        self.show_coords = not self.show_coords
        for p in self.point_objs:
            p.set_label_mode(self.show_coords)

    def redraw(self):
        self.canvas.delete("img_bg")
        self.canvas.delete("skel_line")
        
        vw = int(self.img_w * self.zoom)
        vh = int(self.img_h * self.zoom)
        
        if vw > 0 and vh > 0:
            resized = self.pil_image.resize((vw, vh), Image.Resampling.NEAREST)
            self.tk_image = ImageTk.PhotoImage(resized, master=self.parent)
            self.canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.tk_image, tags="img_bg")
            self.canvas.tag_lower("img_bg")

        self.draw_skeleton()
        
        for p in self.point_objs:
            p.update_screen_position()
            self.canvas.tag_raise(p.canvas_id)
            self.canvas.tag_raise(p.text_id)

    def init_points(self):
        w, h = self.img_w, self.img_h
        margin = w * 0.1
        pts = {}
        
        tl = (margin + w*0.1, h*0.2)
        tr = (w - margin - w*0.1, h*0.2)
        bl = (margin, h - margin)
        br = (w - margin, h - margin)
        
        def interp(p1, p2, t): return (p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t)
        
        pts[0], pts[1], pts[2], pts[3] = tl, tr, br, bl 
        pts[6] = interp(tl, bl, 0.5) 
        pts[7] = interp(tr, br, 0.5) 
        pts[4] = interp(pts[6], tl, 0.33) 
        pts[5] = interp(pts[7], tr, 0.33) 
        pts[9] = interp(pts[6], bl, 0.33) 
        pts[8] = interp(pts[7], br, 0.33) 
        
        net_h = h * 0.25
        cl, cr = pts[6], pts[7]
        pts[10] = (cl[0], cl[1] - net_h) 
        pts[11] = (cr[0], cr[1] - net_h) 
        pts[12] = (cl[0], cl[1] - net_h * 0.4) 
        pts[13] = (cr[0], cr[1] - net_h * 0.4) 
        pts[14] = (pts[10][0], pts[10][1] - 50) 
        pts[15] = (pts[11][0], pts[11][1] - 50) 
        
        self.point_objs = []
        
        # User Defined Short Codes
        # T=Top(Far), B=Bottom(Near), L=Left, R=Right, E=End, A=Attack, M=Mid, N=Net, Ant=Antenna
        short_labels = [
            "T-L-E", "T-R-E", "B-R-E", "B-L-E", 
            "T-L-A", "T-R-A", "M-L", "M-R", 
            "B-R-A", "B-L-A", 
            "N-T-L", "N-T-R", "N-B-L", "N-B-R", 
            "N-A-L", "N-A-R"
        ]
        
        colors = ["red"] * 10 + ["cyan"] * 6
        
        for i in range(16):
            # Format real world coord label
            wp = WORLD_POINTS[i]
            coord_str = f"({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})"
            p = DraggablePoint(self, i, pts[i][0], pts[i][1], short_labels[i], coord_str, color=colors[i])
            self.point_objs.append(p)


    def reset_all(self):
        self.canvas.delete("all")
        self.init_points()
        self.fit_to_window()

    def draw_skeleton(self):
        # Improved Connectivity
        # Court Side Lines
        side_l = [0, 4, 6, 9, 3]
        side_r = [1, 5, 7, 8, 2]
        
        # Cross Lines
        cross = [(0,1), (4,5), (6,7), (9,8), (3,2)]
        
        # Net Box
        net_box = [(10,11), (11,13), (13,12), (12,10)]
        # Pole
        poles = [(12,6), (13,7)]
        # Antennae
        antennas = [(14,10), (14,12), (15,11), (15,13)]
        
        # Helper to draw
        def draw_lines(pairs, color):
            for idx1, idx2 in pairs:
                p1 = self.point_objs[idx1]
                p2 = self.point_objs[idx2]
                if p1.active and p2.active:
                    sx1, sy1 = self.to_screen(p1.lx, p1.ly)
                    sx2, sy2 = self.to_screen(p2.lx, p2.ly)
                    self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=2, tags="skel_line")

        # Process side lines (chains)
        side_pairs = []
        for i in range(len(side_l)-1): side_pairs.append((side_l[i], side_l[i+1]))
        for i in range(len(side_r)-1): side_pairs.append((side_r[i], side_r[i+1]))
        draw_lines(side_pairs, "blue")
        
        # Process others
        draw_lines(cross, "blue")
        draw_lines(net_box, "cyan") # Keep net slightly different maybe?
        draw_lines(poles, "cyan")
        draw_lines(antennas, "orange") # Requested change

        self.canvas.tag_raise("point") # Keep points on top

    def on_done(self):
        res = {}
        count = 0
        for p in self.point_objs:
            if p.active:
                res[p.id] = (p.lx, p.ly)
                count += 1
        
        if count < 6:
            tk.messagebox.showerror("Error", "Need at least 6 points.")
            return

        self.output_points = res
        self.destroy()
