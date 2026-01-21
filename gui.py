import tkinter as tk
import customtkinter as ctk

if __name__ == "__main__":
    # Initialize the main app first (The one and only Root)
    # We delay showing it until splash is done
    from src.gui.app import VolleyballAnalyticsGUI
    
    app = VolleyballAnalyticsGUI()
    app.withdraw() # Hide main window initially

    # Create Splash as a Toplevel of the app
    # (Since app is ctk, we can use ctk.CTkToplevel or tk.Toplevel)
    splash = tk.Toplevel(app)
    splash.title("Loading...")
    
    # Center splash
    width, height = 400, 200
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")
    
    splash.overrideredirect(True)
    splash.configure(bg="#1f538d")
    
    label = tk.Label(splash, text="Volleyball Playtime Extractor\nLoading Pipeline...", 
                     fg="white", bg="#1f538d", font=("Roboto", 16, "bold"))
    label.pack(expand=True)
    
    splash.update()

    # Transition function
    def transition():
        splash.destroy()
        app.deiconify() # Show main window
    
    # Schedule transition
    app.after(1000, transition)
    app.mainloop()