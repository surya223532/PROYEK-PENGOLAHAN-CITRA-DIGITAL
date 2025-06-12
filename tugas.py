import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Processor")
        self.root.configure(bg="#f5f5f5")
        self.root.geometry("1200x800")
        
        # Set window icon
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # Modern color scheme
        self.bg_color = "#f5f5f5"
        self.sidebar_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.text_color = "#333333"
        self.button_color = "#3498db"
        self.button_hover = "#2980b9"
        self.canvas_bg = "#ffffff"
        
        self.original_img = None
        self.processed_img = None
        self.original_photo = None
        self.processed_photo = None
        self.histogram_figures = []

        # Configure grid layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ==== Sidebar Frame ====
        self.sidebar = tk.Frame(root, bg=self.sidebar_color, padx=15, pady=20)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        # App title
        title_label = tk.Label(self.sidebar, text="Image Processor", 
                             bg=self.sidebar_color, fg="white", 
                             font=("Segoe UI", 18, "bold"))
        title_label.pack(pady=(0, 30))

        # Upload button with modern style
        self.upload_btn = tk.Button(self.sidebar, text="Upload Image", 
                                   bg=self.button_color, fg="white",
                                   font=("Segoe UI", 12), 
                                   relief=tk.FLAT, activebackground=self.button_hover,
                                   command=self.upload_image)
        self.upload_btn.pack(fill=tk.X, pady=5)
        self.upload_btn.bind("<Enter>", lambda e: self.upload_btn.config(bg=self.button_hover))
        self.upload_btn.bind("<Leave>", lambda e: self.upload_btn.config(bg=self.button_color))

        # Process selection dropdown with modern style
        tk.Label(self.sidebar, text="Select Process:", bg=self.sidebar_color, 
                fg="white", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(20, 5))
        
        self.options = [
            "Grayscale",
            "Biner (Threshold)",
            "Brightness/Contrast",
            "Operasi Logika",
            "Histogram",
            "Dilasi",
            "Edge Detection"
        ]
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground=self.canvas_bg, background=self.canvas_bg)
        
        self.option_var = tk.StringVar(value=self.options[0])
        self.option_menu = ttk.Combobox(self.sidebar, textvariable=self.option_var, 
                                      values=self.options, state="readonly", 
                                      font=("Segoe UI", 10))
        self.option_menu.pack(fill=tk.X, pady=5)
        self.option_menu.bind("<<ComboboxSelected>>", self.update_parameters)

        # Parameters frame
        self.param_frame = tk.Frame(self.sidebar, bg=self.sidebar_color)
        self.param_frame.pack(fill=tk.X, pady=15)

        # Create initial parameter widgets
        self.create_parameter_widgets()

        # Process button with modern style
        self.process_btn = tk.Button(self.sidebar, text="Process Image", 
                                   bg="#27ae60", fg="white",
                                   font=("Segoe UI", 12, "bold"), 
                                   relief=tk.FLAT, activebackground="#2ecc71",
                                   command=self.process_image)
        self.process_btn.pack(fill=tk.X, pady=20)
        self.process_btn.bind("<Enter>", lambda e: self.process_btn.config(bg="#2ecc71"))
        self.process_btn.bind("<Leave>", lambda e: self.process_btn.config(bg="#27ae60"))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self.sidebar, textvariable=self.status_var, 
                                 bg=self.sidebar_color, fg="#bdc3c7", 
                                 font=("Segoe UI", 9), anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(20, 0), side=tk.BOTTOM)

        # ==== Main Content Frame ====
        self.main_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=20)
        self.main_frame.grid(row=0, column=1, sticky="nsew")

      