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

      # Image display title
        title_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        ori_label_title = tk.Label(title_frame, text="Original Image", bg=self.bg_color,
                                 font=("Segoe UI", 14, "bold"), fg=self.text_color)
        ori_label_title.pack(side=tk.LEFT, padx=15, expand=True)

        proc_label_title = tk.Label(title_frame, text="Processed Result", bg=self.bg_color,
                                  font=("Segoe UI", 14, "bold"), fg=self.text_color)
        proc_label_title.pack(side=tk.LEFT, padx=15, expand=True)

        # Canvas frame with shadow effect
        self.canvas_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Original image canvas with shadow
        original_frame = tk.Frame(self.canvas_frame, bg=self.bg_color)
        original_frame.grid(row=0, column=0, padx=(0, 20), pady=5, sticky="nsew")

        # Shadow effect for canvas
        canvas_shadow = tk.Frame(original_frame, bg="#d5d5d5")
        canvas_shadow.pack(fill=tk.BOTH, expand=True, padx=(0, 3), pady=(0, 3))

        self.original_canvas = tk.Canvas(canvas_shadow, bg=self.canvas_bg, relief=tk.FLAT,
                                       highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Processed image canvas with shadow
        processed_frame = tk.Frame(self.canvas_frame, bg=self.bg_color)
        processed_frame.grid(row=0, column=1, padx=(20, 0), pady=5, sticky="nsew")

        # Shadow effect for canvas
        canvas_shadow2 = tk.Frame(processed_frame, bg="#d5d5d5")
        canvas_shadow2.pack(fill=tk.BOTH, expand=True, padx=(0, 3), pady=(0, 3))

        self.processed_canvas = tk.Canvas(canvas_shadow2, bg=self.canvas_bg, relief=tk.FLAT,
                                        highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        # Add hover effects to all buttons
        self.style_buttons()

    def style_buttons(self):
        """Apply modern styling to all buttons"""
        buttons = [self.upload_btn, self.process_btn]
        
        for btn in buttons:
            btn.config(borderwidth=0, relief=tk.FLAT, 
                      activebackground=self.button_hover,  # Ganti transparansi dengan warna solid
                      padx=10, pady=8)
            
            # Add rounded corners (simulated with canvas)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.button_hover))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=self.button_color))

    def create_parameter_widgets(self):
        """Create parameter widgets based on selected process"""
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        current_choice = self.option_var.get()

        if current_choice == "Biner (Threshold)":
            self.create_slider("Threshold Value:", 0, 255, 128, "threshold_var")

        elif current_choice == "Brightness/Contrast":
            self.create_slider("Brightness:", -100, 100, 0, "brightness_var")
            self.create_slider("Contrast:", 0.1, 3.0, 1.0, "contrast_var", resolution=0.1)

        elif current_choice == "Operasi Logika":
            self.create_dropdown("Operasi:", ["AND", "OR", "XOR", "NOT"], "logic_op_var")

        elif current_choice == "Dilasi":
            self.create_slider("Kernel Size:", 1, 15, 3, "morph_kernel_var")
            self.create_slider("Iterations:", 1, 10, 1, "morph_iter_var")

        elif current_choice == "Edge Detection":
            self.create_dropdown("Metode:", ["Canny", "Sobel"], "edge_method_var")
            
            if hasattr(self, 'edge_method_var') and self.edge_method_var.get() == "Canny":
                self.create_slider("Threshold 1:", 0, 500, 100, "canny_thresh1_var")
                self.create_slider("Threshold 2:", 0, 500, 200, "canny_thresh2_var")

    def create_slider(self, label_text, from_, to, default, var_name, resolution=1):
        """Helper method to create modern sliders"""
        frame = tk.Frame(self.param_frame, bg=self.sidebar_color)
        frame.pack(fill=tk.X, pady=5)
        
        label = tk.Label(frame, text=label_text, bg=self.sidebar_color, 
                        fg="white", font=("Segoe UI", 9))
        label.pack(anchor=tk.W)
        
        var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
        setattr(self, var_name, var)
        
        slider = ttk.Scale(frame, from_=from_, to=to, variable=var, 
                          orient=tk.HORIZONTAL, style="Horizontal.TScale")
        slider.pack(fill=tk.X)
        
        # Value display
        value_label = tk.Label(frame, textvariable=var, bg=self.sidebar_color, 
                             fg="white", font=("Segoe UI", 8))
        value_label.pack(anchor=tk.E)

    def create_dropdown(self, label_text, options, var_name):
        """Helper method to create modern dropdowns"""
        frame = tk.Frame(self.param_frame, bg=self.sidebar_color)
        frame.pack(fill=tk.X, pady=5)
        
        label = tk.Label(frame, text=label_text, bg=self.sidebar_color, 
                        fg="white", font=("Segoe UI", 9))
        label.pack(anchor=tk.W)
        
        var = tk.StringVar(value=options[0])
        setattr(self, var_name, var)
        
        style = ttk.Style()
        style.configure('TCombobox', fieldbackground="white", background="white")
        
        dropdown = ttk.Combobox(frame, textvariable=var, values=options, 
                               state="readonly", font=("Segoe UI", 9))
        dropdown.pack(fill=tk.X)

    def update_parameters(self, event=None):
        """Update parameter widgets when process selection changes"""
        self.create_parameter_widgets()

    def upload_image(self):
        """Upload image from file system"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp *.tiff"), ("All files", ".*")]
        )
        
        if file_path:
            try:
                self.original_img = Image.open(file_path)
                self.display_image(self.original_img, self.original_canvas)
                self.processed_img = None
                self.processed_canvas.delete("all")
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")

    def display_image(self, img, canvas):
        """Display image on canvas"""
        canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width() - 4
        canvas_height = canvas.winfo_height() - 4
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 500
            canvas_height = 400
        
        # Calculate aspect ratio preserving dimensions
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        # Resize and display image
        display_img = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        
        if canvas == self.original_canvas:
            self.original_photo = photo
        else:
            self.processed_photo = photo
        
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        canvas.image = photo

    def process_image(self):
        """Process image based on user selection"""
        if self.original_img is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return

        try:
            cv_img = cv2.cvtColor(np.array(self.original_img), cv2.COLOR_RGB2BGR)
            choice = self.option_var.get()
            processed = None

            if choice == "Grayscale":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                processed = gray

            elif choice == "Biner (Threshold)":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                threshold = getattr(self, "threshold_var").get()
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                processed = binary

            elif choice == "Brightness/Contrast":
                brightness = getattr(self, "brightness_var").get()
                contrast = getattr(self, "contrast_var").get()
                processed = cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)

            elif choice == "Operasi Logika":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                operation = getattr(self, "logic_op_var").get()
                
                mask = np.zeros_like(gray)
                cv2.circle(mask, (gray.shape[1]//2, gray.shape[0]//2), 
                          min(gray.shape)//3, 255, -1)
                
                if operation == "AND":
                    processed = cv2.bitwise_and(gray, mask)
                elif operation == "OR":
                    processed = cv2.bitwise_or(gray, mask)
                elif operation == "XOR":
                    processed = cv2.bitwise_xor(gray, mask)
                elif operation == "NOT":
                    processed = cv2.bitwise_not(gray)

            elif choice == "Histogram":
                self.show_histogram(cv_img)
                return

            elif choice == "Dilasi":
                kernel_size = getattr(self, "morph_kernel_var").get()
                iterations = getattr(self, "morph_iter_var").get()
                
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                processed = cv2.dilate(gray, kernel, iterations=iterations)

            elif choice == "Edge Detection":
                method = getattr(self, "edge_method_var").get()
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                
                if method == "Canny":
                    threshold1 = getattr(self, "canny_thresh1_var").get()
                    threshold2 = getattr(self, "canny_thresh2_var").get()
                    processed = cv2.Canny(gray, threshold1, threshold2)
                elif method == "Sobel":
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    processed = cv2.magnitude(sobelx, sobely)
                    processed = np.uint8(processed)

            # Convert result to displayable format
            if processed is not None:
                if len(processed.shape) == 2:
                    img_to_show = Image.fromarray(processed)
                else:
                    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    img_to_show = Image.fromarray(img_rgb)
                
                self.processed_img = img_to_show
                self.display_image(img_to_show, self.processed_canvas)
                self.status_var.set(f"Processed: {choice}")

        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.status_var.set("Processing error")

            def show_histogram(self, cv_img):
        """Show image histogram in a modern dialog"""
        try:
            # Validate image dimensions
            if cv_img is None or len(cv_img.shape) < 2:
                messagebox.showerror("Error", "Invalid image for histogram.")
                return

            # Use a valid matplotlib style
            plt.style.use('ggplot')  # Ganti 'seaborn' dengan 'ggplot' atau style lain yang tersedia
            fig = plt.figure(figsize=(10, 6), facecolor='#f5f5f5')
            
            # Color histogram (RGB)
            ax1 = fig.add_subplot(211)
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([cv_img], [i], None, [256], [0, 256])
                ax1.plot(hist, color=col, label=col.upper())
            ax1.set_title("Color Histogram (RGB)", fontsize=12)
            ax1.set_xlabel("Intensity Value", fontsize=10)
            ax1.set_ylabel("Pixel Count", fontsize=10)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Grayscale histogram
            ax2 = fig.add_subplot(212)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            ax2.plot(hist_gray, color='black', label='Grayscale')
            ax2.set_title("Grayscale Histogram", fontsize=12)
            ax2.set_xlabel("Intensity Value", fontsize=10)
            ax2.set_ylabel("Pixel Count", fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Create modern histogram window
            hist_win = tk.Toplevel(self.root)
            hist_win.title("Image Histogram")
            hist_win.geometry("800x600")
            hist_win.configure(bg=self.bg_color)
            
            # Add canvas for figure
            canvas = FigureCanvasTkAgg(fig, master=hist_win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Add close button
            close_btn = tk.Button(hist_win, text="Close", command=hist_win.destroy,
                                bg=self.button_color, fg="white", relief=tk.FLAT,
                                font=("Segoe UI", 10))
            close_btn.pack(pady=(0, 15))
            close_btn.bind("<Enter>", lambda e: close_btn.config(bg=self.button_hover))
            close_btn.bind("<Leave>", lambda e: close_btn.config(bg=self.button_color))
            
            self.histogram_figures.append(fig)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create histogram: {str(e)}")

if _name_ == "_main_":
    root = tk.Tk()
    
    # Set window icon (optional)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    # Set theme
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ImageProcessorApp(root)
    root.mainloop()