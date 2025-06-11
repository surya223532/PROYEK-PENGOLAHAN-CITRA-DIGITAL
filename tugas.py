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
        self.root.title("Aplikasi Pengolahan Citra Digital")
        self.root.configure(bg="#FFA500")
        self.root.geometry("1200x800")
        
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        self.original_img = None
        self.processed_img = None
        self.original_photo = None
        self.processed_photo = None
        self.histogram_figures = []

        # ==== Main Frames ====
        self.control_frame = tk.Frame(root, bg="#FFA500", padx=20, pady=20)
        self.control_frame.grid(row=0, column=0, sticky="ns")

        self.image_frame = tk.Frame(root, bg="#FFB84D", padx=20, pady=20)
        self.image_frame.grid(row=0, column=1, sticky="nsew")

        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # ===== Control Frame =====
        title_label = tk.Label(self.control_frame, text="Pengolahan Citra Digital", bg="#FFA500", 
                              fg="white", font=("Arial", 20, "bold"))
        title_label.pack(pady=(0, 25))

        # Action buttons
        button_frame = tk.Frame(self.control_frame, bg="#FFA500")
        button_frame.pack(pady=10)

        self.upload_btn = tk.Button(button_frame, text="Upload Gambar", bg="white", fg="#FFA500",
                                   font=("Arial", 12, "bold"), width=20, command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        # Removed save button as requested

        # Parameter frame
        self.param_frame = tk.Frame(self.control_frame, bg="#FFA500")
        self.param_frame.pack(pady=10, fill=tk.X)

        # Process selection dropdown
        self.options = [
            "Grayscale",
            "Biner (Threshold)",
            "Brightness/Contrast",
            "Operasi Logika",
            "Histogram",
            "Dilasi",  # Only Dilasi kept as requested
            "Edge Detection",
            "Rotasi/Flip",
            "Negative Image"
        ]
        
        self.option_var = tk.StringVar(value=self.options[0])
        self.option_menu = ttk.Combobox(self.control_frame, textvariable=self.option_var, 
                                      values=self.options, state="readonly", font=("Arial", 12))
        self.option_menu.pack(pady=15, fill=tk.X)
        self.option_menu.bind("<<ComboboxSelected>>", self.update_parameters)

        # Create parameter widgets
        self.create_parameter_widgets()

        self.process_btn = tk.Button(self.control_frame, text="Proses", bg="white", fg="#FFA500",
                                    font=("Arial", 14, "bold"), width=20, command=self.process_image)
        self.process_btn.pack(pady=20)

        # Status bar
        self.status_var = tk.StringVar(value="Siap")
        self.status_bar = tk.Label(self.control_frame, textvariable=self.status_var, bg="#FFA500",
                                  fg="white", font=("Arial", 10), anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(20, 0))

        # ===== Image Frame =====
        title_frame = tk.Frame(self.image_frame, bg="#FFB84D")
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        ori_label_title = tk.Label(title_frame, text="Gambar Asli", bg="#FFB84D",
                                  font=("Arial", 16, "bold"))
        ori_label_title.pack(side=tk.LEFT, padx=15, expand=True)

        proc_label_title = tk.Label(title_frame, text="Hasil Proses", bg="#FFB84D",
                                   font=("Arial", 16, "bold"))
        proc_label_title.pack(side=tk.LEFT, padx=15, expand=True)

        # Canvas for images with scrollbars
        self.canvas_frame = tk.Frame(self.image_frame, bg="#FFB84D")
        self.canvas_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Original image canvas
        self.original_canvas = tk.Canvas(self.canvas_frame, bg="white", relief="sunken")
        self.original_canvas.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # Processed image canvas
        self.processed_canvas = tk.Canvas(self.canvas_frame, bg="white", relief="sunken")
        self.processed_canvas.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        # Scrollbars
        self.original_scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.original_canvas.yview)
        self.original_scroll_y.grid(row=0, column=2, sticky="ns")
        self.original_canvas.configure(yscrollcommand=self.original_scroll_y.set)

        self.processed_scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.processed_canvas.yview)
        self.processed_scroll_y.grid(row=0, column=3, sticky="ns")
        self.processed_canvas.configure(yscrollcommand=self.processed_scroll_y.set)

        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(1, weight=1)
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

    def create_parameter_widgets(self):
        """Create parameter widgets based on selected process"""
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        current_choice = self.option_var.get()

        if current_choice == "Biner (Threshold)":
            tk.Label(self.param_frame, text="Threshold Value:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.threshold_var = tk.IntVar(value=128)
            self.threshold_slider = tk.Scale(self.param_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                           variable=self.threshold_var, bg="#FFA500")
            self.threshold_slider.pack(fill=tk.X)

        elif current_choice == "Brightness/Contrast":
            tk.Label(self.param_frame, text="Brightness:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.brightness_var = tk.IntVar(value=0)
            self.brightness_slider = tk.Scale(self.param_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                            variable=self.brightness_var, bg="#FFA500")
            self.brightness_slider.pack(fill=tk.X)

            tk.Label(self.param_frame, text="Contrast:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.contrast_var = tk.DoubleVar(value=1.0)
            self.contrast_slider = tk.Scale(self.param_frame, from_=0.1, to=3.0, resolution=0.1,
                                          orient=tk.HORIZONTAL, variable=self.contrast_var, bg="#FFA500")
            self.contrast_slider.pack(fill=tk.X)

        elif current_choice == "Operasi Logika":
            tk.Label(self.param_frame, text="Operasi:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.logic_op_var = tk.StringVar(value="AND")
            logic_menu = ttk.Combobox(self.param_frame, textvariable=self.logic_op_var, 
                                     values=["AND", "OR", "XOR", "NOT"], state="readonly")
            logic_menu.pack(fill=tk.X)

        elif current_choice == "Dilasi":  # Only Dilasi kept as requested
            tk.Label(self.param_frame, text="Kernel Size:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.morph_kernel_var = tk.IntVar(value=3)
            morph_kernel_slider = tk.Scale(self.param_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                                         variable=self.morph_kernel_var, bg="#FFA500")
            morph_kernel_slider.pack(fill=tk.X)

            tk.Label(self.param_frame, text="Iterations:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.morph_iter_var = tk.IntVar(value=1)
            morph_iter_slider = tk.Scale(self.param_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                       variable=self.morph_iter_var, bg="#FFA500")
            morph_iter_slider.pack(fill=tk.X)

        elif current_choice == "Edge Detection":
            tk.Label(self.param_frame, text="Metode:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.edge_method_var = tk.StringVar(value="Canny")
            edge_menu = ttk.Combobox(self.param_frame, textvariable=self.edge_method_var, 
                                   values=["Canny", "Sobel"], state="readonly")  # Removed Laplacian and Prewitt
            edge_menu.pack(fill=tk.X)

            if self.edge_method_var.get() == "Canny":
                tk.Label(self.param_frame, text="Threshold 1:", bg="#FFA500", 
                        font=("Arial", 10)).pack(anchor=tk.W)
                self.canny_thresh1_var = tk.IntVar(value=100)
                canny_thresh1_slider = tk.Scale(self.param_frame, from_=0, to=500, orient=tk.HORIZONTAL,
                                              variable=self.canny_thresh1_var, bg="#FFA500")
                canny_thresh1_slider.pack(fill=tk.X)

                tk.Label(self.param_frame, text="Threshold 2:", bg="#FFA500", 
                        font=("Arial", 10)).pack(anchor=tk.W)
                self.canny_thresh2_var = tk.IntVar(value=200)
                canny_thresh2_slider = tk.Scale(self.param_frame, from_=0, to=500, orient=tk.HORIZONTAL,
                                              variable=self.canny_thresh2_var, bg="#FFA500")
                canny_thresh2_slider.pack(fill=tk.X)

        elif current_choice == "Rotasi/Flip":
            tk.Label(self.param_frame, text="Transformasi:", bg="#FFA500", 
                    font=("Arial", 10)).pack(anchor=tk.W)
            self.transform_var = tk.StringVar(value="Rotasi 90°")
            transform_menu = ttk.Combobox(self.param_frame, textvariable=self.transform_var, 
                                        values=["Rotasi 90°", "Rotasi 180°", "Rotasi 270°", 
                                               "Flip Horizontal", "Flip Vertical"], state="readonly")
            transform_menu.pack(fill=tk.X)

    def update_parameters(self, event=None):
        """Update parameter widgets when process selection changes"""
        self.create_parameter_widgets()

    def upload_image(self):
        """Upload image from file system"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.original_img = Image.open(file_path)
                self.display_image(self.original_img, self.original_canvas)
                self.processed_img = None
                self.processed_canvas.delete("all")
                self.status_var.set(f"Gambar berhasil diupload: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka gambar: {str(e)}")
                self.status_var.set("Error saat mengupload gambar")

    def display_image(self, img, canvas):
        """Display image on canvas"""
        canvas.delete("all")
        
        # Resize image to fit canvas but maintain aspect ratio
        canvas_width = canvas.winfo_width() - 4
        canvas_height = canvas.winfo_height() - 4
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 500
            canvas_height = 400
        
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
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
            messagebox.showerror("Error", "Silakan upload gambar terlebih dahulu!")
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
                threshold = self.threshold_var.get()
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                processed = binary

            elif choice == "Brightness/Contrast":
                brightness = self.brightness_var.get()
                contrast = self.contrast_var.get()
                processed = cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)

            elif choice == "Operasi Logika":
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                operation = self.logic_op_var.get()
                
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

            elif choice == "Dilasi":  # Only Dilasi kept as requested
                kernel_size = self.morph_kernel_var.get()
                iterations = self.morph_iter_var.get()
                
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                processed = cv2.dilate(gray, kernel, iterations=iterations)

            elif choice == "Edge Detection":
                method = self.edge_method_var.get()
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                
                if method == "Canny":
                    threshold1 = self.canny_thresh1_var.get()
                    threshold2 = self.canny_thresh2_var.get()
                    processed = cv2.Canny(gray, threshold1, threshold2)
                elif method == "Sobel":
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    processed = cv2.magnitude(sobelx, sobely)
                    processed = np.uint8(processed)

            elif choice == "Rotasi/Flip":
                transform = self.transform_var.get()
                img_pil = self.original_img.copy()
                
                if transform == "Rotasi 90°":
                    processed_img = img_pil.rotate(-90, expand=True)
                elif transform == "Rotasi 180°":
                    processed_img = img_pil.rotate(180)
                elif transform == "Rotasi 270°":
                    processed_img = img_pil.rotate(90, expand=True)
                elif transform == "Flip Horizontal":
                    processed_img = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                elif transform == "Flip Vertical":
                    processed_img = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                
                self.processed_img = processed_img
                self.display_image(processed_img, self.processed_canvas)
                self.status_var.set(f"Proses {choice} berhasil")
                return

            elif choice == "Negative Image":
                processed = cv2.bitwise_not(cv_img)

            # Convert result to displayable format
            if processed is not None:
                if len(processed.shape) == 2:
                    img_to_show = Image.fromarray(processed)
                else:
                    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    img_to_show = Image.fromarray(img_rgb)
                
                self.processed_img = img_to_show
                self.display_image(img_to_show, self.processed_canvas)
                self.status_var.set(f"Proses {choice} berhasil")

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat memproses gambar: {str(e)}")
            self.status_var.set("Error saat memproses gambar")

    def show_histogram(self, cv_img):
        """Show image histogram"""
        try:
            fig = plt.figure(figsize=(10, 6))
            
            # Color histogram (RGB)
            ax1 = fig.add_subplot(211)
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([cv_img], [i], None, [256], [0, 256])
                ax1.plot(hist, color=col, label=col.upper())
            ax1.set_title("Histogram Warna (RGB)")
            ax1.set_xlabel("Nilai Intensitas")
            ax1.set_ylabel("Jumlah Pixel")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Grayscale histogram
            ax2 = fig.add_subplot(212)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
            ax2.plot(hist_gray, color='black', label='Grayscale')
            ax2.set_title("Histogram Grayscale")
            ax2.set_xlabel("Nilai Intensitas")
            ax2.set_ylabel("Jumlah Pixel")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            hist_win = tk.Toplevel(self.root)
            hist_win.title("Histogram Gambar")
            
            canvas = FigureCanvasTkAgg(fig, master=hist_win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.histogram_figures.append(fig)
            
            close_btn = tk.Button(hist_win, text="Tutup", command=hist_win.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat histogram: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()