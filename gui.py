import os
import sys
import threading
import json
import time
from queue import Queue
from PIL import Image, ImageTk
import numpy as np
import rasterio

import tensorflow as tf
from tensorflow.keras.models import load_model

import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import functions from train.py
import train as train_script

# --- Set Appearance ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class SmartAgriSys_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Smart Agricultural Land Suitability System using Satellite Imagery")
        self.geometry("1150x850")

        # --- State Variables ---
        self.log_queue = Queue()
        self.training_thread = None
        self.loaded_model = None
        
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="SMART-AGRI AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.sub_logo = ctk.CTkLabel(self.sidebar, text="Hybrid Land Suitability Engine", font=ctk.CTkFont(size=12))
        self.sub_logo.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Dataset Selection
        self.ds_label = ctk.CTkLabel(self.sidebar, text="Analysis Platform:", anchor="w")
        self.ds_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.ds_menu = ctk.CTkOptionMenu(self.sidebar, values=["rgb", "multispectral"])
        self.ds_menu.grid(row=3, column=0, padx=20, pady=10)

        # Engine Config
        self.sep = ctk.CTkLabel(self.sidebar, text="--- Engine Config ---", font=ctk.CTkFont(size=11))
        self.sep.grid(row=4, column=0, pady=10)

        self.epoch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Epochs (25)")
        self.epoch_entry.insert(0, "25")
        self.epoch_entry.grid(row=5, column=0, padx=20, pady=5)

        self.batch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Batch (32)")
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=6, column=0, padx=20, pady=5)

        self.train_btn = ctk.CTkButton(self.sidebar, text="🚀 START TRAINING", command=self.start_training_flow, 
                                      fg_color="#1a5276", hover_color="#154360", font=ctk.CTkFont(weight="bold"))
        self.train_btn.grid(row=7, column=0, padx=20, pady=20)

        # --- Main Tabs ---
        self.tabview = ctk.CTkTabview(self, corner_radius=15)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("HYBRID ANALYSIS")
        self.tabview.add("ENGINE LOGS")

        # --- Analysis View ---
        self.tabview.tab("HYBRID ANALYSIS").grid_columnconfigure(0, weight=1)
        self.p_controls = ctk.CTkFrame(self.tabview.tab("HYBRID ANALYSIS"), fg_color="transparent")
        self.p_controls.pack(pady=20, fill="x", padx=20)
        
        self.load_btn = ctk.CTkButton(self.p_controls, text="1. Load CNN Weights (.h5)", command=self.load_model_action)
        self.load_btn.pack(side="left", padx=5, expand=True)
        
        self.select_btn = ctk.CTkButton(self.p_controls, text="2. Process Satellite Image", command=self.select_predict_image, fg_color="#27ae60")
        self.select_btn.pack(side="left", padx=5, expand=True)

        self.report_frame = ctk.CTkFrame(self.tabview.tab("HYBRID ANALYSIS"), corner_radius=10)
        self.report_frame.pack(pady=10, fill="both", expand=True, padx=20)
        
        self.img_label = ctk.CTkLabel(self.report_frame, text="Awaiting Remote Sensing Data...", width=350, height=350)
        self.img_label.pack(side="left", padx=20, pady=20)

        self.data_panel = ctk.CTkFrame(self.report_frame, fg_color="#1E1E1E", corner_radius=10)
        self.data_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.title_data = ctk.CTkLabel(self.data_panel, text="HYBRID SYSTEM REPORT", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_data.pack(pady=15)
        self.cnn_res = ctk.CTkLabel(self.data_panel, text="CNN Prediction: ---", anchor="w"); self.cnn_res.pack(pady=5, fill="x", padx=20)
        self.ndvi_res = ctk.CTkLabel(self.data_panel, text="Spectral Index (NDVI): ---", anchor="w"); self.ndvi_res.pack(pady=5, fill="x", padx=20)
        self.final_res = ctk.CTkLabel(self.data_panel, text="FINAL SUITABILITY SCORE", font=ctk.CTkFont(size=15, weight="bold"), fg_color="#2c3e50")
        self.final_res.pack(pady=15, fill="x", padx=20)
        self.score_label = ctk.CTkLabel(self.data_panel, text="---", font=ctk.CTkFont(size=22, weight="bold")); self.score_label.pack(pady=5)
        self.recommendation = ctk.CTkTextbox(self.data_panel, height=120, fg_color="transparent"); self.recommendation.pack(pady=10, fill="x", padx=20)

        # --- Logs ---
        self.tabview.tab("ENGINE LOGS").grid_columnconfigure(0, weight=1)
        self.textbox = ctk.CTkTextbox(self.tabview.tab("ENGINE LOGS"), font=ctk.CTkFont(family="Consolas"))
        self.textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.after(100, self.update_logs)

    def log(self, msg): self.log_queue.put(msg)
    def update_logs(self):
        while not self.log_queue.empty(): self.textbox.insert("end", f"{self.log_queue.get()}\n"); self.textbox.see("end")
        self.after(100, self.update_logs)

    def load_model_action(self):
        path = filedialog.askopenfilename(filetypes=[("H5 Weights", "*.h5")])
        if path:
            try:
                self.loaded_model = load_model(path)
                messagebox.showinfo("Ready", "Hybrid Core Initialized.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def select_predict_image(self):
        path = filedialog.askopenfilename(filetypes=[("Satellite Files", "*.jpg *.png *.tif")])
        if not path: return
        
        try:
            if path.lower().endswith('.tif'):
                # PRO FIX: Using Rasterio for 13-band TIF visualization
                with rasterio.open(path) as src:
                    # Extract Bands 4 (R), 3 (G), 2 (B) for display
                    r = src.read(4); g = src.read(3); b = src.read(2)
                    # Simple normalization for display
                    rgb = np.dstack((r, g, b))
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
                    img = Image.fromarray(rgb.astype('uint8'))
            else:
                img = Image.open(path)
                
            d_img = img.resize((350, 350))
            ctk_img = ctk.CTkImage(light_image=d_img, dark_image=d_img, size=(350, 350))
            self.img_label.configure(image=ctk_img, text="")
            
            if self.loaded_model: self.predict(path, img)
            else: messagebox.showwarning("Warning", "Load Model Weights First.")
        except Exception as e:
            messagebox.showerror("IO Error", f"Cannot process image: {e}")

    def start_training_flow(self):
        if self.training_thread and self.training_thread.is_alive(): return
        self.tabview.set("ENGINE LOGS")
        self.textbox.delete("0.0", "end")
        self.log(">>> INITIALIZING TRAINING SEQUENCE...")
        
        class Args: pass
        args = Args()
        args.dataset = self.ds_menu.get()
        args.epochs = int(self.epoch_entry.get())
        args.batch_size = int(self.batch_entry.get())
        args.lr = 1e-4; args.patience = 7

        def run():
            try:
                # We overwrite the print function inside the thread to capture output
                import builtins
                original_print = builtins.print
                def gui_print(*args, **kwargs):
                    msg = " ".join(map(str, args))
                    self.log(msg)
                    # original_print(*args, **kwargs)
                builtins.print = gui_print
                
                train_script.train(args)
                
                builtins.print = original_print
                self.log(">>> TRAINING SEQUENCE COMPLETED.")
            except Exception as e:
                self.log(f"!!! CRITICAL ERROR: {e}")
            finally:
                self.train_btn.configure(state="normal", text="🚀 START TRAINING")

        self.train_btn.configure(state="disabled", text="TRAINING...")
        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def predict(self, path, img):
        classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", 
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        mode = self.ds_menu.get()
        try:
            # 1. Prediction
            if mode == "rgb":
                x = np.array(img.resize((64, 64)).convert("RGB")) / 255.0
            else:
                with rasterio.open(path) as src:
                    x = src.read().transpose(1, 2, 0).astype(np.float32)
                means = np.array([1354.4,1118.2,1042.9,947.6,1199.5,1999.8,2369.2,2296.8,732.1,12.1,1819.0,1118.9,2594.1]).reshape(1,1,13)
                stds = np.array([245.7,333.0,395.1,593.8,566.4,861.2,1086.6,1118.0,404.9,4.8,1002.6,761.3,1231.6]).reshape(1,1,13)
                x = (x - means) / (stds + 1e-8)
            
            preds = self.loaded_model.predict(np.expand_dims(x, axis=0), verbose=0)
            idx = np.argmax(preds[0]); class_name = classes[idx]; conf = preds[0][idx] * 100
            self.cnn_res.configure(text=f"CNN Prediction: {class_name} ({conf:.1f}%)")

            # 2. NDVI
            ndvi_val = 0.0
            if mode == "multispectral":
                with rasterio.open(path) as src:
                    b4 = src.read(4).astype(float); b8 = src.read(8).astype(float)
                    ndvi_val = np.mean((b8 - b4) / (b8 + b4 + 1e-8))
                self.ndvi_res.configure(text=f"Spectral Index (NDVI): {ndvi_val:.2f}")
            else: self.ndvi_res.configure(text="Spectral Index (NDVI): (N/A in RGB)")

            # 3. Hybrid Decision
            final = "NOT SUITABLE"; color = "#e74c3c"; conclusion = "Unsuitable for Agriculture."
            if mode == "multispectral":
                if class_name in ["AnnualCrop", "PermanentCrop"] and ndvi_val > 0.4:
                    final = "HIGHLY SUITABLE"; color = "#2ecc71"
                    conclusion = "Ideal condition: Optimized soil & spectral health."
                elif ndvi_val > 0.2:
                    final = "MODERATE"; color = "#f1c40f"
                    conclusion = "Arable land detected but requires investigation."
            elif class_name in ["AnnualCrop", "PermanentCrop"]: final = "SUITABLE (Estimated)"; color = "#2ecc71"

            self.score_label.configure(text=final, text_color=color)
            self.recommendation.delete("0.0", "end"); self.recommendation.insert("0.0", f"Expert Analysis:\n{conclusion}")

        except Exception as e: messagebox.showerror("Error", f"Fusion failed: {e}")

if __name__ == "__main__":
    app = SmartAgriSys_GUI()
    app.mainloop()
