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
ctk.set_default_color_theme("blue")

class EuroSAT_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EuroSAT Classification - TensorFlow AI")
        self.geometry("1100x700")

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
        self.sidebar.grid_rowconfigure(10, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="EuroSAT TF", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Dataset Selection
        self.ds_label = ctk.CTkLabel(self.sidebar, text="Dataset Mode:", anchor="w")
        self.ds_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.ds_menu = ctk.CTkOptionMenu(self.sidebar, values=["rgb", "multispectral"])
        self.ds_menu.grid(row=2, column=0, padx=20, pady=10)

        # Hyperparams
        self.epoch_label = ctk.CTkLabel(self.sidebar, text="Epochs:", anchor="w")
        self.epoch_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.epoch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="25")
        self.epoch_entry.insert(0, "25")
        self.epoch_entry.grid(row=4, column=0, padx=20, pady=5)

        self.batch_label = ctk.CTkLabel(self.sidebar, text="Batch Size:", anchor="w")
        self.batch_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.batch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="32")
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=6, column=0, padx=20, pady=5)

        # Buttons
        self.train_btn = ctk.CTkButton(self.sidebar, text="START TRAINING", command=self.start_training_flow, 
                                      fg_color="#3498db", hover_color="#2980b9", font=ctk.CTkFont(weight="bold"))
        self.train_btn.grid(row=9, column=0, padx=20, pady=20)

        self.appearance_label = ctk.CTkLabel(self.sidebar, text="UI Theme:", anchor="w")
        self.appearance_label.grid(row=11, column=0, padx=20, pady=(10, 0))
        self.appearance_menu = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], command=ctk.set_appearance_mode)
        self.appearance_menu.grid(row=12, column=0, padx=20, pady=(10, 20))

        # --- Main Content Tabs ---
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Training Monitor")
        self.tabview.add("Testing / Prediction")

        # Training Tab Layout
        self.tabview.tab("Training Monitor").grid_columnconfigure(0, weight=1)
        self.textbox = ctk.CTkTextbox(self.tabview.tab("Training Monitor"), height=400)
        self.textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.textbox.insert("0.0", "TensorFlow Ready. Click Start to begin.\n")

        # Prediction Tab Layout
        self.tabview.tab("Testing / Prediction").grid_columnconfigure(0, weight=1)
        self.p_controls = ctk.CTkFrame(self.tabview.tab("Testing / Prediction"), fg_color="transparent")
        self.p_controls.pack(pady=10)
        
        self.load_btn = ctk.CTkButton(self.p_controls, text="Load Keras Model (.h5)", command=self.load_model_action)
        self.load_btn.pack(side="left", padx=10)
        
        self.select_btn = ctk.CTkButton(self.p_controls, text="Select Image", command=self.select_predict_image)
        self.select_btn.pack(side="left", padx=10)

        self.img_frame = ctk.CTkFrame(self.tabview.tab("Testing / Prediction"), corner_radius=10)
        self.img_frame.pack(pady=20, fill="both", expand=True, padx=20)
        
        self.img_label = ctk.CTkLabel(self.img_frame, text="No Image", width=300, height=300)
        self.img_label.pack(pady=20)
        
        self.res_label = ctk.CTkLabel(self.img_frame, text="Result: ---", font=ctk.CTkFont(size=22, weight="bold"))
        self.res_label.pack(pady=10)

        self.after(100, self.update_logs)

    def log(self, msg):
        self.log_queue.put(msg)

    def update_logs(self):
        while not self.log_queue.empty():
            self.textbox.insert("end", f"{self.log_queue.get()}\n")
            self.textbox.see("end")
        self.after(100, self.update_logs)

    def start_training_flow(self):
        if self.training_thread and self.training_thread.is_alive(): return
        
        class Args:
            pass
        args = Args()
        args.dataset = self.ds_menu.get()
        args.epochs = int(self.epoch_entry.get())
        args.batch_size = int(self.batch_entry.get())
        args.lr = 1e-4
        args.patience = 7

        self.train_btn.configure(state="disabled", text="RUNNING...")
        self.training_thread = threading.Thread(target=self.run_train_thread, args=(args,))
        self.training_thread.start()

    def run_train_thread(self, args):
        try:
            # Re-directing stdout would be complex, so we just run the main train function
            train_script.train(args)
            self.log("Training Finished!")
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.train_btn.configure(state="normal", text="START TRAINING")

    def load_model_action(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.h5")])
        if path:
            try:
                self.loaded_model = load_model(path)
                messagebox.showinfo("Loaded", f"Model loaded from {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_predict_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.tif")])
        if not path: return
        
        # Display image
        img = Image.open(path)
        d_img = img.resize((300, 300))
        ctk_img = ctk.CTkImage(light_image=d_img, dark_image=d_img, size=(300, 300))
        self.img_label.configure(image=ctk_img, text="")
        
        if self.loaded_model:
            self.predict(path, img)
        else:
            self.res_label.configure(text="Result: Load Model First!")

    def predict(self, path, img):
        classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", 
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        mode = self.ds_menu.get()
        
        try:
            if mode == "rgb":
                x = img.resize((64, 64)).convert("RGB")
                x = np.array(x) / 255.0
                x = np.expand_dims(x, axis=0)
            else:
                with rasterio.open(path) as src:
                    x = src.read().transpose(1, 2, 0).astype(np.float32)
                # Normalization stats from train.py
                means = np.array([1354.4,1118.2,1042.9,947.6,1199.5,1999.8,2369.2,2296.8,732.1,12.1,1819.0,1118.9,2594.1]).reshape(1,1,13)
                stds = np.array([245.7,333.0,395.1,593.8,566.4,861.2,1086.6,1118.0,404.9,4.8,1002.6,761.3,1231.6]).reshape(1,1,13)
                x = (x - means) / (stds + 1e-8)
                x = np.expand_dims(x, axis=0)
            
            preds = self.loaded_model.predict(x)
            idx = np.argmax(preds[0])
            conf = preds[0][idx] * 100
            self.res_label.configure(text=f"Result: {classes[idx]} ({conf:.1f}%)")
        except Exception as e:
            self.res_label.configure(text=f"Error: {e}")

if __name__ == "__main__":
    app = EuroSAT_GUI()
    app.mainloop()
