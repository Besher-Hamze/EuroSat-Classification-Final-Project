import os
import sys
import threading
import json
import time
from datetime import datetime
from queue import Queue
from PIL import Image, ImageTk
import numpy as np
import rasterio
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.models import load_model

import customtkinter as ctk
from tkinter import filedialog, messagebox
from fpdf import FPDF

# Import functions from train.py
import train as train_script

# --- Set Appearance ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

RGB_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MS_EXTENSIONS = {".tif", ".tiff"}
MS_REQUIRED_BANDS = 13

class SmartAgriSys_GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Smart Agricultural Land Suitability System - Professional Edition")
        self.geometry("1250x950")

        # --- State Variables ---
        self.log_queue = Queue()
        self.training_thread = None
        self.loaded_model = None
        self.current_analysis_data = None # Store results for reporting
        
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="AGRI-SAT AI", font=ctk.CTkFont(size=26, weight="bold")).grid(row=0, column=0, padx=20, pady=(30, 10))
        ctk.CTkLabel(self.sidebar, text="Expert Decision System", font=ctk.CTkFont(size=13)).grid(row=1, column=0, padx=20, pady=(0, 20))

        # Analysis Selection
        ctk.CTkLabel(self.sidebar, text="Operational Platform:", anchor="w").grid(row=2, column=0, padx=20, pady=(10, 0))
        self.ds_menu = ctk.CTkOptionMenu(self.sidebar, values=["rgb", "multispectral"])
        self.ds_menu.grid(row=3, column=0, padx=20, pady=10)

        # System Controls
        self.train_btn = ctk.CTkButton(self.sidebar, text="RE-TRAIN CORE", command=self.start_training_flow, 
                                      fg_color="#1a5276", hover_color="#154360")
        self.train_btn.grid(row=7, column=0, padx=20, pady=20)

        self.export_btn = ctk.CTkButton(self.sidebar, text="💾 GENERATE PDF REPORT", command=self.export_report_action, 
                                      fg_color="#e67e22", hover_color="#d35400", state="disabled")
        self.export_btn.grid(row=8, column=0, padx=20, pady=10)

        # --- Main Tabs ---
        self.tabview = ctk.CTkTabview(self, corner_radius=15)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("HYBRID LAND ANALYSIS")
        self.tabview.add("SYSTEM DIAGNOSTICS")

        # --- Analysis View ---
        self.tabview.tab("HYBRID LAND ANALYSIS").grid_columnconfigure(0, weight=1)
        
        self.p_controls = ctk.CTkFrame(self.tabview.tab("HYBRID LAND ANALYSIS"), fg_color="transparent")
        self.p_controls.pack(pady=10, fill="x", padx=20)
        
        self.load_btn = ctk.CTkButton(self.p_controls, text="1. Load AI Brain (.h5)", command=self.load_model_action)
        self.load_btn.pack(side="left", padx=5, expand=True)
        
        self.select_btn = ctk.CTkButton(self.p_controls, text="2. Ingest Satellite Frame", command=self.select_predict_image, fg_color="#27ae60")
        self.select_btn.pack(side="left", padx=5, expand=True)

        # Scrollable Report Area
        self.scroll_frame = ctk.CTkScrollableFrame(self.tabview.tab("HYBRID LAND ANALYSIS"), corner_radius=10)
        self.scroll_frame.pack(pady=10, fill="both", expand=True, padx=20)

        # Visualization View (Original & Grad-CAM)
        self.vis_row = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.vis_row.pack(fill="x", pady=10)

        self.orig_preview = ctk.CTkLabel(self.vis_row, text="Spectral/RGB Frame", width=300, height=300, fg_color="#2b2b2b", corner_radius=10)
        self.orig_preview.pack(side="left", padx=10, expand=True)

        self.cam_preview = ctk.CTkLabel(self.vis_row, text="Decision Map (Grad-CAM)", width=300, height=300, fg_color="#2b2b2b", corner_radius=10)
        self.cam_preview.pack(side="left", padx=10, expand=True)

        # Metrics Panel
        self.panel = ctk.CTkFrame(self.scroll_frame, fg_color="#1E1E1E", corner_radius=10)
        self.panel.pack(fill="both", expand=True, padx=10, pady=20)
        
        ctk.CTkLabel(self.panel, text="PROFESSIONAL ASSESSMENT REPORT", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        self.res_class = ctk.CTkLabel(self.panel, text="Detected Land Usage: ---", font=ctk.CTkFont(size=15)); self.res_class.pack(pady=2)
        self.res_ndvi = ctk.CTkLabel(self.panel, text="Biophysical Activity (NDVI): ---"); self.res_ndvi.pack(pady=2)
        self.res_final = ctk.CTkLabel(self.panel, text="DECISION: ---", font=ctk.CTkFont(size=22, weight="bold")); self.res_final.pack(pady=10)
        
        self.final_desc = ctk.CTkTextbox(self.panel, height=100, font=ctk.CTkFont(size=13), fg_color="transparent")
        self.final_desc.pack(fill="x", padx=20)
        self.final_desc.insert("0.0", "Initialize Analysis sequence for live reporting.")

        # --- Diagnostics View ---
        self.tabview.tab("SYSTEM DIAGNOSTICS").grid_columnconfigure(0, weight=1)
        self.textbox = ctk.CTkTextbox(self.tabview.tab("SYSTEM DIAGNOSTICS"), font=ctk.CTkFont(family="Consolas"))
        self.textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.after(100, self.update_logs)

    def log(self, msg): self.log_queue.put(msg)
    def update_logs(self):
        while not self.log_queue.empty(): self.textbox.insert("end", f"{self.log_queue.get()}\n"); self.textbox.see("end")
        self.after(100, self.update_logs)

    def _file_extension(self, path):
        return os.path.splitext(path)[1].lower()

    def _expected_image_hint(self, mode):
        if mode == "rgb":
            return (
                "JPEG or PNG image (3-channel RGB)\n"
                "Example: EuroSAT folder → *.jpg"
            )
        return (
            "Sentinel-2 TIFF with 13 spectral bands\n"
            "Example: EuroSATallBands folder → *.tif"
        )

    def validate_image_for_mode(self, path):
        """Return (ok, error_message). Rejects wrong format for selected mode."""
        mode = self.ds_menu.get()
        ext = self._file_extension(path)
        fname = os.path.basename(path)

        if mode == "rgb":
            if ext in MS_EXTENSIONS:
                return False, (
                    "Wrong image type!\n\n"
                    f"Selected mode: RGB\n"
                    f"Uploaded file: {fname} (Multispectral TIFF)\n\n"
                    "Please upload:\n"
                    f"  • {self._expected_image_hint('rgb')}\n\n"
                    "Or switch Operational Platform to 'multispectral'."
                )
            if ext not in RGB_EXTENSIONS:
                return False, (
                    "Unsupported file format!\n\n"
                    f"Selected mode: RGB\n"
                    f"Uploaded file: {fname}\n\n"
                    "Please upload:\n"
                    f"  • {self._expected_image_hint('rgb')}\n\n"
                    f"Allowed extensions: {', '.join(sorted(RGB_EXTENSIONS))}"
                )
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    if w < 8 or h < 8:
                        return False, (
                            "Invalid RGB image!\n\n"
                            f"File: {fname}\n"
                            f"Size: {w}×{h} (too small)\n\n"
                            "Please upload a valid EuroSAT RGB image (JPEG/PNG)."
                        )
            except Exception as e:
                return False, (
                    "Cannot read RGB image!\n\n"
                    f"File: {fname}\n"
                    f"Error: {e}\n\n"
                    "Please upload a valid JPEG or PNG file."
                )
            return True, ""

        # multispectral
        if ext in RGB_EXTENSIONS:
            return False, (
                "Wrong image type!\n\n"
                f"Selected mode: Multispectral\n"
                f"Uploaded file: {fname} (RGB image)\n\n"
                "Please upload:\n"
                f"  • {self._expected_image_hint('multispectral')}\n\n"
                "Or switch Operational Platform to 'rgb'."
            )
        if ext not in MS_EXTENSIONS:
            return False, (
                "Unsupported file format!\n\n"
                f"Selected mode: Multispectral\n"
                f"Uploaded file: {fname}\n\n"
                "Please upload:\n"
                f"  • {self._expected_image_hint('multispectral')}\n\n"
                f"Allowed extensions: {', '.join(sorted(MS_EXTENSIONS))}"
            )
        try:
            with rasterio.open(path) as src:
                bands = src.count
                if bands < MS_REQUIRED_BANDS:
                    return False, (
                        "Invalid multispectral image!\n\n"
                        f"File: {fname}\n"
                        f"Bands found: {bands}\n"
                        f"Bands required: {MS_REQUIRED_BANDS}\n\n"
                        "Please upload a Sentinel-2 TIFF from EuroSATallBands\n"
                        "(13-band multispectral image)."
                    )
        except Exception as e:
            return False, (
                "Cannot read multispectral image!\n\n"
                f"File: {fname}\n"
                f"Error: {e}\n\n"
                "Please upload a valid 13-band .tif file."
            )
        return True, ""

    def load_model_action(self):
        path = filedialog.askopenfilename(filetypes=[("H5 Model", "*.h5")])
        if path:
            try:
                self.loaded_model = load_model(path)
                messagebox.showinfo("Ready", "Expert System Initialized.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def make_gradcam(self, img_array, model, last_conv_name, pred_idx=None):
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            if pred_idx is None: pred_idx = tf.argmax(preds[0])
            score = preds[:, pred_idx]
        grads = tape.gradient(score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def select_predict_image(self):
        mode = self.ds_menu.get()
        if mode == "rgb":
            filetypes = [("RGB Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        else:
            filetypes = [("Multispectral TIFF", "*.tif *.tiff"), ("All files", "*.*")]

        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return

        ok, err = self.validate_image_for_mode(path)
        if not ok:
            self.log(f"[ERROR] Image rejected: {os.path.basename(path)} — wrong type for '{mode}' mode.")
            messagebox.showerror("Wrong Image Type", err)
            return

        try:
            if self._file_extension(path) in MS_EXTENSIONS:
                with rasterio.open(path) as src:
                    r = src.read(4); g = src.read(3); b = src.read(2)
                    rgb = np.dstack((r, g, b))
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
                    disp_img = Image.fromarray(rgb.astype('uint8'))
            else:
                disp_img = Image.open(path).convert("RGB")
            
            d_img = disp_img.resize((300, 300))
            ctk_img = ctk.CTkImage(light_image=d_img, dark_image=d_img, size=(300, 300))
            self.orig_preview.configure(image=ctk_img, text="")
            
            if self.loaded_model: self.predict(path, disp_img)
            else: messagebox.showwarning("Warning", "Knowledge Base Missing.")
        except Exception as e: messagebox.showerror("IO Error", f"Failed: {e}")

    def start_training_flow(self):
        # Implementation from previous steps...
        pass

    def predict(self, path, disp_img):
        classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", 
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        mode = self.ds_menu.get()

        ok, err = self.validate_image_for_mode(path)
        if not ok:
            messagebox.showerror("Wrong Image Type", err)
            return

        try:
            # 1. Processing Input
            if mode == "rgb":
                x = np.array(disp_img.resize((64, 64))) / 255.0
            else:
                with rasterio.open(path) as src:
                    x = src.read().transpose(1, 2, 0).astype(np.float32)
                means = np.array([1354.4,1118.2,1042.9,947.6,1199.5,1999.8,2369.2,2296.8,732.1,12.1,1819.0,1118.9,2594.1]).reshape(1,1,13)
                stds = np.array([245.7,333.0,395.1,593.8,566.4,861.2,1086.6,1118.0,404.9,4.8,1002.6,761.3,1231.6]).reshape(1,1,13)
                x = (x - means) / (stds + 1e-8)
            
            img_arr = np.expand_dims(x, axis=0)
            preds = self.loaded_model.predict(img_arr, verbose=0)
            idx = np.argmax(preds[0]); class_name = classes[idx]; conf = float(preds[0][idx] * 100)
            
            # 2. Grad-CAM Calculation
            try:
                base_model = self.loaded_model.layers[0] # assuming ResNet50V2 is first
                heatmap = self.make_gradcam(img_arr, base_model, 'post_relu', pred_idx=idx)
                jet_map = cm.get_cmap("jet")(np.uint8(255 * heatmap))[:, :, :3]
                jet_img = Image.fromarray(np.uint8(255 * jet_map)).resize((300, 300))
                overlay = Image.blend(disp_img.resize((300, 300)), jet_img, alpha=0.5)
                cam_img = ctk.CTkImage(light_image=overlay, dark_image=overlay, size=(300, 300))
                self.cam_preview.configure(image=cam_img, text="")
            except: 
                self.cam_preview.configure(text="Grad-CAM Fail")

            # 3. NDVI Analysis
            ndvi_val = 0.0
            if mode == "multispectral":
                with rasterio.open(path) as src:
                    b4 = src.read(4).astype(float); b8 = src.read(8).astype(float)
                    ndvi_val = float(np.mean((b8 - b4) / (b8 + b4 + 1e-8)))
                self.res_ndvi.configure(text=f"Biophysical Activity (NDVI): {ndvi_val:.2f}")

            # 4. Expert Decision
            final = "NOT SUITABLE"; color = "#e74c3c"
            suitability_msg = "Area is unsuitable for agricultural activity based on land usage class or low spectral index values."
            
            if class_name in ["AnnualCrop", "PermanentCrop"]:
                if mode == "multispectral" and ndvi_val > 0.4:
                    final = "HIGHLY SUITABLE"; color = "#2ecc71"
                    suitability_msg = f"Soil classified as {class_name} with strong vegetative signals (NDVI {ndvi_val:.2f}). Highly recommended for precision irrigation."
                else:
                    final = "SUITABLE"; color = "#27ae60"
                    suitability_msg = f"Visual land usage matches Arable Class: {class_name}. Good potential."

            # UI Update
            self.res_class.configure(text=f"Detected Land Usage: {class_name} ({conf:.1f}%)")
            self.res_final.configure(text=f"DECISION: {final}", text_color=color)
            self.final_desc.delete("0.0", "end"); self.final_desc.insert("0.0", f"Expert Assessment:\n{suitability_msg}")

            # Store for Export
            self.current_analysis_data = {
                "class": class_name, "conf": conf, "ndvi": ndvi_val, "final": final,
                "msg": suitability_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "path": path, "mode": mode
            }
            self.export_btn.configure(state="normal")

        except Exception as e: messagebox.showerror("Analysis Error", str(e))

    def export_report_action(self):
        if not self.current_analysis_data: return
        data = self.current_analysis_data
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Report", "*.pdf")])
        if not save_path: return
        
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 20)
            pdf.cell(200, 10, "SMART AGRI-SAT AI REPORT", ln=1, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 10, f"Generated on: {data['date']}", ln=1, align='C')
            pdf.ln(10)
            
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "1. INGESTION METADATA", ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 8, f"Source File: {os.path.basename(data['path'])}", ln=1)
            pdf.cell(200, 8, f"Analysis Mode: {data['mode'].upper()}", ln=1)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "2. AI PREDICTION RESULTS", ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 8, f"Classification: {data['class']}", ln=1)
            pdf.cell(200, 8, f"Confidence Score: {data['conf']:.2f}%", ln=1)
            pdf.cell(200, 8, f"NDVI Score: {data['ndvi']:.2f}", ln=1)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "3. EXPERT DECISION", ln=1)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, f"FINAL SCORE: {data['final']}", ln=1)
            pdf.set_font("Arial", 'I', 12)
            pdf.multi_cell(0, 8, data['msg'])
            
            pdf.ln(20)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(200, 10, "This report is generated automatically by the Agri-Sat AI Decision Engine.", ln=1, align='C')
            
            pdf.output(save_path)
            messagebox.showinfo("Success", f"Professional report saved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

if __name__ == "__main__":
    app = SmartAgriSys_GUI()
    app.mainloop()
