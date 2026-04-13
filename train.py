"""
EuroSAT Classification Training Script (TensorFlow/Keras)
========================================================
Supports two datasets:
  - EuroSAT        : RGB JPEG images (3-band)
  - EuroSATallBands: Sentinel-2 multispectral TIFF images (13-band)
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import rasterio

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 10

DATASET_CONFIG = {
    "rgb": {
        "root": os.path.join(PROJECT_ROOT, "datasets", "EuroSAT"),
        "train_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSAT", "train.csv"),
        "val_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSAT", "validation.csv"),
        "test_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSAT", "test.csv"),
        "label_map": os.path.join(PROJECT_ROOT, "datasets", "EuroSAT", "label_map.json"),
        "in_channels": 3,
    },
    "multispectral": {
        "root": os.path.join(PROJECT_ROOT, "datasets", "EuroSATallBands"),
        "train_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSATallBands", "train.csv"),
        "val_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSATallBands", "validation.csv"),
        "test_csv": os.path.join(PROJECT_ROOT, "datasets", "EuroSATallBands", "test.csv"),
        "label_map": os.path.join(PROJECT_ROOT, "datasets", "EuroSATallBands", "label_map.json"),
        "in_channels": 13,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class EuroSATGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, root_dir, batch_size=32, mode="rgb", augment=False):
        df = pd.read_csv(csv_path)
        if df.columns[0] not in ("Filename", "filename"):
            df = df.iloc[:, 1:]
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.mode = mode
        self.augment = augment
        
        # Multispectral normalization stats
        self.means = np.array([
            1354.40, 1118.24, 1042.92, 947.63, 1199.47,
            1999.79, 2369.22, 2296.82, 732.08, 12.11,
            1819.01, 1118.92, 2594.14
        ], dtype=np.float32).reshape(1, 1, 13)
        self.stds = np.array([
            245.72, 333.00, 395.09, 593.75, 566.43,
            861.18, 1086.63, 1117.98, 404.92, 4.77,
            1002.58, 761.30, 1231.58
        ], dtype=np.float32).reshape(1, 1, 13)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        images = []
        labels = []
        
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.root_dir, row["Filename"])
            if self.mode == "rgb":
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            else:
                with rasterio.open(img_path) as src:
                    img = src.read().transpose(1, 2, 0).astype(np.float32)
                img = (img - self.means) / (self.stds + 1e-8)
                
            if self.augment:
                # Basic TF augmentations
                if np.random.rand() > 0.5: img = np.fliplr(img)
                if np.random.rand() > 0.5: img = np.flipud(img)

            images.append(img)
            labels.append(int(row["Label"]))
            
        return np.array(images), np.array(labels)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_model(mode="rgb", in_channels=3):
    if mode == "rgb":
        base_model = applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    else:
        # For multispectral, we manually adapt the input layer
        base_model = applications.ResNet50V2(weights=None, include_top=False, input_shape=(64, 64, 13))
        # Initializing weights for a 13-band model from scratch is hard, 
        # so for this project we'll just train it.
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    return model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    cfg = DATASET_CONFIG[args.dataset]
    out_dir = os.path.join(PROJECT_ROOT, f"outputs_tf_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    train_gen = EuroSATGenerator(cfg["train_csv"], cfg["root"], batch_size=args.batch_size, mode=args.dataset, augment=True)
    val_gen = EuroSATGenerator(cfg["val_csv"], cfg["root"], batch_size=args.batch_size, mode=args.dataset, augment=False)
    test_gen = EuroSATGenerator(cfg["test_csv"], cfg["root"], batch_size=args.batch_size, mode=args.dataset, augment=False)

    model = build_model(mode=args.dataset, in_channels=cfg["in_channels"])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cbs = [
        callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.h5"), save_best_only=True, monitor="val_accuracy"),
        callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]

    print(f"\nStarting TensorFlow Training for {args.dataset}...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cbs
    )

    # Evaluation
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save training info
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rgb", choices=["rgb", "multispectral"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    train(args)
