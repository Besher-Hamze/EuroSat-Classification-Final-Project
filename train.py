"""
EuroSAT Classification Training Script (TensorFlow/Keras)
========================================================
Supports:
  - EuroSAT        : RGB JPEG images (3-band)
  - EuroSATallBands: Sentinel-2 multispectral TIFF images (13-band)

Models trained:
  1. ResNet50V2  with ImageNet pretrained weights  (transfer learning)
  2. Custom CNN  built from scratch

Outputs:
  - best_model_resnet.h5
  - best_model_custom.h5
  - history_resnet.json
  - history_custom.json
  - curves_resnet.png
  - curves_custom.png
  - comparison_accuracy.png
  - comparison_loss.png
  - comparison_bar.png
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from types import SimpleNamespace
import rasterio

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = "/kaggle/input/eurosat-dataset"   # read-only  — dataset lives here
OUT_ROOT  = "/kaggle/working"                 # writable   — all outputs go here

IMAGE_SIZE  = (64, 64)
NUM_CLASSES = 10

DATASET_CONFIG = {
    "rgb": {
        "root":        os.path.join(DATA_ROOT, "EuroSAT"),
        "train_csv":   os.path.join(DATA_ROOT, "EuroSAT", "train.csv"),
        "val_csv":     os.path.join(DATA_ROOT, "EuroSAT", "validation.csv"),
        "test_csv":    os.path.join(DATA_ROOT, "EuroSAT", "test.csv"),
        "in_channels": 3,
    },
    "multispectral": {
        "root":        os.path.join(DATA_ROOT, "EuroSATallBands"),
        "train_csv":   os.path.join(DATA_ROOT, "EuroSATallBands", "train.csv"),
        "val_csv":     os.path.join(DATA_ROOT, "EuroSATallBands", "validation.csv"),
        "test_csv":    os.path.join(DATA_ROOT, "EuroSATallBands", "test.csv"),
        "in_channels": 13,
    },
}

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

COLORS = {
    "resnet": "#2E86AB",   # blue
    "custom": "#3BB273",   # green
}
LABELS = {
    "resnet": "ResNet50V2 (ImageNet pretrained)",
    "custom": "Custom CNN (from scratch)",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class EuroSATGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, root_dir, batch_size=32, mode="rgb", augment=False):
        df = pd.read_csv(csv_path)
        if df.columns[0] not in ("Filename", "filename"):
            df = df.iloc[:, 1:]
        self.df         = df.reset_index(drop=True)
        self.root_dir   = root_dir
        self.batch_size = batch_size
        self.mode       = mode
        self.augment    = augment

        self.means = np.array([
            1354.40, 1118.24, 1042.92,  947.63, 1199.47,
            1999.79, 2369.22, 2296.82,  732.08,   12.11,
            1819.01, 1118.92, 2594.14,
        ], dtype=np.float32).reshape(1, 1, 13)

        self.stds = np.array([
            245.72, 333.00, 395.09, 593.75,  566.43,
            861.18, 1086.63, 1117.98, 404.92,   4.77,
            1002.58, 761.30, 1231.58,
        ], dtype=np.float32).reshape(1, 1, 13)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for _, row in batch.iterrows():
            img_path = os.path.join(self.root_dir, row["Filename"])

            if self.mode == "rgb":
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=IMAGE_SIZE
                )
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            else:
                with rasterio.open(img_path) as src:
                    img = src.read().transpose(1, 2, 0).astype(np.float32)
                img = (img - self.means) / (self.stds + 1e-8)

            if self.augment:
                if np.random.rand() > 0.5: img = np.fliplr(img)
                if np.random.rand() > 0.5: img = np.flipud(img)

            images.append(img)
            labels.append(int(row["Label"]))

        return np.array(images), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet(in_channels=3):
    """ResNet50V2 with ImageNet weights — transfer learning."""
    base = applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(64, 64, in_channels),
    )
    base.trainable = False  # freeze backbone

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="ResNet50V2_Pretrained")
    return model


def build_custom_cnn(in_channels=3):
    """
    Custom CNN built entirely from scratch.
    Architecture:
        Conv(32)  → BN → ReLU → MaxPool
        Conv(64)  → BN → ReLU → MaxPool
        Conv(128) → BN → ReLU → MaxPool
        Conv(256) → BN → ReLU → GlobalAvgPool
        Dense(256) → Dropout(0.4) → Dense(NUM_CLASSES, softmax)
    """
    inp = layers.Input(shape=(64, 64, in_channels))

    x = layers.Conv2D(32, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inp, out, name="Custom_CNN")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(model, tag, train_gen, val_gen, out_dir, args):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(out_dir, f"best_model_{tag}.h5")
    cbs = [
        callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True,
            monitor="val_accuracy", verbose=1,
        ),
        callbacks.EarlyStopping(
            patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    print(f"\n{'='*60}")
    print(f"  Training : {model.name}")
    print(f"{'='*60}")
    t0 = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=1,
    )

    print(f"\n  Done in {(time.time()-t0)/60:.1f} min  |  saved → {ckpt_path}")

    with open(os.path.join(out_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    return history, model


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_val_curves(history, tag, out_dir):
    """Individual accuracy + loss curves (train vs val) for one model."""
    hist   = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{LABELS[tag]} — Training Curves",
                 fontsize=13, fontweight="bold")

    # — Accuracy —
    axes[0].plot(epochs, hist["accuracy"],
                 label="Train", color=COLORS[tag], linewidth=2)
    axes[0].plot(epochs, hist["val_accuracy"],
                 label="Validation", color=COLORS[tag],
                 linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # — Loss —
    axes[1].plot(epochs, hist["loss"],
                 label="Train", color=COLORS[tag], linewidth=2)
    axes[1].plot(epochs, hist["val_loss"],
                 label="Validation", color=COLORS[tag],
                 linewidth=2, linestyle="--")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"curves_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")


def plot_comparison_metric(histories, metric, out_dir):
    """Val-metric curves for both models on the same axes."""
    ylabel = "Validation Accuracy" if metric == "accuracy" else "Validation Loss"
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag, hist in histories.items():
        vals = hist.history[f"val_{metric}"]
        ax.plot(range(1, len(vals) + 1), vals,
                marker="o", markersize=3, linewidth=2,
                color=COLORS[tag], label=LABELS[tag])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Model Comparison — {ylabel}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")


def plot_bar_comparison(test_results, out_dir):
    """Side-by-side bar chart of final test accuracy."""
    tags = list(test_results.keys())
    accs = [test_results[t] * 100 for t in tags]
    lbls = [LABELS[t] for t in tags]
    cols = [COLORS[t] for t in tags]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lbls, accs, color=cols, width=0.45,
                  edgecolor="white", linewidth=1.2)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Final Test Accuracy — ResNet50V2 vs Custom CNN",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "comparison_bar.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    cfg     = DATASET_CONFIG[args.dataset]
    out_dir = os.path.join(OUT_ROOT, f"outputs_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    in_ch = cfg["in_channels"]

    # ── Generators ────────────────────────────────────────────────────────────
    train_gen = EuroSATGenerator(cfg["train_csv"], cfg["root"],
                                 batch_size=args.batch_size,
                                 mode=args.dataset, augment=True)
    val_gen   = EuroSATGenerator(cfg["val_csv"],   cfg["root"],
                                 batch_size=args.batch_size,
                                 mode=args.dataset, augment=False)
    test_gen  = EuroSATGenerator(cfg["test_csv"],  cfg["root"],
                                 batch_size=args.batch_size,
                                 mode=args.dataset, augment=False)

    # ── Models ────────────────────────────────────────────────────────────────
    model_defs = {
        "resnet": build_resnet(in_ch),
        "custom": build_custom_cnn(in_ch),
    }

    for tag, m in model_defs.items():
        print(f"\n--- {LABELS[tag]} ---")
        m.summary(line_length=80)

    # ── Train ─────────────────────────────────────────────────────────────────
    histories    = {}
    test_results = {}

    for tag, model in model_defs.items():
        hist, trained = train_one_model(
            model, tag, train_gen, val_gen, out_dir, args
        )
        histories[tag] = hist
        plot_train_val_curves(hist, tag, out_dir)

        print(f"\n  Evaluating {LABELS[tag]} ...")
        loss, acc = trained.evaluate(test_gen, verbose=0)
        test_results[tag] = acc
        print(f"  Test Accuracy : {acc*100:.2f}%  |  Loss : {loss:.4f}")

    # ── Comparison plots ──────────────────────────────────────────────────────
    print("\n\nGenerating comparison plots...")
    plot_comparison_metric(histories, "accuracy", out_dir)
    plot_comparison_metric(histories, "loss",     out_dir)
    plot_bar_comparison(test_results, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  FINAL RESULTS  —  {args.dataset.upper()}")
    print("="*60)
    for tag, acc in test_results.items():
        bv = max(histories[tag].history["val_accuracy"])
        print(f"  {LABELS[tag]:<42}  "
              f"test={acc*100:.2f}%   best_val={bv*100:.2f}%")
    print("="*60)
    print(f"\n  All outputs → {out_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

args = SimpleNamespace(
    dataset    = "rgb",   # "rgb"  or  "multispectral"
    epochs     = 30,
    batch_size = 32,
    lr         = 1e-4,
    patience   = 7,
)

train(args)