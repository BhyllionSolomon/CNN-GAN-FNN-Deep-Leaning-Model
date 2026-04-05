"""
=============================================================================
  VISUAL FEATURE DOMINANCE IN CNN-BASED OCCLUSION CLASSIFICATION
  OF ROMA TOMATOES: A CONTROLLED FEATURE ABLATION FRAMEWORK
  ─────────────────────────────────────────────────────────────
  Author  : Olagunju Korede Solomon (Matric: 216882)
  Supervisor: Prof. S.O. Akinola
  University of Ibadan, Nigeria
=============================================================================

  HYPOTHESIS:
    A CNN trained to classify partially occluded Roma tomatoes relies
    primarily on the visual appearance of the occluder rather than on
    the incomplete visibility of the tomato itself, and this finding
    holds consistently across occluder type and CNN architecture.

  STUDY STRUCTURE:
  ─────────────────────────────────────────────────────────────
  PHASE 1 — Dataset Preparation
    Reads ripe + occluded images from your confirmed folders.
    Tags each occluded image with its occluder type from filename.
    Builds train/val/test split (70/15/15).

  PHASE 2 — CNN Training (3 variants)
    CNN-A: dropout=0.4, filters=32/64/128  (baseline)
    CNN-B: dropout=0.2, filters=32/64/128  (low dropout)
    CNN-C: dropout=0.4, filters=16/32/64   (narrow architecture)
    Each variant saved as model_A.keras, model_B.keras, model_C.keras
    Training curves saved per variant.

  PHASE 3 — Experiment 1: Feature Dominance Test
    For every occluded test image, across all 3 CNN variants:
      Baseline  : original image → confidence recorded
      M1 Colour : greyscale image → confidence drop measured
      M2 Texture: blurred image   → confidence drop measured
      M3 Shape  : dilated boundary → confidence drop measured
      M4 Boundary: feathered edge  → confidence drop measured
      M5 Occluder colour: occluder pixels → neutral grey
    Confidence Drop = Baseline Confidence − Modified Confidence
    Result: ranked feature importance table with mean ± std

  PHASE 4 — Experiment 2: Per-Occluder-Type Analysis
    Separates test images by occluder type (leaf/hand/stem/
    tomato/object) using filename tags.
    Runs Experiment 1 per occluder type.
    Answers: does dominant feature change by occluder type?

  PHASE 5 — Experiment 3: Framework Repeatability
    Runs Experiments 1+2 on all three CNN variants.
    Compares: is the dominant feature consistent across models?
    This proves the finding is about the visual data,
    not an artefact of a specific model configuration.

  HOW TO RUN:
    pip install tensorflow opencv-python numpy matplotlib
                seaborn pandas tqdm scikit-learn scipy
    python Feature_Dominance_Study.py

  PATHS — edit the PATHS section below to match your machine.
  SKIP TRAINING — set SKIP_TRAINING = True if models already exist.
=============================================================================
"""

import os, json, csv, time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from scipy import stats

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
#  PATHS  — edit these to match your machine
# ═════════════════════════════════════════════════════════════════════════════
TRAIN_DIR  = Path(r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\train")
VAL_DIR    = Path(r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\val")
TEST_DIR   = Path(r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\test")
OUTPUT_DIR = Path(r"C:\..PhD Thesis\DataSet\Feature_Dominance_Study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  STUDY SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 0.001
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
SKIP_TRAINING = False   # set True to skip training if models already exist

# Occluder type keywords matched against filename stems
OCCLUDER_TYPES = {
    "leaf":    ["leaf"],
    "hand":    ["hand"],
    "stem":    ["stem"],
    "tomato":  ["occ1", "occ2", "tomato"],
    "object":  ["object", "net", "basket", "slat"],
}

# 3 CNN variants for the study
CNN_VARIANTS = [
    {"name": "CNN-A", "tag": "A",
     "dropout": 0.4, "filters": (32, 64, 128),
     "label": "Baseline\n(dropout=0.4, filters=32/64/128)"},
    {"name": "CNN-B", "tag": "B",
     "dropout": 0.2, "filters": (32, 64, 128),
     "label": "Low Dropout\n(dropout=0.2, filters=32/64/128)"},
    {"name": "CNN-C", "tag": "C",
     "dropout": 0.4, "filters": (16, 32, 64),
     "label": "Narrow Filters\n(dropout=0.4, filters=16/32/64)"},
]

# 5 controlled modifications
MODIFICATIONS = [
    ("M1_Colour",   "Colour\nRemoval"),
    ("M2_Texture",  "Texture\nRemoval"),
    ("M3_Shape",    "Shape\nRemoval"),
    ("M4_Boundary", "Boundary\nRemoval"),
    ("M5_OccColour","Occluder\nColour Neutral"),
]

PLOT_COLORS = {
    "CNN-A": "#2980B9",
    "CNN-B": "#27AE60",
    "CNN-C": "#E74C3C",
    "leaf":    "#27AE60",
    "hand":    "#F39C12",
    "stem":    "#795548",
    "tomato":  "#E74C3C",
    "object":  "#7F8C8D",
    "unknown": "#BDC3C7",
}


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def save_plot(fig, name):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {name}.png")


def load_img(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0


def get_occluder_type(stem):
    stem_lower = stem.lower()
    for occ_type, keywords in OCCLUDER_TYPES.items():
        if any(kw in stem_lower for kw in keywords):
            return occ_type
    return "unknown"


def model_path(tag):
    return OUTPUT_DIR / f"model_{tag}.keras"


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — CNN ARCHITECTURE + TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def build_cnn(dropout, filters, name):
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input")
    x = layers.Conv2D(filters[0], (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(0.001),
                      name="conv1")(inp)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2,2), name="pool1")(x)
    x = layers.Conv2D(filters[1], (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(0.001),
                      name="conv2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2,2), name="pool2")(x)
    x = layers.Conv2D(filters[2], (3,3), padding="same",
                      kernel_regularizer=regularizers.l2(0.001),
                      name="conv3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2,2), name="pool3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001),
                     name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)
    return models.Model(inp, out, name=name)


def train_variant(variant):
    tag   = variant["tag"]
    mpath = model_path(tag)

    if SKIP_TRAINING and mpath.exists():
        print(f"    Skipping training — loading {mpath.name}")
        return tf.keras.models.load_model(str(mpath))

    print(f"\n  Training {variant['name']} "
          f"(dropout={variant['dropout']}, "
          f"filters={variant['filters']})...")

    aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, rotation_range=20,
        width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, fill_mode="nearest"
    )
    plain_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    train_data = aug_gen.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        shuffle=True, seed=42
    )
    val_data = plain_gen.flow_from_directory(
        str(VAL_DIR), target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        shuffle=False
    )

    # Class weights for imbalance
    weights = compute_class_weight(
        "balanced", classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weight = dict(enumerate(weights))

    model = build_cnn(variant["dropout"], variant["filters"],
                      variant["name"])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(class_id=1, name="precision"),
            tf.keras.metrics.Recall(class_id=1, name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(mpath),
            monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / f"training_log_{tag}.csv")
        ),
    ]

    history = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS, callbacks=callbacks,
        class_weight=class_weight, verbose=1
    )

    model.save(str(mpath))
    print(f"    Saved: model_{tag}.keras")
    _plot_training_curves(history, variant)
    return model


def _plot_training_curves(history, variant):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Training History — {variant['name']}\n"
        f"dropout={variant['dropout']}  "
        f"filters={variant['filters']}",
        fontsize=13, fontweight="bold"
    )
    pairs = [
        ("accuracy",  "val_accuracy",  "Accuracy",  axes[0,0]),
        ("loss",      "val_loss",      "Loss",      axes[0,1]),
        ("precision", "val_precision", "Precision", axes[1,0]),
        ("recall",    "val_recall",    "Recall",    axes[1,1]),
    ]
    color = PLOT_COLORS[variant["name"]]
    for tk, vk, title, ax in pairs:
        if tk in history.history:
            ax.plot(history.history[tk], label="Train",
                    color=color, lw=2)
        if vk in history.history:
            ax.plot(history.history[vk], label="Val",
                    color=color, lw=2, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, f"phase2_training_{variant['tag']}")


def phase2_train_all():
    print("\n" + "="*65)
    print("  PHASE 2 — CNN Training (3 variants)")
    print("="*65)
    models_dict = {}
    for v in CNN_VARIANTS:
        models_dict[v["name"]] = train_variant(v)
    return models_dict


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — LOAD TEST SET WITH OCCLUDER TAGS
# ═════════════════════════════════════════════════════════════════════════════
def load_test_set():
    """
    Loads test images. For occluded images, tags each with
    its occluder type from the filename stem.
    Returns list of dicts with path, true_label, occluder_type.
    """
    print("\n" + "="*65)
    print("  PHASE 1 — Loading test set with occluder tags")
    print("="*65)

    test_images = []
    for class_name in ["Ripe", "Occluded"]:
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found")
            continue
        files = [p for p in class_dir.iterdir()
                 if p.suffix.lower() in IMG_EXTS]
        for p in files:
            occ_type = ("ripe_class" if class_name == "Ripe"
                        else get_occluder_type(p.stem))
            test_images.append({
                "path":         p,
                "true_label":   class_name,
                "occluder_type": occ_type,
            })

    ripe_n = sum(1 for x in test_images if x["true_label"]=="Ripe")
    occ_n  = sum(1 for x in test_images if x["true_label"]=="Occluded")
    print(f"  Ripe test images    : {ripe_n}")
    print(f"  Occluded test images: {occ_n}")
    print(f"  Total               : {len(test_images)}")

    # Occluder type breakdown
    type_counts = defaultdict(int)
    for x in test_images:
        if x["true_label"] == "Occluded":
            type_counts[x["occluder_type"]] += 1
    print(f"\n  Occluded images by type:")
    for t, n in sorted(type_counts.items()):
        print(f"    {t:<15} : {n}")

    return test_images


# ═════════════════════════════════════════════════════════════════════════════
#  CONTROLLED MODIFICATIONS  (M1 – M5)
# ═════════════════════════════════════════════════════════════════════════════
def m1_colour_removal(img):
    """
    M1: Remove colour information.
    Converts to greyscale then back to 3-channel.
    If CNN confidence drops → colour was driving classification.
    """
    grey = cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    grey_3ch = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    return grey_3ch.astype(np.float32) / 255.0


def m2_texture_removal(img):
    """
    M2: Remove surface texture.
    Strong Gaussian blur eliminates fine texture while
    preserving overall shape and colour distribution.
    If CNN confidence drops → texture was driving classification.
    """
    blurred = cv2.GaussianBlur(
        (img * 255).astype(np.uint8), (21, 21), 0
    )
    return blurred.astype(np.float32) / 255.0


def m3_shape_removal(img):
    """
    M3: Reduce shape sharpness of the occluder region.
    Applies morphological dilation then erosion to soften
    the occluder boundary while keeping colour intact.
    If CNN confidence drops → shape/outline was important.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img_u8  = (img * 255).astype(np.uint8)
    dilated = cv2.dilate(img_u8, kernel, iterations=2)
    softened= cv2.addWeighted(img_u8, 0.5, dilated, 0.5, 0)
    return softened.astype(np.float32) / 255.0


def m4_boundary_removal(img):
    """
    M4: Remove the edge/boundary between occluder and tomato.
    Detects edges with Canny, then blends/feathers those regions.
    If CNN confidence drops → boundary was driving classification.
    """
    img_u8 = (img * 255).astype(np.uint8)
    grey   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(grey, 50, 150)
    # Dilate edges to create a blending mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edge_mask = cv2.dilate(edges, kernel).astype(np.float32) / 255.0
    # Blur the edge regions
    blurred = cv2.GaussianBlur(img_u8, (15, 15), 0)
    edge_mask_3 = np.stack([edge_mask]*3, axis=-1)
    blended = (img_u8.astype(np.float32) * (1 - edge_mask_3) +
               blurred.astype(np.float32) * edge_mask_3)
    return np.clip(blended, 0, 255).astype(np.float32) / 255.0


def m5_occluder_colour_neutral(img):
    """
    M5: Replace occluder-coloured pixels with neutral grey.
    Detects non-red (non-tomato) regions and neutralises them.
    Tomato pixels (red) are preserved. Everything else → grey.
    If CNN confidence drops → the occluder's specific colour mattered.
    """
    img_u8 = (img * 255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    # Red tomato mask (two hue ranges)
    m1 = cv2.inRange(hsv, np.array([0,70,50]),   np.array([12,255,255]))
    m2 = cv2.inRange(hsv, np.array([165,70,50]), np.array([180,255,255]))
    tomato_mask = cv2.bitwise_or(m1, m2)
    # Invert: non-tomato regions
    non_tomato  = cv2.bitwise_not(tomato_mask)
    result      = img_u8.copy()
    result[non_tomato > 0] = [128, 128, 128]   # neutral grey
    return result.astype(np.float32) / 255.0


MODIFICATION_FUNCTIONS = {
    "M1_Colour":    m1_colour_removal,
    "M2_Texture":   m2_texture_removal,
    "M3_Shape":     m3_shape_removal,
    "M4_Boundary":  m4_boundary_removal,
    "M5_OccColour": m5_occluder_colour_neutral,
}


# ═════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ═════════════════════════════════════════════════════════════════════════════
def get_occluded_confidence(model, img, class_indices):
    """
    Returns the model's confidence that the image is Occluded.
    This is the probability we track across all modifications.
    """
    occ_idx = class_indices.get("Occluded", 1)
    inp     = np.expand_dims(img, 0).astype(np.float32)
    pred    = model.predict(inp, verbose=0)[0]
    return float(pred[occ_idx])


def get_class_indices(model):
    """
    Infers class indices from test directory folder names.
    Alphabetical order matches Keras flow_from_directory.
    """
    folders = sorted([
        f.name for f in TEST_DIR.iterdir() if f.is_dir()
    ])
    return {name: idx for idx, name in enumerate(folders)}


# ═════════════════════════════════════════════════════════════════════════════
#  PHASES 3–5 — THE THREE EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════
def run_experiments(models_dict, test_images):
    """
    Runs all three experiments across all three CNN variants.
    Returns a DataFrame with one row per (image × model × modification).
    """
    print("\n" + "="*65)
    print("  PHASES 3–5 — Running 3 Experiments × 3 CNN Variants")
    print("="*65)

    # Only occluded images are used for the ablation
    occ_images = [x for x in test_images if x["true_label"] == "Occluded"]
    print(f"  Occluded images for ablation: {len(occ_images)}")

    all_records = []

    for variant in CNN_VARIANTS:
        model       = models_dict[variant["name"]]
        class_idx   = get_class_indices(model)
        print(f"\n  Processing {variant['name']} "
              f"({len(occ_images)} images × "
              f"{len(MODIFICATIONS)+1} conditions)...")

        for rec in tqdm(occ_images,
                        desc=f"  {variant['name']}"):
            try:
                img      = load_img(rec["path"])
                baseline = get_occluded_confidence(
                    model, img, class_idx
                )

                row = {
                    "model":        variant["name"],
                    "model_label":  variant["label"],
                    "image":        rec["path"].name,
                    "occluder_type":rec["occluder_type"],
                    "baseline_conf":round(baseline, 6),
                }

                for mod_key, _ in MODIFICATIONS:
                    fn          = MODIFICATION_FUNCTIONS[mod_key]
                    mod_img     = fn(img)
                    mod_conf    = get_occluded_confidence(
                        model, mod_img, class_idx
                    )
                    drop        = baseline - mod_conf
                    row[f"{mod_key}_conf"] = round(mod_conf, 6)
                    row[f"{mod_key}_drop"] = round(drop,     6)

                all_records.append(row)

            except Exception as e:
                print(f"    Skipping {rec['path'].name}: {e}")
                continue

    df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "ablation_results_full.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Full results saved: ablation_results_full.csv")
    print(f"  Total records: {len(df)}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING — EXPERIMENT 1
# ═════════════════════════════════════════════════════════════════════════════
def plot_experiment1(df):
    """
    Experiment 1: Feature Dominance — which modification causes
    the largest confidence drop across the full test set?
    One bar chart per CNN variant + a combined comparison.
    """
    print("\n  Plotting Experiment 1: Feature Dominance...")

    drop_cols  = [f"{m[0]}_drop" for m in MODIFICATIONS]
    mod_labels = [m[1] for m in MODIFICATIONS]

    # ── Per-variant bar charts ────────────────────────────────────────────
    for variant in CNN_VARIANTS:
        sub  = df[df["model"] == variant["name"]]
        means= [sub[c].mean() for c in drop_cols]
        stds = [sub[c].std()  for c in drop_cols]

        print(f"\n  {variant['name']} — Mean Confidence Drop per Modification:")
        print(f"  {'Modification':<25} {'Mean Drop':>12} {'Std':>10}")
        print(f"  {'-'*50}")
        ranked = sorted(zip(mod_labels, means, stds),
                        key=lambda x: -x[1])
        for lbl, mn, sd in ranked:
            print(f"  {lbl.replace(chr(10),' '):<25} {mn:>12.4f} {sd:>10.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = ["#E74C3C" if m == max(means) else "#2980B9"
                   for m in means]
        bars    = ax.bar(mod_labels, means, color=colors,
                         edgecolor="white", linewidth=0.8,
                         yerr=stds, capsize=6,
                         error_kw={"elinewidth": 1.5})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    m + s + 0.005,
                    f"{m:.3f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Confidence Drop\n(Baseline − Modified)",
                      fontsize=12)
        ax.set_title(
            f"Experiment 1: Feature Dominance — {variant['name']}\n"
            f"({variant['label'].replace(chr(10),' ')})\n"
            f"Red bar = dominant feature (largest drop)",
            fontsize=12, fontweight="bold"
        )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        save_plot(fig, f"exp1_feature_dominance_{variant['tag']}")

    # ── Combined 3-model comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    x   = np.arange(len(MODIFICATIONS))
    w   = 0.25
    offsets = [-w, 0, w]

    for i, variant in enumerate(CNN_VARIANTS):
        sub   = df[df["model"] == variant["name"]]
        means = [sub[c].mean() for c in drop_cols]
        stds  = [sub[c].std()  for c in drop_cols]
        bars  = ax.bar(x + offsets[i], means, w,
                       label=variant["name"],
                       color=PLOT_COLORS[variant["name"]],
                       edgecolor="white", linewidth=0.8,
                       yerr=stds, capsize=4,
                       error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(mod_labels, fontsize=10)
    ax.set_ylabel("Mean Confidence Drop (Baseline − Modified)",
                  fontsize=12)
    ax.set_title(
        "Experiment 1 + Experiment 3: Feature Dominance Across 3 CNN Variants\n"
        "If the same modification causes the largest drop in all 3 models → "
        "finding is robust (not model-dependent)",
        fontsize=12, fontweight="bold"
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    save_plot(fig, "exp1_exp3_combined_comparison")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING — EXPERIMENT 2
# ═════════════════════════════════════════════════════════════════════════════
def plot_experiment2(df):
    """
    Experiment 2: Per-occluder-type analysis.
    Does the dominant feature change by occluder type?
    Produces a heatmap: rows=occluder type, cols=modification.
    """
    print("\n  Plotting Experiment 2: Per-Occluder-Type Analysis...")

    drop_cols  = [f"{m[0]}_drop" for m in MODIFICATIONS]
    mod_labels = [m[1].replace("\n"," ") for m in MODIFICATIONS]
    occ_types  = sorted([t for t in df["occluder_type"].unique()
                         if t != "ripe_class"])

    for variant in CNN_VARIANTS:
        sub = df[df["model"] == variant["name"]]

        # Build matrix: rows=occluder type, cols=modification
        matrix = []
        for occ_t in occ_types:
            type_sub = sub[sub["occluder_type"] == occ_t]
            if len(type_sub) == 0:
                matrix.append([0.0]*len(drop_cols))
            else:
                matrix.append([type_sub[c].mean() for c in drop_cols])
        matrix = np.array(matrix)

        if matrix.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, max(4, len(occ_types)*1.2)))
        sns.heatmap(
            matrix,
            xticklabels=mod_labels,
            yticklabels=[t.capitalize() for t in occ_types],
            annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"label": "Mean Confidence Drop"}
        )
        ax.set_title(
            f"Experiment 2: Feature Dominance by Occluder Type — "
            f"{variant['name']}\n"
            f"Each row = one occluder type  |  "
            f"Highest value per row = dominant feature for that type",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Controlled Modification", fontsize=11)
        ax.set_ylabel("Occluder Type", fontsize=11)
        plt.tight_layout()
        save_plot(fig, f"exp2_per_occluder_type_{variant['tag']}")

    # ── Cross-variant consistency heatmap ─────────────────────────────────
    # For each occluder type, which modification is dominant in each model?
    fig, axes = plt.subplots(1, len(CNN_VARIANTS),
                             figsize=(6*len(CNN_VARIANTS), max(4, len(occ_types)*1.2)))
    for ax, variant in zip(axes, CNN_VARIANTS):
        sub    = df[df["model"] == variant["name"]]
        matrix = []
        for occ_t in occ_types:
            type_sub = sub[sub["occluder_type"] == occ_t]
            if len(type_sub) == 0:
                matrix.append([0.0]*len(drop_cols))
            else:
                matrix.append([type_sub[c].mean() for c in drop_cols])
        if not matrix:
            continue
        matrix = np.array(matrix)
        sns.heatmap(
            matrix,
            xticklabels=mod_labels,
            yticklabels=[t.capitalize() for t in occ_types],
            annot=True, fmt=".3f", cmap="Blues",
            linewidths=0.5, ax=ax,
            cbar_kws={"label":"Drop"}
        )
        ax.set_title(variant["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Modification", fontsize=9)
        ax.set_ylabel("Occluder Type", fontsize=9)

    fig.suptitle(
        "Experiment 3: Framework Repeatability — "
        "Same Occluder Type Analysed Across 3 CNN Variants\n"
        "Consistent dominant modification across all 3 models = "
        "finding is about the data, not the model",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save_plot(fig, "exp3_repeatability_heatmaps")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING — PER-INSTANCE PANEL
# ═════════════════════════════════════════════════════════════════════════════
def plot_per_instance_panel(df, test_images, models_dict):
    """
    Selects one representative image per occluder type.
    Shows: original + each modification + confidence drop per modification.
    Uses CNN-A (baseline model).
    This is the key visual for the paper.
    """
    print("\n  Plotting per-instance panel...")

    model       = models_dict["CNN-A"]
    class_idx   = get_class_indices(model)
    occ_types   = [t for t in
                   sorted(set(x["occluder_type"] for x in test_images
                              if x["true_label"]=="Occluded"))
                   if t != "unknown"]

    n_types = len(occ_types)
    n_cols  = 1 + len(MODIFICATIONS)   # original + 5 modifications
    fig, axes = plt.subplots(n_types, n_cols,
                             figsize=(n_cols * 3, n_types * 3.2))
    if n_types == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Per-Instance Feature Ablation — CNN-A (Baseline Model)\n"
        "Each row = one occluder type  |  "
        "Number = confidence drop (higher = this feature matters more)",
        fontsize=12, fontweight="bold"
    )

    col_titles = ["Original"] + [m[1] for m in MODIFICATIONS]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9,
                             fontweight="bold", pad=5)

    for row, occ_t in enumerate(occ_types):
        # Pick first matching test image
        candidates = [x for x in test_images
                      if x["occluder_type"] == occ_t]
        if not candidates:
            continue
        rec = candidates[0]
        try:
            img      = load_img(rec["path"])
            baseline = get_occluded_confidence(model, img, class_idx)
        except Exception:
            continue

        # Original
        axes[row, 0].imshow(img)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(
            occ_t.capitalize(), fontsize=10,
            fontweight="bold", rotation=90, labelpad=6
        )
        axes[row, 0].set_xlabel(
            f"Conf: {baseline:.3f}", fontsize=8
        )

        # Each modification
        for col, (mod_key, mod_label) in enumerate(MODIFICATIONS, start=1):
            fn       = MODIFICATION_FUNCTIONS[mod_key]
            mod_img  = fn(img)
            mod_conf = get_occluded_confidence(model, mod_img, class_idx)
            drop     = baseline - mod_conf

            axes[row, col].imshow(mod_img)
            axes[row, col].axis("off")
            color = "#E74C3C" if drop > 0.1 else \
                    "#E67E22" if drop > 0.03 else "#27AE60"
            axes[row, col].set_xlabel(
                f"Drop: {drop:+.3f}",
                fontsize=9, fontweight="bold", color=color
            )

    plt.tight_layout()
    save_plot(fig, "exp_per_instance_panel")


# ═════════════════════════════════════════════════════════════════════════════
#  STATISTICAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def statistical_summary(df):
    """
    For each model × modification:
      - mean ± std confidence drop
      - t-test against zero (is the drop significant?)
      - rank of modifications (1=dominant)
    Saves to summary table and prints to console.
    """
    print("\n" + "="*65)
    print("  STATISTICAL SUMMARY")
    print("="*65)

    drop_cols  = [f"{m[0]}_drop" for m in MODIFICATIONS]
    mod_names  = [m[0] for m in MODIFICATIONS]
    rows       = []

    for variant in CNN_VARIANTS:
        sub   = df[df["model"] == variant["name"]]
        means = [sub[c].mean() for c in drop_cols]
        stds  = [sub[c].std()  for c in drop_cols]
        # Rank (1 = highest drop = dominant)
        ranked = sorted(range(len(means)),
                        key=lambda i: -means[i])
        ranks  = [0]*len(means)
        for rank, idx in enumerate(ranked, start=1):
            ranks[idx] = rank

        for i, mod in enumerate(mod_names):
            t_stat, p_val = stats.ttest_1samp(
                sub[drop_cols[i]].dropna(), 0
            )
            rows.append({
                "Model":        variant["name"],
                "Modification": mod,
                "Mean_Drop":    round(means[i], 4),
                "Std_Drop":     round(stds[i],  4),
                "Rank":         ranks[i],
                "t_stat":       round(t_stat, 3),
                "p_value":      round(p_val, 6),
                "Significant":  "YES" if p_val < 0.05 else "NO",
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        str(OUTPUT_DIR / "statistical_summary.csv"), index=False
    )

    print("\n  " + "-"*75)
    print(f"  {'Model':<8} {'Modification':<18} "
          f"{'Mean Drop':>10} {'Std':>8} "
          f"{'Rank':>6} {'p-value':>10} {'Sig?':>6}")
    print("  " + "-"*75)
    for _, r in summary_df.iterrows():
        print(f"  {r['Model']:<8} {r['Modification']:<18} "
              f"{r['Mean_Drop']:>10.4f} {r['Std_Drop']:>8.4f} "
              f"{r['Rank']:>6} {r['p_value']:>10.6f} "
              f"{r['Significant']:>6}")
    print("  " + "-"*75)

    # Print dominant feature per model
    print("\n  DOMINANT FEATURE PER MODEL (Rank 1 = largest confidence drop):")
    for variant in CNN_VARIANTS:
        dominant = summary_df[
            (summary_df["Model"]==variant["name"]) &
            (summary_df["Rank"]==1)
        ]["Modification"].values
        print(f"    {variant['name']}: {dominant[0] if len(dominant) else 'N/A'}")

    return summary_df


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  VISUAL FEATURE DOMINANCE STUDY")
    print("  CNN-Based Occlusion Classification of Roma Tomatoes")
    print("  Olagunju Korede Solomon — University of Ibadan")
    print("="*65)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Skip training   : {SKIP_TRAINING}")

    # Phase 1: Load test set with occluder tags
    test_images = load_test_set()

    # Phase 2: Train all 3 CNN variants
    models_dict = phase2_train_all()

    # Phase 3–5: Run all 3 experiments
    df = run_experiments(models_dict, test_images)

    # Plots
    plot_experiment1(df)           # Exp 1 + Exp 3 (feature dominance)
    plot_experiment2(df)           # Exp 2 (per occluder type)
    plot_per_instance_panel(       # visual per-image panel
        df, test_images, models_dict
    )

    # Statistical summary
    summary_df = statistical_summary(df)

    # Final JSON summary
    drop_cols = [f"{m[0]}_drop" for m in MODIFICATIONS]
    final = {
        "timestamp":    datetime.now().isoformat(),
        "hypothesis":   (
            "CNN classifies occluded tomatoes by learning occluder "
            "appearance, not tomato incompleteness"
        ),
        "total_images_tested": len(df),
        "models_tested": [v["name"] for v in CNN_VARIANTS],
        "dominant_feature_per_model": {},
        "outputs": {
            "training_curves": [
                f"phase2_training_{v['tag']}.png"
                for v in CNN_VARIANTS
            ],
            "exp1_plots": [
                f"exp1_feature_dominance_{v['tag']}.png"
                for v in CNN_VARIANTS
            ] + ["exp1_exp3_combined_comparison.png"],
            "exp2_plots": [
                f"exp2_per_occluder_type_{v['tag']}.png"
                for v in CNN_VARIANTS
            ] + ["exp3_repeatability_heatmaps.png"],
            "instance_panel": "exp_per_instance_panel.png",
            "data_files": [
                "ablation_results_full.csv",
                "statistical_summary.csv",
            ],
        }
    }
    for variant in CNN_VARIANTS:
        sub = summary_df[
            (summary_df["Model"]==variant["name"]) &
            (summary_df["Rank"]==1)
        ]
        final["dominant_feature_per_model"][variant["name"]] = (
            sub["Modification"].values[0] if len(sub) else "N/A"
        )

    with open(str(OUTPUT_DIR / "study_summary.json"), "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "="*65)
    print("  STUDY COMPLETE")
    print("="*65)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print()
    print("  FILES PRODUCED:")
    print("  Training:")
    for v in CNN_VARIANTS:
        print(f"    model_{v['tag']}.keras")
        print(f"    phase2_training_{v['tag']}.png")
        print(f"    training_log_{v['tag']}.csv")
    print()
    print("  Experiment 1 (Feature Dominance):")
    for v in CNN_VARIANTS:
        print(f"    exp1_feature_dominance_{v['tag']}.png")
    print("    exp1_exp3_combined_comparison.png")
    print()
    print("  Experiment 2 (Per Occluder Type):")
    for v in CNN_VARIANTS:
        print(f"    exp2_per_occluder_type_{v['tag']}.png")
    print("    exp3_repeatability_heatmaps.png")
    print()
    print("  Per-instance visual:")
    print("    exp_per_instance_panel.png")
    print()
    print("  Data and statistics:")
    print("    ablation_results_full.csv")
    print("    statistical_summary.csv")
    print("    study_summary.json")
    print()
    print("  To skip training on next run:")
    print("    Set SKIP_TRAINING = True at the top of the script")


if __name__ == "__main__":
    main()