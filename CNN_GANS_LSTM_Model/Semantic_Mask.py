#!/usr/bin/env python3
"""
Create train/val/test mask folders AND generate pseudo semantic masks (no manual labeling)
for ripe-red tomato images using HSV red thresholding + morphological cleanup.

Inputs:
  - JPG images in IMAGE_ROOT

Outputs:
  OUT_ROOT/
    train/masks/*.png
    val/masks/*.png
    test/masks/*.png
    _previews/*.jpg
    split_manifest.json

Mask naming:
  image: occluded_gan_0001.jpg
  mask : occluded_gan_0001.png   (same stem)

NOTE:
  These are approximate "pseudo-masks", not ground truth.
"""

import os
import glob
import cv2
import json
import random
import numpy as np
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
IMAGE_ROOT = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\GAN_Dataset\Occluded_GAN"

OUT_ROOT = r"C:\..PhD Thesis\DataSet\Semantic Mask\GAN_Dataset\Occluded_GAN_masks"
TRAIN_DIR = os.path.join(OUT_ROOT, "train", "masks")
VAL_DIR   = os.path.join(OUT_ROOT, "val", "masks")
TEST_DIR  = os.path.join(OUT_ROOT, "test", "masks")
PREVIEW_DIR = os.path.join(OUT_ROOT, "_previews")

# -----------------------------
# Split (IMPORTANT: must sum to 1.0)
# Choose ONE:
#   - 0.70/0.15/0.15  (recommended)
#   - 0.75/0.15/0.10
# -----------------------------
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Save masks resized for training pipelines (224,224). Set to None to keep original size.
MASK_SIZE = (224, 224)

# HSV thresholds for red (two ranges because red wraps around hue)
LOW1  = (0,   70,  50)
HIGH1 = (10,  255, 255)
LOW2  = (170, 70,  50)
HIGH2 = (180, 255, 255)

# Morphology parameters (tune if needed)
OPEN_KERNEL  = 5
CLOSE_KERNEL = 9
MIN_AREA_FRAC = 0.0005  # remove very small components

def ensure_dirs():
    """Create required output directories (no manual creation needed)."""
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, PREVIEW_DIR]:
        os.makedirs(d, exist_ok=True)

def list_images():
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(IMAGE_ROOT, e)))
    files.sort()
    return files

def split_files(files):
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0. Got {total}.")

    rng = random.Random(SPLIT_SEED)
    files = files[:]
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = files[:n_train]
    val   = files[n_train:n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test

def red_tomato_mask_bgr(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array(LOW1, dtype=np.uint8), np.array(HIGH1, dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array(LOW2, dtype=np.uint8), np.array(HIGH2, dtype=np.uint8))
    mask = cv2.bitwise_or(m1, m2)

    if OPEN_KERNEL and OPEN_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if CLOSE_KERNEL and CLOSE_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Remove tiny connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        h, w = mask.shape[:2]
        min_area = int(MIN_AREA_FRAC * h * w)

        cleaned = np.zeros_like(mask)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == lab] = 255
        mask = cleaned

    return mask

def save_overlay_preview(img_bgr, mask, out_path):
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255  # BGR red
    alpha = 0.35

    m = (mask > 0)
    overlay[m] = (overlay[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis = np.concatenate([img_bgr, overlay, mask_bgr], axis=1)
    cv2.imwrite(out_path, vis)

def process_split(files, out_dir, preview_prefix, preview_count=25):
    for i, img_path in enumerate(tqdm(files, desc=f"Writing masks -> {out_dir}")):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        mask = red_tomato_mask_bgr(img_bgr)

        if MASK_SIZE is not None:
            mask = cv2.resize(mask, MASK_SIZE, interpolation=cv2.INTER_NEAREST)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(out_dir, f"{stem}.png")
        cv2.imwrite(mask_path, mask)

        if i < preview_count:
            vis_img = img_bgr
            if MASK_SIZE is not None:
                vis_img = cv2.resize(img_bgr, MASK_SIZE, interpolation=cv2.INTER_AREA)
            prev_path = os.path.join(PREVIEW_DIR, f"{preview_prefix}_{i:03d}_{stem}.jpg")
            save_overlay_preview(vis_img, mask, prev_path)

def main():
    print("🍅 Generating pseudo semantic masks (HSV red threshold) ...")
    print("Images:", IMAGE_ROOT)
    print("Out:", OUT_ROOT)

    ensure_dirs()

    files = list_images()
    if not files:
        print("❌ No images found.")
        return

    train, val, test = split_files(files)
    print(f"✅ Images found: {len(files)}")
    print(f"📊 Split: train={len(train)} val={len(val)} test={len(test)} (seed={SPLIT_SEED})")
    print(f"📐 Mask size: {MASK_SIZE if MASK_SIZE is not None else 'original'}")

    manifest = {
        "image_root": IMAGE_ROOT,
        "out_root": OUT_ROOT,
        "seed": SPLIT_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "mask_size": MASK_SIZE,
        "train": [os.path.basename(p) for p in train],
        "val":   [os.path.basename(p) for p in val],
        "test":  [os.path.basename(p) for p in test],
    }
    with open(os.path.join(OUT_ROOT, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    process_split(train, TRAIN_DIR, "train", preview_count=25)
    process_split(val,   VAL_DIR,   "val",   preview_count=15)
    process_split(test,  TEST_DIR,  "test",  preview_count=15)

    print("\n✅ Done.")
    print("Masks written to:")
    print(" ", TRAIN_DIR)
    print(" ", VAL_DIR)
    print(" ", TEST_DIR)
    print("Preview overlays:")
    print(" ", PREVIEW_DIR)
    print("Manifest:")
    print(" ", os.path.join(OUT_ROOT, "split_manifest.json"))

if __name__ == "__main__":
    main()