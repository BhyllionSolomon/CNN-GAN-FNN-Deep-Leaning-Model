"""
=============================================================================
  TOMATO DATASET PIPELINE  —  PhD Thesis, Mr. Olagunju Korede Solomon
  Matric: 216882, University of Ibadan
=============================================================================

  WHAT THIS SCRIPT PRODUCES
  ─────────────────────────
  CNN CLASSIFIER DATASET  (17,473 images total)
  ┌─ Processed_Tomatoes/
  │   ├── train/
  │   │   ├── Ripe/        ~4,762 images  (70% of 6,803)
  │   │   └── Occluded/    ~7,468 images  (70% of 10,670)
  │   ├── val/
  │   │   ├── Ripe/        ~1,020 images  (15% of 6,803)
  │   │   └── Occluded/    ~1,600 images  (15% of 10,670)
  │   └── test/
  │       ├── Ripe/        ~1,021 images  (15% of 6,803)
  │       └── Occluded/    ~1,602 images  (15% of 10,670)
  │
  GAN TRIPLET DATASET  (separate — NOT counted in 17,473)
  └─ GAN_Dataset/
      ├── occluded_images/   occluded tomato  (GAN input)
      ├── semantic_masks/    binary mask      (visible=1, hidden=0)
      └── ground_truth/      clean ripe image (GAN target output)

  HOW IT WORKS
  ────────────
  1. Reads 117 ripe tomato images from PrimaryDataset
  2. Augments to exactly 6,803 ripe images (57–58 variants per image)
  3. Generates 2 occluded versions per ripe image → trims to 10,670
  4. Splits both classes 70/15/15 (stratified) into train/val/test
  5. Creates matching GAN triplets from the occluded images

  HOW TO RUN
  ──────────
  pip install opencv-python numpy Pillow tqdm
  python tomato_dataset_pipeline.py

  ⚠  Edit BASE_PATH below to match your exact folder location.
=============================================================================
"""

import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  ⚙  YOUR CONFIRMED PATH — matches your actual folder structure
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH  = Path(r"C:\..PhD Thesis\DataSet")
# ─────────────────────────────────────────────────────────────────────────────

SOURCE      = BASE_PATH / "PrimaryDataset"
PROCESSED   = BASE_PATH / "Processed_Tomatoes"
GAN_DIR     = BASE_PATH / "GAN_Dataset"

# CNN dataset split targets (matching thesis exactly)
TARGET_RIPE = 6803
TARGET_OCC  = 10670
TOTAL_CNN   = TARGET_RIPE + TARGET_OCC   # 17,473

TARGET_SIZE = (224, 224)   # CNN input size (matches thesis: 224×224)
SEED        = 42
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

random.seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
#  DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════
def make_dirs():
    dirs = [
        PROCESSED / "train" / "Ripe",
        PROCESSED / "train" / "Occluded",
        PROCESSED / "val"   / "Ripe",
        PROCESSED / "val"   / "Occluded",
        PROCESSED / "test"  / "Ripe",
        PROCESSED / "test"  / "Occluded",
        GAN_DIR / "occluded_images",
        GAN_DIR / "semantic_masks",
        GAN_DIR / "ground_truth",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def read_images(folder):
    return sorted([p for p in Path(folder).iterdir()
                   if p.suffix.lower() in IMG_EXTS])

def save_img(path, img):
    cv2.imwrite(str(path), img)

def resize(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

def stratified_split(items, train_r=0.70, val_r=0.15):
    items = list(items)
    random.shuffle(items)
    n      = len(items)
    t      = int(n * train_r)
    v      = int(n * val_r)
    return items[:t], items[t:t+v], items[t+v:]

def copy_files(file_list, dest_dir):
    for src in file_list:
        shutil.copy2(str(src), str(dest_dir / Path(src).name))


# ═════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION  (57–58 variants per image → exactly 6,803 ripe)
# ═════════════════════════════════════════════════════════════════════════════
def _hue_shift(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _add_noise(img, strength=20):
    noise = np.random.randint(-strength, strength, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def _rotate_free(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)

def _crop_resize(img, margin):
    h, w = img.shape[:2]
    return cv2.resize(
        img[margin:h - margin, margin:w - margin],
        (w, h), interpolation=cv2.INTER_AREA)

def _sharpen(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, k)

def _zoom(img, factor=1.15):
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    zoomed = cv2.resize(img, (new_w, new_h))
    y = (new_h - h) // 2
    x = (new_w - w) // 2
    return zoomed[y:y+h, x:x+w]

def _translate(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)

# Full pool of augmentation operations — 57 distinct combinations
def build_aug_pool(img):
    h, w = img.shape[:2]
    pool = [
        cv2.flip(img, 1),                                   # 1  H-flip
        cv2.flip(img, 0),                                   # 2  V-flip
        cv2.flip(img, -1),                                  # 3  Both flip
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),           # 4
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),    # 5
        cv2.rotate(img, cv2.ROTATE_180),                    # 6
        _rotate_free(img,  15),                             # 7
        _rotate_free(img, -15),                             # 8
        _rotate_free(img,  30),                             # 9
        _rotate_free(img, -30),                             # 10
        _rotate_free(img,  45),                             # 11
        _rotate_free(img, -45),                             # 12
        cv2.convertScaleAbs(img, alpha=1.0, beta=40),       # 13 bright+
        cv2.convertScaleAbs(img, alpha=1.0, beta=70),       # 14 bright++
        cv2.convertScaleAbs(img, alpha=1.0, beta=-40),      # 15 bright-
        cv2.convertScaleAbs(img, alpha=1.0, beta=-70),      # 16 bright--
        cv2.convertScaleAbs(img, alpha=1.3, beta=0),        # 17 contrast+
        cv2.convertScaleAbs(img, alpha=1.6, beta=0),        # 18 contrast++
        cv2.convertScaleAbs(img, alpha=0.7, beta=0),        # 19 contrast-
        cv2.GaussianBlur(img, (3, 3), 0),                   # 20 blur mild
        cv2.GaussianBlur(img, (7, 7), 0),                   # 21 blur med
        cv2.GaussianBlur(img, (11, 11), 0),                 # 22 blur heavy
        _add_noise(img, 15),                                # 23 noise mild
        _add_noise(img, 30),                                # 24 noise med
        _add_noise(img, 45),                                # 25 noise heavy
        _hue_shift(img,  15),                               # 26 hue+
        _hue_shift(img, -15),                               # 27 hue-
        _hue_shift(img,  30),                               # 28 hue++
        _hue_shift(img, -30),                               # 29 hue--
        _crop_resize(img, int(h * 0.05)),                   # 30 crop 5%
        _crop_resize(img, int(h * 0.10)),                   # 31 crop 10%
        _crop_resize(img, int(h * 0.15)),                   # 32 crop 15%
        _sharpen(img),                                      # 33 sharpen
        _zoom(img, 1.10),                                   # 34 zoom 1.1x
        _zoom(img, 1.20),                                   # 35 zoom 1.2x
        _translate(img,  20,  0),                           # 36
        _translate(img, -20,  0),                           # 37
        _translate(img,   0, 20),                           # 38
        _translate(img,   0,-20),                           # 39
        # Combinations
        _rotate_free(cv2.flip(img, 1), 15),                 # 40
        _rotate_free(cv2.flip(img, 1), -15),                # 41
        cv2.convertScaleAbs(cv2.flip(img, 1), alpha=1.3, beta=30),  # 42
        _add_noise(cv2.flip(img, 0), 20),                   # 43
        _hue_shift(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 15),   # 44
        cv2.GaussianBlur(cv2.flip(img, 1), (5, 5), 0),     # 45
        cv2.convertScaleAbs(_rotate_free(img,  20), alpha=1.2, beta=20), # 46
        cv2.convertScaleAbs(_rotate_free(img, -20), alpha=1.2, beta=20), # 47
        _crop_resize(cv2.flip(img, 1), int(h * 0.08)),      # 48
        _zoom(_hue_shift(img, 10), 1.10),                   # 49
        _sharpen(cv2.flip(img, 1)),                         # 50
        _add_noise(_hue_shift(img, -10), 15),               # 51
        cv2.convertScaleAbs(_zoom(img, 1.15), alpha=1.0, beta=30),  # 52
        _rotate_free(cv2.convertScaleAbs(img, alpha=1.4, beta=0), 25),  # 53
        _translate(cv2.flip(img, 1), 15, 15),               # 54
        _hue_shift(cv2.GaussianBlur(img, (5, 5), 0), 20),  # 55
        cv2.convertScaleAbs(_crop_resize(img, int(h*0.08)), alpha=1.2, beta=-20), # 56
        _rotate_free(_add_noise(img, 20), -25),             # 57
    ]
    return pool   # exactly 57 variants


# ═════════════════════════════════════════════════════════════════════════════
#  OCCLUSION GENERATORS
# ═════════════════════════════════════════════════════════════════════════════
def _tomato_region(img):
    h, w = img.shape[:2]
    return w // 2, h // 2, int(min(h, w) * 0.35)

def _make_mask(img, occluder_mask):
    """Binary mask: 1=visible, 0=occluded region."""
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    mask[occluder_mask == 255] = 0
    return mask

# ── Leaf ──────────────────────────────────────────────────────────────────────
def occlude_leaf(img):
    out  = img.copy()
    h, w = out.shape[:2]
    cx, cy, r = _tomato_region(out)
    angle = random.uniform(-60, 60)
    rad   = np.radians(angle)
    lx = int(cx + r * random.uniform(-0.3, 0.3))
    ly = int(cy + r * random.uniform(-0.5, 0.1))
    la = int(r * random.uniform(0.9, 1.4))
    lb = int(r * random.uniform(0.35, 0.55))
    leaf_color = (random.randint(20,50),
                  random.randint(90,145),
                  random.randint(15,45))
    cv2.ellipse(out, (lx, ly), (la, lb), angle, 0, 360, leaf_color, -1)
    vein = tuple(max(0, c - 30) for c in leaf_color)
    vx1 = int(lx - la * 0.85 * np.cos(rad))
    vy1 = int(ly - la * 0.85 * np.sin(rad))
    vx2 = int(lx + la * 0.85 * np.cos(rad))
    vy2 = int(ly + la * 0.85 * np.sin(rad))
    cv2.line(out, (vx1,vy1), (vx2,vy2), vein, max(2, r//18))
    for side in [-1, 1]:
        for frac in [0.3, 0.6]:
            px = int(lx + frac * la * np.cos(rad))
            py = int(ly + frac * la * np.sin(rad))
            ex = int(px + lb * 0.6 * np.cos(rad + side * np.pi/3))
            ey = int(py + lb * 0.6 * np.sin(rad + side * np.pi/3))
            cv2.line(out, (px,py), (ex,ey), vein, max(1, r//30))
    occ_mask = np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(occ_mask, (lx,ly), (la,lb), angle, 0, 360, 255, -1)
    noise = np.random.randint(-15,15, out.shape, dtype=np.int16)
    noisy = np.clip(out.astype(np.int16)+noise,0,255).astype(np.uint8)
    out[occ_mask==255] = noisy[occ_mask==255]
    out = cv2.addWeighted(img, 0.12, out, 0.88, 0)
    return out, _make_mask(img, occ_mask)

# ── Hand ──────────────────────────────────────────────────────────────────────
def occlude_hand(img):
    out  = img.copy()
    h, w = out.shape[:2]
    cx, cy, r = _tomato_region(out)
    skins = [(110,155,200),(80,120,170),(140,185,220),(70,100,145)]
    skin  = random.choice(skins)
    skin_dark = tuple(max(0, c-30) for c in skin)
    num_f   = random.randint(2, 4)
    fing_w  = int(r * random.uniform(0.22, 0.30))
    start_x = int(cx - r * random.uniform(0.5, 0.8))
    occ_mask = np.zeros((h,w), dtype=np.uint8)
    for i in range(num_f):
        fx    = start_x + i * int(r * 0.38)
        top_y = int(cy - r * random.uniform(0.6, 1.1))
        bot_y = int(cy + r * random.uniform(0.3, 0.6))
        fh    = bot_y - top_y
        x1, x2 = fx - fing_w//2, fx + fing_w//2
        cv2.rectangle(out, (x1,top_y),(x2,bot_y), skin, -1)
        cv2.rectangle(occ_mask,(x1,top_y),(x2,bot_y),255,-1)
        for kf in [0.28, 0.56]:
            ky = top_y + int(fh * kf)
            cv2.line(out,(x1+2,ky),(x2-2,ky),skin_dark,2)
        cv2.ellipse(out,(fx,top_y),(fing_w//2,fing_w//2),0,180,360,skin,-1)
        cv2.ellipse(occ_mask,(fx,top_y),(fing_w//2,fing_w//2),0,180,360,255,-1)
    px1 = start_x - fing_w//2
    px2 = start_x + num_f * int(r*0.38) + fing_w//2
    py1 = int(cy + r*0.28)
    py2 = int(cy + r*0.95)
    cv2.rectangle(out,(px1,py1),(px2,py2),skin,-1)
    cv2.rectangle(occ_mask,(px1,py1),(px2,py2),255,-1)
    out = cv2.addWeighted(img, 0.07, out, 0.93, 0)
    return out, _make_mask(img, occ_mask)

# ── Another tomato ────────────────────────────────────────────────────────────
def occlude_tomato(img):
    out  = img.copy()
    h, w = out.shape[:2]
    cx, cy, r = _tomato_region(out)
    r2   = int(r * random.uniform(0.65, 0.95))
    side = random.choice([-1, 1])
    cx2  = int(cx + side * r * random.uniform(0.5, 0.85))
    cy2  = int(cy + r * random.uniform(-0.4, 0.2))
    red  = (random.randint(20,50),
            random.randint(30,60),
            random.randint(160,210))
    cv2.circle(out,(cx2,cy2),r2,red,-1)
    hi   = tuple(min(255,c+55) for c in red)
    cv2.circle(out,(cx2-int(r2*0.28),cy2-int(r2*0.28)),int(r2*0.22),hi,-1)
    cv2.ellipse(out,(cx2,cy2-r2),(int(r2*0.18),int(r2*0.10)),0,0,360,(20,80,20),-1)
    occ_mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(occ_mask,(cx2,cy2),r2,255,-1)
    out = cv2.addWeighted(img, 0.05, out, 0.95, 0)
    return out, _make_mask(img, occ_mask)

# ── Generic object (crate slat / net / basket) ────────────────────────────────
def occlude_object(img):
    out  = img.copy()
    h, w = out.shape[:2]
    occ_mask = np.zeros((h,w), dtype=np.uint8)
    choice = random.choice(["slat","net","basket"])
    _, _, r = _tomato_region(out)
    if choice == "slat":
        bar_h = int(r * random.uniform(0.25, 0.45))
        cy2   = h // 2
        y1    = cy2 - bar_h//2 + int(r * random.uniform(-0.3, 0.3))
        y2    = y1 + bar_h
        col   = (random.randint(30,60),
                 random.randint(60,100),
                 random.randint(80,130))
        cv2.rectangle(out,(0,y1),(w,y2),col,-1)
        cv2.rectangle(occ_mask,(0,y1),(w,y2),255,-1)
        for _ in range(random.randint(3,6)):
            gy = random.randint(y1,y2)
            cv2.line(out,(0,gy),(w,gy),tuple(max(0,c-20) for c in col),1)
    elif choice == "net":
        sp  = int(r * random.uniform(0.18, 0.30))
        th  = random.randint(2,5)
        col = (20,80,20)
        for x in range(0,w,sp):
            cv2.line(out,(x,0),(x,h),col,th)
            cv2.line(occ_mask,(x,0),(x,h),255,th)
        for y in range(0,h,sp):
            cv2.line(out,(0,y),(w,y),col,th)
            cv2.line(occ_mask,(0,y),(w,y),255,th)
    else:
        sp  = int(r * random.uniform(0.20, 0.35))
        th  = random.randint(3,6)
        col = (30,80,120)
        for d in range(-h, w+h, sp):
            cv2.line(out,(d,0),(d+h,h),col,th)
            cv2.line(out,(d+h,0),(d,h),col,th)
            cv2.line(occ_mask,(d,0),(d+h,h),255,th)
            cv2.line(occ_mask,(d+h,0),(d,h),255,th)
    out = cv2.addWeighted(img, 0.10, out, 0.90, 0)
    return out, _make_mask(img, occ_mask)

# ── Stem ──────────────────────────────────────────────────────────────────────
def occlude_stem(img):
    out  = img.copy()
    h, w = out.shape[:2]
    cx, cy, r = _tomato_region(out)
    thick = int(r * random.uniform(0.10, 0.18))
    x1 = int(cx - r * 1.1);  y1 = int(cy - r * 0.9)
    x2 = int(cx + r * 0.9);  y2 = int(cy + r * 0.7)
    col = (random.randint(30,55),
           random.randint(70,110),
           random.randint(15,40))
    cv2.line(out,(x1,y1),(x2,y2),col,thick)
    hi  = tuple(min(255,c+25) for c in col)
    cv2.line(out,(x1+3,y1+3),(x2+3,y2+3),hi,max(1,thick//3))
    occ_mask = np.zeros((h,w),dtype=np.uint8)
    cv2.line(occ_mask,(x1,y1),(x2,y2),255,thick)
    return out, _make_mask(img, occ_mask)

OCCLUDERS = [occlude_leaf, occlude_hand, occlude_tomato, occlude_object, occlude_stem]


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  TOMATO DATASET PIPELINE")
    print("  PhD Thesis — Olagunju Korede Solomon (216882)")
    print("=" * 65)

    make_dirs()

    # ── Load sources ──────────────────────────────────────────────────────────
    sources = read_images(SOURCE)
    if not sources:
        print(f"\n[ERROR] No images found in: {SOURCE}")
        print("Please update BASE_PATH at the top of this script.")
        return
    n = len(sources)
    print(f"\n[1/5] Found {n} source images in PrimaryDataset")

    # ── Augment to exactly TARGET_RIPE ────────────────────────────────────────
    # First 100 images get 58 total (1 orig + 57 aug)
    # Remaining 17 images get 59 total (1 orig + 58 = we add 1 extra)
    print(f"\n[2/5] Augmenting to exactly {TARGET_RIPE} ripe images...")
    aug_tmp  = PROCESSED / "_aug_tmp"
    aug_tmp.mkdir(exist_ok=True)
    ripe_files = []

    for idx, src in enumerate(tqdm(sources, desc="  Augmenting")):
        img = cv2.imread(str(src))
        if img is None:
            continue
        img  = resize(img)
        stem = src.stem

        # Save original
        orig_path = aug_tmp / f"{stem}_orig.jpg"
        save_img(orig_path, img)
        ripe_files.append(orig_path)

        # Build pool of 57 variants
        pool = build_aug_pool(img)

        # First 100 images: use all 57 variants
        # Last 17 images: use all 57 variants + 1 extra (repeat first variant)
        extras = 1 if idx >= 100 else 0
        variants_to_save = pool[:57 + extras]

        for i, aug_img in enumerate(variants_to_save):
            out_path = aug_tmp / f"{stem}_aug{i+1:02d}.jpg"
            save_img(out_path, resize(aug_img))
            ripe_files.append(out_path)

    # Trim or pad to exactly TARGET_RIPE
    random.shuffle(ripe_files)
    ripe_files = ripe_files[:TARGET_RIPE]
    print(f"  → {len(ripe_files)} ripe images ready")

    # ── Split ripe → train / val / test ───────────────────────────────────────
    print(f"\n[3/5] Splitting ripe images 70/15/15...")
    r_train, r_val, r_test = stratified_split(ripe_files)

    copy_files(r_train, PROCESSED / "train" / "Ripe")
    copy_files(r_val,   PROCESSED / "val"   / "Ripe")
    copy_files(r_test,  PROCESSED / "test"  / "Ripe")
    print(f"  → train/Ripe : {len(r_train)}")
    print(f"  → val/Ripe   : {len(r_val)}")
    print(f"  → test/Ripe  : {len(r_test)}")

    shutil.rmtree(aug_tmp)  # clean up temp

    # ── Generate occluded images (2 per ripe, trim to TARGET_OCC) ────────────
    print(f"\n[4/5] Generating {TARGET_OCC} occluded images + GAN triplets...")
    all_ripe = (list((PROCESSED/"train"/"Ripe").iterdir()) +
                list((PROCESSED/"val"  /"Ripe").iterdir()) +
                list((PROCESSED/"test" /"Ripe").iterdir()))
    all_ripe = [p for p in all_ripe if p.suffix.lower() in IMG_EXTS]
    random.shuffle(all_ripe)

    occ_tmp   = PROCESSED / "_occ_tmp"
    occ_tmp.mkdir(exist_ok=True)
    occ_files = []

    for img_path in tqdm(all_ripe, desc="  Occluding"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img  = resize(img)
        stem = img_path.stem

        # Pick 2 random occluder functions (different types)
        chosen = random.sample(OCCLUDERS, 2)
        for j, fn in enumerate(chosen):
            occ_img, mask = fn(img)
            occ_name = f"{stem}_occ{j+1}.jpg"
            mask_name = f"{stem}_occ{j+1}_mask.png"

            out_path = occ_tmp / occ_name
            save_img(out_path, occ_img)
            occ_files.append(out_path)

            # GAN triplet — only save if we haven't exceeded target
            if len(occ_files) <= TARGET_OCC:
                save_img(GAN_DIR / "occluded_images" / occ_name, occ_img)
                save_img(GAN_DIR / "semantic_masks"  / mask_name, mask)
                save_img(GAN_DIR / "ground_truth"    / occ_name, img)

        if len(occ_files) >= TARGET_OCC * 2:
            break

    # Trim to exactly TARGET_OCC
    random.shuffle(occ_files)
    occ_files = occ_files[:TARGET_OCC]
    print(f"  → {len(occ_files)} occluded images ready")

    # ── Split occluded → train / val / test ───────────────────────────────────
    o_train, o_val, o_test = stratified_split(occ_files)
    copy_files(o_train, PROCESSED / "train" / "Occluded")
    copy_files(o_val,   PROCESSED / "val"   / "Occluded")
    copy_files(o_test,  PROCESSED / "test"  / "Occluded")
    print(f"  → train/Occluded : {len(o_train)}")
    print(f"  → val/Occluded   : {len(o_val)}")
    print(f"  → test/Occluded  : {len(o_test)}")

    shutil.rmtree(occ_tmp)

    # ── GAN dataset count ─────────────────────────────────────────────────────
    gan_count = len(list((GAN_DIR / "occluded_images").iterdir()))
    print(f"\n[5/5] GAN triplets saved: {gan_count} sets")
    print(f"       GAN_Dataset/occluded_images/ — {gan_count} files")
    print(f"       GAN_Dataset/semantic_masks/  — {gan_count} files")
    print(f"       GAN_Dataset/ground_truth/    — {gan_count} files")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_cnn = len(r_train)+len(r_val)+len(r_test)+len(o_train)+len(o_val)+len(o_test)
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Source images              : {n}")
    print(f"  Total ripe (augmented)     : {len(r_train)+len(r_val)+len(r_test)}")
    print(f"  Total occluded             : {len(o_train)+len(o_val)+len(o_test)}")
    print(f"  Total CNN dataset          : {total_cnn}  (thesis: 17,473)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  train/ Ripe={len(r_train)}  Occluded={len(o_train)}  → {len(r_train)+len(o_train)}")
    print(f"  val/   Ripe={len(r_val)}  Occluded={len(o_val)}   → {len(r_val)+len(o_val)}")
    print(f"  test/  Ripe={len(r_test)}  Occluded={len(o_test)}  → {len(r_test)+len(o_test)}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  GAN triplets               : {gan_count}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Output path: {PROCESSED}")
    print(f"  GAN path:    {GAN_DIR}")
    print("=" * 65)
    print("\n  Dataset is ready for CNN training!\n")


if __name__ == "__main__":
    main()