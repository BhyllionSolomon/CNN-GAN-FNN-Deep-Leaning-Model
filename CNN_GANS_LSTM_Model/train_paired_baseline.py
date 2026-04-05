import os, re, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

RIPE_DIR = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\train\Ripe"
OCC_DIR  = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\train\Occluded"

VAL_RIPE_DIR = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\val\Ripe"
VAL_OCC_DIR  = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\val\Occluded"

OUT_DIR = r"C:\..PhD Thesis\DataSet\GANS\paired_baseline_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Pairing by ID
# =========================
def list_ids(dir_path, prefix):
    ids = []
    for f in os.listdir(dir_path):
        if not f.lower().endswith((".jpg",".jpeg",".png")):
            continue
        stem = os.path.splitext(f)[0]
        m = re.fullmatch(rf"{prefix}_(\d+)", stem)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def load_pair(occ_path, ripe_path):
    occ = cv2.imread(occ_path)
    ripe = cv2.imread(ripe_path)
    if occ is None or ripe is None:
        return None

    occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)
    ripe = cv2.cvtColor(ripe, cv2.COLOR_BGR2RGB)

    occ = cv2.resize(occ, IMG_SIZE).astype(np.float32) / 255.0
    ripe = cv2.resize(ripe, IMG_SIZE).astype(np.float32) / 255.0
    return occ, ripe

def make_dataset(occ_dir, ripe_dir, max_items=None, shuffle=True):
    occ_ids = set(list_ids(occ_dir, "Occluded"))
    ripe_ids = set(list_ids(ripe_dir, "Ripe"))
    common = sorted(list(occ_ids & ripe_ids))
    if shuffle:
        random.shuffle(common)
    if max_items:
        common = common[:max_items]

    X = []
    Y = []
    kept = 0
    for id_ in common:
        occ_path  = os.path.join(occ_dir,  f"Occluded_{id_}.jpg")
        ripe_path = os.path.join(ripe_dir, f"Ripe_{id_}.jpg")
        if not os.path.exists(occ_path):
            occ_path = os.path.join(occ_dir, f"Occluded_{id_}.png")
        if not os.path.exists(ripe_path):
            ripe_path = os.path.join(ripe_dir, f"Ripe_{id_}.png")

        pair = load_pair(occ_path, ripe_path)
        if pair is None:
            continue
        occ, ripe = pair
        X.append(occ)
        Y.append(ripe)
        kept += 1

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    print(f"Paired dataset from {occ_dir} -> {ripe_dir}: {kept} pairs")
    return X, Y

# =========================
# Simple U-Net-ish generator
# =========================
def build_generator():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # encoder
    x1 = layers.Conv2D(64, 4, strides=2, padding="same")(inp); x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(128, 4, strides=2, padding="same")(x1); x2 = layers.BatchNormalization()(x2); x2 = layers.ReLU()(x2)
    x3 = layers.Conv2D(256, 4, strides=2, padding="same")(x2); x3 = layers.BatchNormalization()(x3); x3 = layers.ReLU()(x3)
    x4 = layers.Conv2D(512, 4, strides=2, padding="same")(x3); x4 = layers.BatchNormalization()(x4); x4 = layers.ReLU()(x4)

    # bottleneck
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(x4)
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(b)

    # decoder + skip connections
    d3 = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(b); d3 = layers.BatchNormalization()(d3); d3 = layers.ReLU()(d3)
    d3 = layers.Concatenate()([d3, x3])

    d2 = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(d3); d2 = layers.BatchNormalization()(d2); d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, x2])

    d1 = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(d2); d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, x1])

    out = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(d1); out = layers.ReLU()(out)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(out)

    return Model(inp, out, name="paired_inpaint_generator")

# =========================
# Debug visualization
# =========================
def save_samples(model, X, Y, epoch, n=6):
    n = min(n, len(X))
    idxs = np.random.choice(len(X), n, replace=False)
    preds = model.predict(X[idxs], verbose=0)

    plt.figure(figsize=(12, 2*n))
    for r, i in enumerate(range(n)):
        occ = X[idxs[i]]
        gt  = Y[idxs[i]]
        pr  = preds[i]

        plt.subplot(n, 3, r*3+1); plt.imshow(occ); plt.axis("off"); plt.title("Occluded")
        plt.subplot(n, 3, r*3+2); plt.imshow(pr);  plt.axis("off"); plt.title("Pred")
        plt.subplot(n, 3, r*3+3); plt.imshow(gt);  plt.axis("off"); plt.title("Ripe GT")

    path = os.path.join(OUT_DIR, f"samples_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

# =========================
# Train
# =========================
def main():
    X_train, Y_train = make_dataset(OCC_DIR, RIPE_DIR, max_items=None, shuffle=True)
    X_val, Y_val     = make_dataset(VAL_OCC_DIR, VAL_RIPE_DIR, max_items=300, shuffle=False)

    model = build_generator()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.MeanAbsoluteError()
    )
    model.summary()

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=1, verbose=1)
        # quick val
        val_loss = model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE, verbose=0)
        print("val_loss:", float(val_loss))
        save_samples(model, X_val, Y_val, epoch, n=6)

    model.save(os.path.join(OUT_DIR, "paired_baseline_generator.keras"))
    print("Saved model to:", os.path.join(OUT_DIR, "paired_baseline_generator.keras"))

if __name__ == "__main__":
    main()