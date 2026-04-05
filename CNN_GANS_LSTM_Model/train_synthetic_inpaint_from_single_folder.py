import os, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

# Single folder that contains ONLY images (no train/val subfolders)
DATA_DIR = r"C:\..PhD Thesis\DataSet\Processed_Tomatoes\GAN_Dataset\Occluded_GAN"

# Where outputs will be saved
OUT_DIR  = r"C:\..PhD Thesis\DataSet\GANS\synthetic_inpaint_occluded_gan_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all_images(dir_path):
    files = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    imgs = []
    for f in files:
        p = os.path.join(dir_path, f)
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32)
    print("Loaded:", len(imgs), "images from", dir_path)
    return imgs

def apply_random_occlusion(img, max_holes=6):
    """Farm-like occlusion: random leaf-ish blobs."""
    h, w, _ = img.shape
    out = img.copy()

    num = np.random.randint(1, max_holes + 1)
    for _ in range(num):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(w//20, w//6), np.random.randint(h//20, h//6))
        angle = np.random.randint(0, 180)

        # leaf-ish occluder colors
        color = np.array([
            np.random.uniform(0.05, 0.25),  # R
            np.random.uniform(0.25, 0.60),  # G
            np.random.uniform(0.05, 0.25),  # B
        ], dtype=np.float32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        out[mask > 0] = color

    return out

def make_xy(images):
    X = np.stack([apply_random_occlusion(im) for im in images], axis=0)
    Y = images
    return X, Y

def build_generator():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x1 = layers.Conv2D(64, 4, strides=2, padding="same")(inp); x1 = layers.ReLU()(x1)
    x2 = layers.Conv2D(128, 4, strides=2, padding="same")(x1); x2 = layers.BatchNormalization()(x2); x2 = layers.ReLU()(x2)
    x3 = layers.Conv2D(256, 4, strides=2, padding="same")(x2); x3 = layers.BatchNormalization()(x3); x3 = layers.ReLU()(x3)
    x4 = layers.Conv2D(512, 4, strides=2, padding="same")(x3); x4 = layers.BatchNormalization()(x4); x4 = layers.ReLU()(x4)

    b = layers.Conv2D(512, 3, padding="same", activation="relu")(x4)
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(b)

    d3 = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(b); d3 = layers.BatchNormalization()(d3); d3 = layers.ReLU()(d3)
    d3 = layers.Concatenate()([d3, x3])

    d2 = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(d3); d2 = layers.BatchNormalization()(d2); d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, x2])

    d1 = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(d2); d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, x1])

    out = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(d1); out = layers.ReLU()(out)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(out)

    return Model(inp, out, name="synthetic_inpaint_generator")

def save_samples(model, images, epoch, n=6):
    n = min(n, len(images))
    idxs = np.random.choice(len(images), n, replace=False)
    gt = images[idxs]
    x, y = make_xy(gt)
    pred = model.predict(x, verbose=0)

    plt.figure(figsize=(12, 2*n))
    for r in range(n):
        plt.subplot(n, 3, r*3+1); plt.imshow(x[r]); plt.axis("off"); plt.title("Input (synthetic occl.)")
        plt.subplot(n, 3, r*3+2); plt.imshow(pred[r]); plt.axis("off"); plt.title("Pred")
        plt.subplot(n, 3, r*3+3); plt.imshow(y[r]); plt.axis("off"); plt.title("GT (original)")
    path = os.path.join(OUT_DIR, f"samples_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

def main():
    imgs = load_all_images(DATA_DIR)
    if len(imgs) < 10:
        raise SystemExit("Not enough images found in DATA_DIR.")

    # --------- THIS IS THE SPLIT (90% train / 10% val) ----------
    idx = np.arange(len(imgs))
    np.random.shuffle(idx)
    imgs = imgs[idx]
    split = int(0.9 * len(imgs))
    train_imgs = imgs[:split]
    val_imgs   = imgs[split:]
    # ------------------------------------------------------------

    model = build_generator()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mae")

    for epoch in range(1, EPOCHS+1):
        Xtr, Ytr = make_xy(train_imgs)
        model.fit(Xtr, Ytr, batch_size=BATCH_SIZE, epochs=1, verbose=1)

        Xv, Yv = make_xy(val_imgs)
        val_loss = model.evaluate(Xv, Yv, batch_size=BATCH_SIZE, verbose=0)
        print("val_loss:", float(val_loss))
        save_samples(model, val_imgs, epoch, n=6)

    model.save(os.path.join(OUT_DIR, "generator.keras"))
    print("Saved model:", os.path.join(OUT_DIR, "generator.keras"))

if __name__ == "__main__":
    main()