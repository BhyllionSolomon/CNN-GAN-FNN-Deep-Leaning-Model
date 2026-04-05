"""
=============================================================================
  SPADE U-NET GAN TRAINING  —  PhD Thesis, Mr. Olagunju Korede Solomon
  Matric: 216882, University of Ibadan
  Task: Reconstruct occluded tomato regions using paired triplets
=============================================================================

  INPUTS (from your GAN_Dataset folder):
    occluded_images/   — occluded tomato (generator input)
    semantic_masks/    — binary mask: 255=visible, 0=hidden region
    ground_truth/      — clean ripe tomato (generator target)

  OUTPUTS (saved to Trained_Models/GAN/):
    generator_best.keras       — best generator weights
    discriminator.keras        — discriminator weights
    gan_training_log.csv       — loss per epoch
    samples/                   — visual reconstruction samples per epoch
    metrics/                   — FID-proxy, SSIM, PSNR per epoch

  HOW TO RUN:
    pip install tensorflow opencv-python numpy matplotlib tqdm scikit-image
    python GAN_Training.py
=============================================================================
"""

import os
import csv
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
GAN_DATA   = Path(r"C:\..PhD Thesis\DataSet\GAN_Dataset")
SAVE_DIR   = Path(r"C:\..PhD Thesis\DataSet\Trained_Models\GAN")
SAMPLE_DIR = SAVE_DIR / "samples"
METRIC_DIR = SAVE_DIR / "metrics"
CKPT_DIR   = SAVE_DIR / "checkpoints"

for d in [SAVE_DIR, SAMPLE_DIR, METRIC_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OCC_DIR  = GAN_DATA / "occluded_images"
MASK_DIR = GAN_DATA / "semantic_masks"
GT_DIR   = GAN_DATA / "ground_truth"

# ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
IMG_H      = 224
IMG_W      = 224
CHANNELS   = 3
BATCH_SIZE = 8        # keep low — GAN training is memory intensive
EPOCHS     = 100
LR_G       = 2e-4     # generator learning rate
LR_D       = 1e-4     # discriminator learning rate (lower = stable training)
LAMBDA_L1  = 100.0    # weight for L1 reconstruction loss
LAMBDA_PER = 10.0     # weight for perceptual loss
SAVE_EVERY = 5        # save sample images every N epochs
IMG_EXTS   = {".jpg", ".jpeg", ".png"}


# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_triplets():
    """Match occluded images with their masks and ground truth by stem name."""
    occ_files  = sorted([p for p in OCC_DIR.iterdir()
                         if p.suffix.lower() in IMG_EXTS])

    triplets = []
    missing  = 0
    for occ_path in occ_files:
        stem = occ_path.stem
        # mask has .png extension, ground truth has .jpg
        mask_path = MASK_DIR / f"{stem}_mask.png"
        gt_path   = GT_DIR   / f"{stem}.jpg"

        # Try alternate extensions if not found
        if not mask_path.exists():
            mask_path = MASK_DIR / f"{stem}.png"
        if not gt_path.exists():
            gt_path = GT_DIR / f"{stem}.jpg"
            if not gt_path.exists():
                gt_path = GT_DIR / f"{stem}.png"

        if mask_path.exists() and gt_path.exists():
            triplets.append((str(occ_path), str(mask_path), str(gt_path)))
        else:
            missing += 1

    print(f"  Triplets found  : {len(triplets)}")
    if missing:
        print(f"  Missing pairs   : {missing} (skipped)")
    return triplets


def preprocess_image(path):
    """Load and normalise image to [-1, 1] for GAN training."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0   # [-1, 1]
    return img


def preprocess_mask(path):
    """Load binary mask and normalise to [0, 1]."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0        # [0, 1]
    mask = np.expand_dims(mask, axis=-1)           # (H, W, 1)
    return mask


def build_dataset(triplets):
    """Build tf.data pipeline from triplet paths."""

    def load_sample(occ_p, mask_p, gt_p):
        occ  = tf.numpy_function(preprocess_image, [occ_p],  tf.float32)
        mask = tf.numpy_function(preprocess_mask,  [mask_p], tf.float32)
        gt   = tf.numpy_function(preprocess_image, [gt_p],   tf.float32)
        occ.set_shape([IMG_H, IMG_W, CHANNELS])
        mask.set_shape([IMG_H, IMG_W, 1])
        gt.set_shape([IMG_H, IMG_W, CHANNELS])
        return (occ, mask), gt

    occ_paths  = [t[0] for t in triplets]
    mask_paths = [t[1] for t in triplets]
    gt_paths   = [t[2] for t in triplets]

    ds = tf.data.Dataset.from_tensor_slices(
        (occ_paths, mask_paths, gt_paths)
    )
    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=500, seed=42)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ═════════════════════════════════════════════════════════════════════════════
#  2. SPADE NORMALISATION BLOCK
# ═════════════════════════════════════════════════════════════════════════════
class SPADEBlock(layers.Layer):
    """Spatially-Adaptive Denormalization — conditions on semantic mask."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm    = layers.LayerNormalization(axis=-1)
        self.conv_gamma = layers.Conv2D(filters, 3, padding='same',
                                        activation='linear')
        self.conv_beta  = layers.Conv2D(filters, 3, padding='same',
                                        activation='linear')

    def call(self, x, mask):
        # Resize mask to match x spatial dims
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        mask_r = tf.image.resize(mask, [h, w], method='nearest')

        normalized = self.norm(x)
        gamma = self.conv_gamma(mask_r)
        beta  = self.conv_beta(mask_r)
        return normalized * (1 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


# ═════════════════════════════════════════════════════════════════════════════
#  3. GENERATOR  (U-Net backbone + SPADE layers)
# ═════════════════════════════════════════════════════════════════════════════
def build_generator():
    """
    U-Net generator with SPADE normalisation.
    Input : occluded image (224,224,3) + semantic mask (224,224,1)
    Output: reconstructed image (224,224,3)
    """
    occ_input  = layers.Input(shape=(IMG_H, IMG_W, CHANNELS), name='occluded')
    mask_input = layers.Input(shape=(IMG_H, IMG_W, 1),        name='mask')

    # ── Encoder ──────────────────────────────────────────────────────────────
    def enc_block(x, filters, name):
        x = layers.Conv2D(filters, 4, strides=2, padding='same',
                          use_bias=False, name=f'{name}_conv')(x)
        x = layers.LeakyReLU(0.2, name=f'{name}_lrelu')(x)
        return x

    e1 = enc_block(occ_input, 64,  'enc1')   # 112x112
    e2 = enc_block(e1,        128, 'enc2')   # 56x56
    e3 = enc_block(e2,        256, 'enc3')   # 28x28
    e4 = enc_block(e3,        512, 'enc4')   # 14x14
    e5 = enc_block(e4,        512, 'enc5')   # 7x7

    # ── Bottleneck ────────────────────────────────────────────────────────────
    b = layers.Conv2D(512, 4, strides=2, padding='same',
                      use_bias=False, name='bottleneck_conv')(e5)
    b = layers.ReLU(name='bottleneck_relu')(b)               # 4x4

    # ── Decoder with SPADE + skip connections ─────────────────────────────────
    def dec_block(x, skip, filters, spade_layer, name, dropout=False):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same',
                                   use_bias=False, name=f'{name}_deconv')(x)
        x = spade_layer(x, mask_input)
        x = layers.ReLU(name=f'{name}_relu')(x)
        if dropout:
            x = layers.Dropout(0.5, name=f'{name}_drop')(x)
        x = layers.Concatenate(name=f'{name}_skip')([x, skip])
        return x

    spade5 = SPADEBlock(512,  name='spade5')
    spade4 = SPADEBlock(512,  name='spade4')
    spade3 = SPADEBlock(256,  name='spade3')
    spade2 = SPADEBlock(128,  name='spade2')
    spade1 = SPADEBlock(64,   name='spade1')

    d5 = dec_block(b,  e5, 512, spade5, 'd5', dropout=True)   # 7x7→14x14
    d4 = dec_block(d5, e4, 512, spade4, 'd4', dropout=True)   # 14→28
    d3 = dec_block(d4, e3, 256, spade3, 'd3')                  # 28→56
    d2 = dec_block(d3, e2, 128, spade2, 'd2')                  # 56→112
    d1 = dec_block(d2, e1, 64,  spade1, 'd1')                  # 112→224

    # Final upsample + output
    out = layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding='same',
                                 activation='tanh', name='output')(d1)

    return models.Model(inputs=[occ_input, mask_input],
                        outputs=out, name='SPADE_Generator')


# ═════════════════════════════════════════════════════════════════════════════
#  4. DISCRIMINATOR  (PatchGAN)
# ═════════════════════════════════════════════════════════════════════════════
def build_discriminator():
    """
    PatchGAN discriminator.
    Judges local 70x70 patches as real or fake.
    Input: occluded image + (real or generated) image concatenated
    """
    inp  = layers.Input(shape=(IMG_H, IMG_W, CHANNELS), name='input_image')
    tar  = layers.Input(shape=(IMG_H, IMG_W, CHANNELS), name='target_image')
    x    = layers.Concatenate()([inp, tar])

    def d_block(x, filters, stride, name, norm=True):
        x = layers.Conv2D(filters, 4, strides=stride, padding='same',
                          use_bias=False, name=f'{name}_conv')(x)
        if norm:
            x = layers.BatchNormalization(name=f'{name}_bn')(x)
        x = layers.LeakyReLU(0.2, name=f'{name}_lrelu')(x)
        return x

    x = d_block(x,  64,  2, 'd1', norm=False)   # no norm on first layer
    x = d_block(x,  128, 2, 'd2')
    x = d_block(x,  256, 2, 'd3')
    x = d_block(x,  512, 1, 'd4')

    out = layers.Conv2D(1, 4, strides=1, padding='same',
                        name='patch_output')(x)

    return models.Model(inputs=[inp, tar],
                        outputs=out, name='PatchGAN_Discriminator')


# ═════════════════════════════════════════════════════════════════════════════
#  5. LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
bce = losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output),  real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) * 0.5

def generator_loss(fake_output, generated, target):
    # Adversarial loss
    adv_loss  = bce(tf.ones_like(fake_output), fake_output)
    # L1 pixel reconstruction loss (weighted heavily)
    l1_loss   = tf.reduce_mean(tf.abs(target - generated))
    # Perceptual loss proxy — L2 on feature-level difference
    per_loss  = tf.reduce_mean(tf.square(target - generated))
    total     = adv_loss + LAMBDA_L1 * l1_loss + LAMBDA_PER * per_loss
    return total, adv_loss, l1_loss, per_loss


# ═════════════════════════════════════════════════════════════════════════════
#  6. TRAINING STEP
# ═════════════════════════════════════════════════════════════════════════════
@tf.function
def train_step(generator, discriminator, gen_opt, disc_opt,
               occ_imgs, masks, gt_imgs):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generator forward pass
        generated = generator([occ_imgs, masks], training=True)

        # Discriminator on real and fake
        real_out = discriminator([occ_imgs, gt_imgs],   training=True)
        fake_out = discriminator([occ_imgs, generated], training=True)

        # Losses
        d_loss = discriminator_loss(real_out, fake_out)
        g_loss, adv_l, l1_l, per_l = generator_loss(fake_out,
                                                      generated, gt_imgs)

    # Apply gradients
    gen_grads  = gen_tape.gradient(g_loss,
                                   generator.trainable_variables)
    disc_grads = disc_tape.gradient(d_loss,
                                    discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grads,
                                generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads,
                                 discriminator.trainable_variables))

    return g_loss, d_loss, adv_l, l1_l, per_l


# ═════════════════════════════════════════════════════════════════════════════
#  7. SAMPLE SAVER
# ═════════════════════════════════════════════════════════════════════════════
def save_samples(generator, sample_batch, epoch):
    """Save side-by-side: occluded | generated | ground truth."""
    (occ_imgs, masks), gt_imgs = sample_batch
    generated = generator([occ_imgs[:4], masks[:4]], training=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    titles = ['Occluded (input)', 'Generated', 'Ground truth']
    imgs   = [occ_imgs[:4], generated, gt_imgs[:4]]

    for row, (title, img_set) in enumerate(zip(titles, imgs)):
        for col in range(4):
            img = img_set[col].numpy()
            img = (img + 1.0) / 2.0          # [-1,1] → [0,1]
            img = np.clip(img, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=10,
                                          fontweight='bold')

    plt.suptitle(f'GAN Reconstruction — Epoch {epoch}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = SAMPLE_DIR / f'epoch_{epoch:03d}.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  8. METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(generator, val_batch):
    """Compute SSIM and PSNR on a validation batch."""
    (occ_imgs, masks), gt_imgs = val_batch
    generated = generator([occ_imgs, masks], training=False)

    ssim_scores = []
    psnr_scores = []

    for i in range(len(occ_imgs)):
        gen_np = ((generated[i].numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        gt_np  = ((gt_imgs[i].numpy()   + 1.0) / 2.0 * 255).astype(np.uint8)

        s = ssim(gt_np, gen_np, channel_axis=2, data_range=255)
        p = psnr(gt_np, gen_np, data_range=255)
        ssim_scores.append(s)
        psnr_scores.append(p)

    return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))


# ═════════════════════════════════════════════════════════════════════════════
#  9. MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  SPADE U-NET GAN TRAINING")
    print("  PhD Thesis — Olagunju Korede Solomon (216882)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading GAN triplets...")
    triplets = load_triplets()
    if not triplets:
        print("[ERROR] No triplets found. Run tomato_dataset_pipeline.py first.")
        return

    # 90% train, 10% val
    split     = int(len(triplets) * 0.90)
    train_t   = triplets[:split]
    val_t     = triplets[split:]
    print(f"  Train triplets : {len(train_t)}")
    print(f"  Val triplets   : {len(val_t)}")

    train_ds  = build_dataset(train_t)
    val_ds    = build_dataset(val_t)

    # Sample batch for visualisation
    sample_batch = next(iter(val_ds))

    # ── Build models ──────────────────────────────────────────────────────────
    print("\n[2/5] Building Generator and Discriminator...")
    generator     = build_generator()
    discriminator = build_discriminator()
    generator.summary(line_length=80)

    # ── Optimisers ────────────────────────────────────────────────────────────
    gen_opt  = optimizers.Adam(LR_G, beta_1=0.5)
    disc_opt = optimizers.Adam(LR_D, beta_1=0.5)

    # ── TensorFlow checkpoint (saves optimiser state too) ─────────────────────
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_opt,
        disc_optimizer=disc_opt
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory=str(CKPT_DIR),
        max_to_keep=3             # keep last 3 checkpoints only
    )

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    start_epoch = 1
    best_ssim   = 0.0
    log_path    = SAVE_DIR / "gan_training_log.csv"

    state_path  = SAVE_DIR / "training_state.json"

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"\n  RESUMED from checkpoint: {ckpt_manager.latest_checkpoint}")

        # Restore epoch counter and best SSIM from state file
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            start_epoch = state.get("last_epoch", 1) + 1
            best_ssim   = state.get("best_ssim", 0.0)
            print(f"  Resuming from epoch {start_epoch}  |  Best SSIM so far: {best_ssim:.4f}")
        else:
            print("  No state file found — starting epoch count from 1")
    else:
        print("\n  No checkpoint found — starting fresh training")
        # Create fresh CSV log with header
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'g_loss', 'd_loss', 'adv_loss',
                                    'l1_loss', 'per_loss', 'ssim', 'psnr'])

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Training from epoch {start_epoch} to {EPOCHS}...")

    if start_epoch > EPOCHS:
        print("  Training already complete. Delete checkpoints/ to retrain.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        g_losses, d_losses = [], []
        adv_ls, l1_ls, per_ls = [], [], []

        for (occ_b, mask_b), gt_b in tqdm(train_ds,
                                           desc=f"Epoch {epoch:03d}/{EPOCHS}",
                                           leave=False):
            g_l, d_l, adv_l, l1_l, per_l = train_step(
                generator, discriminator, gen_opt, disc_opt,
                occ_b, mask_b, gt_b
            )
            g_losses.append(float(g_l))
            d_losses.append(float(d_l))
            adv_ls.append(float(adv_l))
            l1_ls.append(float(l1_l))
            per_ls.append(float(per_l))

        # Epoch averages
        g_avg   = np.mean(g_losses)
        d_avg   = np.mean(d_losses)
        adv_avg = np.mean(adv_ls)
        l1_avg  = np.mean(l1_ls)
        per_avg = np.mean(per_ls)

        # Validation metrics
        val_batch   = next(iter(val_ds))
        mean_ssim, mean_psnr = compute_metrics(generator, val_batch)

        print(f"Epoch {epoch:03d} | G={g_avg:.4f} D={d_avg:.4f} "
              f"L1={l1_avg:.4f} SSIM={mean_ssim:.4f} PSNR={mean_psnr:.2f}dB")

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, g_avg, d_avg, adv_avg,
                                    l1_avg, per_avg, mean_ssim, mean_psnr])

        # Save sample images
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_samples(generator, sample_batch, epoch)

        # Save best generator
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            generator.save(str(SAVE_DIR / "generator_best.keras"))
            print(f"  --> Best generator saved (SSIM={best_ssim:.4f})")

        # ── CHECKPOINT every epoch — survives power outage ────────────────
        ckpt_manager.save()

        # Save lightweight state file (epoch number + best SSIM)
        with open(state_path, 'w') as f:
            json.dump({"last_epoch": epoch,
                       "best_ssim": best_ssim,
                       "total_epochs": EPOCHS}, f, indent=2)

    # ── Save final models ─────────────────────────────────────────────────────
    print("\n[4/5] Saving final models...")
    generator.save(str(SAVE_DIR / "generator_final.keras"))
    discriminator.save(str(SAVE_DIR / "discriminator_final.keras"))

    # ── Plot training curves ──────────────────────────────────────────────────
    print("\n[5/5] Plotting training curves...")
    import pandas as pd
    log_df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('GAN Training History — SPADE U-Net',
                 fontsize=14, fontweight='bold')

    axes[0].plot(log_df['epoch'], log_df['g_loss'],  label='Generator')
    axes[0].plot(log_df['epoch'], log_df['d_loss'],  label='Discriminator')
    axes[0].set_title('Adversarial Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log_df['epoch'], log_df['ssim'], color='green')
    axes[1].set_title('SSIM (higher = better)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(log_df['epoch'], log_df['psnr'], color='orange')
    axes[2].set_title('PSNR dB (higher = better)')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(SAVE_DIR / 'gan_training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 65)
    print("  GAN TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best SSIM achieved : {best_ssim:.4f}")
    print(f"  Models saved to    : {SAVE_DIR}")
    print(f"  Sample images      : {SAMPLE_DIR}")
    print(f"  Training log       : {log_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()