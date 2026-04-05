#!/usr/bin/env python3
"""
TOMATO RECONSTRUCTION SYSTEM — SPADE + Conditional GAN
Power-Safe Version with Full Resume & Enhanced Plotting
Patched by Claude — paths and data loader updated for confirmed folder structure.

Confirmed paths:
  Occluded images : C:\..PhD Thesis\DataSet\GAN_Dataset\occluded_images\
  Semantic masks  : C:\..PhD Thesis\DataSet\GAN_Dataset\semantic_masks\
  Ground truth    : C:\..PhD Thesis\DataSet\GAN_Dataset\ground_truth\
  Output          : C:\..PhD Thesis\DataSet\Trained_Models\GAN\
"""

import os
import sys
import json
import glob
import time
import shutil
import signal
import ctypes
import tempfile
import threading
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

tf.random.set_seed(42)
np.random.seed(42)


# ===========================================================================
# POWER MANAGEMENT
# ===========================================================================
class PowerManager:
    ES_CONTINUOUS        = 0x80000000
    ES_SYSTEM_REQUIRED   = 0x00000001
    ES_DISPLAY_REQUIRED  = 0x00000002
    _enabled = False
    _lock    = threading.Lock()

    @classmethod
    def prevent_sleep(cls):
        if sys.platform != "win32":
            print("Power management: non-Windows OS — skipping.")
            return
        try:
            result = ctypes.windll.kernel32.SetThreadExecutionState(
                cls.ES_CONTINUOUS | cls.ES_SYSTEM_REQUIRED | cls.ES_DISPLAY_REQUIRED)
            if result != 0:
                cls._enabled = True
                print("Power management: sleep/hibernate DISABLED.")
            else:
                print("Power management: SetThreadExecutionState failed.")
        except Exception as e:
            print(f"Power management unavailable: {e}")

    @classmethod
    def allow_sleep(cls):
        if sys.platform != "win32" or not cls._enabled:
            return
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(cls.ES_CONTINUOUS)
            cls._enabled = False
            print("Power management: normal sleep restored.")
        except Exception:
            pass

    @classmethod
    def install_signal_handler(cls, trainer):
        def _handler(sig, frame):
            print("\n\nInterrupt received — saving emergency checkpoint...")
            trainer.emergency_save()
            cls.allow_sleep()
            print("Emergency checkpoint saved. Safe to power off.")
            sys.exit(0)
        signal.signal(signal.SIGINT,  _handler)
        signal.signal(signal.SIGTERM, _handler)
        print("Graceful shutdown handler installed (Ctrl+C = safe save).")


# ===========================================================================
# ATOMIC FILE WRITER
# ===========================================================================
def atomic_json_save(data: dict, path: str):
    dir_  = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ===========================================================================
# CONFIGURATION  — confirmed paths for Mr. Solomon's thesis
# ===========================================================================
class Config:
    # ---- Confirmed paths (do not change) ------------------------------------
    OCC_ROOT  = r"C:\..PhD Thesis\DataSet\GAN_Dataset\occluded_images"
    MASK_ROOT = r"C:\..PhD Thesis\DataSet\GAN_Dataset\semantic_masks"
    GT_ROOT   = r"C:\..PhD Thesis\DataSet\GAN_Dataset\ground_truth"
    OUT_DIR   = r"C:\..PhD Thesis\DataSet\Trained_Models\GAN"

    # ---- Image / model -------------------------------------------------------
    IMG_SIZE   = (224, 224)
    BATCH_SIZE = 8
    EPOCHS     = 100

    # ---- Learning rate -------------------------------------------------------
    INIT_LR        = 2e-4
    LR_DECAY_RATE  = 0.5
    LR_DECAY_STEPS = 500
    GRAD_CLIP_NORM = 1.0

    # ---- GAN loss weights ----------------------------------------------------
    L1_LOSS_WEIGHT        = 100.0
    MASKED_L1_WEIGHT      = 200.0
    PERCEPTUAL_LW         = 1.0
    SSIM_LOSS_WEIGHT      = 0.3
    PATCHGAN_LOSS_WEIGHT  = 0.05
    GLOBALGAN_LOSS_WEIGHT = 0.05

    # ---- Training schedule ---------------------------------------------------
    WARMUP_EPOCHS        = 3
    NUM_RESBLOCKS        = 4
    CHECKPOINT_SAVE_FREQ = 2

    # ---- Splits (applied to triplet list) ------------------------------------
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15
    RANDOM_SEED = 42

    ARCH_VERSION = "v2_confirmed_paths"

    @classmethod
    def setup_dirs(cls):
        cls.RESULTS_DIR                = os.path.join(cls.OUT_DIR, "Tomato_Reconstruction")
        cls.CHECKPOINT_DIR             = os.path.join(cls.RESULTS_DIR, "Checkpoints")
        cls.PLOTS_DIR                  = os.path.join(cls.RESULTS_DIR, "Plots")
        cls.SAMPLE_RECONSTRUCTIONS_DIR = os.path.join(cls.RESULTS_DIR, "Samples")
        cls.METRICS_DIR                = os.path.join(cls.RESULTS_DIR, "Metrics")
        cls.TEST_RESULTS_DIR           = os.path.join(cls.RESULTS_DIR, "Test_Results")
        cls.HEARTBEAT_PATH             = os.path.join(cls.RESULTS_DIR, "heartbeat.json")
        cls.HISTORY_PATH               = os.path.join(cls.RESULTS_DIR, "training_history.json")
        cls.STATE_PATH                 = os.path.join(cls.RESULTS_DIR, "training_state.json")
        for d in [cls.RESULTS_DIR, cls.CHECKPOINT_DIR, cls.PLOTS_DIR,
                  cls.SAMPLE_RECONSTRUCTIONS_DIR, cls.METRICS_DIR, cls.TEST_RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)


# ===========================================================================
# DATA LOADER  — uses pre-generated triplets from GAN_Dataset
# ===========================================================================
class TomatoDataLoader:
    """
    Loads pre-generated triplets created by tomato_dataset_pipeline.py:
      occluded_images/  — occluded tomato  (generator input)
      semantic_masks/   — binary mask      (visible=255, hidden=0)
      ground_truth/     — clean ripe image (generator target)

    No runtime occlusion generation — uses the prepared dataset directly.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self):
        self.image_size = Config.IMG_SIZE

    # ------------------------------------------------------------------
    def find_all_triplets(self):
        """Match occluded + mask + ground_truth by stem filename."""
        occ_files = []
        for ext in self.IMG_EXTS:
            occ_files += glob.glob(os.path.join(Config.OCC_ROOT,  f"*{ext}"))
            occ_files += glob.glob(os.path.join(Config.OCC_ROOT,  f"*{ext.upper()}"))
        occ_files.sort()

        if not occ_files:
            print(f"No occluded images found in: {Config.OCC_ROOT}")
            return []

        print(f"Found {len(occ_files)} occluded images")

        triplets = []
        missing  = 0
        for occ_path in tqdm(occ_files, desc="Matching triplets"):
            stem = os.path.splitext(os.path.basename(occ_path))[0]

            # Mask: pipeline saves as {stem}_mask.png, fallback to {stem}.png
            mask_path = os.path.join(Config.MASK_ROOT, f"{stem}_mask.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(Config.MASK_ROOT, f"{stem}.png")

            # Ground truth: same stem, .jpg first then .png
            gt_path = os.path.join(Config.GT_ROOT, f"{stem}.jpg")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(Config.GT_ROOT, f"{stem}.png")

            if os.path.exists(mask_path) and os.path.exists(gt_path):
                triplets.append((occ_path, mask_path, gt_path))
            else:
                missing += 1

        print(f"Valid triplets  : {len(triplets)}")
        if missing:
            print(f"Missing pairs   : {missing} (skipped)")
        return triplets

    # ------------------------------------------------------------------
    def split_triplets(self, triplets):
        """Split triplets 70/15/15 with fixed seed."""
        import random
        rng = random.Random(Config.RANDOM_SEED)
        triplets = list(triplets)
        rng.shuffle(triplets)
        n       = len(triplets)
        n_train = int(n * Config.TRAIN_RATIO)
        n_val   = int(n * Config.VAL_RATIO)
        train   = triplets[:n_train]
        val     = triplets[n_train:n_train + n_val]
        test    = triplets[n_train + n_val:]
        print(f"Split  train={len(train)}  val={len(val)}  test={len(test)}")
        return train, val, test

    # ------------------------------------------------------------------
    def load_triplet(self, occ_path, mask_path, gt_path):
        """Load and normalise one triplet. Returns (occ, mask, gt) as float32."""
        H, W = self.image_size

        # Occluded image — normalise to [0, 1]
        occ = cv2.imread(occ_path)
        if occ is None:
            raise ValueError(f"Cannot read: {occ_path}")
        occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)
        occ = cv2.resize(occ, (W, H)).astype(np.float32) / 255.0

        # Semantic mask — binary [0, 1], shape (H, W, 1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read: {mask_path}")
        mask = cv2.resize(mask, (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        mask = mask[..., np.newaxis]

        # Ground truth — normalise to [0, 1]
        gt = cv2.imread(gt_path)
        if gt is None:
            raise ValueError(f"Cannot read: {gt_path}")
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (W, H)).astype(np.float32) / 255.0

        return occ, mask, gt

    # ------------------------------------------------------------------
    def _load_py(self, occ_p, mask_p, gt_p):
        occ, mask, gt = self.load_triplet(
            occ_p.numpy().decode(),
            mask_p.numpy().decode(),
            gt_p.numpy().decode()
        )
        # occ_mask = inverse of visible mask (1 = hidden region)
        occ_mask = 1.0 - mask
        return occ, mask, occ_mask[..., 0], gt

    # ------------------------------------------------------------------
    def create_tf_dataset(self, triplets, batch_size, is_training=True):
        """
        Returns batches of: (occluded, mask, occ_mask_3d), ground_truth
        All values in [0, 1].
        """
        H, W = self.image_size

        occ_paths  = [t[0] for t in triplets]
        mask_paths = [t[1] for t in triplets]
        gt_paths   = [t[2] for t in triplets]

        def load_sample(occ_p, mask_p, gt_p):
            occ, mask, occ_mask_2d, gt = tf.py_function(
                self._load_py,
                [occ_p, mask_p, gt_p],
                [tf.float32, tf.float32, tf.float32, tf.float32]
            )
            occ.set_shape([H, W, 3])
            mask.set_shape([H, W, 1])
            occ_mask_2d.set_shape([H, W])
            gt.set_shape([H, W, 3])
            occ_mask_3d = tf.expand_dims(occ_mask_2d, axis=-1)
            occ_mask_3d.set_shape([H, W, 1])
            return (occ, mask, occ_mask_3d), gt

        ds = tf.data.Dataset.from_tensor_slices((occ_paths, mask_paths, gt_paths))
        ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            ds = ds.shuffle(buffer_size=min(500, len(triplets)), seed=42).repeat()
        else:
            ds = ds.repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ===========================================================================
# SPADE
# ===========================================================================
class SPADE(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.shared_conv = layers.Conv2D(128, kernel_size, padding="same", activation="relu")
        self.gamma_conv  = layers.Conv2D(filters, kernel_size, padding="same")
        self.beta_conv   = layers.Conv2D(filters, kernel_size, padding="same")

    def call(self, x, seg):
        mean  = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std   = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        x_n   = (x - mean) / (std + 1e-5)
        seg_r = tf.image.resize(seg, [tf.shape(x)[1], tf.shape(x)[2]], method="nearest")
        feat  = self.shared_conv(seg_r)
        return x_n * (1.0 + self.gamma_conv(feat)) + self.beta_conv(feat)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters})
        return cfg


# ===========================================================================
# GENERATOR
# ===========================================================================
def build_generator(img_size=None):
    S = img_size or Config.IMG_SIZE[0]
    inp_img  = layers.Input((S, S, 3), name="occluded_image")
    inp_mask = layers.Input((S, S, 1), name="semantic_mask")

    def spade_resblock(x, filters, mask):
        skip = x
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = SPADE(filters)(x, mask); x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = SPADE(filters)(x, mask)
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, 1, padding="same")(skip)
            skip = SPADE(filters)(skip, mask)
        return layers.ReLU()(layers.Add()([x, skip]))

    x = layers.Conv2D(64,  7, padding="same")(inp_img)
    x = SPADE(64)(x,  inp_mask); x = layers.ReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = SPADE(128)(x, inp_mask); x = layers.ReLU()(x)
    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = SPADE(256)(x, inp_mask); x = layers.ReLU()(x)

    for _ in range(Config.NUM_RESBLOCKS):
        x = spade_resblock(x, 256, inp_mask)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
    x = SPADE(128)(x, inp_mask); x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64,  4, strides=2, padding="same")(x)
    x = SPADE(64)(x,  inp_mask); x = layers.ReLU()(x)
    x = layers.Conv2D(32, 7, padding="same")(x)
    x = SPADE(32)(x,  inp_mask); x = layers.ReLU()(x)
    out = layers.Conv2D(3, 7, padding="same", activation="sigmoid",
                        name="reconstructed_tomato")(x)
    return Model([inp_img, inp_mask], out, name="Generator")


# ===========================================================================
# DISCRIMINATORS
# ===========================================================================
def build_patchgan(img_size=None):
    S = img_size or Config.IMG_SIZE[0]
    inp = layers.Input((S, S, 6), name="concat_input")
    x = layers.Conv2D(64,  4, strides=2, padding="same")(inp); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, 4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(1, 4, strides=1, padding="same", activation="linear")(x)
    return Model(inp, out, name="PatchGAN")


def build_global_disc(img_size=None):
    S = img_size or Config.IMG_SIZE[0]
    inp = layers.Input((S, S, 6), name="concat_input")
    x = layers.Conv2D(64, 4, strides=2, padding="same")(inp); x = layers.LeakyReLU(0.2)(x)
    for filters in (128, 256, 512, 512):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="linear")(x)
    return Model(inp, out, name="GlobalDisc")


# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================
class Losses:
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()

    @staticmethod
    def lsgan(pred, real: bool):
        target = tf.ones_like(pred) if real else tf.zeros_like(pred)
        return Losses.mse(target, pred)

    @staticmethod
    def masked_l1(y_true, y_pred, mask):
        diff   = tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))
        mask_f = tf.cast(mask, tf.float32)
        return tf.reduce_sum(diff * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)

    @staticmethod
    def perceptual(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        return Losses.mse(y_true, y_pred) + 0.5 * Losses.mae(y_true, y_pred)

    @staticmethod
    def ssim_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    @staticmethod
    def discriminator(real_out, fake_out):
        return 0.5 * (Losses.lsgan(real_out, True) + Losses.lsgan(fake_out, False))

    @staticmethod
    def generator_total(y_true, y_pred, patch_fake, global_fake, occ_mask, use_adversarial):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        l1         = Losses.mae(y_true, y_pred)                * Config.L1_LOSS_WEIGHT
        m_l1       = Losses.masked_l1(y_true, y_pred, occ_mask)* Config.MASKED_L1_WEIGHT
        perceptual = Losses.perceptual(y_true, y_pred)         * Config.PERCEPTUAL_LW
        ssim_l     = Losses.ssim_loss(y_true, y_pred)          * Config.SSIM_LOSS_WEIGHT
        adv = (
            Losses.lsgan(patch_fake,  True) * Config.PATCHGAN_LOSS_WEIGHT +
            Losses.lsgan(global_fake, True) * Config.GLOBALGAN_LOSS_WEIGHT
        ) if use_adversarial else 0.0
        total = l1 + m_l1 + perceptual + ssim_l + adv
        return total, {"l1": l1, "masked_l1": m_l1,
                       "perceptual": perceptual, "ssim": ssim_l, "adv": adv}


# ===========================================================================
# VISUALIZER
# ===========================================================================
class TrainingVisualizer:

    @staticmethod
    def plot_training_curves(history, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        eps = list(range(1, len(history["g_loss"]) + 1))
        if not eps:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(
            f"GAN Training Curves — Epoch {eps[-1]}/{Config.EPOCHS}\n"
            f"Best Val SSIM: {max(history['val_ssim']):.4f}  |  "
            f"Best Val PSNR: {max(history['val_psnr']):.2f} dB",
            fontsize=13, fontweight="bold"
        )
        def _plot(ax, k_t, k_v=None, title="", ylabel="", ct="royalblue", cv="tomato"):
            ax.plot(eps, history[k_t], color=ct, lw=2, label="Train", marker="o", markersize=3)
            if k_v and k_v in history and history[k_v]:
                ax.plot(eps, history[k_v], color=cv, lw=2, label="Val", marker="s", markersize=3)
                ax.annotate(f"{history[k_v][-1]:.4f}", xy=(eps[-1], history[k_v][-1]),
                            xytext=(5,5), textcoords="offset points", fontsize=8, color=cv)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        _plot(axes[0,0], "g_loss",   "val_g_loss",  "Generator Loss",  "Loss")
        _plot(axes[0,1], "dp_loss",  "val_dp_loss", "PatchGAN D Loss", "Loss", "seagreen", "darkorange")
        _plot(axes[0,2], "dg_loss",  "val_dg_loss", "Global D Loss",   "Loss", "purple",   "crimson")
        _plot(axes[1,0], "val_ssim", title="Validation SSIM", ylabel="SSIM",  color_t="darkorange")
        _plot(axes[1,1], "val_psnr", title="Validation PSNR", ylabel="dB",    color_t="seagreen")

        g  = np.array(history["g_loss"])
        dp = np.array(history["dp_loss"]) + 1e-8
        axes[1,2].plot(eps, g/dp, color="darkorange", lw=2, marker="o", markersize=3)
        axes[1,2].axhline(1.0, color="red", ls="--", alpha=0.6, label="Balance (ratio=1)")
        axes[1,2].fill_between(eps, 0.5, 2.0, alpha=0.08, color="green", label="Healthy range")
        axes[1,2].set_title("G / D Loss Ratio", fontsize=11, fontweight="bold")
        axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Ratio")
        axes[1,2].legend(fontsize=9); axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(save_dir, f"curves_epoch{eps[-1]:03d}_{timestamp}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        shutil.copy(out, os.path.join(save_dir, "training_curves_LATEST.png"))
        print(f"  Training curves saved")

    @staticmethod
    def save_reconstruction_grid(originals, occludeds, reconstructions,
                                  sem_masks, occ_masks, ssim_vals, psnr_vals,
                                  save_path, epoch=None, split="val"):
        n = len(originals)
        fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        titles = ["Original (GT)", "Occluded (Input)", "Reconstructed (Output)",
                  "Semantic Mask", "Occlusion Mask"]
        for j, t in enumerate(titles):
            axes[0, j].set_title(t, fontsize=10, fontweight="bold", pad=6)
        for i in range(n):
            for j, (im, lbl) in enumerate([
                (originals[i],       "GT"),
                (occludeds[i],       "Input"),
                (reconstructions[i], f"SSIM={ssim_vals[i]:.3f}\nPSNR={psnr_vals[i]:.1f}dB"),
                (sem_masks[i].squeeze() if sem_masks[i].ndim==3 else sem_masks[i], "Semantic"),
                (occ_masks[i],       "Occlusion"),
            ]):
                cmap = "gray" if (isinstance(im, np.ndarray) and im.ndim==2) else None
                axes[i,j].imshow(np.clip(im, 0, 1), cmap=cmap)
                axes[i,j].set_xlabel(lbl, fontsize=8)
                axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            axes[i,0].set_ylabel(f"Sample {i+1}", fontsize=9, fontweight="bold")
        epoch_str = f"Epoch {epoch}" if epoch else ""
        fig.suptitle(
            f"{split.upper()} Reconstructions  {epoch_str}\n"
            f"Mean SSIM: {np.mean(ssim_vals):.4f}  |  Mean PSNR: {np.mean(psnr_vals):.2f} dB",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# ===========================================================================
# GAN TRAINER — Power-Safe
# ===========================================================================
class TomatoGANTrainer:

    def __init__(self):
        S = Config.IMG_SIZE[0]
        self.generator   = build_generator(S)
        self.patchgan    = build_patchgan(S)
        self.global_disc = build_global_disc(S)

        self.g_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Config.INIT_LR,
            decay_steps=Config.LR_DECAY_STEPS,
            decay_rate=Config.LR_DECAY_RATE,
            staircase=True,
        )
        clip = Config.GRAD_CLIP_NORM
        self.g_opt  = tf.keras.optimizers.Adam(self.g_lr_schedule, beta_1=0.5, clipnorm=clip)
        self.dp_opt = tf.keras.optimizers.Adam(Config.INIT_LR,     beta_1=0.5, clipnorm=clip)
        self.dg_opt = tf.keras.optimizers.Adam(Config.INIT_LR,     beta_1=0.5, clipnorm=clip)

        self.arch_version_var = tf.Variable("", trainable=False, dtype=tf.string,
                                             name="arch_version")
        self.ckpt = tf.train.Checkpoint(
            generator=self.generator,
            patchgan=self.patchgan,
            global_disc=self.global_disc,
            g_opt=self.g_opt,
            dp_opt=self.dp_opt,
            dg_opt=self.dg_opt,
            epoch=tf.Variable(0,   dtype=tf.int64,   trainable=False),
            step=tf.Variable(0,    dtype=tf.int64,   trainable=False),
            best_ssim=tf.Variable(0.0, dtype=tf.float32, trainable=False),
            arch_version=self.arch_version_var,
        )
        self.ckpt_mgr = tf.train.CheckpointManager(
            self.ckpt, Config.CHECKPOINT_DIR, max_to_keep=5)

        self.history = {k: [] for k in (
            "g_loss", "dp_loss", "dg_loss",
            "val_g_loss", "val_dp_loss", "val_dg_loss",
            "val_ssim", "val_psnr"
        )}
        self.start_epoch = 0
        self.global_step = 0
        self.best_ssim   = 0.0
        self._build_optimizers()

    def _build_optimizers(self):
        self.g_opt.build(self.generator.trainable_variables)
        self.dp_opt.build(self.patchgan.trainable_variables)
        self.dg_opt.build(self.global_disc.trainable_variables)
        print("Optimizers built")

    def emergency_save(self):
        print("Emergency save in progress...")
        self.ckpt.step.assign(self.global_step)
        self.ckpt_mgr.save()
        atomic_json_save(self.history, Config.HISTORY_PATH)
        self._write_heartbeat(status="emergency_save")
        print(f"Emergency checkpoint saved at step {self.global_step}")

    def _write_heartbeat(self, epoch=None, step=None, status="running"):
        data = {
            "status":    status,
            "epoch":     epoch if epoch is not None else self.start_epoch,
            "step":      step  if step  is not None else self.global_step,
            "best_ssim": float(self.best_ssim),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            atomic_json_save(data, Config.HEARTBEAT_PATH)
        except Exception:
            pass

    def _check_previous_crash(self):
        if not os.path.isfile(Config.HEARTBEAT_PATH):
            return
        try:
            with open(Config.HEARTBEAT_PATH) as f:
                hb = json.load(f)
            if hb.get("status") not in ("completed", "emergency_save"):
                print(f"\nWARNING: Previous run did NOT finish cleanly.")
                print(f"  Last heartbeat: epoch={hb.get('epoch')}  step={hb.get('step')}")
                print(f"  Resuming from last checkpoint automatically.\n")
        except Exception:
            pass

    def restore_or_initialize(self):
        self._check_previous_crash()
        latest = self.ckpt_mgr.latest_checkpoint
        if latest:
            reader = tf.train.load_checkpoint(latest)
            saved_ver = ""
            try:
                saved_ver = reader.get_tensor(
                    "arch_version/.ATTRIBUTES/VARIABLE_VALUE").decode()
            except Exception:
                pass
            if saved_ver != Config.ARCH_VERSION:
                print(f"Architecture changed — removing stale checkpoints.")
                shutil.rmtree(Config.CHECKPOINT_DIR)
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                self.ckpt_mgr = tf.train.CheckpointManager(
                    self.ckpt, Config.CHECKPOINT_DIR, max_to_keep=5)
            else:
                self.ckpt.restore(latest)
                self.start_epoch = int(self.ckpt.epoch.numpy())
                self.global_step = int(self.ckpt.step.numpy())
                self.best_ssim   = float(self.ckpt.best_ssim.numpy())
                print(f"Resumed from epoch {self.start_epoch}  "
                      f"(best SSIM={self.best_ssim:.4f})")
                if os.path.isfile(Config.HISTORY_PATH):
                    with open(Config.HISTORY_PATH) as f:
                        self.history = json.load(f)
                return
        self.arch_version_var.assign(Config.ARCH_VERSION)
        print(f"Starting fresh — arch version: {Config.ARCH_VERSION}")

    @tf.function
    def _train_step_full(self, occluded, mask, target, occ_mask):
        return self.__train_impl(occluded, mask, target, occ_mask, use_adversarial=True)

    @tf.function
    def _train_step_warmup(self, occluded, mask, target, occ_mask):
        return self.__train_impl(occluded, mask, target, occ_mask, use_adversarial=False)

    def __train_impl(self, occluded, mask, target, occ_mask, use_adversarial):
        occluded = tf.cast(occluded, tf.float32); mask     = tf.cast(mask,     tf.float32)
        target   = tf.cast(target,   tf.float32); occ_mask = tf.cast(occ_mask, tf.float32)
        dp_loss  = tf.constant(0.0);              dg_loss  = tf.constant(0.0)

        if use_adversarial:
            with tf.GradientTape(persistent=True) as dtape:
                fake     = self.generator([occluded, mask], training=True)
                real_cat = tf.concat([occluded, target], axis=-1)
                fake_cat = tf.concat([occluded, fake],   axis=-1)
                dp_loss  = Losses.discriminator(self.patchgan(real_cat,     training=True),
                                                self.patchgan(fake_cat,     training=True))
                dg_loss  = Losses.discriminator(self.global_disc(real_cat,  training=True),
                                                self.global_disc(fake_cat,  training=True))
            self.dp_opt.apply_gradients(zip(
                dtape.gradient(dp_loss, self.patchgan.trainable_variables),
                self.patchgan.trainable_variables))
            self.dg_opt.apply_gradients(zip(
                dtape.gradient(dg_loss, self.global_disc.trainable_variables),
                self.global_disc.trainable_variables))

        with tf.GradientTape() as gtape:
            fake = self.generator([occluded, mask], training=True)
            if use_adversarial:
                fake_cat    = tf.concat([occluded, fake], axis=-1)
                patch_fake  = self.patchgan(fake_cat,    training=False)
                global_fake = self.global_disc(fake_cat, training=False)
            else:
                patch_fake  = tf.zeros([1], dtype=tf.float32)
                global_fake = tf.zeros([1], dtype=tf.float32)
            g_loss, _ = Losses.generator_total(
                target, fake, patch_fake, global_fake, occ_mask,
                use_adversarial=use_adversarial)
        self.g_opt.apply_gradients(zip(
            gtape.gradient(g_loss, self.generator.trainable_variables),
            self.generator.trainable_variables))
        return g_loss, dp_loss, dg_loss

    @tf.function
    def _val_step(self, occluded, mask, target):
        occluded = tf.cast(occluded, tf.float32)
        mask     = tf.cast(mask,     tf.float32)
        target   = tf.cast(target,   tf.float32)
        fake     = self.generator([occluded, mask], training=False)
        cat_real = tf.concat([occluded, target], axis=-1)
        cat_fake = tf.concat([occluded, fake],   axis=-1)
        dp_loss  = Losses.discriminator(self.patchgan(cat_real, False),
                                        self.patchgan(cat_fake, False))
        dg_loss  = Losses.discriminator(self.global_disc(cat_real, False),
                                        self.global_disc(cat_fake, False))
        occ_mask = tf.zeros_like(fake[..., :1])
        g_loss, _ = Losses.generator_total(
            target, fake,
            self.patchgan(cat_fake, False),
            self.global_disc(cat_fake, False),
            occ_mask, use_adversarial=True)
        ssim_v = tf.reduce_mean(tf.image.ssim(target, fake, max_val=1.0))
        psnr_v = tf.reduce_mean(tf.image.psnr(target, fake, max_val=1.0))
        return g_loss, dp_loss, dg_loss, ssim_v, psnr_v

    def _do_checkpoint_save(self, epoch):
        self.ckpt.epoch.assign(epoch + 1)
        self.ckpt.step.assign(self.global_step)
        self.ckpt.best_ssim.assign(self.best_ssim)
        self.arch_version_var.assign(Config.ARCH_VERSION)
        self.ckpt_mgr.save()
        atomic_json_save(self.history, Config.HISTORY_PATH)
        state = {
            "epoch":        epoch + 1,
            "global_step":  self.global_step,
            "best_ssim":    float(self.best_ssim),
            "arch_version": Config.ARCH_VERSION,
            "timestamp":    datetime.now().isoformat(),
        }
        atomic_json_save(state, Config.STATE_PATH)
        print(f"  Checkpoint saved — epoch {epoch+1}  step {self.global_step}")

    def train(self, train_ds, val_ds, val_triplets,
              steps_per_epoch, val_steps, loader):
        self.restore_or_initialize()
        PowerManager.install_signal_handler(self)

        train_iter = iter(train_ds)
        val_iter   = iter(val_ds)

        for epoch in range(self.start_epoch, Config.EPOCHS):
            print(f"\n{'='*65}")
            use_adv  = epoch >= Config.WARMUP_EPOCHS
            phase    = "FULL" if use_adv else f"WARMUP ({epoch+1}/{Config.WARMUP_EPOCHS})"
            print(f"  Epoch {epoch+1}/{Config.EPOCHS}  [{phase}]  "
                  f"[{datetime.now().strftime('%H:%M:%S')}]")
            print('='*65)

            g_sum = dp_sum = dg_sum = 0.0
            train_fn = self._train_step_full if use_adv else self._train_step_warmup

            for step in range(steps_per_epoch):
                (occ, msk, occ_msk), tgt = next(train_iter)
                g_l, dp_l, dg_l = train_fn(occ, msk, tgt, occ_msk)
                g_sum  += float(g_l)
                dp_sum += float(dp_l)
                dg_sum += float(dg_l)
                self.global_step += 1
                self._write_heartbeat(epoch=epoch+1, step=self.global_step)
                if step % 20 == 0:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"step {step:4d}/{steps_per_epoch}  "
                          f"G={g_l:.4f}  Dp={dp_l:.4f}  Dg={dg_l:.4f}")

            avg_g  = g_sum  / steps_per_epoch
            avg_dp = dp_sum / steps_per_epoch
            avg_dg = dg_sum / steps_per_epoch

            vg=vdp=vdg=vssim=vpsnr=0.0
            for _ in range(val_steps):
                (occ, msk, _), tgt = next(val_iter)
                vg_l, vdp_l, vdg_l, vssim_l, vpsnr_l = self._val_step(occ, msk, tgt)
                vg+=float(vg_l); vdp+=float(vdp_l); vdg+=float(vdg_l)
                vssim+=float(vssim_l); vpsnr+=float(vpsnr_l)
            avg_vg=vg/val_steps; avg_vssim=vssim/val_steps; avg_vpsnr=vpsnr/val_steps

            if avg_vssim > self.best_ssim:
                self.best_ssim = avg_vssim
                self.generator.save(os.path.join(Config.RESULTS_DIR, "generator_best.keras"))
                print(f"  New best SSIM: {self.best_ssim:.4f} — generator saved")

            self.history["g_loss"].append(avg_g)
            self.history["dp_loss"].append(avg_dp)
            self.history["dg_loss"].append(avg_dg)
            self.history["val_g_loss"].append(avg_vg)
            self.history["val_dp_loss"].append(vdp/val_steps)
            self.history["val_dg_loss"].append(vdg/val_steps)
            self.history["val_ssim"].append(avg_vssim)
            self.history["val_psnr"].append(avg_vpsnr)

            print(f"\n  Train  G={avg_g:.4f}  Dp={avg_dp:.4f}  Dg={avg_dg:.4f}")
            print(f"  Val    G={avg_vg:.4f}  SSIM={avg_vssim:.4f}  PSNR={avg_vpsnr:.2f} dB")

            if (epoch+1) % 2 == 0:
                self._save_samples(val_triplets, epoch, loader)

            if (epoch+1) % Config.CHECKPOINT_SAVE_FREQ == 0:
                self._do_checkpoint_save(epoch)
                TrainingVisualizer.plot_training_curves(self.history, Config.PLOTS_DIR)

        self._do_checkpoint_save(Config.EPOCHS - 1)
        TrainingVisualizer.plot_training_curves(self.history, Config.PLOTS_DIR)
        self._write_heartbeat(epoch=Config.EPOCHS, status="completed")
        print("\nTraining complete!")

    def _save_samples(self, val_triplets, epoch, loader):
        save_dir = os.path.join(Config.SAMPLE_RECONSTRUCTIONS_DIR, f"epoch_{epoch+1:03d}")
        os.makedirs(save_dir, exist_ok=True)
        originals=[]; occludeds=[]; recons=[]; masks=[]; occ_masks=[]
        ssim_vals=[]; psnr_vals=[]
        for occ_p, mask_p, gt_p in val_triplets[:6]:
            occ, mask, gt = loader.load_triplet(occ_p, mask_p, gt_p)
            recon = self.generator.predict([occ[np.newaxis], mask[np.newaxis]], verbose=0)[0]
            ssim_v = ssim(gt,  recon, channel_axis=-1, data_range=1.0)
            psnr_v = psnr(gt,  recon, data_range=1.0)
            originals.append(gt); occludeds.append(occ); recons.append(recon)
            masks.append(mask); occ_masks.append(1.0 - mask.squeeze())
            ssim_vals.append(ssim_v); psnr_vals.append(psnr_v)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_path = os.path.join(save_dir, f"grid_{timestamp}.png")
        TrainingVisualizer.save_reconstruction_grid(
            originals, occludeds, recons, masks, occ_masks,
            ssim_vals, psnr_vals, grid_path, epoch=epoch+1, split="val")
        print(f"  Sample grid saved  "
              f"Mean SSIM={np.mean(ssim_vals):.4f}  PSNR={np.mean(psnr_vals):.2f}dB")

    def evaluate_test_set(self, test_triplets, loader):
        print("\nEvaluating on test set...")
        rows=[]; originals=[]; occludeds=[]; recons=[]; masks=[]; occ_masks=[]
        ssim_vals=[]; psnr_vals=[]
        for idx, (occ_p, mask_p, gt_p) in enumerate(tqdm(test_triplets, desc="Testing")):
            occ, mask, gt = loader.load_triplet(occ_p, mask_p, gt_p)
            recon  = self.generator.predict([occ[np.newaxis], mask[np.newaxis]], verbose=0)[0]
            ssim_v = ssim(gt, recon, channel_axis=-1, data_range=1.0)
            psnr_v = psnr(gt, recon, data_range=1.0)
            mse_v  = float(np.mean((gt - recon)**2))
            rows.append({"id": idx, "image": os.path.basename(occ_p),
                         "ssim": ssim_v, "psnr": psnr_v, "mse": mse_v})
            if idx < 12:
                originals.append(gt); occludeds.append(occ); recons.append(recon)
                masks.append(mask); occ_masks.append(1.0 - mask.squeeze())
                ssim_vals.append(ssim_v); psnr_vals.append(psnr_v)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_path = os.path.join(Config.TEST_RESULTS_DIR, f"test_grid_{timestamp}.png")
        TrainingVisualizer.save_reconstruction_grid(
            originals, occludeds, recons, masks, occ_masks,
            ssim_vals, psnr_vals, grid_path, split="test")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(Config.TEST_RESULTS_DIR, "test_metrics.csv"), index=False)
        summary = {
            "n": len(df), "ssim_mean": float(df.ssim.mean()),
            "ssim_std": float(df.ssim.std()), "psnr_mean": float(df.psnr.mean()),
            "psnr_std": float(df.psnr.std()), "mse_mean": float(df.mse.mean()),
            "best_ssim": float(df.ssim.max()), "worst_ssim": float(df.ssim.min()),
        }
        atomic_json_save(summary, os.path.join(Config.TEST_RESULTS_DIR, "test_summary.json"))
        print(f"\nTest Results:")
        print(f"  Samples : {summary['n']}")
        print(f"  SSIM    : {summary['ssim_mean']:.4f} +/- {summary['ssim_std']:.4f}")
        print(f"  PSNR    : {summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f} dB")
        print(f"  MSE     : {summary['mse_mean']:.6f}")
        return summary


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main():
    Config.setup_dirs()
    print("TOMATO RECONSTRUCTION GAN — Power-Safe Version")
    print("PhD Thesis: Olagunju Korede Solomon (216882)")
    print("=" * 65)
    print(f"  Occluded   : {Config.OCC_ROOT}")
    print(f"  Masks      : {Config.MASK_ROOT}")
    print(f"  Ground truth: {Config.GT_ROOT}")
    print(f"  Output     : {Config.RESULTS_DIR}")
    print("=" * 65)

    PowerManager.prevent_sleep()
    loader   = TomatoDataLoader()
    triplets = loader.find_all_triplets()

    if not triplets:
        print("No valid triplets found.")
        print("Make sure tomato_dataset_pipeline.py has been run first.")
        PowerManager.allow_sleep()
        return

    train_t, val_t, test_t = loader.split_triplets(triplets)
    if not train_t:
        print("No training triplets.")
        PowerManager.allow_sleep()
        return

    steps_per_epoch      = max(1, len(train_t) // Config.BATCH_SIZE)
    val_steps            = max(1, len(val_t)   // Config.BATCH_SIZE)
    Config.LR_DECAY_STEPS = steps_per_epoch * 10

    print(f"\n  Steps/epoch : {steps_per_epoch}")
    print(f"  Val steps   : {val_steps}\n")

    train_ds = loader.create_tf_dataset(train_t, Config.BATCH_SIZE, is_training=True)
    val_ds   = loader.create_tf_dataset(val_t,   Config.BATCH_SIZE, is_training=False)

    trainer = TomatoGANTrainer()
    try:
        trainer.train(train_ds, val_ds, val_t,
                      steps_per_epoch, val_steps, loader)
        if test_t:
            trainer.evaluate_test_set(test_t, loader)
        else:
            print("No test triplets — skipping evaluation.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        trainer.emergency_save()
        raise
    finally:
        PowerManager.allow_sleep()

    print("\nDone! Results in:", Config.RESULTS_DIR)


if __name__ == "__main__":
    main()