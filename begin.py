# begin.py
"""
TOMATO GRIP FORCE ESTIMATION SYSTEM
Author: OLAGUNJU KOREDE SOLOMON (Student ID: 216882)
Supervisor: Prof. S.O. Akinola

PIPELINE:
  RIPE TOMATO:
    Camera → Detect → CNN (ripe) → FNN (visual features from ROI) → Frontend

  OCCLUDED TOMATO:
    Camera → Detect → CNN (occluded) → GAN (reconstruct) →
    CNN again (reconstructed image) → FNN (visual features from reconstructed image) →
    Frontend (reconstructed image + grip force + physical properties)

NO DUMMY VALUES. NO SIMULATED DATA. NO FAKE RESULTS.
If a component is unavailable, its output is None — never fabricated.
"""

import cv2
import numpy as np
import time
import threading
import os
import json
import math
import base64
import traceback
import queue
import platform
import signal
import sys
import psutil

from datetime import datetime
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

import multiprocessing

# ==================== ARDUINO ====================
try:
    from arduino import ArduinoClient, BLINK_PATTERNS, VALID_COMMANDS
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False
    BLINK_PATTERNS = {}
    VALID_COMMANDS = []

# ==================== PATHS ====================
RECONSTRUCTED_IMAGES_DIR = r"C:\..PhD Thesis\GripForce\ReconstructedImages"
os.makedirs(RECONSTRUCTED_IMAGES_DIR, exist_ok=True)
print(f"Reconstructed images -> {RECONSTRUCTED_IMAGES_DIR}")

IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model')
print(f"Platform: {'Raspberry Pi' if IS_RASPBERRY_PI else 'Local Computer'}")

# ==================== JSON SAFETY ====================
def json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {json_safe(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

# ==================== IMAGE UTILITIES ====================
def to_uint8(img):
    if img is None or (hasattr(img, 'size') and img.size == 0):
        return None
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def encode_to_base64(image_bgr, quality=85):
    try:
        img = to_uint8(image_bgr)
        if img is None:
            return None
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=quality, optimize=True)
        buf.seek(0)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"encode_to_base64: {e}")
        return None


def save_image(image_bgr, tomato_id, suffix=""):
    try:
        img = to_uint8(image_bgr)
        if img is None:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECONSTRUCTED_IMAGES_DIR,
                            f"tomato_{tomato_id}_{ts}{suffix}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")
        return path
    except Exception as e:
        print(f"save_image: {e}")
        return None


def create_blank_frame(text="No Camera Feed", width=640, height=480):
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = cv2.getTextSize(text, font, 1, 2)[0]
    cv2.putText(blank, text, ((width - tw) // 2, height // 2),
                font, 1, (255, 255, 255), 2)
    cv2.putText(blank, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10, height - 10), font, 0.5, (100, 100, 100), 1)
    return blank

# ==================== SPADE LAYER ====================
@register_keras_serializable(package="Custom", name="SPADE")
class SPADE(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
        self.conv_gamma = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv_beta = layers.Conv2D(filters, kernel_size, padding='same')

    def call(self, x, segmentation_map):
        mean = K.mean(x, axis=[1, 2], keepdims=True)
        std = K.std(x, axis=[1, 2], keepdims=True)
        x_norm = (x - mean) / (std + 1e-5)
        seg = tf.image.resize(segmentation_map,
                              [tf.shape(x)[1], tf.shape(x)[2]], method='nearest')
        feat = self.conv(seg)
        return x_norm * (1 + self.conv_gamma(feat)) + self.conv_beta(feat)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return cfg

# ==================== VISUAL FEATURE EXTRACTOR ====================
def extract_visual_features(image_bgr):
    """
    Extract real visual features from a BGR image for FNN input.
    21 features: RGB stats(6) + HSV stats(6) + GLCM texture(4) +
                 contour shape(3) + color ratios(2)
    """
    try:
        img = to_uint8(image_bgr)
        if img is None or img.size == 0:
            return np.zeros(21, dtype=np.float32)

        r = cv2.resize(img, (64, 64))
        features = []

        # RGB mean + std (6)
        for c in range(3):
            ch = r[:, :, c].astype(float)
            features += [float(np.mean(ch) / 255.0), float(np.std(ch) / 255.0)]

        # HSV mean + std (6)
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
        for c in range(3):
            ch = hsv[:, :, c].astype(float)
            mx = 179.0 if c == 0 else 255.0
            features += [float(np.mean(ch) / mx), float(np.std(ch) / mx)]

        # GLCM texture (4)
        gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(float)
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        features.append(float(np.mean(dx**2 + dy**2)) / (255**2))
        features.append(float(np.mean(gray**2)) / (255**2))
        features.append(float(1.0 / (1.0 + np.mean(np.abs(dx)) + np.mean(np.abs(dy)))))
        features.append(float(np.std(gray) / 255.0))

        # Contour shape (3)
        _, binary = cv2.threshold(
            cv2.cvtColor(r, cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            ct = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(ct)
            perim = cv2.arcLength(ct, True)
            circ = 4 * math.pi * area / (perim**2) if perim > 0 else 0.0
            xb, yb, wb, hb = cv2.boundingRect(ct)
            asp = float(wb) / float(hb) if hb > 0 else 1.0
            ext = area / (wb * hb) if wb * hb > 0 else 0.0
        else:
            circ = asp = ext = 0.0
        features += [float(circ), float(asp), float(ext)]

        # Red / green ratio (2)
        hsv64 = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
        total = 64 * 64
        red = cv2.bitwise_or(
            cv2.inRange(hsv64, np.array([0,  80, 80]), np.array([10,  255, 255])),
            cv2.inRange(hsv64, np.array([160, 80, 80]), np.array([179, 255, 255])))
        grn = cv2.inRange(hsv64, np.array([35, 50, 50]), np.array([85, 255, 255]))
        features.append(float(np.sum(red > 0)) / total)
        features.append(float(np.sum(grn > 0)) / total)

        arr = np.array(features, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    except Exception as e:
        print(f"extract_visual_features: {e}")
        return np.zeros(21, dtype=np.float32)


def adapt_features(features, expected):
    """Trim or zero-pad feature vector to match model input size."""
    n = len(features)
    if n == expected:
        return features
    if n > expected:
        return features[:expected]
    return np.concatenate([features, np.zeros(expected - n, dtype=np.float32)])

# ==================== CAMERA MANAGER ====================
class CameraManager:
    def __init__(self):
        self.cap = None
        self.pi_camera = None
        self.raw_capture = None
        self.camera_type = None
        self.last_frame = None
        self.frame_lock = threading.RLock()
        self.running = False
        self.frame_count = 0
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.width = 640
        self.height = 480
        self.fps_target = 30
        self.stop_event = threading.Event()

    def initialize(self):
        print("Initializing camera...")
        for idx in [0, 1, 2, 3]:
            try:
                backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
                cap = cv2.VideoCapture(idx, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.5)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.cap = cap
                        self.camera_type = f"USB Camera {idx}"
                        self.running = True
                        self.stop_event.clear()
                        self.capture_thread = threading.Thread(
                            target=self._capture_loop, daemon=True)
                        self.capture_thread.start()
                        print(f"Camera {self.camera_type} initialized")
                        return True
                cap.release()
            except Exception as e:
                print(f"Camera {idx}: {e}")

        if IS_RASPBERRY_PI:
            try:
                import picamera
                from picamera.array import PiRGBArray
                self.pi_camera = picamera.PiCamera()
                self.pi_camera.resolution = (self.width, self.height)
                self.pi_camera.framerate = self.fps_target
                self.raw_capture = PiRGBArray(
                    self.pi_camera, size=(self.width, self.height))
                self.camera_type = "Pi Camera"
                self.running = True
                self.stop_event.clear()
                self.capture_thread = threading.Thread(
                    target=self._pi_capture_loop, daemon=True)
                self.capture_thread.start()
                print("Pi Camera initialized")
                return True
            except Exception as e:
                print(f"Pi Camera: {e}")

        print("No camera found")
        return False

    def _capture_loop(self):
        failures = 0
        while not self.stop_event.is_set():
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        failures = 0
                        with self.frame_lock:
                            self.last_frame = frame.copy()
                            self.frame_count += 1
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(frame.copy())
                    else:
                        failures += 1
                        if failures >= 5:
                            self._reinitialize()
                            failures = 0
                    time.sleep(0.01)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Capture loop: {e}")
                failures += 1
                time.sleep(0.1)

    def _pi_capture_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.pi_camera and self.raw_capture:
                    self.raw_capture.truncate(0)
                    self.pi_camera.capture(
                        self.raw_capture, format="bgr", use_video_port=True)
                    frame = self.raw_capture.array
                    if frame is not None and frame.size > 0:
                        with self.frame_lock:
                            self.last_frame = frame.copy()
                            self.frame_count += 1
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(frame.copy())
                    time.sleep(0.01)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Pi capture: {e}")
                time.sleep(0.1)

    def _reinitialize(self):
        print("Reinitializing camera...")
        with self.frame_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        time.sleep(1)
        for idx in [0, 1, 2, 3]:
            try:
                cap = cv2.VideoCapture(idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        with self.frame_lock:
                            self.cap = cap
                            self.camera_type = f"USB Camera {idx}"
                        print(f"Camera reinitialized: index {idx}")
                        return
                cap.release()
            except Exception as e:
                print(f"Reinit {idx}: {e}")
        print("Camera reinit failed")

    def read_frame(self):
        try:
            try:
                frame = self.frame_queue.get_nowait()
                if frame is not None and frame.size > 0:
                    return frame.copy()
            except queue.Empty:
                pass
            with self.frame_lock:
                if self.last_frame is not None and self.last_frame.size > 0:
                    return self.last_frame.copy()
            return None
        except Exception as e:
            print(f"read_frame: {e}")
            return None

    def release(self):
        print("Releasing camera...")
        self.stop_event.set()
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        with self.frame_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.pi_camera:
                self.pi_camera.close()
                self.pi_camera = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.last_frame = None
        print("Camera released")

# ==================== TOMATO DETECTOR ====================
class TomatoDetector:
    def __init__(self):
        self.red_lower1 = np.array([0,   100, 100])
        self.red_upper1 = np.array([10,  255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([179, 255, 255])
        self.green_lower = np.array([35,  50,  50])
        self.green_upper = np.array([85,  255, 255])
        self.kernel = np.ones((5, 5), np.uint8)
        self.min_area = 500

        self.tracked = None
        self.tracking_id = None
        self.tomato_counter = 0
        self.lost_frames = 0
        self.max_lost = 30
        self.consecutive = 0
        self.min_consecutive = 3

        self.position_history = []
        self.filtered_positions = []
        self.max_history = 10
        self.filter_size = 5

    def detect(self, frame):
        if frame is None:
            return []
        candidates = []
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            H, W = frame.shape[:2]
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, self.red_lower1, self.red_upper1),
                cv2.inRange(hsv, self.red_lower2, self.red_upper2))
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            combined = cv2.morphologyEx(
                cv2.bitwise_or(red_mask, green_mask),
                cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                perim = cv2.arcLength(contour, True)
                circ = 4 * math.pi * area / perim**2 if perim > 0 else 0.0

                roi_hsv = hsv[y:y+h, x:x+w]
                if roi_hsv.size == 0:
                    continue
                total = w * h
                red_px = (np.sum(cv2.inRange(roi_hsv, self.red_lower1, self.red_upper1) > 0) +
                          np.sum(cv2.inRange(roi_hsv, self.red_lower2, self.red_upper2) > 0))
                grn_px = np.sum(cv2.inRange(roi_hsv, self.green_lower, self.green_upper) > 0)
                red_r = red_px / total if total > 0 else 0.0
                grn_r = grn_px / total if total > 0 else 0.0

                if red_r > 0.3 and red_r > grn_r:
                    ttype, color_score = "RIPE", red_r
                elif grn_r > 0.3:
                    ttype, color_score = "GREEN", grn_r
                else:
                    ttype, color_score = "UNKNOWN", max(red_r, grn_r)

                score = color_score * 0.5 + circ * 0.3 + min(1.0, area / 10000) * 0.2
                candidates.append({
                    'bbox_pixel': [x, y, w, h],
                    'bbox_norm':  [cx / W, cy / H, w / W, h / H],
                    'center':     (cx, cy),
                    'type':       ttype,
                    'score':      score,
                    'area':       area,
                    'circularity': circ,
                    'red_ratio':  red_r,
                    'green_ratio': grn_r,
                })
        except Exception as e:
            print(f"detect: {e}")
        return candidates

    def update(self, frame):
        candidates = self.detect(frame)

        if self.tracked:
            best, best_d = None, float('inf')
            for c in candidates:
                d = math.dist(self.tracked['center'], c['center'])
                if d < 100 and d < best_d:
                    best_d, best = d, c
            if best:
                self.tracked = best
                self.lost_frames = 0
                self._update_positions(best['bbox_pixel'])
                return {**best, 'id': self.tracking_id, 'tracking_lost': False}
            else:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost:
                    print(f"Lost Tomato {self.tracking_id}")
                    self.tracked = None
                    self.tracking_id = None
                    self.consecutive = 0
                    self.position_history.clear()
                    self.filtered_positions.clear()
                    return None
                return {**self.tracked, 'id': self.tracking_id, 'tracking_lost': True}

        if candidates:
            best = max(candidates, key=lambda c: c['score'])
            if best['score'] > 0.3:
                self.consecutive += 1
                if self.consecutive >= self.min_consecutive:
                    self.tomato_counter += 1
                    self.tracking_id = self.tomato_counter
                    self.tracked = best
                    self.lost_frames = 0
                    self.position_history.clear()
                    self.filtered_positions.clear()
                    self._update_positions(best['bbox_pixel'])
                    print(f"Tracking Tomato {self.tracking_id}")
                    return {**best, 'id': self.tracking_id, 'tracking_lost': False}
            else:
                self.consecutive = 0
        return None

    def _update_positions(self, bbox_pixel):
        x, y, w, h = bbox_pixel
        self.position_history.append(
            {'x': x + w / 2, 'y': y + h / 2, 'time': time.time()})
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        if len(self.position_history) >= self.filter_size:
            recent = self.position_history[-self.filter_size:]
            self.filtered_positions.append({
                'x': sum(p['x'] for p in recent) / len(recent),
                'y': sum(p['y'] for p in recent) / len(recent),
                'time': time.time()
            })
            if len(self.filtered_positions) > self.filter_size:
                self.filtered_positions.pop(0)

    def get_filtered_position(self):
        if self.filtered_positions:
            return self.filtered_positions[-1]
        if self.position_history:
            return self.position_history[-1]
        return None

    def reset(self):
        self.tracked = None
        self.tracking_id = None
        self.tomato_counter = 0
        self.lost_frames = 0
        self.consecutive = 0
        self.position_history.clear()
        self.filtered_positions.clear()

# ==================== MODEL MANAGER ====================
class ModelManager:
    CNN_SIZE = (224, 224)
    GAN_SIZE  = (224, 224)
    CNN_LABELS = ['Ripe Tomato', 'Occluded Tomato']

    def __init__(self):
        self.cnn = None
        self.fnn = None
        self.gan = None
        self.fnn_input_size  = None
        self.fnn_output_size = None

    def load_all(self):
        self.load_cnn()
        self.load_fnn()
        self.load_gan()

    def load_cnn(self):
        path = r"C:\..PhD Thesis\DataSet\Trained_models\tomato_classifier_final.h5"
        if not os.path.exists(path):
            print(f"CNN not found: {path}")
            return
        try:
            self.cnn = load_model(path)
            print(f"CNN loaded — input: {self.cnn.input_shape}")
        except Exception as e:
            print(f"CNN load error: {e}")

    def load_fnn(self):
        path = (r"C:\..PhD Thesis\DataSet\FNN_Regression"
                r"\FNN_Regression_20251216_160441\fnn_regression_model.h5")
        if not os.path.exists(path):
            print(f"FNN not found: {path}")
            return
        try:
            self.fnn = load_model(path, custom_objects={"mse": MeanSquaredError()})
            self.fnn_input_size  = self.fnn.input_shape[1]
            self.fnn_output_size = self.fnn.output_shape[1]
            print(f"FNN loaded — input: {self.fnn_input_size}  output: {self.fnn_output_size}")
        except Exception as e:
            print(f"FNN load error: {e}")

    def load_gan(self):
        path = r"C:\..PhD Thesis\DataSet\GANS\tomato_reconstruction_generator.keras"
        if not os.path.exists(path):
            print(f"GAN not found: {path}")
            return
        try:
            self.gan = load_model(path, custom_objects={"SPADE": SPADE})
            print(f"GAN loaded — input: {self.gan.input_shape}")
        except Exception as e:
            print(f"GAN load error: {e}")

    # ── CNN inference ────────────────────────────────────────────────────────
    def run_cnn(self, image_bgr):
        """
        Classify a BGR image.
        Returns dict or None.
        """
        if self.cnn is None:
            return None
        img = to_uint8(image_bgr)
        if img is None:
            return None
        try:
            inp = np.expand_dims(
                cv2.resize(img, self.CNN_SIZE).astype(np.float32) / 255.0, axis=0)
            probs = self.cnn.predict(inp, verbose=0)[0]
            cid   = int(np.argmax(probs))
            conf  = float(np.max(probs))
            label = self.CNN_LABELS[cid] if cid < len(self.CNN_LABELS) else f"Class {cid}"
            return {
                'class_id':   cid,
                'label':      label,
                'confidence': conf,
                'is_ripe':    cid == 0,
                'is_occluded': cid == 1,
                'all_probs':  probs.tolist(),
            }
        except Exception as e:
            print(f"CNN inference: {e}")
            traceback.print_exc()
            return None

    # ── GAN inference ────────────────────────────────────────────────────────
    def run_gan(self, roi_bgr):
        """
        Reconstruct an occluded ROI.
        Returns reconstructed BGR image (uint8) or None.
        """
        if self.gan is None:
            return None
        img = to_uint8(roi_bgr)
        if img is None:
            return None
        try:
            resized = cv2.resize(img, self.GAN_SIZE).astype(np.float32) / 255.0

            # Real segmentation map via Otsu thresholding of the ROI
            gray = cv2.cvtColor(cv2.resize(img, self.GAN_SIZE), cv2.COLOR_BGR2GRAY)
            _, seg_bin = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            seg_map = (seg_bin / 255.0).astype(np.float32).reshape(
                self.GAN_SIZE[1], self.GAN_SIZE[0], 1)

            t0 = time.time()
            output = self.gan.predict(
                [np.expand_dims(resized, 0), np.expand_dims(seg_map, 0)],
                verbose=0)[0]
            print(f"GAN inference: {(time.time()-t0)*1000:.1f} ms")
            return to_uint8(output)
        except Exception as e:
            print(f"GAN inference: {e}")
            traceback.print_exc()
            return None

    # ── FNN inference ────────────────────────────────────────────────────────
    def run_fnn(self, image_bgr):
        """
        Predict physical properties from a BGR image.
        Features are extracted from real pixel content.
        Returns dict or None.
        """
        if self.fnn is None:
            return None
        img = to_uint8(image_bgr)
        if img is None:
            return None
        try:
            features = extract_visual_features(img)
            adapted  = adapt_features(features, self.fnn_input_size)
            raw = self.fnn.predict(adapted.reshape(1, -1), verbose=0)[0]

            def g(i):
                return float(max(0.0, raw[i])) if i < len(raw) else None

            result = {
                'grip_force_N':  g(0),
                'weight_g':      g(1),
                'size_mm':       g(2),
                'pressure_kPa':  g(3),
                'force_N':       g(4),
                'torque_Nm':     g(5),
                'raw_outputs':   raw.tolist(),
                'features_used': int(len(adapted)),
            }
            print(f"FNN -> grip_force={result['grip_force_N']:.4f} N  "
                  f"weight={result['weight_g']:.2f} g")
            return result
        except Exception as e:
            print(f"FNN inference: {e}")
            traceback.print_exc()
            return None

    # ── Reconstruction quality metrics ───────────────────────────────────────
    @staticmethod
    def reconstruction_metrics(original_bgr, reconstructed_bgr):
        try:
            orig = to_uint8(original_bgr)
            rec  = to_uint8(reconstructed_bgr)
            if orig is None or rec is None:
                return {}
            if orig.shape[:2] != rec.shape[:2]:
                orig = cv2.resize(orig, (rec.shape[1], rec.shape[0]))
            og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(float)
            rg = cv2.cvtColor(rec,  cv2.COLOR_BGR2GRAY).astype(float)

            mse  = float(np.mean((og - rg)**2))
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 100.0

            mx, my = np.mean(og), np.mean(rg)
            sx, sy = np.std(og),  np.std(rg)
            sxy    = float(np.mean((og - mx) * (rg - my)))
            c1, c2 = (0.01*255)**2, (0.03*255)**2
            ssim = float(
                ((2*mx*my + c1)*(2*sxy + c2)) /
                ((mx**2 + my**2 + c1)*(sx**2 + sy**2 + c2)))

            mae  = float(np.mean(np.abs(og - rg)))
            corr = 0.0
            if sx > 0 and sy > 0:
                m = np.corrcoef(og.flatten(), rg.flatten())
                corr = float(m[0, 1]) if m.shape == (2, 2) else 0.0

            eo   = cv2.Canny(og.astype(np.uint8), 50, 150)
            er   = cv2.Canny(rg.astype(np.uint8), 50, 150)
            un   = np.sum((eo > 0) | (er > 0))
            ep   = float(np.sum((eo > 0) & (er > 0)) / un) if un > 0 else 1.0

            pq   = float(min(1.0, max(0.0,
                    ssim*0.4 + min(1.0, psnr/50)*0.3 + corr*0.2 + ep*0.1)))
            grade = ("EXCELLENT" if pq > 0.8 else
                     "GOOD"      if pq > 0.6 else
                     "FAIR"      if pq > 0.4 else
                     "POOR"      if pq > 0.2 else "UNACCEPTABLE")
            return {
                'mse': mse, 'psnr': psnr, 'ssim': ssim, 'mae': mae,
                'correlation': corr, 'edge_preservation': ep,
                'perceptual_quality': pq, 'quality_grade': grade,
            }
        except Exception as e:
            print(f"reconstruction_metrics: {e}")
            return {}

# ==================== PIPELINE ====================
class TomatoPipeline:
    """
    Executes the full processing pipeline:

    RIPE:
      ROI -> CNN(ripe) -> FNN(ROI) -> result

    OCCLUDED:
      ROI -> CNN(occluded) -> GAN(ROI) -> reconstructed
          -> CNN(reconstructed) -> FNN(reconstructed) -> result
    """
    def __init__(self, models: ModelManager):
        self.models = models
        self._cache  = {}
        self._cache_ttl = 360.0

    def process(self, tomato, frame):
        tid = tomato.get('id')
        roi = self._crop_roi(frame, tomato)
        if roi is None:
            return self._empty(tid, "ROI extraction failed")

        # ── Step 1: CNN on raw ROI ───────────────────────────────────────────
        cnn_raw = self.models.run_cnn(roi)
        if cnn_raw is None:
            return self._empty(tid, "CNN model not loaded")

        is_occluded = cnn_raw['is_occluded']

        # ── RIPE PATH ────────────────────────────────────────────────────────
        if not is_occluded:
            cached = self._get_cache(tid)
            if cached and cached.get('path') == 'RIPE':
                cached['is_cached'] = True
                cached['cnn'] = cnn_raw
                return cached

            fnn = self.models.run_fnn(roi)
            result = {
                'tomato_id':               tid,
                'path':                    'RIPE',
                'cnn':                     cnn_raw,
                'cnn_on_reconstruction':   None,
                'gan':                     None,
                'fnn':                     fnn,
                'roi_base64':              encode_to_base64(roi),
                'reconstructed_base64':    None,
                'comparison_base64':       None,
                'reconstruction_metrics':  None,
                'is_cached':               False,
                'timestamp':               time.time(),
                'error':                   None,
            }
            self._set_cache(tid, result)
            return result

        # ── OCCLUDED PATH ────────────────────────────────────────────────────
        # Step 2: GAN reconstruction
        reconstructed = self.models.run_gan(roi)
        if reconstructed is None:
            return {
                'tomato_id':               tid,
                'path':                    'OCCLUDED_NO_GAN',
                'cnn':                     cnn_raw,
                'cnn_on_reconstruction':   None,
                'gan':                     None,
                'fnn':                     None,
                'roi_base64':              encode_to_base64(roi),
                'reconstructed_base64':    None,
                'comparison_base64':       None,
                'reconstruction_metrics':  None,
                'is_cached':               False,
                'timestamp':               time.time(),
                'error':                   'GAN model not loaded or reconstruction failed',
            }

        # Step 3: CNN on reconstructed image
        cnn_rec = self.models.run_cnn(reconstructed)

        # Step 4: FNN on reconstructed image
        fnn = self.models.run_fnn(reconstructed)

        # Step 5: quality metrics
        metrics    = ModelManager.reconstruction_metrics(roi, reconstructed)
        comparison = self._side_by_side(roi, reconstructed)

        # Save to disk
        save_image(roi,           tid, "_original")
        save_image(reconstructed, tid, "_reconstructed")
        if comparison is not None:
            save_image(comparison, tid, "_comparison")

        result = {
            'tomato_id':               tid,
            'path':                    'OCCLUDED',
            'cnn':                     cnn_raw,
            'cnn_on_reconstruction':   cnn_rec,
            'gan':                     {'success': True, 'metrics': metrics},
            'fnn':                     fnn,
            'roi_base64':              encode_to_base64(roi),
            'reconstructed_base64':    encode_to_base64(reconstructed),
            'comparison_base64':       encode_to_base64(comparison) if comparison is not None else None,
            'reconstruction_metrics':  metrics,
            'is_cached':               False,
            'timestamp':               time.time(),
            'error':                   None,
        }
        self._set_cache(tid, result)
        return result

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _crop_roi(frame, tomato):
        if frame is None or tomato is None:
            return None
        x, y, w, h = tomato['bbox_pixel']
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return None
        roi = frame[y:y+h, x:x+w]
        return roi if roi.size > 0 else None

    @staticmethod
    def _side_by_side(roi_bgr, rec_bgr):
        try:
            orig = to_uint8(roi_bgr)
            rec  = to_uint8(rec_bgr)
            if orig is None or rec is None:
                return None
            H = 300
            ow = max(50, int(orig.shape[1] * H / orig.shape[0]))
            rw = max(50, int(rec.shape[1]  * H / rec.shape[0]))
            canvas = np.zeros((H + 40, ow + rw + 20, 3), dtype=np.uint8)
            canvas[20:20+H, 10:10+ow]               = cv2.resize(orig, (ow, H))
            canvas[20:20+H, 10+ow+10:10+ow+10+rw]   = cv2.resize(rec,  (rw, H))
            cv2.putText(canvas, "Original (Occluded)",
                        (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, "GAN Reconstructed",
                        (10+ow+10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return canvas
        except Exception as e:
            print(f"_side_by_side: {e}")
            return None

    def _get_cache(self, tid):
        e = self._cache.get(tid)
        if e and time.time() - e.get('timestamp', 0) < self._cache_ttl:
            return e
        return None

    def _set_cache(self, tid, result):
        self._cache[tid] = result
        if len(self._cache) > 5:
            oldest = min(self._cache,
                         key=lambda k: self._cache[k].get('timestamp', 0))
            del self._cache[oldest]

    @staticmethod
    def _empty(tid, msg):
        return {
            'tomato_id': tid, 'path': None,
            'cnn': None, 'cnn_on_reconstruction': None,
            'gan': None, 'fnn': None,
            'roi_base64': None, 'reconstructed_base64': None,
            'comparison_base64': None, 'reconstruction_metrics': None,
            'is_cached': False, 'timestamp': time.time(), 'error': msg,
        }

# ==================== ARDUINO CONTROLLER ====================
class ArduinoController:
    def __init__(self, socketio):
        self.client = ArduinoClient(socketio) if ARDUINO_AVAILABLE else None
        self.last_cmd_time = 0
        self.min_interval  = 2.0
        self.last_blink    = None

    def start(self):
        if self.client:
            self.client.start()

    def stop(self):
        if self.client:
            self.client.stop()

    def get_status(self):
        return self.client.get_status() if self.client else None

    def get_live_status(self):
        return self.client.get_live_status() if self.client else None

    def blink(self, direction):
        now = time.time()
        if not self.client:
            return None
        if now - self.last_cmd_time < self.min_interval:
            return None
        if direction == self.last_blink:
            return None
        result = self.client.blink(direction)
        if result and result.get('success'):
            self.last_blink    = direction
            self.last_cmd_time = now
        return result

    def move(self, direction):
        return self.client.move(direction) if self.client else None

# ==================== MAIN SYSTEM ====================
class TomatoSystem:
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
        self.camera   = CameraManager()
        self.detector = TomatoDetector()
        self.models   = ModelManager()
        self.pipeline = TomatoPipeline(self.models)
        self.arduino  = ArduinoController(socketio_instance)

        self.running    = False
        self.stop_event = threading.Event()
        self._lock      = threading.RLock()

        self.last_frame     = None
        self.last_tomato    = None
        self.last_result    = None
        self.last_annotated = None

        self.fps          = 0.0
        self._frame_count = 0
        self._start_time  = time.time()

        # zone state
        self.current_zone    = "CENTER"
        self.last_stable_zone = "CENTER"
        self.zone_candidate  = "CENTER"
        self.zone_stability  = 0
        self.zone_threshold  = 5
        self.zone_changed    = False
        self.last_blink_dir  = None
        self.frontend_target = None

        # stats
        self.total_ripe            = 0
        self.total_occluded        = 0
        self.total_reconstructions = 0

        # pipeline runs in its own thread per detection
        self._pipeline_lock = threading.Lock()
        self._pipeline_running = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        print("=" * 60)
        print("STARTING TOMATO GRIP FORCE ESTIMATION SYSTEM")
        print("=" * 60)
        self.models.load_all()
        if not self.camera.initialize():
            print("Camera init failed")
            return False
        self.arduino.start()
        self.running = True
        self.stop_event.clear()
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        print("System running")
        return True

    def stop(self):
        print("Stopping...")
        self.stop_event.set()
        self.running = False
        self.arduino.stop()
        if hasattr(self, '_tracking_thread') and self._tracking_thread.is_alive():
            self._tracking_thread.join(timeout=3.0)
        self.camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Stopped")

    # ── tracking loop ─────────────────────────────────────────────────────────
    def _tracking_loop(self):
        print("Tracking loop started")
        while self.running and not self.stop_event.is_set():
            try:
                t0 = time.time()
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.033)
                    continue

                self._frame_count += 1
                if self._frame_count % 30 == 0:
                    elapsed = time.time() - self._start_time
                    self.fps = self._frame_count / elapsed if elapsed > 0 else 0.0

                tomato = self.detector.update(frame)
                with self._lock:
                    self.last_frame  = frame.copy()
                    self.last_tomato = tomato

                if tomato and not tomato.get('tracking_lost', False):
                    # Run pipeline in background (non-blocking)
                    if not self._pipeline_running:
                        t = threading.Thread(
                            target=self._run_pipeline,
                            args=(tomato.copy(), frame.copy()),
                            daemon=True)
                        t.start()

                    self._update_zone(tomato, frame)
                    self._handle_arduino()
                    self._emit_target_position(tomato, frame)

                annotated = self._annotate(frame, tomato)
                with self._lock:
                    self.last_annotated = annotated

                time.sleep(max(0, 0.033 - (time.time() - t0)))

            except Exception as e:
                print(f"Tracking loop: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _run_pipeline(self, tomato, frame):
        self._pipeline_running = True
        try:
            result = self.pipeline.process(tomato, frame)
            with self._lock:
                self.last_result = result

            if result.get('path') == 'RIPE':
                self.total_ripe += 1
            elif result.get('path') == 'OCCLUDED':
                self.total_occluded += 1
                self.total_reconstructions += 1
            elif result.get('path') == 'OCCLUDED_NO_GAN':
                self.total_occluded += 1

            self.socketio.emit(
                'pipeline_result', json_safe(self._to_frontend(result)))
        except Exception as e:
            print(f"_run_pipeline: {e}")
            traceback.print_exc()
        finally:
            self._pipeline_running = False

    # ── format for frontend ────────────────────────────────────────────────────
    def _to_frontend(self, result):
        if result is None:
            return {'success': False, 'message': 'No result'}
        path    = result.get('path')
        cnn     = result.get('cnn') or {}
        cnn_rec = result.get('cnn_on_reconstruction')
        fnn     = result.get('fnn') or {}
        metrics = result.get('reconstruction_metrics') or {}
        return {
            'success':   result.get('error') is None,
            'tomato_id': result.get('tomato_id'),
            'path':      path,
            'error':     result.get('error'),
            'timestamp': result.get('timestamp'),
            'is_cached': result.get('is_cached', False),

            'cnn': {
                'label':       cnn.get('label'),
                'confidence':  cnn.get('confidence'),
                'is_ripe':     cnn.get('is_ripe'),
                'is_occluded': cnn.get('is_occluded'),
            } if cnn else None,

            'cnn_on_reconstruction': {
                'label':       cnn_rec.get('label'),
                'confidence':  cnn_rec.get('confidence'),
                'is_ripe':     cnn_rec.get('is_ripe'),
                'is_occluded': cnn_rec.get('is_occluded'),
            } if cnn_rec else None,

            'grip_force': {
                'grip_force_N': fnn.get('grip_force_N'),
                'weight_g':     fnn.get('weight_g'),
                'size_mm':      fnn.get('size_mm'),
                'pressure_kPa': fnn.get('pressure_kPa'),
                'force_N':      fnn.get('force_N'),
                'torque_Nm':    fnn.get('torque_Nm'),
                'fnn_source':   'reconstructed_image' if path == 'OCCLUDED' else 'original_roi',
            } if fnn else None,

            'images': {
                'roi':           result.get('roi_base64'),
                'reconstructed': result.get('reconstructed_base64'),
                'comparison':    result.get('comparison_base64'),
            },

            'reconstruction_quality': {
                'psnr':               metrics.get('psnr'),
                'ssim':               metrics.get('ssim'),
                'perceptual_quality': metrics.get('perceptual_quality'),
                'quality_grade':      metrics.get('quality_grade'),
            } if metrics else None,
        }

    # ── zone / Arduino ─────────────────────────────────────────────────────────
    def _update_zone(self, tomato, frame):
        H, W = frame.shape[:2]
        fpos = self.detector.get_filtered_position()
        cx, cy = ((fpos['x'], fpos['y']) if fpos
                  else (tomato['bbox_norm'][0]*W, tomato['bbox_norm'][1]*H))

        if cx < W*0.4:
            zone = ("TOP_LEFT" if cy < H*0.4 else
                    "BOTTOM_LEFT" if cy > H*0.6 else "LEFT")
        elif cx > W*0.6:
            zone = ("TOP_RIGHT" if cy < H*0.4 else
                    "BOTTOM_RIGHT" if cy > H*0.6 else "RIGHT")
        else:
            zone = ("TOP_CENTER" if cy < H*0.4 else
                    "BOTTOM_CENTER" if cy > H*0.6 else "CENTER")

        if zone != self.zone_candidate:
            self.zone_candidate = zone
            self.zone_stability = 1
            self.zone_changed   = False
        else:
            self.zone_stability += 1

        if self.zone_stability >= self.zone_threshold:
            self.zone_changed    = (self.last_stable_zone != zone)
            self.last_stable_zone = zone
            self.current_zone    = zone
        else:
            self.current_zone  = self.last_stable_zone
            self.zone_changed  = False

    def _handle_arduino(self):
        if not self.zone_changed:
            return
        zone = self.current_zone
        direction = (
            "LEFT"  if "LEFT"   in zone else
            "RIGHT" if "RIGHT"  in zone else
            "UP"    if "TOP"    in zone else
            "DOWN"  if "BOTTOM" in zone else
            "GRIP"  if zone == "CENTER" else None)
        if direction:
            result = self.arduino.blink(direction)
            if result and result.get('success'):
                self.last_blink_dir = direction
                print(f"Arduino blink: {direction}")

    def _emit_target_position(self, tomato, frame):
        H, W = frame.shape[:2]
        fpos = self.detector.get_filtered_position()
        cx, cy = ((fpos['x'], fpos['y']) if fpos
                  else (tomato['bbox_norm'][0]*W, tomato['bbox_norm'][1]*H))
        if abs(cx - W//2) > W*0.1:
            direction = "left" if cx < W//2 else "right"
        elif abs(cy - H//2) > H*0.1:
            direction = "up" if cy < H//2 else "down"
        else:
            direction = "center"
        self.frontend_target = direction
        try:
            self.socketio.emit('robot_target_position', json_safe({
                'target_direction': direction,
                'active':           direction != "center",
                'tomato_id':        tomato.get('id'),
                'zone':             self.current_zone,
                'timestamp':        time.time(),
            }))
        except Exception as e:
            print(f"emit target position: {e}")

    # ── annotated frame ────────────────────────────────────────────────────────
    def _annotate(self, frame, tomato):
        out = frame.copy()
        H, W = out.shape[:2]

        if tomato and not tomato.get('tracking_lost', False):
            x, y, w, h = tomato['bbox_pixel']
            result = self.last_result

            is_occ = False
            label  = tomato.get('type', 'TOMATO')
            conf   = tomato.get('score', 0.0)

            if result and result.get('tomato_id') == tomato.get('id'):
                cnn = result.get('cnn')
                if cnn:
                    is_occ = cnn.get('is_occluded', False)
                    label  = cnn.get('label', label)
                    conf   = cnn.get('confidence', conf)

            color = (0, 0, 255) if is_occ else (0, 255, 0)
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)

            # Label
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            ov = out.copy()
            cv2.rectangle(ov, (x, y-lh-10), (x+lw+10, y), (0,0,0), -1)
            cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
            cv2.putText(out, label, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence
            ct = f"Conf: {conf:.2f}"
            cw, ch = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            ov = out.copy()
            cv2.rectangle(ov, (x, y+h+5), (x+cw+10, y+h+ch+15), (0,0,0), -1)
            cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
            cv2.putText(out, ct, (x+5, y+h+ch+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # FNN overlay
            if result and result.get('fnn'):
                fnn = result['fnn']
                lines = []
                if fnn.get('grip_force_N') is not None:
                    lines.append(f"Grip: {fnn['grip_force_N']:.4f} N")
                if fnn.get('weight_g') is not None:
                    lines.append(f"Weight: {fnn['weight_g']:.1f} g")
                if result.get('path') == 'OCCLUDED':
                    lines.append("FNN src: GAN reconstruction")
                elif result.get('path') == 'RIPE':
                    lines.append("FNN src: direct ROI")

                sx, sy = W - 240, 50
                for i, line in enumerate(lines):
                    tw2, th2 = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                    ov = out.copy()
                    cv2.rectangle(ov, (sx-5, sy+i*24-th2-3),
                                  (sx+tw2+5, sy+i*24+5), (0,0,0), -1)
                    cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
                    cv2.putText(out, line, (sx, sy+i*24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 255, 255) if i == 0 else (200,200,200), 1)

            # GAN active banner
            if is_occ:
                bt = "GAN RECONSTRUCTION ACTIVE"
                bw, bh = cv2.getTextSize(bt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                ov = out.copy()
                cv2.rectangle(ov, (5, H-30), (bw+15, H-5), (0,0,0), -1)
                cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
                cv2.putText(out, bt, (10, H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

        # Status bar
        status = (f"Tomato {tomato.get('id','?')}" if tomato
                  else "Waiting for tomato...")
        sw, sh = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        ov = out.copy()
        cv2.rectangle(ov, (5, 5), (sw+15, sh+15), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
        cv2.putText(out, status, (10, sh+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255) if tomato else (150,150,150), 2)

        # FPS
        ft = f"FPS: {self.fps:.1f}"
        fw, fh = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(out, ft, (W-fw-10, fh+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        return out

    # ── public accessors ───────────────────────────────────────────────────────
    def get_annotated_frame(self):
        with self._lock:
            return self.last_annotated.copy() if self.last_annotated is not None else None

    def get_latest_result(self):
        with self._lock:
            r = self.last_result
            t = self.last_tomato
        if r is None:
            return {'success': False, 'message': 'No result yet'}
        return json_safe({'success': True, 'tomato': t,
                          'result': self._to_frontend(r)})

    def get_analytics(self):
        arduino_s = self.arduino.get_status()
        arduino_l = self.arduino.get_live_status()
        proc = psutil.Process(os.getpid())
        return json_safe({
            'success': True,
            'system': {
                'running':    self.running,
                'uptime_s':   round(time.time() - self._start_time, 2),
                'fps':        round(self.fps, 2),
                'memory_mb':  round(proc.memory_info().rss / 1024 / 1024, 2),
                'camera_type': self.camera.camera_type,
                'platform':   'Raspberry Pi' if IS_RASPBERRY_PI else 'Local Computer',
            },
            'models': {
                'cnn_loaded':      self.models.cnn is not None,
                'fnn_loaded':      self.models.fnn is not None,
                'gan_loaded':      self.models.gan is not None,
                'fnn_input_size':  self.models.fnn_input_size,
                'fnn_output_size': self.models.fnn_output_size,
            },
            'tracking': {
                'active':            self.last_tomato is not None,
                'tomato_count':      self.detector.tomato_counter,
                'current_tomato_id': self.last_tomato.get('id') if self.last_tomato else None,
                'current_zone':      self.current_zone,
                'last_blink_dir':    self.last_blink_dir,
                'frontend_target':   self.frontend_target,
            },
            'pipeline_stats': {
                'total_ripe':            self.total_ripe,
                'total_occluded':        self.total_occluded,
                'total_reconstructions': self.total_reconstructions,
            },
            'arduino': {
                'available': ARDUINO_AVAILABLE,
                'status':    arduino_s,
                'live':      arduino_l,
            },
            'latest_result': self._to_frontend(self.last_result) if self.last_result else None,
        })

    def get_health(self):
        arduino_s = self.arduino.get_status()
        arduino_l = self.arduino.get_live_status()
        return json_safe({
            'success':          True,
            'running':          self.running,
            'fps':              round(self.fps, 2),
            'uptime_s':         round(time.time() - self._start_time, 2),
            'tracking':         self.last_tomato is not None,
            'models': {
                'cnn': self.models.cnn is not None,
                'fnn': self.models.fnn is not None,
                'gan': self.models.gan is not None,
            },
            'arduino_connected': arduino_s.get('connected') if arduino_s else None,
            'emergency_stop':    arduino_l.get('emergency_stop') if arduino_l else None,
            'current_zone':      self.current_zone,
            'last_blink_dir':    self.last_blink_dir,
        })


# ==================== FLASK + SOCKETIO ====================
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode='threading', ping_timeout=60, ping_interval=25)

system: TomatoSystem = None


@app.route('/')
def index():
    return jsonify({
        'name':    'Tomato Grip Force Estimation System',
        'author':  'OLAGUNJU KOREDE SOLOMON (216882)',
        'pipeline': {
            'RIPE':     'Camera → Detect → CNN(ripe) → FNN(ROI) → Frontend',
            'OCCLUDED': 'Camera → Detect → CNN(occluded) → GAN → CNN(reconstructed) → FNN(reconstructed) → Frontend',
        },
        'arduino':   ARDUINO_AVAILABLE,
        'endpoints': ['/start', '/stop', '/video_feed', '/detect', '/analytics', '/health'],
    })


@app.route('/start', methods=['POST'])
def start_system():
    global system
    if system is None:
        system = TomatoSystem(socketio)
    if system.running:
        return jsonify({'success': True, 'message': 'Already running'})
    ok = system.start()
    return jsonify({'success': ok,
                    'message': 'Started' if ok else 'Failed — check camera and model paths'})


@app.route('/stop', methods=['POST'])
def stop_system():
    global system
    if system is None:
        return jsonify({'success': False, 'message': 'Not initialized'})
    system.stop()
    return jsonify({'success': True, 'message': 'Stopped'})


@app.route('/video_feed')
def video_feed():
    def generate():
        t0 = time.time()
        while time.time() - t0 < 5:
            if system and system.running:
                break
            time.sleep(0.1)
        blank_n = 0
        while True:
            try:
                if system and system.running:
                    frame = system.get_annotated_frame()
                    if frame is not None:
                        ret, jpeg = cv2.imencode(
                            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n'
                                   + jpeg.tobytes() + b'\r\n')
                else:
                    msg = "Starting..." if blank_n < 30 else "Waiting for system..."
                    ret, jpeg = cv2.imencode('.jpg', create_blank_frame(msg))
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + jpeg.tobytes() + b'\r\n')
                    blank_n += 1
                time.sleep(0.033)
            except GeneratorExit:
                break
            except Exception as e:
                print(f"video_feed: {e}")
                time.sleep(0.1)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['GET'])
def detect():
    if system is None or not system.running:
        return jsonify({'success': False, 'message': 'System not running'})
    return jsonify(system.get_latest_result())


@app.route('/analytics', methods=['GET'])
def analytics():
    if system is None:
        return jsonify({'success': False, 'message': 'Not initialized'})
    return jsonify(system.get_analytics())


@app.route('/health', methods=['GET'])
def health():
    if system is None:
        return jsonify({'success': True, 'running': False,
                        'models': {'cnn': False, 'fnn': False, 'gan': False},
                        'arduino_connected': None})
    return jsonify(system.get_health())


# ── WebSocket ─────────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print('Frontend connected')
    emit('connected', {'message': 'Tomato Grip Force Estimation System ready'})


@socketio.on('disconnect')
def on_disconnect():
    print('Frontend disconnected')


@socketio.on('request_latest_result')
def on_request_latest():
    if system:
        emit('pipeline_result',
             json_safe(system._to_frontend(system.last_result)))


@socketio.on('request_force_data')
def on_request_force():
    if system is None or system.arduino.client is None:
        emit('force_sensor_data', {'error': 'Arduino not connected'})
        return
    live    = system.arduino.get_live_status()
    sensors = live.get('force_sensors', {}) if live else {}
    emit('force_sensor_data', json_safe({
        'sensor1':       sensors.get('sensor1'),
        'sensor2':       sensors.get('sensor2'),
        'grip_force':    sensors.get('grip_force'),
        'total_force':   sensors.get('total'),
        'emergency_stop': live.get('emergency_stop') if live else None,
        'timestamp':     time.time(),
    }))


@socketio.on('request_target_position')
def on_request_target():
    if system and system.last_tomato:
        emit('robot_target_position', json_safe({
            'target_direction': system.frontend_target,
            'active':           system.frontend_target != 'center',
            'tomato_id':        system.last_tomato.get('id'),
            'zone':             system.current_zone,
            'timestamp':        time.time(),
        }))


# ── signal / main ─────────────────────────────────────────────────────────────
def _signal_handler(sig, frame):
    print("\nInterrupt — shutting down")
    global system
    if system:
        system.stop()
    sys.exit(0)


def main():
    if platform.system() == 'Windows':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    global system
    print("\n" + "="*70)
    print("  TOMATO GRIP FORCE ESTIMATION SYSTEM")
    print("  OLAGUNJU KOREDE SOLOMON — Student ID 216882")
    print("  Supervisor: Prof. S.O. Akinola")
    print("="*70)
    print(f"  Arduino  : {'YES' if ARDUINO_AVAILABLE else 'NO (arduino.py not found)'}")
    print(f"  GAN dir  : {RECONSTRUCTED_IMAGES_DIR}")
    print("  Server   : http://127.0.0.1:5000")
    print("  Ctrl+C to stop\n")

    system = TomatoSystem(socketio)

    try:
        socketio.run(app, host='0.0.0.0', port=5000,
                     debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if system:
            system.stop()
    except Exception as e:
        print(f"\nFatal: {e}")
        traceback.print_exc()
        if system:
            system.stop()


if __name__ == '__main__':
    main()