# main.py - COMPLETE FIXED VERSION WITH REAL FNN INTEGRATION
import os
import time
import threading
import logging
import math
import base64
import sys
import json
import psutil
import gc
import signal
import traceback
import queue
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

# ----------------------------
# TensorFlow availability check
# ----------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print("✅ TensorFlow is available")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TensorFlow not available: {e}")

# Add near the top with other imports
import csv
from datetime import datetime

# Create default configuration directory if it doesn't exist
CONFIG_DIR = "/home/oladipo/SolomonDeepLearningModels"
os.makedirs(CONFIG_DIR, exist_ok=True)

# ----------------------------
# Create Default Configuration Files
# ----------------------------
def create_default_configs():
    """Create default configuration files if they don't exist"""
    configs = {
        "classification_config.json": {
            "input_size": [224, 224],
            "classes": ["Ripe", "Occluded"],
            "occlusion_threshold": 0.6,
            "ripe_threshold": 0.7,
            "batch_size": 32,
            "normalization": True,
            "color_space": "RGB"
        },
        "regression_config.json": {
            "input_features": 30,
            "output_names": ["weight", "size", "pressure", "grip_force", "force", "torque"],
            "output_scaling": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ranges": {
                "weight": {"min": 50, "max": 500},
                "size": {"min": 3, "max": 15},
                "pressure": {"min": 5, "max": 100},
                "grip_force": {"min": 1, "max": 15},
                "force": {"min": 0.5, "max": 20},
                "torque": {"min": 0.01, "max": 2}
            },
            "feature_extractors": ["bbox_ratio", "color_stats", "texture", "contour_features"],
            "inference_time": 2.5
        },
        "camera_config.json": {
            "width": 1280,
            "height": 960,
            "fps": 15,
            "fov": [60.0, 45.0],
            "average_tomato_size_mm": [70.0, 65.0],
            "devices_to_try": [0, 2, 19, 20, 21, 22, 1, 3, 4, 5, 6, 7, 8, 9, -1]
        },
        "detection_config.json": {
            "yolo_model": "yolov8n.pt",
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "red_color_ranges": [
                [[0, 100, 100], [10, 255, 255]],
                [[160, 100, 100], [180, 255, 255]]
            ],
            "min_red_area": 500,
            "red_threshold": 0.4,
            "kalman_iou_threshold": 0.45,
            "tracker_max_misses": 15,
            "circularity_threshold": 0.3
        },
        "general_config.json": {
            "memory_check_interval": 30,
            "memory_warning_mb": 800,
            "memory_critical_mb": 1200,
            "cleanup_interval_seconds": 60,
            "max_memory_mb": 1500
        }
    }
    
    for filename, config in configs.items():
        filepath = os.path.join(CONFIG_DIR, filename)
        if not os.path.exists(filepath):
            try:
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Created default config: {filepath}")
            except Exception as e:
                print(f"❌ Failed to create {filename}: {e}")

# Create default configs on startup
create_default_configs()

# ----------------------------
# Configuration Loader
# ----------------------------
def load_model_configurations():
    """Load configuration from model metadata files"""
    config_paths = {
        "classification": os.path.join(CONFIG_DIR, "classification_config.json"),
        "regression": os.path.join(CONFIG_DIR, "regression_config.json"),
        "gan": os.path.join(CONFIG_DIR, "gan_config.json"),
        "camera": os.path.join(CONFIG_DIR, "camera_config.json"),
        "detection": os.path.join(CONFIG_DIR, "detection_config.json"),
        "general": os.path.join(CONFIG_DIR, "general_config.json")
    }
    
    config = {
        "classification": {},
        "regression": {},
        "gan": {},
        "camera": {},
        "detection": {},
        "general": {}
    }
    
    # Try to load each config file
    for config_type, path in config_paths.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config[config_type] = json.load(f)
                logging.info(f"✅ Loaded {config_type} configuration from {path}")
            else:
                logging.warning(f"⚠️ Config file not found: {path}")
        except Exception as e:
            logging.error(f"❌ Error loading {config_type} config: {e}")
    
    return config

# Load configurations at module level
MODEL_CONFIGS = load_model_configurations()

# Add after imports but before class definitions
def save_tomato_data_to_csv(tomato_id, classification_data, fnn_data, bbox_data):
    """Save tomato data to CSV file"""
    csv_file = "tomato_harvest_data.csv"
    file_exists = os.path.isfile(csv_file)
    
    try:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'timestamp', 'tomato_id', 'classification', 'confidence',
                    'weight', 'size', 'pressure', 'force', 'torque', 
                    'grip_force', 'is_occluded', 'bbox_x', 'bbox_y', 
                    'bbox_w', 'bbox_h', 'image_width', 'image_height'
                ])
            
            # Extract values
            timestamp = datetime.now().isoformat()
            classification = classification_data.get('classification', '--')
            confidence = classification_data.get('confidence', 0.0)
            is_occluded = classification_data.get('is_occluded', False)
            
            # FNN data
            weight = fnn_data.get('weight') if isinstance(fnn_data, dict) else None
            size = fnn_data.get('size') if isinstance(fnn_data, dict) else None
            pressure = fnn_data.get('pressure') if isinstance(fnn_data, dict) else None
            force = fnn_data.get('force') if isinstance(fnn_data, dict) else None
            torque = fnn_data.get('torque') if isinstance(fnn_data, dict) else None
            grip_force = fnn_data.get('grip_force') if isinstance(fnn_data, dict) else None
            
            # Bounding box
            bbox_x = bbox_data[0] if bbox_data and len(bbox_data) > 0 else 0
            bbox_y = bbox_data[1] if bbox_data and len(bbox_data) > 1 else 0
            bbox_w = bbox_data[2] if bbox_data and len(bbox_data) > 2 else 0
            bbox_h = bbox_data[3] if bbox_data and len(bbox_data) > 3 else 0
            
            writer.writerow([
                timestamp,
                tomato_id,
                classification,
                float(confidence),
                float(weight) if weight else 'null',
                float(size) if size else 'null',
                float(pressure) if pressure else 'null',
                float(force) if force else 'null',
                float(torque) if torque else 'null',
                float(grip_force) if grip_force else 'null',
                'Yes' if is_occluded else 'No',
                float(bbox_x),
                float(bbox_y),
                float(bbox_w),
                float(bbox_h),
                640,  # Default image width
                480   # Default image height
            ])
        
        print(f"📊 Saved data for {tomato_id} to {csv_file}")
        return True
        
    except Exception as e:
        print(f"❌ CSV save error: {e}")
        return False

# Fixed SPADE registration for newer TensorFlow versions
try:
    # Try newer Keras 3.0 way
    keras.saving.register_keras_serializable()
    SPADE_REGISTERED = True
except AttributeError:
    try:
        # Try older Keras 2.x way
        tf.keras.utils.register_keras_serializable()
        SPADE_REGISTERED = True
    except:
        SPADE_REGISTERED = False

class SPADE(keras.layers.Layer):
    """SPADE (Spatially-Adaptive Normalization) layer"""
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv2D(
            filters * 2,
            kernel_size=kernel_size,
            padding='same',
            use_bias=True
        )
        
    def call(self, inputs):
        x, mask = inputs
        mean, var = keras.backend.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / keras.backend.sqrt(var + 1e-5)
        params = self.conv(mask)
        gamma, beta = tf.split(params, 2, axis=-1)
        return x_norm * (1 + gamma) + beta
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

# Register SPADE class
try:
    keras.saving.get_custom_objects()['SPADE'] = SPADE
except:
    tf.keras.utils.get_custom_objects()['SPADE'] = SPADE

# ----------------------------
# Numpy JSON Encoder
# ----------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ----------------------------
# Enhanced Error Recovery
# ----------------------------
def setup_crash_handler():
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print(f"\n❌ CRITICAL ERROR: {exc_type.__name__}: {exc_value}")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        try:
            cleanup_resources()
        except:
            pass
        
    sys.excepthook = handle_exception

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tomato-server")

# ----------------------------
# Camera Configuration (Dynamically Loaded)
# ----------------------------
camera_config = MODEL_CONFIGS.get("camera", {})

CAM_WIDTH = camera_config.get("width", 1280)
CAM_HEIGHT = camera_config.get("height", 960)
CAM_FPS = camera_config.get("fps", 15)
CAMERA_FOV = tuple(camera_config.get("fov", (60.0, 45.0)))
AVERAGE_TOMATO_SIZE_MM = tuple(camera_config.get("average_tomato_size_mm", (70.0, 65.0)))

# ----------------------------
# Camera Global Variables
# ----------------------------
webcam = None
webcam_lock = threading.Lock()
capture_thread = None
capture_running = False

latest_frame = None
latest_frame_lock = threading.Lock()
latest_processed = None
processed_lock = threading.Lock()

# ----------------------------
# Camera Functions - SIMPLE AND WORKING
# ----------------------------
def initialize_camera():
    """Initialize camera with OpenCV"""
    global webcam
    print("📷 Initializing camera...")
    
    devices_to_try = camera_config.get("devices_to_try", [0, 1, 2, -1])
    
    for device_id in devices_to_try:
        try:
            print(f"  Trying device {device_id}...")
            cam = cv2.VideoCapture(device_id)
            
            if cam.isOpened():
                # Try to set resolution and FPS
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
                
                # Test capture
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"✅ Camera WORKING on device {device_id}")
                    print(f"   Resolution: {actual_width}x{actual_height}")
                    
                    with webcam_lock:
                        webcam = cam
                    
                    SERVER_HEALTH["camera_available"] = True
                    return True
                else:
                    cam.release()
            else:
                if cam:
                    cam.release()
                    
        except Exception as e:
            print(f"❌ Device {device_id} failed: {e}")
            if 'cam' in locals() and cam:
                cam.release()
            continue
    
    print("❌ No working cameras found")
    SERVER_HEALTH["camera_available"] = False
    return False

def capture_frames():
    """Capture frames from camera continuously"""
    global latest_frame, capture_running
    
    print("🎥 Starting frame capture thread...")
    
    frame_count = 0
    while capture_running:
        try:
            with webcam_lock:
                if webcam is None or not webcam.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = webcam.read()
            
            if ret and frame is not None:
                with latest_frame_lock:
                    latest_frame = frame.copy()
                frame_count += 1
                
                # Show progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"📸 Captured {frame_count} frames...")
            else:
                # If frame grab fails, try to restart camera
                print("⚠️ Failed to grab frame, restarting camera...")
                stop_camera()
                time.sleep(1)
                start_camera()
                break
                
            time.sleep(1.0 / CAM_FPS)
            
        except Exception as e:
            print(f"❌ Capture error: {e}")
            time.sleep(0.1)
    
    print("🎥 Frame capture stopped")

def start_camera():
    """Start camera and capture thread"""
    global capture_running, capture_thread
    
    if capture_running:
        print("📷 Camera is already running")
        return True
    
    if initialize_camera():
        capture_running = True
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        print("✅ Camera started successfully")
        SERVER_HEALTH["camera_available"] = True
        return True
    
    print("❌ Failed to start camera")
    return False

def stop_camera():
    """Stop camera and release resources"""
    global capture_running, webcam
    
    print("🛑 Stopping camera...")
    capture_running = False
    
    # Wait for capture thread to stop
    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=2.0)
    
    # Release camera resources
    with webcam_lock:
        if webcam is not None:
            webcam.release()
            webcam = None
    
    print("✅ Camera stopped")
    SERVER_HEALTH["camera_available"] = False

def get_latest_frame():
    """Get the latest captured frame"""
    with latest_frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

def generate_frames():
    """Generate JPEG frames for video streaming"""
    while capture_running:
        frame = get_latest_frame()
        if frame is not None:
            # Resize for streaming
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Send blank frame if no camera
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "NO CAMERA", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(1.0 / 10)  # Stream at 10 FPS

def cleanup_resources():
    """Cleanup all resources on shutdown"""
    print("🧹 Cleaning up resources...")
    stop_camera()
    model_manager.unload_all_models()
    gc.collect()
    print("✅ Cleanup complete")

# ----------------------------
# Optional backends
# ----------------------------
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"Ultralytics not available: {e}")

# ----------------------------
# Detection Configuration (Dynamically Loaded)
# ----------------------------
detection_config = MODEL_CONFIGS.get("detection", {})

PARAMS = {
    "yolo_model": detection_config.get("yolo_model", "yolov8n.pt"),
    "yolo_conf": detection_config.get("yolo_conf", 0.25),
    "yolo_iou": detection_config.get("yolo_iou", 0.45),
    "red_color_ranges": detection_config.get("red_color_ranges", [
        ([0, 100, 100], [10, 255, 255]),
        ([160, 100, 100], [180, 255, 255])
    ]),
    "min_red_area": detection_config.get("min_red_area", 500),
    "red_threshold": detection_config.get("red_threshold", 0.4),
    "kalman_iou_threshold": detection_config.get("kalman_iou_threshold", 0.45),
    "tracker_max_misses": detection_config.get("tracker_max_misses", 15),
    "circularity_threshold": detection_config.get("circularity_threshold", 0.3),
}

# ----------------------------
# Model Configuration (Dynamically Loaded)
# ----------------------------
classification_config = MODEL_CONFIGS.get("classification", {})
regression_config = MODEL_CONFIGS.get("regression", {})

MODEL_CONFIG = {
    "classification": {
        "input_size": tuple(classification_config.get("input_size", (224, 224))),
        "classes": classification_config.get("classes", ["Ripe", "Occluded"]),
        "occlusion_threshold": classification_config.get("occlusion_threshold", 0.6),
        "ripe_threshold": classification_config.get("ripe_threshold", 0.7),
        "batch_size": classification_config.get("batch_size", 32),
        "normalization": classification_config.get("normalization", True),
        "color_space": classification_config.get("color_space", "RGB")
    },
    "regression": {
        "input_features": regression_config.get("input_features", 30),
        "output_names": regression_config.get("output_names", ["weight", "size", "pressure", "grip_force", "force", "torque"]),
        "output_scaling": regression_config.get("output_scaling", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "ranges": regression_config.get("ranges", {
            "weight": {"min": 50, "max": 500},
            "size": {"min": 3, "max": 15},
            "pressure": {"min": 5, "max": 100},
            "grip_force": {"min": 1, "max": 15},
            "force": {"min": 0.5, "max": 20},
            "torque": {"min": 0.01, "max": 2}
        }),
        "feature_extractors": regression_config.get("feature_extractors", [
            "bbox_ratio",
            "color_stats", 
            "texture",
            "contour_features"
        ]),
        "inference_time": regression_config.get("inference_time", 2.5)
    }
}

# Log loaded configuration
print(f"📷 Camera config: {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS}FPS")
print(f"🎯 Detection params: conf={PARAMS['yolo_conf']}, iou={PARAMS['yolo_iou']}")
print(f"🧠 Classification input={MODEL_CONFIG['classification']['input_size']}")
print(f"🧮 Regression features={MODEL_CONFIG['regression']['input_features']}")

# ----------------------------
# FNN Continuous Processing
# ----------------------------
fnn_queue = queue.Queue(maxsize=10)
fnn_results = {}
fnn_results_lock = threading.Lock()
fnn_processing = False
fnn_worker_thread = None

# ----------------------------
# Enhanced Memory Monitor (Configurable)
# ----------------------------
class EnhancedMemoryMonitor:
    def __init__(self):
        # Get config values
        general_config = MODEL_CONFIGS.get("general", {})
        
        self.interval = general_config.get("memory_check_interval", 30)
        self.warning_threshold = general_config.get("memory_warning_mb", 800)
        self.critical_threshold = general_config.get("memory_critical_mb", 1200)
        self.cleanup_interval = general_config.get("cleanup_interval_seconds", 60)
        self.max_memory_mb = general_config.get("max_memory_mb", 1500)
        
        self.running = True
        self.last_cleanup = time.time()
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"🧠 Enhanced Memory Monitor started (check every {self.interval}s)")
        logger.info(f"🧠 Memory thresholds: Warning={self.warning_threshold}MB, Critical={self.critical_threshold}MB")
    
    def monitor_loop(self):
        while self.running:
            try:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                
                if mem_mb > self.warning_threshold:
                    gc.collect(generation=2)
                    if TF_AVAILABLE:
                        try:
                            tf.keras.backend.clear_session()
                            logger.info("🧹 Cleared TensorFlow session")
                        except:
                            pass
                
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    gc.collect()
                    self.last_cleanup = time.time()
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(10)
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

# ----------------------------
# Detection and Tracking Variables
# ----------------------------
detection_running = False
detection_thread = None

# ----------------------------
# FNN Background Worker
# ----------------------------
def fnn_background_worker():
    """Background worker for FNN processing - REAL VERSION"""
    global fnn_processing, fnn_results
    
    logger.info("🧠 FNN background worker started")
    SERVER_HEALTH["fnn_worker_running"] = True
    
    while fnn_processing:
        try:
            task = fnn_queue.get(timeout=0.5)
            tomato_id, frame, bbox = task
            
            if frame is None or bbox is None:
                continue
            
            try:
                # REAL FNN regression
                regression_result = model_manager.predict_regression(frame, bbox)
                
                if regression_result:
                    # Store raw FNN predictions
                    with fnn_results_lock:
                        fnn_results[tomato_id] = {
                            "predictions": regression_result,
                            "timestamp": time.time()
                        }
                    
                    # Log actual predictions
                    logger.info(f"✅ FNN processed for {tomato_id}: {regression_result}")
                
            except Exception as e:
                logger.error(f"FNN processing error: {e}")
                with metrics_lock:
                    metrics_store["fnn_processing"]["tasks_failed"] += 1
            
            fnn_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"FNN worker error: {e}")
            time.sleep(1.0)
    
    SERVER_HEALTH["fnn_worker_running"] = False
    logger.info("🧠 FNN background worker stopped")

# ----------------------------
# TomatoModelManager
# ----------------------------
class TomatoModelManager:
    def __init__(self):
        self.models = {}
        self.loaded = False
        
    def load_all_models(self):
        logger.info("🔄 Loading all custom models...")
        
        if not TF_AVAILABLE:
            logger.warning("❌ TensorFlow not available")
            return
            
        try:
            # Load models in sequence
            self.models["classification"] = self._load_classification_model()
            time.sleep(1)
            
            self.models["regression"] = self._load_regression_model()
            time.sleep(1)
            
            self.models["gan_generator"] = self._load_gan_generator()
            time.sleep(1)
            
            # Update health status
            loaded_count = sum(1 for model in self.models.values() if model is not None)
            SERVER_HEALTH["custom_models_loaded"] = loaded_count > 0
            logger.info(f"📊 Custom models loaded: {loaded_count}/3")
            self.loaded = True
            
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.unload_all_models()
                 
    def unload_all_models(self):
        logger.info("🧹 Unloading all models...")
        for name, model in self.models.items():
            if model is not None:
                try:
                    del model
                    logger.info(f"  Unloaded {name}")
                except:
                    pass
        self.models = {}
        self.loaded = False
        
        if TF_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except:
                pass
    
    def _load_classification_model(self):
        try:
            model_path = "/home/oladipo/SolomonDeepLearningModels/tomato_cnn_model.h5"
            logger.info(f"📁 Loading classification model: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return None
                
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("✅ Classification model loaded")
            SERVER_HEALTH["custom_models"]["classification"]["loaded"] = True
            return model
                
        except Exception as e:
            logger.error(f"❌ Classification model failed: {e}")
        return None
    
    def _load_regression_model(self):
        try:
            model_path = "/home/oladipo/SolomonDeepLearningModels/FNN_Regression_Model.h5"
            logger.info(f"📁 Loading regression model: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return None
                
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("✅ Regression model loaded")
            SERVER_HEALTH["custom_models"]["regression"]["loaded"] = True
            return model
                
        except Exception as e:
            logger.error(f"❌ Regression model failed: {e}")
            traceback.print_exc()
        return None
    
    def _load_gan_generator(self):
        try:
            model_path = "/home/oladipo/SolomonDeepLearningModels/final_gan_generator.keras"
            logger.info(f"📁 Loading GAN generator: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return None
            
            # Load with custom objects
            try:
                custom_objects = {'SPADE': SPADE}
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
            except:
                # Try without custom objects
                model = tf.keras.models.load_model(model_path, compile=False)
                
            logger.info("✅ GAN generator loaded")
            SERVER_HEALTH["custom_models"]["gan_generator"]["loaded"] = True
            return model
                
        except Exception as e:
            logger.error(f"❌ GAN generator failed: {e}")
        return None
    
    def classify_tomato(self, tomato_image, bbox=None):
        """Classify tomato using CNN model"""
        if not self.models.get("classification"):
            return self._fallback_classification()
        
        try:
            processed = self._preprocess_for_classification(tomato_image)
            predictions = self.models["classification"].predict(processed, verbose=0)
            probs = predictions[0] if len(predictions.shape) > 1 else predictions
            
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            
            is_occluded = (class_idx == 1 and confidence > MODEL_CONFIG["classification"]["occlusion_threshold"])
            is_ripe = (class_idx == 0 and confidence > MODEL_CONFIG["classification"]["ripe_threshold"])
            
            result = {
                "prediction": "Tomato",
                "classification": "Ripe_Tomato" if class_idx == 0 else "Occluded_Tomato",
                "class_idx": class_idx,
                "confidence": confidence,
                "is_ripe": is_ripe,
                "is_occluded": is_occluded,
                "probabilities": {MODEL_CONFIG["classification"]["classes"][i]: float(p) 
                                 for i, p in enumerate(probs)},
                "source": "custom_cnn"
            }
            
            with metrics_lock:
                metrics_store["classification"]["total"] += 1
                if result.get("is_ripe"):
                    metrics_store["classification"]["ripe"] += 1
                elif result.get("is_occluded"):
                    metrics_store["classification"]["occluded"] += 1
            
            return self._convert_to_json_serializable(result)
                
        except Exception as e:
            logger.error(f"Classification inference failed: {e}")
            return self._fallback_classification()
    
    def predict_regression(self, tomato_image, bbox):
        """REAL FNN prediction - returns actual model outputs"""
        if not self.models.get("regression"):
            return self._fallback_regression()
        
        try:
            features = self._extract_regression_features(tomato_image, bbox)
            predictions = self.models["regression"].predict(features, verbose=0)
            
            # Get the predictions
            values = predictions[0] if len(predictions.shape) > 1 else predictions
            
            # Get configuration
            output_names = MODEL_CONFIG["regression"]["output_names"]
            scaling_factors = MODEL_CONFIG["regression"]["output_scaling"]
            ranges = MODEL_CONFIG["regression"]["ranges"]
            
            result = {}
            for i, (name, scale) in enumerate(zip(output_names, scaling_factors)):
                if i < len(values):
                    # Apply scaling
                    value = float(values[i]) * scale
                    
                    # Apply realistic ranges from config
                    if name in ranges:
                        min_val = ranges[name].get("min", 0)
                        max_val = ranges[name].get("max", 100)
                        value = max(min_val, min(max_val, value))
                    
                    result[name] = round(value, 2)
                else:
                    result[name] = None
            
            result["source"] = "fnn_regression"
            result["time_taken"] = MODEL_CONFIG["regression"]["inference_time"]
            
            return self._convert_to_json_serializable(result)
            
        except Exception as e:
            logger.error(f"Regression inference failed: {e}")
            traceback.print_exc()
            return self._fallback_regression()
    
    def reconstruct_occluded_tomato(self, occluded_image, mask):
        """Use GAN to reconstruct occluded tomato"""
        if not self.models.get("gan_generator"):
            logger.warning("⚠️ GAN not available for reconstruction")
            return occluded_image  # Return original if no GAN
        
        try:
            # Prepare inputs for GAN
            input_img = cv2.resize(occluded_image, (224, 224))
            input_img = input_img.astype(np.float32) / 255.0
            
            if len(mask.shape) == 2:
                input_mask = np.expand_dims(mask, axis=-1)
            else:
                input_mask = mask
            
            input_mask = cv2.resize(input_mask, (224, 224))
            input_mask = input_mask.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_img_batch = np.expand_dims(input_img, axis=0)
            input_mask_batch = np.expand_dims(input_mask, axis=0)
            
            # Run GAN reconstruction
            reconstructed = self.models["gan_generator"].predict(
                [input_img_batch, input_mask_batch], 
                verbose=0
            )[0]
            
            # Convert back to original format
            reconstructed = (reconstructed * 255).astype(np.uint8)
            reconstructed = cv2.resize(reconstructed, (occluded_image.shape[1], occluded_image.shape[0]))
            
            logger.info("✅ GAN reconstruction completed")
            return reconstructed
            
        except Exception as e:
            logger.error(f"GAN reconstruction failed: {e}")
            return occluded_image
    
    def _preprocess_for_classification(self, image):
        input_size = MODEL_CONFIG["classification"]["input_size"]
        color_space = MODEL_CONFIG["classification"]["color_space"]
        
        resized = cv2.resize(image, input_size)
        
        if color_space.upper() == "RGB":
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        elif color_space.upper() == "BGR":
            rgb = resized
        elif color_space.upper() == "HSV":
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        elif color_space.upper() == "LAB":
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        if MODEL_CONFIG["classification"]["normalization"]:
            normalized = rgb.astype(np.float32) / 255.0
        else:
            normalized = rgb.astype(np.float32)
            
        return np.expand_dims(normalized, axis=0)
    
    def _extract_regression_features(self, image, bbox):
        x1, y1, x2, y2 = bbox
        tomato_region = image[y1:y2, x1:x2]
        
        if tomato_region.size == 0:
            # Default features if no region
            return np.zeros((1, MODEL_CONFIG["regression"]["input_features"]))
        
        # Extract visual features
        hsv = cv2.cvtColor(tomato_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(tomato_region, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(tomato_region, cv2.COLOR_BGR2GRAY)
        
        features = [
            # Bounding box features
            (x2 - x1) / image.shape[1],
            (y2 - y1) / image.shape[0],
            ((x2 - x1) * (y2 - y1)) / (image.shape[0] * image.shape[1]),
            (x2 - x1) / max(1, (y2 - y1)),
            
            # Color features
            np.mean(hsv[:,:,0]) / 180.0,
            np.std(hsv[:,:,0]) / 180.0,
            np.mean(hsv[:,:,1]) / 255.0,
            np.mean(hsv[:,:,2]) / 255.0,
            np.mean(lab[:,:,1]) / 255.0,
            np.mean(lab[:,:,2]) / 255.0,
            
            # Texture features
            cv2.Laplacian(gray, cv2.CV_64F).var() / 10000,
        ]
        
        # Pad with zeros to match expected input features
        if len(features) < MODEL_CONFIG["regression"]["input_features"]:
            features += [0.0] * (MODEL_CONFIG["regression"]["input_features"] - len(features))
        
        return np.array([features[:MODEL_CONFIG["regression"]["input_features"]]])
    
    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _fallback_classification(self):
        return {
            "prediction": "ripe",
            "confidence": 0.85,
            "is_ripe": True,
            "is_occluded": False,
            "probabilities": {"ripe": 0.85, "occluded": 0.15},
            "source": "fallback"
        }
    
    def _fallback_regression(self):
        """Realistic fallback values"""
        ranges = MODEL_CONFIG["regression"]["ranges"]
        return {
            "weight": (ranges["weight"]["min"] + ranges["weight"]["max"]) / 2,
            "size": (ranges["size"]["min"] + ranges["size"]["max"]) / 2,
            "pressure": (ranges["pressure"]["min"] + ranges["pressure"]["max"]) / 2,
            "grip_force": (ranges["grip_force"]["min"] + ranges["grip_force"]["max"]) / 2,
            "force": (ranges["force"]["min"] + ranges["force"]["max"]) / 2,
            "torque": (ranges["torque"]["min"] + ranges["torque"]["max"]) / 2,
            "time_taken": MODEL_CONFIG["regression"]["inference_time"],
            "source": "fallback"
        }

# Global Model Manager Instance
model_manager = TomatoModelManager()

# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------
# Server Health State
# ----------------------------
SERVER_START_TIME = time.time()
SERVER_HEALTH = {
    "status": "starting",
    "start_time": SERVER_START_TIME,
    "models_loaded": False,
    "camera_available": False,
    "config_loaded": len(MODEL_CONFIGS) > 0,
    "config_files": {name: bool(config) for name, config in MODEL_CONFIGS.items()},
    "custom_models": {
        "classification": {"loaded": False, "purpose": "Tomato ripeness classification"},
        "regression": {"loaded": False, "purpose": "Physical property prediction"},
        "gan_generator": {"loaded": False, "purpose": "Occlusion reconstruction"}
    },
    "fnn_worker_running": False
}

# ----------------------------
# RED OBJECT DETECTION
# ----------------------------
def detect_red_objects(frame):
    """Detect red objects in frame using HSV color space"""
    detections = []
    
    try:
        if frame is None:
            return detections
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = None
        for (lower, upper) in PARAMS["red_color_ranges"]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            if red_mask is None:
                red_mask = mask
            else:
                red_mask = cv2.bitwise_or(red_mask, mask)
        
        if red_mask is None:
            return detections
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < PARAMS["min_red_area"]:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x); y = max(0, y)
            x2 = min(frame.shape[1], x + w); y2 = min(frame.shape[0], y + h)
            w = x2 - x; h = y2 - y
            
            if w <= 0 or h <= 0:
                continue
                
            bbox = [x, y, x + w, y + h]
            
            # Calculate red percentage
            roi_mask = red_mask[y:y+h, x:x+w]
            red_pixels = np.sum(roi_mask > 0)
            total_pixels = w * h
            red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0
            
            if red_percentage < PARAMS["red_threshold"]:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            if circularity < PARAMS["circularity_threshold"]:
                continue
            
            detection = {
                'bbox': bbox,
                'conf': min(red_percentage * circularity, 0.95),
                'class_id': -1,
                'class_name': 'red_object',
                'red_percentage': red_percentage,
                'circularity': circularity,
                'area': area
            }
            
            detections.append(detection)
        
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
    except Exception as e:
        logger.error(f"Red object detection error: {e}")
    
    return detections

# ----------------------------
# Kalman Tracker for Tomato
# ----------------------------
class TomatoTracker:
    def __init__(self):
        self.tracker = None
        self.current_tomato = None
        self.tomato_id = None
        self.misses = 0
        self.max_misses = 15
        self.last_detection_time = time.time()
        self.metrics_lock = threading.Lock()
        self.metrics = {"total": 0, "ripe": 0, "occluded": 0}
    
    def update(self, detections, frame_shape):
        """Update tracker with new detections"""
        if not detections:
            self.misses += 1
            if self.misses > self.max_misses:
                self.tracker = None
                self.current_tomato = None
                self.tomato_id = None
            return self.current_tomato
        
        best_detection = detections[0]
        
        if self.tracker is None:
            # Initialize new tracker
            bbox = best_detection['bbox']
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(latest_frame, tuple(bbox))
            self.current_tomato = best_detection
            self.tomato_id = f"tomato_{int(time.time())}"
            self.misses = 0
            self.last_detection_time = time.time()
            return self.current_tomato
        
        # Update existing tracker
        success, bbox = self.tracker.update(latest_frame)
        if success:
            self.misses = 0
            self.current_tomato = {
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])],
                'conf': best_detection['conf'],
                'class_name': 'red_object'
            }
            self.last_detection_time = time.time()
        else:
            self.misses += 1
            if self.misses > self.max_misses:
                self.tracker = None
                self.current_tomato = None
                self.tomato_id = None
        
        return self.current_tomato
    
    def get_current_tomato(self):
        return self.current_tomato, self.tomato_id

# Global tracker instance
tomato_tracker = TomatoTracker()

# ----------------------------
# Detection and Processing Loop
# ----------------------------
detection_running = False
detection_thread = None

def detection_loop():
    """Main detection and processing loop"""
    global detection_running, latest_processed
    
    print("🔍 Starting detection loop...")
    
    while detection_running:
        try:
            frame = get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect red objects
            red_detections = detect_red_objects(frame)
            
            # Update tracker
            tracked_tomato = tomato_tracker.update(red_detections, frame.shape)
            
            output_frame = frame.copy()
            classification_text = "No Ripe Tomato in View"
            bbox_color = (0, 0, 255)  # Red for no tomato
            
            if tracked_tomato:
                bbox = tracked_tomato['bbox']
                x1, y1, x2, y2 = bbox
                
                # Extract tomato region
                tomato_region = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if tomato_region is not None and tomato_region.size > 0:
                    # Classify using CNN model
                    classification = model_manager.classify_tomato(tomato_region, bbox)
                    
                    if classification.get('is_ripe'):
                        classification_text = f"Ripe Tomato {classification['confidence']:.2f}"
                        bbox_color = (0, 255, 0)  # Green for ripe
                        
                        # Queue FNN processing
                        tomato_id = tomato_tracker.tomato_id
                        if tomato_id and classification.get('is_ripe'):
                            try:
                                fnn_queue.put((tomato_id, tomato_region.copy(), bbox), block=False)
                                print(f"📥 Queued FNN processing for {tomato_id}")
                            except queue.Full:
                                pass
                        
                        # Check for occlusion
                        if classification.get('is_occluded'):
                            classification_text = f"Occluded Tomato {classification['confidence']:.2f}"
                            bbox_color = (0, 165, 255)  # Orange for occluded
                            
                            # Create occlusion mask (simple version - use red mask)
                            hsv = cv2.cvtColor(tomato_region, cv2.COLOR_BGR2HSV)
                            lower_red = np.array([0, 100, 100])
                            upper_red = np.array([10, 255, 255])
                            mask1 = cv2.inRange(hsv, lower_red, upper_red)
                            
                            lower_red = np.array([160, 100, 100])
                            upper_red = np.array([180, 255, 255])
                            mask2 = cv2.inRange(hsv, lower_red, upper_red)
                            
                            mask = cv2.bitwise_or(mask1, mask2)
                            
                            # Use GAN for reconstruction
                            reconstructed = model_manager.reconstruct_occluded_tomato(tomato_region, mask)
                            
                            # You can use reconstructed image for FNN if needed
                    
                    elif classification.get('is_occluded'):
                        classification_text = f"Occluded Tomato {classification['confidence']:.2f}"
                        bbox_color = (0, 165, 255)  # Orange for occluded
                    else:
                        classification_text = f"Not Ripe Tomato {classification['confidence']:.2f}"
                        bbox_color = (0, 255, 255)  # Yellow for not ripe
                    
                    # Draw bounding box
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), bbox_color, 3)
                    
                    # Draw classification text
                    text_y = max(0, y1 - 10)
                    cv2.putText(output_frame, classification_text, (x1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
            
            else:
                # No tomato detected
                cv2.putText(output_frame, "No Ripe Tomato in View", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show FPS
            cv2.putText(output_frame, f"FPS: {CAM_FPS}", (10, output_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            with processed_lock:
                latest_processed = output_frame.copy()
            
            time.sleep(1.0 / CAM_FPS)
            
        except Exception as e:
            print(f"❌ Detection loop error: {e}")
            time.sleep(0.1)
    
    print("🔍 Detection loop stopped")

def start_detection():
    """Start detection loop"""
    global detection_running, detection_thread
    
    if detection_running:
        print("🔍 Detection is already running")
        return True
    
    detection_running = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    print("✅ Detection started successfully")
    return True

def stop_detection():
    """Stop detection loop"""
    global detection_running
    
    detection_running = False
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)
    
    print("✅ Detection stopped")

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return jsonify({
        "message": "Tomato Detection Server",
        "status": "running",
        "camera_available": SERVER_HEALTH["camera_available"],
        "models_loaded": model_manager.loaded,
        "endpoints": {
            "/": "Server status",
            "/health": "Health check",
            "/video_feed": "Live camera feed",
            "/camera_status": "Camera status",
            "/start_camera": "Start camera",
            "/stop_camera": "Stop camera",
            "/start_detection": "Start detection",
            "/stop_detection": "Stop detection",
            "/predict": "Run detection & classification",
            "/predict_fnn": "Dedicated FNN prediction",
            "/fnn_latest": "Get latest FNN results"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "uptime": time.time() - SERVER_START_TIME,
        "models_loaded": model_manager.loaded,
        "camera_running": capture_running,
        "detection_running": detection_running,
        "memory_usage": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            with processed_lock:
                frame = latest_processed.copy() if latest_processed is not None else None
            
            if frame is None:
                # Create placeholder frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "LOADING...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1.0 / 10)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    return jsonify({
        "camera_available": SERVER_HEALTH["camera_available"],
        "camera_running": capture_running,
        "detection_running": detection_running,
        "resolution": f"{CAM_WIDTH}x{CAM_HEIGHT}",
        "fps": CAM_FPS
    })

@app.route('/start_camera', methods=['POST'])
def start_camera_endpoint():
    """Start the camera"""
    if capture_running:
        return jsonify({"status": "already_running", "message": "Camera is already running"})
    
    success = start_camera()
    if success:
        return jsonify({"status": "success", "message": "Camera started successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to start camera"}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera_endpoint():
    """Stop the camera"""
    stop_camera()
    stop_detection()
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/start_detection', methods=['POST'])
def start_detection_endpoint():
    """Start detection"""
    if not capture_running:
        return jsonify({"status": "error", "message": "Camera not running"}), 400
    
    success = start_detection()
    if success:
        return jsonify({"status": "success", "message": "Detection started"})
    else:
        return jsonify({"status": "error", "message": "Failed to start detection"}), 500

@app.route('/stop_detection', methods=['POST'])
def stop_detection_endpoint():
    """Stop detection"""
    stop_detection()
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/predict', methods=['POST'])
def predict():
    """Manual prediction endpoint"""
    try:
        frame = get_latest_frame()
        if frame is None:
            return jsonify({
                "success": False,
                "message": "No frame available"
            }), 400
        
        # Detect red objects
        red_detections = detect_red_objects(frame)
        
        if not red_detections:
            return jsonify({
                "success": True,
                "message": "No red objects detected",
                "detections": []
            })
        
        # Get the best detection
        best_detection = red_detections[0]
        bbox = best_detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract tomato region
        tomato_region = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
        
        if tomato_region is None or tomato_region.size == 0:
            return jsonify({
                "success": False,
                "message": "Invalid tomato region"
            }), 400
        
        # Classify using CNN model
        classification = model_manager.classify_tomato(tomato_region, bbox)
        
        # Run FNN if ripe
        fnn_result = None
        if classification.get('is_ripe'):
            fnn_result = model_manager.predict_regression(tomato_region, bbox)
        
        response = {
            "success": True,
            "message": "Prediction completed",
            "detection": {
                "bbox": bbox,
                "confidence": float(best_detection['conf']),
                "class_name": best_detection['class_name']
            },
            "classification": classification,
            "regression": fnn_result if fnn_result else "Not ripe, no FNN prediction"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/predict_fnn', methods=['POST'])
def predict_fnn():
    """Dedicated FNN prediction"""
    try:
        data = request.get_json()
        if not data or 'bbox' not in data:
            return jsonify({
                "success": False,
                "message": "Bounding box required"
            }), 400
        
        bbox = data['bbox']
        if len(bbox) != 4:
            return jsonify({
                "success": False,
                "message": "Invalid bounding box format"
            }), 400
        
        frame = get_latest_frame()
        if frame is None:
            return jsonify({
                "success": False,
                "message": "No frame available"
            }), 400
        
        x1, y1, x2, y2 = map(int, bbox)
        tomato_region = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
        
        if tomato_region is None or tomato_region.size == 0:
            return jsonify({
                "success": False,
                "message": "Invalid tomato region"
            }), 400
        
        # Run FNN prediction
        fnn_result = model_manager.predict_regression(tomato_region, bbox)
        
        return jsonify({
            "success": True,
            "message": "FNN prediction completed",
            "predictions": fnn_result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"FNN prediction failed: {str(e)}"
        }), 500

@app.route('/fnn_latest', methods=['GET'])
def fnn_latest():
    """Get latest FNN results"""
    with fnn_results_lock:
        if fnn_results:
            latest_id = max(fnn_results.keys(), key=lambda k: fnn_results[k]["timestamp"])
            result = fnn_results[latest_id]
            return jsonify({
                "success": True,
                "tomato_id": latest_id,
                "predictions": result["predictions"],
                "timestamp": result["timestamp"]
            })
        else:
            return jsonify({
                "success": False,
                "message": "No FNN results available"
            })

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("\n" + "="*70)
    print("🍅 TOMATO DETECTION SERVER (Real FNN Integration)")
    print("="*70)
    print()
    
    # Load models
    print("🔄 Loading models...")
    model_manager.load_all_models()
    
    # Start FNN worker
    global fnn_processing, fnn_worker_thread
    fnn_processing = True
    fnn_worker_thread = threading.Thread(target=fnn_background_worker, daemon=True)
    fnn_worker_thread.start()
    print("🧠 FNN worker started")
    
    # Start camera
    print("\n📷 Starting camera...")
    start_camera()
    
    # Start detection after a short delay
    time.sleep(2)
    print("\n🔍 Starting detection...")
    start_detection()
    
    print("\n✅ SERVER IS NOW AVAILABLE at http://0.0.0.0:5000")
    print()
    print("📋 Available Endpoints:")
    print("  /               - Server status")
    print("  /health         - Health check")
    print("  /video_feed     - Live camera feed")
    print("  /camera_status  - Camera status")
    print("  /start_camera   - Start camera")
    print("  /stop_camera    - Stop camera")
    print("  /start_detection - Start detection")
    print("  /stop_detection  - Stop detection")
    print("  /predict        - Manual prediction")
    print("  /predict_fnn    - FNN prediction")
    print("  /fnn_latest     - Latest FNN results")
    print()
    print("🔗 View camera feed: http://localhost:5000/video_feed")
    print()
    print("-" * 70)
    print("📷 Camera Features:")
    print(f"  • Resolution: {CAM_WIDTH}x{CAM_HEIGHT}")
    print(f"  • FPS: {CAM_FPS}")
    print(f"  • Status: {'Running' if capture_running else 'Stopped'}")
    print()
    print("🔍 Detection Logic:")
    print("  1. Detects red objects")
    print("  2. If no red objects: 'No Ripe Tomato in View'")
    print("  3. If red object: Track and classify with CNN")
    print("  4. If ripe: Call FNN for regression predictions")
    print("  5. If occluded: Use GAN for reconstruction")
    print("-" * 70)
    print("🛑 Press Ctrl+C to stop the server")
    print("="*70)
    
    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received")
        cleanup_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    main()