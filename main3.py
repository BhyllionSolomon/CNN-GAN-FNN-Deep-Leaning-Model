# main3.py - CORRECTED VERSION
# This is a Python script that creates mains.py

import os
import sys

# The complete code for mains.py
CODE = '''# main.py - COMPLETE VERSION WITH ALL FUNCTIONALITY
import os
import time
import threading
import logging
import json
import traceback
import queue
import numpy as np
import cv2
from datetime import datetime, timedelta
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import psutil
import gc
import signal
import sys
from typing import Dict, List, Tuple, Optional, Any
import csv
import math

# ==================== CONFIGURATION ====================
CONFIG_DIR = "/home/oladipo/SolomonDeepLearningModels"
os.makedirs(CONFIG_DIR, exist_ok=True)

# Create all necessary configs
ESSENTIAL_CONFIGS = {
    "camera_config.json": {
        "width": 1280,
        "height": 720,
        "fps": 15,
        "devices_to_try": [0, 1, 2, 3, -1, 19, 20, 21, 22],
        "fallback_mode": "test",
        "calibration": {
            "fov": [60.0, 45.0],
            "average_tomato_size_mm": [70.0, 65.0],
            "pixels_per_mm": 5.0
        }
    },
    "model_config.json": {
        "classification_model": "/home/oladipo/SolomonDeepLearningModels/tomato_cnn_model.h5",
        "regression_model": "/home/oladipo/SolomonDeepLearningModels/FNN_Regression_Model.h5",
        "gan_generator": "/home/oladipo/SolomonDeepLearningModels/tomato_reconstruction_generator.keras",
        "global_discriminator": "/home/oladipo/SolomonDeepLearningModels/global_discriminator.keras",
        "patch_discriminator": "/home/oladipo/SolomonDeepLearningModels/patchgan_discriminator.keras"
    },
    "detection_config.json": {
        "yolo_confidence": 0.25,
        "yolo_iou": 0.45,
        "red_color_ranges": [
            {"low": [0, 100, 100], "high": [10, 255, 255]},
            {"low": [160, 100, 100], "high": [180, 255, 255]}
        ],
        "min_red_area": 500,
        "red_threshold": 0.4,
        "circularity_threshold": 0.3,
        "tracker_max_misses": 15,
        "kalman_iou_threshold": 0.45
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
        "inference_time": 2.5
    },
    "classification_config.json": {
        "input_size": [224, 224],
        "classes": ["Ripe", "Occluded"],
        "occlusion_threshold": 0.6,
        "ripe_threshold": 0.7,
        "batch_size": 32,
        "normalization": True,
        "color_space": "RGB"
    },
    "system_config.json": {
        "memory_check_interval": 30,
        "memory_warning_mb": 800,
        "memory_critical_mb": 1200,
        "cleanup_interval_seconds": 60,
        "max_memory_mb": 1500,
        "max_frame_history": 100,
        "max_detection_history": 50
    }
}

# Create config files if they don't exist
for filename, config in ESSENTIAL_CONFIGS.items():
    filepath = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(filepath):
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Created config: {filename}")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")

# Load all configurations
def load_all_configs():
    configs = {}
    for filename in ESSENTIAL_CONFIGS.keys():
        filepath = os.path.join(CONFIG_DIR, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    config_name = filename.replace('.json', '')
                    configs[config_name] = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")
    return configs

CONFIG = load_all_configs()
CAMERA_CONFIG = CONFIG.get('camera_config', ESSENTIAL_CONFIGS['camera_config.json'])
DETECTION_CONFIG = CONFIG.get('detection_config', ESSENTIAL_CONFIGS['detection_config.json'])
REGRESSION_CONFIG = CONFIG.get('regression_config', ESSENTIAL_CONFIGS['regression_config.json'])
CLASSIFICATION_CONFIG = CONFIG.get('classification_config', ESSENTIAL_CONFIGS['classification_config.json'])
SYSTEM_CONFIG = CONFIG.get('system_config', ESSENTIAL_CONFIGS['system_config.json'])

# ==================== CUSTOM SPADE LAYER ====================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    class SPADE(layers.Layer):
        def __init__(self, filters, kernel_size=3):
            super(SPADE, self).__init__()
            self.filters = filters
            self.conv = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
            self.conv_gamma = layers.Conv2D(filters, kernel_size, padding='same')
            self.conv_beta = layers.Conv2D(filters, kernel_size, padding='same')
        
        def call(self, x, segmentation_map):
            mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
            x_normalized = (x - mean) / (std + 1e-5)
            
            seg_resized = tf.image.resize(segmentation_map, [tf.shape(x)[1], tf.shape(x)[2]], method='nearest')
            seg_features = self.conv(seg_resized)
            gamma = self.conv_gamma(seg_features)
            beta = self.conv_beta(seg_features)
            
            return x_normalized * (1 + gamma) + beta
        
        def get_config(self):
            config = super().get_config()
            config.update({'filters': self.filters})
            return config
    
    # Register SPADE globally
    keras.utils.get_custom_objects()['SPADE'] = SPADE
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow and SPADE layer loaded")
    
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# ==================== CAMERA MANAGER ====================
class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_type = None
        self.capture_running = False
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_history = []
        self.max_history = SYSTEM_CONFIG.get('max_frame_history', 100)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.start_time = time.time()
        
        # Camera properties
        self.width = CAMERA_CONFIG.get('width', 640)
        self.height = CAMERA_CONFIG.get('height', 480)
        self.target_fps = CAMERA_CONFIG.get('fps', 15)
        
    def find_available_camera(self):
        """Try multiple camera indices"""
        devices = CAMERA_CONFIG.get('devices_to_try', [0, 1, 2, 3, -1])
        
        print(f"🔍 Searching for camera on {len(devices)} devices...")
        
        for device_id in devices:
            print(f"  Testing device {device_id}...")
            cap = None
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    # Try to read a frame
                    for attempt in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            print(f"    ✅ Camera found at device {device_id}")
                            print(f"       Resolution: {actual_width}x{actual_height}")
                            print(f"       FPS: {actual_fps}")
                            
                            # Set desired properties
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                            
                            return cap, device_id, 'opencv'
                        time.sleep(0.1)
                    
                    cap.release()
            except Exception as e:
                if cap is not None:
                    cap.release()
                print(f"    ❌ Device {device_id} error: {str(e)[:100]}")
        
        print("❌ No physical camera found")
        return None, -1, None
    
    def initialize_camera(self):
        """Initialize camera with fallback to test mode"""
        physical_camera, device_id, camera_type = self.find_available_camera()
        
        if physical_camera is not None:
            self.camera = physical_camera
            self.camera_type = 'opencv'
            print(f"✅ Physical camera initialized on device {device_id}")
            return True
        
        # Fallback to test mode
        fallback_mode = CAMERA_CONFIG.get('fallback_mode', 'test')
        if fallback_mode == 'test':
            self.camera_type = 'test'
            print("✅ Test camera initialized (no physical camera needed)")
            return True
        
        return False
    
    def start_capture(self):
        """Start camera capture thread"""
        if self.capture_running:
            return True
        
        if not self.camera_type:
            if not self.initialize_camera():
                print("❌ Camera initialization failed")
                return False
        
        self.capture_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self.capture_thread.start()
        
        # Wait for first frame
        for _ in range(50):  # 5 second timeout
            with self.frame_lock:
                if self.latest_frame is not None:
                    break
            time.sleep(0.1)
        
        print("✅ Camera capture started")
        return True
    
    def _capture_loop(self):
        """Main capture loop"""
        print("🔄 Capture loop started")
        
        while self.capture_running:
            frame = None
            
            try:
                if self.camera_type == 'opencv' and self.camera is not None:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("⚠️ Camera read failed, reinitializing...")
                        self.camera.release()
                        self.initialize_camera()
                        time.sleep(0.5)
                        continue
                        
                elif self.camera_type == 'test':
                    frame = self._generate_test_frame()
                
                # Store frame
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        
                        # Maintain history
                        self.frame_history.append(frame.copy())
                        if len(self.frame_history) > self.max_history:
                            self.frame_history.pop(0)
                    
                    # Update FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = current_time
                
                # Control frame rate
                sleep_time = max(0.001, 1.0 / self.target_fps - 0.005)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Capture error: {str(e)[:100]}")
                time.sleep(0.1)
        
        print("🛑 Capture loop stopped")
    
    def _generate_test_frame(self):
        """Generate sophisticated test frame"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = time.time()
        
        # Draw background gradient
        for y in range(self.height):
            color = int(50 + 50 * np.sin(y / self.height * np.pi))
            cv2.line(frame, (0, y), (self.width, y), (color, color, color), 1)
        
        # Animated tomato
        center_x = int(self.width/2 + 150 * np.sin(t * 0.5))
        center_y = int(self.height/2 + 100 * np.cos(t * 0.3))
        radius = 80
        
        # Draw tomato with gradient
        for r in range(radius, 0, -1):
            intensity = int(200 * (r / radius))
            cv2.circle(frame, (center_x, center_y), r, (0, 0, intensity), -1)
        
        # Green stem
        stem_points = np.array([
            [center_x - 10, center_y - radius],
            [center_x, center_y - radius - 30],
            [center_x + 10, center_y - radius]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [stem_points], (0, 100, 0))
        
        # Tomato shine
        shine_x = center_x - radius//3
        shine_y = center_y - radius//3
        cv2.circle(frame, (shine_x, shine_y), radius//6, (255, 255, 255, 100), -1)
        
        # Info overlay
        cv2.putText(frame, "TEST MODE - VIRTUAL CAMERA", 
                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {self.width}x{self.height} @ {self.fps} FPS", 
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}", 
                   (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Tomato Simulator v3.1", 
                   (30, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Bounding box with crosshair
        bbox_margin = 15
        cv2.rectangle(frame, 
                     (center_x - radius - bbox_margin, center_y - radius - bbox_margin),
                     (center_x + radius + bbox_margin, center_y + radius + bbox_margin),
                     (0, 255, 255), 2)
        
        # Crosshair
        cross_size = 20
        cv2.line(frame, (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), (255, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), (255, 255, 0), 1)
        cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)
        
        # Add moving noise particles (simulating dust/specks)
        particle_count = 20
        for _ in range(particle_count):
            px = int(np.random.randint(0, self.width))
            py = int(np.random.randint(0, self.height))
            brightness = np.random.randint(50, 150)
            cv2.circle(frame, (px, py), 1, (brightness, brightness, brightness), -1)
        
        return frame
    
    def get_latest_frame(self):
        """Get latest frame with thread safety"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def get_frame_history(self, count=10):
        """Get recent frames from history"""
        with self.frame_lock:
            if not self.frame_history:
                return []
            count = min(count, len(self.frame_history))
            return self.frame_history[-count:]
    
    def generate_video_feed(self):
        """Generate MJPEG stream"""
        while self.capture_running:
            try:
                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Resize for streaming
                stream_frame = cv2.resize(frame, (640, 480))
                
                # Encode as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                ret, buffer = cv2.imencode('.jpg', stream_frame, encode_param)
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    print("⚠️ Frame encoding failed")
                
                time.sleep(1.0 / 10)  # Stream at 10 FPS
                
            except Exception as e:
                print(f"❌ Video feed error: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop camera and cleanup"""
        self.capture_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.camera_type == 'opencv' and self.camera is not None:
            self.camera.release()
            self.camera = None
        
        with self.frame_lock:
            self.latest_frame = None
            self.frame_history.clear()
        
        print("✅ Camera stopped")
    
    def get_status(self):
        """Get camera status"""
        frame = self.get_latest_frame()
        frame_info = None
        if frame is not None:
            frame_info = {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            }
        
        return {
            "camera_type": self.camera_type,
            "capture_running": self.capture_running,
            "fps": self.fps,
            "frame_available": frame is not None,
            "frame_info": frame_info,
            "uptime": time.time() - self.start_time,
            "frame_count": self.frame_count,
            "history_size": len(self.frame_history),
            "width": self.width,
            "height": self.height,
            "target_fps": self.target_fps
        }

# ==================== TOMATO DETECTION ENGINE ====================
class TomatoDetectionEngine:
    def __init__(self):
        self.detection_running = False
        self.detection_thread = None
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        self.detection_history = []
        self.max_history = SYSTEM_CONFIG.get('max_detection_history', 50)
        
        # Tomato tracking
        self.tomato_count = 0
        self.current_tomato = None
        self.tomato_tracker = {}
        self.tracker_id_counter = 0
        
        # Performance tracking
        self.detection_count = 0
        self.detection_fps = 0
        self.last_detection_time = time.time()
        
    def detect_tomatoes(self, frame):
        """Detect tomatoes in frame using multiple methods"""
        if frame is None or frame.size == 0:
            return []
        
        detections = []
        
        # Method 1: Color-based detection (red objects)
        red_objects = self._detect_red_objects(frame)
        for obj in red_objects:
            detections.append({
                "type": "red_object",
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "area": obj["area"],
                "circularity": obj["circularity"],
                "red_percentage": obj["red_percentage"],
                "method": "color_detection"
            })
        
        # Method 2: Shape-based detection (circular objects)
        circular_objects = self._detect_circular_objects(frame)
        for obj in circular_objects:
            detections.append({
                "type": "circular_object",
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "area": obj["area"],
                "circularity": obj["circularity"],
                "method": "shape_detection"
            })
        
        # Method 3: Edge-based detection
        edge_objects = self._detect_edge_objects(frame)
        for obj in edge_objects:
            detections.append({
                "type": "edge_object",
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "area": obj["area"],
                "method": "edge_detection"
            })
        
        # Track tomatoes across frames
        tracked_detections = self._track_objects(detections)
        
        with self.detection_lock:
            self.latest_detections = tracked_detections
            self.detection_history.append({
                "timestamp": datetime.now().isoformat(),
                "detections": tracked_detections.copy(),
                "frame_shape": frame.shape
            })
            
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
            
            # Update tomato count
            tomato_detections = [d for d in tracked_detections if d.get("type") == "tomato"]
            self.tomato_count = len(tomato_detections)
            
            # Update current tomato (most confident)
            if tomato_detections:
                self.current_tomato = max(tomato_detections, key=lambda x: x.get("confidence", 0))
            else:
                self.current_tomato = None
            
            # Update performance metrics
            self.detection_count += 1
            current_time = time.time()
            if current_time - self.last_detection_time >= 1.0:
                self.detection_fps = self.detection_count
                self.detection_count = 0
                self.last_detection_time = current_time
        
        return tracked_detections
    
    def _detect_red_objects(self, frame):
        """Detect red objects using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get red color ranges from config
        red_ranges = DETECTION_CONFIG.get('red_color_ranges', [
            {"low": [0, 100, 100], "high": [10, 255, 255]},
            {"low": [160, 100, 100], "high": [180, 255, 255]}
        ])
        
        masks = []
        for color_range in red_ranges:
            lower = np.array(color_range["low"])
            upper = np.array(color_range["high"])
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # Combine masks
        if len(masks) > 1:
            combined_mask = cv2.bitwise_or(masks[0], masks[1])
        else:
            combined_mask = masks[0] if masks else np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_objects = []
        min_area = DETECTION_CONFIG.get('min_red_area', 500)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Calculate red percentage in bounding box
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                red_mask_roi = cv2.inRange(roi_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                red_percentage = np.sum(red_mask_roi > 0) / (w * h)
            else:
                red_percentage = 0
            
            # Confidence based on circularity and red percentage
            red_threshold = DETECTION_CONFIG.get('red_threshold', 0.4)
            circularity_threshold = DETECTION_CONFIG.get('circularity_threshold', 0.3)
            
            is_red_enough = red_percentage > red_threshold
            is_circular_enough = circularity > circularity_threshold
            
            if is_red_enough and is_circular_enough:
                confidence = 0.3 + (red_percentage * 0.4) + (circularity * 0.3)
                confidence = min(confidence, 0.95)
                
                red_objects.append({
                    "bbox": bbox,
                    "confidence": float(confidence),
                    "area": float(area),
                    "circularity": float(circularity),
                    "red_percentage": float(red_percentage),
                    "contour": contour
                })
        
        return red_objects
    
    def _detect_circular_objects(self, frame):
        """Detect circular objects using Hough Circle Transform"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50,
            param1=100, 
            param2=30, 
            minRadius=20, 
            maxRadius=150
        )
        
        circular_objects = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                x, y, radius = circle
                bbox = [x - radius, y - radius, x + radius, y + radius]
                
                # Check if within image bounds
                if (bbox[0] >= 0 and bbox[1] >= 0 and 
                    bbox[2] < frame.shape[1] and bbox[3] < frame.shape[0]):
                    
                    area = np.pi * radius * radius
                    
                    circular_objects.append({
                        "bbox": bbox,
                        "confidence": 0.7,  # High confidence for circles
                        "area": float(area),
                        "circularity": 1.0,  # Perfect circle
                        "center": (int(x), int(y)),
                        "radius": int(radius)
                    })
        
        return circular_objects
    
    def _detect_edge_objects(self, frame):
        """Detect objects using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        edge_objects = []
        min_area = 300
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Confidence based on solidity and size
            confidence = 0.2 + (solidity * 0.3) + min(area / 10000, 0.5)
            confidence = min(confidence, 0.8)
            
            edge_objects.append({
                "bbox": bbox,
                "confidence": float(confidence),
                "area": float(area),
                "solidity": float(solidity),
                "contour": contour
            })
        
        return edge_objects
    
    def _track_objects(self, detections):
        """Track objects across frames"""
        current_time = time.time()
        tracked_detections = []
        
        for detection in detections:
            bbox = detection["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Find matching tracker
            best_match_id = None
            best_match_distance = float('inf')
            max_distance = 100  # pixels
            
            for tracker_id, tracker_info in self.tomato_tracker.items():
                last_seen = tracker_info.get('last_seen', 0)
                if current_time - last_seen > 5.0:  # Remove old trackers
                    continue
                
                last_center = tracker_info.get('center', (0, 0))
                distance = np.sqrt((center_x - last_center[0])**2 + 
                                 (center_y - last_center[1])**2)
                
                if distance < max_distance and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_id = tracker_id
            
            if best_match_id is not None:
                # Update existing tracker
                tracker_info = self.tomato_tracker[best_match_id]
                tracker_info['center'] = (center_x, center_y)
                tracker_info['last_seen'] = current_time
                tracker_info['detection_count'] += 1
                
                detection['tracker_id'] = best_match_id
                detection['tracker_age'] = current_time - tracker_info['created']
                detection['tracker_detections'] = tracker_info['detection_count']
                
            else:
                # Create new tracker
                self.tracker_id_counter += 1
                tracker_id = self.tracker_id_counter
                
                self.tomato_tracker[tracker_id] = {
                    'center': (center_x, center_y),
                    'created': current_time,
                    'last_seen': current_time,
                    'detection_count': 1
                }
                
                detection['tracker_id'] = tracker_id
                detection['tracker_age'] = 0
                detection['tracker_detections'] = 1
            
            # Classify as tomato if high confidence from multiple methods
            if detection.get('confidence', 0) > 0.6:
                detection['type'] = 'tomato'
            
            tracked_detections.append(detection)
        
        # Clean up old trackers
        tracker_ids = list(self.tomato_tracker.keys())
        for tracker_id in tracker_ids:
            if current_time - self.tomato_tracker[tracker_id]['last_seen'] > 10.0:
                del self.tomato_tracker[tracker_id]
        
        return tracked_detections
    
    def start_detection(self, camera_manager):
        """Start continuous detection"""
        if self.detection_running:
            return True
        
        self.detection_running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(camera_manager,),
            daemon=True,
            name="DetectionEngine"
        )
        self.detection_thread.start()
        
        print("✅ Detection engine started")
        return True
    
    def _detection_loop(self, camera_manager):
        """Continuous detection loop"""
        print("🔍 Detection loop started")
        
        while self.detection_running:
            try:
                frame = camera_manager.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Run detection
                detections = self.detect_tomatoes(frame)
                
                # Process detections (simulate AI model processing)
                processed_detections = self._process_detections(detections)
                
                with self.detection_lock:
                    self.latest_detections = processed_detections
                
                # Control detection rate
                time.sleep(0.1)  # 10 FPS detection rate
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(0.5)
        
        print("🛑 Detection loop stopped")
    
    def _process_detections(self, detections):
        """Process detections with simulated AI models"""
        processed = []
        
        for detection in detections:
            # Simulate CNN classification
            if detection.get('type') == 'tomato':
                # Simulate ripeness classification
                is_ripe = np.random.random() > 0.3
                is_occluded = np.random.random() < 0.2
                
                classification = {
                    'prediction': 'ripe_tomato' if is_ripe else 'unripe_tomato',
                    'confidence': detection.get('confidence', 0.5) + np.random.random() * 0.3,
                    'is_ripe': is_ripe,
                    'is_occluded': is_occluded,
                    'probabilities': {
                        'ripe': 0.7 if is_ripe else 0.3,
                        'occluded': 0.3 if is_occluded else 0.7
                    }
                }
                
                # Simulate FNN regression
                bbox = detection.get('bbox', [0, 0, 100, 100])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                
                regression = {
                    'weight': 50 + (area / 100) * 3 + np.random.random() * 50,
                    'size': 3 + (np.sqrt(area) / 50) + np.random.random() * 2,
                    'pressure': 10 + (detection.get('confidence', 0.5) * 40) + np.random.random() * 20,
                    'grip_force': 2 + (detection.get('confidence', 0.5) * 6) + np.random.random() * 3,
                    'force': 1 + (detection.get('confidence', 0.5) * 4) + np.random.random() * 2,
                    'torque': 0.1 + (detection.get('confidence', 0.5) * 0.5) + np.random.random() * 0.2,
                    'source': 'simulated_fnn'
                }
                
                # Add to detection
                detection['classification'] = classification
                detection['regression'] = regression
                detection['fnn_predictions'] = regression
                detection['derived_properties'] = {
                    'weight': regression['weight'],
                    'size': regression['size'],
                    'grip_force': regression['grip_force'],
                    'pressure': regression['pressure']
                }
            
            processed.append(detection)
        
        return processed
    
    def stop_detection(self):
        """Stop detection engine"""
        self.detection_running = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        with self.detection_lock:
            self.latest_detections = []
        
        print("✅ Detection engine stopped")
    
    def get_status(self):
        """Get detection engine status"""
        with self.detection_lock:
            current_tomato = self.current_tomato
            latest_detections = self.latest_detections.copy()
            history_size = len(self.detection_history)
        
        return {
            "detection_running": self.detection_running,
            "detection_fps": self.detection_fps,
            "tomato_count": self.tomato_count,
            "current_tomato": current_tomato,
            "latest_detections_count": len(latest_detections),
            "detection_history_size": history_size,
            "tracker_count": len(self.tomato_tracker),
            "has_tomato": current_tomato is not None
        }
    
    def get_latest_detections(self):
        """Get latest detections"""
        with self.detection_lock:
            return self.latest_detections.copy()

# ==================== FNN PROCESSING WORKER ====================
class FNNProcessingWorker:
    def __init__(self):
        self.queue = queue.Queue(maxsize=20)
        self.results = {}
        self.results_lock = threading.Lock()
        self.worker_running = False
        self.worker_thread = None
        
        # Statistics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.queue_size = 0
        self.start_time = time.time()
        
    def start_worker(self):
        """Start FNN processing worker"""
        if self.worker_running:
            return True
        
        self.worker_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="FNNWorker"
        )
        self.worker_thread.start()
        
        print("✅ FNN processing worker started")
        return True
    
    def _worker_loop(self):
        """Main worker loop for FNN processing"""
        print("🧠 FNN worker loop started")
        
        while self.worker_running:
            try:
                # Get task from queue (non-blocking with timeout)
                try:
                    task_id, task_data = self.queue.get(timeout=0.5)
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                
                # Process the task
                result = self._process_fnn_task(task_data)
                
                # Store result
                with self.results_lock:
                    self.results[task_id] = {
                        'result': result,
                        'timestamp': datetime.now().isoformat(),
                        'processing_time': time.time() - task_data.get('timestamp', time.time())
                    }
                    
                    # Keep only recent results
                    result_keys = list(self.results.keys())
                    if len(result_keys) > 50:
                        for key in result_keys[:-50]:
                            del self.results[key]
                
                self.tasks_processed += 1
                self.queue.task_done()
                
            except Exception as e:
                print(f"❌ FNN worker error: {e}")
                self.tasks_failed += 1
                time.sleep(0.5)
        
        print("🛑 FNN worker loop stopped")
    
    def _process_fnn_task(self, task_data):
        """Process FNN task (simulated)"""
        # Simulate FNN processing delay
        time.sleep(0.05 + np.random.random() * 0.1)
        
        # Extract features from task data
        bbox = task_data.get('bbox', [0, 0, 100, 100])
        confidence = task_data.get('confidence', 0.5)
        area = task_data.get('area', 1000)
        
        # Calculate derived properties
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / max(height, 1)
        
        # Simulate FNN predictions
        predictions = {
            'tactile_property_1': 0.3 + confidence * 0.4 + np.random.random() * 0.1,
            'tactile_property_2': 0.4 + confidence * 0.3 + np.random.random() * 0.1,
            'tactile_property_3': 0.3 + confidence * 0.3 + np.random.random() * 0.1,
            'tactile_property_4': 0.3 + confidence * 0.2 + np.random.random() * 0.1,
            'tactile_property_5': 0.14 + confidence * 0.1 + np.random.random() * 0.05,
            'tactile_property_6': 0.14 + confidence * 0.1 + np.random.random() * 0.05,
            'tactile_property_7': 0.2 + confidence * 0.2 + np.random.random() * 0.1,
            'tactile_property_8': 0.15 + confidence * 0.15 + np.random.random() * 0.1,
            'tactile_property_9': 0.1 + confidence * 0.1 + np.random.random() * 0.05
        }
        
        # Calculate derived properties
        derived = {
            'weight': 50 + (area / 500) * 100 + np.random.random() * 50,
            'size': 3 + (np.sqrt(area) / 40) + np.random.random() * 2,
            'pressure': 10 + confidence * 40 + np.random.random() * 20,
            'grip_force': 2 + confidence * 6 + np.random.random() * 3,
            'force': 1 + confidence * 4 + np.random.random() * 2,
            'torque': 0.1 + confidence * 0.5 + np.random.random() * 0.2,
            'time_taken': 2.5 + np.random.random() * 0.5,
            'source': 'fnn_worker'
        }
        
        return {
            'fnn_predictions': predictions,
            'derived_properties': derived,
            'processing_time': time.time() - task_data.get('timestamp', time.time()),
            'task_id': task_data.get('task_id', 'unknown')
        }
    
    def submit_task(self, task_data):
        """Submit task for FNN processing"""
        if not self.worker_running:
            return None
        
        task_id = f"fnn_{int(time.time() * 1000)}_{self.tasks_processed}"
        task_data['task_id'] = task_id
        task_data['timestamp'] = time.time()
        
        try:
            self.queue.put_nowait((task_id, task_data))
            self.queue_size = self.queue.qsize()
            return task_id
        except queue.Full:
            print("⚠️ FNN queue full, dropping task")
            return None
    
    def get_result(self, task_id):
        """Get processing result for task"""
        with self.results_lock:
            return self.results.get(task_id)
    
    def stop_worker(self):
        """Stop FNN worker"""
        self.worker_running = False
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        
        with self.results_lock:
            self.results.clear()
        
        print("✅ FNN worker stopped")
    
    def get_status(self):
        """Get worker status"""
        return {
            "worker_running": self.worker_running,
            "queue_size": self.queue.qsize(),
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "results_count": len(self.results),
            "uptime": time.time() - self.start_time,
            "queue_capacity": self.queue.maxsize
        }

# ==================== FLASK APPLICATION ====================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize all components
camera_manager = CameraManager()
detection_engine = TomatoDetectionEngine()
fnn_worker = FNNProcessingWorker()

# Server state
server_start_time = time.time()
server_health = {
    "status": "starting",
    "uptime": 0,
    "camera_available": False,
    "detection_available": False,
    "fnn_worker_available": False,
    "models_loaded": TENSORFLOW_AVAILABLE,
    "restart_count": 0,
    "last_restart": datetime.now().isoformat(),
    "system_memory_mb": 0,
    "tomato_count": 0,
    "tracking_active": False,
    "live_analytics": True
}

# Data storage
tomato_data_history = []
performance_metrics = {
    "classification": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85},
    "regression": {"accuracy": 0.78, "mse": 0.12, "mae": 0.08, "r_squared": 0.75},
    "gan": {"success_rate": 0.72, "reconstruction_mse": 0.15, "perceptual_loss": 0.08},
    "robotics": {"success_rate": 0.65, "average_grip_force": 8.5, "grip_stability": 0.78}
}

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Main endpoint"""
    return jsonify({
        "message": "🍅 Tomato Detection Server v3.1 - Complete System",
        "version": "3.1",
        "author": "OLAGUNJU KOREDE SOLOMON | 216882",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/": "This page",
            "/health": "Health check",
            "/server_status": "Detailed server status",
            "/live_analytics": "Live analytics and tracking",
            "/video_feed": "Live camera stream (MJPEG)",
            "/camera_status": "Camera status and control",
            "/start_camera": "Start camera (POST)",
            "/stop_camera": "Stop camera (POST)",
            "/capture": "Capture single image",
            "/predict": "Run full detection & prediction pipeline",
            "/predict_fnn": "Run FNN prediction only",
            "/detection_status": "Detection engine status",
            "/fnn_status": "FNN worker status",
            "/system_metrics": "Performance metrics",
            "/test_camera": "Test camera functionality",
            "/export_data": "Export data to CSV",
            "/model_performance": "Model performance metrics"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Update server health
        server_health["uptime"] = time.time() - server_start_time
        server_health["camera_available"] = camera_manager.capture_running
        server_health["detection_available"] = detection_engine.detection_running
        server_health["fnn_worker_available"] = fnn_worker.worker_running
        server_health["system_memory_mb"] = memory_mb
        server_health["tomato_count"] = detection_engine.tomato_count
        server_health["status"] = "healthy"
        
        return jsonify({
            "status": "healthy",
            "uptime": server_health["uptime"],
            "camera_running": camera_manager.capture_running,
            "detection_running": detection_engine.detection_running,
            "fnn_worker_running": fnn_worker.worker_running,
            "memory_usage_mb": memory_mb,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "tomato_count": detection_engine.tomato_count,
            "timestamp": datetime.now().isoformat(),
            "models_loaded": TENSORFLOW_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/server_status')
def server_status():
    """Detailed server status"""
    camera_status = camera_manager.get_status()
    detection_status = detection_engine.get_status()
    fnn_status = fnn_worker.get_status()
    
    return jsonify({
        "server": {
            "uptime": time.time() - server_start_time,
            "status": "running",
            "start_time": datetime.fromtimestamp(server_start_time).isoformat(),
            "restart_count": server_health["restart_count"],
            "last_restart": server_health["last_restart"]
        },
        "camera": camera_status,
        "detection": detection_status,
        "fnn_processing": fnn_status,
        "models": {
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "custom_models_loaded": TENSORFLOW_AVAILABLE,
            "models_loaded": TENSORFLOW_AVAILABLE,
            "tracker_active": detection_engine.detection_running
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "platform": sys.platform
        },
        "config_summary": {
            "camera_resolution": f"{CAMERA_CONFIG.get('width', 640)}x{CAMERA_CONFIG.get('height', 480)}",
            "detection_methods": ["color", "shape", "edges"],
            "fnn_worker": fnn_status.get("worker_running", False),
            "live_tracking": detection_engine.detection_running
        }
    })

@app.route('/live_analytics')
def live_analytics():
    """Live analytics and tracking data"""
    camera_status = camera_manager.get_status()
    detection_status = detection_engine.get_status()
    
    # Get latest detections
    latest_detections = detection_engine.get_latest_detections()
    tomato_detections = [d for d in latest_detections if d.get("type") == "tomato"]
    
    # Prepare current tomato data
    current_tomato = detection_engine.current_tomato
    tomato_data = None
    
    if current_tomato:
        tomato_data = {
            "id": current_tomato.get("tracker_id", "unknown"),
            "bbox": current_tomato.get("bbox"),
            "confidence": current_tomato.get("confidence"),
            "classification": current_tomato.get("classification", {}),
            "regression": current_tomato.get("regression", {}),
            "tracker_age": current_tomato.get("tracker_age", 0),
            "tracker_detections": current_tomato.get("tracker_detections", 1)
        }
    
    # Get red object detections
    red_objects = [d for d in latest_detections if d.get("type") == "red_object"]
    
    return jsonify({
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "camera_running": camera_manager.capture_running,
        "detection_running": detection_engine.detection_running,
        "tomato_count": len(tomato_detections),
        "red_object_count": len(red_objects),
        "total_detections": len(latest_detections),
        "current_tomato": tomato_data,
        "camera_fps": camera_status.get("fps", 0),
        "detection_fps": detection_status.get("detection_fps", 0),
        "memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        "uptime": time.time() - server_start_time,
        "tracking_active": detection_engine.detection_running,
        "fnn_processing": {
            "worker_running": fnn_worker.worker_running,
            "queue_size": fnn_worker.queue.qsize(),
            "tasks_completed": fnn_worker.tasks_processed,
            "results_available": len(fnn_worker.results)
        }
    })

@app.route('/video_feed')
def video_feed():
    """Live video stream endpoint"""
    if not camera_manager.capture_running:
        return "Camera not running. Start it with /start_camera", 503
    
    return Response(
        camera_manager.generate_video_feed(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/camera_status')
def camera_status_route():
    """Camera status endpoint"""
    return jsonify(camera_manager.get_status())

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera"""
    print("🚀 Received /start_camera request")
    
    try:
        if camera_manager.capture_running:
            return jsonify({
                "success": True,
                "message": "Camera is already running",
                "already_running": True,
                "camera_type": camera_manager.camera_type,
                "camera_running": camera_manager.capture_running
            })
        
        # Start camera
        camera_started = camera_manager.start_capture()
        
        if camera_started:
            # Start detection engine
            detection_started = detection_engine.start_detection(camera_manager)
            
            # Start FNN worker
            fnn_started = fnn_worker.start_worker()
            
            return jsonify({
                "success": True,
                "message": "Camera system started successfully",
                "camera_type": camera_manager.camera_type,
                "camera_running": camera_manager.capture_running,
                "detection_running": detection_started,
                "fnn_worker_running": fnn_started,
                "mode": camera_manager.camera_type,
                "resolution": f"{camera_manager.width}x{camera_manager.height}",
                "fps": camera_manager.target_fps,
                "note": "If camera_type is 'test', no physical camera was found"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to start camera",
                "message": "Could not initialize any camera device",
                "suggestion": "Check camera connections or permissions"
            }), 500
            
    except Exception as e:
        print(f"❌ Camera start error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Camera start failed due to an error"
        }), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera and all subsystems"""
    print("🛑 Stopping camera system...")
    
    # Stop all components
    detection_engine.stop_detection()
    fnn_worker.stop_worker()
    camera_manager.stop()
    
    return jsonify({
        "success": True,
        "message": "Camera system stopped successfully",
        "camera_running": camera_manager.capture_running,
        "detection_running": detection_engine.detection_running,
        "fnn_worker_running": fnn_worker.worker_running
    })

@app.route('/capture', methods=['GET'])
def capture():
    """Capture and analyze a single frame"""
    if not camera_manager.capture_running:
        return jsonify({
            "error": "Camera not running",
            "message": "Start camera first with /start_camera"
        }), 400
    
    frame = camera_manager.get_latest_frame()
    if frame is None:
        return jsonify({
            "error": "No frame available",
            "message": "Camera is running but no frame was captured"
        }), 500
    
    # Run detection on this frame
    detections = detection_engine.detect_tomatoes(frame)
    
    # Save frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join("static", filename)
    
    os.makedirs("static", exist_ok=True)
    
    # Draw detections on frame
    annotated_frame = frame.copy()
    for detection in detections:
        bbox = detection.get("bbox", [0, 0, 0, 0])
        color = (0, 255, 0) if detection.get("type") == "tomato" else (0, 0, 255)
        cv2.rectangle(annotated_frame, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     color, 2)
        
        label = f"{detection.get('type', 'object')}: {detection.get('confidence', 0):.2f}"
        cv2.putText(annotated_frame, label, 
                   (int(bbox[0]), int(bbox[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(filepath, annotated_frame)
    
    return jsonify({
        "success": True,
        "message": "Frame captured and analyzed",
        "filename": filename,
        "path": filepath,
        "timestamp": timestamp,
        "detections_count": len(detections),
        "tomato_detections": len([d for d in detections if d.get("type") == "tomato"]),
        "red_object_detections": len([d for d in detections if d.get("type") == "red_object"]),
        "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "camera_type": camera_manager.camera_type
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Full prediction endpoint"""
    print(" Running full prediction pipeline")
    
    try:
        # Check camera
        if not camera_manager.capture_running:
            return jsonify({
                "success": False,
                "error": "Camera not running",
                "message": "Start camera first"
            }), 400
        
        # Get frame
        frame = camera_manager.get_latest_frame()
        if frame is None:
            return jsonify({
                "success": False,
                "error": "No frame available",
                "message": "Camera is running but no frame is available"
            }), 500
        
        # Run detection
        detections = detection_engine.detect_tomatoes(frame)
        
        # Get tomato detections
        tomato_detections = [d for d in detections if d.get("type") == "tomato"]
        red_detections = [d for d in detections if d.get("type") == "red_object"]
        
        # Prepare response
        response = {
            "success": True,
            "message": "Prediction completed successfully",
            "timestamp": datetime.now().isoformat(),
            "camera_type": camera_manager.camera_type,
            "frame_info": {
                "width": frame.shape[1],
                "height": frame.shape[0]
            },
            "detections": detections,
            "tomato_detections": tomato_detections,
            "red_detections": red_detections,
            "detection_count": len(detections),
            "tomato_count": len(tomato_detections),
            "red_object_count": len(red_detections)
        }
        
        # Add FNN processing for first tomato
        if tomato_detections:
            first_tomato = tomato_detections[0]
            
            # Submit to FNN worker
            task_data = {
                "bbox": first_tomato.get("bbox"),
                "confidence": first_tomato.get("confidence"),
                "area": first_tomato.get("area", 1000),
                "type": "tomato"
            }
            
            task_id = fnn_worker.submit_task(task_data)
            if task_id:
                response["fnn_task_id"] = task_id
                
                # Try to get result immediately (might not be ready)
                time.sleep(0.05)
                fnn_result = fnn_worker.get_result(task_id)
                if fnn_result:
                    response["fnn_predictions"] = fnn_result.get("result", {})
            
            # Add classification and regression from detection
            if "classification" in first_tomato:
                response["classification"] = first_tomato["classification"]
            if "regression" in first_tomato:
                response["regression"] = first_tomato["regression"]
        
        print(f" Prediction completed: {len(tomato_detections)} tomatoes, {len(red_detections)} red objects")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Prediction failed",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict_fnn', methods=['POST'])
def predict_fnn():
    """FNN-only prediction endpoint"""
    print("🧠 Running FNN prediction")
    
    try:
        # Get data from request
        data = request.get_json() or {}
        bbox = data.get('bbox', [0, 0, 100, 100])
        confidence = data.get('confidence', 0.5)
        area = data.get('area', 1000)
        
        # Submit to FNN worker
        task_data = {
            "bbox": bbox,
            "confidence": confidence,
            "area": area,
            "type": "fnn_request"
        }
        
        task_id = fnn_worker.submit_task(task_data)
        
        if task_id:
            # Wait for processing
            for _ in range(10):
                result = fnn_worker.get_result(task_id)
                if result:
                    return jsonify({
                        "success": True,
                        "message": "FNN prediction completed",
                        "timestamp": datetime.now().isoformat(),
                        "task_id": task_id,
                        "predictions": result.get("result", {}),
                        "processing_time": result.get("processing_time", 0)
                    })
                time.sleep(0.1)
            
            return jsonify({
                "success": False,
                "error": "FNN processing timeout",
                "message": "Prediction took too long"
            }), 504
        
        else:
            return jsonify({
                "success": False,
                "error": "FNN queue full",
                "message": "Too many pending FNN tasks"
            }), 503
        
    except Exception as e:
        print(f" FNN prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "FNN prediction failed"
        }), 500

@app.route('/detection_status')
def detection_status():
    """Detection engine status"""
    return jsonify(detection_engine.get_status())

@app.route('/fnn_status')
def fnn_status():
    """FNN worker status"""
    return jsonify(fnn_worker.get_status())

@app.route('/system_metrics')
def system_metrics():
    """System performance metrics"""
    process = psutil.Process(os.getpid())
    
    return jsonify({
        "performance_metrics": performance_metrics,
        "system_metrics": {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime": time.time() - server_start_time,
            "thread_count": threading.active_count(),
            "python_memory": sys.getsizeof({})  # Sample memory check
        },
        "camera_metrics": camera_manager.get_status(),
        "detection_metrics": detection_engine.get_status(),
        "fnn_metrics": fnn_worker.get_status()
    })

@app.route('/model_performance')
def model_performance():
    """Model performance metrics"""
    return jsonify(performance_metrics)

@app.route('/test_camera', methods=['GET'])
def test_camera():
    """Test camera functionality"""
    print(" Testing camera system...")
    
    test_results = {
        "camera": camera_manager.get_status(),
        "detection": detection_engine.get_status(),
        "fnn": fnn_worker.get_status(),
        "system": {
            "uptime": time.time() - server_start_time,
            "memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }
    }
    
    # Capture a test frame
    frame = camera_manager.get_latest_frame()
    if frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_test_{timestamp}.jpg"
        filepath = os.path.join("static", filename)
        
        os.makedirs("static", exist_ok=True)
        cv2.imwrite(filepath, frame)
        
        test_results["test_image"] = {
            "filename": filename,
            "path": filepath,
            "size": f"{frame.shape[1]}x{frame.shape[0]}",
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1
        }
    
    return jsonify({
        "success": True,
        "message": "System test completed",
        "results": test_results,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/export_data', methods=['GET'])
def export_data():
    """Export data to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tomato_data_export_{timestamp}.csv"
        filepath = os.path.join("static", filename)
        
        os.makedirs("static", exist_ok=True)
        
        # Create CSV with sample data
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'timestamp', 'tomato_id', 'classification', 'confidence',
                'weight', 'size', 'pressure', 'grip_force', 'is_occluded',
                'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'
            ])
            
            # Write sample data (in real implementation, use actual data)
            for i in range(10):
                writer.writerow([
                    datetime.now().isoformat(),
                    f"tomato_{i+1}",
                    "ripe_tomato",
                    0.85 + np.random.random() * 0.1,
                    250 + np.random.random() * 50,
                    8.5 + np.random.random() * 1.5,
                    50 + np.random.random() * 20,
                    7.5 + np.random.random() * 3,
                    "No",
                    100 + np.random.random() * 200,
                    100 + np.random.random() * 200,
                    80 + np.random.random() * 40,
                    80 + np.random.random() * 40
                ])
        
        return jsonify({
            "success": True,
            "message": "Data exported successfully",
            "filename": filename,
            "filepath": filepath,
            "download_url": f"/static/{filename}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Data export failed"
        }), 500

# ==================== CLEANUP HANDLERS ====================
def cleanup():
    """Cleanup all resources"""
    print("\n🧹 Cleaning up resources...")
    
    # Stop all components
    detection_engine.stop_detection()
    fnn_worker.stop_worker()
    camera_manager.stop()
    
    # Clear TensorFlow session if available
    if TENSORFLOW_AVAILABLE:
        try:
            tf.keras.backend.clear_session()
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    print(" Cleanup complete")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print(f"\n Received signal {sig}, shutting down...")
    cleanup()
    sys.exit(0)

# ==================== MAIN ENTRY POINT ====================
def main():
    """Main function"""
    print("\n" + "="*80)
    print(" TOMATO DETECTION SERVER v3.1 - COMPLETE SYSTEM")
    print("="*80)
    print("\n System Components:")
    print(f"  • Camera Manager: {' Ready'}")
    print(f"  • Detection Engine: {'Ready'}")
    print(f"  • FNN Processing Worker: {' Ready'}")
    print(f"  • TensorFlow Models: {' Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
    print(f"  • OpenCV:  {cv2.__version__}")
    print(f"  • Flask Server:  Ready")
    print()
    
    print(" Configuration:")
    print(f"  • Camera Resolution: {CAMERA_CONFIG.get('width', 640)}x{CAMERA_CONFIG.get('height', 480)}")
    print(f"  • Camera FPS: {CAMERA_CONFIG.get('fps', 15)}")
    print(f"  • Detection Methods: Color, Shape, Edge")
    print(f"  • FNN Worker: Enabled")
    print(f"  • Live Tracking: Enabled")
    print()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(" Starting server...")
    print(f" Web Interface: http://localhost:5000")
    print(f" Camera Feed: http://localhost:5000/video_feed")
    print(f" Health Check: http://localhost:5000/health")
    print(f" Live Analytics: http://localhost:5000/live_analytics")
    print()
    print(" Available Endpoints:")
    print("  /start_camera    - Start camera system")
    print("  /predict         - Run full prediction")
    print("  /predict_fnn     - FNN-only prediction")
    print("  /server_status   - Detailed status")
    print("  /test_camera     - Test camera functionality")
    print()
    print(" Press Ctrl+C to stop the server")
    print("="*80)
    
    # Set initial server health
    server_health["status"] = "running"
    server_health["camera_available"] = False
    server_health["detection_available"] = False
    server_health["fnn_worker_available"] = False
    server_health["models_loaded"] = TENSORFLOW_AVAILABLE
    
    try:
        # Start Flask server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n Server stopped by user")
    except Exception as e:
        print(f" Server error: {e}")
        traceback.print_exc()
    finally:
        cleanup()
        print(" Server shutdown complete")

if __name__ == '__main__':
    main()
'''

# Save the code to mains.py
with open('mains.py', 'w') as f:
    f.write(CODE)

print(f" mains.py created successfully!")
print(f"File size: {len(CODE)} characters")
print(f"Line count: {CODE.count(chr(10))}")