# Model_Tracking.py - Complete Analytics and Tracking System WITH Distance Measurement
import time
import json
import csv
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field, asdict
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-tracker")

# ============================================================================
# DATA CLASSES - ALL ORIGINAL PLUS NEW POSITION CLASSES
# ============================================================================

@dataclass
class PositionRecord:
    """Object position and distance measurement - NEW CLASS"""
    timestamp: float
    # 2D bounding box position
    x: int  # X-coordinate in pixels
    y: int  # Y-coordinate in pixels
    width: int  # Width in pixels
    height: int  # Height in pixels
    
    # 3D position estimation (relative to camera)
    distance_mm: Optional[float] = None  # Distance from camera in millimeters
    real_width_mm: Optional[float] = None  # Estimated real width in mm
    real_height_mm: Optional[float] = None  # Estimated real height in mm
    camera_angle_x: Optional[float] = None  # Angle from camera center X-axis (degrees)
    camera_angle_y: Optional[float] = None  # Angle from camera center Y-axis (degrees)
    
    # Position tracking
    velocity_x: Optional[float] = None  # Pixel movement per second
    velocity_y: Optional[float] = None
    is_stable: Optional[bool] = None  # If object is stationary

@dataclass
class ClassificationRecord:
    """Single classification prediction record - UPDATED WITH POSITION"""
    timestamp: float
    prediction: str
    confidence: float
    is_occluded: bool
    source: str  # "custom_cnn" or "fallback"
    position: Optional[PositionRecord] = None  # ADDED position tracking

@dataclass
class RegressionRecord:
    """Single regression prediction record - UPDATED WITH POSITION"""
    timestamp: float
    tactile_properties: Dict[str, float]  # 9 tactile properties
    source: str  # "custom_fnn" or "fallback"
    position: Optional[PositionRecord] = None  # ADDED position tracking

@dataclass
class GANRecord:
    """GAN reconstruction event record - ORIGINAL UNCHANGED"""
    timestamp: float
    reconstruction_quality: Optional[float] = None
    occlusion_severity: Optional[float] = None
    successful: bool = True

@dataclass
class RoboticsRecord:
    """Robotics success/failure event record - ORIGINAL UNCHANGED"""
    timestamp: float
    success: bool
    grip_force: Optional[float] = None
    estimated_weight: Optional[float] = None
    failure_reason: Optional[str] = None
    recovery_action: Optional[str] = None

@dataclass
class TomatoTracking:
    """Complete tracking for a single tomato - UPDATED WITH POSITION HISTORY"""
    tomato_id: str
    detection_timestamp: float
    classification_history: List[ClassificationRecord] = field(default_factory=list)
    regression_history: List[RegressionRecord] = field(default_factory=list)
    gan_reconstructions: List[GANRecord] = field(default_factory=list)
    robotics_events: List[RoboticsRecord] = field(default_factory=list)
    position_history: List[PositionRecord] = field(default_factory=list)  # ADDED position history
    
    @property
    def is_occluded(self) -> bool:
        """Check if tomato was ever occluded"""
        if not self.classification_history:
            return False
        return any(rec.is_occluded for rec in self.classification_history)
    
    @property
    def final_classification(self) -> Optional[ClassificationRecord]:
        """Get most recent classification"""
        return self.classification_history[-1] if self.classification_history else None
    
    @property
    def final_regression(self) -> Optional[RegressionRecord]:
        """Get most recent regression"""
        return self.regression_history[-1] if self.regression_history else None
    
    @property
    def current_position(self) -> Optional[PositionRecord]:
        """Get most recent position"""
        return self.position_history[-1] if self.position_history else None
    
    @property
    def distance_from_camera(self) -> Optional[float]:
        """Get current distance from camera"""
        if self.position_history and self.position_history[-1].distance_mm:
            return self.position_history[-1].distance_mm
        return None

@dataclass
class HarvestingSession:
    """Complete harvesting session tracking - ORIGINAL UNCHANGED"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    tomatoes: Dict[str, TomatoTracking] = field(default_factory=dict)
    total_predictions: int = 0
    total_classifications: int = 0
    total_regressions: int = 0
    total_gan_reconstructions: int = 0
    total_robotics_events: int = 0
    
    @property
    def is_active(self) -> bool:
        return self.end_time is None
    
    @property
    def duration(self) -> float:
        """Session duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time

# ============================================================================
# NEW: DISTANCE CALCULATOR CLASS
# ============================================================================

class DistanceCalculator:
    """
    Calculate distance and 3D position from webcam images
    
    Methods available:
    1. Known size method (requires knowing real object size)
    2. Reference object method (requires calibration object in scene)
    3. Focus-based method (requires webcam with focus info)
    """
    
    def __init__(self, 
                 camera_fov: Tuple[float, float] = (60.0, 45.0),  # Camera field of view (horizontal, vertical)
                 known_object_width_mm: float = 70.0,  # Average tomato width in mm
                 known_object_height_mm: float = 65.0,  # Average tomato height in mm
                 image_width: int = 640,  # Webcam resolution
                 image_height: int = 480):
        """
        Initialize distance calculator
        
        Args:
            camera_fov: Field of view in degrees (horizontal, vertical)
            known_object_width_mm: Real-world width of target object in mm
            known_object_height_mm: Real-world height of target object in mm
        """
        self.camera_fov = camera_fov
        self.known_width_mm = known_object_width_mm
        self.known_height_mm = known_object_height_mm
        
        # Camera intrinsic parameters (simplified)
        self.focal_length_pixels = self._calculate_focal_length(image_width, image_height)
        self.image_center_x = image_width / 2
        self.image_center_y = image_height / 2
        
        # For tracking velocity
        self.previous_positions = {}  # tomato_id -> (timestamp, x, y)
        
        logger.info(f"✅ Distance Calculator initialized with known object size: {known_object_width_mm}x{known_object_height_mm}mm")
    
    def _calculate_focal_length(self, image_width: int, image_height: int) -> float:
        """
        Calculate focal length in pixels from FOV
        
        Formula: focal_length = (image_width / 2) / tan(FOV_horizontal / 2)
        """
        fov_horizontal_rad = math.radians(self.camera_fov[0])
        focal_length = (image_width / 2) / math.tan(fov_horizontal_rad / 2)
        return focal_length
    
    def calculate_position(self,
                          tomato_id: str,
                          bbox: Tuple[int, int, int, int],  # (x, y, width, height)
                          timestamp: float,
                          method: str = "size_based") -> PositionRecord:
        """
        Calculate distance and 3D position from bounding box
        
        Args:
            bbox: (x, y, width, height) in pixels
            method: "size_based", "reference_object", or "focus_based"
        """
        x, y, width_pixels, height_pixels = bbox
        
        # Calculate distance using selected method
        distance_mm = None
        real_width_mm = None
        real_height_mm = None
        
        if method == "size_based":
            # Method 1: Known size method (most reliable if object size is consistent)
            distance_mm, real_width_mm, real_height_mm = self._size_based_calculation(
                width_pixels, height_pixels
            )
        elif method == "reference_object":
            # Method 2: Reference object in scene (requires calibration)
            distance_mm = self._reference_object_calculation(width_pixels)
        elif method == "focus_based":
            # Method 3: Using webcam focus information (if available)
            distance_mm = self._focus_based_calculation(width_pixels)
        
        # Calculate angles from camera center
        angle_x, angle_y = self._calculate_angles(x, y, width_pixels, height_pixels)
        
        # Calculate velocity (if we have previous position)
        velocity_x, velocity_y, is_stable = self._calculate_velocity(
            tomato_id, x, y, timestamp
        )
        
        return PositionRecord(
            timestamp=timestamp,
            x=x,
            y=y,
            width=width_pixels,
            height=height_pixels,
            distance_mm=distance_mm,
            real_width_mm=real_width_mm,
            real_height_mm=real_height_mm,
            camera_angle_x=angle_x,
            camera_angle_y=angle_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            is_stable=is_stable
        )
    
    def _size_based_calculation(self, width_pixels: int, height_pixels: int) -> Tuple[float, float, float]:
        """
        Calculate distance using known real object size
        
        Formula: distance = (real_width * focal_length) / pixel_width
        """
        if width_pixels == 0 or height_pixels == 0:
            return None, None, None
        
        # Calculate distance based on width (more reliable for symmetric objects)
        distance_from_width = (self.known_width_mm * self.focal_length_pixels) / width_pixels
        
        # Calculate distance based on height
        distance_from_height = (self.known_height_mm * self.focal_length_pixels) / height_pixels
        
        # Average both estimates for better accuracy
        distance_mm = (distance_from_width + distance_from_height) / 2
        
        # Calculate estimated real size (can vary if tomatoes aren't uniform)
        real_width_mm = distance_mm * width_pixels / self.focal_length_pixels
        real_height_mm = distance_mm * height_pixels / self.focal_length_pixels
        
        return distance_mm, real_width_mm, real_height_mm
    
    def _reference_object_calculation(self, width_pixels: int) -> Optional[float]:
        """
        Calculate distance using reference object in scene
        
        Note: This requires a calibration object of known size in the image
        """
        # This would need implementation based on your specific calibration object
        # For now, returns None - you'd need to implement based on your setup
        return None
    
    def _focus_based_calculation(self, width_pixels: int) -> Optional[float]:
        """
        Calculate distance using webcam focus information
        
        Note: This requires webcam that provides focus distance information
        """
        # This would need webcam-specific API access
        return None
    
    def _calculate_angles(self, x: int, y: int, width: int, height: int) -> Tuple[float, float]:
        """
        Calculate horizontal and vertical angles from camera center
        
        Formula: angle = arctan((pixel_position - center) / focal_length)
        """
        # Calculate center of object
        center_x = x + (width / 2)
        center_y = y + (height / 2)
        
        # Calculate angles in radians, then convert to degrees
        angle_x_rad = math.atan2(center_x - self.image_center_x, self.focal_length_pixels)
        angle_y_rad = math.atan2(center_y - self.image_center_y, self.focal_length_pixels)
        
        angle_x_deg = math.degrees(angle_x_rad)
        angle_y_deg = math.degrees(angle_y_rad)
        
        return angle_x_deg, angle_y_deg
    
    def _calculate_velocity(self, 
                           tomato_id: str, 
                           x: int, y: int, 
                           timestamp: float) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
        """
        Calculate velocity and stability of object
        
        Returns: (velocity_x_px_per_sec, velocity_y_px_per_sec, is_stable)
        """
        if tomato_id in self.previous_positions:
            prev_time, prev_x, prev_y = self.previous_positions[tomato_id]
            time_delta = timestamp - prev_time
            
            if time_delta > 0:  # Avoid division by zero
                velocity_x = (x - prev_x) / time_delta
                velocity_y = (y - prev_y) / time_delta
                
                # Consider object stable if velocity < 5 pixels per second
                is_stable = (abs(velocity_x) < 5 and abs(velocity_y) < 5)
                
                # Update previous position
                self.previous_positions[tomato_id] = (timestamp, x, y)
                
                return velocity_x, velocity_y, is_stable
        
        # First detection or no previous data
        self.previous_positions[tomato_id] = (timestamp, x, y)
        return None, None, None
    
    def calibrate_with_reference(self, 
                                reference_width_mm: float, 
                                reference_width_pixels: int,
                                reference_distance_mm: float):
        """
        Calibrate using a reference object at known distance
        
        Args:
            reference_width_mm: Real width of reference object in mm
            reference_width_pixels: Width of reference object in pixels
            reference_distance_mm: Known distance of reference object from camera
        """
        # Update focal length based on reference object
        self.focal_length_pixels = (reference_width_pixels * reference_distance_mm) / reference_width_mm
        logger.info(f"✅ Camera calibrated. New focal length: {self.focal_length_pixels:.2f} pixels")
    
    def estimate_camera_height(self, ground_y_position: int, object_height_mm: float = 1000.0) -> Optional[float]:
        """
        Estimate camera height from object on ground plane
        
        Args:
            ground_y_position: Y-position of object on ground (in pixels)
            object_height_mm: Height of object from ground (mm)
        """
        # This assumes object is on the ground plane
        # Formula: camera_height = object_height * tan(vertical_angle)
        vertical_angle = self._calculate_angles(0, ground_y_position, 0, 0)[1]
        vertical_angle_rad = math.radians(vertical_angle)
        
        camera_height_mm = object_height_mm * math.tan(abs(vertical_angle_rad))
        return camera_height_mm

# ============================================================================
# MAIN TRACKING CLASS - COMPLETELY UPDATED WITH ALL ORIGINAL + NEW FEATURES
# ============================================================================

class ModelTracker:
    """Main analytics and tracking engine with distance measurement"""
    
    def __init__(self, 
                 max_history_days: int = 7,
                 enable_distance_tracking: bool = True,
                 camera_fov: Tuple[float, float] = (60.0, 45.0),
                 known_tomato_size_mm: Tuple[float, float] = (70.0, 65.0)):
        """
        Initialize the model tracker
        
        Args:
            max_history_days: Maximum days to keep session history
            enable_distance_tracking: Enable distance calculation features
            camera_fov: Camera field of view (horizontal, vertical) in degrees
            known_tomato_size_mm: Known tomato size (width, height) in mm
        """
        self.sessions: Dict[str, HarvestingSession] = {}
        self.current_session: Optional[HarvestingSession] = None
        self.session_lock = threading.Lock()
        
        # Distance calculation module - NEW
        self.distance_calculator = None
        if enable_distance_tracking:
            self.distance_calculator = DistanceCalculator(
                camera_fov=camera_fov,
                known_object_width_mm=known_tomato_size_mm[0],
                known_object_height_mm=known_tomato_size_mm[1]
            )
        
        # Live analytics buffers (last 100 data points for real-time graphs)
        self.live_classification_accuracy = deque(maxlen=100)
        self.live_regression_values = {
            'tactile_1': deque(maxlen=100),
            'tactile_2': deque(maxlen=100),
            'tactile_3': deque(maxlen=100),
            'tactile_4': deque(maxlen=100),
            'tactile_5': deque(maxlen=100),
            'tactile_6': deque(maxlen=100),
            'tactile_7': deque(maxlen=100),
            'tactile_8': deque(maxlen=100),
            'tactile_9': deque(maxlen=100),
        }
        self.live_success_rate = deque(maxlen=100)
        self.live_gan_quality = deque(maxlen=100)
        self.live_distances = deque(maxlen=100)  # ADDED for distance tracking
        
        # Performance metrics
        self.performance_metrics = {
            'avg_inference_time': deque(maxlen=500),
            'model_load_times': {},
            'error_counts': defaultdict(int)
        }
        
        # Tomato counter
        self.tomato_counter = 0
        self.max_history_days = max_history_days
        
        logger.info("✅ Model Tracker initialized with distance tracking" if enable_distance_tracking else "✅ Model Tracker initialized")
    
    # ============================================================================
    # SESSION MANAGEMENT - ALL ORIGINAL METHODS PRESERVED
    # ============================================================================
    
    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new harvesting session"""
        with self.session_lock:
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, overwriting")
            
            self.current_session = HarvestingSession(
                session_id=session_id,
                start_time=time.time()
            )
            self.sessions[session_id] = self.current_session
            self.tomato_counter = 0
            
            logger.info(f"📊 Started new harvesting session: {session_id}")
            return session_id
    
    def stop_current_session(self) -> Dict[str, Any]:
        """Stop current session and generate final report"""
        with self.session_lock:
            if not self.current_session:
                logger.warning("No active session to stop")
                return {}
            
            self.current_session.end_time = time.time()
            
            # Generate final session report
            report = self._generate_session_report(self.current_session.session_id)
            
            logger.info(f"📊 Stopped session {self.current_session.session_id}")
            logger.info(f"   Duration: {self.current_session.duration:.1f}s")
            logger.info(f"   Tomatoes: {len(self.current_session.tomatoes)}")
            logger.info(f"   Success Rate: {report.get('success_rate', 0):.1%}")
            
            self.current_session = None
            return report
    
    def get_or_create_tomato(self, tomato_id: Optional[str] = None) -> str:
        """Get or create a new tomato tracking ID"""
        with self.session_lock:
            if not self.current_session:
                # Auto-start session if none exists
                self.start_new_session()
            
            if tomato_id is None:
                self.tomato_counter += 1
                tomato_id = f"tomato_{self.tomato_counter:03d}"
            
            if tomato_id not in self.current_session.tomatoes:
                self.current_session.tomatoes[tomato_id] = TomatoTracking(
                    tomato_id=tomato_id,
                    detection_timestamp=time.time()
                )
            
            return tomato_id
    
    # ============================================================================
    # RECORDING METHODS - ALL ORIGINAL PLUS NEW POSITION METHODS
    # ============================================================================
    
    def record_classification(
        self, 
        tomato_id: str,
        prediction: str,
        confidence: float,
        is_occluded: bool,
        source: str = "custom_cnn"
    ) -> None:
        """Record a classification prediction - ORIGINAL METHOD PRESERVED"""
        with self.session_lock:
            if not self.current_session:
                return
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            record = ClassificationRecord(
                timestamp=time.time(),
                prediction=prediction,
                confidence=confidence,
                is_occluded=is_occluded,
                source=source
            )
            
            tomato.classification_history.append(record)
            self.current_session.total_classifications += 1
            self.current_session.total_predictions += 1
            
            # Update live analytics
            self.live_classification_accuracy.append(confidence)
            
            # Log performance
            self._log_performance('classification', confidence >= 0.7)
    
    def record_regression(
        self,
        tomato_id: str,
        tactile_properties: Dict[str, float],
        source: str = "custom_fnn"
    ) -> None:
        """Record regression predictions (9 tactile properties) - ORIGINAL METHOD"""
        with self.session_lock:
            if not self.current_session:
                return
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            record = RegressionRecord(
                timestamp=time.time(),
                tactile_properties=tactile_properties,
                source=source
            )
            
            tomato.regression_history.append(record)
            self.current_session.total_regressions += 1
            self.current_session.total_predictions += 1
            
            # Update live analytics for each tactile property
            for i in range(1, 10):
                key = f"tactile_property_{i}"
                if key in tactile_properties:
                    self.live_regression_values[f'tactile_{i}'].append(tactile_properties[key])
            
            # Log performance
            has_valid_values = any(v != 0 for v in tactile_properties.values())
            self._log_performance('regression', has_valid_values)
    
    def record_gan_reconstruction(
        self,
        tomato_id: str,
        reconstruction_quality: Optional[float] = None,
        occlusion_severity: Optional[float] = None,
        successful: bool = True
    ) -> None:
        """Record a GAN reconstruction event - ORIGINAL METHOD"""
        with self.session_lock:
            if not self.current_session:
                return
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            record = GANRecord(
                timestamp=time.time(),
                reconstruction_quality=reconstruction_quality,
                occlusion_severity=occlusion_severity,
                successful=successful
            )
            
            tomato.gan_reconstructions.append(record)
            self.current_session.total_gan_reconstructions += 1
            
            # Update live analytics
            if reconstruction_quality is not None:
                self.live_gan_quality.append(reconstruction_quality)
            
            # Log performance
            self._log_performance('gan_reconstruction', successful)
    
    def record_robotics_event(
        self,
        tomato_id: str,
        success: bool,
        grip_force: Optional[float] = None,
        estimated_weight: Optional[float] = None,
        failure_reason: Optional[str] = None,
        recovery_action: Optional[str] = None
    ) -> None:
        """Record a robotics success/failure event - ORIGINAL METHOD"""
        with self.session_lock:
            if not self.current_session:
                return
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            record = RoboticsRecord(
                timestamp=time.time(),
                success=success,
                grip_force=grip_force,
                estimated_weight=estimated_weight,
                failure_reason=failure_reason,
                recovery_action=recovery_action
            )
            
            tomato.robotics_events.append(record)
            self.current_session.total_robotics_events += 1
            
            # Update live analytics
            self.live_success_rate.append(1.0 if success else 0.0)
            
            # Log performance
            self._log_performance('robotics', success, not success)
    
    def record_inference_time(self, model_type: str, inference_time: float):
        """Record model inference time for performance tracking - ORIGINAL METHOD"""
        self.performance_metrics['avg_inference_time'].append(inference_time)
    
    # ============================================================================
    # NEW METHODS FOR DISTANCE AND POSITION TRACKING
    # ============================================================================
    
    def record_classification_with_position(
        self, 
        tomato_id: str,
        prediction: str,
        confidence: float,
        is_occluded: bool,
        bbox: Tuple[int, int, int, int],  # (x, y, width, height)
        source: str = "custom_cnn"
    ) -> Tuple[Optional[PositionRecord], Optional[ClassificationRecord]]:
        """NEW: Record classification with position tracking"""
        with self.session_lock:
            if not self.current_session:
                return None, None
            
            # Get or create tomato
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            # Calculate position if distance calculator is available
            position_record = None
            if self.distance_calculator and bbox:
                position_record = self.distance_calculator.calculate_position(
                    tomato_id, bbox, time.time()
                )
                
                # Add to tomato's position history
                tomato.position_history.append(position_record)
                
                # Update live analytics
                if position_record.distance_mm:
                    self.live_distances.append(position_record.distance_mm)
            
            # Create classification record with position
            record = ClassificationRecord(
                timestamp=time.time(),
                prediction=prediction,
                confidence=confidence,
                is_occluded=is_occluded,
                source=source,
                position=position_record
            )
            
            tomato.classification_history.append(record)
            self.current_session.total_classifications += 1
            self.current_session.total_predictions += 1
            
            # Update live analytics
            self.live_classification_accuracy.append(confidence)
            
            # Log performance
            self._log_performance('classification', confidence >= 0.7)
            
            # Log position info if available
            if position_record and position_record.distance_mm:
                logger.info(f"📏 Tomato {tomato_id} at {position_record.distance_mm:.0f}mm, "
                          f"Position: ({position_record.x}, {position_record.y}), "
                          f"Angles: ({position_record.camera_angle_x:.1f}°, {position_record.camera_angle_y:.1f}°)")
            
            return position_record, record
    
    def record_regression_with_position(
        self,
        tomato_id: str,
        tactile_properties: Dict[str, float],
        bbox: Tuple[int, int, int, int],  # (x, y, width, height)
        source: str = "custom_fnn"
    ) -> Tuple[Optional[PositionRecord], Optional[RegressionRecord]]:
        """NEW: Record regression with position tracking"""
        with self.session_lock:
            if not self.current_session:
                return None, None
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                tomato_id = self.get_or_create_tomato(tomato_id)
                tomato = self.current_session.tomatoes[tomato_id]
            
            # Calculate position
            position_record = None
            if self.distance_calculator and bbox:
                position_record = self.distance_calculator.calculate_position(
                    tomato_id, bbox, time.time()
                )
                tomato.position_history.append(position_record)
            
            record = RegressionRecord(
                timestamp=time.time(),
                tactile_properties=tactile_properties,
                source=source,
                position=position_record
            )
            
            tomato.regression_history.append(record)
            self.current_session.total_regressions += 1
            self.current_session.total_predictions += 1
            
            # Update live analytics for each tactile property
            for i in range(1, 10):
                key = f"tactile_property_{i}"
                if key in tactile_properties:
                    self.live_regression_values[f'tactile_{i}'].append(tactile_properties[key])
            
            # Log performance
            has_valid_values = any(v != 0 for v in tactile_properties.values())
            self._log_performance('regression', has_valid_values)
            
            return position_record, record
    
    def record_position_only(
        self,
        tomato_id: str,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[PositionRecord]:
        """NEW: Record just position information without classification/regression"""
        with self.session_lock:
            if not self.current_session:
                return None
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                return None
            
            if self.distance_calculator:
                position_record = self.distance_calculator.calculate_position(
                    tomato_id, bbox, time.time()
                )
                tomato.position_history.append(position_record)
                
                # Update live analytics
                if position_record.distance_mm:
                    self.live_distances.append(position_record.distance_mm)
                
                return position_record
            
            return None
    
    # ============================================================================
    # ANALYTICS AND REPORTING - ENHANCED WITH DISTANCE DATA
    # ============================================================================
    
    def get_live_analytics(self) -> Dict[str, Any]:
        """Get current live analytics for real-time graphs - ENHANCED WITH DISTANCE"""
        with self.session_lock:
            if not self.current_session:
                return self._get_empty_analytics()
            
            # Calculate current session averages
            session_stats = self._calculate_session_stats(self.current_session)
            
            # Prepare live graph data
            live_graphs = {
                'classification_accuracy': list(self.live_classification_accuracy),
                'success_rate': list(self.live_success_rate),
                'gan_quality': list(self.live_gan_quality),
                'distances': list(self.live_distances),  # ADDED
                'regression_trends': {
                    f'tactile_{i}': list(self.live_regression_values[f'tactile_{i}'])
                    for i in range(1, 10)
                },
                'timestamps': [time.time() - (i * 2) for i in range(len(self.live_classification_accuracy))]
            }
            
            # Current tomato being processed
            current_tomato = None
            current_position = None
            if self.current_session.tomatoes:
                latest_tomato_id = max(self.current_session.tomatoes.keys())
                tomato = self.current_session.tomatoes[latest_tomato_id]
                current_tomato = self._get_tomato_summary(latest_tomato_id)
                current_position = tomato.current_position
            
            return {
                'session_id': self.current_session.session_id,
                'session_active': self.current_session.is_active,
                'session_duration': self.current_session.duration,
                'current_tomato': current_tomato,
                'current_position': self._position_to_dict(current_position) if current_position else None,
                'tomato_count': len(self.current_session.tomatoes),
                'live_graphs': live_graphs,
                'current_stats': session_stats,
                'performance': {
                    'avg_inference_time': np.mean(self.performance_metrics['avg_inference_time']) 
                        if self.performance_metrics['avg_inference_time'] else 0,
                    'total_predictions': self.current_session.total_predictions,
                    'error_counts': dict(self.performance_metrics['error_counts'])
                },
                'distance_stats': self._calculate_distance_stats()  # ADDED
            }
    
    def generate_session_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive session report - ORIGINAL METHOD PRESERVED"""
        with self.session_lock:
            if session_id is None:
                if not self.current_session:
                    return {}
                session_id = self.current_session.session_id
            
            session = self.sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return {}
            
            return self._generate_session_report(session_id)
    
    def _generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """Internal method to generate session report - ENHANCED WITH DISTANCE"""
        session = self.sessions[session_id]
        
        # Calculate comprehensive statistics
        stats = self._calculate_session_stats(session)
        
        # Per-tomato analysis
        tomato_summaries = []
        for tomato_id, tomato in session.tomatoes.items():
            tomato_summary = self._get_tomato_summary(tomato_id, session)
            tomato_summaries.append(tomato_summary)
        
        # Generate graphs
        graph_data = self._generate_session_graphs(session)
        
        # Prepare downloadable data
        csv_data = self._session_to_csv(session)
        
        return {
            'session_id': session_id,
            'start_time': datetime.fromtimestamp(session.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(session.end_time or time.time()).isoformat() 
                if session.end_time else None,
            'duration_seconds': session.duration,
            'total_tomatoes': len(session.tomatoes),
            'summary_statistics': stats,
            'tomato_summaries': tomato_summaries,
            'graph_data': graph_data,
            'csv_data': csv_data,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_session_stats(self, session: HarvestingSession) -> Dict[str, Any]:
        """Calculate statistics for a session - ENHANCED WITH DISTANCE"""
        if not session.tomatoes:
            return self._get_empty_stats()
        
        stats = {
            'classification': {
                'total_predictions': session.total_classifications,
                'average_confidence': 0.0,
                'ripe_count': 0,
                'unripe_count': 0,
                'occluded_count': 0,
                'accuracy_rate': 0.0
            },
            'regression': {
                'total_predictions': session.total_regressions,
                'average_tactile_values': {f'tactile_{i}': 0.0 for i in range(1, 10)},
                'value_ranges': {f'tactile_{i}': {'min': 0.0, 'max': 0.0} for i in range(1, 10)},
                'consistency_score': 0.0
            },
            'gan': {
                'total_reconstructions': session.total_gan_reconstructions,
                'success_rate': 0.0,
                'average_quality': 0.0,
                'occlusion_handled': 0
            },
            'robotics': {
                'total_events': session.total_robotics_events,
                'success_rate': 0.0,
                'average_grip_force': 0.0,
                'common_failure_reasons': [],
                'recovery_success_rate': 0.0
            },
            'distance': {  # ADDED SECTION
                'average_distance_mm': 0.0,
                'min_distance_mm': 0.0,
                'max_distance_mm': 0.0,
                'distance_std_mm': 0.0,
                'position_variability': 0.0,
                'total_position_samples': 0
            },
            'overall': {
                'success_rate': 0.0,
                'efficiency_score': 0.0,
                'average_time_per_tomato': 0.0
            }
        }
        
        # Classification statistics - ORIGINAL
        confidences = []
        for tomato in session.tomatoes.values():
            for cls in tomato.classification_history:
                confidences.append(cls.confidence)
                if cls.prediction.lower() == 'ripe':
                    stats['classification']['ripe_count'] += 1
                elif cls.prediction.lower() == 'unripe':
                    stats['classification']['unripe_count'] += 1
                if cls.is_occluded:
                    stats['classification']['occluded_count'] += 1
        
        if confidences:
            stats['classification']['average_confidence'] = float(np.mean(confidences))
            stats['classification']['accuracy_rate'] = float(np.mean([c for c in confidences if c > 0.7]))
        
        # Regression statistics - ORIGINAL
        tactile_values = {f'tactile_{i}': [] for i in range(1, 10)}
        for tomato in session.tomatoes.values():
            for reg in tomato.regression_history:
                for i in range(1, 10):
                    key = f"tactile_property_{i}"
                    if key in reg.tactile_properties:
                        tactile_values[f'tactile_{i}'].append(reg.tactile_properties[key])
        
        for i in range(1, 10):
            values = tactile_values[f'tactile_{i}']
            if values:
                stats['regression']['average_tactile_values'][f'tactile_{i}'] = float(np.mean(values))
                stats['regression']['value_ranges'][f'tactile_{i}'] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # GAN statistics - ORIGINAL
        gan_qualities = []
        gan_successes = 0
        for tomato in session.tomatoes.values():
            for gan in tomato.gan_reconstructions:
                if gan.reconstruction_quality is not None:
                    gan_qualities.append(gan.reconstruction_quality)
                if gan.successful:
                    gan_successes += 1
                if gan.occlusion_severity is not None:
                    stats['gan']['occlusion_handled'] += 1
        
        if tomato.gan_reconstructions:
            stats['gan']['success_rate'] = gan_successes / len(tomato.gan_reconstructions)
            if gan_qualities:
                stats['gan']['average_quality'] = float(np.mean(gan_qualities))
        
        # Robotics statistics - ORIGINAL
        grip_forces = []
        successes = 0
        failures_by_reason = defaultdict(int)
        recoveries = 0
        total_recoveries = 0
        
        for tomato in session.tomatoes.values():
            for robot in tomato.robotics_events:
                if robot.success:
                    successes += 1
                if robot.grip_force is not None:
                    grip_forces.append(robot.grip_force)
                if robot.failure_reason:
                    failures_by_reason[robot.failure_reason] += 1
                if robot.recovery_action:
                    total_recoveries += 1
                    if robot.success:
                        recoveries += 1
        
        if session.total_robotics_events > 0:
            stats['robotics']['success_rate'] = successes / session.total_robotics_events
            if grip_forces:
                stats['robotics']['average_grip_force'] = float(np.mean(grip_forces))
            if failures_by_reason:
                stats['robotics']['common_failure_reasons'] = [
                    {'reason': k, 'count': v} 
                    for k, v in sorted(failures_by_reason.items(), key=lambda x: x[1], reverse=True)[:3]
                ]
            if total_recoveries > 0:
                stats['robotics']['recovery_success_rate'] = recoveries / total_recoveries
        
        # Distance statistics - NEW SECTION
        all_distances = []
        all_positions = []
        
        for tomato in session.tomatoes.values():
            for pos in tomato.position_history:
                if pos.distance_mm:
                    all_distances.append(pos.distance_mm)
                    all_positions.append({
                        'x': pos.x,
                        'y': pos.y,
                        'distance': pos.distance_mm
                    })
        
        if all_distances:
            stats['distance']['average_distance_mm'] = float(np.mean(all_distances))
            stats['distance']['min_distance_mm'] = float(np.min(all_distances))
            stats['distance']['max_distance_mm'] = float(np.max(all_distances))
            stats['distance']['distance_std_mm'] = float(np.std(all_distances))
            stats['distance']['position_variability'] = self._calculate_position_variability(all_positions)
            stats['distance']['total_position_samples'] = len(all_distances)
        
        # Overall statistics - ORIGINAL
        if session.tomatoes:
            total_successes = stats['robotics']['success_rate'] * session.total_robotics_events
            total_predictions = session.total_predictions
            if total_predictions > 0:
                stats['overall']['success_rate'] = total_successes / total_predictions
            
            if session.duration > 0 and len(session.tomatoes) > 0:
                stats['overall']['average_time_per_tomato'] = session.duration / len(session.tomatoes)
                
                # Efficiency score (higher is better)
                efficiency = (
                    stats['classification']['accuracy_rate'] * 0.3 +
                    stats['robotics']['success_rate'] * 0.4 +
                    (1.0 - (session.duration / (len(session.tomatoes) * 60))) * 0.3  # Time efficiency
                )
                stats['overall']['efficiency_score'] = float(np.clip(efficiency, 0, 1))
        
        return stats
    
    def _get_tomato_summary(self, tomato_id: str, session: Optional[HarvestingSession] = None) -> Dict[str, Any]:
        """Get summary for a specific tomato - ENHANCED WITH POSITION"""
        if session is None:
            if not self.current_session:
                return {}
            session = self.current_session
        
        tomato = session.tomatoes.get(tomato_id)
        if not tomato:
            return {}
        
        # Calculate tomato-specific statistics
        classification = tomato.final_classification
        regression = tomato.final_regression
        
        # Robotics success for this tomato
        tomato_success = any(r.success for r in tomato.robotics_events) if tomato.robotics_events else False
        
        # GAN reconstructions for this tomato
        gan_success = any(g.successful for g in tomato.gan_reconstructions) if tomato.gan_reconstructions else False
        
        # Position information
        position = tomato.current_position
        
        summary = {
            'tomato_id': tomato_id,
            'detection_time': datetime.fromtimestamp(tomato.detection_timestamp).isoformat(),
            'classification': {
                'prediction': classification.prediction if classification else None,
                'confidence': classification.confidence if classification else None,
                'is_occluded': tomato.is_occluded,
                'total_predictions': len(tomato.classification_history)
            },
            'regression': {
                'tactile_properties': regression.tactile_properties if regression else {},
                'total_predictions': len(tomato.regression_history)
            },
            'gan': {
                'reconstructions': len(tomato.gan_reconstructions),
                'successful': gan_success
            },
            'robotics': {
                'success': tomato_success,
                'total_attempts': len(tomato.robotics_events),
                'final_grip_force': tomato.robotics_events[-1].grip_force if tomato.robotics_events else None
            },
            'position': self._position_to_dict(position) if position else None,  # ADDED
            'processing_time': time.time() - tomato.detection_timestamp if tomato.detection_timestamp else 0
        }
        
        return summary
    
    # ============================================================================
    # NEW HELPER METHODS FOR DISTANCE TRACKING
    # ============================================================================
    
    def _calculate_distance_stats(self) -> Dict[str, Any]:
        """Calculate statistics about object distances"""
        if not self.current_session or not self.live_distances:
            return {
                'avg_distance_mm': 0,
                'min_distance_mm': 0,
                'max_distance_mm': 0,
                'distance_std': 0,
                'optimal_range_percentage': 0,
                'too_close_count': 0,
                'too_far_count': 0
            }
        
        distances = list(self.live_distances)
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        
        # Define optimal range for harvesting (example: 200-400mm)
        optimal_min = 200
        optimal_max = 400
        in_range = [d for d in distances if optimal_min <= d <= optimal_max]
        too_close = [d for d in distances if d < optimal_min]
        too_far = [d for d in distances if d > optimal_max]
        
        return {
            'avg_distance_mm': float(avg_distance),
            'min_distance_mm': float(min_distance),
            'max_distance_mm': float(max_distance),
            'distance_std': float(std_distance),
            'optimal_range_percentage': len(in_range) / len(distances) if distances else 0,
            'too_close_count': len(too_close),
            'too_far_count': len(too_far),
            'optimal_range_mm': (optimal_min, optimal_max)
        }
    
    def _calculate_position_variability(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate how much objects move around (0-1 scale, 1=very variable)"""
        if len(positions) < 2:
            return 0.0
        
        x_values = [p['x'] for p in positions]
        y_values = [p['y'] for p in positions]
        
        # Calculate normalized standard deviation
        x_std = np.std(x_values)
        y_std = np.std(y_values)
        
        # Normalize to 0-1 range (assuming image width/height of 640x480)
        normalized_variability = (x_std / 640 + y_std / 480) / 2
        
        return float(np.clip(normalized_variability, 0, 1))
    
    def _position_to_dict(self, position: PositionRecord) -> Dict[str, Any]:
        """Convert PositionRecord to dictionary"""
        if not position:
            return {}
        
        return {
            'timestamp': position.timestamp,
            'time_formatted': datetime.fromtimestamp(position.timestamp).strftime('%H:%M:%S.%f')[:-3],
            'x_pixels': position.x,
            'y_pixels': position.y,
            'width_pixels': position.width,
            'height_pixels': position.height,
            'distance_mm': position.distance_mm,
            'distance_cm': position.distance_mm / 10 if position.distance_mm else None,
            'real_width_mm': position.real_width_mm,
            'real_height_mm': position.real_height_mm,
            'camera_angle_x': position.camera_angle_x,
            'camera_angle_y': position.camera_angle_y,
            'velocity_x': position.velocity_x,
            'velocity_y': position.velocity_y,
            'is_stable': position.is_stable,
            'center_x': position.x + (position.width / 2),
            'center_y': position.y + (position.height / 2)
        }
    
    def get_tomato_position_history(self, tomato_id: str) -> List[Dict[str, Any]]:
        """Get complete position history for a tomato"""
        with self.session_lock:
            if not self.current_session:
                return []
            
            tomato = self.current_session.tomatoes.get(tomato_id)
            if not tomato:
                return []
            
            return [self._position_to_dict(pos) for pos in tomato.position_history]
    
    # ============================================================================
    # GRAPH GENERATION - ALL ORIGINAL METHODS PRESERVED
    # ============================================================================
    
    def _generate_session_graphs(self, session: HarvestingSession) -> Dict[str, Any]:
        """Generate graph data for session visualization - ORIGINAL"""
        if not session.tomatoes:
            return {}
        
        # Prepare time-series data
        time_points = []
        classification_accuracies = []
        success_rates = []
        grip_forces = []
        
        # Collect data in chronological order
        all_events = []
        for tomato in session.tomatoes.values():
            for cls in tomato.classification_history:
                all_events.append(('classification', cls.timestamp, cls.confidence))
            for robot in tomato.robotics_events:
                all_events.append(('robotics', robot.timestamp, 1.0 if robot.success else 0.0))
                if robot.grip_force:
                    all_events.append(('grip_force', robot.timestamp, robot.grip_force))
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x[1])
        
        # Process to create smooth time series
        for event_type, timestamp, value in all_events:
            time_points.append(timestamp - session.start_time)
            if event_type == 'classification':
                classification_accuracies.append(value)
                success_rates.append(success_rates[-1] if success_rates else 0.5)
                grip_forces.append(grip_forces[-1] if grip_forces else 0)
            elif event_type == 'robotics':
                success_rates.append(value)
                classification_accuracies.append(classification_accuracies[-1] if classification_accuracies else 0.5)
                grip_forces.append(grip_forces[-1] if grip_forces else 0)
            elif event_type == 'grip_force':
                grip_forces.append(value)
                classification_accuracies.append(classification_accuracies[-1] if classification_accuracies else 0.5)
                success_rates.append(success_rates[-1] if success_rates else 0.5)
        
        # Generate actual plot images (simplified - returns plot data)
        graph_data = {
            'time_series': {
                'timestamps': time_points,
                'classification_accuracy': classification_accuracies,
                'success_rate': success_rates,
                'grip_force': grip_forces
            },
            'summary_charts': {
                'model_performance': self._generate_model_performance_chart(session),
                'tactile_property_distribution': self._generate_tactile_distribution_chart(session),
                'success_failure_breakdown': self._generate_success_failure_chart(session)
            }
        }
        
        return graph_data
    
    def _generate_model_performance_chart(self, session: HarvestingSession) -> Dict[str, Any]:
        """Generate model performance comparison chart data - ORIGINAL"""
        if not session.tomatoes:
            return {}
        
        # Simplified chart data (frontend can use this to render charts)
        return {
            'labels': ['Classification', 'Regression', 'GAN', 'Robotics'],
            'accuracy': [
                self._calculate_average_confidence(session),
                0.85,  # Placeholder for regression accuracy
                self._calculate_gan_success_rate(session),
                self._calculate_robotics_success_rate(session)
            ],
            'response_times': [0.15, 0.08, 0.25, 0.12]  # Placeholder
        }
    
    def _generate_tactile_distribution_chart(self, session: HarvestingSession) -> Dict[str, Any]:
        """Generate tactile property distribution chart - ORIGINAL"""
        if not session.tomatoes:
            return {}
        
        tactile_means = {f'tactile_{i}': [] for i in range(1, 10)}
        
        for tomato in session.tomatoes.values():
            for reg in tomato.regression_history:
                for i in range(1, 10):
                    key = f"tactile_property_{i}"
                    if key in reg.tactile_properties:
                        tactile_means[f'tactile_{i}'].append(reg.tactile_properties[key])
        
        return {
            'properties': [f'TP{i}' for i in range(1, 10)],
            'averages': [float(np.mean(vals)) if vals else 0 for vals in tactile_means.values()],
            'std_dev': [float(np.std(vals)) if vals else 0 for vals in tactile_means.values()]
        }
    
    def _generate_success_failure_chart(self, session: HarvestingSession) -> Dict[str, Any]:
        """Generate success/failure breakdown chart - ORIGINAL"""
        if not session.tomatoes:
            return {}
        
        successes = 0
        failures = 0
        failure_reasons = defaultdict(int)
        
        for tomato in session.tomatoes.values():
            for robot in tomato.robotics_events:
                if robot.success:
                    successes += 1
                else:
                    failures += 1
                    if robot.failure_reason:
                        failure_reasons[robot.failure_reason] += 1
        
        return {
            'overall': {'success': successes, 'failure': failures},
            'failure_breakdown': [
                {'reason': reason, 'count': count}
                for reason, count in failure_reasons.items()
            ]
        }
    
    # ============================================================================
    # EXPORT METHODS - ALL ORIGINAL METHODS PRESERVED
    # ============================================================================
    
    def export_session_to_csv(self, session_id: str) -> str:
        """Export session data to CSV string - ORIGINAL"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return ""
            
            return self._session_to_csv(session)
    
    def _session_to_csv(self, session: HarvestingSession) -> str:
        """Convert session data to CSV format - ENHANCED WITH POSITION"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header - ENHANCED WITH POSITION COLUMNS
        writer.writerow([
            'tomato_id', 'timestamp', 'model_type', 'prediction_type',
            'value', 'confidence', 'success', 'distance_mm', 'x_pixels', 'y_pixels', 'details'
        ])
        
        # Write data rows
        for tomato_id, tomato in session.tomatoes.items():
            # Classification records
            for cls in tomato.classification_history:
                writer.writerow([
                    tomato_id,
                    datetime.fromtimestamp(cls.timestamp).isoformat(),
                    'classification',
                    cls.prediction,
                    '',
                    cls.confidence,
                    cls.confidence > 0.7,
                    cls.position.distance_mm if cls.position else '',
                    cls.position.x if cls.position else '',
                    cls.position.y if cls.position else '',
                    f"source:{cls.source},occluded:{cls.is_occluded}"
                ])
            
            # Regression records
            for reg in tomato.regression_history:
                for prop_name, prop_value in reg.tactile_properties.items():
                    writer.writerow([
                        tomato_id,
                        datetime.fromtimestamp(reg.timestamp).isoformat(),
                        'regression',
                        prop_name,
                        prop_value,
                        '',
                        prop_value > 0,  # Simple success criteria
                        reg.position.distance_mm if reg.position else '',
                        reg.position.x if reg.position else '',
                        reg.position.y if reg.position else '',
                        f"source:{reg.source}"
                    ])
            
            # GAN records
            for gan in tomato.gan_reconstructions:
                writer.writerow([
                    tomato_id,
                    datetime.fromtimestamp(gan.timestamp).isoformat(),
                    'gan_reconstruction',
                    'reconstruction',
                    gan.reconstruction_quality if gan.reconstruction_quality else '',
                    '',
                    gan.successful,
                    '',
                    '',
                    '',
                    f"occlusion_severity:{gan.occlusion_severity}"
                ])
            
            # Robotics records
            for robot in tomato.robotics_events:
                writer.writerow([
                    tomato_id,
                    datetime.fromtimestamp(robot.timestamp).isoformat(),
                    'robotics',
                    'grip_attempt',
                    robot.grip_force if robot.grip_force else '',
                    '',
                    robot.success,
                    '',
                    '',
                    '',
                    f"failure_reason:{robot.failure_reason},recovery:{robot.recovery_action}"
                ])
            
            # Position-only records (for frames without classification/regression)
            for pos in tomato.position_history:
                # Check if this position isn't already recorded with a classification/regression
                has_classification = any(cls.position == pos for cls in tomato.classification_history)
                has_regression = any(reg.position == pos for reg in tomato.regression_history)
                
                if not has_classification and not has_regression:
                    writer.writerow([
                        tomato_id,
                        datetime.fromtimestamp(pos.timestamp).isoformat(),
                        'position_tracking',
                        'position_update',
                        '',
                        '',
                        pos.is_stable if pos.is_stable is not None else '',
                        pos.distance_mm if pos.distance_mm else '',
                        pos.x,
                        pos.y,
                        f"width:{pos.width},height:{pos.height}"
                    ])
        
        return output.getvalue()
    
    def export_graph_as_png(self, graph_type: str, session_id: Optional[str] = None) -> bytes:
        """Generate and export graph as PNG bytes - ORIGINAL"""
        try:
            if session_id is None:
                if not self.current_session:
                    return b""
                session_id = self.current_session.session_id
            
            session = self.sessions.get(session_id)
            if not session:
                return b""
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if graph_type == 'performance':
                self._plot_performance_graph(ax, session)
            elif graph_type == 'tactile_distribution':
                self._plot_tactile_distribution(ax, session)
            elif graph_type == 'success_timeline':
                self._plot_success_timeline(ax, session)
            else:
                self._plot_summary_graph(ax, session)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            return b""
    
    def _plot_performance_graph(self, ax, session: HarvestingSession):
        """Plot model performance comparison - ORIGINAL"""
        labels = ['CNN Classification', 'FNN Regression', 'GAN Reconstruction', 'Robotics Control']
        accuracy = [
            self._calculate_average_confidence(session),
            0.85,  # Regression accuracy placeholder
            self._calculate_gan_success_rate(session),
            self._calculate_robotics_success_rate(session)
        ]
        
        colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800']
        bars = ax.bar(labels, accuracy, color=colors)
        ax.set_ylabel('Accuracy / Success Rate')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracy):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom')
    
    def _plot_tactile_distribution(self, ax, session: HarvestingSession):
        """Plot tactile property distribution - ORIGINAL"""
        tactile_data = self._generate_tactile_distribution_chart(session)
        
        x = np.arange(len(tactile_data['properties']))
        width = 0.35
        
        ax.bar(x - width/2, tactile_data['averages'], width, label='Average', color='#2196F3')
        ax.errorbar(x - width/2, tactile_data['averages'], yerr=tactile_data['std_dev'], 
                   fmt='none', color='black', capsize=3)
        
        ax.set_xlabel('Tactile Properties')
        ax.set_ylabel('Value')
        ax.set_title('Tactile Property Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(tactile_data['properties'])
        ax.legend()
    
    def _plot_success_timeline(self, ax, session: HarvestingSession):
        """Plot success rate over time - ORIGINAL"""
        graph_data = self._generate_session_graphs(session)
        time_data = graph_data.get('time_series', {})
        
        if not time_data:
            return
        
        ax.plot(time_data['timestamps'], time_data['success_rate'], 
                color='#4CAF50', linewidth=2, label='Success Rate')
        ax.plot(time_data['timestamps'], time_data['classification_accuracy'],
                color='#2196F3', linewidth=2, label='Classification Accuracy', alpha=0.7)
        
        ax.set_xlabel('Time (seconds from session start)')
        ax.set_ylabel('Rate / Accuracy')
        ax.set_title('Performance Over Time')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_graph(self, ax, session: HarvestingSession):
        """Plot general summary graph - ORIGINAL"""
        stats = self._calculate_session_stats(session)
        
        categories = ['Classification', 'Regression', 'GAN', 'Robotics', 'Overall']
        values = [
            stats['classification']['accuracy_rate'],
            0.85,  # Regression placeholder
            stats['gan']['success_rate'],
            stats['robotics']['success_rate'],
            stats['overall']['success_rate']
        ]
        
        colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800', '#FF5722']
        bars = ax.bar(categories, values, color=colors)
        
        ax.set_ylabel('Success Rate')
        ax.set_title('Session Performance Summary')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.1%}', ha='center', va='bottom')
    
    # ============================================================================
    # HELPER METHODS - ALL ORIGINAL PLUS NEW ONES
    # ============================================================================
    
    def _log_performance(self, model_type: str, success: bool, error: bool = False):
        """Log performance metrics - ORIGINAL"""
        if error:
            self.performance_metrics['error_counts'][model_type] += 1
    
    def _calculate_average_confidence(self, session: HarvestingSession) -> float:
        """Calculate average classification confidence - ORIGINAL"""
        confidences = []
        for tomato in session.tomatoes.values():
            for cls in tomato.classification_history:
                confidences.append(cls.confidence)
        return float(np.mean(confidences)) if confidences else 0.0
    
    def _calculate_gan_success_rate(self, session: HarvestingSession) -> float:
        """Calculate GAN reconstruction success rate - ORIGINAL"""
        successes = 0
        total = 0
        for tomato in session.tomatoes.values():
            for gan in tomato.gan_reconstructions:
                total += 1
                if gan.successful:
                    successes += 1
        return successes / total if total > 0 else 0.0
    
    def _calculate_robotics_success_rate(self, session: HarvestingSession) -> float:
        """Calculate robotics success rate - ORIGINAL"""
        successes = 0
        total = 0
        for tomato in session.tomatoes.values():
            for robot in tomato.robotics_events:
                total += 1
                if robot.success:
                    successes += 1
        return successes / total if total > 0 else 0.0
    
    def _get_empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure - ENHANCED WITH DISTANCE"""
        return {
            'session_id': None,
            'session_active': False,
            'session_duration': 0,
            'current_tomato': None,
            'current_position': None,
            'tomato_count': 0,
            'live_graphs': {
                'classification_accuracy': [],
                'success_rate': [],
                'gan_quality': [],
                'distances': [],  # ADDED
                'regression_trends': {f'tactile_{i}': [] for i in range(1, 10)},
                'timestamps': []
            },
            'current_stats': self._get_empty_stats(),
            'performance': {
                'avg_inference_time': 0,
                'total_predictions': 0,
                'error_counts': {}
            },
            'distance_stats': self._calculate_distance_stats()  # ADDED
        }
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure - ENHANCED WITH DISTANCE"""
        return {
            'classification': {
                'total_predictions': 0,
                'average_confidence': 0.0,
                'ripe_count': 0,
                'unripe_count': 0,
                'occluded_count': 0,
                'accuracy_rate': 0.0
            },
            'regression': {
                'total_predictions': 0,
                'average_tactile_values': {f'tactile_{i}': 0.0 for i in range(1, 10)},
                'value_ranges': {f'tactile_{i}': {'min': 0.0, 'max': 0.0} for i in range(1, 10)},
                'consistency_score': 0.0
            },
            'gan': {
                'total_reconstructions': 0,
                'success_rate': 0.0,
                'average_quality': 0.0,
                'occlusion_handled': 0
            },
            'robotics': {
                'total_events': 0,
                'success_rate': 0.0,
                'average_grip_force': 0.0,
                'common_failure_reasons': [],
                'recovery_success_rate': 0.0
            },
            'distance': {  # ADDED
                'average_distance_mm': 0.0,
                'min_distance_mm': 0.0,
                'max_distance_mm': 0.0,
                'distance_std_mm': 0.0,
                'position_variability': 0.0,
                'total_position_samples': 0
            },
            'overall': {
                'success_rate': 0.0,
                'efficiency_score': 0.0,
                'average_time_per_tomato': 0.0
            }
        }
    
    def get_active_session_id(self) -> Optional[str]:
        """Get ID of active session - ORIGINAL"""
        return self.current_session.session_id if self.current_session else None
    
    def cleanup_old_sessions(self):
        """Remove sessions older than max_history_days - ORIGINAL"""
        cutoff_time = time.time() - (self.max_history_days * 24 * 3600)
        to_delete = []
        
        for session_id, session in self.sessions.items():
            if session.end_time and session.end_time < cutoff_time:
                to_delete.append(session_id)
        
        for session_id in to_delete:
            del self.sessions[session_id]
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old sessions")
    
    def get_session_ids(self) -> List[str]:
        """Get list of all session IDs - ORIGINAL"""
        return list(self.sessions.keys())
    
    # ============================================================================
    # NEW DISTANCE-SPECIFIC METHODS
    # ============================================================================
    
    def calibrate_distance_calculator(self,
                                     reference_width_mm: float,
                                     reference_width_pixels: int,
                                     reference_distance_mm: float) -> bool:
        """
        Calibrate distance calculator with reference object
        
        Args:
            reference_width_mm: Real width of reference object
            reference_width_pixels: Pixel width in image
            reference_distance_mm: Known distance from camera
        """
        if not self.distance_calculator:
            logger.error("Distance calculator not enabled")
            return False
        
        try:
            self.distance_calculator.calibrate_with_reference(
                reference_width_mm,
                reference_width_pixels,
                reference_distance_mm
            )
            logger.info("✅ Distance calculator calibrated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to calibrate distance calculator: {e}")
            return False
    
    def estimate_camera_height(self, ground_y_position: int, object_height_mm: float = 1000.0) -> Optional[float]:
        """Estimate camera height from ground objects"""
        if not self.distance_calculator:
            return None
        
        return self.distance_calculator.estimate_camera_height(ground_y_position, object_height_mm)
    
    def get_optimal_harvesting_distance(self, min_mm: float = 200, max_mm: float = 400) -> Dict[str, Any]:
        """
        Check if current tomatoes are at optimal harvesting distance
        
        Returns statistics about how many tomatoes are in optimal range
        """
        with self.session_lock:
            if not self.current_session:
                return {'in_range': 0, 'total': 0, 'percentage': 0}
            
            in_range = 0
            total_with_distance = 0
            
            for tomato in self.current_session.tomatoes.values():
                distance = tomato.distance_from_camera
                if distance:
                    total_with_distance += 1
                    if min_mm <= distance <= max_mm:
                        in_range += 1
            
            return {
                'in_range': in_range,
                'total': total_with_distance,
                'percentage': in_range / total_with_distance if total_with_distance > 0 else 0,
                'optimal_range_mm': (min_mm, max_mm)
            }


# ============================================================================
# SINGLETON INSTANCE FOR IMPORT
# ============================================================================

_tracker_instance = None

def get_tracker(enable_distance_tracking: bool = True) -> ModelTracker:
    """Get or create the global tracker instance with distance tracking"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ModelTracker(enable_distance_tracking=enable_distance_tracking)
    return _tracker_instance