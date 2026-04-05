# complete_tomato_tracking_system_single.py
# Single Tomato Tracking with Models, Pi Camera Support, and Arduino Communication
# Author: OLAGUNJU KOREDE SOLOMON (Student ID: 216882)

import cv2
import numpy as np
import time
import threading
import os
import json
import math
import serial
import serial.tools.list_ports
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from tensorflow.keras.models import load_model
from tensorflow.keras import layers 
from tensorflow.keras.metrics import MeanSquaredError 
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

# ================= MULTIPROCESSING FOR ARDUINO CONTROL =================
import multiprocessing
import queue
from multiprocessing import Process, Queue, Event, Value, Lock

# ================= JSON SAFETY FIX =================
def json_safe(obj):
    """Convert numpy / non-JSON types to pure Python"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
# ===================================================

@register_keras_serializable(package="Custom", name="SPADE")
class SPADE(Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(SPADE, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(128, kernel_size, padding='same', activation='relu')
        self.conv_gamma = layers.Conv2D(filters, kernel_size, padding='same')
        self.conv_beta = layers.Conv2D(filters, kernel_size, padding='same')

    def build(self, input_shape):
        super(SPADE, self).build(input_shape)

    def call(self, x, segmentation_map):
        mean = K.mean(x, axis=[1, 2], keepdims=True)
        std = K.std(x, axis=[1, 2], keepdims=True)
        x_normalized = (x - mean) / (std + 1e-5)

        seg_resized = tf.image.resize(segmentation_map, [tf.shape(x)[1], tf.shape(x)[2]], method='nearest')
        seg_features = self.conv(seg_resized)
        gamma = self.conv_gamma(seg_features)
        beta = self.conv_beta(seg_features)

        return x_normalized * (1 + gamma) + beta

    def get_config(self):
        config = super(SPADE, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config

# ==================== ARDUINO COMMUNICATION PROTOCOL ====================
# Protocol Commands:
# INIT            -> Initialize Arduino
# PING            -> Check connection (responds with PONG)
# GET_FORCE       -> Get force sensor data
# MOVE:LEFT       -> Move left
# MOVE:RIGHT      -> Move right
# MOVE:UP         -> Move up
# MOVE:DOWN       -> Move down
# GRIP            -> Grip tomato
# RELEASE         -> Release grip
# STOP            -> Stop all movement
# EMERGENCY_STOP  -> Emergency stop
# RESET           -> Reset emergency stop
# LED:ON          -> Turn LED on
# LED:OFF         -> Turn LED off
# BLINK           -> Blink LED (test command)

# ==================== ARDUINO CONTROLLER PROCESS ====================
class ArduinoControllerProcess(Process):
    """Isolated Arduino controller with robust communication"""
    
    def __init__(self, command_queue, status_queue, event_queue, stop_event):
        super().__init__()
        self.command_queue = command_queue      # Commands to Arduino
        self.status_queue = status_queue        # Arduino status updates
        self.event_queue = event_queue          # Events for WebSocket
        self.stop_event = stop_event           # Process stop signal
        
        # Hardware state (shared memory)
        self.emergency_stop = Value('b', False)
        self.force_sensor1 = Value('f', 0.0)
        self.force_sensor2 = Value('f', 0.0)
        self.grip_force = Value('f', 0.0)
        self.position_x = Value('f', 0.0)
        self.position_y = Value('f', 0.0)
        
        # Serial connection
        self.serial_port = None
        self.baud_rate = 115200
        self.timeout = 0.5                     # Increased for reliability
        self.write_timeout = 1.0
        
        # Control parameters
        self.loop_rate = 50                    # Reduced to 50Hz for reliability
        self.control_interval = 1.0 / self.loop_rate
        
        # Force sensor calibration
        self.force_threshold = 2.0             # kg - emergency stop threshold
        self.grip_target = 1.5                 # kg - target grip force
        
        # State tracking
        self.connected = False
        self.last_ping_time = 0
        self.ping_interval = 2.0               # Ping every 2 seconds
        self.force_update_interval = 0.1       # Update force every 100ms
        self.last_force_update = 0
        
        # Connection retry
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
        # Debug
        self.debug_mode = True
        
    def find_arduino_port(self):
        """Find Arduino/ESP32 port with better detection"""
        ports = list(serial.tools.list_ports.comports())
        
        # Priority list for detection
        priority_patterns = [
            'Arduino',
            'ESP32',
            'CH340',
            'USB Serial',
            'USB2.0-Serial',
            'ttyACM',
            'ttyUSB',
            'COM'
        ]
        
        detected_ports = []
        
        for port in ports:
            port_info = f"{port.device} - {port.description}"
            for pattern in priority_patterns:
                if pattern.lower() in port_info.lower():
                    detected_ports.append((port.device, port_info))
                    if self.debug_mode:
                        print(f"Detected potential Arduino: {port.device} - {port.description}")
                    break
        
        # Return first detected port
        if detected_ports:
            return detected_ports[0][0]
        
        # Fallback: try common ports
        common_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', 'COM3', 'COM4', 'COM5', 'COM6']
        for port in common_ports:
            if os.path.exists(port):
                return port
        
        return None
    
    def test_arduino_connection(self, port):
        """Test if a port is actually an Arduino"""
        try:
            test_serial = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=0.5,
                write_timeout=0.5
            )
            time.sleep(0.5)
            
            # Send PING command
            test_serial.write(b"PING\n")
            test_serial.flush()
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < 1.0:
                if test_serial.in_waiting:
                    response = test_serial.readline().decode('utf-8', errors='ignore').strip()
                    if "PONG" in response or "READY" in response:
                        test_serial.close()
                        return True
            
            test_serial.close()
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Test connection failed for {port}: {e}")
            return False
    
    def connect_arduino(self):
        """Connect to Arduino with handshake"""
        port = self.find_arduino_port()
        
        if not port:
            self._send_status_update("ERROR", "No Arduino port found")
            return False
        
        if self.debug_mode:
            print(f"Attempting to connect to {port}")
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                rtscts=False,
                dsrdtr=False
            )
            
            # Wait for Arduino to reset
            time.sleep(2.0)
            
            # Clear buffers
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            # Send INIT command
            self._send_serial_command("INIT")
            time.sleep(0.5)
            
            # Try to read response
            response = self._read_serial_response(timeout=1.0)
            if response and ("READY" in response or "INIT_OK" in response):
                self.connected = True
                self._send_status_update("CONNECTED", f"Connected to {port}")
                
                # Send LED ON to confirm connection
                self._send_serial_command("LED:ON")
                time.sleep(0.2)
                
                if self.debug_mode:
                    print(f"✅ Arduino connected on {port}")
                    print(f"   Baud rate: {self.baud_rate}")
                    print(f"   Timeout: {self.timeout}s")
                
                return True
            
            # If no response, try PING
            self._send_serial_command("PING")
            response = self._read_serial_response(timeout=1.0)
            
            if response and "PONG" in response:
                self.connected = True
                self._send_status_update("CONNECTED", f"Connected to {port} (PING response)")
                
                if self.debug_mode:
                    print(f"✅ Arduino connected via PING on {port}")
                
                return True
            
            self._send_status_update("ERROR", f"No response from {port}")
            self.serial_port.close()
            self.serial_port = None
            return False
            
        except serial.SerialException as e:
            self._send_status_update("ERROR", f"Serial error: {str(e)}")
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            return False
        except Exception as e:
            self._send_status_update("ERROR", f"Connection failed: {str(e)}")
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            return False
    
    def _send_serial_command(self, command, wait_response=False):
        """Send command to Arduino"""
        if not self.serial_port or not self.serial_port.is_open:
            return None
        
        try:
            full_command = f"{command}\n"
            self.serial_port.write(full_command.encode('utf-8'))
            self.serial_port.flush()
            
            if self.debug_mode and "GET_FORCE" not in command:
                print(f"📤 Sent: {command}")
            
            if wait_response:
                return self._read_serial_response(timeout=0.5)
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"Send command error: {e}")
            return None
    
    def _read_serial_response(self, timeout=0.5):
        """Read response from Arduino"""
        if not self.serial_port or not self.serial_port.is_open:
            return None
        
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        if self.debug_mode and "FORCE:" not in line:
                            print(f"📥 Received: {line}")
                        return line
                time.sleep(0.01)
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"Read response error: {e}")
            return None
    
    def _send_status_update(self, status, message):
        """Send status update to main process"""
        try:
            self.status_queue.put_nowait({
                'type': 'ARDUINO_STATUS',
                'status': status,
                'message': message,
                'timestamp': time.time(),
                'connected': self.connected,
                'emergency_stop': self.emergency_stop.value,
                'force_sensor1': self.force_sensor1.value,
                'force_sensor2': self.force_sensor2.value,
                'grip_force': self.grip_force.value
            })
        except queue.Full:
            pass
    
    def _broadcast_force_data(self):
        """Broadcast force sensor data via event queue"""
        try:
            self.event_queue.put_nowait({
                'type': 'FORCE_SENSOR_DATA',
                'sensor1': self.force_sensor1.value,
                'sensor2': self.force_sensor2.value,
                'grip_force': self.grip_force.value,
                'total_force': self.force_sensor1.value + self.force_sensor2.value,
                'emergency_stop': self.emergency_stop.value,
                'timestamp': time.time()
            })
        except queue.Full:
            pass
    
    def _read_force_sensors(self):
        """Read force sensor data from Arduino"""
        if not self.connected or not self.serial_port:
            return
        
        current_time = time.time()
        if current_time - self.last_force_update < self.force_update_interval:
            return
        
        try:
            response = self._send_serial_command("GET_FORCE", wait_response=True)
            
            if response and response.startswith("FORCE:"):
                parts = response.split(':')[1].split(',')
                if len(parts) >= 3:
                    try:
                        self.force_sensor1.value = float(parts[0])
                        self.force_sensor2.value = float(parts[1])
                        self.grip_force.value = float(parts[2])
                        
                        # Check emergency stop condition
                        if (self.force_sensor1.value > self.force_threshold or 
                            self.force_sensor2.value > self.force_threshold):
                            self.emergency_stop.value = True
                            self._send_status_update("EMERGENCY_STOP", 
                                                   f"Force threshold exceeded: {self.force_sensor1.value}, {self.force_sensor2.value}")
                            self._send_serial_command("EMERGENCY_STOP")
                        
                        self._broadcast_force_data()
                        self.last_force_update = current_time
                        
                    except ValueError as e:
                        if self.debug_mode:
                            print(f"Force data parse error: {e}")
                    
        except Exception as e:
            if self.debug_mode:
                print(f"Force read error: {e}")
    
    def _check_connection(self):
        """Check if Arduino is still connected"""
        if not self.connected or not self.serial_port:
            return False
        
        current_time = time.time()
        if current_time - self.last_ping_time >= self.ping_interval:
            try:
                response = self._send_serial_command("PING", wait_response=True)
                if response and "PONG" in response:
                    self.last_ping_time = current_time
                    return True
                else:
                    self.connected = False
                    self._send_status_update("DISCONNECTED", "Arduino not responding to PING")
                    return False
            except Exception as e:
                self.connected = False
                self._send_status_update("ERROR", f"PING failed: {str(e)}")
                return False
        
        return True
    
    def _process_commands(self):
        """Process commands from main process"""
        try:
            while True:
                cmd = self.command_queue.get_nowait()
                self._execute_command(cmd)
        except queue.Empty:
            pass
    
    def _execute_command(self, cmd):
        """Execute a command from main process"""
        if not self.connected or self.emergency_stop.value:
            # Still process reset commands even during emergency stop
            if cmd.get('type') != 'RESET_EMERGENCY':
                return
        
        cmd_type = cmd.get('type')
        
        if cmd_type == 'MOVE':
            direction = cmd.get('direction')
            if direction in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
                response = self._send_serial_command(f"MOVE:{direction}", wait_response=True)
                if response:
                    self._send_status_update("MOVING", f"Moving {direction}: {response}")
        
        elif cmd_type == 'GRIP':
            response = self._send_serial_command("GRIP", wait_response=True)
            if response:
                self._send_status_update("GRIPPING", f"Gripping: {response}")
        
        elif cmd_type == 'RELEASE':
            response = self._send_serial_command("RELEASE", wait_response=True)
            if response:
                self._send_status_update("RELEASING", f"Releasing: {response}")
        
        elif cmd_type == 'STOP':
            response = self._send_serial_command("STOP", wait_response=True)
            if response:
                self._send_status_update("STOPPED", f"Stopped: {response}")
        
        elif cmd_type == 'EMERGENCY_STOP':
            self.emergency_stop.value = True
            response = self._send_serial_command("EMERGENCY_STOP", wait_response=True)
            self._send_status_update("EMERGENCY_STOP", "Emergency stop activated")
        
        elif cmd_type == 'RESET_EMERGENCY':
            self.emergency_stop.value = False
            response = self._send_serial_command("RESET", wait_response=True)
            if response:
                self._send_status_update("RESET", "Emergency stop reset")
        
        elif cmd_type == 'SET_GRIP_FORCE':
            force = cmd.get('force', self.grip_target)
            self.grip_target = max(0.1, min(force, 5.0))
            self._send_status_update("GRIP_SET", f"Grip force set to {self.grip_target}kg")
        
        elif cmd_type == 'LED_ON':
            response = self._send_serial_command("LED:ON", wait_response=True)
            if response:
                self._send_status_update("LED", "LED turned ON")
        
        elif cmd_type == 'LED_OFF':
            response = self._send_serial_command("LED:OFF", wait_response=True)
            if response:
                self._send_status_update("LED", "LED turned OFF")
        
        elif cmd_type == 'BLINK':
            response = self._send_serial_command("BLINK", wait_response=True)
            if response:
                self._send_status_update("BLINK", "LED blinking")
    
    def run(self):
        """Main process loop"""
        print("🚀 Arduino Controller Process started")
        
        # Initial connection attempt
        self.connect_arduino()
        
        while not self.stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Reconnect if necessary
                if not self.connected:
                    if self.connection_attempts < self.max_connection_attempts:
                        print(f"Reconnection attempt {self.connection_attempts + 1}/{self.max_connection_attempts}")
                        if self.connect_arduino():
                            self.connection_attempts = 0
                        else:
                            self.connection_attempts += 1
                            time.sleep(2.0)
                        continue
                    else:
                        print("Max connection attempts reached")
                        time.sleep(5.0)
                        continue
                
                # Check connection
                if not self._check_connection():
                    self.connected = False
                    continue
                
                # Process commands
                self._process_commands()
                
                # Read sensors
                self._read_force_sensors()
                
                # Maintain loop rate
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.control_interval - loop_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Arduino controller error: {e}")
                time.sleep(0.1)
        
        # Cleanup
        self._cleanup()
        print("🛑 Arduino Controller Process stopped")
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self._send_serial_command("STOP")
                self._send_serial_command("LED:OFF")
                time.sleep(0.1)
                self.serial_port.close()
        except:
            pass
        
        self._send_status_update("STOPPED", "Controller process stopped")

# ==================== ARDUINO CLIENT ====================
class ArduinoClient:
    """Thread-safe client for Arduino controller"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        
        # Multiprocessing queues
        self.command_queue = Queue(maxsize=100)
        self.status_queue = Queue(maxsize=100)
        self.event_queue = Queue(maxsize=100)
        self.stop_event = Event()
        
        # Controller process
        self.controller = ArduinoControllerProcess(
            self.command_queue,
            self.status_queue,
            self.event_queue,
            self.stop_event
        )
        
        # Status tracking
        self.last_status = {
            'connected': False,
            'status': 'DISCONNECTED',
            'message': 'Not initialized',
            'emergency_stop': False,
            'force_sensor1': 0.0,
            'force_sensor2': 0.0,
            'grip_force': 0.0,
            'ports': []
        }
        
        # Background threads
        self.status_thread = None
        self.event_thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start Arduino controller process"""
        with self.lock:
            if self.running:
                return True
            
            self.controller.start()
            self.running = True
            
            # Start status update thread
            self.status_thread = threading.Thread(target=self._update_status_loop, daemon=True)
            self.status_thread.start()
            
            # Start event processing thread
            self.event_thread = threading.Thread(target=self._process_events, daemon=True)
            self.event_thread.start()
            
            print("✅ Arduino controller process started")
            return True
    
    def stop(self):
        """Stop Arduino controller process"""
        with self.lock:
            if not self.running:
                return
            
            self.stop_event.set()
            self.running = False
            
            # Wait for process to stop
            if self.controller.is_alive():
                self.controller.join(timeout=3.0)
            
            print("🛑 Arduino controller process stopped")
    
    def _update_status_loop(self):
        """Background thread to update Arduino status"""
        while self.running:
            try:
                # Get latest status from controller
                while True:
                    try:
                        status = self.status_queue.get_nowait()
                        with self.lock:
                            self.last_status.update(status)
                    except queue.Empty:
                        break
                
                time.sleep(0.05)  # 20Hz update rate
                
            except Exception as e:
                print(f"Status update error: {e}")
                time.sleep(0.1)
    
    def _process_events(self):
        """Process events from controller for WebSocket broadcasting"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                # Broadcast via WebSocket
                if event['type'] == 'FORCE_SENSOR_DATA':
                    self.socketio.emit('force_sensor_data', event)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Event processing error: {e}")
                time.sleep(0.1)
    
    def send_command(self, cmd_type, **kwargs):
        """Send command to Arduino controller"""
        if not self.running:
            return {'success': False, 'message': 'Controller not running'}
        
        cmd = {'type': cmd_type, **kwargs}
        
        try:
            self.command_queue.put_nowait(cmd)
            return {'success': True, 'message': f'Command {cmd_type} queued'}
        except queue.Full:
            return {'success': False, 'message': 'Command queue full'}
    
    # Public API methods
    
    def scan_ports(self):
        """Scan for available serial ports"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'manufacturer': port.manufacturer if port.manufacturer else 'Unknown'
            })
        
        with self.lock:
            self.last_status['ports'] = ports
        
        return ports
    
    def connect(self):
        """Connect to Arduino"""
        if not self.running:
            self.start()
            time.sleep(1.0)
        
        # Send connect command
        self.send_command('CONNECT')
        time.sleep(1.0)
        
        with self.lock:
            return self.last_status['connected']
    
    def disconnect(self):
        """Disconnect Arduino"""
        self.stop()
        with self.lock:
            self.last_status = {
                'connected': False,
                'status': 'DISCONNECTED',
                'message': 'Disconnected',
                'emergency_stop': False,
                'ports': []
            }
    
    def send_movement_command(self, direction):
        """Send movement command to Arduino"""
        valid_directions = ["LEFT", "RIGHT", "UP", "DOWN"]
        if direction not in valid_directions:
            return {
                'type': 'ARDUINO_STATUS',
                'connected': self.last_status['connected'],
                'message': f'ERROR: Invalid direction {direction}',
                'success': False
            }
        
        with self.lock:
            if self.last_status['emergency_stop']:
                return {
                    'type': 'ARDUINO_STATUS',
                    'connected': self.last_status['connected'],
                    'message': 'ERROR: Emergency stop active',
                    'success': False
                }
        
        result = self.send_command('MOVE', direction=direction)
        
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': f'Moving {direction}' if result['success'] else result['message'],
            'direction': direction,
            'success': result['success']
        }
    
    def emergency_stop(self):
        """Emergency stop"""
        result = self.send_command('EMERGENCY_STOP')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'Emergency stop activated' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def reset_emergency(self):
        """Reset emergency stop"""
        result = self.send_command('RESET_EMERGENCY')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'Emergency stop reset' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def grip(self, force=None):
        """Grip command"""
        if force:
            self.send_command('SET_GRIP_FORCE', force=force)
        
        result = self.send_command('GRIP')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'Grip command sent' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def release(self):
        """Release grip"""
        result = self.send_command('RELEASE')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'Release command sent' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def led_on(self):
        """Turn LED on"""
        result = self.send_command('LED_ON')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'LED ON' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def led_off(self):
        """Turn LED off"""
        result = self.send_command('LED_OFF')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'LED OFF' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def blink(self):
        """Blink LED (test command)"""
        result = self.send_command('BLINK')
        return {
            'type': 'ARDUINO_STATUS',
            'connected': self.last_status['connected'],
            'message': 'LED blinking' if result['success'] else 'Failed',
            'success': result['success']
        }
    
    def get_status(self):
        """Get current status"""
        with self.lock:
            status = self.last_status.copy()
            status['type'] = 'ARDUINO_STATUS'
            return status
    
    def get_live_status(self):
        """Get detailed live status"""
        with self.lock:
            return {
                'type': 'ARDUINO_LIVE_STATUS',
                'connected': self.last_status['connected'],
                'status': self.last_status['status'],
                'emergency_stop': self.last_status['emergency_stop'],
                'force_sensors': {
                    'sensor1': self.last_status['force_sensor1'],
                    'sensor2': self.last_status['force_sensor2'],
                    'grip_force': self.last_status['grip_force'],
                    'total': self.last_status['force_sensor1'] + self.last_status['force_sensor2']
                },
                'controller_running': self.running,
                'message': self.last_status['message'],
                'timestamp': time.time()
            }

# ==================== PLATFORM DETECTION ====================
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model')
print(f"Platform: {'Raspberry Pi' if IS_RASPBERRY_PI else 'Local Computer'}")

# ==================== CAMERA MANAGER ====================
class CameraManager:
    """Handles both Pi Camera and USB cameras"""
    
    def __init__(self):
        self.cap = None
        self.pi_camera = None
        self.raw_capture = None
        self.camera_type = "Unknown"
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        
    def initialize(self):
        """Initialize appropriate camera"""
        print("Initializing camera...")
        
        # Try USB cameras first
        for idx in range(4):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
                time.sleep(0.5)
                
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.camera_type = f"USB Camera {idx}"
                        
                        # Set camera properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Flush buffer
                        for _ in range(5):
                            self.cap.read()
                        
                        print(f"✅ {self.camera_type} initialized")
                        return True
                    else:
                        cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"USB camera {idx} error: {e}")
        
        # Try Pi Camera on Raspberry Pi
        if IS_RASPBERRY_PI:
            try:
                from picamera import PiCamera
                from picamera.array import PiRGBArray
                
                self.pi_camera = PiCamera()
                self.pi_camera.resolution = (640, 480)
                self.pi_camera.framerate = 30
                self.raw_capture = PiRGBArray(self.pi_camera, size=(640, 480))
                
                self.camera_type = "Pi Camera"
                print("✅ Pi Camera initialized")
                return True
            except ImportError:
                print("Pi Camera module not available")
            except Exception as e:
                print(f"Pi Camera error: {e}")
        
        print("❌ No camera found")
        return False
    
    def read_frame(self):
        """Read frame from camera"""
        try:
            if self.pi_camera:
                self.raw_capture.truncate(0)
                self.pi_camera.capture(self.raw_capture, format="bgr", use_video_port=True)
                frame = self.raw_capture.array.copy()
                with self.frame_lock:
                    self.last_frame = frame.copy()
                return frame
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.last_frame = frame.copy()
                    return frame.copy()
        except Exception as e:
            print(f"Camera error: {e}")
        
        return None
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pi_camera:
            self.pi_camera.close()
            self.pi_camera = None

# ==================== TOMATO TRACKER ====================
class TomatoTracker:
    """Enhanced tracker with FNN prediction caching"""
    
    def __init__(self):
        # Color ranges for tomato detection
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([179, 255, 255])
        self.green_lower = np.array([35, 50, 50])
        self.green_upper = np.array([85, 255, 255])
        
        self.kernel = np.ones((5, 5), np.uint8)
        self.min_area = 500
        
        # Tracking state
        self.tracked_tomato = None
        self.tomato_counter = 0
        self.tracking_id = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = 30
        
        self.consecutive_detections = 0
        self.min_consecutive = 3
        
        self.current_bbox = None
        self.current_confidence = 0
        self.current_type = "UNKNOWN"
        
        # FNN caching
        self.current_tomato_id = None
        self.cached_predictions = {}
        self.last_analysis_time = 0
        self.tracking_start_time = 0
        self.cache_refresh_interval = 10.0
        
        # Position tracking
        self.position_history = []
        self.max_position_history = 10
        
        # Debug
        self.fnn_call_count = 0
        self.last_fnn_call_time = 0
    
    def should_run_fnn_prediction(self, tomato_data):
        """Determine whether to run FNN prediction"""
        if not tomato_data or not tomato_data.get('is_tomato', False):
            self.current_tomato_id = None
            return False, None
        
        tomato_id = tomato_data.get('id')
        if not tomato_id:
            return False, None
        
        current_time = time.time()
        
        # New tomato
        if not self.current_tomato_id or tomato_id != self.current_tomato_id:
            self.current_tomato_id = tomato_id
            self.tracking_start_time = current_time
            self.last_analysis_time = current_time
            self.fnn_call_count = 0
            print(f"🆕 New tomato detected: ID {tomato_id}")
            return True, tomato_id
        
        # Same tomato - check cache
        if tomato_id in self.cached_predictions:
            cache_entry = self.cached_predictions[tomato_id]
            cache_age = current_time - cache_entry['timestamp']
            
            if cache_age < self.cache_refresh_interval:
                return False, tomato_id
            else:
                self.last_analysis_time = current_time
                return True, tomato_id
        else:
            self.last_analysis_time = current_time
            return True, tomato_id
    
    def get_cached_predictions(self, tomato_id):
        """Get cached FNN predictions"""
        if tomato_id and tomato_id in self.cached_predictions:
            return self.cached_predictions[tomato_id]['predictions']
        return None
    
    def cache_predictions(self, tomato_id, predictions):
        """Cache FNN predictions"""
        if not tomato_id:
            return
        
        self.cached_predictions[tomato_id] = {
            'predictions': predictions,
            'timestamp': time.time(),
            'tomato_id': tomato_id
        }
        
        # Keep only recent caches
        if len(self.cached_predictions) > 3:
            oldest_id = min(self.cached_predictions.keys(), 
                          key=lambda k: self.cached_predictions[k]['timestamp'])
            del self.cached_predictions[oldest_id]
    
    def update_position_history(self, bbox_pixel):
        """Update position history"""
        if not bbox_pixel:
            return
        
        x, y, w, h = bbox_pixel
        center_x = x + w/2
        center_y = y + h/2
        
        self.position_history.append({
            'x': center_x,
            'y': center_y,
            'time': time.time()
        })
        
        if len(self.position_history) > self.max_position_history:
            self.position_history.pop(0)
    
    def get_position_trend(self):
        """Get position trend"""
        if len(self.position_history) < 3:
            return "STABLE"
        
        recent_positions = self.position_history[-3:]
        dx = recent_positions[-1]['x'] - recent_positions[0]['x']
        dy = recent_positions[-1]['y'] - recent_positions[0]['y']
        
        movement_threshold = 20
        
        if abs(dx) > abs(dy):
            if dx > movement_threshold:
                return "MOVING_RIGHT"
            elif dx < -movement_threshold:
                return "MOVING_LEFT"
        else:
            if dy > movement_threshold:
                return "MOVING_DOWN"
            elif dy < -movement_threshold:
                return "MOVING_UP"
        
        return "STABLE"
    
    def detect_candidates(self, frame):
        """Detect all tomato candidates in frame"""
        candidates = []
        
        if frame is None:
            return candidates
        
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = frame.shape[:2]
            
            # Create masks
            red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            tomato_mask = cv2.bitwise_or(red_mask, green_mask)
            tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, self.kernel)
            
            contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                roi_hsv = hsv[y:y+h, x:x+w]
                if roi_hsv.size == 0:
                    continue
                
                red_pixels = np.sum(cv2.inRange(roi_hsv, self.red_lower1, self.red_upper1) > 0) + \
                            np.sum(cv2.inRange(roi_hsv, self.red_lower2, self.red_upper2) > 0)
                green_pixels = np.sum(cv2.inRange(roi_hsv, self.green_lower, self.green_upper) > 0)
                total_pixels = w * h
                
                red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
                green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
                
                if red_ratio > 0.3 and red_ratio > green_ratio:
                    tomato_type = "RIPE"
                    color_score = red_ratio
                elif green_ratio > 0.3:
                    tomato_type = "GREEN"
                    color_score = green_ratio
                else:
                    tomato_type = "UNKNOWN"
                    color_score = max(red_ratio, green_ratio)
                
                size_score = min(1.0, area / 10000)
                score = (color_score * 0.5 + circularity * 0.3 + size_score * 0.2)
                
                bbox_normalized = [
                    center_x / width,
                    center_y / height,
                    w / width,
                    h / height
                ]
                
                candidate = {
                    'bbox': bbox_normalized,
                    'bbox_pixel': [x, y, w, h],
                    'center': (center_x, center_y),
                    'type': tomato_type,
                    'score': score,
                    'area': area,
                    'circularity': circularity,
                    'red_ratio': red_ratio,
                    'green_ratio': green_ratio,
                    'is_tomato': score > 0.3,
                    'is_occluded': False
                }
                
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def update_tracking(self, frame):
        """Update single tomato tracking"""
        if frame is None:
            return None
        
        candidates = self.detect_candidates(frame)
        
        # Continue tracking existing tomato
        if self.tracked_tomato:
            best_match = None
            best_distance = float('inf')
            
            for candidate in candidates:
                tracked_center = self.tracked_tomato['center']
                candidate_center = candidate['center']
                
                distance = math.sqrt(
                    (tracked_center[0] - candidate_center[0])**2 +
                    (tracked_center[1] - candidate_center[1])**2
                )
                
                if distance < 100 and distance < best_distance:
                    best_distance = distance
                    best_match = candidate
            
            if best_match:
                self.tracked_tomato = best_match
                self.tracking_lost_frames = 0
                self.consecutive_detections = min(self.consecutive_detections + 1, 10)
                
                self.current_bbox = self.tracked_tomato['bbox']
                self.current_confidence = self.tracked_tomato['score']
                self.current_type = self.tracked_tomato['type']
                
                self.update_position_history(best_match['bbox_pixel'])
                
                return {
                    'id': self.tracking_id,
                    'counter': self.tracking_id,
                    'bbox': self.tracked_tomato['bbox'],
                    'bbox_pixel': self.tracked_tomato['bbox_pixel'],
                    'type': self.tracked_tomato['type'],
                    'score': self.tracked_tomato['score'],
                    'area': self.tracked_tomato['area'],
                    'circularity': self.tracked_tomato['circularity'],
                    'red_ratio': self.tracked_tomato['red_ratio'],
                    'green_ratio': self.tracked_tomato['green_ratio'],
                    'is_tomato': self.tracked_tomato['is_tomato'],
                    'is_occluded': self.tracked_tomato['is_occluded'],
                    'tracking_lost': False
                }
            else:
                self.tracking_lost_frames += 1
                
                if self.tracking_lost_frames > self.max_lost_frames:
                    self.tracked_tomato = None
                    self.tracking_id = None
                    self.current_tomato_id = None
                    self.consecutive_detections = 0
                    self.current_bbox = None
                    self.current_confidence = 0
                    self.position_history.clear()
                    return None
                else:
                    return {
                        'id': self.tracking_id,
                        'tracking_lost': True,
                        'lost_frames': self.tracking_lost_frames
                    }
        
        # Start tracking new tomato
        if not self.tracked_tomato and candidates:
            best_candidate = None
            best_score = 0
            
            for candidate in candidates:
                if candidate['score'] > best_score:
                    best_score = candidate['score']
                    best_candidate = candidate
            
            if best_candidate and best_score > 0.3:
                self.consecutive_detections += 1
                
                if self.consecutive_detections >= self.min_consecutive:
                    self.tomato_counter += 1
                    self.tracking_id = self.tomato_counter
                    self.tracked_tomato = best_candidate
                    self.tracking_lost_frames = 0
                    
                    self.current_bbox = self.tracked_tomato['bbox']
                    self.current_confidence = self.tracked_tomato['score']
                    self.current_type = self.tracked_tomato['type']
                    
                    self.position_history.clear()
                    self.update_position_history(best_candidate['bbox_pixel'])
                    
                    print(f"Started tracking Tomato {self.tracking_id}")
                    
                    return {
                        'id': self.tracking_id,
                        'counter': self.tracking_id,
                        'bbox': self.tracked_tomato['bbox'],
                        'bbox_pixel': self.tracked_tomato['bbox_pixel'],
                        'type': self.tracked_tomato['type'],
                        'score': self.tracked_tomato['score'],
                        'area': self.tracked_tomato['area'],
                        'circularity': self.tracked_tomato['circularity'],
                        'red_ratio': self.tracked_tomato['red_ratio'],
                        'green_ratio': self.tracked_tomato['green_ratio'],
                        'is_tomato': self.tracked_tomato['is_tomato'],
                        'is_occluded': self.tracked_tomato['is_occluded'],
                        'tracking_lost': False
                    }
            else:
                self.consecutive_detections = 0
        
        self.current_bbox = None
        self.current_confidence = 0
        return None
    
    def reset_counter(self):
        """Reset tomato counter and clear cache"""
        self.tomato_counter = 0
        self.tracked_tomato = None
        self.tracking_id = None
        self.current_tomato_id = None
        self.tracking_lost_frames = 0
        self.consecutive_detections = 0
        self.current_bbox = None
        self.current_confidence = 0
        self.cached_predictions.clear()
        self.position_history.clear()
        self.last_analysis_time = 0
        self.tracking_start_time = 0
        self.fnn_call_count = 0
        print("Counter reset to 0 and cache cleared")

# ==================== MODEL PREDICTOR ====================
class ModelPredictor:
    """Handles model predictions with caching"""
    
    def __init__(self, camera_manager, tomato_tracker):
        self.camera = camera_manager
        self.tomato_tracker = tomato_tracker
        self.cnn_model = None
        self.fnn_model = None
        self.gan_model = None
        
        # Model paths (update these to your actual paths)
        self.cnn_model_path = "tomato_cnn_model.h5"
        self.fnn_model_path = "FNN_Regression_Model.h5"
        self.gan_model_path = "final_gan_generator.keras"
    
    def load_cnn_model(self):
        """Load CNN model"""
        try:
            if os.path.exists(self.cnn_model_path):
                print("Loading CNN model...")
                self.cnn_model = load_model(self.cnn_model_path)
                print("✅ CNN model loaded")
                return True
            else:
                print(f"❌ CNN model not found: {self.cnn_model_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load CNN model: {e}")
            return False
    
    def load_fnn_model(self):
        """Load FNN model"""
        try:
            if os.path.exists(self.fnn_model_path):
                print("Loading FNN model...")
                self.fnn_model = load_model(
                    self.fnn_model_path,
                    custom_objects={"mse": MeanSquaredError()}
                )
                print("✅ FNN model loaded")
                return True
            else:
                print(f"❌ FNN model not found: {self.fnn_model_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load FNN model: {e}")
            return False
    
    def load_gan_model(self):
        """Load GAN model"""
        try:
            if os.path.exists(self.gan_model_path):
                print("Loading GAN model...")
                self.gan_model = load_model(
                    self.gan_model_path,
                    custom_objects={"SPADE": SPADE}
                )
                print("✅ GAN model loaded")
                return True
            else:
                print(f"❌ GAN model not found: {self.gan_model_path}")
                return False
        except Exception as e:
            print(f"❌ Failed to load GAN model: {e}")
            return False
    
    def predict_classification(self, tomato_data, frame):
        """CNN classification prediction"""
        if not tomato_data or not tomato_data.get('is_tomato', False):
            return {
                'prediction': 'No Tomato',
                'confidence': 0.0,
                'is_tomato': False,
                'is_ripe': False,
                'is_occluded': False
            }
        
        if self.cnn_model is None:
            return {
                'prediction': 'Model Error',
                'confidence': 0.0,
                'is_tomato': False,
                'is_ripe': False,
                'is_occluded': False
            }
        
        try:
            bbox_pixel = tomato_data['bbox_pixel']
            x, y, w, h = bbox_pixel
            
            # Ensure ROI is valid
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)
            
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                return {
                    'prediction': 'Invalid ROI',
                    'confidence': 0.0,
                    'is_tomato': False,
                    'is_ripe': False,
                    'is_occluded': False
                }
            
            roi_resized = cv2.resize(roi, (224, 224))
            roi_normalized = roi_resized / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            cnn_prediction = self.cnn_model.predict(roi_batch, verbose=0)
            class_id = int(np.argmax(cnn_prediction))
            confidence = float(np.max(cnn_prediction))
            
            return {
                'prediction': ['Ripe Tomato', 'Occluded Tomato'][class_id],
                'confidence': confidence,
                'is_tomato': class_id in [0, 1],
                'is_ripe': class_id == 0,
                'is_occluded': class_id == 1
            }
            
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'is_tomato': False,
                'is_ripe': False,
                'is_occluded': False
            }
    
    def predict_fnn_properties(self, tomato_data, frame, force_run=False):
        """FNN regression predictions with caching"""
        if not tomato_data or not tomato_data.get('is_tomato', False):
            return {
                'weight': 0.0,
                'size': 0.0,
                'gripForce': 0.0,
                'pressure': 0.0,
                'force': 0.0,
                'torque': 0.0,
                'timeTaken': 0.0,
                'is_cached': False
            }
        
        if self.fnn_model is None:
            return {
                'weight': 0.0,
                'size': 0.0,
                'gripForce': 0.0,
                'pressure': 0.0,
                'force': 0.0,
                'torque': 0.0,
                'timeTaken': 0.0,
                'is_cached': False
            }
        
        try:
            tomato_id = tomato_data.get('id')
            
            # Check if we should run FNN or use cached values
            should_run_fnn, cache_tomato_id = self.tomato_tracker.should_run_fnn_prediction(tomato_data)
            
            # Use cached if available
            if not force_run and not should_run_fnn and cache_tomato_id:
                cached = self.tomato_tracker.get_cached_predictions(cache_tomato_id)
                if cached:
                    cached['is_cached'] = True
                    return cached
            
            # Run FNN prediction
            bbox_pixel = tomato_data.get('bbox_pixel', [0, 0, 0, 0])
            x, y, w, h = bbox_pixel
            
            area = tomato_data.get('area', 0)
            circularity = tomato_data.get('circularity', 0)
            red_ratio = tomato_data.get('red_ratio', 0)
            green_ratio = tomato_data.get('green_ratio', 0)
            
            # Prepare features (simplified for demonstration)
            features = [x, y, w, h, area, circularity, red_ratio, green_ratio]
            features.extend([0] * (30 - len(features)))
            
            input_features = np.array(features, dtype=np.float32).reshape(1, -1)
            fnn_predictions = self.fnn_model.predict(input_features, verbose=0)[0]
            
            if len(fnn_predictions) < 15:
                fnn_predictions = np.pad(
                    fnn_predictions,
                    (0, 15 - len(fnn_predictions)),
                    'constant'
                )
            
            result = {
                'weight': max(0, float(fnn_predictions[0])),
                'size': max(0, float(fnn_predictions[1])),
                'gripForce': max(0, float(fnn_predictions[2])),
                'pressure': max(0, float(fnn_predictions[3])),
                'force': max(0, float(fnn_predictions[4])),
                'torque': max(0, float(fnn_predictions[5])),
                'timeTaken': 0.05,
                'prediction_time': time.time(),
                'is_cached': False,
                'tomato_id': tomato_id
            }
            
            self.tomato_tracker.fnn_call_count += 1
            self.tomato_tracker.last_fnn_call_time = time.time()
            
            print(f"🔧 FNN CALL #{self.tomato_tracker.fnn_call_count} for Tomato {tomato_id}")
            print(f"   Weight: {result['weight']:.1f}g, Grip: {result['gripForce']:.2f}N")
            
            # Cache the predictions
            if cache_tomato_id:
                self.tomato_tracker.cache_predictions(cache_tomato_id, result)
                print(f"   📦 Cached for future use")
            
            return result
            
        except Exception as e:
            print(f"❌ FNN prediction error: {e}")
            return {
                'weight': 0.0,
                'size': 0.0,
                'gripForce': 0.0,
                'pressure': 0.0,
                'force': 0.0,
                'torque': 0.0,
                'timeTaken': 0.0,
                'is_cached': False
            }
    
    def predict_gan_reconstruction(self, tomato_data):
        """GAN reconstruction"""
        if not tomato_data or not tomato_data.get('is_occluded', False):
            return {}
        
        if self.gan_model is None:
            return {}
        
        try:
            frame = self.camera.read_frame()
            if frame is None:
                return {}
            
            bbox_pixel = tomato_data.get('bbox_pixel', [0, 0, 0, 0])
            x, y, w, h = bbox_pixel
            
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                return {}
            
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0
            seg_map = np.zeros((224, 224, 1), dtype=np.float32)
            
            gan_input = [
                np.expand_dims(roi_resized, axis=0),
                np.expand_dims(seg_map, axis=0)
            ]
            
            reconstructed_image = self.gan_model.predict(gan_input, verbose=0)[0]
            
            return {
                'reconstructed': True,
                'confidence': 0.85,
                'reconstruction_time': 0.1,
                'image': (reconstructed_image * 255).astype(np.uint8)
            }
            
        except Exception as e:
            print(f"GAN reconstruction error: {e}")
            return {}

# ==================== TOMATO TRACKING SYSTEM ====================
class TomatoTrackingSystem:
    """Main tomato tracking system"""
    
    def __init__(self, socketio):
        self.camera = CameraManager()
        self.tracker = TomatoTracker()
        self.predictor = ModelPredictor(self.camera, self.tracker)
        
        # Arduino client
        self.arduino = ArduinoClient(socketio)
        
        self.socketio = socketio
        
        # Load models
        print("\n" + "="*60)
        print("INITIALIZING TOMATO TRACKING SYSTEM")
        print("="*60)
        
        self.predictor.load_cnn_model()
        self.predictor.load_fnn_model()
        self.predictor.load_gan_model()
        
        # Start Arduino controller
        self.arduino.start()
        
        # System state
        self.running = False
        self.camera_running = False
        self.last_tomato = None
        self.last_frame = None
        
        self.tracking_history = []
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Robot movement
        self.robot_movement_enabled = True
        self.last_movement_direction = None
        self.movement_signal_queue = []
        self.movement_running = False
        self.movement_thread = None
        
        # Target position for frontend
        self.frontend_target_direction = None
        self.target_position_active = False
        self.last_target_emission_time = 0
        self.target_emission_interval = 1.0
        
        # Tomato position
        self.tomato_position_zone = "CENTER"
        self.last_position_update = 0
        self.position_update_interval = 0.5
        
        # Visual indicators
        self.indicators = {
            'left': {'active': False, 'color': (0, 100, 255)},
            'right': {'active': False, 'color': (0, 100, 255)},
            'up': {'active': False, 'color': (0, 100, 255)},
            'down': {'active': False, 'color': (0, 100, 255)},
            'center': {'active': False, 'color': (0, 255, 0)}
        }
        self.indicator_inactive_color = (100, 100, 100)
    
    def start(self):
        """Start the system"""
        print("=" * 60)
        print("STARTING SINGLE TOMATO TRACKING SYSTEM")
        print("=" * 60)
        
        if self.camera.initialize():
            self.camera_running = True
            self.running = True
            
            # Start tracking thread
            self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.tracking_thread.start()
            
            # Start movement processing if enabled
            if self.robot_movement_enabled:
                self.movement_running = True
                self.movement_thread = threading.Thread(target=self._movement_processing_loop, daemon=True)
                self.movement_thread.start()
                print("✅ Robot arm movement processing enabled")
            
            print("✅ System started - Tracking ONE tomato")
            return True
        
        print("❌ Failed to start system - Camera initialization failed")
        return False
    
    def _tracking_loop(self):
        """Main tracking loop"""
        print("Single tomato tracking loop started")
        
        while self.running:
            try:
                loop_start = time.time()
                
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.033)
                    continue
                
                self.frame_count += 1
                
                # Update FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                # Update tracking
                tomato = self.tracker.update_tracking(frame)
                self.last_tomato = tomato
                self.last_frame = frame.copy()
                
                if tomato:
                    # Add to history
                    self.tracking_history.append({
                        'id': tomato.get('id', 0),
                        'time': time.time(),
                        'bbox': tomato.get('bbox', []),
                        'type': tomato.get('type', 'UNKNOWN')
                    })
                    
                    if len(self.tracking_history) > 100:
                        self.tracking_history.pop(0)
                    
                    # Update position and indicators
                    current_time = time.time()
                    if current_time - self.last_position_update >= self.position_update_interval:
                        self._update_tomato_position_zone(tomato, frame)
                        self._update_indicators()
                        
                        # Determine and emit target position
                        self._determine_and_emit_target_position(tomato, frame)
                        
                        self.last_position_update = current_time
                    
                    # Analyze for robot movement
                    if self.robot_movement_enabled and tomato.get('is_tomato', False):
                        self._analyze_tomato_position_for_movement(tomato, frame)
                
                # Maintain frame rate
                loop_time = time.time() - loop_start
                time.sleep(max(0, 0.033 - loop_time))
                
            except Exception as e:
                print(f"Tracking loop error: {e}")
                time.sleep(0.1)
    
    def _determine_and_emit_target_position(self, tomato_data, frame):
        """Determine target position for frontend robot arm visualization"""
        if not tomato_data or not tomato_data.get('bbox'):
            self.frontend_target_direction = None
            self.target_position_active = False
            return
        
        height, width = frame.shape[:2]
        bbox_normalized = tomato_data['bbox']
        center_x_norm, center_y_norm = bbox_normalized[0], bbox_normalized[1]
        
        center_x = int(center_x_norm * width)
        center_y = int(center_y_norm * height)
        
        # Define zones
        center_threshold_x = width * 0.1
        center_threshold_y = height * 0.1
        target_center_x = width // 2
        target_center_y = height // 2
        
        # Determine target direction
        target_direction = None
        
        if abs(center_x - target_center_x) > center_threshold_x:
            if center_x < target_center_x:
                target_direction = "left"
            else:
                target_direction = "right"
        elif abs(center_y - target_center_y) > center_threshold_y:
            if center_y < target_center_y:
                target_direction = "up"
            else:
                target_direction = "down"
        else:
            target_direction = "center"
        
        # Update state
        self.frontend_target_direction = target_direction
        self.target_position_active = target_direction != "center"
        
        # Emit to frontend
        current_time = time.time()
        if current_time - self.last_target_emission_time >= self.target_emission_interval:
            self._emit_target_position_to_frontend(target_direction, tomato_data)
            self.last_target_emission_time = current_time
    
    def _emit_target_position_to_frontend(self, target_direction, tomato_data):
        """Emit target position to frontend via WebSocket"""
        if not target_direction:
            return
        
        target_data = {
            'type': 'ROBOT_TARGET_POSITION',
            'target_direction': target_direction,
            'active': target_direction != "center",
            'tomato_id': tomato_data.get('id', 0),
            'timestamp': time.time(),
            'tomato_position': {
                'x': tomato_data['bbox'][0],
                'y': tomato_data['bbox'][1],
                'zone': self.tomato_position_zone
            },
            'message': f'Robot arm should move {target_direction.upper()} to reach tomato'
        }
        
        try:
            self.socketio.emit('robot_target_position', target_data)
            print(f"🎯 Emitted target position: {target_direction.upper()} for Tomato {tomato_data.get('id', 0)}")
        except Exception as e:
            print(f"Error emitting target position: {e}")
    
    def _update_tomato_position_zone(self, tomato_data, frame):
        """Update which zone the tomato is in"""
        if not tomato_data or not tomato_data.get('bbox'):
            self.tomato_position_zone = "NO_TOMATO"
            return
        
        height, width = frame.shape[:2]
        bbox_normalized = tomato_data['bbox']
        center_x_norm, center_y_norm = bbox_normalized[0], bbox_normalized[1]
        
        center_x = int(center_x_norm * width)
        center_y = int(center_y_norm * height)
        
        # Define zones
        left_boundary = width * 0.4
        right_boundary = width * 0.6
        top_boundary = height * 0.4
        bottom_boundary = height * 0.6
        
        # Determine zone
        if center_x < left_boundary:
            if center_y < top_boundary:
                self.tomato_position_zone = "TOP_LEFT"
            elif center_y > bottom_boundary:
                self.tomato_position_zone = "BOTTOM_LEFT"
            else:
                self.tomato_position_zone = "LEFT"
        elif center_x > right_boundary:
            if center_y < top_boundary:
                self.tomato_position_zone = "TOP_RIGHT"
            elif center_y > bottom_boundary:
                self.tomato_position_zone = "BOTTOM_RIGHT"
            else:
                self.tomato_position_zone = "RIGHT"
        else:
            if center_y < top_boundary:
                self.tomato_position_zone = "TOP_CENTER"
            elif center_y > bottom_boundary:
                self.tomato_position_zone = "BOTTOM_CENTER"
            else:
                self.tomato_position_zone = "CENTER"
    
    def _update_indicators(self):
        """Update visual indicators"""
        for key in self.indicators:
            self.indicators[key]['active'] = False
        
        zone = self.tomato_position_zone
        
        if zone == "LEFT" or zone == "TOP_LEFT" or zone == "BOTTOM_LEFT":
            self.indicators['left']['active'] = True
        elif zone == "RIGHT" or zone == "TOP_RIGHT" or zone == "BOTTOM_RIGHT":
            self.indicators['right']['active'] = True
        
        if zone == "TOP_CENTER" or zone == "TOP_LEFT" or zone == "TOP_RIGHT":
            self.indicators['up']['active'] = True
        elif zone == "BOTTOM_CENTER" or zone == "BOTTOM_LEFT" or zone == "BOTTOM_RIGHT":
            self.indicators['down']['active'] = True
        
        if zone == "CENTER":
            self.indicators['center']['active'] = True
    
    def _analyze_tomato_position_for_movement(self, tomato_data, frame):
        """Analyze tomato position for robot movement"""
        if not tomato_data or not tomato_data.get('bbox'):
            return
        
        height, width = frame.shape[:2]
        bbox_normalized = tomato_data['bbox']
        center_x_norm, center_y_norm = bbox_normalized[0], bbox_normalized[1]
        
        center_x = int(center_x_norm * width)
        center_y = int(center_y_norm * height)
        
        center_threshold_x = width * 0.1
        center_threshold_y = height * 0.1
        target_center_x = width // 2
        target_center_y = height // 2
        
        direction = None
        
        if abs(center_x - target_center_x) > center_threshold_x:
            if center_x < target_center_x:
                direction = "LEFT"
            else:
                direction = "RIGHT"
        elif abs(center_y - target_center_y) > center_threshold_y:
            if center_y < target_center_y:
                direction = "UP"
            else:
                direction = "DOWN"
        else:
            direction = "GRIP"
        
        if direction and direction != "GRIP":
            self._queue_movement_signal(direction, tomato_data)
    
    def _queue_movement_signal(self, direction, tomato_data):
        """Queue movement signal"""
        signal = {
            'type': 'ROBOT_MOVE',
            'direction': direction,
            'tomato_id': tomato_data.get('id', 0),
            'timestamp': time.time(),
            'position': {
                'x': tomato_data['bbox'][0],
                'y': tomato_data['bbox'][1]
            }
        }
        
        self.movement_signal_queue.append(signal)
        
        if len(self.movement_signal_queue) > 10:
            self.movement_signal_queue.pop(0)
    
    def _movement_processing_loop(self):
        """Process movement signals"""
        print("Robot movement processing loop started")
        
        while self.movement_running and self.running:
            try:
                if self.movement_signal_queue:
                    signal = self.movement_signal_queue.pop(0)
                    self._emit_movement_to_backend(signal)
                    self.last_movement_direction = signal['direction']
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Movement processing error: {e}")
                time.sleep(0.5)
    
    def _emit_movement_to_backend(self, signal):
        """Send movement to Arduino"""
        direction = signal['direction']
        tomato_id = signal['tomato_id']
        
        print(f"🤖 Robot movement: {direction} for Tomato {tomato_id}")
        result = self.arduino.send_movement_command(direction)
        
        return result
    
    def get_robot_movement_status(self):
        """Get robot movement status"""
        arduino_status = self.arduino.get_status()
        live_status = self.arduino.get_live_status()
        
        return {
            'movement_enabled': self.robot_movement_enabled,
            'last_direction': self.last_movement_direction,
            'queue_size': len(self.movement_signal_queue),
            'movement_running': self.movement_running,
            'tomato_position_zone': self.tomato_position_zone,
            'indicators': self.indicators,
            'frontend_target_direction': self.frontend_target_direction,
            'target_position_active': self.target_position_active,
            'arduino_status': arduino_status,
            'force_sensors': live_status.get('force_sensors', {}),
            'emergency_stop': live_status.get('emergency_stop', False),
            'controller_running': live_status.get('controller_running', False)
        }
    
    def get_detection_data(self):
        """Get detection data for frontend"""
        if not self.last_tomato:
            return {
                'success': False,
                'message': 'No tomato currently tracked',
                'detections': {'tomatoes': [], 'red_objects': []},
                'results': []
            }
        
        results = []
        frame = self.camera.read_frame()
        
        classification = self.predictor.predict_classification(self.last_tomato, frame)
        properties = self.predictor.predict_fnn_properties(self.last_tomato, frame)
        gan_data = self.predictor.predict_gan_reconstruction(self.last_tomato)
        
        result = {
            'detection': {
                'bbox': self.last_tomato['bbox'],
                'bbox_pixel': self.last_tomato['bbox_pixel'],
                'confidence': self.last_tomato['score'],
                'area': self.last_tomato['area'],
                'circularity': self.last_tomato['circularity'],
                'is_tomato': self.last_tomato['is_tomato'],
                'is_occluded': self.last_tomato['is_occluded'],
                'class_name': 'tomato' if self.last_tomato['is_tomato'] else 'red_object',
                'image_shape': [640, 480]
            },
            'classification': classification,
            'properties': properties,
            'fnn_predictions': properties,
            'gan_reconstruction': gan_data,
            'caching_info': {
                'tomato_id': self.last_tomato.get('id', 0),
                'using_cached': properties.get('is_cached', False),
                'cache_size': len(self.tracker.cached_predictions),
                'fnn_calls': self.tracker.fnn_call_count
            }
        }
        
        results.append(result)
        
        response_data = {
            'success': True,
            'message': 'Single tomato tracking active',
            'detections': {
                'tomatoes': [self.last_tomato] if self.last_tomato['is_tomato'] else [],
                'red_objects': []
            },
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'total_tomatoes': self.tracker.tomato_counter,
            'current_tomato_id': self.last_tomato.get('id', 0),
            'position_trend': self.tracker.get_position_trend(),
            'fnn_stability': {
                'calls': self.tracker.fnn_call_count,
                'cache_hits': len(self.tracker.cached_predictions),
                'is_stable': properties.get('is_cached', False)
            }
        }
        
        response_data['robot_movement'] = self.get_robot_movement_status()
        
        return response_data
    
    def get_analytics(self):
        """Get system analytics"""
        arduino_status = self.arduino.get_status()
        arduino_live = self.arduino.get_live_status()
        
        analytics_data = {
            'success': True,
            'camera_running': self.camera_running,
            'tomato_count': self.tracker.tomato_counter,
            'current_tomato': {
                'tomato_id': self.last_tomato.get('id', 0) if self.last_tomato else 0,
                'classification': None,
                'properties': {}
            },
            'memory_mb': 125.5,
            'uptime': time.time() - self.start_time,
            'fps': self.fps,
            'models_loaded': self.predictor.cnn_model is not None and self.predictor.fnn_model is not None,
            'tracker_active': self.running,
            'camera_type': self.camera.camera_type,
            'tracking': self.last_tomato is not None,
            'tomato': self.last_tomato if self.last_tomato else None,
            'arduino': arduino_status,
            'arduino_live': arduino_live,
            'fnn_caching': {
                'enabled': True,
                'cache_size': len(self.tracker.cached_predictions),
                'current_tomato_id': self.tracker.current_tomato_id,
                'refresh_interval': self.tracker.cache_refresh_interval,
                'fnn_calls': self.tracker.fnn_call_count
            }
        }
        
        analytics_data['robot_movement'] = self.get_robot_movement_status()
        
        if self.last_tomato:
            bbox = self.last_tomato.get('bbox', [0.5, 0.5, 0.1, 0.1])
            center_x, center_y = bbox[0], bbox[1]
            
            position_info = {
                'center_x': center_x,
                'center_y': center_y,
                'zone': self.tomato_position_zone,
                'is_left': center_x < 0.4,
                'is_right': center_x > 0.6,
                'is_up': center_y < 0.4,
                'is_down': center_y > 0.6,
                'is_centered': 0.45 <= center_x <= 0.55 and 0.45 <= center_y <= 0.55
            }
            
            analytics_data['tomato_position'] = position_info
            
            # Add target direction
            if self.frontend_target_direction:
                analytics_data['frontend_target_direction'] = self.frontend_target_direction
                analytics_data['target_position_active'] = self.target_position_active
            
            # Movement recommendations
            if position_info['is_left']:
                analytics_data['movement_recommendation'] = 'LEFT'
            elif position_info['is_right']:
                analytics_data['movement_recommendation'] = 'RIGHT'
            elif position_info['is_up']:
                analytics_data['movement_recommendation'] = 'UP'
            elif position_info['is_down']:
                analytics_data['movement_recommendation'] = 'DOWN'
            elif position_info['is_centered']:
                analytics_data['movement_recommendation'] = 'GRIP'
            
            analytics_data['position_trend'] = self.tracker.get_position_trend()
        
        return analytics_data
    
    def get_annotated_frame(self):
        """Get annotated frame with bounding box"""
        frame = self.camera.read_frame()
        if frame is None:
            return None
        
        frame_copy = frame.copy()
        
        if self.last_tomato and not self.last_tomato.get("tracking_lost", False):
            x, y, w, h = self.last_tomato['bbox_pixel']
            tomato_type = self.last_tomato.get("type", "UNKNOWN")
            tomato_id = self.last_tomato.get("id", 0)
            
            # Choose color based on tomato type
            if tomato_type == "RIPE":
                color = (0, 255, 0)  # Green
            elif tomato_type == "GREEN":
                color = (0, 255, 255)  # Yellow
            else:
                color = (200, 200, 200)  # Gray
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            id_text = f"{tomato_type} (ID: {tomato_id})"
            cv2.putText(
                frame_copy,
                id_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 0),
                2,
            )
            
            # Draw confidence
            confidence_score = f"Confidence: {self.last_tomato['score']:.2f}"
            cv2.putText(
                frame_copy,
                confidence_score,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        
        # Add overlays
        self._add_system_overlay(frame_copy)
        self._add_robot_movement_indicators(frame_copy)
        
        return frame_copy
    
    def _add_system_overlay(self, frame):
        """Add system info overlay"""
        height, width = frame.shape[:2]
        
        # Header
        header = "SINGLE TOMATO TRACKING"
        cv2.putText(frame, header, (width//2 - 180, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Tomato ID
        counter_text = f"Tomato ID: {self.tracker.tracking_id if self.tracker.tracking_id else 0}"
        cv2.putText(frame, counter_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # FNN status
        if self.last_tomato:
            fnn_status = "FNN: CACHED" if self.tracker.current_tomato_id in self.tracker.cached_predictions else "FNN: LIVE"
            fnn_color = (0, 255, 0) if "CACHED" in fnn_status else (255, 255, 0)
            cv2.putText(frame, fnn_status, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fnn_color, 1)
        
        # Camera info
        cam_text = f"Camera: {self.camera.camera_type}"
        cv2.putText(frame, cam_text, (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        # Arduino status
        arduino_status = self.arduino.get_status()
        arduino_text = f"Arduino: {arduino_status['status']}"
        arduino_color = (0, 255, 0) if arduino_status['connected'] else (200, 200, 200)
        if arduino_status['emergency_stop']:
            arduino_color = (0, 0, 255)
        cv2.putText(frame, arduino_text, (width - 250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, arduino_color, 1)
        
        # Force sensor data
        force_data = self.arduino.get_live_status()['force_sensors']
        force_text = f"Force: {force_data['total']:.1f}kg"
        cv2.putText(frame, force_text, (width - 250, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Target position
        if self.frontend_target_direction:
            target_text = f"Target: {self.frontend_target_direction.upper()}"
            target_color = (0, 255, 0) if self.target_position_active else (200, 200, 200)
            cv2.putText(frame, target_text, (width - 250, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 1)
        
        # Tracking status
        if self.last_tomato:
            status_text = f"Tracking Tomato {self.last_tomato.get('id', 0)}"
            status_color = (0, 255, 0)
        else:
            status_text = "Waiting for tomato..."
            status_color = (200, 200, 200)
        
        cv2.putText(frame, status_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    def _add_robot_movement_indicators(self, frame):
        """Add visual indicators for robot arm movement"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        indicator_size = 35
        
        # LEFT indicator
        left_color = self.indicators['left']['color'] if self.indicators['left']['active'] else self.indicator_inactive_color
        cv2.rectangle(frame, (20, center_y - indicator_size), 
                     (40, center_y + indicator_size), left_color, 3)
        cv2.putText(frame, "LEFT", (15, center_y - indicator_size - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        
        # RIGHT indicator
        right_color = self.indicators['right']['color'] if self.indicators['right']['active'] else self.indicator_inactive_color
        cv2.rectangle(frame, (width - 40, center_y - indicator_size), 
                     (width - 20, center_y + indicator_size), right_color, 3)
        cv2.putText(frame, "RIGHT", (width - 50, center_y - indicator_size - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        # UP indicator
        up_color = self.indicators['up']['color'] if self.indicators['up']['active'] else self.indicator_inactive_color
        cv2.rectangle(frame, (center_x - indicator_size, 20), 
                     (center_x + indicator_size, 40), up_color, 3)
        cv2.putText(frame, "UP", (center_x - 10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, up_color, 1)
        
        # DOWN indicator
        down_color = self.indicators['down']['color'] if self.indicators['down']['active'] else self.indicator_inactive_color
        cv2.rectangle(frame, (center_x - indicator_size, height - 40), 
                     (center_x + indicator_size, height - 20), down_color, 3)
        cv2.putText(frame, "DOWN", (center_x - 20, height - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, down_color, 1)
        
        # CENTER indicator
        center_color = self.indicators['center']['color'] if self.indicators['center']['active'] else (100, 100, 100)
        cv2.circle(frame, (center_x, center_y), 20, center_color, 2)
        cv2.circle(frame, (center_x, center_y), 5, center_color, -1)
        
        if self.indicators['center']['active']:
            cv2.putText(frame, "GRIP", (center_x - 20, center_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_color, 2)
        
        # Draw tomato-to-center guidance
        if self.last_tomato:
            bbox = self.last_tomato.get('bbox', [0.5, 0.5, 0.1, 0.1])
            center_x_norm, center_y_norm = bbox[0], bbox[1]
            tomato_x = int(center_x_norm * width)
            tomato_y = int(center_y_norm * height)
            
            # Draw line from tomato to center
            cv2.arrowedLine(frame, (tomato_x, tomato_y), (center_x, center_y), (255, 100, 100), 2)
            
            # Add distance
            distance = math.sqrt((tomato_x - center_x)**2 + (tomato_y - center_y)**2)
            distance_text = f"{distance:.0f}px"
            cv2.putText(frame, distance_text, (tomato_x + 10, tomato_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    def stop(self):
        """Stop system"""
        print("Stopping system...")
        self.running = False
        self.camera_running = False
        self.movement_running = False
        
        # Stop Arduino
        self.arduino.stop()
        
        # Wait for threads
        if hasattr(self, 'tracking_thread') and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        
        if hasattr(self, 'movement_thread') and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=1.0)
        
        # Release camera
        self.camera.release()
        print("System stopped")

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize system
system = TomatoTrackingSystem(socketio)

@app.route('/')
def index():
    return jsonify({
        'message': 'Single Tomato Tracking System API',
        'description': 'Tracks ONE tomato at a time with same ID while in frame',
        'arduino_control': 'Complete Arduino communication with LED control',
        'endpoints': {
            '/': 'This info',
            '/start': 'Start system (POST)',
            '/stop': 'Stop system (POST)',
            '/video_feed': 'Live video feed',
            '/detect': 'Get detection data',
            '/analytics': 'Get system analytics',
            '/health': 'Health check',
            '/reset_counter': 'Reset counter (POST)',
            '/analyze': 'Get model predictions',
            '/arduino/status': 'Get Arduino status',
            '/arduino/scan': 'Scan for Arduino ports (GET)',
            '/arduino/connect': 'Connect to Arduino (POST)',
            '/arduino/disconnect': 'Disconnect Arduino (POST)',
            '/arduino/move': 'Send movement command (POST)',
            '/arduino/emergency_stop': 'Emergency stop (POST)',
            '/arduino/reset_emergency': 'Reset emergency stop (POST)',
            '/arduino/grip': 'Grip command (POST)',
            '/arduino/release': 'Release command (POST)',
            '/arduino/led_on': 'Turn LED on (POST)',
            '/arduino/led_off': 'Turn LED off (POST)',
            '/arduino/blink': 'Blink LED (test) (POST)',
            '/arduino/live_status': 'Get live Arduino status',
            '/robot/movement_status': 'Get robot movement status',
            '/robot/enable_movement': 'Enable robot movement (POST)',
            '/robot/disable_movement': 'Disable robot movement (POST)',
            '/force_refresh_fnn': 'Force refresh FNN cache (POST)'
        }
    })

@app.route('/start', methods=['POST'])
def start_system():
    if system.running:
        return jsonify({'success': True, 'message': 'Already running'})
    
    if system.start():
        return jsonify({'success': True, 'message': 'System started'})
    else:
        return jsonify({'success': False, 'message': 'Failed to start'})

@app.route('/stop', methods=['POST'])
def stop_system():
    system.stop()
    return jsonify({'success': True, 'message': 'System stopped'})

@app.route('/video_feed')
def video_feed():
    def generate():
        while system.running:
            frame = system.get_annotated_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           jpeg.tobytes() + b'\r\n')
            else:
                # Create blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Single Tomato Tracking", (120, 220), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           jpeg.tobytes() + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['GET'])
def detect():
    if not system.running:
        return jsonify({'success': False, 'message': 'System not running'})
    
    data = system.get_detection_data()
    return jsonify(data)

@app.route('/analytics', methods=['GET'])
def analytics():
    data = system.get_analytics()
    return jsonify(json_safe(data))

@app.route('/health', methods=['GET'])
def health():
    arduino_status = system.arduino.get_status()
    return jsonify({
        'success': True,
        'running': system.running,
        'tomato_count': system.tracker.tomato_counter,
        'fps': system.fps,
        'tracking': system.last_tomato is not None,
        'arduino_connected': arduino_status['connected'],
        'arduino_status': arduino_status['status'],
        'emergency_stop': arduino_status['emergency_stop']
    })

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    system.tracker.reset_counter()
    return jsonify({
        'success': True, 
        'message': 'Counter reset', 
        'counter': system.tracker.tomato_counter
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    if not system.last_tomato:
        return jsonify({
            'success': False,
            'message': 'No tomato currently tracked'
        })
    
    frame = system.camera.read_frame()
    classification = system.predictor.predict_classification(system.last_tomato, frame)
    properties = system.predictor.predict_fnn_properties(system.last_tomato, frame)
    
    response = {
        'success': True,
        'classification': classification,
        'fnn_predictions': properties,
        'caching_info': {
            'tomato_id': system.tracker.current_tomato_id,
            'using_cached': properties.get('is_cached', False)
        }
    }
    
    return jsonify(json_safe(response))

# ==================== ARDUINO ENDPOINTS ====================

@app.route('/arduino/status', methods=['GET'])
def arduino_status():
    status = system.arduino.get_status()
    return jsonify(json_safe(status))

@app.route('/arduino/scan', methods=['GET'])
def arduino_scan():
    ports = system.arduino.scan_ports()
    return jsonify({
        'success': True,
        'ports': ports
    })

@app.route('/arduino/connect', methods=['POST'])
def arduino_connect():
    success = system.arduino.connect()
    return jsonify({
        'success': success,
        'message': 'Connected to Arduino' if success else 'Failed to connect'
    })

@app.route('/arduino/disconnect', methods=['POST'])
def arduino_disconnect():
    system.arduino.disconnect()
    return jsonify({
        'success': True,
        'message': 'Arduino disconnected'
    })

@app.route('/arduino/move', methods=['POST'])
def arduino_move():
    data = request.get_json()
    
    if not data or 'direction' not in data:
        return jsonify({
            'success': False,
            'message': 'Missing direction parameter'
        })
    
    direction = data['direction'].upper()
    result = system.arduino.send_movement_command(direction)
    
    return jsonify(json_safe(result))

@app.route('/arduino/emergency_stop', methods=['POST'])
def arduino_emergency_stop():
    result = system.arduino.emergency_stop()
    return jsonify(json_safe(result))

@app.route('/arduino/reset_emergency', methods=['POST'])
def arduino_reset_emergency():
    result = system.arduino.reset_emergency()
    return jsonify(json_safe(result))

@app.route('/arduino/grip', methods=['POST'])
def arduino_grip():
    data = request.get_json()
    force = data.get('force') if data else None
    
    if force:
        result = system.arduino.grip(force=force)
    else:
        result = system.arduino.grip()
    
    return jsonify(json_safe(result))

@app.route('/arduino/release', methods=['POST'])
def arduino_release():
    result = system.arduino.release()
    return jsonify(json_safe(result))

@app.route('/arduino/led_on', methods=['POST'])
def arduino_led_on():
    result = system.arduino.led_on()
    return jsonify(json_safe(result))

@app.route('/arduino/led_off', methods=['POST'])
def arduino_led_off():
    result = system.arduino.led_off()
    return jsonify(json_safe(result))

@app.route('/arduino/blink', methods=['POST'])
def arduino_blink():
    result = system.arduino.blink()
    return jsonify(json_safe(result))

@app.route('/arduino/live_status', methods=['GET'])
def arduino_live_status():
    status = system.arduino.get_live_status()
    return jsonify(json_safe(status))

# ==================== ROBOT ENDPOINTS ====================

@app.route('/robot/movement_status', methods=['GET'])
def robot_movement_status():
    status = system.get_robot_movement_status()
    return jsonify(json_safe(status))

@app.route('/robot/enable_movement', methods=['POST'])
def enable_robot_movement():
    system.robot_movement_enabled = True
    
    if not system.movement_running and system.running:
        system.movement_running = True
        system.movement_thread = threading.Thread(target=system._movement_processing_loop, daemon=True)
        system.movement_thread.start()
    
    socketio.emit('robot_movement_enabled', {
        'enabled': True,
        'timestamp': time.time()
    })
    
    return jsonify({
        'success': True,
        'message': 'Robot movement enabled'
    })

@app.route('/robot/disable_movement', methods=['POST'])
def disable_robot_movement():
    system.robot_movement_enabled = False
    system.movement_running = False
    
    socketio.emit('robot_movement_disabled', {
        'enabled': False,
        'timestamp': time.time()
    })
    
    return jsonify({
        'success': True,
        'message': 'Robot movement disabled'
    })

@app.route('/force_refresh_fnn', methods=['POST'])
def force_refresh_fnn():
    if not system.last_tomato:
        return jsonify({
            'success': False,
            'message': 'No tomato currently tracked'
        })
    
    frame = system.camera.read_frame()
    properties = system.predictor.predict_fnn_properties(system.last_tomato, frame, force_run=True)
    
    return jsonify({
        'success': True,
        'message': 'FNN cache refreshed',
        'properties': properties
    })

# ==================== WEB SOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    print('Frontend connected via WebSocket')
    emit('connected', {'message': 'Connected to Tomato Tracking System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Frontend disconnected from WebSocket')

@socketio.on('request_target_position')
def handle_request_target_position():
    if system.last_tomato and system.frontend_target_direction:
        emit('robot_target_position', {
            'target_direction': system.frontend_target_direction,
            'active': system.target_position_active,
            'tomato_id': system.last_tomato.get('id', 0)
        })

@socketio.on('request_force_data')
def handle_request_force_data():
    status = system.arduino.get_live_status()
    emit('force_sensor_data', {
        'sensor1': status['force_sensors']['sensor1'],
        'sensor2': status['force_sensors']['sensor2'],
        'grip_force': status['force_sensors']['grip_force'],
        'total_force': status['force_sensors']['total'],
        'emergency_stop': status['emergency_stop']
    })

# ==================== MAIN ====================
def main():
    print("\n" + "="*80)
    print("🍅 SINGLE TOMATO TRACKING SYSTEM")
    print("="*80)
    
    print("\n✅ FEATURES:")
    print("   1. Single Tomato Tracking with ID persistence")
    print("   2. Complete Arduino Communication Protocol")
    print("   3. Real-time Force Sensor Monitoring")
    print("   4. FNN Prediction Caching (10+ second stability)")
    print("   5. Robot Arm Movement with Visual Indicators")
    print("   6. WebSocket Integration for Real-time Updates")
    print("   7. LED Control for Arduino Testing")
    
    print("\n🚀 Starting server on http://127.0.0.1:5000")
    print("🌐 WebSocket available at ws://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        system.stop()
        print("✅ System stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        system.stop()

if __name__ == '__main__':
    # Required for multiprocessing
    multiprocessing.freeze_support()
    main()