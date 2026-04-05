# arduino.py
"""
Arduino communication module for tomato tracking system
Separate module for better separation of concerns
"""
import serial
import serial.tools.list_ports
import time
import threading
import queue
import random
from multiprocessing import Process, Queue, Event, Value

# ==================== CONSTANTS ====================
BLINK_PATTERNS = {
    'LEFT': 1,      # 1 blink
    'RIGHT': 2,     # 2 blinks
    'UP': 4,        # 4 blinks
    'DOWN': 5,      # 5 blinks
    'GRIP': 6,      # 6 blinks
    'RELEASE': 3    # 3 blinks
}

VALID_COMMANDS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'GRIP', 'RELEASE']
DEFAULT_BAUD_RATE = 115200
DEFAULT_TIMEOUT = 1.0
CONTROL_LOOP_RATE = 20
COMMAND_COOLDOWN = 2.0  # Still needed to prevent rapid-fire zone changes

# ==================== PROCESS CLASS ====================
class ArduinoControllerProcess(Process):
    """Isolated Arduino controller running at 20Hz with BLINK patterns"""
    
    def __init__(self, command_queue, status_queue, event_queue, stop_event):
        super().__init__()
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.event_queue = event_queue
        self.stop_event = stop_event
        
        # Hardware state
        self.emergency_stop = Value('b', False)
        self.force_sensor1 = Value('f', 0.0)
        self.force_sensor2 = Value('f', 0.0)
        self.grip_force = Value('f', 0.0)
        
        # Serial connection
        self.serial_port = None
        self.baud_rate = DEFAULT_BAUD_RATE
        self.timeout = DEFAULT_TIMEOUT
        
        # Control parameters
        self.loop_rate = CONTROL_LOOP_RATE
        self.control_interval = 1.0 / self.loop_rate
        
        # Blink patterns
        self.blink_patterns = BLINK_PATTERNS.copy()
        
        # State tracking
        self.last_status_time = 0
        self.status_interval = 0.5
        self.force_broadcast_interval = 0.5
        self.last_force_broadcast = 0
        self.last_command_time = 0
        self.command_cooldown = COMMAND_COOLDOWN
        
        # Force request tracking
        self.force_request_interval = 1.0
        self.last_force_request_time = 0
        
        # Reconnection tracking
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2.0
        
        # Pending commands tracking
        self.pending_command = None
        self.pending_command_time = 0
        self.command_timeout = 5.0
        
        # Command response tracking
        self.awaiting_ready = False
        self.last_response = ""
        
        # Force data cache
        self.last_force_data = None
        self.force_data_timestamp = 0
        
        print(f"🔧 ArduinoControllerProcess initialized (20Hz)")
    
    def find_arduino_port(self):
        """Find Arduino port"""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if ('Arduino' in port.description or 
                'CH340' in port.description or 
                'USB Serial' in port.description or
                'COM5' in port.device):
                print(f"🔌 Found Arduino on port: {port.device}")
                return port.device
        return None
    
    def connect_arduino(self):
        """Connect to Arduino"""
        port = self.find_arduino_port()
        if not port:
            return False
            
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=1.0
            )
            time.sleep(2)
            
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            print(f"✅ Connected to Arduino on {port}")
            
            self._test_communication()
            
            self._send_status_update("CONNECTED", f"Connected to {port}")
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            print(f"❌ Arduino connection failed: {str(e)}")
            self._send_status_update("ERROR", f"Connection failed: {str(e)}")
            return False
    
    def _test_communication(self):
        """Test communication with Arduino"""
        try:
            print("🔄 Testing Arduino communication...")
            time.sleep(0.5)
            
            success, response = self._send_command_to_arduino("STATUS", wait_for_ready=True)
            
            if success:
                print(f"✅ Arduino communication test passed")
            else:
                print(f"⚠️ Arduino communication test: {response}")
                
        except Exception as e:
            print(f"⚠️ Communication test failed: {e}")
    
    def _send_command_to_arduino(self, command, wait_for_ready=True):
        """Send raw command to Arduino with error handling"""
        if not self.serial_port or not self.serial_port.is_open:
            return False, "Serial port not open"
        
        try:
            cmd_str = f"{command}\n"
            
            if command != "GET_FORCE":
                print(f"📤 Sending to Arduino: {cmd_str.strip()}")
            
            self.serial_port.reset_input_buffer()
            
            bytes_written = self.serial_port.write(cmd_str.encode())
            self.serial_port.flush()
            
            if bytes_written == 0:
                return False, "No bytes written"
            
            if not wait_for_ready:
                return True, "Command sent"
            
            time.sleep(0.3)
            
            responses = []
            ready_detected = False
            timeout = time.time() + self.command_timeout
            
            while time.time() < timeout:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        if command != "GET_FORCE" and not line.startswith("FORCE:"):
                            print(f"📥 Arduino: {line}")
                        responses.append(line)
                        
                        if line.startswith("FORCE:"):
                            try:
                                parts = line.split(':')[1].split(',')
                                if len(parts) >= 3:
                                    self.force_sensor1.value = float(parts[0])
                                    self.force_sensor2.value = float(parts[1])
                                    self.grip_force.value = float(parts[2])
                                    self.last_force_data = {
                                        'sensor1': float(parts[0]),
                                        'sensor2': float(parts[1]),
                                        'grip': float(parts[2])
                                    }
                                    self.force_data_timestamp = time.time()
                            except:
                                pass
                        
                        if "--- Ready ---" in line or "Ready" in line:
                            ready_detected = True
                            break
                            
                        if line.startswith("ERROR:"):
                            return False, line
                
                time.sleep(0.05)
            
            if ready_detected:
                return True, "Command completed successfully"
            elif responses:
                return True, f"Command sent, responses: {responses[-1] if responses else 'none'}"
            else:
                return False, "No response from Arduino"
            
        except serial.SerialTimeoutException:
            return False, "Write timeout"
        except Exception as e:
            if command != "GET_FORCE":
                print(f"❌ Serial communication error: {e}")
            return False, str(e)
    
    def _send_blink_command(self, direction):
        """Send blink command to Arduino"""
        if not self.serial_port or not self.serial_port.is_open:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            time_left = self.command_cooldown - (current_time - self.last_command_time)
            print(f"⏱️  Command cooldown: {time_left:.1f}s remaining")
            return False
        
        if direction not in VALID_COMMANDS:
            print(f"❌ Invalid command: {direction}")
            return False
        
        try:
            success, response = self._send_command_to_arduino(direction, wait_for_ready=True)
            
            if success:
                self.last_command_time = current_time
                self.pending_command = direction
                self.pending_command_time = current_time
                
                try:
                    self.event_queue.put_nowait({
                        'type': 'BLINK_EVENT',
                        'direction': direction,
                        'blink_count': self.blink_patterns.get(direction, 1),
                        'timestamp': current_time,
                        'response': response
                    })
                except:
                    pass
                
                print(f"💡 Blink command completed: {direction}")
                return True
            else:
                print(f"❌ Arduino command failed: {response}")
                return False
            
        except Exception as e:
            print(f"❌ Error sending blink command: {e}")
            return False
    
    def _send_status_update(self, status, message):
        """Send status update to main process"""
        try:
            self.status_queue.put_nowait({
                'type': 'ARDUINO_STATUS',
                'status': status,
                'message': message,
                'timestamp': time.time(),
                'emergency_stop': self.emergency_stop.value,
                'force_sensor1': self.force_sensor1.value,
                'force_sensor2': self.force_sensor2.value,
                'grip_force': self.grip_force.value,
                'blink_patterns': self.blink_patterns,
                'last_command_time': self.last_command_time,
                'command_cooldown': self.command_cooldown
            })
        except:
            pass
    
    def _broadcast_force_data(self):
        """Broadcast force sensor data"""
        current_time = time.time()
        if current_time - self.last_force_broadcast >= self.force_broadcast_interval:
            if self.last_force_data:
                try:
                    self.event_queue.put_nowait({
                        'type': 'FORCE_SENSOR_DATA',
                        'sensor1': self.last_force_data['sensor1'],
                        'sensor2': self.last_force_data['sensor2'],
                        'grip_force': self.last_force_data['grip'],
                        'total_force': self.last_force_data['sensor1'] + self.last_force_data['sensor2'],
                        'emergency_stop': self.emergency_stop.value,
                        'timestamp': current_time
                    })
                    self.last_force_broadcast = current_time
                except:
                    pass
    
    def _request_force_data(self):
        """Request force data from Arduino"""
        if not self.serial_port or not self.serial_port.is_open:
            return
        
        try:
            if random.random() < 0.1:
                print(f"📤 Requesting force data")
            
            self._send_command_to_arduino("GET_FORCE", wait_for_ready=False)
            
        except Exception as e:
            pass
    
    def _process_commands(self):
        """Process commands from main process"""
        try:
            commands_processed = 0
            while commands_processed < 2:
                cmd = self.command_queue.get_nowait()
                self._execute_command(cmd)
                commands_processed += 1
        except queue.Empty:
            pass
    
    def _execute_command(self, cmd):
        """Execute a command from main process"""
        if not self.serial_port or not self.serial_port.is_open:
            print("⚠️ Cannot execute command - serial port not open")
            return
            
        cmd_type = cmd.get('type')
        
        if cmd_type in ['MOVE', 'BLINK']:
            direction = cmd.get('direction')
            if direction in VALID_COMMANDS:
                print(f"🤖 Executing command: {direction}")
                self._send_blink_command(direction)
                
        elif cmd_type == 'EMERGENCY_STOP':
            self.emergency_stop.value = True
            self._send_command_to_arduino("EMERGENCY_STOP", wait_for_ready=False)
            self._send_status_update("EMERGENCY_STOP", "Emergency stop activated")
            
        elif cmd_type == 'RESET_EMERGENCY':
            self.emergency_stop.value = False
            self._send_command_to_arduino("RESET", wait_for_ready=False)
            self._send_status_update("RESET", "Emergency stop reset")
    
    def run(self):
        """Main process loop"""
        print("🚀 Arduino Controller Process STARTED (20Hz)")
        
        try:
            if not self.connect_arduino():
                print("❌ Failed to connect to Arduino")
                self._send_status_update("DISCONNECTED", "No Arduino found")
                print("🔧 Running in SIMULATION mode (no Arduino)")
            
            last_loop_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    loop_start = time.time()
                    current_time = time.time()
                    
                    self._process_commands()
                    
                    if self.serial_port and self.serial_port.is_open:
                        if current_time - self.last_force_request_time >= self.force_request_interval:
                            self._request_force_data()
                            self.last_force_request_time = current_time
                        
                        if current_time - self.last_status_time >= self.status_interval:
                            self._send_status_update("RUNNING", "Controller running")
                            self.last_status_time = current_time
                        
                        self._broadcast_force_data()
                    else:
                        if self.reconnect_attempts < self.max_reconnect_attempts:
                            self.reconnect_attempts += 1
                            print(f"⚠️ Attempting reconnect {self.reconnect_attempts}/{self.max_reconnect_attempts}...")
                            time.sleep(self.reconnect_delay)
                            self.connect_arduino()
                        else:
                            if current_time - self.last_status_time >= self.status_interval:
                                self._send_status_update("DISCONNECTED", "Lost connection to Arduino")
                                self.last_status_time = current_time
                    
                    loop_time = time.time() - loop_start
                    sleep_time = max(0, self.control_interval - loop_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"Arduino controller loop error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Fatal Arduino controller error: {e}")
            
        finally:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.close()
                except:
                    pass
            
            self._send_status_update("STOPPED", "Controller stopped")
            print("🛑 Arduino Controller Process STOPPED")


# ==================== CLIENT CLASS ====================
class ArduinoClient:
    """Thread-safe client for Arduino controller process with auto-reconnection"""
    
    def __init__(self, socketio=None):
        self.socketio = socketio
        
        self.command_queue = Queue(maxsize=5)
        self.status_queue = Queue(maxsize=5)
        self.event_queue = Queue(maxsize=10)
        self.stop_event = Event()
        
        self.controller = None
        self._create_controller()
        
        self.last_status = {
            'connected': False,
            'status': 'DISCONNECTED',
            'message': 'Not initialized',
            'emergency_stop': False,
            'force_sensor1': 0.0,
            'force_sensor2': 0.0,
            'grip_force': 0.0,
            'blink_patterns': BLINK_PATTERNS,
            'last_command_time': 0,
            'command_cooldown': COMMAND_COOLDOWN
        }
        
        self.status_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.monitor_thread = None
        
        self.last_command_direction = None
        self.last_command_time = 0
        
        self.command_cooldown = COMMAND_COOLDOWN
        
        if socketio:
            self.event_thread = threading.Thread(target=self._process_events, daemon=True)
            self.event_thread.start()
    
    def _create_controller(self):
        """Create a new controller process"""
        self.stop_event.clear()
        self.controller = ArduinoControllerProcess(
            self.command_queue,
            self.status_queue,
            self.event_queue,
            self.stop_event
        )
    
    def _clear_queues(self):
        """Clear all queues"""
        for q in [self.command_queue, self.status_queue, self.event_queue]:
            try:
                while True:
                    q.get_nowait()
            except:
                pass
    
    def _monitor_controller(self):
        """Monitor controller process and restart if needed"""
        while self.running:
            time.sleep(5)
            if not self.running:
                break
                
            if self.controller and not self.controller.is_alive():
                print("⚠️ Arduino controller died - restarting...")
                with self.lock:
                    self._clear_queues()
                    self._create_controller()
                    self.controller.start()
                    print("✅ Arduino controller restarted")
    
    def start(self):
        """Start Arduino controller process"""
        with self.lock:
            if self.running:
                return True
                
            self.controller.start()
            self.running = True
            
            self.status_thread = threading.Thread(target=self._update_status_loop, daemon=True)
            self.status_thread.start()
            
            self.monitor_thread = threading.Thread(target=self._monitor_controller, daemon=True)
            self.monitor_thread.start()
            
            print("✅ Arduino controller client started")
            return True
    
    def stop(self):
        """Stop Arduino controller process"""
        with self.lock:
            if not self.running:
                return
                
            self.stop_event.set()
            self.running = False
            
            if self.controller and self.controller.is_alive():
                self.controller.join(timeout=2.0)
            
            print("🛑 Arduino controller client stopped")
    
    def _update_status_loop(self):
        """Background thread to update Arduino status"""
        while self.running:
            try:
                while True:
                    try:
                        status = self.status_queue.get_nowait()
                        with self.lock:
                            self.last_status.update(status)
                            self.last_status['connected'] = status['status'] in ['CONNECTED', 'RUNNING']
                    except queue.Empty:
                        break
                time.sleep(0.1)
            except Exception as e:
                print(f"Status update error: {e}")
                time.sleep(0.1)
    
    def _process_events(self):
        """Process events for WebSocket broadcasting"""
        while True:
            try:
                event = self.event_queue.get(timeout=1.0)
                if self.socketio:
                    if event['type'] == 'FORCE_SENSOR_DATA':
                        self.socketio.emit('force_sensor_data', event)
                    elif event['type'] == 'BLINK_EVENT':
                        self.socketio.emit('arduino_blink', event)
                        self.last_command_direction = event['direction']
                        self.last_command_time = event['timestamp']
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Event processing error: {e}")
                time.sleep(0.1)
    
    def can_send_command(self):
        """Check if enough time has passed since last command"""
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False
        return True
    
    def send_command(self, cmd_type, **kwargs):
        """Send command to Arduino controller with throttling"""
        if not self.running:
            return {'success': False, 'message': 'Controller not running'}
        
        if not self.can_send_command():
            time_left = self.command_cooldown - (time.time() - self.last_command_time)
            return {
                'success': False, 
                'message': f'Command cooldown active - wait {time_left:.1f}s'
            }
        
        if self.controller and not self.controller.is_alive():
            print("⚠️ Controller dead - attempting restart...")
            with self.lock:
                self._clear_queues()
                self._create_controller()
                self.controller.start()
            time.sleep(1)
        
        cmd = {'type': cmd_type, **kwargs}
        
        try:
            self.command_queue.put_nowait(cmd)
            self.last_command_time = time.time()
            return {
                'success': True, 
                'message': f'Command {cmd_type} queued',
                'direction': kwargs.get('direction')
            }
        except queue.Full:
            return {'success': False, 'message': 'Command queue full - try again later'}
    
    def blink(self, direction):
        """Send blink command"""
        if direction not in VALID_COMMANDS:
            return {
                'success': False,
                'message': f'Invalid direction: {direction}'
            }
        
        result = self.send_command('BLINK', direction=direction)
        if result['success']:
            return {
                'success': True,
                'message': f'Blink command sent for {direction}',
                'direction': direction,
                'blink_count': BLINK_PATTERNS.get(direction, 1)
            }
        return result
    
    def move(self, direction):
        """Send movement command"""
        return self.blink(direction)
    
    def emergency_stop(self):
        """Emergency stop"""
        result = self.send_command('EMERGENCY_STOP')
        return {
            'success': result['success'],
            'message': 'Emergency stop activated' if result['success'] else 'Failed'
        }
    
    def reset_emergency(self):
        """Reset emergency stop"""
        result = self.send_command('RESET_EMERGENCY')
        return {
            'success': result['success'],
            'message': 'Emergency stop reset' if result['success'] else 'Failed'
        }
    
    def grip(self, force=None):
        """Grip command"""
        return self.blink('GRIP')
    
    def release(self):
        """Release command"""
        return self.blink('RELEASE')
    
    def get_status(self):
        """Get current status"""
        with self.lock:
            status = self.last_status.copy()
            status['last_command_direction'] = self.last_command_direction
            status['last_command_time'] = self.last_command_time
            status['command_cooldown'] = self.command_cooldown
            status['can_send_command'] = self.can_send_command()
            return status
    
    def get_live_status(self):
        """Get detailed live status"""
        with self.lock:
            controller_alive = self.controller and self.controller.is_alive() if hasattr(self, 'controller') else False
            
            return {
                'connected': self.last_status['connected'],
                'status': self.last_status['status'],
                'emergency_stop': self.last_status['emergency_stop'],
                'force_sensors': {
                    'sensor1': self.last_status['force_sensor1'],
                    'sensor2': self.last_status['force_sensor2'],
                    'grip_force': self.last_status['grip_force'],
                    'total': self.last_status['force_sensor1'] + self.last_status['force_sensor2']
                },
                'blink_patterns': BLINK_PATTERNS,
                'last_command_time': self.last_command_time,
                'last_command_direction': self.last_command_direction,
                'command_cooldown': self.command_cooldown,
                'can_send_command': self.can_send_command(),
                'controller_running': self.running and controller_alive,
                'message': self.last_status['message'],
                'timestamp': time.time()
            }


# ==================== EXPORTS ====================
__all__ = [
    'ArduinoClient',
    'ArduinoControllerProcess',
    'BLINK_PATTERNS',
    'VALID_COMMANDS'
]