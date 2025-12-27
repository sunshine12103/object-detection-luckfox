#!/usr/bin/python3
"""
Vehicle Logo Follower - ADVANCED VERSION with PID + Kalman Filter
Option A: Full Advanced Control System
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import serial
import threading


class PIDController:
    """PID Controller for smooth control"""
    def __init__(self, kp, ki, kd, output_limits=(-255, 255)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def compute(self, error):
        """Compute PID output"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Total output
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()


class KalmanFilter:
    """Simple Kalman Filter for position tracking"""
    def __init__(self):
        # State: [x, y, vx, vy]
        self.state = np.array([0, 0, 0, 0], dtype=float)
        
        # State covariance
        self.P = np.eye(4) * 1000
        
        # Process noise
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise
        self.R = np.eye(2) * 10
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.initialized = False
    
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return x, y
    
    def update(self, measurement):
        """Update with measurement"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return self.state[:2]
        
        # Measurement residual
        y = measurement - self.H @ self.state
        
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]
    
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.state[2:4]


class ExponentialSmoother:
    """Exponential smoothing for values"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


class VehicleLogoFollowerAdvanced:
    def __init__(self):
        print("="*60)
        print("VEHICLE LOGO FOLLOWER - ADVANCED CONTROL SYSTEM")
        print("Features: PID + Kalman Filter + PWM Control")
        print("Target Distance: 60cm")
        print("="*60)
        
        # MQTT Setup
        self.MQTT_BROKER = "mqtt.fuvitech.vn"
        self.MQTT_PORT = 2883
        self.MQTT_TOPIC = "DATH/DKXE"
        
        print("Connecting to MQTT...")
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print(f"✅ MQTT connected")
        except Exception as e:
            print(f"⚠️ MQTT failed: {e}")
        
        # Radar Setup
        try:
            self.ser = serial.Serial("/dev/ttyS2", 115200, timeout=0.1)
            print("✅ Radar connected")
            self.radar_available = True
        except:
            print("⚠️ No radar")
            self.radar_available = False
            self.ser = None
        
        # Load model
        print("Loading YOLO model...")
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        print("✅ Model loaded!")
        
        # Config
        self.IMG_SIZE = 320
        self.CONF_THRESH = 0.5
        
        # Distance calculation
        self.logo_real_width = 7.5
        self.logo_real_height = 3.0
        self.focal_length = 1206.4
        
        # Control parameters
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.CENTER_X = self.FRAME_WIDTH // 2
        self.CENTER_Y = self.FRAME_HEIGHT // 2
        
        # Target settings
        self.TARGET_DISTANCE = 60
        self.DISTANCE_TOLERANCE = 10  # ±10cm tolerance for distance
        self.X_TOLERANCE = 80  # ±80px tolerance for horizontal position
        
        # PID Controllers
        # Steering PID (controls left-right balance)
        self.steering_pid = PIDController(kp=0.8, ki=0.01, kd=0.15, output_limits=(-100, 100))
        
        # Speed PID (controls forward-backward speed based on distance)
        self.speed_pid = PIDController(kp=1.5, ki=0.02, kd=0.25, output_limits=(-100, 100))
        
        # Kalman Filter for position tracking
        self.kalman = KalmanFilter()
        
        # Exponential smoothers
        self.speed_smoother = ExponentialSmoother(alpha=0.4)
        self.steering_smoother = ExponentialSmoother(alpha=0.35)
        
        # Search mode
        self.last_turn_direction = None
        self.frames_without_logo = 0
        self.last_known_position = None
        
        # Radar data
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        print("✅ Advanced control system ready!")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT connected successfully")
        else:
            print(f"❌ MQTT connection failed with code {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        print("⚠️ MQTT disconnected")
    
    def send_pwm_command(self, left_speed, right_speed):
        """Send PWM command to vehicle via MQTT
        Speed range: -255 to 255 (negative = backward)
        """
        try:
            # Clamp values
            left_speed = int(np.clip(left_speed, -255, 255))
            right_speed = int(np.clip(right_speed, -255, 255))
            
            payload = {
                "command": "PWM",
                "left": left_speed,
                "right": right_speed,
                "timestamp": time.time()
            }
            self.mqtt_client.publish(self.MQTT_TOPIC, json.dumps(payload))
            # print(f"📤 PWM: L={left_speed}, R={right_speed}")
        except Exception as e:
            print(f"❌ MQTT send error: {e}")
    
    def send_command(self, command):
        """Send simple command (for STOP, STRAFE, etc.)"""
        try:
            payload = {"command": command, "timestamp": time.time()}
            self.mqtt_client.publish(self.MQTT_TOPIC, json.dumps(payload))
            print(f"📤 Sent: {command}")
        except Exception as e:
            print(f"❌ MQTT send error: {e}")
    
    def determine_strafe_or_turn(self, steering_error, distance_error):
        """Decide between STRAFE (mecanum) or TURN based on distance
        
        Strategy for mecanum wheels:
        - If at target distance: STRAFE to adjust position (keeps camera stable)
        - If far from target: TURN while moving forward/backward
        """
        STRAFE_THRESHOLD = 80  # Use strafe if horizontal error is not too large
        
        # At target distance - prefer STRAFE for stability
        if abs(distance_error) < self.DISTANCE_TOLERANCE:
            if abs(steering_error) > self.X_TOLERANCE:
                if abs(steering_error) < STRAFE_THRESHOLD:
                    # Small offset - STRAFE only
                    return "STRAFE_LEFT" if steering_error < 0 else "STRAFE_RIGHT"
                else:
                    # Large offset - still STRAFE but might need TURN later
                    return "STRAFE_LEFT" if steering_error < 0 else "STRAFE_RIGHT"
            else:
                return "STOP"
        else:
            # Not at target distance - return None to use normal control
            return None
    
    def calculate_distance(self, w, h):
        try:
            d1 = (self.logo_real_width * self.focal_length) / w
            d2 = (self.logo_real_height * self.focal_length) / h
            return (d1 + d2) / 2
        except:
            return 0.0
    
    def compute_control(self, logo_cx, logo_cy, distance):
        """Compute motor speeds using PID controllers"""
        
        # Steering error (horizontal position)
        steering_error = logo_cx - self.CENTER_X
        
        # Distance error
        distance_error = self.TARGET_DISTANCE - distance
        
        # Compute PID outputs
        steering_output = self.steering_pid.compute(steering_error)
        speed_output = self.speed_pid.compute(distance_error)
        
        # Smooth the outputs
        steering_output = self.steering_smoother.update(steering_output)
        speed_output = self.speed_smoother.update(speed_output)
        
        # Base speed (from distance PID)
        base_speed = speed_output
        
        # Differential drive: adjust left/right based on steering
        left_speed = base_speed - steering_output
        right_speed = base_speed + steering_output
        
        # Scale to PWM range (0-255), with minimum threshold
        MIN_SPEED = 120  # Minimum speed to overcome friction (user confirmed)
        MAX_SPEED = 200  # Maximum safe speed
        
        def scale_speed(speed):
            if abs(speed) < 10:  # Dead zone
                return 0
            sign = 1 if speed > 0 else -1
            scaled = abs(speed) * (MAX_SPEED - MIN_SPEED) / 100 + MIN_SPEED
            return int(sign * min(scaled, MAX_SPEED))
        
        left_pwm = scale_speed(left_speed)
        right_pwm = scale_speed(right_speed)
        
        return left_pwm, right_pwm, steering_error, distance_error
    
    def get_adaptive_search_speed(self):
        """Adaptive search with decreasing speed stages"""
        frames = self.frames_without_logo
        
        if frames <= 20:
            # Stage 1: Slow search (minimum working speed)
            base_speed = 125
            stage = "STAGE 1"
        elif frames <= 40:
            # Stage 2: Slower but still above threshold
            base_speed = 120
            stage = "STAGE 2"
        elif frames <= 60:
            # Stage 3: At minimum threshold
            base_speed = 120
            stage = "STAGE 3"
        else:
            # Stage 4: Stop
            return 0, 0, "STOP"
        
        # Turn in last known direction
        if self.last_turn_direction == "LEFT":
            return -base_speed, base_speed, stage  # Turn left
        elif self.last_turn_direction == "RIGHT":
            return base_speed, -base_speed, stage  # Turn right
        else:
            return 0, 0, "STOP"
    
    def read_radar_worker(self):
        """Background thread to read radar"""
        while self.radar_running and self.radar_available:
            try:
                data_all = []
                while self.ser.in_waiting > 0:
                    data_all.append(self.ser.read().hex())
                
                if len(data_all) >= 7:
                    if int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 85 and \
                       int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2:
                        self.heart_rate = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                    
                    elif int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 81 and \
                         int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2:
                        self.breath_rate = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                
                time.sleep(0.1)
            except:
                pass
    
    def run(self):
        cap = cv2.VideoCapture('/dev/video45')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("="*60)
        print("ADVANCED CONTROL ACTIVE")
        print("PID Steering: Kp=0.8, Ki=0.01, Kd=0.15")
        print("PID Speed: Kp=1.5, Ki=0.02, Kd=0.25")
        print("Search: 4-stage adaptive (70→50→35→0)")
        print("Press 'q' to quit, SPACE to pause")
        print("="*60)
        
        # Start radar thread
        if self.radar_available:
            self.radar_running = True
            threading.Thread(target=self.read_radar_worker, daemon=True).start()
        
        fps_list = []
        paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            start = time.time()
            
            # Detection
            results = self.model(frame, imgsz=self.IMG_SIZE, 
                               conf=self.CONF_THRESH, verbose=False)
            
            logo_found = False
            distance = 0
            search_mode = False
            left_pwm = 0
            right_pwm = 0
            steering_err = 0
            dist_err = 0
            search_stage = ""
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    logo_found = True
                    self.frames_without_logo = 0
                    
                    box = result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    logo_cx = (x1 + x2) // 2
                    logo_cy = (y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    distance = self.calculate_distance(w, h)
                    
                    # Update Kalman filter
                    measurement = np.array([logo_cx, logo_cy])
                    predicted_pos = self.kalman.update(measurement)
                    
                    # Store last known position and direction
                    self.last_known_position = (logo_cx, logo_cy)
                    if logo_cx < self.CENTER_X - 40:
                        self.last_turn_direction = "LEFT"
                    elif logo_cx > self.CENTER_X + 40:
                        self.last_turn_direction = "RIGHT"
                    
                    if not paused:
                        # Check if we should use mecanum STRAFE
                        strafe_cmd = self.determine_strafe_or_turn(
                            logo_cx - self.CENTER_X, 
                            self.TARGET_DISTANCE - distance
                        )
                        
                        if strafe_cmd:
                            # Use STRAFE for lateral adjustment (mecanum advantage!)
                            self.send_command(strafe_cmd)
                            steering_err = logo_cx - self.CENTER_X
                            dist_err = self.TARGET_DISTANCE - distance
                            left_pwm = 0
                            right_pwm = 0
                        else:
                            # Use normal PID control
                            left_pwm, right_pwm, steering_err, dist_err = \
                                self.compute_control(logo_cx, logo_cy, distance)
                            self.send_pwm_command(left_pwm, right_pwm)
                    
                    # Draw detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (logo_cx, logo_cy), 8, (0, 0, 255), -1)
                    
                    # Draw predicted position
                    pred_x, pred_y = int(predicted_pos[0]), int(predicted_pos[1])
                    cv2.circle(frame, (pred_x, pred_y), 6, (255, 0, 255), 2)
                    
                    cv2.putText(frame, f'LOGO: {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    dist_color = (0, 255, 0) if abs(distance - 60) < 10 else (0, 255, 255)
                    cv2.putText(frame, f'{distance:.1f}cm', (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
                    
                    cv2.line(frame, (logo_cx, 0), (logo_cx, self.FRAME_HEIGHT), 
                            (0, 255, 255), 1)
                    break
            
            # Handle logo lost - ADAPTIVE SEARCH
            if not logo_found:
                self.frames_without_logo += 1
                
                if not paused:
                    left_pwm, right_pwm, search_stage = self.get_adaptive_search_speed()
                    
                    if search_stage != "STOP":
                        search_mode = True
                        self.send_pwm_command(left_pwm, right_pwm)
                        
                        if self.frames_without_logo % 15 == 0:
                            print(f"🔍 Search {search_stage}: {self.frames_without_logo} frames - PWM({left_pwm},{right_pwm})")
                    else:
                        self.send_command("STOP")
                        self.steering_pid.reset()
                        self.speed_pid.reset()
                        self.speed_smoother.reset()
                        self.steering_smoother.reset()
                        if self.frames_without_logo == 61:
                            print("❌ Logo lost - stopped")
            
            # Draw center crosshair
            cv2.line(frame, (self.CENTER_X, 0), (self.CENTER_X, self.FRAME_HEIGHT),
                    (255, 0, 0), 2)
            cv2.line(frame, (0, self.CENTER_Y), (self.FRAME_WIDTH, self.CENTER_Y),
                    (255, 0, 0), 1)
            
            # RADAR PANEL
            if logo_found and self.radar_available:
                cv2.rectangle(frame, (10, 60), (260, 160), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 60), (260, 160), (0, 255, 0), 2)
                cv2.putText(frame, "VITAL SIGNS", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Heart: {self.heart_rate} bpm", (20, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Breath: {self.breath_rate} /min", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS
            fps = 1000 / ((time.time() - start) * 1000 + 0.001)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            cv2.putText(frame, f'FPS: {np.mean(fps_list):.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Control info display
            if search_mode:
                status_text = f'SEARCH {search_stage}'
                status_color = (0, 165, 255)
                cv2.putText(frame, status_text, (400, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            elif logo_found:
                if left_pwm == 0 and right_pwm == 0:
                    status_text = 'STRAFE'  # Mecanum strafe mode
                    status_color = (255, 128, 0)
                else:
                    status_text = f'TRACK L:{left_pwm} R:{right_pwm}'
                    status_color = (0, 255, 0)
                cv2.putText(frame, status_text, (320, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                cv2.putText(frame, 'LOST', (450, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # PID errors
            if logo_found:
                cv2.putText(frame, f'Steer Err: {steering_err:.0f}', (10, 450),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f'Dist Err: {dist_err:.1f}cm', (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Pause overlay
            if paused:
                cv2.putText(frame, 'PAUSED', (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            cv2.imshow('Vehicle Logo Follower - ADVANCED', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    self.send_command("STOP")
                print(f"{'PAUSED' if paused else 'RESUMED'}")
        
        # Cleanup
        self.send_command("STOP")
        self.radar_running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        print("✅ System stopped")


if __name__ == "__main__":
    follower = VehicleLogoFollowerAdvanced()
    follower.run()
