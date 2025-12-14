#!/usr/bin/python3
"""
Complete System - With hysteresis for stable control
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import threading
import paho.mqtt.client as mqtt
import json
from collections import deque

class CompleteSystem:
    def __init__(self):
        print("="*60)
        print("COMPLETE HEALTH MONITORING + VEHICLE CONTROL")
        print("="*60)
        
        # MQTT
        self.MQTT_BROKER = "mqtt.fuvitech.vn"
        self.MQTT_PORT = 2883
        self.MQTT_TOPIC = "DATH/DKXE"
        
        print("Connecting to MQTT...")
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print(f"✅ MQTT connected")
        except:
            print("⚠️ MQTT failed")
        
        # Radar
        try:
            self.ser = serial.Serial("/dev/ttyS2", 115200, timeout=0.1)
            print("✅ Radar connected")
            self.radar_available = True
        except:
            print("⚠️ No radar")
            self.radar_available = False
            self.ser = None
        
        # Models
        print("Loading models...")
        self.logo_model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.logo_model.fuse()
        self.pose_model = YOLO('yolov8n-pose.pt')
        print("✅ Models loaded!")
        
        # Config
        self.IMG_SIZE = 320
        self.POSE_SIZE = 224
        
        # Distance
        self.logo_real_width = 7.5
        self.logo_real_height = 3.0
        self.focal_length = 1206.4
        
        # Radar
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        # Fall
        self.fall_count = 0
        self.fall_detected = False
        self.fall_alert_time = 0
        
        # Vehicle control with HYSTERESIS
        self.TARGET_DISTANCE = 100
        self.CENTER_X = 320
        self.X_TOL = 100
        
        # Hysteresis zones to prevent oscillation
        self.current_state = "STOP"  # Track current state
        
        # Command stability
        self.cmd_history = deque(maxlen=15)  # Increased history
        self.last_sent_cmd = None
        self.cmd_send_interval = 0.3
        self.last_send_time = 0
        
        # Logo state
        self.last_logo = None
        self.logo_timeout = 0
        
        print("✅ System ready!")
    
    def calculate_distance(self, w, h):
        try:
            return ((self.logo_real_width * self.focal_length) / w + 
                   (self.logo_real_height * self.focal_length) / h) / 2
        except:
            return 0.0
    
    def check_fall(self, kp):
        try:
            nose, l_sh, r_sh = kp[0][0][:2], kp[0][5][:2], kp[0][6][:2]
            if kp[0][0][2] < 0.5 or kp[0][5][2] < 0.5 or kp[0][6][2] < 0.5:
                return False
            
            sh_dx = abs(r_sh[0] - l_sh[0])
            sh_dy = abs(r_sh[1] - l_sh[1])
            sh_cx = (l_sh[0] + r_sh[0]) / 2
            sh_cy = (l_sh[1] + r_sh[1]) / 2
            
            return sum([sh_dx / (sh_dy + 1) > 3.0,
                       abs(nose[0] - sh_cx) / (abs(sh_cy - nose[1]) + 1) > 2.0,
                       nose[1] > sh_cy + 50]) >= 2
        except:
            return False
    
    def determine_command(self, cx, dist):
        """
        Determine command with HYSTERESIS to prevent oscillation
        """
        dx = cx - self.CENTER_X
        
        # Distance zones with hysteresis
        # Current state affects thresholds
        if self.current_state == "BACKWARD":
            # If backing up, need to be farther before stopping
            too_close = dist < 85  # Lower threshold
            good_distance = 85 <= dist <= 130
            too_far = dist > 130
        elif self.current_state == "FORWARD":
            # If moving forward, need to be closer before stopping
            too_close = dist < 70
            good_distance = 70 <= dist <= 115  # Higher threshold
            too_far = dist > 115
        else:
            # Default thresholds
            too_close = dist < 75
            good_distance = 75 <= dist <= 125
            too_far = dist > 125
        
        # Determine command
        if too_close:
            cmd = "BACKWARD"
        elif too_far:
            if abs(dx) > self.X_TOL:
                cmd = "LEFT" if dx < 0 else "RIGHT"
            else:
                cmd = "FORWARD"
        elif good_distance:
            if abs(dx) > self.X_TOL:
                cmd = "STRAFE_LEFT" if dx < 0 else "STRAFE_RIGHT"
            else:
                cmd = "STOP"
        else:
            cmd = "STOP"
        
        return cmd
    
    def get_stable_command(self):
        """Get most common command from history"""
        if len(self.cmd_history) < 5:
            return self.current_state
        
        # Count occurrences
        cmd_counts = {}
        for cmd in self.cmd_history:
            cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
        
        # Return most common
        most_common = max(cmd_counts, key=cmd_counts.get)
        
        # Only change if significantly different (>60% votes)
        if cmd_counts[most_common] > len(self.cmd_history) * 0.6:
            return most_common
        else:
            return self.current_state
    
    def send_mqtt(self, cmd):
        """Send MQTT with rate limiting"""
        current_time = time.time()
        
        if (cmd != self.last_sent_cmd or 
            current_time - self.last_send_time > self.cmd_send_interval):
            
            try:
                self.mqtt_client.publish(self.MQTT_TOPIC, 
                                        json.dumps({"command": cmd, "timestamp": current_time}))
                print(f"📤 {cmd} (dist: {self.last_logo[5]:.0f}cm)" if self.last_logo else f"📤 {cmd}")
                self.last_sent_cmd = cmd
                self.last_send_time = current_time
            except:
                pass
    
    def read_radar_worker(self):
        while self.radar_running and self.radar_available:
            try:
                data = []
                while self.ser.in_waiting > 0:
                    data.append(self.ser.read().hex())
                
                if len(data) >= 7:
                    if int(data[2][0], 16)*10 + int(data[2][1], 16) == 85 and \
                       int(data[3][0], 16)*10 + int(data[3][1], 16) == 2:
                        self.heart_rate = int(data[6][0], 16) * 16 + int(data[6][1], 16)
                    elif int(data[2][0], 16)*10 + int(data[2][1], 16) == 81 and \
                         int(data[3][0], 16)*10 + int(data[3][1], 16) == 2:
                        self.breath_rate = int(data[6][0], 16) * 16 + int(data[6][1], 16)
                
                time.sleep(0.1)
            except:
                pass
    
    def run(self):
        cap = cv2.VideoCapture('/dev/video45')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ No camera!")
            return
        
        print("Press 'q' to quit, SPACE to pause, 'r' to reset fall")
        
        # Start radar
        if self.radar_available:
            self.radar_running = True
            threading.Thread(target=self.read_radar_worker, daemon=True).start()
        
        fps_list = []
        fc = 0
        paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            fc += 1
            start = time.time()
            
            # Logo detection
            raw_cmd = "STOP"
            dist = 0
            
            if fc % 2 == 0:
                results = self.logo_model(frame, imgsz=self.IMG_SIZE, conf=0.5, verbose=False)
                
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        box = r.boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        cx = (x1 + x2) // 2
                        w, h = x2 - x1, y2 - y1
                        dist = self.calculate_distance(w, h)
                        
                        self.last_logo = (x1, y1, x2, y2, conf, dist, cx)
                        self.logo_timeout = 0
                        
                        if not paused:
                            raw_cmd = self.determine_command(cx, dist)
                        break
                
                if r.boxes is None or len(r.boxes) == 0:
                    self.logo_timeout += 1
            else:
                self.logo_timeout += 1
            
            if self.logo_timeout > 3:
                self.last_logo = None
                raw_cmd = "STOP"
            
            # Add to history
            self.cmd_history.append(raw_cmd)
            
            # Get stable command
            stable_cmd = self.get_stable_command()
            
            # Update current state
            self.current_state = stable_cmd
            
            # Send
            if not paused:
                self.send_mqtt(stable_cmd)
            
            # Draw logo
            if self.last_logo:
                x1, y1, x2, y2, conf, dist, cx = self.last_logo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'LOGO: {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Color code distance
                if 75 <= dist <= 125:
                    dist_color = (0, 255, 0)  # Green = good
                else:
                    dist_color = (0, 255, 255)  # Yellow = adjusting
                
                cv2.putText(frame, f'{dist:.1f}cm', (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
                cv2.line(frame, (cx, 0), (cx, 480), (0, 255, 255), 1)
            
            # Pose detection
            pose_res = self.pose_model(frame, imgsz=self.POSE_SIZE, conf=0.5, verbose=False)
            
            curr_fall = False
            for r in pose_res:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kp = r.keypoints.data.cpu().numpy()
                    if self.check_fall(kp):
                        curr_fall = True
                    
                    if not self.fall_detected:
                        for pk in kp:
                            if pk[0][2] > 0.5:
                                cv2.circle(frame, (int(pk[0][0]), int(pk[0][1])), 5, (255, 0, 0), -1)
                            for i in [5, 6]:
                                if pk[i][2] > 0.5:
                                    cv2.circle(frame, (int(pk[i][0]), int(pk[i][1])), 4, (0, 255, 0), -1)
            
            # Fall state
            if curr_fall:
                self.fall_count += 1
                if self.fall_count >= 5 and not self.fall_detected:
                    self.fall_detected = True
                    self.fall_alert_time = time.time()
                    print("⚠️ FALL!")
            else:
                self.fall_count = 0
            
            if self.fall_detected and time.time() - self.fall_alert_time > 10:
                self.fall_detected = False
            
            # Display
            if self.fall_detected:
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 15)
                cv2.rectangle(frame, (120, 160), (520, 320), (0, 0, 255), -1)
                cv2.rectangle(frame, (120, 160), (520, 320), (255, 255, 255), 3)
                cv2.putText(frame, "!!! FALL DETECTED !!!", (140, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                cv2.putText(frame, "Press 'R' to reset", (200, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Radar panel
            elif self.last_logo and self.radar_available:
                cv2.rectangle(frame, (10, 60), (260, 160), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 60), (260, 160), (0, 255, 0), 2)
                cv2.putText(frame, "VITAL SIGNS", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Heart: {self.heart_rate} bpm", (20, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Breath: {self.breath_rate} /min", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Center line
            cv2.line(frame, (self.CENTER_X, 0), (self.CENTER_X, 480), (255, 0, 0), 2)
            
            # FPS & Command
            fps = 1000 / ((time.time() - start) * 1000 + 0.001)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            cv2.putText(frame, f'FPS: {np.mean(fps_list):.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if not self.fall_detected:
                cv2.putText(frame, f'CMD: {stable_cmd}', (400, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(frame, 'PAUSED', (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            cv2.imshow('Complete System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    self.send_mqtt("STOP")
            elif key == ord('r') or key == ord('R'):
                self.fall_detected = False
                self.fall_count = 0
        
        self.send_mqtt("STOP")
        self.radar_running = False
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        print("✅ Stopped")

if __name__ == "__main__":
    CompleteSystem().run()
