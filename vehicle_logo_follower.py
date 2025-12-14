#!/usr/bin/python3
"""
Vehicle Logo Follower with MQTT Control + Radar
Maintains 100cm distance + reads vital signs
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import serial
import threading

class VehicleLogoFollower:
    def __init__(self):
        print("="*60)
        print("VEHICLE LOGO FOLLOWER + VITAL SIGNS")
        print("Target Distance: 100cm")
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
        
        # Thresholds
        self.X_TOLERANCE = 80
        self.TARGET_DISTANCE = 100
        self.DISTANCE_TOLERANCE = 20
        
        # State
        self.last_command = None
        self.command_count = 0
        self.COMMAND_STABLE = 3
        
        # Radar data
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        print("✅ System ready!")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT connected successfully")
        else:
            print(f"❌ MQTT connection failed with code {rc}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        print("⚠️ MQTT disconnected")
    
    def send_command(self, command):
        """Send command to vehicle via MQTT"""
        try:
            payload = {"command": command, "timestamp": time.time()}
            self.mqtt_client.publish(self.MQTT_TOPIC, json.dumps(payload))
            print(f"📤 Sent: {command}")
        except Exception as e:
            print(f"❌ MQTT send error: {e}")
    
    def calculate_distance(self, w, h):
        try:
            d1 = (self.logo_real_width * self.focal_length) / w
            d2 = (self.logo_real_height * self.focal_length) / h
            return (d1 + d2) / 2
        except:
            return 0.0
    
    def determine_command(self, logo_center_x, logo_center_y, distance):
        """Determine vehicle command based on logo position"""
        dx = logo_center_x - self.CENTER_X
        
        if distance < self.TARGET_DISTANCE - self.DISTANCE_TOLERANCE:
            return "BACKWARD"
        elif distance > self.TARGET_DISTANCE + self.DISTANCE_TOLERANCE:
            if abs(dx) > self.X_TOLERANCE:
                return "LEFT" if dx < 0 else "RIGHT"
            else:
                return "FORWARD"
        else:
            if abs(dx) > self.X_TOLERANCE:
                return "STRAFE_LEFT" if dx < 0 else "STRAFE_RIGHT"
            else:
                return "STOP"
    
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
        print("VEHICLE CONTROL ACTIVE")
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
            
            command = "STOP"
            logo_found = False
            distance = 0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    logo_found = True
                    box = result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    logo_cx = (x1 + x2) // 2
                    logo_cy = (y1 + y2) // 2
                    w, h = x2 - x1, y2 - y1
                    distance = self.calculate_distance(w, h)
                    
                    if not paused:
                        command = self.determine_command(logo_cx, logo_cy, distance)
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (logo_cx, logo_cy), 8, (0, 0, 255), -1)
                    
                    cv2.putText(frame, f'LOGO: {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    dist_color = (0, 255, 0) if abs(distance - 100) < 20 else (0, 255, 255)
                    cv2.putText(frame, f'{distance:.1f}cm', (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
                    
                    cv2.line(frame, (logo_cx, 0), (logo_cx, self.FRAME_HEIGHT), 
                            (0, 255, 255), 1)
                    break
            
            if not logo_found:
                command = "STOP"
            
            # Send command with stability
            if command == self.last_command:
                self.command_count += 1
                if self.command_count >= self.COMMAND_STABLE and not paused:
                    self.send_command(command)
                    self.command_count = 0
            else:
                self.last_command = command
                self.command_count = 0
            
            # Draw center crosshair
            cv2.line(frame, (self.CENTER_X, 0), (self.CENTER_X, self.FRAME_HEIGHT),
                    (255, 0, 0), 2)
            
            # RADAR PANEL (when logo detected)
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
            
            # Command display
            cmd_color = (0, 255, 0) if logo_found else (0, 0, 255)
            cv2.putText(frame, f'CMD: {command}', (400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 2)
            
            # Status
            if paused:
                cv2.putText(frame, 'PAUSED', (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            status = "TRACKING" if logo_found else "SEARCHING"
            cv2.putText(frame, status, (500, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)
            
            cv2.imshow('Vehicle Logo Follower', frame)
            
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
    follower = VehicleLogoFollower()
    follower.run()
