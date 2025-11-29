#!/usr/bin/python3
"""
Logo Detection + Radar Reading
Detect logo -> Read heart rate & breath rate from radar
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import threading
from collections import deque

class LogoRadarSystem:
    def __init__(self):
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        print("✅ Model loaded!")
        
        # Serial for radar
        self.SERIAL_PORT = "/dev/ttyS2"
        self.BAUDRATE = 115200
        
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
            print(f"✅ Radar connected: {self.SERIAL_PORT}")
            self.radar_available = True
        except:
            print("⚠️ Radar not connected - running without radar")
            self.radar_available = False
            self.ser = None
        
        # Detection config
        self.IMG_SIZE = 320
        self.CONF_THRESH = 0.5
        self.SKIP_FRAMES = 2
        
        # Radar data
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        # Detection state
        self.logo_detected = False
        self.detection_stable_count = 0
        self.STABLE_THRESHOLD = 5  # Phải detect ổn định 5 frames
    
    def read_radar_worker(self):
        """Background thread để đọc radar"""
        while self.radar_running and self.radar_available:
            try:
                data_all = []
                
                # Đọc data
                while self.ser.in_waiting > 0:
                    byte_data = self.ser.read()
                    hex_str = byte_data.hex()
                    data_all.append(hex_str)
                
                # Parse data
                if len(data_all) >= 7:
                    # NHỊP TIM (ID: 85 02)
                    if (int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 85) and \
                       (int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2):
                        self.heart_rate = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                    
                    # NHỊP THỞ (ID: 81 02)
                    elif (int(data_all[2][0], 16)*10 + int(data_all[2][1], 16) == 81) and \
                         (int(data_all[3][0], 16)*10 + int(data_all[3][1], 16) == 2):
                        self.breath_rate = int(data_all[6][0], 16) * 16 + int(data_all[6][1], 16)
                
                time.sleep(0.1)
            except:
                pass
    
    def run(self):
        # Open camera
        CAMERA = '/dev/video45'
        cap = cv2.VideoCapture(CAMERA)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("="*60)
        print("LOGO DETECTION + RADAR SYSTEM")
        print("Press 'q' to quit, 's' to save")
        print("="*60)
        
        frame_count = 0
        last_results = None
        fps_list = []
        
        # Start radar thread
        if self.radar_available:
            self.radar_running = True
            radar_thread = threading.Thread(target=self.read_radar_worker, daemon=True)
            radar_thread.start()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection
            if frame_count % self.SKIP_FRAMES == 0:
                start = time.time()
                results = self.model(frame, imgsz=self.IMG_SIZE, 
                                   conf=self.CONF_THRESH, verbose=False)
                inf_time = (time.time() - start) * 1000
                last_results = (results, inf_time)
                
                fps = 1000 / inf_time if inf_time > 0 else 0
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
            
            # Draw results
            current_detected = False
            
            if last_results:
                results, inf_time = last_results
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        current_detected = True
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            
                            # Draw box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'LOGO: {conf:.2f}'
                            cv2.putText(frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # FPS
                fps_text = f'FPS: {np.mean(fps_list):.1f} | Inf: {inf_time:.1f}ms'
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update detection state
            if current_detected:
                self.detection_stable_count += 1
                if self.detection_stable_count >= self.STABLE_THRESHOLD:
                    self.logo_detected = True
            else:
                self.detection_stable_count = 0
                self.logo_detected = False
            
            # Display radar data when logo detected
            if self.logo_detected and self.radar_available:
                # Radar panel
                cv2.rectangle(frame, (10, 60), (250, 150), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 60), (250, 150), (0, 255, 0), 2)
                
                cv2.putText(frame, "RADAR ACTIVE", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                heart_text = f"Heart: {self.heart_rate} bpm"
                cv2.putText(frame, heart_text, (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                breath_text = f"Breath: {self.breath_rate} /min"
                cv2.putText(frame, breath_text, (20, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            elif self.logo_detected and not self.radar_available:
                cv2.putText(frame, "LOGO DETECTED (No Radar)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show
            cv2.imshow('Logo + Radar Detection', frame)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
                if self.logo_detected:
                    print(f"   Heart: {self.heart_rate} bpm, Breath: {self.breath_rate} /min")
        
        # Cleanup
        self.radar_running = False
        cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        
        print("✅ System stopped")

if __name__ == "__main__":
    system = LogoRadarSystem()
    system.run()
