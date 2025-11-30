#!/usr/bin/python3
"""
Complete Health Monitoring System
Radar only shows when logo detected
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import threading

class HealthMonitorSystem:
    def __init__(self):
        print("="*60)
        print("HEALTH MONITORING SYSTEM")
        print("="*60)
        
        # Models
        print("Loading models...")
        self.logo_model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.logo_model.fuse()
        self.pose_model = YOLO('yolov8n-pose.pt')
        print("✅ Models loaded!")
        
        # Radar
        try:
            self.ser = serial.Serial("/dev/ttyS2", 115200, timeout=0.1)
            print("✅ Radar connected")
            self.radar_available = True
        except:
            print("⚠️ No radar")
            self.radar_available = False
            self.ser = None
        
        # Config
        self.LOGO_IMG_SIZE = 320
        self.POSE_IMG_SIZE = 224
        
        # Distance
        self.logo_real_width = 7.5
        self.logo_real_height = 3.0
        self.focal_length = 1206.4
        
        # Radar data
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        # Fall detection
        self.fall_count = 0
        self.fall_detected = False
        self.fall_alert_time = 0
        
        # Logo state
        self.last_logo_box = None
        self.logo_frame_count = 0
        self.LOGO_TIMEOUT = 3
        
        print("✅ Ready!")
    
    def calculate_distance(self, w, h):
        try:
            d1 = (self.logo_real_width * self.focal_length) / w
            d2 = (self.logo_real_height * self.focal_length) / h
            return (d1 + d2) / 2
        except:
            return 0.0
    
    def check_fall(self, kp):
        try:
            nose = kp[0][0][:2]
            l_sh = kp[0][5][:2]
            r_sh = kp[0][6][:2]
            
            if kp[0][0][2] < 0.5 or kp[0][5][2] < 0.5 or kp[0][6][2] < 0.5:
                return False
            
            sh_dx = abs(r_sh[0] - l_sh[0])
            sh_dy = abs(r_sh[1] - l_sh[1])
            sh_ratio = sh_dx / (sh_dy + 1)
            
            sh_cx = (l_sh[0] + r_sh[0]) / 2
            sh_cy = (l_sh[1] + r_sh[1]) / 2
            h_dx = abs(nose[0] - sh_cx)
            h_dy = abs(sh_cy - nose[1])
            h_ratio = h_dx / (h_dy + 1)
            
            cond = sum([sh_ratio > 3.0, h_ratio > 2.0, nose[1] > sh_cy + 50])
            return cond >= 2
        except:
            return False
    
    def read_radar_worker(self):
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ No camera!")
            return
        
        print("Press 'q' to quit, 'r' to reset fall")
        
        # Start radar
        if self.radar_available:
            self.radar_running = True
            threading.Thread(target=self.read_radar_worker, daemon=True).start()
        
        fps_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontally
            frame = cv2.flip(frame, 1)
            
            frame_count += 1
            start = time.time()
            
            # === LOGO DETECTION ===
            if frame_count % 2 == 0:
                results = self.logo_model(frame, imgsz=self.LOGO_IMG_SIZE, 
                                         conf=0.5, verbose=False)
                
                logo_found = False
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        box = result.boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        w = x2 - x1
                        h = y2 - y1
                        dist = self.calculate_distance(w, h)
                        
                        self.last_logo_box = (x1, y1, x2, y2, conf, dist)
                        self.logo_frame_count = 0
                        logo_found = True
                        break
                
                if not logo_found:
                    self.logo_frame_count += 1
            else:
                self.logo_frame_count += 1
            
            if self.logo_frame_count > self.LOGO_TIMEOUT:
                self.last_logo_box = None
            
            # Check if logo is currently detected
            logo_is_detected = self.last_logo_box is not None
            
            # === DRAW LOGO ===
            if logo_is_detected:
                x1, y1, x2, y2, conf, dist = self.last_logo_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'LOGO: {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'{dist:.1f}cm', (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # === POSE DETECTION ===
            pose_results = self.pose_model(frame, imgsz=self.POSE_IMG_SIZE,
                                          conf=0.5, verbose=False)
            
            current_fall = False
            for result in pose_results:
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    kp = result.keypoints.data.cpu().numpy()
                    if self.check_fall(kp):
                        current_fall = True
                    
                    if not self.fall_detected:
                        for pk in kp:
                            if pk[0][2] > 0.5:
                                cv2.circle(frame, (int(pk[0][0]), int(pk[0][1])), 
                                         5, (255, 0, 0), -1)
                            for i in [5, 6]:
                                if pk[i][2] > 0.5:
                                    cv2.circle(frame, (int(pk[i][0]), int(pk[i][1])), 
                                             4, (0, 255, 0), -1)
            
            # Fall state
            if current_fall:
                self.fall_count += 1
                if self.fall_count >= 5 and not self.fall_detected:
                    self.fall_detected = True
                    self.fall_alert_time = time.time()
                    print("⚠️ FALL DETECTED!")
            else:
                self.fall_count = 0
            
            if self.fall_detected and time.time() - self.fall_alert_time > 10:
                self.fall_detected = False
            
            # === DISPLAY ===
            
            # FALL ALERT (highest priority)
            if self.fall_detected:
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 15)
                
                cv2.rectangle(frame, (120, 160), (520, 320), (0, 0, 255), -1)
                cv2.rectangle(frame, (120, 160), (520, 320), (255, 255, 255), 3)
                cv2.putText(frame, "!!! FALL DETECTED !!!", (140, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                cv2.putText(frame, "Press 'R' to reset", (200, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # RADAR PANEL - ONLY when logo detected
            elif logo_is_detected and self.radar_available:
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
            
            # Status
            if not self.fall_detected:
                status = "MONITORING" if logo_is_detected else "SEARCHING"
                color = (0, 255, 0) if logo_is_detected else (255, 255, 0)
                cv2.putText(frame, status, (500, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Health Monitor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.fall_detected = False
                self.fall_count = 0
                print("✅ Fall reset")
        
        self.radar_running = False
        cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        print("✅ Stopped")

if __name__ == "__main__":
    HealthMonitorSystem().run()
