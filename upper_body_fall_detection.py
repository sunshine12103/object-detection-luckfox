#!/usr/bin/python3
"""
Upper Body Fall Detection - Calibrated version
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import math

class UpperBodyFallDetector:
    def __init__(self):
        print("Loading YOLOv8n-Pose...")
        self.model = YOLO('yolov8n-pose.pt')
        print("✅ Model loaded!")
        
        self.IMG_SIZE = 224
        self.CONF_THRESH = 0.5
        
        self.fall_detected = False
        self.fall_count = 0
        self.FALL_CONFIRM_FRAMES = 5  # Tăng lên 5 frames
        
    def check_fall_upper_body(self, keypoints):
        """
        Detect fall - stricter conditions
        """
        try:
            kp = keypoints[0]
            
            # Get nose and shoulders
            nose = kp[0][:2]
            l_shoulder = kp[5][:2]
            r_shoulder = kp[6][:2]
            
            # Check confidence
            if kp[0][2] < 0.5 or kp[5][2] < 0.5 or kp[6][2] < 0.5:
                return False
            
            # Calculate shoulder line
            shoulder_dx = abs(r_shoulder[0] - l_shoulder[0])
            shoulder_dy = abs(r_shoulder[1] - l_shoulder[1])
            
            # Shoulder center
            shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
            shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2
            
            # Head to shoulder vector
            head_dx = abs(nose[0] - shoulder_center_x)
            head_dy = abs(shoulder_center_y - nose[1])  # Positive if head above
            
            # === STRICTER CONDITIONS ===
            
            # 1. Shoulder angle - MUST be very horizontal
            # Normal standing: dy > dx (shoulders vertical)
            # Falling: dx >> dy (shoulders horizontal)
            shoulder_angle_ratio = shoulder_dx / (shoulder_dy + 1)
            is_shoulder_horizontal = shoulder_angle_ratio > 3.0  # Very strict
            
            # 2. Head position - MUST be beside or below shoulders
            # Normal: head is above shoulders (head_dy > head_dx)
            # Falling: head is beside shoulders (head_dx > head_dy)
            head_position_ratio = head_dx / (head_dy + 1)
            is_head_beside = head_position_ratio > 2.0  # Very strict
            
            # 3. Head below shoulders (face down)
            is_head_below = nose[1] > shoulder_center_y + 50  # Large margin
            
            # FALL only if MULTIPLE conditions are true
            conditions_met = sum([
                is_shoulder_horizontal,
                is_head_beside,
                is_head_below
            ])
            
            # Need at least 2 conditions to confirm fall
            return conditions_met >= 2
            
        except:
            return False
    
    def run(self):
        cap = cv2.VideoCapture('/dev/video45')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("="*60)
        print("UPPER BODY FALL DETECTION - CALIBRATED")
        print("Press 'q' to quit, 'd' to toggle debug")
        print("="*60)
        
        fps_list = []
        show_debug = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Pose detection
            results = self.model(frame, imgsz=self.IMG_SIZE, 
                               conf=self.CONF_THRESH, verbose=False)
            
            current_fall = False
            debug_info = []
            
            for result in results:
                keypoints = result.keypoints
                if keypoints is not None and len(keypoints.data) > 0:
                    kp = keypoints.data.cpu().numpy()
                    
                    # Get values for debug
                    if len(kp) > 0 and kp[0][0][2] > 0.5:
                        person_kp = kp[0]
                        nose = person_kp[0][:2]
                        l_shoulder = person_kp[5][:2]
                        r_shoulder = person_kp[6][:2]
                        
                        shoulder_dx = abs(r_shoulder[0] - l_shoulder[0])
                        shoulder_dy = abs(r_shoulder[1] - l_shoulder[1])
                        shoulder_ratio = shoulder_dx / (shoulder_dy + 1)
                        
                        shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2
                        shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
                        head_dx = abs(nose[0] - shoulder_center_x)
                        head_dy = abs(shoulder_center_y - nose[1])
                        head_ratio = head_dx / (head_dy + 1)
                        
                        debug_info = [
                            f"Shoulder ratio: {shoulder_ratio:.2f} (>3.0 = fall)",
                            f"Head ratio: {head_ratio:.2f} (>2.0 = fall)",
                            f"Head Y: {nose[1]:.0f}, Shoulder Y: {shoulder_center_y:.0f}"
                        ]
                    
                    # Check fall
                    if self.check_fall_upper_body(kp):
                        current_fall = True
                    
                    # Draw keypoints
                    for person_kp in kp:
                        # Nose
                        if person_kp[0][2] > 0.5:
                            x, y = int(person_kp[0][0]), int(person_kp[0][1])
                            color = (0, 0, 255) if current_fall else (255, 0, 0)
                            cv2.circle(frame, (x, y), 10, color, -1)
                            cv2.putText(frame, 'HEAD', (x-20, y-15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Shoulders
                        for i, name in [(5, 'L'), (6, 'R')]:
                            if person_kp[i][2] > 0.5:
                                x, y = int(person_kp[i][0]), int(person_kp[i][1])
                                color = (0, 0, 255) if current_fall else (0, 255, 0)
                                cv2.circle(frame, (x, y), 8, color, -1)
                        
                        # Shoulder line
                        if person_kp[5][2] > 0.5 and person_kp[6][2] > 0.5:
                            p1 = (int(person_kp[5][0]), int(person_kp[5][1]))
                            p2 = (int(person_kp[6][0]), int(person_kp[6][1]))
                            color = (0, 0, 255) if current_fall else (0, 255, 0)
                            cv2.line(frame, p1, p2, color, 3)
            
            # Update fall counter
            if current_fall:
                self.fall_count += 1
                if self.fall_count >= self.FALL_CONFIRM_FRAMES:
                    self.fall_detected = True
            else:
                self.fall_count = 0
                self.fall_detected = False
            
            # FPS
            inf_time = (time.time() - start) * 1000
            fps = 1000 / inf_time if inf_time > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            # FALL ALERT
            if self.fall_detected:
                cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 15)
                cv2.rectangle(frame, (150, 180), (490, 300), (0, 0, 255), -1)
                cv2.rectangle(frame, (150, 180), (490, 300), (255, 255, 255), 3)
                cv2.putText(frame, "FALL DETECTED!", (170, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Status
            status = "FALL!" if self.fall_detected else "Normal"
            color = (0, 0, 255) if self.fall_detected else (0, 255, 0)
            cv2.putText(frame, f'Status: {status} ({self.fall_count}/5)', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, f'FPS: {np.mean(fps_list):.1f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Debug info
            if show_debug and debug_info:
                y_pos = 100
                for info in debug_info:
                    cv2.putText(frame, info, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_pos += 25
            
            cv2.imshow('Fall Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug: {'ON' if show_debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Stopped")

if __name__ == "__main__":
    detector = UpperBodyFallDetector()
    detector.run()
