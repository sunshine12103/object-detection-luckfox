"""
Smooth real-time detection with box smoothing
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
from collections import deque

class SmoothDetector:
    def __init__(self):
        print("Loading model...")
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        
        # Threading
        self.frame_queue = deque(maxlen=1)
        self.result_queue = deque(maxlen=1)
        self.running = False
        
        # Config
        self.IMG_SIZE = 224
        self.CONF_THRESH = 0.5
        
        # Smoothing - lưu lại box trước đó
        self.prev_boxes = []
        self.SMOOTH_FACTOR = 0.7  # 0-1, càng cao càng smooth
        
        print("Model loaded!")
    
    def smooth_box(self, new_box, prev_boxes):
        """Làm mịn bounding box"""
        if len(prev_boxes) == 0:
            return new_box
        
        # Tìm box gần nhất trong frame trước
        min_dist = float('inf')
        closest_box = None
        
        for prev_box in prev_boxes:
            # Tính khoảng cách center
            new_cx = (new_box[0] + new_box[2]) / 2
            new_cy = (new_box[1] + new_box[3]) / 2
            prev_cx = (prev_box[0] + prev_box[2]) / 2
            prev_cy = (prev_box[1] + prev_box[3]) / 2
            
            dist = np.sqrt((new_cx - prev_cx)**2 + (new_cy - prev_cy)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_box = prev_box
        
        # Nếu box gần (cùng object), làm mịn
        if min_dist < 100:  # threshold
            smoothed = [
                int(self.SMOOTH_FACTOR * closest_box[i] + 
                    (1 - self.SMOOTH_FACTOR) * new_box[i])
                for i in range(4)
            ]
            return smoothed
        
        return new_box
    
    def detection_worker(self):
        while self.running:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft()
                
                start = time.time()
                results = self.model(frame, imgsz=self.IMG_SIZE, 
                                   conf=self.CONF_THRESH, verbose=False)
                inf_time = (time.time() - start) * 1000
                
                self.result_queue.append({
                    'results': results,
                    'inf_time': inf_time
                })
            else:
                time.sleep(0.001)
    
    def run(self):
        CAMERA = '/dev/video45'
        cap = cv2.VideoCapture(CAMERA)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Cannot open camera!")
            return
        
        # Start thread
        self.running = True
        thread = threading.Thread(target=self.detection_worker, daemon=True)
        thread.start()
        
        print("Camera ready! Press 'q' to quit, 's' to save")
        
        fps_list = []
        last_time = time.time()
        latest_result = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send frame
            if len(self.frame_queue) == 0:
                self.frame_queue.append(frame.copy())
            
            # Get result
            if len(self.result_queue) > 0:
                latest_result = self.result_queue[-1]
            
            # Draw
            display_frame = frame.copy()
            current_boxes = []
            
            if latest_result:
                results = latest_result['results']
                inf_time = latest_result['inf_time']
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            
                            # Smooth box
                            new_box = [x1, y1, x2, y2]
                            smoothed_box = self.smooth_box(new_box, self.prev_boxes)
                            current_boxes.append(smoothed_box)
                            
                            x1, y1, x2, y2 = smoothed_box
                            
                            # Draw
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                        (0, 255, 0), 2)
                            
                            label = f'LOGO: {conf:.2f}'
                            cv2.putText(display_frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                       (0, 255, 0), 2)
                
                # Update prev boxes
                self.prev_boxes = current_boxes
                
                # Info
                cv2.putText(display_frame, f'Inf: {inf_time:.0f}ms', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 255), 2)
            
            # Display FPS
            current_time = time.time()
            display_fps = 1.0 / (current_time - last_time)
            last_time = current_time
            fps_list.append(display_fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            cv2.putText(display_frame, f'FPS: {np.mean(fps_list):.1f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
            
            # Show
            cv2.imshow('Smooth Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"Saved: {filename}")
        
        # Cleanup
        self.running = False
        thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Done! Avg FPS: {np.mean(fps_list):.1f}")

if __name__ == "__main__":
    detector = SmoothDetector()
    detector.run()
