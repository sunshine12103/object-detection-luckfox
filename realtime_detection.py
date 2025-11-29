"""
Real-time detection with threading
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
from collections import deque

class RealtimeDetector:
    def __init__(self):
        print("Loading model...")
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        
        # Threading
        self.frame_queue = deque(maxlen=1)  # Only keep latest frame
        self.result_queue = deque(maxlen=1)  # Only keep latest result
        self.running = False
        
        # Config
        self.IMG_SIZE = 224  # Even smaller
        self.CONF_THRESH = 0.5
        
        print("Model loaded!")
    
    def detection_worker(self):
        """Background thread for detection"""
        while self.running:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft()
                
                start = time.time()
                results = self.model(frame, imgsz=self.IMG_SIZE, 
                                   conf=self.CONF_THRESH, verbose=False)
                inf_time = (time.time() - start) * 1000
                
                self.result_queue.append({
                    'results': results,
                    'inf_time': inf_time,
                    'timestamp': time.time()
                })
            else:
                time.sleep(0.001)  # Short sleep
    
    def run(self):
        # Open camera
        CAMERA = '/dev/video45'
        cap = cv2.VideoCapture(CAMERA)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Cannot open camera!")
            return
        
        # Start detection thread
        self.running = True
        thread = threading.Thread(target=self.detection_worker, daemon=True)
        thread.start()
        
        print("Camera ready! Press 'q' to quit")
        
        fps_list = []
        last_time = time.time()
        latest_result = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send frame to detection (non-blocking)
            if len(self.frame_queue) == 0:
                self.frame_queue.append(frame.copy())
            
            # Get latest result (non-blocking)
            if len(self.result_queue) > 0:
                latest_result = self.result_queue[-1]
            
            # Draw results
            display_frame = frame.copy()
            
            if latest_result:
                results = latest_result['results']
                inf_time = latest_result['inf_time']
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            
                            # Draw box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                        (0, 255, 0), 2)
                            
                            # Label
                            label = f'LOGO: {conf:.2f}'
                            cv2.putText(display_frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                       (0, 255, 0), 2)
                
                # FPS info
                fps_text = f'Inf: {inf_time:.0f}ms'
                cv2.putText(display_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate display FPS
            current_time = time.time()
            display_fps = 1.0 / (current_time - last_time)
            last_time = current_time
            fps_list.append(display_fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            
            # Display FPS
            fps_text = f'FPS: {np.mean(fps_list):.1f}'
            cv2.putText(display_frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show
            cv2.imshow('Realtime Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.running = False
        thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Done! Avg display FPS: {np.mean(fps_list):.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector()
    detector.run()
