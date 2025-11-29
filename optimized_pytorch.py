"""
Optimized PyTorch inference - Fast version
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load model
print("Loading model...")
model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
model.fuse()  # Fuse layers
print("Model loaded!")

# Open camera
CAMERA = '/dev/video45'
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera ready! Press 'q' to quit")

# Config
IMG_SIZE = 320  # Smaller = faster
CONF_THRESH = 0.5
SKIP_FRAMES = 2  # Process every 3rd frame

frame_count = 0
last_results = None
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Only process every Nth frame
    if frame_count % SKIP_FRAMES == 0:
        start = time.time()
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
        inf_time = (time.time() - start) * 1000
        last_results = (results, inf_time)
        
        fps = 1000 / inf_time if inf_time > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
    
    # Draw last results
    if last_results:
        results, inf_time = last_results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'LOGO: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        fps_text = f'FPS: {np.mean(fps_list):.1f} | Inf: {inf_time:.1f}ms'
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Avg FPS: {np.mean(fps_list):.2f}")
