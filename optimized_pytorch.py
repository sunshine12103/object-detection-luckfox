"""
YOLOv5 Detection for Luckfox/Raspberry Pi
Optimized for embedded devices
"""
import cv2
import torch
import numpy as np
import time

# Load YOLOv5 model using torch.hub
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='runs/train/yolov5_logo_tdtu/weights/best.pt',
                       force_reload=False)

# Optimize for inference
model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU")
else:
    print("Using CPU")

# Set model parameters
model.conf = 0.5  # Confidence threshold
model.iou = 0.45  # NMS IOU threshold
model.max_det = 10  # Maximum detections

print("Model loaded successfully!")

# Camera setup
CAMERA = '/dev/video45'
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Camera ready! Press 'q' to quit")

# Performance settings
IMG_SIZE = 320  # Smaller = faster (try 320, 416, or 640)
SKIP_FRAMES = 2  # Process every N frames (2 = every other frame)

frame_count = 0
last_results = None
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    
    frame_count += 1
    
    # Only process every Nth frame for speed
    if frame_count % SKIP_FRAMES == 0:
        start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            results = model(frame, size=IMG_SIZE)
        
        inf_time = (time.time() - start_time) * 1000
        last_results = (results, inf_time)
        
        # Calculate FPS
        fps = 1000 / inf_time if inf_time > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
    
    # Draw results from last inference
    if last_results:
        results, inf_time = last_results
        
        # Get predictions as pandas DataFrame
        predictions = results.pandas().xyxy[0]
        
        # Draw bounding boxes
        for _, row in predictions.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'LOGO TDTU: {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS and inference time
        avg_fps = np.mean(fps_list) if fps_list else 0
        fps_text = f'FPS: {avg_fps:.1f} | Inference: {inf_time:.1f}ms'
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw detection count
        det_count = f'Detections: {len(predictions)}'
        cv2.putText(frame, det_count, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow('TDTU Logo Detection - Luckfox', frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

if fps_list:
    print(f"\nAverage FPS: {np.mean(fps_list):.2f}")
    print(f"Min FPS: {min(fps_list):.2f}")
    print(f"Max FPS: {max(fps_list):.2f}")
print("Done!")
