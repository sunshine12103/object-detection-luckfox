"""
ONNX Detection for Luckfox - FASTEST Option!
2-3x faster than PyTorch .pt model
"""
import cv2
import numpy as np
import onnxruntime as ort
import time

print("Loading ONNX model...")

# Load ONNX model
model_path = "runs/train/yolov5_logo_tdtu/weights/best.onnx"  # Or full path: runs/train/yolov5_logo_tdtu/weights/best.onnx
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Get model info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2]  # Usually 320, 416, or 640

print(f"✓ Model loaded!")
print(f"  Input size: {img_size}x{img_size}")
print(f"  Input name:{input_name}")

# Camera setup
CAMERA = '/dev/video45'
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("✓ Camera ready! Press 'q' to quit\n")

# Config
CONF_THRESH = 0.5
IOU_THRESH = 0.45
SKIP_FRAMES = 2  # Process every Nth frame

frame_count = 0
last_boxes = []
fps_list = []

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding"""
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    # Sort by confidence
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Calculate IoU
        xx1 = np.maximum(current_box[0], other_boxes[:, 0])
        yy1 = np.maximum(current_box[1], other_boxes[:, 1])
        xx2 = np.minimum(current_box[2], other_boxes[:, 2])
        yy2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = current_area + other_areas - intersection
        
        iou = intersection / union
        indices = indices[1:][iou < iou_threshold]
    
    return keep

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    
    frame_count += 1
    h0, w0 = frame.shape[:2]
    
    # Only process every Nth frame
    if frame_count % SKIP_FRAMES == 0:
        start_time = time.time()
        
        # Preprocess
        img, ratio, (dw, dh) = letterbox(frame, (img_size, img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Inference
        outputs = session.run(None, {input_name: img})[0]
        
        # Post-process
        boxes = []
        scores = []
        
        # YOLOv5 output format: [batch, num_detections, 85]
        # 85 = x, y, w, h, conf, class_scores(80)
        for detection in outputs[0]:
            confidence = detection[4]
            if confidence > CONF_THRESH:
                # Get box coordinates
                x, y, w, h = detection[:4]
                
                # Convert from center format to corner format
                x1 = int((x - w/2 - dw) / ratio)
                y1 = int((y - h/2 - dh) / ratio)
                x2 = int((x + w/2 - dw) / ratio)
                y2 = int((y + h/2 - dh) / ratio)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, w0))
                y1 = max(0, min(y1, h0))
                x2 = max(0, min(x2, w0))
                y2 = max(0, min(y2, h0))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
        
        # NMS
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            keep = non_max_suppression(boxes, scores, IOU_THRESH)
            last_boxes = [(boxes[i], scores[i]) for i in keep]
        else:
            last_boxes = []
        
        inf_time = (time.time() - start_time) * 1000
        fps = 1000 / inf_time if inf_time > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
    
    # Draw results
    for box, conf in last_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f'LOGO TDTU: {conf:.2f}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw FPS
    avg_fps = np.mean(fps_list) if fps_list else 0
    fps_text = f'FPS: {avg_fps:.1f} | Detections: {len(last_boxes)}'
    cv2.putText(frame, fps_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('ONNX Detection - Luckfox', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if fps_list:
    print(f"\n✓ Average FPS: {np.mean(fps_list):.2f}")
print("Done!")
