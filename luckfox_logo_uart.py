#!/usr/bin/python3
"""
Luckfox Logo Detection + UART Communication với Arduino Nano
Khi phát hiện logo TDTU → gửi "hello" qua UART sang Arduino
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import serial

# ============== UART CONFIG ==============
SERIAL_PORT = "/dev/ttyS2"  # Luckfox UART port
BAUDRATE = 115200
SEND_MESSAGE = "hello\n"  # Message gửi khi detect logo

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    print(f"✅ UART connected: {SERIAL_PORT} @ {BAUDRATE} baud")
except Exception as e:
    print(f"❌ Cannot open UART: {e}")
    print("💡 Chạy: ls /dev/ttyS* để tìm port đúng")
    exit(1)

# ============== ONNX MODEL ==============
print("Loading ONNX model...")
model_path = "runs/train/yolov5_logo_tdtu/weights/best.onnx"  # Or full path: runs/train/yolov5_logo_tdtu/weights/best.onnx
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2]  # 320, 416, or 640

print(f"✅ Model loaded! Input size: {img_size}x{img_size}")

# ============== CAMERA CONFIG ==============
CAMERA = '/dev/video45'
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    ser.close()
    exit(1)

print("✅ Camera ready!")

# ============== DETECTION CONFIG ==============
CONF_THRESH = 0.5
SKIP_FRAMES = 2
SEND_INTERVAL = 1.0  # Gửi "hello" mỗi 1 giây (tránh spam)

frame_count = 0
last_boxes = []
fps_list = []
last_send_time = 0

# ============== HELPER FUNCTIONS ==============
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with padding"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)

def send_uart(message):
    """Gửi message qua UART"""
    try:
        ser.write(message.encode())
        ser.flush()
        return True
    except Exception as e:
        print(f"❌ UART send error: {e}")
        return False

print("\n🔍 Starting detection... (Press 'q' to quit)\n")

# ============== MAIN LOOP ==============
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break
        
        frame_count += 1
        h0, w0 = frame.shape[:2]
        logo_detected = False
        
        # Process every Nth frame
        if frame_count % SKIP_FRAMES == 0:
            start_time = time.time()
            
            # Preprocess
            img, ratio, (dw, dh) = letterbox(frame, (img_size, img_size))
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Inference
            outputs = session.run(None, {input_name: img})[0]
            
            # Post-process
            boxes = []
            scores = []
            
            for detection in outputs[0]:
                confidence = detection[4]
                if confidence > CONF_THRESH:
                    x, y, w, h = detection[:4]
                    x1 = int((x - w/2 - dw) / ratio)
                    y1 = int((y - h/2 - dh) / ratio)
                    x2 = int((x + w/2 - dw) / ratio)
                    y2 = int((y + h/2 - dh) / ratio)
                    
                    x1 = max(0, min(x1, w0))
                    y1 = max(0, min(y1, h0))
                    x2 = max(0, min(x2, w0))
                    y2 = max(0, min(y2, h0))
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
            
            if len(boxes) > 0:
                last_boxes = [(boxes[i], scores[i]) for i in range(len(boxes))]
                logo_detected = True
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
        
        # UART: Gửi "hello" khi detect logo
        current_time = time.time()
        if logo_detected and (current_time - last_send_time) >= SEND_INTERVAL:
            if send_uart(SEND_MESSAGE):
                print(f"📤 UART Sent: {SEND_MESSAGE.strip()} (Logo detected!)")
                last_send_time = current_time
                
                # Visual feedback
                cv2.putText(frame, "UART: SENT!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw FPS and detection count
        avg_fps = np.mean(fps_list) if fps_list else 0
        fps_text = f'FPS: {avg_fps:.1f} | Logos: {len(last_boxes)}'
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Logo Detection + UART', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("✅ Cleanup complete")
    
    if fps_list:
        print(f"📊 Average FPS: {np.mean(fps_list):.2f}")
