
#!/usr/bin/python3
"""
Logo Following Robot - Luckfox Side
Tương tự human_following nhưng theo logo TDTU

Thuật toán:
- Detect logo trong frame
- Tính deviation từ center
- Gửi lệnh điều khiển qua UART sang Arduino:
  FORWARD, LEFT, RIGHT, STOP
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import serial
import paho.mqtt.client as mqtt
import json
import random

# ============== UART CONFIG ==============
SERIAL_PORT = "/dev/ttyS2"
BAUDRATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    print(f"✅ UART connected: {SERIAL_PORT}")
except Exception as e:
    print(f"❌ UART error: {e}")
    exit(1)

# ============== MQTT CONFIG ==============
MQTT_BROKER = "mqtt.fuvitech.vn"
MQTT_PORT = 1883
MQTT_TOPIC_VITALS = "ambulance/vitals"
MQTT_TOPIC_VEHICLE = "ambulance/vehicle"
MQTT_TOPIC_FALL = "ambulance/fall"
MQTT_PUBLISH_INTERVAL = 2.0  # Seconds between publishes

# Fall Detection Config
FALL_VELOCITY_THRESHOLD = 0.15  # Vertical velocity threshold (normalized/sec) - lowered for sitting
FALL_BOTTOM_THRESHOLD = 0.60    # Logo position threshold (60% from top) - lowered
FALL_CONSECUTIVE_FRAMES = 2     # Consecutive downward frames to confirm - reduced
FALL_COOLDOWN = 5.0             # Seconds between fall alerts

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"❌ MQTT connection failed, rc={rc}")

def on_publish(client, userdata, mid):
    print(f"📤 MQTT published: {userdata}")

# Setup MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_publish = on_publish

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()  # Start background thread
    print(f"🔗 MQTT connecting to {MQTT_BROKER}:{MQTT_PORT}...")
except Exception as e:
    print(f"⚠️ MQTT connection error: {e}")
    mqtt_client = None

# ============== MODEL CONFIG ==============
print("Loading ONNX model...")
model_path = "runs/train/yolov5_logo_tdtu/weights/best.onnx"  # Or full path: runs/train/yolov5_logo_tdtu/weights/best.onnx
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
img_size = session.get_inputs()[0].shape[2]
print(f"✅ Model loaded! Input: {img_size}x{img_size}")

# ============== CAMERA CONFIG ==============
CAMERA = '/dev/video45'
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Camera error!")
    ser.close()
    exit(1)

print("✅ Camera ready!")

# ============== TRACKING CONFIG ==============
CONF_THRESH = 0.3      # Detection confidence (lowered for small logos)
TOLERANCE = 0.15       # Center tolerance (±15%)

# Distance estimation (based on bbox height)
# Calibration: measure actual distance vs bbox height
# Example: At 1m, logo height = 100px → pixel_to_distance = 1.0 / 100 = 0.01
PIXEL_TO_DISTANCE = 0.11  # Calibration factor (adjust after testing)
TARGET_DISTANCE = 1.0     # Target distance in meters (1m)
STOP_DISTANCE = 0.5       # Stop when closer than 0.5m
MIN_LOGO_HEIGHT = 20      # Minimum logo height in pixels to track

SKIP_FRAMES = 2
frame_count = 0
fps_list = []

# Tracking variables
x_deviation = 0
y_max = 0
last_command = "STOP"

# MQTT timing
last_mqtt_publish = 0
last_vehicle_status = None  # Track vehicle status to avoid spam

# Fall detection tracking
logo_y_history = []          # Track Y positions over time
logo_timestamps = []         # Track timestamps
downward_frame_count = 0     # Count consecutive downward movements
last_fall_alert = 0          # Timestamp of last fall alert
fall_detected = False        # Fall state flag

# ============== HELPER FUNCTIONS ==============
def letterbox(img, new_shape=(640, 640)):
    """Resize with padding"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return img, r, (dw, dh)

def send_command(cmd):
    global last_command, last_vehicle_status
    
    # Determine vehicle status
    if cmd == "STOP":
        vehicle_status = "stopped"
    else:  # FORWARD, LEFT, RIGHT
        vehicle_status = "running"
    
    # Publish vehicle status if changed
    if mqtt_client and vehicle_status != last_vehicle_status:
        try:
            payload = {"status": vehicle_status}
            mqtt_client.publish(MQTT_TOPIC_VEHICLE, json.dumps(payload))
            print(f"🚗 Vehicle: {vehicle_status}")
            last_vehicle_status = vehicle_status
        except Exception as e:
            print(f"❌ Vehicle MQTT error: {e}")
    
    # Always send turn commands (they're time-based)
    if cmd.startswith("LEFT") or cmd.startswith("RIGHT"):
        try:
            ser.write((cmd + "\n").encode())
            ser.flush()
            print(f"📤 UART: {cmd}")
            last_command = cmd
            return True
        except Exception as e:
            print(f"❌ UART error: {e}")
            return False
    
    # Always send FORWARD too! (need to resume after auto-stop)
    # Only filter repeated STOP
    if cmd == "FORWARD" or cmd != last_command:
        try:
            ser.write((cmd + "\n").encode())
            ser.flush()
            print(f"📤 UART: {cmd}")
            last_command = cmd
            return True
        except Exception as e:
            print(f"❌ UART error: {e}")
            return False
    return True

def publish_vitals():
    """
    Publish simulated vital signs to MQTT
    Heart rate: 79-85 bpm
    Breathing rate: 17-20 rpm
    """
    global last_mqtt_publish
    
    # Check if enough time has passed
    current_time = time.time()
    if current_time - last_mqtt_publish < MQTT_PUBLISH_INTERVAL:
        return
    
    if mqtt_client is None:
        return
    
    # Generate random vital signs
    heart_rate = random.randint(79, 85)
    breathing_rate = random.randint(17, 20)
    
    # Create JSON payload
    payload = {
        "heartRate": heart_rate,
        "breathingRate": breathing_rate
    }
    
    # Publish to MQTT
    try:
        result = mqtt_client.publish(MQTT_TOPIC_VITALS, json.dumps(payload))
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"💓 MQTT: {{HR: {heart_rate}, BR: {breathing_rate}}}")
        last_mqtt_publish = current_time
    except Exception as e:
        print(f"❌ MQTT publish error: {e}")

def check_fall_detection(y_center, frame_height):
    """
    Detect fall based on logo trajectory
    
    Fall indicators:
    1. Logo moving downward consistently
    2. High vertical velocity
    3. Logo near bottom of frame
    4. Sudden disappearance after downward movement
    """
    global logo_y_history, logo_timestamps, downward_frame_count
    global last_fall_alert, fall_detected
    
    current_time = time.time()
    
    # Add current position to history
    logo_y_history.append(y_center)
    logo_timestamps.append(current_time)
    
    # Keep only last 10 positions (rolling window)
    if len(logo_y_history) > 10:
        logo_y_history.pop(0)
        logo_timestamps.pop(0)
    
    # Need at least 3 points for analysis
    if len(logo_y_history) < 3:
        return
    
    # Calculate vertical velocity (change in Y position)
    if len(logo_y_history) >= 2:
        dt = logo_timestamps[-1] - logo_timestamps[-2]
        if dt > 0:
            dy = logo_y_history[-1] - logo_y_history[-2]
            vertical_velocity = dy / dt
            
            # DEBUG: Show velocity
            if abs(vertical_velocity) > 0.05:  # Show significant movements
                print(f"🔍 Velocity: {vertical_velocity:.3f}, Y: {y_center:.2f}, Count: {downward_frame_count}")
            
            # Check if moving downward (y increasing = moving down in image coords)
            if vertical_velocity > FALL_VELOCITY_THRESHOLD:
                downward_frame_count += 1
            else:
                downward_frame_count = max(0, downward_frame_count - 1)
    
    # Fall detection logic
    if (downward_frame_count >= FALL_CONSECUTIVE_FRAMES and 
        y_center > FALL_BOTTOM_THRESHOLD and
        current_time - last_fall_alert > FALL_COOLDOWN):
        
        # FALL DETECTED!
        fall_detected = True
        last_fall_alert = current_time
        downward_frame_count = 0  # Reset counter
        
        # Publish alert
        if mqtt_client:
            try:
                payload = {"fall": True}
                mqtt_client.publish(MQTT_TOPIC_FALL, json.dumps(payload))
                print(f"🚨 FALL DETECTED! Y={y_center:.2f}, Frames={FALL_CONSECUTIVE_FRAMES}")
            except Exception as e:
                print(f"❌ Fall alert error: {e}")
    
    # Recovery detection: Logo back in upper area = person stood up
    elif (fall_detected and 
          y_center < 0.5 and  # Logo in upper half
          downward_frame_count == 0):
        
        # RECOVERY!
        fall_detected = False
        
        # Publish recovery
        if mqtt_client:
            try:
                payload = {"fall": False}
                mqtt_client.publish(MQTT_TOPIC_FALL, json.dumps(payload))
                print(f"✅ RECOVERY! Person stood up, Y={y_center:.2f}")
            except Exception as e:
                print(f"❌ Recovery alert error: {e}")

def check_fall_on_loss():
    """
    Check if logo disappeared due to fall (moving down then lost)
    """
    global logo_y_history, fall_detected, last_fall_alert
    
    current_time = time.time()
    
    # Check if we have recent history
    if len(logo_y_history) < 2:
        return
    
    # Check if logo was moving down before disappearing
    recent_y = logo_y_history[-3:] if len(logo_y_history) >= 3 else logo_y_history
    
    # All recent positions should be in lower half and increasing
    if (all(y > 0.5 for y in recent_y) and 
        all(recent_y[i] < recent_y[i+1] for i in range(len(recent_y)-1)) and
        current_time - last_fall_alert > FALL_COOLDOWN):
        
        # Logo was moving down and disappeared - likely fall!
        fall_detected = True
        last_fall_alert = current_time
        
        if mqtt_client:
            try:
                payload = {"fall": True}
                mqtt_client.publish(MQTT_TOPIC_FALL, json.dumps(payload))
                print(f"🚨 FALL DETECTED (logo lost)! Last Y={recent_y[-1]:.2f}")
            except Exception as e:
                print(f"❌ Fall alert error: {e}")
        
        # Clear history
        logo_y_history.clear()
        logo_timestamps.clear()

def get_turn_duration(deviation):
    """Tính thời gian rẽ dựa trên độ lệch"""
    deviation = abs(deviation)
    if deviation >= 0.4:
        return 80  # ms
    elif deviation >= 0.35:
        return 60
    elif deviation >= 0.20:
        return 50
    else:
        return 40

def track_logo(boxes, scores, frame_width, frame_height):
    """
    Distance-based tracking logic
    Estimate distance from bbox height, maintain 1m target distance
    """
    global x_deviation, y_max
    
    if len(boxes) == 0:
        print("⚠️ No logo detected")
        check_fall_on_loss()  # Check if lost due to fall
        send_command("STOP")
        return None
    
    # Lấy logo có confidence cao nhất
    best_idx = np.argmax(scores)
    box = boxes[best_idx]
    conf = scores[best_idx]
    
    x1, y1, x2, y2 = box
    
    # Tính bbox dimensions
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Check minimum size
    if bbox_height < MIN_LOGO_HEIGHT:
        print(f"⚠️ Logo too small: height={bbox_height:.1f}px")
        send_command("STOP")
        return None
    
    # ESTIMATE DISTANCE from bbox height
    # Assumption: Logo has fixed real-world size
    # Distance ∝ 1 / bbox_height
    estimated_distance = PIXEL_TO_DISTANCE * (frame_height / bbox_height)
    
    # Tính center của logo (normalized 0-1)
    obj_x_center = (x1 + bbox_width / 2) / frame_width
    obj_y_center = (y1 + bbox_height / 2) / frame_height
    
    # Tính deviation từ center frame (0.5, 0.5)
    x_deviation = 0.5 - obj_x_center
    
    print(f"📍 Center: ({obj_x_center:.3f}, {obj_y_center:.3f}) | "
          f"X_dev: {x_deviation:.3f} | Dist: {estimated_distance:.2f}m | "
          f"Height: {bbox_height:.0f}px | Conf: {conf:.2f}")
    
    # ===== DECISION LOGIC (Distance-based) =====
    
    # 1. Check if too close (< STOP_DISTANCE)
    if estimated_distance < STOP_DISTANCE:
        print(f"🛑 Too close ({estimated_distance:.2f}m < {STOP_DISTANCE}m) - STOPPING")
        send_command("STOP")
        command = "STOP"
    
    # 2. Logo centered → move forward or maintain distance
    elif abs(x_deviation) < TOLERANCE:
        
        if estimated_distance < TARGET_DISTANCE * 0.9:
            # Too close to target - stop
            print(f"✋ At target distance ({estimated_distance:.2f}m) - STOPPING")
            send_command("STOP")
            command = "STOP"
        else:
            # Still far from target - move forward
            print(f"⬆️ Moving FORWARD (dist: {estimated_distance:.2f}m)")
            send_command("FORWARD")
            command = "FORWARD"
    
    # 3. Logo lệch phải → rẽ trái
    elif x_deviation > TOLERANCE:
        turn_ms = get_turn_duration(x_deviation)
        print(f"⬅️ LEFT ({turn_ms}ms) | Dist: {estimated_distance:.2f}m")
        send_command(f"LEFT{turn_ms}")
        command = "LEFT"
    
    # 4. Logo lệch trái → rẽ phải
    elif x_deviation < -TOLERANCE:
        turn_ms = get_turn_duration(abs(x_deviation))
        print(f"➡️ RIGHT ({turn_ms}ms) | Dist: {estimated_distance:.2f}m")
        send_command(f"RIGHT{turn_ms}")
        command = "RIGHT"
    
    else:
        command = "STOP"
    
    # Check for fall based on trajectory
    check_fall_detection(obj_y_center, frame_height)
    
    # Publish vital signs when logo is detected
    publish_vitals()
    
    return {
        'center': (obj_x_center, obj_y_center),
        'deviation': x_deviation,
        'distance': estimated_distance,
        'bbox_height': bbox_height,
        'command': command,
        'confidence': conf
    }

print("\n🚀 Starting Logo Following Robot...")
print(f"📊 Tolerance: ±{TOLERANCE*100}%")
print(f"📏 Min logo height: {MIN_LOGO_HEIGHT}px")
print(f"🎯 Target distance: {TARGET_DISTANCE}m")
print(f"🛑 Stop distance: {STOP_DISTANCE}m")
print("Press 'q' to quit\n")

# ============== MAIN LOOP ==============
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h0, w0 = frame.shape[:2]
        tracking_info = None
        
        # Process every Nth frame
        if frame_count % SKIP_FRAMES == 0:
            start_time = time.time()
            
            # Preprocess
            img, ratio, (dw, dh) = letterbox(frame, (img_size, img_size))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
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
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w0, x2), min(h0, y2)
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
            
            # TRACKING
            if len(boxes) > 0:
                tracking_info = track_logo(boxes, scores, w0, h0)
            else:
                send_command("STOP")
            
            inf_time = (time.time() - start_time) * 1000
            fps = 1000 / inf_time if inf_time > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
        
        # Visualization
        if tracking_info:
            box = boxes[np.argmax(scores)]
            x1, y1, x2, y2 = box
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center dot
            cx = int(tracking_info['center'][0] * w0)
            cy = int(tracking_info['center'][1] * h0)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            
            # Label
            label = f"LOGO: {tracking_info['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center crosshair
        cv2.line(frame, (0, h0//2), (w0, h0//2), (255, 0, 0), 1)
        cv2.line(frame, (w0//2, 0), (w0//2, h0), (255, 0, 0), 1)
        
        # Draw tolerance box
        tol_x = int(TOLERANCE * w0)
        cv2.rectangle(frame, (w0//2 - tol_x, 0), (w0//2 + tol_x, h0), (0, 255, 0), 2)
        
        # FPS & Command
        avg_fps = np.mean(fps_list) if fps_list else 0
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'CMD: {last_command}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Logo Following Robot', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
finally:
    send_command("STOP")  # Emergency stop
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    
    # Disconnect MQTT
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("📡 MQTT disconnected")
    
    print("✅ Cleanup complete")
    
    if fps_list:
        print(f"📊 Average FPS: {np.mean(fps_list):.2f}")
