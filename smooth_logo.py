
#!/usr/bin/python3
"""
Real-time detection with threading + Radar + MQTT Control
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
from collections import deque
import serial
import paho.mqtt.client as mqtt
import json

class RealtimeDetector:
    def __init__(self):
        print("Loading model...")
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        
        # Threading
        self.frame_queue = deque(maxlen=1)  # Only keep latest frame
        self.result_queue = deque(maxlen=1)  # Only keep latest result
        self.running = False
        
        # Detection config
        self.IMG_SIZE = 224  # Even smaller
        self.CONF_THRESH = 0.5
        
        # Distance calculation
        self.logo_real_width = 7.5   # cm
        self.logo_real_height = 3.0  # cm
        self.focal_length = 1477.0
        
        # Radar config (UART2)
        self.SERIAL_PORT = "/dev/ttyS2"
        self.BAUDRATE = 115200
        
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
            print("✅ Radar connected!")
            self.radar_available = True
        except:
            print("⚠️ Radar not connected - running without radar")
            self.radar_available = False
            self.ser = None
        
        # Radar data
        self.heart_rate = 0
        self.breath_rate = 0
        self.radar_running = False
        
        # Detection state
        self.logo_detected = False
        self.detection_stable_count = 0
        self.STABLE_THRESHOLD = 5
        self.lost_logo_count = 0
        self.LOST_THRESHOLD = 10  # 10 frames không thấy -> bắt đầu tìm
        self.searching_mode = False
        self.last_turn_direction = "RIGHT"  # Hướng quay cuối cùng
        
        # MQTT config
        self.MQTT_BROKER = "mqtt.fuvitech.vn"
        self.MQTT_PORT = 2883
        self.MQTT_TOPIC = "DATH/DKXE"
        
        # Vehicle control params
        self.FRAME_CENTER_X = 320  # 640/2
        self.DEAD_ZONE = 150  # pixels - vùng chết giữa (tăng lên để ít quay hơn)
        self.TARGET_DISTANCE = 100  # cm - khoảng cách mong muốn (1m)
        self.DISTANCE_TOLERANCE = 20  # cm
        self.last_command = "STOP"
        self.command_interval = 0.5  # giây - tăng lên để xe có thời gian điều chỉnh
        self.last_command_time = 0
        
        # Setup MQTT
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print("✅ MQTT connected!")
            self.mqtt_available = True
        except:
            print("⚠️ MQTT not connected")
            self.mqtt_available = False
        
        print("Model loaded!")
    
    def calculate_distance(self, bbox_width, bbox_height):
        """Tính khoảng cách từ camera đến logo"""
        try:
            distance_by_width = (self.logo_real_width * self.focal_length) / bbox_width
            distance_by_height = (self.logo_real_height * self.focal_length) / bbox_height
            distance = (distance_by_width + distance_by_height) / 2
            return distance
        except ZeroDivisionError:
            return 0.0
    
    def send_vehicle_command(self, command):
        """Gửi lệnh điều khiển xe qua MQTT"""
        current_time = time.time()
        
        # Tránh gửi lệnh quá nhanh
        if current_time - self.last_command_time < self.command_interval:
            return
        
        # Chỉ gửi nếu lệnh thay đổi
        if command == self.last_command:
            return
        
        if self.mqtt_available:
            try:
                # Gửi command dạng text đơn giản cho ESP32
                self.mqtt_client.publish(self.MQTT_TOPIC, command)
                print(f"🚗 Command: {command}")
                self.last_command = command
                self.last_command_time = current_time
            except Exception as e:
                print(f"❌ MQTT error: {e}")
    
    def calculate_vehicle_command(self, bbox_x1, bbox_x2, distance):
        """Tính toán lệnh điều khiển xe dựa trên vị trí logo"""
        # Tâm của bbox
        bbox_center_x = (bbox_x1 + bbox_x2) / 2
        
        # Độ lệch so với tâm frame
        offset_x = bbox_center_x - self.FRAME_CENTER_X
        
        # Kiểm tra khoảng cách
        distance_error = distance - self.TARGET_DISTANCE
        
        # Logic điều khiển với proportional control
        if abs(offset_x) > self.DEAD_ZONE:
            # Logo lệch -> cần quay
            # Tính mức độ lệch (0.0 - 1.0)
            offset_ratio = min(abs(offset_x) / 320.0, 1.0)  # normalize
            
            # Phân cấp tốc độ quay
            if offset_ratio > 0.6:
                # Lệch nhiều -> quay nhanh (spin)
                if offset_x < 0:
                    return "TURN_LEFT_FAST"
                else:
                    return "TURN_RIGHT_FAST"
            else:
                # Lệch ít -> quay chậm (arc)
                if offset_x < 0:
                    return "TURN_LEFT_SLOW"
                else:
                    return "TURN_RIGHT_SLOW"
        else:
            # Logo ở giữa -> điều chỉnh khoảng cách
            if distance_error > self.DISTANCE_TOLERANCE:
                return "FORWARD"  # Quá xa -> tiến
            elif distance_error < -self.DISTANCE_TOLERANCE:
                return "BACKWARD"  # Quá gần -> lùi
            else:
                return "STOP"  # Khoảng cách OK
        
        return "STOP"
    
    def read_radar_worker(self):
        """Background thread để đọc radar"""
        while self.radar_running and self.radar_available:
            try:
                data_all = []
                
                while self.ser.in_waiting > 0:
                    byte_data = self.ser.read()
                    hex_str = byte_data.hex()
                    data_all.append(hex_str)
                
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
        detect_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detect_thread.start()
        
        # Start radar thread
        if self.radar_available:
            self.radar_running = True
            radar_thread = threading.Thread(target=self.read_radar_worker, daemon=True)
            radar_thread.start()
        
        print("="*60)
        print("LOGO DETECTION + DISTANCE + RADAR SYSTEM")
        print("Press 'q' to quit")
        print("="*60)
        
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
            current_detected = False
            
            if latest_result:
                results = latest_result['results']
                inf_time = latest_result['inf_time']
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        current_detected = True
                        
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            
                            # Calculate distance
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            distance = self.calculate_distance(bbox_width, bbox_height)
                            
                            # Draw box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                        (0, 255, 0), 2)
                            
                            # Label with distance
                            if distance:
                                label = f'LOGO: {conf:.2f} | {distance:.1f}cm'
                            else:
                                label = f'LOGO: {conf:.2f}'
                            cv2.putText(display_frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                       (0, 255, 0), 2)
                
                # FPS info
                fps_text = f'Inf: {inf_time:.0f}ms'
                cv2.putText(display_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update detection state and vehicle control
            if current_detected:
                self.detection_stable_count += 1
                self.lost_logo_count = 0  # Reset đếm mất logo
                self.searching_mode = False
                
                if self.detection_stable_count >= self.STABLE_THRESHOLD:
                    self.logo_detected = True
                    
                    # Tính lệnh điều khiển xe (lấy bbox đầu tiên)
                    if latest_result:
                        results = latest_result['results']
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None and len(boxes) > 0:
                                box = boxes[0]  # Lấy bbox đầu tiên
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                bbox_width = x2 - x1
                                bbox_height = y2 - y1
                                distance = self.calculate_distance(bbox_width, bbox_height)
                                
                                # Lưu hướng quay cuối
                                bbox_center_x = (x1 + x2) / 2
                                if bbox_center_x < self.FRAME_CENTER_X:
                                    self.last_turn_direction = "LEFT"
                                else:
                                    self.last_turn_direction = "RIGHT"
                                
                                # Tính và gửi lệnh
                                command = self.calculate_vehicle_command(x1, x2, distance)
                                self.send_vehicle_command(command)
                                break
            else:
                self.detection_stable_count = 0
                self.logo_detected = False
                self.lost_logo_count += 1
                
                # Nếu mất logo quá lâu -> vào chế độ tìm kiếm
                if self.lost_logo_count > self.LOST_THRESHOLD:
                    self.searching_mode = True
                    # Quay NGƯỢC lại hướng cuối cùng để tìm lại logo
                    if self.last_turn_direction == "LEFT":
                        search_command = "TURN_RIGHT_SLOW"
                    else:
                        search_command = "TURN_LEFT_SLOW"
                    self.send_vehicle_command(search_command)
                else:
                    # Mới mất logo -> dừng xe
                    self.send_vehicle_command("STOP")
            
            # Display radar data when logo detected
            if self.logo_detected and self.radar_available:
                # Radar panel
                cv2.rectangle(display_frame, (10, 90), (250, 180), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (10, 90), (250, 180), (0, 255, 0), 2)
                
                cv2.putText(display_frame, "RADAR ACTIVE", (20, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                heart_text = f"Heart: {self.heart_rate} bpm"
                cv2.putText(display_frame, heart_text, (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                breath_text = f"Breath: {self.breath_rate} /min"
                cv2.putText(display_frame, breath_text, (20, 165),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            elif self.logo_detected and not self.radar_available:
                cv2.putText(display_frame, "LOGO DETECTED (No Radar)", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
            
            # Display vehicle command
            if self.searching_mode:
                cmd_text = f'CMD: {self.last_command} [SEARCHING...]'
                cv2.putText(display_frame, cmd_text, (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cmd_text = f'CMD: {self.last_command}'
                cv2.putText(display_frame, cmd_text, (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Vẽ center line và dead zone
            cv2.line(display_frame, (self.FRAME_CENTER_X, 0), 
                    (self.FRAME_CENTER_X, 480), (0, 255, 0), 1)
            cv2.line(display_frame, (self.FRAME_CENTER_X - self.DEAD_ZONE, 0), 
                    (self.FRAME_CENTER_X - self.DEAD_ZONE, 480), (255, 0, 0), 1)
            cv2.line(display_frame, (self.FRAME_CENTER_X + self.DEAD_ZONE, 0), 
                    (self.FRAME_CENTER_X + self.DEAD_ZONE, 480), (255, 0, 0), 1)
            
            # Show
            cv2.imshow('Realtime Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.running = False
        self.radar_running = False
        
        # Dừng xe trước khi thoát
        self.send_vehicle_command("STOP")
        time.sleep(0.5)
        
        detect_thread.join(timeout=1)
        if self.radar_available:
            radar_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        if self.mqtt_available:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        print(f"✅ Done! Avg display FPS: {np.mean(fps_list):.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector()
    detector.run()