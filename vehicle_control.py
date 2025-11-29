"""
Vehicle control with smooth object tracking
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
from collections import deque
import serial  # pip install pyserial

class KalmanFilter:
    """Simple Kalman filter for box smoothing"""
    def __init__(self):
        self.x = None  # State [x, y, w, h]
        self.P = np.eye(4) * 1000  # Covariance
        self.Q = np.eye(4) * 0.1  # Process noise
        self.R = np.eye(4) * 10   # Measurement noise
        
    def predict(self):
        if self.x is not None:
            self.P = self.P + self.Q
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        
        # Kalman gain
        K = self.P @ np.linalg.inv(self.P + self.R)
        
        # Update state
        self.x = self.x + K @ (measurement - self.x)
        
        # Update covariance
        self.P = (np.eye(4) - K) @ self.P
        
        return self.x

class VehicleController:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200):
        print("Initializing Vehicle Controller...")
        
        # Load model
        self.model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')
        self.model.fuse()
        
        # Serial connection (comment out if testing without vehicle)
        try:
            self.serial = serial.Serial(serial_port, baudrate, timeout=0.1)
            print(f"Serial connected: {serial_port}")
        except:
            self.serial = None
            print("Serial not connected - running in test mode")
        
        # Threading
        self.frame_queue = deque(maxlen=1)
        self.result_queue = deque(maxlen=1)
        self.running = False
        
        # Config
        self.IMG_SIZE = 224
        self.CONF_THRESH = 0.5
        
        # Kalman filter for smooth tracking
        self.kalman = KalmanFilter()
        
        # PID controller
        self.prev_error = 0
        self.integral = 0
        
        # Control parameters
        self.Kp = 1.0  # Proportional gain
        self.Ki = 0.1  # Integral gain
        self.Kd = 0.5  # Derivative gain
        
        # Target position (center of frame)
        self.target_x = 320  # Half of 640
        self.target_y = 240  # Half of 480
        
        print("Controller ready!")
    
    def detection_worker(self):
        while self.running:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft()
                
                results = self.model(frame, imgsz=self.IMG_SIZE, 
                                   conf=self.CONF_THRESH, verbose=False)
                
                self.result_queue.append(results)
            else:
                time.sleep(0.001)
    
    def pid_control(self, error):
        """PID controller for smooth steering"""
        self.integral += error
        derivative = error - self.prev_error
        
        output = (self.Kp * error + 
                 self.Ki * self.integral + 
                 self.Kd * derivative)
        
        self.prev_error = error
        
        # Clamp output
        return np.clip(output, -100, 100)
    
    def send_command(self, steering, speed):
        """Send command to vehicle via serial"""
        if self.serial:
            # Format: "S:steering,V:speed\n"
            cmd = f"S:{int(steering)},V:{int(speed)}\n"
            self.serial.write(cmd.encode())
    
    def run(self):
        # Open camera
        CAMERA = '/dev/video45'
        cap = cv2.VideoCapture(CAMERA)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Cannot open camera!")
            return
        
        # Start detection thread
        self.running = True
        thread = threading.Thread(target=self.detection_worker, daemon=True)
        thread.start()
        
        print("="*60)
        print("VEHICLE TRACKING ACTIVE")
        print("Controls: q=quit, SPACE=pause, s=save")
        print("="*60)
        
        paused = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Send frame for detection
            if not paused and len(self.frame_queue) == 0:
                self.frame_queue.append(frame.copy())
            
            # Get latest result
            display_frame = frame.copy()
            steering = 0
            speed = 0
            
            if len(self.result_queue) > 0 and not paused:
                results = self.result_queue[-1]
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # Get first (highest confidence) box
                        box = boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Calculate center and size
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Apply Kalman filter
                        measurement = np.array([cx, cy, w, h])
                        self.kalman.predict()
                        smoothed = self.kalman.update(measurement)
                        
                        cx, cy, w, h = smoothed
                        x1 = int(cx - w/2)
                        y1 = int(cy - h/2)
                        x2 = int(cx + w/2)
                        y2 = int(cy + h/2)
                        
                        # Calculate error (horizontal offset from center)
                        error_x = cx - self.target_x
                        
                        # PID control for steering
                        steering = self.pid_control(error_x)
                        
                        # Speed based on distance (size of box)
                        # Larger box = closer = slower
                        box_area = w * h
                        max_area = 640 * 480 * 0.3  # 30% of frame
                        
                        if box_area < max_area:
                            speed = 50  # Move forward
                        else:
                            speed = 0  # Stop when too close
                        
                        # Send command to vehicle
                        self.send_command(steering, speed)
                        
                        # Draw smoothed box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), 
                                    (0, 255, 0), 2)
                        
                        # Draw center crosshair
                        cv2.circle(display_frame, (int(cx), int(cy)), 5, 
                                 (0, 0, 255), -1)
                        
                        # Draw target
                        cv2.circle(display_frame, (self.target_x, self.target_y), 
                                 10, (255, 0, 0), 2)
                        
                        # Info
                        label = f'LOGO: {conf:.2f}'
                        cv2.putText(display_frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                   (0, 255, 0), 2)
                        
                        # Control info
                        info = f'Steering: {steering:.1f} | Speed: {speed}'
                        cv2.putText(display_frame, info, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (0, 255, 255), 2)
                        
                        error_info = f'Error: {error_x:.1f}px'
                        cv2.putText(display_frame, error_info, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (255, 255, 0), 2)
            else:
                # No detection - stop vehicle
                self.send_command(0, 0)
                cv2.putText(display_frame, 'NO TARGET', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Pause indicator
            if paused:
                cv2.putText(display_frame, 'PAUSED', (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            # Show
            cv2.imshow('Vehicle Control', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    self.send_command(0, 0)  # Stop when paused
            elif key == ord('s'):
                filename = f'tracking_{int(time.time())}.jpg'
                cv2.imwrite(filename, display_frame)
                print(f"Saved: {filename}")
        
        # Cleanup - stop vehicle
        self.send_command(0, 0)
        self.running = False
        thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        if self.serial:
            self.serial.close()
        
        print("Vehicle control stopped")

if __name__ == "__main__":
    # Adjust serial port if needed
    controller = VehicleController(serial_port='/dev/ttyUSB0')
    controller.run()
