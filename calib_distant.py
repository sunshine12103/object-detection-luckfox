import cv2
from ultralytics import YOLO
import numpy as np

class TDTULogoDetector:
    def __init__(self, model_path="runs/train/logo_tdtu_150epochs/weights/best.pt"):
        print("Dang khoi tao YOLO model...")
        try:
            self.model = YOLO(model_path)
            print(f"Da load model: {model_path}")
        except:
            print(f"Khong tim thay model trained, su dung model goc...")
            self.model = YOLO("yolov8n.pt")
        
        self.colors = [(0, 255, 0)]
        
        # Tham số tính khoảng cách
        self.logo_real_width = 7.5   # cm - CẦN ĐO THẬT
        self.logo_real_height = 3.0  # cm - CẦN ĐO THẬT
        self.focal_length = 800      # pixel - Sẽ được calibrate
        
        # Calibration mode
        self.calibration_mode = False
        self.known_distance = 52.0   # cm - Khoảng cách calibration
        
    def calibrate_camera(self):
        """Chế độ calibration tự động"""
        print("=" * 50)
        print("CHẾ ĐỘ CALIBRATION")
        print("=" * 50)
        print("1. Đo kích thước logo TDTU thật bằng thước kẻ")
        print(f"2. Đặt logo ở khoảng cách {self.known_distance}cm từ camera")
        print("3. Nhấn 'c' để capture và tính focal length")
        print("4. Nhấn 'q' để thoát calibration")
        print("=" * 50)
        
        self.calibration_mode = True
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, conf=0.5, verbose=False)
            annotated_frame = self.draw_calibration_results(frame, results)
            
            cv2.imshow('CALIBRATION MODE', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture để calibrate
                self.perform_calibration(results)
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.calibration_mode = False
    
    def perform_calibration(self, results):
        """Thực hiện calibration"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.3:
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Tính focal length
                        focal_length_width = (self.known_distance * bbox_width) / self.logo_real_width
                        focal_length_height = (self.known_distance * bbox_height) / self.logo_real_height
                        
                        # Lấy trung bình
                        self.focal_length = (focal_length_width + focal_length_height) / 2
                        
                        print("=" * 50)
                        print("KẾT QUẢ CALIBRATION:")
                        print(f"Bbox width: {bbox_width} pixel")
                        print(f"Bbox height: {bbox_height} pixel")
                        print(f"Focal length (width): {focal_length_width:.1f}")
                        print(f"Focal length (height): {focal_length_height:.1f}")
                        print(f"Focal length (final): {self.focal_length:.1f}")
                        print("=" * 50)
                        print("Calibration hoàn thành! Có thể sử dụng chế độ đo khoảng cách.")
                        return
        
        print("Không phát hiện được logo để calibrate!")
    
    def draw_calibration_results(self, frame, results):
        """Vẽ kết quả trong chế độ calibration"""
        annotated_frame = frame.copy()
        
        # Vẽ hướng dẫn
        cv2.putText(annotated_frame, f"Dat logo o khoang cach: {self.known_distance}cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "Nhan 'c' de calibrate, 'q' de thoat", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.3:
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Vẽ bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Hiển thị thông tin calibration
                        info = f'Size: {bbox_width}x{bbox_height}px'
                        cv2.putText(annotated_frame, info, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def detect_from_webcam(self):
        print("Dang mo webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Khong the mo webcam!")
            return
        
        print("Webcam da san sang!")
        print("Nhan 'q' de thoat")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, conf=0.5, verbose=False)
            annotated_frame = self.draw_results(frame, results)
            
            cv2.imshow('TDTU Logo Detection with Distance', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_results(self, frame, results):
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.3:
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Tính khoảng cách
                        distance = self.calculate_distance(bbox_width, bbox_height)
                        
                        # Vẽ bounding box
                        color = self.colors[0]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Labels
                        label = f'LOGO TDTU: {confidence:.2f}'
                        distance_label = f'Distance: {distance:.1f}cm'
                        
                        # Vẽ text
                        cv2.putText(annotated_frame, label, (x1, y1-25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(annotated_frame, distance_label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame
    
    def calculate_distance(self, bbox_width, bbox_height):
        """Tính khoảng cách từ camera đến logo"""
        try:
            distance_by_width = (self.logo_real_width * self.focal_length) / bbox_width
            distance_by_height = (self.logo_real_height * self.focal_length) / bbox_height
            distance = (distance_by_width + distance_by_height) / 2
            return distance
        except ZeroDivisionError:
            return 0.0

def main():
    print("TDTU Logo Detector với đo khoảng cách")
    print("=" * 50)
    
    detector = TDTULogoDetector()
    
    while True:
        print("\nChọn chế độ:")
        print("1. Calibrate camera (cần làm trước)")
        print("2. Detect với đo khoảng cách")
        print("3. Thoát")
        
        choice = input("Nhập lựa chọn (1-3): ").strip()
        
        if choice == "1":
            detector.calibrate_camera()
        elif choice == "2":
            detector.detect_from_webcam()
        elif choice == "3":
            break
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()