import cv2
from ultralytics import YOLO
import numpy as np
import math

class TDTULogoDetector:
    def __init__(self, model_path="runs/train/logo_tdtu_150epochs/weights/best.pt"):
        print("Dang khoi tao YOLO model...")
        try:
            self.model = YOLO(model_path)
            print(f"Da load model: {model_path}")
        except:
            print(f"Khong tim thay model")
        
        self.colors = [(0, 255, 0)]
        
        self.logo_real_width = 7.5  # Chiều rộng logo
        self.logo_real_height = 3.0 # Chiều cao logo
        self.focal_length = 1206.4  # pixel
        
    def detect_from_webcam(self):
        print("Dang mo webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Khong the mo webcam")
            return
    
        print("q thoat")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, conf=0.5, verbose=False)
            
            # Ve ket qua len 
            annotated_frame = self.draw_results(frame, results)
            
            # Hien thi frame
            cv2.imshow('TDTU Logo Detection', annotated_frame)
            
            # Xu ly phim bam
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Da dong webcam!")
    
    def draw_results(self, frame, results):
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lay toa do bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Chi hien thi neu confidence du cao
                    if confidence > 0.3:
                        # Tinh kich thuoc bounding box
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Tinh khoang cach
                        distance = self.calculate_distance(bbox_width, bbox_height)
                        
                        # Ve bounding box
                        color = self.colors[0]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Tao label với khoảng cách
                        label = f'LOGO TDTU: {confidence:.2f}'
                        distance_label = f'Distance: {distance:.1f}cm'
                        
                        # Ve background cho text confidence
                        (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w1, y1), color, -1)
                        
                        # Ve text confidence
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Ve background cho text distance
                        (w2, h2), _ = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_frame, (x1, y2), (x1 + w2, y2 + 20), color, -1)
                        
                        # Ve text distance
                        cv2.putText(annotated_frame, distance_label, (x1, y2 + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        print(f"Confidence: {confidence:.2f}, Distance: {distance:.1f}cm")
        
        return annotated_frame
    
    def calculate_distance(self, bbox_width, bbox_height):
     
        try:
            # Sử dụng chiều rộng để tính khoảng cách
            distance_by_width = (self.logo_real_width * self.focal_length) / bbox_width
            
            # Sử dụng chiều cao để tính khoảng cách  
            distance_by_height = (self.logo_real_height * self.focal_length) / bbox_height
            
            # Lấy trung bình để tăng độ chính xác
            distance = (distance_by_width + distance_by_height) / 2
            
            return distance
        except ZeroDivisionError:
            return 0.0

def main():
    print("TDTU Logo Detector - YOLOv8")
    print("=" * 40)
    
    detector = TDTULogoDetector()
    detector.detect_from_webcam()

if __name__ == "__main__":
    main()