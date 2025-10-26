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
                print("Khong the doc frame tu webcam!")
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
                        # Ve bounding box
                        color = self.colors[0]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Tao label
                        label = f'LOGO TDTU: {confidence:.2f}'
                        
                        # Ve background cho text
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        
                        # Ve text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        print(f"Confidence: {confidence:.2f}")
        
        return annotated_frame

def main():
    print("TDTU Logo Detector - YOLOv8")
    print("=" * 40)
    
    detector = TDTULogoDetector()
    detector.detect_from_webcam()

if __name__ == "__main__":
    main()