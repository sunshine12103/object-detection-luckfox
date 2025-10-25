import cv2
from ultralytics import YOLO
import numpy as np

class TDTULogoDetector:
    def __init__(self, model_path="runs/train/tdtu_v1_basic/weights/best.pt", data_path="LOGO-TDTU.v2i.yolov8/data.yaml"):
        print("Đang khởi tạo YOLO model...")
        try:
            self.model = YOLO(model_path)
            print(f"Đã load model: {model_path}")
        except:
            print(f"Không tìm thấy model trained, sử dụng model gốc...")
            self.model = YOLO("yolov8n.pt")
        self.data_path = data_path
        self.class_names = ['logo_tdtu']
        
        self.colors = [(0, 255, 0)]  
        
    def train_model(self, epochs=50, img_size=640):
     
        print("Bắt đầu training model...")
        try:
            results = self.model.train(
                data=self.data_path,
                epochs=epochs,
                imgsz=img_size,
                patience=10,
                save=True,
                device='cpu'  # Có thể đổi thành 'cuda' nếu có GPU
            )
            print("✅ Training hoàn thành!")
            return results
        except Exception as e:
            print(f"❌ Lỗi khi training: {e}")
            return None
    
    def detect_from_webcam(self, camera_id=0, confidence_threshold=0.5):
        """
        Detect logo TDTU từ webcam real-time
        """
        print("📹 Đang mở webcam...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Không thể mở webcam!")
            return
        
        print("✅ Webcam đã sẵn sàng!")
        print("📝 Nhấn 'q' để thoát, 'r' để reset, 's' để chụp ảnh")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame từ webcam!")
                break
            
            frame_count += 1
            
            # Detect objects trong frame
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            
            # Vẽ kết quả lên frame
            annotated_frame = self.draw_results(frame, results)
            
            # Hiển thị FPS
            fps_text = f"Frame: {frame_count}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hiển thị frame
            cv2.imshow('TDTU Logo Detection', annotated_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                frame_count = 0
                print("🔄 Reset frame counter")
            elif key == ord('s'):
                filename = f'screenshot_{frame_count}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Đã lưu ảnh: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Đã đóng webcam!")
    
    def draw_results(self, frame, results):
        """
        Vẽ bounding box và label lên frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Chỉ hiển thị nếu confidence đủ cao
                    if confidence > 0.3:
                        # Vẽ bounding box
                        color = self.colors[0]  # Màu xanh lá cho logo TDTU
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Tạo label
                        label = f'LOGO TDTU: {confidence:.2f}'
                        
                        # Vẽ background cho text
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        
                        # Vẽ text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        print(f"🎯 Phát hiện logo TDTU! Confidence: {confidence:.2f}")
        
        return annotated_frame
    
    def test_on_images(self, image_folder="LOGO-TDTU.v2i.yolov8/test/images"):
        """
        Test model trên các ảnh trong thư mục test
        """
        import os
        print(f"🧪 Testing model trên ảnh trong thư mục: {image_folder}")
        
        if not os.path.exists(image_folder):
            print(f"❌ Không tìm thấy thư mục: {image_folder}")
            return
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("❌ Không tìm thấy ảnh nào trong thư mục!")
            return
        
        for img_file in image_files[:5]:  # Test 5 ảnh đầu
            img_path = os.path.join(image_folder, img_file)
            print(f"📄 Testing: {img_file}")
            
            # Load và predict
            results = self.model(img_path, conf=0.5)
            
            # Hiển thị kết quả
            for result in results:
                result.show()
                
        print("✅ Test hoàn thành!")

def main():
    """
    Hàm main để chạy detector
    """
    print("🎓 TDTU Logo Detector - YOLOv8")
    print("=" * 40)
    
    detector = TDTULogoDetector()
    
    while True:
        print("\nChọn chức năng:")
        print("1. Train model với dataset TDTU")
        print("2. Detect logo từ webcam")
        print("3. Test model trên ảnh mẫu")
        print("4. Thoát")
        
        choice = input("Nhập lựa chọn (1-4): ").strip()
        
        if choice == "1":
            epochs = input("Số epochs (mặc định 50): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 50
            detector.train_model(epochs=epochs)
            
        elif choice == "2":
            camera_id = input("Camera ID (mặc định 0): ").strip()
            camera_id = int(camera_id) if camera_id.isdigit() else 0
            
            confidence = input("Confidence threshold (mặc định 0.5): ").strip()
            confidence = float(confidence) if confidence else 0.5
            
            detector.detect_from_webcam(camera_id=camera_id, 
                                      confidence_threshold=confidence)
            
        elif choice == "3":
            detector.test_on_images()
            
        elif choice == "4":
            print("👋 Tạm biệt!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()