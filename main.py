"""
TDTU Logo Detector - Main Detection Script
==========================================
Script chính để detect logo TDTU từ webcam hoặc ảnh
Tách biệt hoàn toàn khỏi training - chỉ tập trung vào detection
"""

from ultralytics import YOLO
import cv2
import os

def detect_from_webcam():
    """Detect logo TDTU từ webcam real-time"""
    print("Chế độ Webcam Detection")
    print("-" * 40)
    
    # Kiểm tra mô hình custom đã train
    custom_model = "runs/train/tdtu_logo_detector/weights/best.pt"
    alternative_model = "runs/train/logo_tdtu4/weights/best.pt"
    
    if os.path.exists(custom_model):
        print(f"Sử dụng mô hình custom: {custom_model}")
        model = YOLO(custom_model)
    elif os.path.exists(alternative_model):
        print(f"Sử dụng mô hình đã train: {alternative_model}")
        model = YOLO(alternative_model)
    else:
        print("Không tìm thấy mô hình custom, sử dụng mô hình base YOLOv8")
        print("Hãy chạy train_tdtu_logo.py để train mô hình custom")
        model = YOLO("yolov8n.pt")
    
    print("Bắt đầu detect từ webcam...")
    print("Nhấn 'q' hoặc ESC để thoát")
    
    # Detect từ webcam
    try:
        results = model.predict(
            source=0,  # Webcam
            show=True,  # Hiển thị real-time
            save=False,  # Không lưu video
            conf=0.3,  # Confidence threshold thấp hơn để detect dễ hơn
            verbose=False  # Ít thông tin debug
        )
        print("Webcam detection hoàn thành!")
    except Exception as e:
        print(f"Lỗi khi detect từ webcam: {e}")

def detect_from_image(image_path):
    """Detect logo TDTU từ ảnh"""
    print("Chế độ Image Detection")
    print("-" * 40)
    
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return
    
    # Kiểm tra mô hình custom
    custom_model = "runs/train/tdtu_logo_detector/weights/best.pt"
    alternative_model = "runs/train/logo_tdtu4/weights/best.pt"
    
    if os.path.exists(custom_model):
        print(f"Sử dụng mô hình custom: {custom_model}")
        model = YOLO(custom_model)
    elif os.path.exists(alternative_model):
        print(f"Sử dụng mô hình đã train: {alternative_model}")
        model = YOLO(alternative_model)
    else:
        print("Sử dụng mô hình base YOLOv8")
        model = YOLO("yolov8n.pt")
    
    print(f"Đang detect ảnh: {image_path}")
    
    try:
        # Detect từ ảnh
        results = model.predict(
            source=image_path,
            show=True,  # Hiển thị kết quả
            save=True,  # Lưu kết quả
            conf=0.3,  # Confidence threshold
            project="runs/detect",  # Thư mục lưu
            name="image_results"  # Tên folder
        )
        
        print("Detect hoàn thành! Kết quả được lưu trong runs/detect/image_results/")
        
        # Hiển thị thông tin detect
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Phát hiện {len(boxes)} đối tượng!")
                for i, box in enumerate(boxes):
                    conf = box.conf.item()
                    print(f"   • Đối tượng {i+1}: Confidence = {conf:.2f}")
            else:
                print("Không phát hiện đối tượng nào!")
                
    except Exception as e:
        print(f"Lỗi khi detect ảnh: {e}")

def main():
    """Menu chính"""
    print("TDTU Logo Detector - Main Detection Script")
    print("=" * 50)
    print("Chỉ tập trung vào detection - không train")
    print("Để train mô hình, hãy chạy: python train_tdtu_logo.py")
    print("=" * 50)
    
    while True:
        print("\nChọn chế độ detect:")
        print("1. Detect từ webcam (real-time)")
        print("2. Detect từ ảnh local")
        print("3. Chuyển sang training (train_tdtu_logo.py)")
        print("4. Thoát")
        
        choice = input("\nNhập lựa chọn (1-4): ").strip()
        
        if choice == "1":
            detect_from_webcam()
        
        elif choice == "2":
            image_path = input("Nhập đường dẫn ảnh: ").strip()
            if image_path:
                detect_from_image(image_path)
            else:
                print("Đường dẫn không hợp lệ!")
            
        elif choice == "3":
            print("Để train mô hình, hãy chạy lệnh:")
            print("   python train_tdtu_logo.py")
            
            # Hỏi có muốn chạy luôn không
            run_now = input("Chạy training ngay bây giờ? (y/n): ").strip().lower()
            if run_now == 'y':
                try:
                    import subprocess
                    subprocess.run(["python", "train_tdtu_logo.py"])
                except Exception as e:
                    print(f"Không thể chạy script training: {e}")
        
        elif choice == "4":
            print("Tạm biệt!")
            break
        
        else:
            print("Lựa chọn không hợp lệ! Vui lòng chọn 1-4.")

if __name__ == "__main__":
    main()