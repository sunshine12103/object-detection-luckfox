from ultralytics import YOLO
import os

# Chọn chế độ hoạt động
MODE = "train"  # Có thể thay đổi thành "predict", "webcam", hoặc "test"

if MODE == "train":
    print("Bắt đầu training mô hình YOLOv8 với dataset TDTU logo...")
    
    # Load mô hình pretrained
    model = YOLO("yolov8n.pt")
    
    # Train với dataset
    results = model.train(
        data='LOGO-TDTU.v2i.yolov8/data.yaml',
        epochs=50,                    # Giảm epochs để test nhanh
        imgsz=640,
        batch=8,                      # Giảm batch size cho máy yếu
        device='cpu',                 # Dùng CPU
        project='runs/train',
        name='logo_tdtu',
        save=True,
        plots=True
    )
    
elif MODE == "predict":
    # Predict với mô hình đã train
    model_path = 'runs/train/logo_tdtu/weights/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        results = model.predict(
            source='LOGO-TDTU.v2i.yolov8/test/images',
            save=True,
            project='runs/predict',
            name='results'
        )
    else:
        print("Chưa có mô hình trained! Hãy chạy training trước.")

elif MODE == "webcam":
    # Predict từ webcam
    model_path = 'runs/train/logo_tdtu/weights/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
        from ultralytics import YOLO
import cv2

def train_tdtu_model():
    """Train model với dataset TDTU logo"""
    print("🎯 Bắt đầu training model TDTU...")
    model = YOLO("yolov8n.pt")
    
    # Train model
    results = model.train(
        data="LOGO-TDTU.v2i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        patience=10,
        save=True,
        device='cpu'
    )
    print("✅ Training hoàn thành!")
    return results

def webcam_detect():
    """Detect logo TDTU từ webcam"""
    print("📹 Đang khởi động webcam detection...")
    
    # Load model (có thể dùng model đã train hoặc pretrained)
    try:
        # Thử load model đã train trước
        model = YOLO("runs/detect/train/weights/best.pt")
        print("✅ Đã load model đã train!")
    except:
        # Nếu chưa có model train, dùng pretrained
        model = YOLO("yolov8n.pt")
        print("⚠️  Dùng pretrained model. Nên train model trước để detect logo TDTU chính xác hơn!")
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    print("🎓 TDTU Logo Detection đang chạy...")
    print("📝 Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = model(frame, conf=0.5, verbose=False)
        
        # Vẽ kết quả
        annotated_frame = results[0].plot()
        
        # Thêm text
        cv2.putText(annotated_frame, "TDTU Logo Detection - Nhan 'q' de thoat", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow('TDTU Logo Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã đóng webcam!")

def main():
    """Menu chính"""
    print("🎓 TDTU Logo Detector - YOLOv8")
    print("=" * 40)
    
    while True:
        print("\n🔧 Chọn chức năng:")
        print("1. Train model với dataset TDTU")
        print("2. Detect logo từ webcam")
        print("3. Test predict từ URL")
        print("4. Thoát")
        
        choice = input("➡️  Nhập lựa chọn (1-4): ").strip()
        
        if choice == "1":
            train_tdtu_model()
        elif choice == "2":
            webcam_detect()
        elif choice == "3":
            # Test predict từ URL như code cũ
            model = YOLO("yolov8n.pt")
            results = model.predict(source="https://nextcity.org/images/made/219951734_2838e034bb_o_840_630_80.jpg")
            print(results)
        elif choice == "4":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
    if os.path.exists(model_path):
        model = YOLO(model_path)
        results = model.val(data='LOGO-TDTU.v2i.yolov8/data.yaml')
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
    else:
        print("Chưa có mô hình trained! Hãy chạy training trước.")
