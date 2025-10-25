from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

def train_model():
    """
    Train mô hình YOLOv8 với dataset TDTU logo
    """
    print("Bắt đầu train mô hình YOLOv8...")
    
    # Load mô hình YOLOv8 pretrained
    model = YOLO('yolov8n.pt')
    
    # Train mô hình với dataset
    results = model.train(
        data='LOGO-TDTU.v2i.yolov8/data.yaml',  # Đường dẫn đến file data.yaml
        epochs=100,                              # Số epochs train
        imgsz=640,                              # Kích thước ảnh input
        batch=16,                               # Batch size
        device='cpu',                           # Sử dụng CPU (có thể đổi thành 'cuda' nếu có GPU)
        project='runs/train',                   # Thư mục lưu kết quả train
        name='logo_tdtu_detection',             # Tên experiment
        save=True,                              # Lưu checkpoint
        plots=True,                             # Tạo các plots
        verbose=True                            # Hiển thị chi tiết quá trình train
    )
    
    print("Hoàn thành training!")
    return model

def test_model(model_path='runs/train/logo_tdtu_detection/weights/best.pt'):
    """
    Test mô hình đã train với test dataset
    """
    print("Bắt đầu test mô hình...")
    
    # Load mô hình đã train
    model = YOLO(model_path)
    
    # Validate trên test set
    results = model.val(
        data='LOGO-TDTU.v2i.yolov8/data.yaml',
        split='test'
    )
    
    print("Kết quả test:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

def predict_single_image(model_path='runs/train/logo_tdtu_detection/weights/best.pt', 
                        image_path='LOGO-TDTU.v2i.yolov8/test/images'):
    """
    Predict trên một ảnh cụ thể
    """
    print("Predict trên ảnh test...")
    
    # Load mô hình
    model = YOLO(model_path)
    
    # Lấy danh sách ảnh test
    if os.path.isdir(image_path):
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join(image_path, image_files[0])
        else:
            print("Không tìm thấy ảnh test!")
            return
    else:
        test_image = image_path
    
    # Predict
    results = model.predict(
        source=test_image,
        save=True,
        project='runs/predict',
        name='logo_detection_results',
        show_labels=True,
        show_conf=True
    )
    
    print(f"Kết quả predict đã được lưu trong runs/predict/logo_detection_results/")
    return results

def predict_webcam(model_path='runs/train/logo_tdtu_detection/weights/best.pt'):
    """
    Predict real-time từ webcam
    """
    print("Bắt đầu predict từ webcam...")
    
    # Load mô hình
    model = YOLO(model_path)
    
    # Predict từ webcam (source='0' là webcam mặc định)
    results = model.predict(
        source=0,
        show=True,
        save=False,
        stream=True
    )
    
    # Hiển thị kết quả real-time
    for result in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== YOLO v8 Training và Testing cho TDTU Logo Detection ===")
    
    # Kiểm tra xem đã có mô hình trained chưa
    trained_model_path = 'runs/train/logo_tdtu_detection/weights/best.pt'
    
    if not os.path.exists(trained_model_path):
        print("\n1. Training mô hình...")
        model = train_model()
    else:
        print("\nĐã tìm thấy mô hình đã train!")
    
    print("\n2. Testing mô hình...")
    test_results = test_model(trained_model_path)
    
    print("\n3. Predict trên ảnh test...")
    predict_single_image(trained_model_path)
    
    # Tùy chọn: Predict từ webcam
    choice = input("\nBạn có muốn test real-time từ webcam không? (y/n): ")
    if choice.lower() == 'y':
        print("\n4. Predict từ webcam (nhấn 'q' để thoát)...")
        predict_webcam(trained_model_path)
    
    print("\nHoàn thành!")