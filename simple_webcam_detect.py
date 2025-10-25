from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO("yolov8n.pt")  # Có thể thay bằng model đã train: "runs/detect/train/weights/best.pt"

# Mở webcam
cap = cv2.VideoCapture(0)

print("🎓 TDTU Logo Detection đang chạy...")
print("📝 Nhấn 'q' để thoát")

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Không thể đọc frame từ webcam!")
        break
    
    # Detect objects
    results = model(frame, conf=0.5, verbose=False)
    
    # Vẽ kết quả lên frame
    annotated_frame = results[0].plot()
    
    # Thêm text hướng dẫn
    cv2.putText(annotated_frame, "TDTU Logo Detection - Nhan 'q' de thoat", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Hiển thị frame
    cv2.imshow('TDTU Logo Detection', annotated_frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("👋 Đã đóng webcam!")