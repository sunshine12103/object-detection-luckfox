# TDTU Logo Detection with YOLOv8

Dự án nhận diện logo TDTU sử dụng YOLOv8 với webcam real-time.

## 🚀 Cài đặt

1. Clone repository:
```bash
git clone https://github.com/sunshine12103/object-detection-logo-TDTU.git
cd object-detection-logo-TDTU
```

2. Tạo virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# hoặc source venv/bin/activate  # Linux/Mac
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc dự án

```
├── main.py                     # Script chính - Detection từ webcam/ảnh  
├── train_tdtu_logo.py         # Script training riêng biệt
├── yolov8n.pt                 # Mô hình YOLOv8 gốc
├── requirements.txt           # Dependencies Python
├── README.md                  # Hướng dẫn sử dụng
├── .gitignore                 # Git ignore file
├── LOGO-TDTU.v2i.yolov8/      # Dataset
│   ├── data.yaml              # Cấu hình dataset
│   ├── train/                 # Ảnh training
│   ├── valid/                 # Ảnh validation  
│   └── test/                  # Ảnh test
└── runs/                      # Kết quả training/detection
    ├── train/                 # Kết quả training
    │   └── tdtu_logo_detector/
    │       └── weights/
    │           ├── best.pt    # Mô hình tốt nhất
    │           └── last.pt    # Mô hình cuối cùng
    └── detect/                # Kết quả detection
```

## 🎯 Sử dụng

### 1. Training mô hình
```bash
python train_tdtu_logo.py
```

### 2. Detection (webcam/ảnh) 
```bash
python main.py
```

**Hoặc chạy trực tiếp:**
- Webcam: `python main.py` → chọn option 1
- Ảnh: `python main.py` → chọn option 2

## 📊 Kết quả Training

- **Epochs**: 150
- **mAP50**: ~37.6%
- **mAP50-95**: ~10.3%
- **Model size**: 6.2MB

## 🎮 Controls

- **ESC** hoặc **Q**: Thoát chương trình
- **SPACE**: Tạm dừng/tiếp tục

## 📋 Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
- Webcam

## 🔧 Cấu hình

Chỉnh sửa `data.yaml` để thay đổi đường dẫn dataset hoặc số lượng classes.

## 📈 Cải thiện độ chính xác

1. Tăng số epochs training
2. Thêm data augmentation
3. Điều chỉnh confidence threshold
4. Thêm nhiều ảnh training

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới
3. Commit changes
4. Push to branch  
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

## 🔌 Embedded Deployment (Luckfox Omni 3576)

### Deploy YOLOv8 lên Luckfox với NPU acceleration

**Quick Start:**
```bash
# 1. Export ONNX
python export_onnx.py --model runs/train/logo_tdtu_v5/weights/best.pt --imgsz 320

# 2. Convert to RKNN
python export_rknn.py --onnx exported_models/best_320.onnx

# 3. Transfer to Luckfox
scp exported_models/best_320_int8.rknn root@luckfox_ip:/root/
scp inference_luckfox.py root@luckfox_ip:/root/

# 4. Run on Luckfox
python inference_luckfox.py --model best_320_int8.rknn --source 0
```

**Performance:**
- FPS: 15-25 (INT8, 320x320)
- Model size: ~2MB
- NPU accelerated

**Hướng dẫn chi tiết:**
- [QUICKSTART_LUCKFOX.md](./QUICKSTART_LUCKFOX.md) - Hướng dẫn nhanh
- [DEPLOYMENT_LUCKFOX.md](./DEPLOYMENT_LUCKFOX.md) - Hướng dẫn đầy đủ + troubleshooting