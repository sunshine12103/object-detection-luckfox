"""
🚀 TDTU Logo Detector - Training Script
=====================================
Script riêng để train mô hình YOLOv8 detect logo TDTU
Tách biệt hoàn toàn khỏi main.py
"""

from ultralytics import YOLO
import os
import yaml
from datetime import datetime

def setup_training_config():
    """Cấu hình các tham số training"""
    config = {
        'model_path': 'yolov8n.pt',  # Mô hình base
        'data_yaml': 'LOGO-TDTU.v2i.yolov8/data.yaml',  # Dataset config
        'epochs': 150,  # Số epochs
        'batch_size': 16,  # Batch size
        'imgsz': 640,  # Image size
        'lr0': 0.01,  # Learning rate
        'patience': 30,  # Early stopping patience
        'save_period': 10,  # Save checkpoint mỗi 10 epochs
        'device': 'cpu',  # Sử dụng CPU (có thể đổi thành 'cuda' nếu có GPU)
        'workers': 4,  # Số worker threads
        'project': 'runs/train',  # Thư mục lưu kết quả
        'name': 'tdtu_logo_detector',  # Tên experiment
    }
    return config

def check_dataset():
    """Kiểm tra dataset có tồn tại không"""
    data_yaml = 'LOGO-TDTU.v2i.yolov8/data.yaml'
    
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy file {data_yaml}")
        return False
    
    # Đọc config dataset
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"✅ Dataset config loaded:")
    print(f"   • Classes: {data_config.get('nc', 0)}")
    print(f"   • Names: {data_config.get('names', [])}")
    
    # Kiểm tra thư mục train/valid/test
    base_dir = 'LOGO-TDTU.v2i.yolov8'
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"   • {split.capitalize()}: {img_count} images, {label_count} labels")
        else:
            print(f"   ⚠️  {split.capitalize()}: Thư mục không tồn tại")
    
    return True

def train_model():
    """Hàm chính để train mô hình"""
    print("🚀 Bắt đầu training TDTU Logo Detector")
    print("=" * 60)
    
    # Lấy config
    config = setup_training_config()
    
    # Hiển thị config
    print("📊 Cấu hình training:")
    for key, value in config.items():
        print(f"   • {key}: {value}")
    print("-" * 60)
    
    # Kiểm tra dataset
    if not check_dataset():
        print("❌ Dataset không hợp lệ. Thoát chương trình.")
        return
    
    print("-" * 60)
    
    try:
        # Khởi tạo mô hình
        print(f"🔄 Loading mô hình base: {config['model_path']}")
        model = YOLO(config['model_path'])
        
        # Bắt đầu training
        print("🎯 Bắt đầu quá trình training...")
        start_time = datetime.now()
        
        results = model.train(
            data=config['data_yaml'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            lr0=config['lr0'],
            patience=config['patience'],
            save_period=config['save_period'],
            device=config['device'],
            workers=config['workers'],
            project=config['project'],
            name=config['name'],
            verbose=True,  # Hiển thị thông tin chi tiết
            save=True,  # Lưu checkpoints
            plots=True,  # Tạo plots
            val=True,  # Validation trong quá trình training
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print("🎉 Training hoàn thành!")
        print("=" * 60)
        print(f"⏱️  Thời gian training: {training_time}")
        print(f"📁 Kết quả được lưu tại: {config['project']}/{config['name']}")
        print(f"🏆 Mô hình tốt nhất: {config['project']}/{config['name']}/weights/best.pt")
        print(f"💾 Mô hình cuối cùng: {config['project']}/{config['name']}/weights/last.pt")
        
        # Hiển thị một số metrics từ kết quả training
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\n📊 Kết quả training:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"   • mAP50: {metrics['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   • mAP50-95: {metrics['metrics/mAP50-95(B)']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {str(e)}")
        return False

def validate_model(model_path=None):
    """Validate mô hình đã train"""
    if model_path is None:
        model_path = "runs/train/tdtu_logo_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình: {model_path}")
        return
    
    print(f"🔍 Đang validate mô hình: {model_path}")
    
    try:
        model = YOLO(model_path)
        results = model.val(data='LOGO-TDTU.v2i.yolov8/data.yaml')
        
        print("✅ Validation hoàn thành!")
        print(f"📊 mAP50: {results.box.map50:.3f}")
        print(f"📊 mAP50-95: {results.box.map:.3f}")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình validation: {str(e)}")

if __name__ == "__main__":
    print("🎓 TDTU Logo Detector - Training Script")
    print("========================================")
    
    # Menu lựa chọn
    while True:
        print("\n🔧 Chọn hành động:")
        print("1. 🚀 Train mô hình mới")
        print("2. 🔍 Validate mô hình đã train")
        print("3. 📊 Kiểm tra dataset")
        print("4. ❌ Thoát")
        
        choice = input("\nNhập lựa chọn (1-4): ").strip()
        
        if choice == "1":
            success = train_model()
            if success:
                print("\n🎉 Training thành công! Bạn có thể sử dụng mô hình để detect.")
        
        elif choice == "2":
            model_path = input("Nhập đường dẫn mô hình (Enter để dùng mặc định): ").strip()
            validate_model(model_path if model_path else None)
        
        elif choice == "3":
            check_dataset()
        
        elif choice == "4":
            print("👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")