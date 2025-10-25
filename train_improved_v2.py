from ultralytics import YOLO
import os

def train_model():
    """Train YOLOv8 model với 150 epochs"""
    try:
        print("🚀 Bắt đầu training TDTU Logo Detector - 150 Epochs")
        print("=" * 60)
        
        print("📊 Cấu hình training:")
        print(f"   • Epochs: 150")
        print(f"   • Batch size: 16")
        print(f"   • Image size: 640")
        print(f"   • Learning rate: 0.01")
        print(f"   • Device: cpu")
        print("-" * 60)
        
        # Load model
        model = YOLO("yolov8n.pt")
        
        # Train với cấu hình đơn giản
        results = model.train(
            data="LOGO-TDTU.v2i.yolov8/data.yaml",
            epochs=150,
            batch=16,
            imgsz=640,
            lr0=0.01,
            device='cpu',
            project='runs/train',
            name='logo_tdtu_150epochs',
            exist_ok=True,
            plots=True,
            save=True,
            val=True,
            patience=30,
            cos_lr=True,
            close_mosaic=10
        )
        
        print(f"\n✅ Training hoàn thành!")
        print(f"📁 Model được lưu tại: runs/train/logo_tdtu_150epochs/weights/")
        print(f"🏆 Best model: runs/train/logo_tdtu_150epochs/weights/best.pt")
        print(f"📊 Last model: runs/train/logo_tdtu_150epochs/weights/last.pt")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {e}")
        return None

def validate_model():
    """Validate model sau khi train"""
    try:
        print("\n🔍 Đang validate model...")
        
        # Tìm model mới nhất
        best_model_path = "runs/train/logo_tdtu_150epochs/weights/best.pt"
        
        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
            results = model.val(data="LOGO-TDTU.v2i.yolov8/data.yaml")
            print(f"✅ Validation hoàn thành!")
            return results
        else:
            print(f"❌ Không tìm thấy model: {best_model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi trong quá trình validation: {e}")
        return None

def test_model():
    """Test model với test set"""
    try:
        print("\n🧪 Đang test model với test set...")
        
        best_model_path = "runs/train/logo_tdtu_150epochs/weights/best.pt"
        
        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
            
            # Test với test images
            test_results = model.predict(
                source="LOGO-TDTU.v2i.yolov8/test/images",
                save=True,
                project="runs/predict",
                name="test_results_150epochs",
                exist_ok=True
            )
            
            print(f"✅ Testing hoàn thành!")
            print(f"📁 Kết quả được lưu tại: runs/predict/test_results_150epochs/")
            return test_results
        else:
            print(f"❌ Không tìm thấy model: {best_model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi trong quá trình testing: {e}")
        return None

if __name__ == "__main__":
    print("🎓 TDTU Logo Detector - 150 Epochs Training")
    print("=" * 50)
    
    # 1. Train model
    train_results = train_model()
    
    if train_results:
        # 2. Validate model
        val_results = validate_model()
        
        # 3. Test model
        test_results = test_model()
        
        print("\n🎉 Hoàn thành toàn bộ quá trình training, validation và testing!")
        print("📋 Để sử dụng model mới, cập nhật đường dẫn trong webcam_detect.py:")
        print("   model = YOLO('runs/train/logo_tdtu_150epochs/weights/best.pt')")
    
    print("\n✨ Hoàn thành!")