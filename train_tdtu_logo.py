from ultralytics import YOLO
import os

def train_model():
    try:
        print("Bắt đầu train")
        # Load model
        model = YOLO("yolov8n.pt")
        
        results = model.train(
            data="LOGO-TDTU.v5i.yolov8/data.yaml",
            epochs=150,
            batch=16,
            imgsz=640,
            lr0=0.001,
            device='cpu',
            project='runs/train',
            name='logo_tdtu_v5',
            exist_ok=True,
            plots=True,
            save=True,
            val=True,
            patience=50,
            cos_lr=True,
            close_mosaic=10
        )
        
        print(f"\n Train xong")
        
        return results
        
    except Exception as e:
        print("lỗi")
        return None

def validate_model():
    try:
        print("\n Đang validate model...")
        print("=" * 50)
        
        # Tìm model mới nhất
        best_model_path = "runs/train/logo_tdtu_v5/weights/best.pt"

        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
            results = model.val(data="LOGO-TDTU.v5i.yolov8/data.yaml")
            
            print("\nKẾT QUẢ VALIDATION:")
            print("=" * 50)
            
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                
                print(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
                print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
                print(f"Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
                print(f"Recall: {metrics.get('metrics/recall(B)', 0):.3f}")
                print(f"F1-Score: {(2 * metrics.get('metrics/precision(B)', 0) * metrics.get('metrics/recall(B)', 0)) / (metrics.get('metrics/precision(B)', 0) + metrics.get('metrics/recall(B)', 0) + 1e-6):.3f}")
            else:
                print(f"mAP50: {getattr(results, 'box_map50', 'N/A')}")
                print(f"mAP50-95: {getattr(results, 'box_map', 'N/A')}")
                print(f"Precision: {getattr(results, 'box_p', 'N/A')}")
                print(f"Recall: {getattr(results, 'box_r', 'N/A')}")
            
            print("=" * 50)
            print("Validation hoàn thành!")
            
            return results
        else:
            print(f"Không tìm thấy model: {best_model_path}")
            return None
            
    except Exception as e:
        print(f"Lỗi trong quá trình validation: {e}")
        return None

def test_model():
    try:
        print("\n Đang test model với test set...")
        
        best_model_path = "runs/train/logo_tdtu_v5/weights/best.pt"
        
        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
            
            test_results = model.predict(
                source="LOGO-TDTU.v5i.yolov8/test/images",
                save=True,
                project="runs/predict",
                name="test_results_v5",
                exist_ok=True
            )
            
            print(f"Testing hoàn thành!")
            print(f"Kết quả được lưu tại: runs/predict/test_results_v5/")
            return test_results
        else:
            print(f"Không tìm thấy model: {best_model_path}")
            return None
            
    except Exception as e:
        print(f"Lỗi trong quá trình testing: {e}")
        return None

if __name__ == "__main__":
    print("TDTU Logo Detector")
    print("=" * 50)
    
    # 1. Train model
    train_results = train_model()
    
    if train_results:
        # 2. Validate model
        val_results = validate_model()
        
        # 3. Test model
        test_results = test_model()

        print("\nHoàn thành toàn bộ quá trình training, validation và test")
    print("\nHoàn thành!")