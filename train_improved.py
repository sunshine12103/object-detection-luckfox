#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDTU Logo Detector - Improved Training
Train YOLOv8 với nhiều epochs hơn để độ chính xác cao hơn
"""

from ultralytics import YOLO
import os

def train_improved_model():
    """Train YOLOv8 model với cấu hình tối ưu"""
    
    print("🚀 Bắt đầu training TDTU Logo Detector - Improved Version")
    print("=" * 60)
    
    # Load pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # Hoặc yolov8s.pt cho model lớn hơn
    
    # Training parameters - Tăng epochs và tối ưu tham số
    training_config = {
        'data': 'LOGO-TDTU.v2i.yolov8/data.yaml',  # Đường dẫn tới dataset
        'epochs': 150,  # Tăng từ 50 lên 150 epochs
        'patience': 30,  # Early stopping patience
        'batch': 16,  # Batch size
        'imgsz': 640,  # Image size
        'save': True,  # Save checkpoints
        'save_period': 25,  # Save every 25 epochs
        'cache': False,  # Cache images for faster training
        'device': 'cpu',  # Sử dụng CPU (có thể đổi thành 'cuda' nếu có GPU)
        'workers': 4,  # Number of worker threads
        'project': 'runs/train',  # Project directory
        'name': 'logo_tdtu_improved',  # Experiment name
        'exist_ok': True,  # Overwrite existing
        'pretrained': True,  # Use pretrained weights
        'optimizer': 'SGD',  # Optimizer (SGD, Adam, AdamW)
        'verbose': True,  # Verbose output
        'seed': 42,  # Random seed
        'deterministic': True,  # Reproducible results
        'single_cls': False,  # Single class training
        'rect': False,  # Rectangular training
        'resume': False,  # Resume from checkpoint
        'nosave': False,  # Save final checkpoint
        'noval': False,  # Skip validation
        'noautoanchor': False,  # Disable autoanchor
        'evolve': False,  # Evolve hyperparameters
        'bucket': '',  # gsutil bucket
        'cache_ram': False,  # Cache images in RAM
        'image_weights': False,  # Use weighted image selection
        'multi_scale': False,  # Multi-scale training
        'overlap_mask': True,  # Overlap masks
        'mask_ratio': 4,  # Mask ratio
        'dropout': 0.0,  # Dropout rate
        'val': True,  # Validate during training
        
        # Learning rate parameters
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        
        # Augmentation parameters
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
        'degrees': 0.0,  # Rotation degrees
        'translate': 0.1,  # Translation
        'scale': 0.5,  # Scale
        'shear': 0.0,  # Shear
        'perspective': 0.0,  # Perspective
        'flipud': 0.0,  # Flip up-down
        'fliplr': 0.5,  # Flip left-right
        'mosaic': 1.0,  # Mosaic augmentation
        'mixup': 0.0,  # Mixup augmentation
        'copy_paste': 0.0  # Copy-paste augmentation
    }
    
    print(f"📊 Cấu hình training:")
    print(f"   • Epochs: {training_config['epochs']}")
    print(f"   • Batch size: {training_config['batch']}")
    print(f"   • Image size: {training_config['imgsz']}")
    print(f"   • Learning rate: {training_config['lr0']}")
    print(f"   • Device: {training_config['device']}")
    print("-" * 60)
    
    try:
        # Bắt đầu training
        results = model.train(**training_config)
        
        print("\n✅ Training hoàn thành!")
        print(f"📁 Kết quả được lưu tại: runs/train/{training_config['name']}")
        
        # Load best model for validation
        best_model_path = f"runs/train/{training_config['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"🏆 Best model: {best_model_path}")
            
            # Load best model và test
            best_model = YOLO(best_model_path)
            
            # Validate trên test set
            print("\n🧪 Đang validate model...")
            val_results = best_model.val(data='LOGO-TDTU.v2i.yolov8/data.yaml')
            
            print(f"\n📈 Kết quả validation:")
            print(f"   • mAP50: {val_results.box.map50:.3f}")
            print(f"   • mAP50-95: {val_results.box.map:.3f}")
            
            return best_model_path
        else:
            print("❌ Không tìm thấy best model!")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {e}")
        return None

def test_model_accuracy(model_path):
    """Test độ chính xác của model"""
    if not model_path or not os.path.exists(model_path):
        print("❌ Model không tồn tại!")
        return
    
    print(f"\n🔍 Testing model: {model_path}")
    model = YOLO(model_path)
    
    # Test trên các ảnh trong test set
    test_results = model.predict(
        source='LOGO-TDTU.v2i.yolov8/test/images',
        save=True,
        save_txt=True,
        save_conf=True,
        project='runs/predict',
        name='test_improved',
        exist_ok=True
    )
    
    print("✅ Test hoàn thành! Kết quả được lưu tại runs/predict/test_improved")

if __name__ == "__main__":
    print("🎓 TDTU Logo Detector - Improved Training")
    print("=" * 60)
    
    # Train model với cấu hình tối ưu
    best_model = train_improved_model()
    
    # Test model
    if best_model:
        test_model_accuracy(best_model)
        
        print("\n🎯 Để sử dụng model mới cho webcam detection:")
        print(f"   Cập nhật đường dẫn trong webcam_detect.py thành: {best_model}")
    
    print("\n✨ Hoàn thành!")