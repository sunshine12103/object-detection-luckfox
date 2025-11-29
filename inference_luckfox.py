"""
Optimized YOLOv8 inference for Luckfox Omni 3576
Uses RKNN-Toolkit-Lite2 for NPU acceleration
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from rknnlite.api import RKNNLite

class LuckfoxDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Args:
            model_path: Path to .rknn model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Extract image size from filename
        try:
            self.imgsz = int(self.model_path.stem.split('_')[1])
        except:
            self.imgsz = 320
        
        # Distance calculation parameters (calibrate these!)
        self.logo_real_width = 7.5   # cm
        self.logo_real_height = 3.0  # cm
        self.focal_length = 1206.4   # pixels - NEED TO RECALIBRATE ON LUCKFOX
        
        # Initialize RKNN
        print("Initializing RKNN model...")
        self.rknn = RKNNLite()
        
        # Load model
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise Exception(f"Load RKNN model failed! ret={ret}")
        
        # Init runtime on NPU
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise Exception(f"Init runtime failed! ret={ret}")
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"✓ Input size: {self.imgsz}x{self.imgsz}")
        print(f"✓ NPU initialized")
    
    def preprocess(self, image):
        """Preprocess image for RKNN inference"""
        # Resize
        img_resized = cv2.resize(image, (self.imgsz, self.imgsz))
        
        # RKNN expects RGB uint8 format (normalization done in model)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def postprocess(self, outputs, orig_shape):
        """
        Postprocess RKNN outputs to get bounding boxes
        YOLOv8 output format: [batch, num_boxes, 5+num_classes]
        """
        predictions = outputs[0]  # Shape: [1, num_boxes, 5+num_classes]
        
        # Remove batch dimension
        predictions = predictions[0]  # Shape: [num_boxes, 5+num_classes]
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > self.conf_threshold:
                # Convert to xyxy format
                x1 = int((x_center - width / 2) * orig_shape[1] / self.imgsz)
                y1 = int((y_center - height / 2) * orig_shape[0] / self.imgsz)
                x2 = int((x_center + width / 2) * orig_shape[1] / self.imgsz)
                y2 = int((y_center + height / 2) * orig_shape[0] / self.imgsz)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]
        
        return boxes, scores, class_ids
    
    def calculate_distance(self, bbox_width, bbox_height):
        """Calculate distance from camera to logo"""
        try:
            distance_by_width = (self.logo_real_width * self.focal_length) / bbox_width
            distance_by_height = (self.logo_real_height * self.focal_length) / bbox_height
            distance = (distance_by_width + distance_by_height) / 2
            return distance
        except ZeroDivisionError:
            return 0.0
    
    def detect(self, image):
        """Run detection on single image"""
        orig_shape = image.shape[:2]
        
        # Preprocess
        img_preprocessed = self.preprocess(image)
        
        # Inference
        start_time = time.time()
        outputs = self.rknn.inference(inputs=[img_preprocessed])
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Postprocess
        boxes, scores, class_ids = self.postprocess(outputs, orig_shape)
        
        return boxes, scores, class_ids, inference_time
    
    def draw_results(self, image, boxes, scores, class_ids):
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            
            # Calculate distance
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            distance = self.calculate_distance(bbox_width, bbox_height)
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels
            label = f'LOGO TDTU: {score:.2f}'
            distance_label = f'Distance: {distance:.1f}cm'
            
            # Confidence label
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w1, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Distance label
            (w2, h2), _ = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y2), (x1 + w2, y2 + 20), color, -1)
            cv2.putText(annotated, distance_label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated
    
    def run_webcam(self, camera_id=0):
        """Run real-time detection from webcam"""
        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise Exception("Cannot open camera!")
        
        print("Camera ready! Press 'q' to quit")
        
        fps_history = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                boxes, scores, class_ids, inference_time = self.detect(frame)
                
                # Draw results
                annotated = self.draw_results(frame, boxes, scores, class_ids)
                
                # Calculate FPS
                fps = 1000 / inference_time if inference_time > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = np.mean(fps_history)
                
                # Draw FPS
                fps_text = f'FPS: {avg_fps:.1f} | Inference: {inference_time:.1f}ms'
                cv2.putText(annotated, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show
                cv2.imshow('Luckfox TDTU Logo Detection', annotated)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nAverage FPS: {np.mean(fps_history):.2f}")
    
    def run_image(self, image_path):
        """Run detection on single image"""
        print(f"Processing image: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise Exception(f"Cannot read image: {image_path}")
        
        # Detect
        boxes, scores, class_ids, inference_time = self.detect(image)
        
        # Draw results
        annotated = self.draw_results(image, boxes, scores, class_ids)
        
        print(f"Detections: {len(boxes)}")
        print(f"Inference time: {inference_time:.2f}ms")
        
        # Save result
        output_path = Path(image_path).stem + "_result.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"Saved to: {output_path}")
        
        # Display
        cv2.imshow('Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'rknn'):
            self.rknn.release()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 inference on Luckfox')
    parser.add_argument('--model', type=str,
                       default='exported_models/best_320_int8.rknn',
                       help='Path to RKNN model')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or image path)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    print("YOLOv8 TDTU Logo Detection - Luckfox Omni 3576")
    print("=" * 60)
    
    # Initialize detector
    detector = LuckfoxDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection
    if args.source.isdigit():
        # Webcam
        detector.run_webcam(camera_id=int(args.source))
    else:
        # Image file
        detector.run_image(args.source)


if __name__ == "__main__":
    main()
