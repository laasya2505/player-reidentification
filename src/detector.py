
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple
import logging

class Detection:
    """Single detection result"""
    
    def __init__(self, bbox, confidence, class_id, class_name="player"):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = self._calculate_center()
        self.area = self._calculate_area()
    
    def _calculate_center(self):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_area(self):
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_area(self):
        """Get bounding box area"""
        return self.area
    
    def get_width(self):
        """Get bounding box width"""
        return self.bbox[2] - self.bbox[0]
    
    def get_height(self):
        """Get bounding box height"""
        return self.bbox[3] - self.bbox[1]
    
    def get_aspect_ratio(self):
        """Get width/height ratio"""
        height = self.get_height()
        if height > 0:
            return self.get_width() / height
        return 1.0
    
    def is_valid(self):
        """Check if detection is valid"""
        return (self.get_width() > 0 and 
                self.get_height() > 0 and 
                self.confidence > 0)

class PlayerDetector:
    """YOLOv11-based player detector"""
    
    def __init__(self, model_config):
        self.model_path = model_config['path']
        self.conf_threshold = model_config['confidence_threshold']
        self.iou_threshold = model_config['iou_threshold']
        self.min_area = model_config.get('min_detection_area', 100)
        self.max_area = model_config.get('max_detection_area', 50000)
        
        # Device selection
        self.device = self._select_device()
        
        # Load YOLO model
        self._load_model()
        
        # Statistics
        self.total_detections = 0
        self.valid_detections = 0
    
    def _select_device(self):
        """Select best available device"""
        if torch.cuda.is_available():
            device = 'cuda'
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon
            logging.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = 'cpu'
            logging.info("Using CPU")
        
        return device
    
    def _load_model(self):
        """Load and validate YOLO model"""
        # Validate model file
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not self.model_path.endswith('.pt'):
            raise ValueError(f"Expected .pt file, got: {self.model_path}")
        
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # Move to selected device
            self.model.to(self.device)
            
            logging.info(f"✓ Successfully loaded model from {self.model_path}")
            logging.info(f"✓ Model classes: {self.model.names}")
            logging.info(f"✓ Device: {self.device}")
            
            # Store class names
            self.class_names = self.model.names
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def detect(self, frame) -> List[Detection]:
        """
        Detect players in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(
                frame, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                device=self.device,
                verbose=False  # Suppress output
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # Create detection object
                        detection = Detection(
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        
                        # Filter detections
                        if self._is_valid_detection(detection):
                            detections.append(detection)
                            self.valid_detections += 1
                        
                        self.total_detections += 1
            
            return detections
            
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return []
    
    def _is_valid_detection(self, detection):
        """Validate detection based on size and confidence"""
        # Check basic validity
        if not detection.is_valid():
            return False
        
        # Check area constraints
        area = detection.get_area()
        if area < self.min_area or area > self.max_area:
            return False
        
        # Check aspect ratio (players should be taller than wide)
        aspect_ratio = detection.get_aspect_ratio()
        if aspect_ratio > 2.0:  # Too wide to be a person
            return False
        
        return True
    
    def visualize_detections(self, frame, detections, show_confidence=True):
        """Draw detection boxes on frame"""
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            class_name = detection.class_name
            
            # Choose color based on confidence
            if confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1-label_height-10), 
                         (x1+label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_frame
    
    def get_statistics(self):
        """Get detection statistics"""
        if self.total_detections > 0:
            validity_rate = self.valid_detections / self.total_detections
        else:
            validity_rate = 0.0
        
        return {
            'total_detections': self.total_detections,
            'valid_detections': self.valid_detections,
            'validity_rate': validity_rate
        }
