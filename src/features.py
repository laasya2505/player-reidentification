
import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging

class FeatureExtractor:
    """Extract features for player re-identification"""
    
    def __init__(self, config):
        self.color_bins = config.get('color_bins', 32)
        self.use_hog = config.get('use_hog_features', False)
        self.use_lbp = config.get('use_lbp_features', False)
        
        # ROI configuration
        self.roi_padding = config.get('roi_padding', 5)
        self.min_roi_size = config.get('min_roi_size', 20)
        
        logging.debug(f"FeatureExtractor initialized with {self.color_bins} color bins")
    
    def extract_features(self, frame, detection) -> Dict[str, Any]:
        """
        Extract features from detection region
        
        Args:
            frame: Input frame
            detection: Detection object
            
        Returns:
            Dictionary of features
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Add padding and ensure bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1 - self.roi_padding)
        y1 = max(0, y1 - self.roi_padding)
        x2 = min(w, x2 + self.roi_padding)
        y2 = min(h, y2 + self.roi_padding)
        
        # Crop player region
        player_roi = frame[y1:y2, x1:x2]
        
        if player_roi.size == 0 or min(player_roi.shape[:2]) < self.min_roi_size:
            return self._get_default_features()
        
        features = {}
        
        # Color histogram features
        features['color_hist'] = self._extract_color_histogram(player_roi)
        
        # Spatial features
        features['bbox_ratio'] = self._safe_divide(x2 - x1, y2 - y1)
        features['bbox_area'] = detection.get_area()
        features['bbox_width'] = x2 - x1
        features['bbox_height'] = y2 - y1
        
        # Position features (normalized by frame size)
        features['center_x_norm'] = detection.center[0] / w
        features['center_y_norm'] = detection.center[1] / h
        
        # Additional visual features
        if self.use_hog:
            features['hog'] = self._extract_hog_features(player_roi)
        
        if self.use_lbp:
            features['lbp'] = self._extract_lbp_features(player_roi)
        
        # Dominant color
        features['dominant_color'] = self._extract_dominant_color(player_roi)
        
        # Texture features
        features['texture_variance'] = self._extract_texture_variance(player_roi)
        
        return features
    
    def _extract_color_histogram(self, roi):
        """Extract color histogram from ROI"""
        try:
            # Convert to HSV for better color representation
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            h_hist = cv2.calcHist([hsv_roi], [0], None, [self.color_bins], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [self.color_bins], [0, 256])
            v_hist = cv2.calcHist([hsv_roi], [2], None, [self.color_bins], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Concatenate histograms
            color_hist = np.concatenate([h_hist, s_hist, v_hist])
            
            return color_hist
            
        except Exception as e:
            logging.warning(f"Color histogram extraction failed: {e}")
            return np.zeros(self.color_bins * 3)
    
    def _extract_dominant_color(self, roi):
        """Extract dominant color using k-means clustering"""
        try:
            # Reshape image to be a list of pixels
            data = roi.reshape((-1, 3))
            data = np.float32(data)
            
            # Apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3  # Number of clusters
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8 and find the most common cluster
            centers = np.uint8(centers)
            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            dominant_color = centers[dominant_cluster]
            
            return dominant_color.flatten()
            
        except Exception as e:
            logging.warning(f"Dominant color extraction failed: {e}")
            return np.array([0, 0, 0])
    
    def _extract_texture_variance(self, roi):
        """Extract texture variance features"""
        try:
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate local variance using a sliding window
            kernel = np.ones((3, 3), np.float32) / 9
            mean = cv2.filter2D(gray_roi.astype(np.float32), -1, kernel)
            variance = cv2.filter2D((gray_roi.astype(np.float32) - mean)**2, -1, kernel)
            
            # Return mean variance as texture feature
            return np.mean(variance)
            
        except Exception as e:
            logging.warning(f"Texture variance extraction failed: {e}")
            return 0.0
    
    def _extract_hog_features(self, roi):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        try:
            # Resize ROI to standard size
            roi_resized = cv2.resize(roi, (64, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor(
                _winSize=(64, 128),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9
            )
            
            # Compute HOG features
            features = hog.compute(gray)
            return features.flatten()
            
        except Exception as e:
            logging.warning(f"HOG feature extraction failed: {e}")
            return np.zeros(3780)  # Default HOG feature length
    
    def _extract_lbp_features(self, roi):
        """Extract Local Binary Pattern features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Simple LBP implementation
            rows, cols = gray.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray[i, j]
                    code = 0
                    
                    # 8-neighbor LBP
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            # Calculate histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            return hist
            
        except Exception as e:
            logging.warning(f"LBP feature extraction failed: {e}")
            return np.zeros(256)
    
    def _safe_divide(self, a, b):
        """Safe division to avoid division by zero"""
        return a / b if b != 0 else 1.0
    
    def _get_default_features(self):
        """Return default features when ROI is invalid"""
        features = {
            'color_hist': np.zeros(self.color_bins * 3),
            'bbox_ratio': 1.0,
            'bbox_area': 0,
            'bbox_width': 0,
            'bbox_height': 0,
            'center_x_norm': 0.5,
            'center_y_norm': 0.5,
            'dominant_color': np.array([0, 0, 0]),
            'texture_variance': 0.0
        }
        
        if self.use_hog:
            features['hog'] = np.zeros(3780)
        
        if self.use_lbp:
            features['lbp'] = np.zeros(256)
        
        return features
    
    def compare_features(self, features1, features2):
        """
        Compare two feature sets
        
        Returns:
            Distance between features (lower = more similar)
        """
        total_distance = 0.0
        weight_sum = 0.0
        
        # Color histogram comparison using chi-square distance
        hist1 = features1.get('color_hist', np.zeros(self.color_bins * 3))
        hist2 = features2.get('color_hist', np.zeros(self.color_bins * 3))
        
        if hist1.size > 0 and hist2.size > 0:
            # Avoid division by zero
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            chi_square = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2))
            color_distance = min(chi_square, 1.0)
            
            total_distance += 0.5 * color_distance
            weight_sum += 0.5
        
        # Dominant color comparison
        color1 = features1.get('dominant_color', np.array([0, 0, 0]))
        color2 = features2.get('dominant_color', np.array([0, 0, 0]))
        
        if color1.size > 0 and color2.size > 0:
            color_diff = np.linalg.norm(color1.astype(float) - color2.astype(float)) / 441.6  # Normalize by max possible distance
            total_distance += 0.2 * color_diff
            weight_sum += 0.2
        
        # Spatial feature comparison
        ratio1 = features1.get('bbox_ratio', 1.0)
        ratio2 = features2.get('bbox_ratio', 1.0)
        ratio_distance = abs(ratio1 - ratio2) / max(ratio1, ratio2, 1.0)
        
        total_distance += 0.1 * ratio_distance
        weight_sum += 0.1
        
        # Texture comparison
        texture1 = features1.get('texture_variance', 0.0)
        texture2 = features2.get('texture_variance', 0.0)
        texture_distance = abs(texture1 - texture2) / max(texture1, texture2, 1.0)
        
        total_distance += 0.1 * texture_distance
        weight_sum += 0.1
        
        # HOG comparison if available
        if self.use_hog and 'hog' in features1 and 'hog' in features2:
            hog1 = features1['hog']
            hog2 = features2['hog']
            
            if hog1.size > 0 and hog2.size > 0:
                hog_distance = np.linalg.norm(hog1 - hog2) / np.sqrt(hog1.size)
                total_distance += 0.1 * hog_distance
                weight_sum += 0.1
        
        # Normalize by total weight
        if weight_sum > 0:
            total_distance /= weight_sum
        
        return min(total_distance, 1.0)  # Clamp to [0, 1]
