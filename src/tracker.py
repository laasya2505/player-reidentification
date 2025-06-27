
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
import logging
import cv2

from features import FeatureExtractor
from detector import Detection

class Track:
    """Individual player track"""
    
    def __init__(self, track_id, detection, features, frame_id):
        self.id = track_id
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.features = features
        self.last_frame = frame_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        
        # For velocity calculation
        self.position_history = [detection.center]
        self.velocity = (0.0, 0.0)
        self.max_history = 10
        
        # Track state
        self.state = 'active'  # 'active', 'lost', 'inactive'
        
        # Re-identification features
        self.appearance_features = []
        self.add_appearance_feature(features)
        
        # Track quality metrics
        self.avg_confidence = detection.confidence
        self.detection_count = 1
    
    def update(self, detection, features, frame_id):
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_frame = frame_id
        self.hits += 1
        self.time_since_update = 0
        self.state = 'active'
        self.detection_count += 1
        
        # Update average confidence
        self.avg_confidence = (
            (self.avg_confidence * (self.detection_count - 1) + detection.confidence) / 
            self.detection_count
        )
        
        # Update position history and velocity
        self.position_history.append(detection.center)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        self._update_velocity()
        
        # Update features
        self.features = features
        self.add_appearance_feature(features)
    
    def _update_velocity(self):
        """Update velocity based on position history"""
        if len(self.position_history) >= 2:
            current_pos = self.position_history[-1]
            prev_pos = self.position_history[-2]
            
            self.velocity = (
                current_pos[0] - prev_pos[0],
                current_pos[1] - prev_pos[1]
            )
        else:
            self.velocity = (0.0, 0.0)
    
    def predict_next_position(self, steps=1):
        """Predict next position based on velocity"""
        if not self.position_history:
            return (0, 0)
        
        current_pos = self.position_history[-1]
        predicted_pos = (
            current_pos[0] + self.velocity[0] * steps,
            current_pos[1] + self.velocity[1] * steps
        )
        return predicted_pos
    
    def add_appearance_feature(self, features):
        """Add appearance feature for re-identification"""
        self.appearance_features.append(features)
        
        # Keep only recent features
        max_features = 5
        if len(self.appearance_features) > max_features:
            self.appearance_features.pop(0)
    
    def get_average_appearance(self):
        """Get average appearance features"""
        if not self.appearance_features:
            return self.features
        
        # Simple averaging for now - could be improved
        return self.appearance_features[-1]  # Use most recent
    
    def mark_lost(self):
        """Mark track as lost"""
        self.time_since_update += 1
        self.state = 'lost'
    
    def get_speed(self):
        """Get current speed (magnitude of velocity)"""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    def is_stationary(self, threshold=5.0):
        """Check if track is stationary"""
        return self.get_speed() < threshold

class PlayerTracker:
    """Multi-object tracker with re-identification"""
    
    def __init__(self, tracking_config, reid_config):
        # Tracking configuration
        self.max_disappeared = tracking_config['max_disappeared_frames']
        self.max_distance = tracking_config['max_distance_threshold']
        self.feature_weights = tracking_config['feature_weights']
        
        # Re-identification configuration
        self.reid_timeout = reid_config['inactive_track_timeout']
        self.similarity_threshold = reid_config['similarity_threshold']
        
        # Track management
        self.active_tracks: Dict[int, Track] = {}
        self.inactive_tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(reid_config)
        
        # Statistics
        self.total_tracks_created = 0
        self.successful_reidentifications = 0
        
        logging.info("Player tracker initialized")
    
    def update(self, detections: List[Detection], frame) -> List[Track]:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        if not detections:
            # No detections - mark all active tracks as lost
            for track in self.active_tracks.values():
                track.mark_lost()
            self._cleanup_tracks()
            return list(self.active_tracks.values())
        
        # Extract features for all detections
        detection_features = []
        for detection in detections:
            features = self.feature_extractor.extract_features(frame, detection)
            detection_features.append(features)
        
        # Associate detections with existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections, detection_features
        )
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            self.active_tracks[track_id].update(
                detections[det_idx], 
                detection_features[det_idx], 
                self.frame_count
            )
        
        # Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            self.active_tracks[track_id].mark_lost()
        
        # Handle unmatched detections
        for det_idx in unmatched_detections:
            self._handle_unmatched_detection(
                detections[det_idx], 
                detection_features[det_idx]
            )
        
        # Cleanup tracks
        self._cleanup_tracks()
        
        return list(self.active_tracks.values())
    
    def _associate_detections(self, detections, features):
        """Associate detections with existing tracks using Hungarian algorithm"""
        if not self.active_tracks:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((len(detections), len(track_ids)), 1e6)
        
        for i, (detection, det_features) in enumerate(zip(detections, features)):
            for j, track_id in enumerate(track_ids):
                track = self.active_tracks[track_id]
                cost = self._calculate_association_cost(detection, det_features, track)
                cost_matrix[i, j] = cost
        
        # Solve assignment problem
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with high cost
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < self.max_distance:
                matched_pairs.append((det_idx, track_ids[track_idx]))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        # Convert unmatched track indices to track IDs
        unmatched_track_ids = [track_ids[idx] for idx in unmatched_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_track_ids
    
    def _calculate_association_cost(self, detection, features, track):
        """Calculate cost for associating detection with track"""
        # Spatial cost (position + size)
        spatial_cost = self._spatial_distance(detection, track)
        
        # Visual cost (feature similarity)
        visual_cost = self._visual_distance(features, track.features)
        
        # Temporal cost (based on time since last update)
        temporal_cost = track.time_since_update * 10  # Penalty for old tracks
        
        # Motion consistency cost
        motion_cost = self._motion_consistency_cost(detection, track)
        
        # Weighted combination
        total_cost = (
            self.feature_weights['spatial'] * spatial_cost +
            self.feature_weights['visual'] * visual_cost +
            self.feature_weights['temporal'] * temporal_cost +
            0.1 * motion_cost  # Small weight for motion
        )
        
        return total_cost
    
    def _spatial_distance(self, detection, track):
        """Calculate spatial distance between detection and track"""
        det_center = detection.center
        
        # Use predicted position if available
        if track.state == 'active' and len(track.position_history) > 1:
            predicted_pos = track.predict_next_position()
        else:
            predicted_pos = track.position_history[-1] if track.position_history else (0, 0)
        
        # Euclidean distance
        distance = np.sqrt(
            (det_center[0] - predicted_pos[0])**2 + 
            (det_center[1] - predicted_pos[1])**2
        )
        
        return distance
    
    def _visual_distance(self, features1, features2):
        """Calculate visual distance between feature vectors"""
        return self.feature_extractor.compare_features(features1, features2)
    
    def _motion_consistency_cost(self, detection, track):
        """Calculate motion consistency cost"""
        if len(track.position_history) < 2:
            return 0  # No motion history
        
        # Expected position based on motion
        expected_pos = track.predict_next_position()
        actual_pos = detection.center
        
        # Distance from expected position
        motion_error = np.sqrt(
            (actual_pos[0] - expected_pos[0])**2 + 
            (actual_pos[1] - expected_pos[1])**2
        )
        
        return motion_error
    
    def _handle_unmatched_detection(self, detection, features):
        """Handle detection that couldn't be matched to existing track"""
        # First, try re-identification with inactive tracks
        reidentified_id = self._attempt_reidentification(detection, features)
        
        if reidentified_id is not None:
            # Reactivate the track
            track = self.inactive_tracks.pop(reidentified_id)
            track.update(detection, features, self.frame_count)
            self.active_tracks[reidentified_id] = track
            self.successful_reidentifications += 1
            logging.debug(f"Re-identified player {reidentified_id}")
        else:
            # Create new track
            new_track = Track(self.next_id, detection, features, self.frame_count)
            self.active_tracks[self.next_id] = new_track
            self.total_tracks_created += 1
            logging.debug(f"Created new track {self.next_id}")
            self.next_id += 1
    
    def _attempt_reidentification(self, detection, features):
        """Attempt to re-identify detection with inactive tracks"""
        if not self.inactive_tracks:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for track_id, track in self.inactive_tracks.items():
            # Skip tracks that have been inactive too long
            if self.frame_count - track.last_frame > self.reid_timeout:
                continue
            
            # Calculate similarity
            similarity = self._calculate_reidentification_similarity(
                detection, features, track
            )
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = track_id
        
        return best_match_id
    
    def _calculate_reidentification_similarity(self, detection, features, track):
        """Calculate similarity for re-identification"""
        # Visual similarity (most important for re-ID)
        visual_distance = self._visual_distance(features, track.get_average_appearance())
        visual_sim = max(0, 1.0 - visual_distance)
        
        # Spatial plausibility
        spatial_distance = self._spatial_distance(detection, track)
        spatial_sim = max(0, 1.0 - spatial_distance / 300)  # Normalize by max expected distance
        
        # Time penalty (more recent = better)
        time_diff = self.frame_count - track.last_frame
        time_penalty = max(0, 1.0 - time_diff / self.reid_timeout)
        
        # Size consistency
        det_area = detection.get_area()
        track_bbox = track.bbox
        track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
        
        if track_area > 0:
            size_ratio = min(det_area, track_area) / max(det_area, track_area)
        else:
            size_ratio = 0
        
        # Combined similarity
        similarity = (
            0.5 * visual_sim +
            0.2 * spatial_sim +
            0.2 * time_penalty +
            0.1 * size_ratio
        )
        
        return similarity
    
    def _cleanup_tracks(self):
        """Remove old tracks and move lost tracks to inactive"""
        # Move lost tracks to inactive
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
                
                # Move to inactive for potential re-identification
                track.state = 'inactive'
                self.inactive_tracks[track_id] = track
                logging.debug(f"Moved track {track_id} to inactive")
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        # Remove very old inactive tracks
        inactive_to_remove = []
        for track_id, track in self.inactive_tracks.items():
            if self.frame_count - track.last_frame > self.reid_timeout:
                inactive_to_remove.append(track_id)
        
        for track_id in inactive_to_remove:
            del self.inactive_tracks[track_id]
            logging.debug(f"Removed old inactive track {track_id}")
    
    def get_statistics(self):
        """Get tracking statistics"""
        active_count = len(self.active_tracks)
        inactive_count = len(self.inactive_tracks)
        
        return {
            'active_tracks': active_count,
            'inactive_tracks': inactive_count,
            'total_tracks_created': self.total_tracks_created,
            'successful_reidentifications': self.successful_reidentifications,
            'frame_count': self.frame_count
        }
