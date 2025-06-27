
import cv2
import logging
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/player_tracking.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        # Convert HSV to BGR
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color = tuple(map(int, bgr[0, 0]))
        colors.append(color)
    
    return colors

def draw_text_with_background(img, text, position, font_scale=0.6, thickness=2, 
                            text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_height - baseline), 
                 (x + text_width, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

class VideoProcessor:
    """Handle video input/output and processing loop"""
    
    def __init__(self, video_config):
        self.input_path = video_config['input_path']  # FIXED: Correct key
        self.output_path = video_config.get('output_path', 'data/output/tracked_video.mp4')
        self.save_output = video_config.get('save_output', True)
        self.target_fps = video_config.get('fps', 30)
        
        # Visualization settings
        self.show_detections = video_config.get('show_detections', True)
        self.show_tracks = video_config.get('show_tracks', True)
        self.show_trajectories = video_config.get('show_trajectories', False)
        self.trajectory_length = video_config.get('trajectory_length', 30)
        
        # Create output directory if it doesn't exist
        if self.save_output:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Color palette for tracks
        self.track_colors = generate_colors(50)  # Support up to 50 different tracks
        
        # Track trajectories for visualization
        self.trajectories = {}
    
    def process_video(self, detector, tracker, debug=False):
        """Main video processing loop"""
        # Open input video
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        logging.info(f"Input: {self.input_path}")
        if self.save_output:
            logging.info(f"Output: {self.output_path}")
        
        # Setup video writer
        out = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        start_time = time.time()
        processing_times = []
        
        # Statistics
        total_detections = 0
        max_active_tracks = 0
        
        try:
            while True:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect players
                detections = detector.detect(frame)
                total_detections += len(detections)
                
                # Update tracker
                tracks = tracker.update(detections, frame)
                max_active_tracks = max(max_active_tracks, len(tracks))
                
                # Update trajectories
                self._update_trajectories(tracks)
                
                # Visualize results
                vis_frame = self._visualize_frame(frame, detections, tracks, debug)
                
                # Add frame information
                self._add_frame_info(vis_frame, frame_count, total_frames, 
                                   len(detections), len(tracks))
                
                # Write frame
                if self.save_output and out is not None:
                    out.write(vis_frame)
                
                # Calculate processing time
                frame_time = time.time() - frame_start_time
                processing_times.append(frame_time)
                
                # Progress logging
                if frame_count % 30 == 0 or frame_count == total_frames:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    avg_frame_time = np.mean(processing_times[-30:])
                    
                    logging.info(f"Frame {frame_count}/{total_frames} "
                               f"({100 * frame_count / total_frames:.1f}%) - "
                               f"FPS: {fps_current:.1f}, "
                               f"Frame time: {avg_frame_time*1000:.1f}ms")
                
                # Debug mode - show frame
                if debug:
                    cv2.imshow('Player Tracking', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):  # Pause
                        cv2.waitKey(0)
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            if debug:
                cv2.destroyAllWindows()
        
        # Final statistics
        self._log_final_statistics(frame_count, start_time, total_detections, 
                                 max_active_tracks, detector, tracker)
    
    def _update_trajectories(self, tracks):
        """Update trajectory history for visualization"""
        # Add current positions to trajectories
        current_track_ids = set()
        
        for track in tracks:
            track_id = track.id
            current_track_ids.add(track_id)
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            
            # Add current position
            center = track.position_history[-1] if track.position_history else (0, 0)
            self.trajectories[track_id].append(center)
            
            # Limit trajectory length
            if len(self.trajectories[track_id]) > self.trajectory_length:
                self.trajectories[track_id].pop(0)
        
        # Remove trajectories for inactive tracks
        inactive_tracks = set(self.trajectories.keys()) - current_track_ids
        for track_id in inactive_tracks:
            if track_id in self.trajectories:
                del self.trajectories[track_id]
    
    def _visualize_frame(self, frame, detections, tracks, debug=False):
        """Create visualization of frame with detections and tracks"""
        vis_frame = frame.copy()
        
        # Draw detections (if enabled and debug mode)
        if self.show_detections and debug:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                label = f"Det: {detection.confidence:.2f}"
                draw_text_with_background(vis_frame, label, (x1, y1-5), 
                                        font_scale=0.4, bg_color=(0, 255, 0))
        
        # Draw tracks
        if self.show_tracks:
            for track in tracks:
                track_id = track.id
                color = self.track_colors[track_id % len(self.track_colors)]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID and info
                label = f"ID: {track_id}"
                if debug:
                    label += f" ({track.hits}h, {track.confidence:.2f})"
                
                draw_text_with_background(vis_frame, label, (x1, y1-10), 
                                        text_color=(255, 255, 255), bg_color=color)
                
                # Draw center point
                center = tuple(map(int, track.position_history[-1] if track.position_history else (0, 0)))
                cv2.circle(vis_frame, center, 3, color, -1)
                
                # Draw velocity vector (debug mode)
                if debug and track.velocity != (0, 0):
                    center_x, center_y = center
                    vel_x, vel_y = track.velocity
                    end_x = int(center_x + vel_x * 5)  # Scale velocity for visibility
                    end_y = int(center_y + vel_y * 5)
                    cv2.arrowedLine(vis_frame, (center_x, center_y), 
                                  (end_x, end_y), color, 2)
        
        # Draw trajectories
        if self.show_trajectories:
            for track_id, trajectory in self.trajectories.items():
                if len(trajectory) > 1:
                    color = self.track_colors[track_id % len(self.track_colors)]
                    
                    # Draw trajectory as connected line segments
                    points = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(vis_frame, [points], False, color, 2)
        
        return vis_frame
    
    def _add_frame_info(self, frame, frame_num, total_frames, num_detections, num_tracks):
        """Add frame information overlay"""
        h, w = frame.shape[:2]
        
        # Frame counter
        frame_text = f"Frame: {frame_num}/{total_frames}"
        draw_text_with_background(frame, frame_text, (10, 30), 
                                font_scale=0.6, bg_color=(0, 0, 0))
        
        # Detection and track counts
        stats_text = f"Detections: {num_detections} | Active Tracks: {num_tracks}"
        draw_text_with_background(frame, stats_text, (10, 60), 
                                font_scale=0.6, bg_color=(0, 0, 0))
    
    def _log_final_statistics(self, frame_count, start_time, total_detections, 
                            max_active_tracks, detector, tracker):
        """Log final processing statistics"""
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        
        logging.info("=" * 50)
        logging.info("PROCESSING COMPLETE")
        logging.info("=" * 50)
        logging.info(f"Total frames processed: {frame_count}")
        logging.info(f"Total processing time: {elapsed:.2f}s")
        logging.info(f"Average FPS: {avg_fps:.2f}")
        logging.info(f"Total detections: {total_detections}")
        logging.info(f"Max simultaneous tracks: {max_active_tracks}")
        
        # Detector statistics
        det_stats = detector.get_statistics()
        logging.info(f"Detection validity rate: {det_stats['validity_rate']:.2%}")
        
        # Tracker statistics
        track_stats = tracker.get_statistics()
        logging.info(f"Total tracks created: {track_stats['total_tracks_created']}")
        logging.info(f"Successful re-identifications: {track_stats['successful_reidentifications']}")
        
        if track_stats['total_tracks_created'] > 0:
            reid_rate = track_stats['successful_reidentifications'] / track_stats['total_tracks_created']
            logging.info(f"Re-identification rate: {reid_rate:.2%}")
        
        if self.save_output:
            logging.info(f"Output saved to: {self.output_path}")
        
        logging.info("=" * 50)