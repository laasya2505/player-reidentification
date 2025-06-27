# player-reidentification
# Player Re-identification System

## Overview

This project implements a comprehensive player re-identification system for sports footage using YOLOv11 for detection and advanced tracking algorithms. The system successfully maintains consistent player IDs even when players temporarily leave and re-enter the frame, achieving a **65.79% re-identification rate** with **25 successful re-identifications** in a 15-second soccer video.

## Features

- ✅ **YOLOv11-based Player Detection**: State-of-the-art object detection for players, goalkeepers, and referees
- ✅ **Multi-Object Tracking**: Robust tracking with motion prediction and feature-based association
- ✅ **Player Re-identification**: Advanced re-identification when players reappear after leaving the frame
- ✅ **Visual Output**: Real-time visualization with colored bounding boxes and track IDs
- ✅ **Performance Optimization**: GPU acceleration support for faster processing
- ✅ **Configurable Parameters**: Extensive configuration options for fine-tuning

## System Requirements

### Hardware
- **Minimum**: Intel i5 / AMD Ryzen 5, 8GB RAM
- **Recommended**: Intel i7 / AMD Ryzen 7, 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage**: 2GB free space

### Software
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **CUDA**: 11.0+ (optional, for GPU acceleration)

## Installation

### 1. Clone/Download the Project
```bash
# Download and extract the project files
# Ensure you have the following structure:
player_reidentification/
├── src/
├── config/
├── models/
├── data/
└── requirements.txt
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Files

#### Model File
- Download the YOLOv11 model from the provided Google Drive link
- Place the `.pt` file in `models/yolov11_player_detection.pt`

#### Video File
- Download the 15-second test video (`15sec_input_720p.mp4`)
- Place it in `data/input/15sec_input_720p.mp4`

### 5. Verify Installation
```bash
python test_setup.py
```

Expected output:
```
✓ Model file found
✓ Model loaded successfully
✓ Video file found
✓ Video: 1280x720, 30.0 FPS, 375 frames
✓ Setup test passed! Ready to run the tracker.
```

## Usage

### Basic Usage
```bash
# Run with default settings
python src/main.py
```

### Debug Mode (Recommended for first run)
```bash
# Run with live visualization
python src/main.py --debug
```

### Advanced Usage
```bash
# Custom input/output
python src/main.py --input "path/to/video.mp4" --output "results/output.mp4"

# Custom configuration
python src/main.py --config "config/custom_config.yaml"

# Fast processing (no output video)
python src/main.py --no-output
```

## Configuration

The system uses `config/config.yaml` for configuration. Key parameters:

### Model Settings
```yaml
model:
  confidence_threshold: 0.7    # Detection confidence threshold
  iou_threshold: 0.4          # Non-maximum suppression threshold
  min_detection_area: 500     # Minimum bounding box area
```

### Tracking Settings
```yaml
tracking:
  max_disappeared_frames: 45  # Frames before track goes inactive
  max_distance_threshold: 80  # Maximum association distance
```

### Re-identification Settings
```yaml
reidentification:
  similarity_threshold: 0.4     # Re-identification threshold
  inactive_track_timeout: 150   # Frames before inactive track deletion
```

### Performance Settings
```yaml
performance:
  device: "auto"        # "auto", "cuda", "cpu", or "mps"
  batch_size: 4         # Batch size for inference
```

## Output

### Generated Files
- **`data/output/tracked_output.mp4`**: Main output video with tracking visualization
- **`logs/player_tracking.log`**: Detailed processing logs
- **Console output**: Real-time processing statistics

### Visualization Elements
- **Colored bounding boxes**: Each player has a unique colored box
- **Track IDs**: Persistent ID numbers displayed on each player
- **Frame statistics**: Detection and track counts overlay
- **Motion vectors**: Optional velocity visualization in debug mode

## Performance Benchmarks

### Achieved Results
- **Re-identification Rate**: 65.79%
- **Successful Re-identifications**: 25 in 15-second video
- **Total Tracks Created**: 38
- **Max Simultaneous Tracks**: 27
- **Detection Validity Rate**: 98.20%
- **Processing Speed**: 0.48 FPS (CPU), up to 30 FPS (GPU)

### Speed Optimization
For faster processing:
1. **Use GPU**: Set `device: "cuda"` in config
2. **Increase confidence threshold**: Higher values = fewer detections
3. **Disable trajectories**: Set `show_trajectories: false`
4. **Reduce batch size**: Lower memory usage

## Troubleshooting

### Common Issues

#### Issue 1: Model not found
```bash
# Error: Model file not found
# Solution: Download model to models/yolov11_player_detection.pt
```

#### Issue 2: Video not found
```bash
# Error: Could not open video
# Solution: Place video in data/input/15sec_input_720p.mp4
```

#### Issue 3: CUDA out of memory
```yaml
# Solution: In config.yaml
performance:
  device: "cpu"
  batch_size: 1
```

#### Issue 4: Slow processing
```yaml
# Solution: Optimize settings
model:
  confidence_threshold: 0.8  # Higher threshold
video:
  show_trajectories: false   # Disable trajectories
```

### Performance Tips
- **GPU recommended**: 25-50x faster than CPU
- **Adjust confidence threshold**: Higher = faster but fewer detections
- **Monitor memory usage**: Reduce batch size if needed
- **Use SSD storage**: Faster video I/O

## Project Structure

```
player_reidentification/
├── src/
│   ├── main.py              # Main entry point
│   ├── detector.py          # YOLOv11 detection
│   ├── tracker.py           # Multi-object tracking
│   ├── features.py          # Feature extraction
│   └── utils.py             # Video processing utilities
├── config/
│   └── config.yaml          # Configuration file
├── models/
│   └── yolov11_player_detection.pt  # YOLOv11 model
├── data/
│   ├── input/
│   │   └── 15sec_input_720p.mp4     # Input video
│   └── output/
│       └── tracked_output.mp4       # Output video
├── logs/
│   └── player_tracking.log          # Processing logs
├── requirements.txt         # Python dependencies
├── test_setup.py           # Setup verification
└── README.md               # This file
```

## Technical Details

### Architecture
1. **Detection**: YOLOv11 model detects players, goalkeepers, referees
2. **Feature Extraction**: Color histograms, spatial features, dominant colors
3. **Association**: Hungarian algorithm for optimal detection-to-track matching
4. **Tracking**: Kalman filter-inspired motion prediction
5. **Re-identification**: Feature-based similarity matching for inactive tracks

### Key Algorithms
- **Detection**: YOLOv11 with confidence and IoU filtering
- **Association**: Hungarian algorithm with weighted cost function
- **Re-identification**: Cosine similarity on appearance features
- **Motion Prediction**: Velocity-based position prediction

## Dependencies

### Core Dependencies
```
ultralytics>=8.0.0    # YOLOv11 implementation
opencv-python>=4.8.0  # Computer vision operations
torch>=2.0.0          # Deep learning framework
numpy>=1.24.0         # Numerical computations
scipy>=1.10.0         # Scientific computing
PyYAML>=6.0           # Configuration file parsing
```

### Optional Dependencies
```
matplotlib>=3.7.0     # Plotting and visualization
tqdm>=4.65.0         # Progress bars
tensorrt             # GPU optimization (NVIDIA only)
```

## License

This project is developed for educational purposes as part of an AI internship assignment.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your setup with `python test_setup.py`
3. Review the logs in `logs/player_tracking.log`
4. Ensure all dependencies are correctly installed

## Acknowledgments

- **YOLOv11**: Ultralytics team for the object detection model
- **Assignment Provider**: Liat.ai for the project requirements
- **Dataset**: Soccer footage for testing and validation
