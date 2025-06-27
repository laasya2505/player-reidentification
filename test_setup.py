# test_setup.py
import cv2
import os
from ultralytics import YOLO

def test_complete_setup():
    print("Testing complete setup...")
    
    # Test model file
    model_path = "/Users/srinivas/Desktop/player_reidentification/models/yolov11_player_detection.pt"
    if os.path.exists(model_path):
        print("✓ Model file found")
        try:
            model = YOLO(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
    else:
        print("✗ Model file not found")
        return False
    
    # Test video file
    video_path = "/Users/srinivas/Desktop/player_reidentification/data/15sec_input_720p.mp4"
    if os.path.exists(video_path):
        print("✓ Video file found")
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"✓ Video: {width}x{height}, {fps} FPS, {frame_count} frames")
                print(f"✓ Duration: {frame_count/fps:.1f} seconds")
                cap.release()
            else:
                print("✗ Could not open video file")
                return False
        except Exception as e:
            print(f"✗ Video loading failed: {e}")
            return False
    else:
        print("✗ Video file not found")
        print(f"Expected location: {video_path}")
        return False
    
    print("✓ Setup test passed! Ready to run the tracker.")
    return True

if __name__ == "__main__":
    test_complete_setup()