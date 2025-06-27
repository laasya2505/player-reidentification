import os
import sys
import yaml
import shutil

def fix_file_locations():
    """Fix file locations and update config"""
    
    print("Checking file locations...")
    
    # Check where your video file actually is
    possible_video_paths = [
        "data/input/15sec_input_720p.mp4",
        "data/15sec_input_720p.mp4",
        "15sec_input_720p.mp4",
        "/Users/srinivas/Desktop/player_reidentification/data/15sec_input_720p.mp4"
    ]
    
    video_path = None
    for path in possible_video_paths:
        if os.path.exists(path):
            video_path = path
            print(f"✓ Found video file at: {path}")
            break
    
    if not video_path:
        print("✗ Video file not found in any expected location")
        print("Please check if you have downloaded 15sec_input_720p.mp4")
        return False
    
    # Check model file
    model_path = None
    possible_model_paths = [
        "models/yolov11_player_detection.pt",
        "yolov11_player_detection.pt"
    ]
    
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✓ Found model file at: {path}")
            break
    
    if not model_path:
        print("✗ Model file not found")
        return False
    
    # Move files to correct locations if needed
    correct_video_path = "/Users/srinivas/Desktop/player_reidentification/data/15sec_input_720p.mp4"
    correct_model_path = "/Users/srinivas/Desktop/player_reidentification/models/yolov11_player_detection.pt"
    
    # Create directories
    os.makedirs("data/", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Move video file if needed
    if video_path != correct_video_path:
        print(f"Moving video file to correct location...")
        shutil.move(video_path, correct_video_path)
        print(f"✓ Moved video to: {correct_video_path}")
    
    # Move model file if needed
    if model_path != correct_model_path:
        print(f"Moving model file to correct location...")
        shutil.move(model_path, correct_model_path)
        print(f"✓ Moved model to: {correct_model_path}")
    
    # Update config file
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update paths
        config['video']['input_path'] = correct_video_path
        config['model']['path'] = correct_model_path
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"✓ Updated config file")
    
    return True

def run_system():
    """Run the player re-identification system"""
    print("\nRunning player re-identification system...")
    
    # Change to src directory and run
    os.system("python3 src/main.py --debug")

if __name__ == "__main__":
    print("Player Re-identification Quick Fix")
    print("=" * 40)
    
    if fix_file_locations():
        print("\n✅ Files are in correct locations")
        print("Now running the system...")
        run_system()
    else:
        print("\n❌ Please fix file locations manually:")
        print("1. Put video file in: data/input/15sec_input_720p.mp4")
        print("2. Put model file in: models/yolov11_player_detection.pt")
        print("3. Run: python3 src/main.py")