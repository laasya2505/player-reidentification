import argparse
import cv2
import yaml
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import PlayerDetector
from tracker import PlayerTracker
from utils import VideoProcessor, setup_logging

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        sys.exit(1)

def validate_paths(config):
    """Validate input paths exist"""
    input_path = config['video']['input_path']  # FIXED: Use correct key
    model_path = config['model']['path']        # FIXED: Use correct key
    
    if not os.path.exists(input_path):
        logging.error(f"Input video not found: {input_path}")
        return False
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Player Re-identification System')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Config file path')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with visualization')
    parser.add_argument('--input', help='Input video path (overrides config)')
    parser.add_argument('--output', help='Output video path (overrides config)')
    parser.add_argument('--no-output', action='store_true', 
                       help='Skip saving output video (faster processing)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logging.info("Starting Player Re-identification System")
    
    # Load configuration
    config = load_config(args.config)
    
    
    if args.input:
        config['video']['input_path'] = args.input
    if args.output:
        config['video']['output_path'] = args.output
    if args.no_output:
        config['video']['save_output'] = False
    else:
        config['video']['save_output'] = True
    
    # Validate paths
    if not validate_paths(config):
        sys.exit(1)
    
    try:
        # Initialize components
        logging.info("Initializing detector...")
        detector = PlayerDetector(config['model'])
        
        logging.info("Initializing tracker...")
        tracker = PlayerTracker(config['tracking'], config['reidentification'])
        
        logging.info("Initializing video processor...")
        video_processor = VideoProcessor(config['video'])
        
        # Process video
        logging.info("Starting video processing...")
        video_processor.process_video(detector, tracker, debug=args.debug)
        
        logging.info("Processing complete!")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()