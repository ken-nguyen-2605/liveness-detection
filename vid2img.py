"""
This script extracts frames from a video file and saves them as images.
It uses OpenCV to read the video and save frames at specified intervals.
Usage:
    python vid2img.py --input <input_video_path> --output <output_directory> --interval <frame_interval>
    --input: Path to the input video file.
    --output: Directory where the extracted images will be saved.
    --interval: Interval (in seconds) at which frames will be extracted.
    --help: Show this help message.
"""

import cv2
import os
import argparse
from tqdm import tqdm
import glob

def extract_frames(input_video, output_dir, interval=1.0):
    """
    Extract frames from a video at specified intervals
    
    Args:
        input_video: Path to input video file
        output_dir: Directory to save extracted frames
        interval: Interval in seconds between extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    
    # Get base filename without extension
    base_name = os.path.basename(input_video).split('.')[0]
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc=f"Processing {base_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Save frame as image
                output_path = os.path.join(output_dir, f"{base_name}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {saved_count} frames from {input_video}")
    return saved_count

def process_directory(input_dir, output_base_dir, interval=1.0, extensions=('*.mp4', '*.MOV')):
    """
    Process all videos in a directory and its subdirectories
    """
    total_processed = 0
    total_frames = 0
    
    # Find all video files
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        # Create relative output directory structure
        rel_path = os.path.relpath(os.path.dirname(video_file), input_dir)
        output_dir = os.path.join(output_base_dir, rel_path)
        
        # Extract frames
        frames = extract_frames(video_file, output_dir, interval)
        total_frames += frames
        total_processed += 1
    
    print(f"Processed {total_processed} videos, extracted {total_frames} frames total")
    

'''
python3 vid2img.py --input ./videos --output ./images --interval 1.0 --recursive
'''
def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos at specified intervals")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory for extracted frames")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between extracted frames")
    parser.add_argument("--recursive", action="store_true", help="Process all videos in directory recursively")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input) and args.recursive:
        process_directory(args.input, args.output, args.interval)
    elif os.path.isfile(args.input):
        extract_frames(args.input, args.output, args.interval)
    else:
        print("Input must be a video file or a directory (with --recursive flag)")

if __name__ == "__main__":
    main()