"""
    python vid2img.py --input <input_video_path> --output <output_directory>
    --input: Path to the input video file or directory.
    --output: Base directory where extracted images will be saved.
"""

import cv2
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np

class VideoFrameExtractor:
    def __init__(self, output_dir):
        
        self.output_dir = output_dir
        self.interval = 0.2  # Fixed interval in seconds
        
        # Create output directories with real and fake subfolders
        self.real_dir = os.path.join(output_dir, "real")
        self.fake_dir = os.path.join(output_dir, "fake")
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        
        # Video formats to process
        self.extensions = ('*.mp4', '*.MOV', '*.avi', '*.mkv')
        
        # Stats
        self.total_processed = 0
        self.total_frames = 0
    
    def extract_frames(self, input_video, is_real=True):
        """
        Extract frames from a video at specified intervals
        
        Args:
            input_video: Path to input video file
            is_real: If True, save to real folder, otherwise to fake folder
        
        Returns:
            Number of frames extracted
        """
        # Determine output directory based on real/fake
        parent_dir = self.real_dir if is_real else self.fake_dir
        
        # Get base filename without extension
        base_name = os.path.basename(input_video).split('.')[0]
        
        # Create subfolder for this specific video
        video_dir = os.path.join(parent_dir, base_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Error opening video file: {input_video}")
            return 0
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = int(fps * self.interval)
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=total_frames, desc=f"Processing {base_name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Save frame as image in the video's subfolder
                    output_path = os.path.join(video_dir, f"{base_name}_frame_{saved_count:04d}.jpg")
                    cv2.imwrite(output_path, frame)
                    saved_count += 1
                    
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        print(f"Extracted {saved_count} frames from {input_video} to {video_dir}")
        return saved_count
    
    def process_directory(self, input_dir):
        """
        Process all videos in a directory and its subdirectories
        
        Args:
            input_dir: Directory containing video files
        """
        # Find all video files
        video_files = []
        for ext in self.extensions:
            video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        
        if not video_files:
            print(f"No video files found in {input_dir} with extensions: {self.extensions}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            # Skip live_selfie folder
            if "live_selfie" in video_file:
                print(f"Skipping {video_file} (in live_selfie folder)")
                continue
                
            # Determine if the video is "real" or "fake" based on folder name
            is_real = "live_video" in video_file
                
            # Extract frames
            frames = self.extract_frames(video_file, is_real)
            if frames:
                self.total_frames += frames
                self.total_processed += 1
        
        print(f"Processed {self.total_processed} videos, extracted {self.total_frames} frames total")
    
    def process_single_video(self, video_path):
        """
        Process a single video file
        
        Args:
            video_path: Path to the video file
        """
        # Determine if the video is "real" or "fake" based on folder name
        is_real = "live_video" in video_path
        
        frames = self.extract_frames(video_path, is_real)
        if frames:
            self.total_frames += frames
            self.total_processed += 1


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos at specified intervals")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory for extracted frames")
    
    args = parser.parse_args()
    
    # Create extractor instance
    extractor = VideoFrameExtractor(args.output)
    
    if os.path.isdir(args.input):
        extractor.process_directory(args.input)
    elif os.path.isfile(args.input):
        extractor.process_single_video(args.input)
    else:
        print("Input must be a video file or a directory")

if __name__ == "__main__":
    main()