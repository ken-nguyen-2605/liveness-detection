"""
This module provides functionality to capture video from webcam for liveness detection.
It includes functions to capture video frames, save them, and prepare them for processing.
"""

import cv2
import os
import time
import numpy as np
from datetime import datetime

class WebcamCapture:
    def __init__(self, output_dir="captured_videos", duration=5):
        """
        Initialize webcam capture
        
        Args:
            output_dir (str): Directory to save captured videos
            duration (int): Duration of video capture in seconds
        """
        self.output_dir = output_dir
        self.duration = duration
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
            
        # Get webcam properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
    def capture_video(self, show_preview=True):
        """
        Capture video from webcam for specified duration
        
        Args:
            show_preview (bool): Whether to show live preview while capturing
            
        Returns:
            str: Path to saved video file
        """
        # Prepare video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"capture_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                            (self.frame_width, self.frame_height))
        
        start_time = time.time()
        frames = []
        
        print("Starting video capture...")
        print("Press 'q' to stop early")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Add frame to video writer
            out.write(frame)
            frames.append(frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('Webcam Capture', frame)
                
            # Check if duration has elapsed or 'q' is pressed
            if time.time() - start_time >= self.duration:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved to: {output_path}")
        return output_path
    
    def capture_frames(self, interval=0.1, show_preview=True):
        """
        Capture individual frames from webcam
        
        Args:
            interval (float): Time interval between frames in seconds
            show_preview (bool): Whether to show live preview while capturing
            
        Returns:
            list: List of captured frames
        """
        frames = []
        start_time = time.time()
        
        print("Starting frame capture...")
        print("Press 'q' to stop")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            current_time = time.time() - start_time
            if current_time >= self.duration:
                break
                
            # Capture frame at specified intervals
            if len(frames) == 0 or current_time - (len(frames) * interval) >= interval:
                frames.append(frame)
                
            # Show preview if requested
            if show_preview:
                cv2.imshow('Frame Capture', frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return frames
    
    def save_frames(self, frames, output_dir=None):
        """
        Save captured frames as images
        
        Args:
            frames (list): List of frames to save
            output_dir (str): Directory to save frames (optional)
            
        Returns:
            list: List of paths to saved frames
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"frames_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)
            
        print(f"Saved {len(saved_paths)} frames to: {output_dir}")
        return saved_paths
    
    def __del__(self):
        """Release webcam when object is destroyed"""
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    """Example usage"""
    # Initialize webcam capture
    webcam = WebcamCapture(duration=5)
    
    # Capture video
    video_path = webcam.capture_video(show_preview=True)
    print(f"Captured video: {video_path}")
    
    # Capture frames
    frames = webcam.capture_frames(interval=0.1, show_preview=True)
    frame_paths = webcam.save_frames(frames)
    print(f"Captured {len(frame_paths)} frames")

if __name__ == "__main__":
    main() 