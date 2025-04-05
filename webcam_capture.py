import cv2
import os
import time
import numpy as np
from datetime import datetime

class WebcamCapture:
    def __init__(self, output_dir="captured_videos", duration=5):
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
        
    def capture_and_process(self, show_preview=True, mirror_preview=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"capture_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                            (self.frame_width, self.frame_height))
        
        frames = []
        start_time = time.time()
        
        print("Starting capture...")
        print("Press 'q' to stop early")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Add original frame to video writer and store for individual frames
            out.write(frame)
            frames.append(frame)
            
            # Show preview if requested
            if show_preview:
                # Create mirrored version only for display if requested
                if mirror_preview:
                    display_frame = cv2.flip(frame, 1)  # 1 = horizontal flip
                else:
                    display_frame = frame
                    
                cv2.imshow('Recording', display_frame)
                
            # Check if duration has elapsed or 'q' is pressed
            if time.time() - start_time >= self.duration:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved to: {output_path}")
        
        # Save individual frames
        frame_paths = self.save_frames(frames)
        
        return output_path, frame_paths
    
    def save_frames(self, frames, output_dir=None):
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
    # Initialize webcam capture
    webcam = WebcamCapture(duration=10)
    
    video_path, frame_paths = webcam.capture_and_process()
    print(f"Captured video: {video_path}")
    print(f"Captured {len(frame_paths)} frames")

if __name__ == "__main__":
    main() 