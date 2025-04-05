# Liveness Detection System

This repository contains a liveness detection system designed to distinguish between real and fake images or videos. The system includes scripts for data loading, video-to-image conversion, and webcam capture.

## Folder Structure

-   **captured_videos/**: Contains videos and frames captured using the `webcam_capture.py` script.
    -   `capture_*.mp4`: Videos captured from the webcam.
    -   `frames_*`: Extracted frames from the captured videos.
-   **images/**: Contains categorized images for training and testing.
    -   `fake/`: Images classified as fake.
    -   `real/`: Images classified as real.
-   **videos/**: Contains video datasets for liveness detection.
    -   `anti-spoofing.csv`: Metadata for videos.
    -   `cut-out printouts/`, `live_selfie/`, `live_video/`, `printouts/`, `replay/`: Subfolders for different video categories.
-   **data_loader.py**: Script for loading and preprocessing image datasets.
-   **vid2img.py**: Script for extracting frames from videos.
-   **webcam_capture.py**: Script for capturing videos and frames using a webcam.

## How to Run the Scripts

### 1. Data Loader (`data_loader.py`)

This script loads images from the `images/` directory and preprocesses them for training or testing.

#### Example Usage:

```bash
python data_loader.py
```

### 2. Video to Image Conversion (`vid2img.py`)

This script extracts frames from videos at fixed intervals in the raw dataset folder `videos/` and saves them in the `images/` directory.

#### Example Usage:

```bash
python vid2img.py --input <input_video_path_or_directory> --output <output_directory>
```

-   `--input`: Path to the input video file or directory.
-   `--output`: Directory where extracted frames will be saved.

### 3. Webcam Capture (`webcam_capture.py`)

This script captures videos and frames using a webcam and saves them in the `captured_videos/` directory.

#### Example Usage:

```bash
python webcam_capture.py
```

-   The script will record a video for a default duration of 10 seconds. Press `q` to stop early.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Notes

-   Ensure that the `images/` directory contains `real/` and `fake/` subdirectories for the `data_loader.py` script to work correctly.
-   The `vid2img.py` script skips videos in the `live_selfie/` folder by default.
-   The `webcam_capture.py` script requires a functional webcam.
