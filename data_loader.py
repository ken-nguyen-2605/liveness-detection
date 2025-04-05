"""
This module provides data loading functionality for the liveness detection system.
It includes functions to load images from directories and preprocess them.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

class LivenessDataset(Dataset):
    """
    Custom Dataset for loading liveness detection images based on the provided folder structure
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory with all the images (should contain 'fake' and 'real' subdirectories)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Define the subdirectories for real and fake images
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get all image paths
        self.real_images = glob.glob(os.path.join(self.real_dir, '**', '*.jpg'), recursive=True)
        if not self.real_images:  # Try other extensions if no jpg files found
            self.real_images = glob.glob(os.path.join(self.real_dir, '**', '*.png'), recursive=True)
        
        self.fake_images = glob.glob(os.path.join(self.fake_dir, '**', '*.jpg'), recursive=True)
        if not self.fake_images:  # Try other extensions if no jpg files found
            self.fake_images = glob.glob(os.path.join(self.fake_dir, '**', '*.png'), recursive=True)
        
        # Combine all images and create labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = [1] * len(self.real_images) + [0] * len(self.fake_images)
        
        print(f"Loaded {len(self.real_images)} real images and {len(self.fake_images)} fake images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_default_transforms():
    """
    Get the standard image transforms for preprocessing
    
    Returns:
        transforms: Image transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])

def create_data_loader(data_dir, batch_size=32, num_workers=4, transform=None):
    """
    Create a data loader for the liveness detection dataset
    
    Args:
        data_dir (string): Root directory containing the data (with 'fake' and 'real' subdirectories)
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of workers for data loading
        transform (callable, optional): Custom transform to apply to the images
    
    Returns:
        data_loader: DataLoader object for the dataset
    """
    # Create dataset
    if transform is None:
        transform = get_default_transforms()
    
    dataset = LivenessDataset(
        root_dir=data_dir,
        transform=transform
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader

if __name__ == "__main__":
    # Example usage
    data_dir = "images"  # Directory containing 'fake' and 'real' subdirectories
    
    # Create data loader
    data_loader = create_data_loader(data_dir)
    
    # Example of iterating through the data
    for images, labels in data_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break