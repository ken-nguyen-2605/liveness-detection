"""
This module provides data loading and preprocessing functionality for the liveness detection system.
It includes functions to load images from directories, preprocess them, and create data batches.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

class LivenessDataset(Dataset):
    """
    Custom Dataset for loading liveness detection images
    """
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): If True, load training data, else load validation data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Define the subdirectories for real and fake images
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get all image paths
        self.real_images = glob.glob(os.path.join(self.real_dir, '**', '*.jpg'), recursive=True)
        self.fake_images = glob.glob(os.path.join(self.fake_dir, '**', '*.jpg'), recursive=True)
        
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(is_train=True):
    """
    Get the image transforms for training or validation
    
    Args:
        is_train (bool): If True, return training transforms, else return validation transforms
    
    Returns:
        transforms: Image transforms
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders
    
    Args:
        data_dir (string): Root directory containing the data
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of workers for data loading
    
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create datasets
    train_dataset = LivenessDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    val_dataset = LivenessDataset(
        root_dir=os.path.join(data_dir, 'val'),
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def prepare_data_structure(base_dir):
    """
    Prepare the data directory structure for training
    
    Args:
        base_dir (string): Base directory containing all the frames
    """
    # Create main directories
    os.makedirs(os.path.join(base_dir, 'train', 'real'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'fake'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'real'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'fake'), exist_ok=True)
    
    print("Created data directory structure:")
    print(f"{base_dir}/")
    print("├── train/")
    print("│   ├── real/")
    print("│   └── fake/")
    print("└── val/")
    print("    ├── real/")
    print("    └── fake/")

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/data"
    prepare_data_structure(data_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir)
    
    # Example of iterating through the data
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break 