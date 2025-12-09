"""
Step 4.1: Image Semantic Segmentation

Uses DeepLabV3 pretrained on Cityscapes for semantic segmentation.
Cityscapes classes include: road, car, person, building, etc.

The key class we care about for SST-Calib is 'car' (vehicle).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import base class
from .base import ImageSegmentorBase


# Cityscapes class names (21 classes in PASCAL VOC pretrained model)
# For driving scenes, we'll use the COCO pretrained model which has 'car' class
COCO_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Classes we care about for calibration (vehicles are most reliable)
TARGET_CLASSES = {
    'car': 7,
    'bus': 6,
    'motorbike': 14,
    'bicycle': 2,
    'person': 15,
}


class ImageSegmentor(ImageSegmentorBase):
    """
    Semantic segmentation for RGB images using DeepLabV3.
    
    Extracts binary masks for target classes (primarily vehicles).
    """
    
    def __init__(self, device=None, target_classes=None):
        """
        Initialize the image segmentor.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            target_classes: list of class names to extract, default ['car']
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"ImageSegmentor using device: {self.device}")
        
        # Set target classes
        if target_classes is None:
            target_classes = ['car']
        self.target_classes = target_classes
        self.target_class_ids = [TARGET_CLASSES[c] for c in target_classes]
        
        print(f"Target classes: {self.target_classes} (IDs: {self.target_class_ids})")
        
        # Load pretrained DeepLabV3
        print("Loading DeepLabV3 model...")
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def segment(self, image):
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: numpy array (H, W, 3) RGB image, dtype uint8
            
        Returns:
            full_mask: (H, W) array with class IDs for each pixel
            binary_mask: (H, W) boolean array, True for target classes
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Preprocess
        input_tensor = self.transform(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Get class predictions
        full_mask = output.argmax(0).cpu().numpy()  # (H, W)
        
        # Create binary mask for target classes
        binary_mask = np.zeros_like(full_mask, dtype=bool)
        for class_id in self.target_class_ids:
            binary_mask |= (full_mask == class_id)
        
        return full_mask, binary_mask
    
    def get_mask_pixels(self, binary_mask):
        """
        Get pixel coordinates where mask is True.
        
        Args:
            binary_mask: (H, W) boolean array
            
        Returns:
            pixels: (N, 2) array of [u, v] coordinates
        """
        v_coords, u_coords = np.where(binary_mask)
        pixels = np.stack([u_coords, v_coords], axis=1)  # (N, 2) as [u, v]
        return pixels
    
    def visualize_segmentation(self, image, full_mask, binary_mask, save_path=None):
        """
        Visualize segmentation results.
        
        Args:
            image: original RGB image
            full_mask: full class predictions
            binary_mask: binary mask for target classes
            save_path: optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Full segmentation
        axes[1].imshow(full_mask, cmap='tab20')
        axes[1].set_title('Full Segmentation (all classes)')
        axes[1].axis('off')
        
        # Binary mask overlay
        axes[2].imshow(image)
        mask_overlay = np.zeros((*binary_mask.shape, 4))
        mask_overlay[binary_mask] = [1, 1, 0, 0.5]  # Yellow with transparency
        axes[2].imshow(mask_overlay)
        axes[2].set_title(f'Target Classes: {self.target_classes}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return fig


def test_image_segmentation():
    """Test image segmentation on KITTI data."""
    import sys
    sys.path.append('src')
    from data_loader import KITTIDataLoader
    
    # =====================================================
    # CONFIGURE THESE PATHS FOR YOUR SYSTEM
    # =====================================================
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    # Load data
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    
    # Create segmentor
    segmentor = ImageSegmentor(target_classes=['car', 'bus'])
    
    # Test on a few frames
    test_frames = [0, 50, 100]
    
    for frame_idx in test_frames:
        if frame_idx >= len(loader):
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_idx}")
        print('='*60)
        
        # Load image
        image = loader.load_image(frame_idx)
        
        # Segment
        full_mask, binary_mask = segmentor.segment(image)
        
        # Statistics
        n_target_pixels = np.sum(binary_mask)
        total_pixels = binary_mask.size
        percentage = 100 * n_target_pixels / total_pixels
        
        print(f"Target pixels: {n_target_pixels} ({percentage:.2f}% of image)")
        
        # Get pixel coordinates
        target_pixels = segmentor.get_mask_pixels(binary_mask)
        print(f"Target pixel coordinates shape: {target_pixels.shape}")
        
        # Visualize
        segmentor.visualize_segmentation(
            image, full_mask, binary_mask,
            save_path=f'outputs/image_seg_frame_{frame_idx}.png'
        )


if __name__ == "__main__":
    test_image_segmentation()