"""
Abstract base classes for semantic segmentation.

Defines common interfaces for image and LiDAR segmentors.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSegmentor(ABC):
    """Abstract base class for semantic segmentation.
    
    All segmentors (image, LiDAR, etc.) should inherit from this class
    and implement the segment() method.
    """
    
    @abstractmethod
    def segment(self, data):
        """Segment input data into semantic classes.
        
        Args:
            data: Input data (image, point cloud, etc.)
            
        Returns:
            Segmentation result (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def get_car_points(self, data):
        """Extract car-specific data from input.
        
        Args:
            data: Input data (image, point cloud, etc.)
            
        Returns:
            Car-specific data (pixels, points, etc.)
        """
        pass


class ImageSegmentorBase(BaseSegmentor):
    """Base class for image semantic segmentation.
    
    Provides common interface for image-based segmentation models.
    """
    
    @abstractmethod
    def segment(self, image):
        """Segment image into semantic classes.
        
        Args:
            image: (H,W,3) RGB image
            
        Returns:
            segmentation_map: (H,W) class labels
            binary_mask: (H,W) boolean mask for target classes
        """
        pass
    
    def get_car_points(self, image):
        """Get car pixels from image.
        
        Args:
            image: (H,W,3) RGB image
            
        Returns:
            car_pixels: (N,2) array of [x,y] pixel coordinates
        """
        _, binary_mask = self.segment(image)
        car_pixels = np.argwhere(binary_mask)  # Returns [y,x] pairs
        return car_pixels[:, [1, 0]]  # Convert to [x,y]


class LiDARSegmentorBase(BaseSegmentor):
    """Base class for LiDAR point cloud segmentation.
    
    Provides common interface for point cloud segmentation models.
    """
    
    @abstractmethod
    def segment(self, point_cloud):
        """Segment point cloud into semantic classes.
        
        Args:
            point_cloud: (N,4) array [x,y,z,intensity]
            
        Returns:
            labels: (N,) array with class labels
            car_mask: (N,) boolean mask for car points
        """
        pass
    
    @abstractmethod
    def get_car_points(self, point_cloud):
        """Extract car points from point cloud.
        
        Args:
            point_cloud: (N,4) array [x,y,z,intensity]
            
        Returns:
            car_indices: Indices of car points
            car_points: (M,4) array of car points [x,y,z,intensity]
        """
        pass
