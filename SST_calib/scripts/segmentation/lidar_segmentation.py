"""
LiDAR Segmentation using SalsaNext
Wrapper for semantic segmentation of LiDAR point clouds
"""
import numpy as np
import torch
import torch.nn.functional as F
import os
import yaml

# Import local SalsaNext model
from models.salsanext_model import SalsaNext

# Import base class
from .base import LiDARSegmentorBase

# FOV Filtering Constants (camera field-of-view constraints)
FOV_MIN_DISTANCE = 8.0   # meters - minimum forward distance
FOV_MAX_DISTANCE = 40.0  # meters - maximum detection range
FOV_LATERAL_RANGE = 12.0 # meters - lateral (Y-axis) range ±

# DBSCAN Clustering Parameters (remove isolated ghost points)
DBSCAN_EPS = 0.5           # meters - neighborhood radius
DBSCAN_MIN_SAMPLES = 5     # minimum points to form a cluster
DBSCAN_MIN_CLUSTER_SIZE = 30  # minimum cluster size to keep


class LidarSegmenter(LiDARSegmentorBase):
    """
    Wrapper for SalsaNext (Semantic Segmentation of LiDAR Point Clouds).
    Uses pretrained weights and configuration files.
    """
    
    def __init__(self, 
                 weights_path=None,
                 arch_config_path=None,
                 data_config_path=None):
        """
        Initialize LiDAR segmenter with pretrained SalsaNext model.
        
        Args:
            weights_path: Path to pretrained model weights
            arch_config_path: Path to architecture configuration YAML
            data_config_path: Path to data configuration YAML
        """
        print("Initializing SalsaNext Segmenter...")
        
        # Set default paths relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Dataset is two levels up (scripts/segmentation -> scripts -> SST_calib)
        parent_dataset_dir = os.path.join(script_dir, '..', '..', 'dataset')
        
        if weights_path is None:
            weights_path = os.path.join(parent_dataset_dir, "models/pretrained/SalsaNext")
        if arch_config_path is None:
            arch_config_path = os.path.join(parent_dataset_dir, "models/pretrained/arch_cfg.yaml")
        if data_config_path is None:
            data_config_path = os.path.join(parent_dataset_dir, "models/pretrained/data_cfg.yaml")
        
        # 1. Device Setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"  > Device: {self.device}")

        # 2. Load Architecture Configuration
        if not os.path.exists(arch_config_path):
            raise FileNotFoundError(f"Architecture config not found: {arch_config_path}")
        
        print(f"  > Loading architecture config from: {arch_config_path}")
        with open(arch_config_path, 'r') as f:
            self.arch_cfg = yaml.safe_load(f)
        
        # 3. Load Data Configuration
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(f"Data config not found: {data_config_path}")
        
        print(f"  > Loading data config from: {data_config_path}")
        with open(data_config_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)

        # 4. Extract Sensor Parameters from arch_cfg
        sensor_cfg = self.arch_cfg['dataset']['sensor']
        self.fov_up = sensor_cfg['fov_up']
        self.fov_down = sensor_cfg['fov_down']
        self.proj_H = sensor_cfg['img_prop']['height']
        self.proj_W = sensor_cfg['img_prop']['width']
        
        # Convert to radians
        self.fov_up_rad = self.fov_up * np.pi / 180.0
        self.fov_down_rad = self.fov_down * np.pi / 180.0
        self.fov_diff = abs(self.fov_down_rad) + abs(self.fov_up_rad)
        
        # Normalization parameters (from arch_cfg)
        # Order: [range, x, y, z, signal]
        self.img_means = np.array(sensor_cfg['img_means'], dtype=np.float32).reshape(1, 5, 1, 1)
        self.img_stds = np.array(sensor_cfg['img_stds'], dtype=np.float32).reshape(1, 5, 1, 1)
        
        # 5. Get learning map (maps original labels to training labels)
        self.learning_map = self.data_cfg['learning_map']
        
        # Car class: original label 10 -> learning label 1
        self.target_class_id = self.learning_map[10]  # Should be 1
        print(f"  > Target class ID (car): {self.target_class_id}")
        
        # Number of classes in learning map
        self.nclasses = max(self.learning_map.values()) + 1  # 20 classes (0-19)
        print(f"  > Number of classes: {self.nclasses}")

        # 6. Load Model
        try:
            # Get dropout from arch config (default 0.01)
            dropout = self.arch_cfg.get('train', {}).get('epsilon_w', 0.01)
            
            # Create model
            self.model = SalsaNext(self.nclasses, params={'dropout': dropout})
            
            # Load weights
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            print(f"  > Loading weights from: {weights_path}")
            # Load to CPU first to avoid MPS float64 issues
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Clean keys (remove 'module.' prefix if present from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            
            # Load state dict
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("  > SalsaNext model loaded successfully!")
            
        except Exception as e:
            print(f"[MODEL LOAD ERROR] {e}")
            raise e

    def project_to_spherical(self, points):
        """
        Project 3D points to spherical image (range view).
        
        Args:
            points: Nx4 array [x, y, z, intensity]
        
        Returns:
            input_tensor: (1, 5, H, W) tensor for model input
            proj_coords: (N, 2) array of [u, v] pixel coordinates
            valid_mask: (N,) boolean array indicating valid points
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros_like(x)
        
        # Calculate depth (range)
        depth = np.linalg.norm(points[:, :3], axis=1)
        
        # Calculate spherical coordinates
        yaw = -np.arctan2(y, x)  # Horizontal angle
        pitch = np.arcsin(z / np.clip(depth, 1e-5, None))  # Vertical angle
        
        # Project to image coordinates
        proj_x = 0.5 * (yaw / np.pi + 1.0) * self.proj_W
        proj_y = (1.0 - (pitch + abs(self.fov_down_rad)) / self.fov_diff) * self.proj_H
        
        # Round to integers
        proj_x = np.floor(proj_x).astype(np.int64)
        proj_y = np.floor(proj_y).astype(np.int64)
        
        # Clip to valid range
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)
        
        # Valid points (positive depth)
        valid = (depth > 0)
        
        # Create input tensor: (1, 5, H, W)
        # Channels: [range, x, y, z, intensity] (Matches official SalsaNext)
        input_tensor = np.zeros((1, 5, self.proj_H, self.proj_W), dtype=np.float32)
        
        # Get valid indices
        indices = np.where(valid)[0]
        
        # Fill tensor (only valid points)
        # Channel 0: Range (depth)
        input_tensor[0, 0, proj_y[indices], proj_x[indices]] = depth[indices]
        # Channel 1: X
        input_tensor[0, 1, proj_y[indices], proj_x[indices]] = x[indices]
        # Channel 2: Y
        input_tensor[0, 2, proj_y[indices], proj_x[indices]] = y[indices]
        # Channel 3: Z
        input_tensor[0, 3, proj_y[indices], proj_x[indices]] = z[indices]
        # Channel 4: Intensity
        input_tensor[0, 4, proj_y[indices], proj_x[indices]] = intensity[indices]
        
        # Normalize using config parameters
        # Config order is [range, x, y, z, signal], which now matches our tensor
        input_tensor = (input_tensor - self.img_means) / self.img_stds
        
        return torch.from_numpy(input_tensor).float(), np.stack([proj_x, proj_y], axis=1), valid

    def segment(self, point_cloud):
        """
        Segment point cloud into semantic classes.
        
        Args:
            point_cloud: (N,4) array [x,y,z,intensity]
            
        Returns:
            labels: (N,) array with class labels
            car_mask: (N,) boolean mask for car points
        """
        # Project to spherical image
        input_tensor, proj_coords, valid_mask = self.project_to_spherical(point_cloud)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Map predictions back to points
        us = proj_coords[:, 0]
        vs = proj_coords[:, 1]
        point_labels = pred_mask[vs, us]
        
        # Create full labels array
        labels = np.zeros(len(point_cloud), dtype=np.int32)
        labels[valid_mask] = point_labels
        
        # Create car mask
        car_mask = (labels == self.target_class_id)
        
        return labels, car_mask

    def get_car_points(self, points):
        """
        Segment point cloud and extract car points.
        
        Args:
            points: Nx4 array [x, y, z, intensity]
        
        Returns:
            car_indices: Indices of points classified as cars
            car_points: Nx4 array of car points
        """
        # Project to spherical image
        input_tensor, proj_coords, valid_mask = self.project_to_spherical(points)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            # logits shape: (1, nclasses, H, W)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Map predictions back to points
        us = proj_coords[:, 0]
        vs = proj_coords[:, 1]
        
        # Get predicted labels for each point
        point_labels = pred_mask[vs, us]
        
        # Find car points (class 1 in learning map)
        is_car = (point_labels == self.target_class_id)
        
        # Create final mask (accounting for invalid points)
        final_mask = np.zeros(len(points), dtype=bool)
        valid_indices = np.where(valid_mask)[0]
        final_mask[valid_indices] = is_car
        
        # Extract car points
        car_indices = np.where(final_mask)[0]
        car_points = points[final_mask]
        
        # CRITICAL: Tighter FOV filtering to match geometric method's distribution
        # Diagnosis showed SalsaNext was detecting cars EVERYWHERE (Y: -54m to 59m!)
        # This caused calibration divergence. We need to filter to camera's actual FOV.
        # 
        # Target: Reduce from ~2,855 to ~1,000-1,200 points (similar to geometric: 1,029)
        # Match geometric method's spatial distribution: Y [2.8m, 27.2m], dist 13.9±4.4m
        if len(car_points) > 0:
            x = car_points[:, 0]
            y = car_points[:, 1]
            dist = np.linalg.norm(car_points[:, :3], axis=1)
            
            # Apply FOV constraints
            FOV_MIN_DISTANCE = 8.0  # At least 8m in front (avoid very close points)
            FOV_MAX_DISTANCE = 40.0 # Within 40m (typical camera range for calibration)
            FOV_LATERAL_RANGE = 12.0 # Within ±12m laterally (camera horizontal FOV)
            
            fov_mask = (
                (x > FOV_MIN_DISTANCE) &           # At least 8m in front (avoid very close points)
                (dist < FOV_MAX_DISTANCE) &       # Within 40m (typical camera range for calibration)
                (np.abs(y) < FOV_LATERAL_RANGE)    # Within ±12m laterally (camera horizontal FOV)
            )
            
            car_indices = car_indices[fov_mask]
            car_points = car_points[fov_mask]
        
        # ADDITIONAL: Geometric clustering to remove ghost points
        # SalsaNext might misclassify isolated points as cars
        # Use DBSCAN to cluster points and keep only large clusters
        if len(car_points) > 0:
            from sklearn.cluster import DBSCAN
            
            # Cluster in 3D space (parameters defined at module level)
            clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(car_points[:, :3])
            labels = clustering.labels_
            
            # Find large clusters (likely actual cars)
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            large_clusters = unique_labels[counts >= DBSCAN_MIN_CLUSTER_SIZE]
            
            if len(large_clusters) > 0:
                # Keep only points in large clusters
                cluster_mask = np.isin(labels, large_clusters)
                car_indices = car_indices[cluster_mask]
                car_points = car_points[cluster_mask]
            else:
                # No large clusters found, return empty
                car_indices = np.array([])
                car_points = np.array([]).reshape(0, 4)
        
        return car_indices, car_points

    def hybrid_get_car_points(self, points, car_mask_image, calib_current, img_h, img_w):
        """
        Hybrid car detection: Combines LiDAR segmentation with image segmentation.
        
        Process:
        1. Use SalsaNext to find candidate car points in 3D (LiDAR segmentation)
        2. Project these candidates using CURRENT calibration estimate
        3. Filter to only keep points that land on image car pixels
        
        This removes ground truth dependency while still ensuring 2D-3D correspondence.
        
        Args:
            points: Nx4 array [x, y, z, intensity]
            car_mask_image: HxW boolean mask of car pixels from image segmentation
            calib_current: Current calibration estimate (not ground truth!)
            img_h, img_w: Image dimensions
        
        Returns:
            car_indices: Indices of points classified as cars
            car_points: Nx4 array of car points
        """
        # Step 1: Get LiDAR segmentation (candidate car points)
        lidar_car_indices, lidar_car_points = self.get_car_points(points)
        
        if len(lidar_car_points) == 0:
            return np.array([]), np.array([]).reshape(0, 4)
        
        # Step 2: Project candidates using current calibration
        projected, depths = calib_current.project_velo_to_image(lidar_car_points)
        
        # Step 3: Filter to points that:
        # - Are in front of camera (depth > 0)
        # - Land within image bounds
        # - Land on car pixels in the image
        valid = (
            (depths > 0) &
            (projected[:, 0] >= 0) &
            (projected[:, 0] < img_w) &
            (projected[:, 1] >= 0) &
            (projected[:, 1] < img_h)
        )
        
        filtered_indices = []
        for i in range(len(lidar_car_points)):
            if valid[i]:
                u, v = int(projected[i, 0]), int(projected[i, 1])
                if car_mask_image[v, u]:
                    filtered_indices.append(i)
        
        if len(filtered_indices) == 0:
            return np.array([]), np.array([]).reshape(0, 4)
        
        # Return the filtered subset
        filtered_car_points = lidar_car_points[filtered_indices]
        filtered_car_indices = lidar_car_indices[filtered_indices]
        
        return filtered_car_indices, filtered_car_points


if __name__ == "__main__":
    """Test the segmenter"""
    try:
        print("Testing LiDAR Segmenter...")
        seg = LidarSegmenter()
        print("\n✓ Integration Successful!")
        print(f"  - Model ready on {seg.device}")
        print(f"  - Image size: {seg.proj_H}x{seg.proj_W}")
        print(f"  - Number of classes: {seg.nclasses}")
        print(f"  - Target class (car): {seg.target_class_id}")
    except Exception as e:
        print(f"\n✗ Integration Error: {e}")
        import traceback
        traceback.print_exc()