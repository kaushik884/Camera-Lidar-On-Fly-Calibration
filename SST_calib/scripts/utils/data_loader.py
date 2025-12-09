"""
Step 3: KITTI Data Loader

This module handles:
1. Loading RGB images from image_02
2. Loading Velodyne point clouds from velodyne_points
3. Parsing calibration files (intrinsics, extrinsics)
4. Projecting points to camera frame

The KITTI coordinate systems:
- Velodyne: x=forward, y=left, z=up
- Camera: x=right, y=down, z=forward
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path


class KITTIDataLoader:
    """
    Data loader for KITTI raw dataset.
    
    Handles loading synchronized image and velodyne data,
    along with calibration parameters.
    """
    
    def __init__(self, base_path, date, drive):
        """
        Initialize the data loader.
        
        Args:
            base_path: Path to 'dataset' folder
            date: Date string like '2011_09_26'
            drive: Drive string like '0005'
        """
        self.base_path = Path(base_path)
        self.date = date
        self.drive = drive
        
        # Build paths
        self.date_folder = self.base_path / date
        self.drive_folder = self.date_folder / f"{date}_drive_{drive}_sync"
        
        # Data folders
        self.image_folder = self.drive_folder / "image_02" / "data"
        self.velodyne_folder = self.drive_folder / "velodyne_points" / "data"
        
        # Calibration files
        self.calib_cam_path = self.date_folder / "calib_cam_to_cam.txt"
        self.calib_velo_path = self.date_folder / "calib_velo_to_cam.txt"
        
        # Load calibration
        self.calib = self._load_calibration()
        
        # Get frame list
        self.frames = self._get_frame_list()
        
        print(f"KITTIDataLoader initialized:")
        print(f"  Drive: {date}_drive_{drive}_sync")
        print(f"  Frames: {len(self.frames)}")
        print(f"  Image size: {self.get_image_size()}")
    
    def _get_frame_list(self):
        """Get sorted list of frame indices."""
        image_files = sorted(os.listdir(self.image_folder))
        # Extract frame numbers (remove .png extension)
        frames = [int(f.split('.')[0]) for f in image_files if f.endswith('.png')]
        return frames
    
    def _load_calibration(self):
        """
        Load all calibration data.
        
        Returns dict with:
            - K: 3x3 camera intrinsic matrix
            - P_rect_02: 3x4 projection matrix for camera 2
            - R_velo_to_cam: 3x3 rotation from velodyne to camera 0
            - T_velo_to_cam: 3x1 translation from velodyne to camera 0
            - R_rect_00: 3x3 rectification rotation for camera 0
            - T_cam0_to_cam2: translation from camera 0 to camera 2
        """
        calib = {}
        
        # ----- Load velo to cam calibration -----
        velo_calib = self._read_calib_file(self.calib_velo_path)
        
        # Rotation and translation from velodyne to camera 0
        calib['R_velo_to_cam'] = np.array(
            [float(x) for x in velo_calib['R'].split()]
        ).reshape(3, 3)
        
        calib['T_velo_to_cam'] = np.array(
            [float(x) for x in velo_calib['T'].split()]
        ).reshape(3, 1)
        
        # ----- Load camera calibration -----
        cam_calib = self._read_calib_file(self.calib_cam_path)
        
        # Rectification rotation (for camera 0)
        calib['R_rect_00'] = np.array(
            [float(x) for x in cam_calib['R_rect_00'].split()]
        ).reshape(3, 3)
        
        # Projection matrix for camera 2 (3x4)
        calib['P_rect_02'] = np.array(
            [float(x) for x in cam_calib['P_rect_02'].split()]
        ).reshape(3, 4)
        
        # Extract intrinsic matrix K from P_rect_02
        # P = K @ [R | t], for rectified camera R=I, so K = P[:3,:3]
        calib['K'] = calib['P_rect_02'][:3, :3].copy()
        
        # The 4th column of P_rect_02 gives the translation to camera 2
        # P_rect_02 = K @ [I | t], so t = K^{-1} @ P_rect_02[:, 3]
        calib['T_cam0_to_cam2'] = np.linalg.inv(calib['K']) @ calib['P_rect_02'][:, 3:4]
        
        # ----- Build complete transformation matrix -----
        # Transform from velodyne to rectified camera 2
        calib['T_velo_to_cam2'] = self._build_velo_to_cam2_transform(calib)
        
        return calib
    
    def _read_calib_file(self, filepath):
        """Read a KITTI calibration file into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
        return data
    
    def _build_velo_to_cam2_transform(self, calib):
        """
        Build the full 4x4 transformation matrix from velodyne to camera 2.
        
        The transformation chain is:
        1. Velodyne -> Camera 0 (using R_velo_to_cam, T_velo_to_cam)
        2. Camera 0 -> Rectified Camera 0 (using R_rect_00)
        3. Rectified Camera 0 -> Rectified Camera 2 (using T_cam0_to_cam2)
        """
        # Step 1: Velodyne to Camera 0
        T_v2c = np.eye(4)
        T_v2c[:3, :3] = calib['R_velo_to_cam']
        T_v2c[:3, 3:4] = calib['T_velo_to_cam']
        
        # Step 2: Apply rectification
        T_rect = np.eye(4)
        T_rect[:3, :3] = calib['R_rect_00']
        
        # Step 3: Camera 0 to Camera 2 (just translation)
        T_c0_c2 = np.eye(4)
        T_c0_c2[:3, 3:4] = calib['T_cam0_to_cam2']
        
        # Combined transformation
        T_velo_to_cam2 = T_c0_c2 @ T_rect @ T_v2c
        
        return T_velo_to_cam2
    
    def __len__(self):
        """Return number of frames."""
        return len(self.frames)
    
    def get_image_size(self):
        """Get image dimensions (width, height)."""
        if len(self.frames) == 0:
            return None
        sample_path = self.image_folder / f"{self.frames[0]:010d}.png"
        img = Image.open(sample_path)
        return img.size  # (width, height)
    
    def load_image(self, frame_idx):
        """
        Load RGB image for a given frame.
        
        Args:
            frame_idx: Frame index (0, 1, 2, ...)
            
        Returns:
            numpy array of shape (H, W, 3) with dtype uint8
        """
        frame_num = self.frames[frame_idx]
        image_path = self.image_folder / f"{frame_num:010d}.png"
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        return img_array
    
    def load_velodyne(self, frame_idx):
        """
        Load Velodyne point cloud for a given frame.
        
        Args:
            frame_idx: Frame index (0, 1, 2, ...)
            
        Returns:
            numpy array of shape (N, 4) with columns [x, y, z, reflectance]
            Coordinates are in Velodyne frame (x=forward, y=left, z=up)
        """
        frame_num = self.frames[frame_idx]
        velo_path = self.velodyne_folder / f"{frame_num:010d}.bin"
        
        # Read binary file
        point_cloud = np.fromfile(str(velo_path), dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)
        
        return point_cloud
    
    def load_frame_pair(self, frame_idx):
        """
        Load both image and point cloud for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            tuple: (image, point_cloud)
        """
        image = self.load_image(frame_idx)
        point_cloud = self.load_velodyne(frame_idx)
        return image, point_cloud
    
    def get_ground_truth_extrinsics(self):
        """
        Get the ground truth extrinsic parameters (R, t) from velodyne to camera 2.
        
        This is what SST-Calib is trying to estimate.
        
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        T = self.calib['T_velo_to_cam2']
        R = T[:3, :3]
        t = T[:3, 3:4]
        return R, t
    
    def project_velodyne_to_image(self, point_cloud, R=None, t=None):
        """
        Project velodyne points onto the image plane.
        
        Args:
            point_cloud: (N, 4) array of velodyne points [x, y, z, reflectance]
            R: Optional 3x3 rotation matrix (uses ground truth if None)
            t: Optional 3x1 translation vector (uses ground truth if None)
            
        Returns:
            points_2d: (M, 2) array of pixel coordinates [u, v]
            points_3d: (M, 4) array of corresponding 3D points (filtered)
            valid_mask: boolean mask of which original points are valid
        """
        # Use ground truth if no extrinsics provided
        if R is None or t is None:
            R_gt, t_gt = self.get_ground_truth_extrinsics()
            R = R_gt if R is None else R
            t = t_gt if t is None else t
        
        # Get intrinsic matrix
        K = self.calib['K']
        
        # Get image dimensions
        img_width, img_height = self.get_image_size()
        
        # Extract XYZ coordinates
        points_xyz = point_cloud[:, :3]  # (N, 3)
        
        # Transform to camera frame: p_cam = R @ p_velo + t
        points_cam = (R @ points_xyz.T + t).T  # (N, 3)
        
        # Filter points behind the camera (z <= 0)
        valid_depth = points_cam[:, 2] > 0
        
        # Project to image plane
        # p_img = K @ p_cam (normalized by z)
        points_cam_valid = points_cam[valid_depth]
        
        # Homogeneous projection
        points_proj = (K @ points_cam_valid.T).T  # (M, 3)
        
        # Normalize by depth
        points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
        
        # Filter points outside image bounds
        valid_bounds = (
            (points_2d[:, 0] >= 0) & 
            (points_2d[:, 0] < img_width) &
            (points_2d[:, 1] >= 0) & 
            (points_2d[:, 1] < img_height)
        )
        
        # Create full valid mask
        full_valid_mask = np.zeros(len(point_cloud), dtype=bool)
        valid_depth_indices = np.where(valid_depth)[0]
        full_valid_mask[valid_depth_indices[valid_bounds]] = True
        
        # Filter outputs
        points_2d_valid = points_2d[valid_bounds]
        points_3d_valid = point_cloud[full_valid_mask]
        
        return points_2d_valid, points_3d_valid, full_valid_mask
    
    def get_intrinsic_matrix(self):
        """Get the 3x3 camera intrinsic matrix K."""
        return self.calib['K'].copy()
    
    def get_projection_matrix(self):
        """Get the 3x4 projection matrix P_rect_02."""
        return self.calib['P_rect_02'].copy()


def test_data_loader():
    """Test the data loader with visualization."""
    import matplotlib.pyplot as plt
    
    # =====================================================
    # CONFIGURE THESE PATHS FOR YOUR SYSTEM
    # =====================================================
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    # Create loader
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    
    # Print calibration info
    print("\n" + "=" * 60)
    print("Calibration Information")
    print("=" * 60)
    
    R_gt, t_gt = loader.get_ground_truth_extrinsics()
    print(f"\nGround Truth Rotation R:\n{R_gt}")
    print(f"\nGround Truth Translation t:\n{t_gt.flatten()}")
    print(f"\nIntrinsic Matrix K:\n{loader.get_intrinsic_matrix()}")
    
    # Load a sample frame
    print("\n" + "=" * 60)
    print("Loading Sample Frame")
    print("=" * 60)
    
    frame_idx = 0
    image, point_cloud = loader.load_frame_pair(frame_idx)
    
    print(f"\nFrame {frame_idx}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Point cloud shape: {point_cloud.shape}")
    
    # Project points to image
    points_2d, points_3d, valid_mask = loader.project_velodyne_to_image(point_cloud)
    
    print(f"  Projected points: {len(points_2d)} (of {len(point_cloud)} total)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Image with projected points
    axes[0].imshow(image)
    
    # Color points by depth
    depths = points_3d[:, 0]  # x in velodyne = forward = depth
    scatter = axes[0].scatter(
        points_2d[:, 0], 
        points_2d[:, 1], 
        c=depths, 
        cmap='jet', 
        s=1, 
        alpha=0.5
    )
    plt.colorbar(scatter, ax=axes[0], label='Depth (m)')
    axes[0].set_title(f'Frame {frame_idx}: Image with Projected LIDAR Points')
    axes[0].set_xlabel('u (pixels)')
    axes[0].set_ylabel('v (pixels)')
    
    # Bird's eye view of point cloud
    ax_bev = axes[1]
    bev_scatter = ax_bev.scatter(
        point_cloud[:, 0],  # x = forward
        point_cloud[:, 1],  # y = left
        c=point_cloud[:, 2],  # z = height for color
        cmap='viridis',
        s=0.5,
        alpha=0.5
    )
    plt.colorbar(bev_scatter, ax=ax_bev, label='Height (m)')
    ax_bev.set_xlabel('X (forward, m)')
    ax_bev.set_ylabel('Y (left, m)')
    ax_bev.set_title("Bird's Eye View of Point Cloud")
    ax_bev.set_aspect('equal')
    ax_bev.set_xlim(0, 80)
    ax_bev.set_ylim(-40, 40)
    
    plt.tight_layout()
    plt.savefig('outputs/data_loader_test.png', dpi=150)
    plt.show()
    
    print(f"\nVisualization saved to 'outputs/data_loader_test.png'")
    
    return loader


if __name__ == "__main__":
    test_data_loader()