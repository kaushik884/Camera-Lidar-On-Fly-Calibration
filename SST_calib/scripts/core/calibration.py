import numpy as np

def load_calib_cam_to_cam(filepath):
    """Load camera-to-camera calibration from KITTI format file.
    
    Args:
        filepath: Path to calib_cam_to_cam.txt file
        
    Returns:
        P2: (3,4) projection matrix for camera 2
        R0_rect: (3,3) rectification rotation matrix
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
    
    # Parse P2 (projection matrix for left color camera - image_02)
    # P2 is a 3x4 matrix that projects 3D points to image_02
    P2 = np.array([float(x) for x in data['P_rect_02'].split()]).reshape(3, 4)
    
    # Parse R0_rect (rectification rotation matrix)
    R0_rect = np.array([float(x) for x in data['R_rect_00'].split()]).reshape(3, 3)
    
    return P2, R0_rect

def load_calib_velo_to_cam(filepath):
    """Load velodyne-to-camera calibration from KITTI format file.
    
    Args:
        filepath: Path to calib_velo_to_cam.txt file
        
    Returns:
        Tr_velo_to_cam: (4,4) transformation matrix from velodyne to camera
        R: (3,3) rotation matrix
        T: (3,1) translation vector
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
    
    # Parse rotation matrix (3x3)
    R = np.array([float(x) for x in data['R'].split()]).reshape(3, 3)
    
    # Parse translation vector (3x1)
    T = np.array([float(x) for x in data['T'].split()]).reshape(3, 1)
    
    # Create 4x4 transformation matrix
    Tr_velo_to_cam = np.vstack([
        np.hstack([R, T]),
        [0, 0, 0, 1]
    ])
    
    return Tr_velo_to_cam, R, T

class Calibration:
    def __init__(self, calib_dir):
        """Initialize calibration by loading KITTI calibration files.
        
        Args:
            calib_dir: Directory containing calibration files
        """
        # Camera intrinsics and rectification
        self.P2, self.R0_rect = load_calib_cam_to_cam(
            f"{calib_dir}/calib_cam_to_cam.txt"
        )
        
        # LIDAR to camera extrinsics
        self.Tr_velo_to_cam, self.R_velo, self.T_velo = load_calib_velo_to_cam(
            f"{calib_dir}/calib_velo_to_cam.txt"
        )
        
        # Extend R0_rect to 4x4 for matrix multiplication
        self.R0_rect_4x4 = np.eye(4)
        self.R0_rect_4x4[:3, :3] = self.R0_rect
        
        print("Calibration loaded successfully!")
        print(f"P2 (projection matrix) shape: {self.P2.shape}")
        print(f"Tr_velo_to_cam shape: {self.Tr_velo_to_cam.shape}")
    
    def project_velo_to_image(self, points):
        """
        Project LIDAR points to image plane
        
        Args:
            points: Nx4 array (x, y, z, reflectance)
        
        Returns:
            projected: Nx2 array (u, v) pixel coordinates
            depths: N array of depth values
        """
        # Keep only xyz, make homogeneous (Nx4)
        pts_3d = points[:, :3]
        num_points = pts_3d.shape[0]
        pts_3d_hom = np.hstack([pts_3d, np.ones((num_points, 1))])
        
        # Step 1: LIDAR frame -> Camera frame
        # (4x4) @ (4xN) = (4xN)
        pts_cam = self.Tr_velo_to_cam @ pts_3d_hom.T
        
        # Step 2: Apply rectification
        # (4x4) @ (4xN) = (4xN)
        pts_rect = self.R0_rect_4x4 @ pts_cam
        
        # Step 3: Project to image plane
        # (3x4) @ (4xN) = (3xN)
        pts_2d = self.P2 @ pts_rect
        
        # Normalize by depth (z)
        depths = pts_2d[2, :]
        pts_2d[0, :] /= depths
        pts_2d[1, :] /= depths
        
        # Return as Nx2
        projected = pts_2d[:2, :].T
        
        return projected, depths


# Test it
if __name__ == "__main__":
    calib = Calibration("dataset/2011_09_26")
    
    print(f"\nExtrinsic rotation R:\n{calib.R_velo}")
    print(f"\nExtrinsic translation T:\n{calib.T_velo.flatten()}")