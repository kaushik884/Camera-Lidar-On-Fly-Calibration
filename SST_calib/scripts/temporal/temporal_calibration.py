"""
Step 7: Temporal Calibration

Estimates the time delay (δ) between LIDAR and camera using visual odometry.

The key insight:
- If there's a time delay δ, the LIDAR sees the world at time t
- But the camera sees it at time t + δ
- This causes a spatial offset proportional to velocity: offset = v * δ

We estimate velocity using visual odometry between consecutive frames,
then optimize δ to minimize the alignment loss.
"""

import numpy as np
import cv2
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class VisualOdometry:
    """
    Estimates camera motion between consecutive frames using optical flow.
    
    Based on:
    - FAST feature detection
    - Lucas-Kanade optical flow tracking
    - Essential matrix estimation with RANSAC
    """
    
    def __init__(self, K):
        """
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        
        # FAST feature detector parameters
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def estimate_motion(self, img1, img2):
        """
        Estimate camera motion from img1 to img2.
        
        Args:
            img1: first RGB image (numpy array)
            img2: second RGB image (numpy array)
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector (unit length)
            velocity: estimated velocity in camera frame
            success: whether estimation was successful
        """
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = img2
        
        # Detect features in first image
        pts1 = cv2.goodFeaturesToTrack(gray1, **self.feature_params)
        
        if pts1 is None or len(pts1) < 10:
            return None, None, None, False
        
        # Track features to second image
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None, **self.lk_params)
        
        # Filter good matches
        good_mask = status.flatten() == 1
        pts1_good = pts1[good_mask].reshape(-1, 2)
        pts2_good = pts2[good_mask].reshape(-1, 2)
        
        if len(pts1_good) < 8:
            return None, None, None, False
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1_good, pts2_good, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None, None, None, False
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1_good, pts2_good, self.K)
        
        # t is unit vector, we need to estimate scale
        # For driving, we can assume forward motion and estimate from feature displacement
        avg_flow = np.mean(pts2_good - pts1_good, axis=0)
        
        # Rough velocity estimate (this is approximate)
        # In real applications, you'd use IMU or known baseline
        velocity = t.flatten()
        
        return R, t, velocity, True
    
    def estimate_velocity(self, img1, img2, dt=0.1):
        """
        Estimate velocity between two frames.
        
        Args:
            img1: first image
            img2: second image
            dt: time between frames (seconds)
            
        Returns:
            velocity: 3D velocity vector in camera frame (m/s)
            success: whether estimation succeeded
        """
        R, t, _, success = self.estimate_motion(img1, img2)
        
        if not success:
            return np.zeros(3), False
        
        # The translation from visual odometry is up to scale
        # For KITTI, we can estimate scale from typical driving speeds
        # or use GPS/IMU data
        
        # Heuristic: assume typical driving speed of ~10 m/s
        # The visual odometry gives direction, we scale by typical speed
        
        # A better approach would use the actual scale from GPS
        # For now, we'll use optical flow magnitude as a proxy
        
        # Convert to grayscale for flow computation
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Average flow magnitude (pixels per frame)
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_flow = np.mean(flow_mag)
        
        # Convert to approximate velocity using camera parameters
        # This is a rough estimate: v ≈ flow * depth / focal_length / dt
        # Assuming average depth of ~20m for driving scenes
        avg_depth = 20.0  # meters
        scale = avg_flow * avg_depth / self.fx / dt
        
        velocity = t.flatten() * scale
        
        return velocity, True


class TemporalCalibrator:
    """
    Estimates time delay between LIDAR and camera.
    """
    
    def __init__(self, spatial_calib_R, spatial_calib_t, K, img_size):
        """
        Args:
            spatial_calib_R: calibrated rotation matrix
            spatial_calib_t: calibrated translation vector
            K: camera intrinsic matrix
            img_size: (width, height)
        """
        self.R = spatial_calib_R
        self.t_base = spatial_calib_t
        self.K = K
        self.img_size = img_size
        self.vo = VisualOdometry(K)
    
    def estimate_time_delay(self, frames_data, dt=0.1, verbose=True):
        """
        Estimate time delay using multiple frame pairs.
        
        Args:
            frames_data: list of (image, point_cloud, binary_mask, car_points_3d) tuples
            dt: time between consecutive frames (seconds)
            verbose: print progress
            
        Returns:
            delta: estimated time delay (seconds)
            result: optimization result dict
        """
        if len(frames_data) < 2:
            raise ValueError("Need at least 2 frames")
        
        if verbose:
            print("="*60)
            print("Temporal Calibration")
            print("="*60)
            print(f"Number of frames: {len(frames_data)}")
        
        # Estimate velocities between consecutive frames
        velocities = []
        valid_pairs = []
        
        for i in range(len(frames_data) - 1):
            img1 = frames_data[i][0]
            img2 = frames_data[i + 1][0]
            
            vel, success = self.vo.estimate_velocity(img1, img2, dt)
            
            if success:
                velocities.append(vel)
                valid_pairs.append((i, i + 1))
                if verbose:
                    print(f"  Pair {i}-{i+1}: velocity = {vel}")
        
        if len(velocities) == 0:
            print("Failed to estimate any velocities")
            return 0.0, {'success': False}
        
        avg_velocity = np.mean(velocities, axis=0)
        if verbose:
            print(f"\nAverage velocity: {avg_velocity}")
            print(f"Speed: {np.linalg.norm(avg_velocity):.2f} m/s")
        
        # Now optimize time delay
        # For each frame, compute loss with time-compensated translation
        
        from core.losses import SemanticAlignmentLoss, CalibrationLoss
        
        loss_calc = SemanticAlignmentLoss(pixel_downsample_rate=0.05, seed=42)
        
        def compute_loss_with_delay(delta):
            """Compute average loss across frames with given time delay."""
            total_loss = 0
            count = 0
            
            for i, (img, pc, mask, car_pts) in enumerate(frames_data):
                if len(car_pts) < 20:
                    continue
                
                # Compute velocity for this frame (use average if not available)
                if i < len(velocities):
                    vel = velocities[i]
                else:
                    vel = avg_velocity
                
                # Time-compensated translation
                # The LIDAR is "behind" in time, so we shift it forward
                t_compensated = self.t_base + (vel * delta).reshape(3, 1)
                
                calib_loss = CalibrationLoss(
                    self.K, self.img_size, mask, car_pts, loss_calc
                )
                
                loss = calib_loss.compute(self.R, t_compensated, weight_i2p=1.0)
                
                if not np.isinf(loss):
                    total_loss += loss
                    count += 1
            
            return total_loss / max(count, 1)
        
        # Search for optimal delta
        # KITTI has 10Hz sensors, so max delay is ~100ms
        if verbose:
            print("\nSearching for optimal time delay...")
        
        # Grid search first
        deltas = np.linspace(-0.15, 0.15, 31)  # -150ms to +150ms
        losses = [compute_loss_with_delay(d) for d in deltas]
        
        best_idx = np.argmin(losses)
        best_delta_coarse = deltas[best_idx]
        
        if verbose:
            print(f"  Coarse search: best delta = {best_delta_coarse*1000:.1f} ms")
        
        # Fine search around best
        result = minimize_scalar(
            compute_loss_with_delay,
            bounds=(best_delta_coarse - 0.02, best_delta_coarse + 0.02),
            method='bounded'
        )
        
        best_delta = result.x
        
        if verbose:
            print(f"  Fine search: best delta = {best_delta*1000:.1f} ms")
            print(f"  Loss at delta=0: {compute_loss_with_delay(0):.4f}")
            print(f"  Loss at best delta: {result.fun:.4f}")
        
        # Plot loss vs delta
        plt.figure(figsize=(10, 4))
        plt.plot(np.array(deltas) * 1000, losses, 'b-', linewidth=2)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Zero delay')
        plt.axvline(x=best_delta*1000, color='red', linestyle='-', label=f'Optimal: {best_delta*1000:.1f}ms')
        plt.xlabel('Time Delay (ms)')
        plt.ylabel('Loss')
        plt.title('Loss vs Time Delay')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/temporal_calibration.png', dpi=150)
        plt.show()
        
        return best_delta, {
            'success': True,
            'delta_ms': best_delta * 1000,
            'avg_velocity': avg_velocity,
            'losses': losses,
            'deltas': deltas
        }


def test_temporal_calibration():
    """Test temporal calibration."""
    import sys
    sys.path.append('src')
    from utils.data_loader import KITTIDataLoader
    from segmentation.image_segmentation import ImageSegmentor
    from core.losses import segment_car_points_geometric
    from core.optimizer import compute_calibration_error
    
    # =====================================================
    # CONFIGURE THESE PATHS
    # =====================================================
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    print("Loading data...")
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    img_segmentor = ImageSegmentor(target_classes=['car', 'bus'])
    
    R_gt, t_gt = loader.get_ground_truth_extrinsics()
    K = loader.get_intrinsic_matrix()
    img_size = loader.get_image_size()
    
    # Prepare frames (use consecutive frames for velocity estimation)
    print("\nPreparing frames...")
    frame_indices = list(range(0, 50, 5))  # Every 5th frame from 0 to 45
    frames_data = []
    
    for idx in frame_indices:
        image, point_cloud = loader.load_frame_pair(idx)
        _, binary_mask = img_segmentor.segment(image)
        
        car_points_3d = segment_car_points_geometric(
            point_cloud, R_gt, t_gt, K, img_size, binary_mask
        )
        
        frames_data.append((image, point_cloud, binary_mask, car_points_3d))
        print(f"  Frame {idx}: {len(car_points_3d)} car points")
    
    # Run temporal calibration
    # Use ground truth spatial calibration (or optimized one)
    temporal_calib = TemporalCalibrator(R_gt, t_gt, K, img_size)
    
    # KITTI has 10Hz capture rate, so dt = 0.1s between frames
    # But we're using every 5th frame, so dt = 0.5s
    delta, result = temporal_calib.estimate_time_delay(
        frames_data, 
        dt=0.5,  # 5 frames * 0.1s/frame
        verbose=True
    )
    
    print("\n" + "="*60)
    print("TEMPORAL CALIBRATION RESULT")
    print("="*60)
    print(f"Estimated time delay: {delta * 1000:.2f} ms")
    
    # In KITTI raw synced, the data should be synchronized
    # So we expect delta ≈ 0
    print(f"\nNote: KITTI synced data should have ~0ms delay")
    print(f"Non-zero values indicate either:")
    print(f"  1. Residual synchronization error in the dataset")
    print(f"  2. Motion during capture causing apparent offset")
    print(f"  3. Segmentation boundary effects")


if __name__ == "__main__":
    test_temporal_calibration()