"""
Step 6 (Fixed): Robust Calibration Optimizer

Key fixes:
1. Use axis-angle for rotation (avoids gimbal lock)
2. Better initial perturbation
3. More robust optimization settings
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import time
import matplotlib.pyplot as plt

# Optimization Bounds
DEFAULT_ROTATION_BOUND_DEG = 15  # degrees - max rotation deviation from initial
DEFAULT_TRANSLATION_BOUND_M = 0.20  # meters - max translation deviation from initial

# Regularization Parameters (Equation 9 in paper)
REGULARIZATION_LAMBDA_T = 1e4  # translation regularization weight
REGULARIZATION_LAMBDA_R = 1e7  # rotation regularization weight

# Optimization Tolerance
OPTIMIZATION_FTOL = 1e-6  # function tolerance for convergence


def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation."""
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()  # axis * angle
    return rotvec


def axis_angle_to_rotation_matrix(rotvec):
    """Convert axis-angle to rotation matrix."""
    rot = Rotation.from_rotvec(rotvec)
    return rot.as_matrix()


class MultiFrameOptimizer:
    """
    Optimizer that uses multiple frames for robust calibration.
    Uses axis-angle representation to avoid gimbal lock.
    """
    
    def __init__(self, calib_loss_funcs, R_init, t_init):
        """
        Args:
            calib_loss_funcs: list of CalibrationLoss instances
            R_init: initial rotation matrix
            t_init: initial translation vector
        """
        self.calib_losses = calib_loss_funcs
        self.R_init = R_init.copy()
        self.t_init = t_init.copy()
        
        # Convert to axis-angle
        self.rotvec_init = rotation_matrix_to_axis_angle(R_init)
        
        self.n_evals = 0
        self.best_loss = float('inf')
        self.best_params = None
    
    def _params_to_Rt(self, params):
        """
        params: [rx, ry, rz, tx, ty, tz]
                rotation as axis-angle (radians), translation in meters
        """
        rotvec = params[:3]
        t = params[3:6].reshape(3, 1)
        R = axis_angle_to_rotation_matrix(rotvec)
        return R, t
    
    def _Rt_to_params(self, R, t):
        """Convert R, t to optimization parameters."""
        rotvec = rotation_matrix_to_axis_angle(R)
        return np.concatenate([rotvec, t.flatten()])
    
    def _objective(self, params, weight_i2p=1.0, reg_lambda_t=0.0, reg_lambda_r=0.0):
        """Compute average loss across all frames."""
        R, t = self._params_to_Rt(params)
        
        total_loss = 0
        valid_count = 0
        
        for calib_loss in self.calib_losses:
            loss = calib_loss.compute(R, t, weight_i2p=weight_i2p)
            if not np.isinf(loss):
                total_loss += loss
                valid_count += 1
        
        if valid_count == 0:
            return 1e10
        
        avg_loss = total_loss / valid_count
        
        # 2. Regularization (Eq 9 in paper)
        # Penalize deviation from initialization (R_init, t_init)
        
        # Translation L2 difference
        t_diff = np.linalg.norm(t - self.t_init) ** 2
        loss_reg_t = reg_lambda_t * t_diff
        
        # Rotation difference (using geodesic distance roughly via frobenius norm)
        # || R * R_init^-1 || is related to the magnitude of the rotation difference
        R_diff = R @ self.R_init.T
        loss_reg_r = reg_lambda_r * (np.linalg.norm(R_diff - np.eye(3)) ** 2)
        
        if self.n_evals % 100 == 0:
            print(f"  Eval {self.n_evals}: loss={avg_loss:.4f}, best={self.best_loss:.4f}")
        
        return avg_loss
    
    def optimize(self, weight_schedule=None, verbose=True, regularize=False):
        """Run optimization with weight scheduling."""
        self.n_evals = 0
        self.best_loss = float('inf')
        self.best_params = None
        
        params_init = self._Rt_to_params(self.R_init, self.t_init)
        
        if verbose:
            print("="*60)
            print("Multi-Frame Calibration Optimization")
            print("="*60)
            print(f"Number of frames: {len(self.calib_losses)}")
            print(f"Initial axis-angle (rad): {params_init[:3]}")
            print(f"Initial translation (m): {params_init[3:]}")
        
        # Weight schedule from paper
        if weight_schedule is None:
            weight_schedule = [
                (20.0, 20),   # High I2P weight: pull points into mask
                (1.0, 30),    # Balanced
                (0.02, 20),   # Low I2P: fine-tune P2I
            ]
        
        params = params_init.copy()
        
        # Bounds from constants
        rot_bound = DEFAULT_ROTATION_BOUND_DEG * np.pi / 180  # Convert to radians
        trans_bound = DEFAULT_TRANSLATION_BOUND_M
        
        bounds = [
            (params_init[0] - rot_bound, params_init[0] + rot_bound),
            (params_init[1] - rot_bound, params_init[1] + rot_bound),
            (params_init[2] - rot_bound, params_init[2] + rot_bound),
            (params_init[3] - trans_bound, params_init[3] + trans_bound),
            (params_init[4] - trans_bound, params_init[4] + trans_bound),
            (params_init[5] - trans_bound, params_init[5] + trans_bound),
        ]
        
        start_time = time.time()
        reg_t = REGULARIZATION_LAMBDA_T if regularize else 0.0
        reg_r = REGULARIZATION_LAMBDA_R if regularize else 0.0
        for stage, (weight_i2p, n_iter) in enumerate(weight_schedule):
            if verbose:
                print(f"\nStage {stage+1}: weight_i2p={weight_i2p}, max_iter={n_iter}")
            
            def obj_func(p):
                return self._objective(p, weight_i2p=weight_i2p, reg_lambda_t=reg_t, reg_lambda_r=reg_r)
            
            result = minimize(
                obj_func,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': n_iter, 'disp': False, 'ftol': OPTIMIZATION_FTOL}
            )
            
            params = result.x
            
            if verbose:
                print(f"  Stage loss: {result.fun:.4f}")
        
        # Use best found parameters
        if self.best_params is not None:
            params = self.best_params
        
        elapsed = time.time() - start_time
        R_opt, t_opt = self._params_to_Rt(params)
        
        if verbose:
            print(f"\nOptimization complete in {elapsed:.2f}s")
            print(f"Total evaluations: {self.n_evals}")
            print(f"Best loss: {self.best_loss:.4f}")
            euler_opt = Rotation.from_matrix(R_opt).as_euler('xyz', degrees=True)
            print(f"Final Euler (deg): {euler_opt}")
            print(f"Final translation (m): {t_opt.flatten()}")
        
        return R_opt, t_opt, {
            'R': R_opt, 
            't': t_opt,
            'final_loss': self.best_loss,
            'elapsed_time': elapsed,
            'n_evals': self.n_evals
        }


def compute_calibration_error(R_est, t_est, R_gt, t_gt):
    """Compute calibration error metrics."""
    # Translation error (ATD - Average Translation Difference) in cm
    t_diff = np.abs(t_est.flatten() - t_gt.flatten())
    atd = np.mean(t_diff) * 100  # Convert to cm
    
    # Per-axis translation error
    t_err_xyz = t_diff * 100  # cm
    
    # Rotation error - QAD (Quaternion Angle Difference)
    q_est = Rotation.from_matrix(R_est).as_quat()
    q_gt = Rotation.from_matrix(R_gt).as_quat()
    dot = np.clip(np.abs(np.dot(q_est, q_gt)), 0, 1)
    qad = 2 * np.arccos(dot) * 180 / np.pi
    
    # AEAD (Average Euler Angle Difference)
    euler_est = Rotation.from_matrix(R_est).as_euler('xyz', degrees=True)
    euler_gt = Rotation.from_matrix(R_gt).as_euler('xyz', degrees=True)
    
    # Handle angle wrapping
    euler_diff = euler_est - euler_gt
    euler_diff = np.abs((euler_diff + 180) % 360 - 180)
    aead = np.mean(euler_diff)
    
    return {
        'atd_cm': atd,
        't_err_xyz_cm': t_err_xyz,
        'qad_deg': qad,
        'aead_deg': aead,
        'euler_diff_deg': euler_diff
    }


def test_optimizer():
    """Test the calibration optimizer."""
    import sys
    sys.path.append('src')
    from utils.data_loader import KITTIDataLoader
    from segmentation.image_segmentation import ImageSegmentor
    from losses import (SemanticAlignmentLoss, CalibrationLoss, 
                        segment_car_points_geometric)
    
    # =====================================================
    # CONFIGURE THESE PATHS FOR YOUR SYSTEM
    # =====================================================
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    print("Loading data and models...")
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    img_segmentor = ImageSegmentor(target_classes=['car', 'bus'])
    
    R_gt, t_gt = loader.get_ground_truth_extrinsics()
    K = loader.get_intrinsic_matrix()
    img_size = loader.get_image_size()
    
    euler_gt = Rotation.from_matrix(R_gt).as_euler('xyz', degrees=True)
    print(f"\nGround truth:")
    print(f"  Translation (m): {t_gt.flatten()}")
    print(f"  Euler angles (deg): {euler_gt}")
    
    loss_calc = SemanticAlignmentLoss(pixel_downsample_rate=0.05, seed=42)
    
    # Prepare frames
    test_frames = [0, 5, 10, 15, 20, 25, 30]
    calib_losses = []
    
    print(f"\nPreparing {len(test_frames)} frames...")
    for frame_idx in test_frames:
        image, point_cloud = loader.load_frame_pair(frame_idx)
        _, binary_mask = img_segmentor.segment(image)
        
        car_points_3d = segment_car_points_geometric(
            point_cloud, R_gt, t_gt, K, img_size, binary_mask
        )
        
        if len(car_points_3d) >= 50:
            calib_loss = CalibrationLoss(K, img_size, binary_mask, car_points_3d, loss_calc)
            calib_losses.append(calib_loss)
            print(f"  Frame {frame_idx}: {len(car_points_3d)} car points")
    
    print(f"\nUsing {len(calib_losses)} frames")
    
    # ===== Test: Optimize from perturbed initial guess =====
    print("\n" + "="*60)
    print("Test: Optimize from perturbed initial guess")
    print("="*60)
    
    # Create controlled perturbation
    np.random.seed(123)
    
    # Translation noise: ±5cm per axis
    t_noise = np.random.uniform(-0.05, 0.05, (3, 1))
    t_init = t_gt + t_noise
    
    # Rotation noise: ±3 degrees per axis
    euler_noise = np.random.uniform(-3, 3, 3)
    R_noise = Rotation.from_euler('xyz', euler_noise, degrees=True).as_matrix()
    R_init = R_gt @ R_noise
    
    euler_init = Rotation.from_matrix(R_init).as_euler('xyz', degrees=True)
    
    print(f"\nInitial guess (perturbed):")
    print(f"  Translation (m): {t_init.flatten()}")
    print(f"  Euler angles (deg): {euler_init}")
    print(f"  Translation noise (cm): {(t_noise.flatten() * 100)}")
    print(f"  Rotation noise (deg): {euler_noise}")
    
    # Compute initial error
    init_error = compute_calibration_error(R_init, t_init, R_gt, t_gt)
    print(f"\nInitial error:")
    print(f"  ATD: {init_error['atd_cm']:.2f} cm")
    print(f"  QAD: {init_error['qad_deg']:.2f}°")
    print(f"  AEAD: {init_error['aead_deg']:.2f}°")
    
    # Optimize
    optimizer = MultiFrameOptimizer(calib_losses, R_init, t_init)
    R_opt, t_opt, result = optimizer.optimize(verbose=True)
    
    # Compute final error
    final_error = compute_calibration_error(R_opt, t_opt, R_gt, t_gt)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Initial':<15} {'Final':<15} {'Improvement':<15}")
    print("-"*65)
    print(f"{'ATD (cm)':<20} {init_error['atd_cm']:<15.2f} {final_error['atd_cm']:<15.2f} {init_error['atd_cm']-final_error['atd_cm']:<15.2f}")
    print(f"{'QAD (deg)':<20} {init_error['qad_deg']:<15.2f} {final_error['qad_deg']:<15.2f} {init_error['qad_deg']-final_error['qad_deg']:<15.2f}")
    print(f"{'AEAD (deg)':<20} {init_error['aead_deg']:<15.2f} {final_error['aead_deg']:<15.2f} {init_error['aead_deg']-final_error['aead_deg']:<15.2f}")
    
    print(f"\nPer-axis translation error (cm):")
    print(f"  X: {init_error['t_err_xyz_cm'][0]:.2f} -> {final_error['t_err_xyz_cm'][0]:.2f}")
    print(f"  Y: {init_error['t_err_xyz_cm'][1]:.2f} -> {final_error['t_err_xyz_cm'][1]:.2f}")
    print(f"  Z: {init_error['t_err_xyz_cm'][2]:.2f} -> {final_error['t_err_xyz_cm'][2]:.2f}")
    
    print(f"\nPer-axis rotation error (deg):")
    print(f"  Roll:  {init_error['euler_diff_deg'][0]:.2f} -> {final_error['euler_diff_deg'][0]:.2f}")
    print(f"  Pitch: {init_error['euler_diff_deg'][1]:.2f} -> {final_error['euler_diff_deg'][1]:.2f}")
    print(f"  Yaw:   {init_error['euler_diff_deg'][2]:.2f} -> {final_error['euler_diff_deg'][2]:.2f}")
    
    # ===== Visualize =====
    print("\nGenerating visualization...")
    
    # Load frame for visualization
    frame_idx = 0
    image, point_cloud = loader.load_frame_pair(frame_idx)
    _, binary_mask = img_segmentor.segment(image)
    car_points_3d = segment_car_points_geometric(
        point_cloud, R_gt, t_gt, K, img_size, binary_mask
    )
    
    def project_points(R, t, points_3d):
        pts_cam = (R @ points_3d.T + t).T
        valid = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[valid]
        if len(pts_cam) == 0:
            return np.array([]).reshape(0, 2)
        proj = (K @ pts_cam.T).T
        pts_2d = proj[:, :2] / proj[:, 2:3]
        in_bounds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_size[0]) & \
                    (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_size[1])
        return pts_2d[in_bounds]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    mask_overlay = np.zeros((*binary_mask.shape, 4))
    mask_overlay[binary_mask] = [1, 1, 0, 0.4]
    
    titles = [
        f"Initial Guess\nATD={init_error['atd_cm']:.1f}cm, QAD={init_error['qad_deg']:.1f}°",
        f"Optimized\nATD={final_error['atd_cm']:.1f}cm, QAD={final_error['qad_deg']:.1f}°",
        "Ground Truth"
    ]
    Rs = [R_init, R_opt, R_gt]
    ts = [t_init, t_opt, t_gt]
    
    for ax, R, t, title in zip(axes, Rs, ts, titles):
        ax.imshow(image)
        ax.imshow(mask_overlay)
        pts = project_points(R, t, car_points_3d)
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c='red', s=3, alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/optimization_result.png', dpi=150)
    plt.show()
    
    print("\nResults saved to outputs/optimization_result.png")
    print("\n" + "="*60)
    print("Optimization test complete!")
    print("="*60)


if __name__ == "__main__":
    test_optimizer()