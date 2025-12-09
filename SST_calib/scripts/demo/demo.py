"""
Step 9 (Fixed): Interactive Online Calibration Demo

CLEAR LOGIC:
- Ground truth (R_gt, t_gt, δ_gt=0) is the CORRECT calibration
- Current estimate (R_est, t_est, δ_est) is what our algorithm thinks
- Perturbation adds NOISE to our estimate (simulating drift)
- Calibration tries to recover back to ground truth

No simulation of "true unknown" values - just GT vs our estimate.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys

sys.path.append('src')


class OnlineCalibrationDemo:
    """
    Interactive demo for SST-Calib.
    
    Simple logic:
    - GT calibration is known (from KITTI)
    - We maintain an ESTIMATE that can drift
    - Perturbation = add noise to estimate
    - Calibration = recover estimate back to GT
    """
    
    def __init__(self, data_loader, image_segmentor):
        self.loader = data_loader
        self.img_segmentor = image_segmentor
        
        self.K = data_loader.get_intrinsic_matrix()
        self.img_size = data_loader.get_image_size()
        
        # Ground truth calibration (FIXED, from KITTI)
        self.R_gt, self.t_gt = data_loader.get_ground_truth_extrinsics()
        self.delta_gt = 0.0  # KITTI is synchronized, no delay
        
        # Our current ESTIMATE (starts at ground truth)
        self.R_est = self.R_gt.copy()
        self.t_est = self.t_gt.copy()
        self.delta_est = 0.0
        
        # State
        self.frame_idx = 0
        self.error_history = []
        self.delta_error_history = []
        self.step_count = 0
        
        # Simulated time delay for the scenario
        # This is what we ADD to GT to simulate a sensor with delay
        self.simulated_delay = 0.0  # Will be set when we "perturb"
        
        # Pre-load frames
        print("Pre-loading frames...")
        self._preload_frames()
    
    def _preload_frames(self):
        """Pre-load frames with car segmentation."""
        from core.losses import segment_car_points_salsanext
        from temporal.temporal_calibration import VisualOdometry
        
        self.frames = []
        self.velocities = []
        
        vo = VisualOdometry(self.K)
        prev_image = None
        
        n_frames = min(80, len(self.loader))
        
        for i in range(n_frames):
            image, point_cloud = self.loader.load_frame_pair(i)
            _, binary_mask = self.img_segmentor.segment(image)
            
            # Segment car points using SalsaNext (NO GROUND TRUTH NEEDED)
            car_points_3d = segment_car_points_salsanext(point_cloud)
            
            self.frames.append({
                'image': image,
                'binary_mask': binary_mask,
                'car_points_3d': car_points_3d,
                'n_car_points': len(car_points_3d)
            })
            
            # Estimate velocity
            if prev_image is not None:
                vel, success = vo.estimate_velocity(prev_image, image, dt=0.1)
                self.velocities.append(vel if success else np.array([5.0, 0.0, 0.0]))
            else:
                self.velocities.append(np.array([5.0, 0.0, 0.0]))
            
            prev_image = image
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i+1}/{n_frames} frames ({len(car_points_3d)} car points)")
        
        self.avg_velocity = np.mean(self.velocities, axis=0)
        print(f"\nLoaded {len(self.frames)} frames")
        print(f"Average velocity: {self.avg_velocity}")
    
    def get_velocity(self, frame_idx):
        """Get velocity at frame."""
        if frame_idx < len(self.velocities):
            return self.velocities[frame_idx]
        return self.avg_velocity
    
    def compute_error(self):
        """Compute error between our estimate and ground truth."""
        import sys
        import os
        # Ensure we import from current directory, not parent
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from core.optimizer import compute_calibration_error
        
        err = compute_calibration_error(self.R_est, self.t_est, self.R_gt, self.t_gt)
        
        # Temporal error: difference between our estimate and simulated delay
        delta_err_ms = abs(self.delta_est + self.simulated_delay) * 1000
        
        return err['atd_cm'], err['qad_deg'], delta_err_ms
    
    def project_car_points(self, car_points_3d, R, t, delta, velocity):
        """Project car points with temporal compensation."""
        if len(car_points_3d) == 0:
            return np.array([]).reshape(0, 2)
        
        # Temporal compensation
        t_comp = t + (velocity * delta).reshape(3, 1)
        
        # Transform
        pts_cam = (R @ car_points_3d.T + t_comp).T
        
        # Filter behind camera
        valid = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[valid]
        
        if len(pts_cam) == 0:
            return np.array([]).reshape(0, 2)
        
        # Project
        proj = (self.K @ pts_cam.T).T
        pts_2d = proj[:, :2] / proj[:, 2:3]
        
        # Filter bounds
        w, h = self.img_size
        in_bounds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
                    (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        
        return pts_2d[in_bounds]
    
    def perturb_estimate(self, t_noise_cm=5.0, r_noise_deg=3.0, delta_noise_ms=30.0):
        """
        Add noise to our estimate (simulating calibration drift).
        """
        print("\n" + "="*50)
        print("⚠ PERTURBATION: Adding noise to calibration estimate")
        print("="*50)
        
        # Add noise to translation estimate
        t_noise = np.random.uniform(-t_noise_cm/100, t_noise_cm/100, (3, 1))
        self.t_est = self.t_est + t_noise
        
        # Add noise to rotation estimate
        euler_noise = np.random.uniform(-r_noise_deg, r_noise_deg, 3)
        R_noise = Rotation.from_euler('xyz', euler_noise, degrees=True).as_matrix()
        self.R_est = self.R_est @ R_noise
        
        # Simulate a time delay (GT has no delay, but we pretend sensor has one)
        self.simulated_delay = np.random.uniform(-delta_noise_ms/1000, delta_noise_ms/1000)
        # Our estimate is still 0, so we have error
        
        print(f"\nNoise added to estimate:")
        print(f"  Translation: [{t_noise[0,0]*100:.1f}, {t_noise[1,0]*100:.1f}, {t_noise[2,0]*100:.1f}] cm")
        print(f"  Rotation: [{euler_noise[0]:.1f}, {euler_noise[1]:.1f}, {euler_noise[2]:.1f}]°")
        print(f"  Simulated delay: {self.simulated_delay*1000:.1f}ms (our estimate: {self.delta_est*1000:.1f}ms)")
        
        atd, qad, delta_err = self.compute_error()
        print(f"\nNew errors:")
        print(f"  Spatial: ATD={atd:.2f}cm, QAD={qad:.2f}°")
        print(f"  Temporal: {delta_err:.1f}ms")
    
    def run_spatial_calibration(self, n_frames=5):
        """Run spatial-only calibration (R, t)."""
        print("\n🔧 Running SPATIAL-ONLY calibration...")
        
        # Check if already good
        atd, qad, _ = self.compute_error()
        if atd < 1.0 and qad < 0.5:
            print("  Already well calibrated, skipping.")
            return
        
        from core.losses import SemanticAlignmentLoss, CalibrationLoss
        from core.optimizer import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
        
        loss_calc = SemanticAlignmentLoss(pixel_downsample_rate=0.1, seed=42)
        
        # Collect frames
        start = max(0, self.frame_idx - n_frames // 2)
        end = min(len(self.frames), start + n_frames)
        
        calib_losses = []
        for i in range(start, end):
            frame = self.frames[i]
            if frame['n_car_points'] >= 30:
                cl = CalibrationLoss(
                    self.K, self.img_size,
                    frame['binary_mask'],
                    frame['car_points_3d'],
                    loss_calc
                )
                calib_losses.append(cl)
        
        if len(calib_losses) < 2:
            print("  ❌ Not enough frames with car points")
            return
        
        print(f"  Using {len(calib_losses)} frames")
        
        # Optimize
        # rotvec_init = rotation_matrix_to_axis_angle(self.R_est)
        # params_init = np.concatenate([rotvec_init, self.t_est.flatten()])
        
        # def objective(params):
        #     R = axis_angle_to_rotation_matrix(params[:3])
        #     t = params[3:6].reshape(3, 1)
            
        #     total = 0
        #     for cl in calib_losses:
        #         loss = cl.compute(R, t, weight_i2p=1.0)
        #         if not np.isinf(loss):
        #             total += loss
        #     return total / len(calib_losses)
        
        # # Bounds around current estimate
        # rot_bound = 20 * np.pi / 180
        # trans_bound = 0.3
        
        # bounds = [(p - rot_bound, p + rot_bound) for p in rotvec_init]
        # bounds += [(p - trans_bound, p + trans_bound) for p in self.t_est.flatten()]
        
        # result = minimize(objective, params_init, method='L-BFGS-B',
        #                  bounds=bounds, options={'maxiter': 100})
        
        # self.R_est = axis_angle_to_rotation_matrix(result.x[:3])
        # self.t_est = result.x[3:6].reshape(3, 1)
        from core.optimizer import MultiFrameOptimizer
        
        # Initialize Optimizer with CURRENT estimate
        # This uses the Weight Schedule defined in optimizer.py
        opt = MultiFrameOptimizer(calib_losses, self.R_est, self.t_est)
        
        # Run optimization (No regularization for static spatial per paper, 
        # or mild regularization to prevent explosion if data is bad)
        R_new, t_new, res = opt.optimize(verbose=True, regularize=True)
        
        self.R_est = R_new
        self.t_est = t_new
        
        atd, qad, delta_err = self.compute_error()
        print(f"  ✓ Done! ATD={atd:.2f}cm, QAD={qad:.2f}°")
        print(f"    (Temporal error unchanged: {delta_err:.1f}ms)")
    
    def run_joint_calibration(self, n_frames=5):
        """Run joint spatial-temporal calibration (R, t, δ)."""
        print("\n🔧 Running JOINT SPATIAL-TEMPORAL calibration...")
        
        # Check if already good
        atd, qad, delta_err = self.compute_error()
        if atd < 1.0 and qad < 0.5 and delta_err < 5:
            print("  Already well calibrated, skipping.")
            return
        
        from core.losses import SemanticAlignmentLoss, CalibrationLoss
        from core.optimizer import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
        
        loss_calc = SemanticAlignmentLoss(pixel_downsample_rate=0.1, seed=42)
        
        # Collect frames with velocities
        start = max(0, self.frame_idx - n_frames // 2)
        end = min(len(self.frames), start + n_frames)
        
        frames_data = []
        for i in range(start, end):
            frame = self.frames[i]
            if frame['n_car_points'] >= 30:
                cl = CalibrationLoss(
                    self.K, self.img_size,
                    frame['binary_mask'],
                    frame['car_points_3d'],
                    loss_calc
                )
                frames_data.append({
                    'calib_loss': cl,
                    'velocity': self.get_velocity(i)
                })
        
        if len(frames_data) < 2:
            print("  ❌ Not enough frames with car points")
            return
        
        print(f"  Using {len(frames_data)} frames")
        
        # --- WEIGHT SCHEDULE (The Fix) ---
        # 1. High weight (20.0): Pulls points roughly into the mask (Coarse alignment)
        # 2. Med weight (1.0): Refines the fit
        # 3. Low weight (0.02): Final polish
        schedule = [
            (20.0, 30),  # Weight 20.0, 30 iterations
            (1.0, 50),   # Weight 1.0, 50 iterations
            (0.02, 20)   # Weight 0.02, 20 iterations
        ]
        
        # Initialization
        delta_est = self.delta_est # Start with current delta guess
        rotvec_est = rotation_matrix_to_axis_angle(self.R_est)
        t_est = self.t_est.flatten()
        
        # Current params: [rx, ry, rz, tx, ty, tz, delta]
        params = np.concatenate([rotvec_est, t_est, [delta_est]])

        # Bounds (Prevent vanishing points without pinning to bad state)
        # We allow ±20 degrees rotation, ±50cm translation, ±150ms delay
        rot_bound = 20 * np.pi / 180
        trans_bound = 0.5
        
        bounds = [(p - rot_bound, p + rot_bound) for p in rotvec_est]
        bounds += [(p - trans_bound, p + trans_bound) for p in t_est]
        bounds += [(-0.15, 0.15)]  # Delta bounds
        
        # Optimization Loop
        for stage, (w_i2p, max_iter) in enumerate(schedule):
            print(f"  Stage {stage+1}/3: weight={w_i2p}, iter={max_iter}")
            
            # Regularization strength (decreases across stages)
            lambda_reg = [0.1, 100.0, 10000.0][stage]
            
            def joint_objective(p):
                R = axis_angle_to_rotation_matrix(p[:3])
                t = p[3:6].reshape(3, 1)
                delta = p[6]
                
                total = 0
                count = 0
                for fd in frames_data:
                    # Apply time delay compensation
                    t_comp = t + (fd['velocity'] * delta).reshape(3, 1)
                    
                    # Compute loss
                    loss = fd['calib_loss'].compute(R, t_comp, weight_i2p=w_i2p)
                    if not np.isinf(loss):
                        total += loss
                        count += 1
                
                if count == 0: return 1e9
                
                # Regularization: penalize deviation from initial
                rot_dev = np.sum((p[:3] - rotvec_est)**2)
                trans_dev = np.sum((p[3:6] - t_est)**2)
                delta_dev = (p[6] - delta_est)**2
                reg_term = lambda_reg * (rot_dev + trans_dev + delta_dev)
                
                return total / count + reg_term

            # Run optimizer for this stage
            res = minimize(joint_objective, params, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-4})
            
            # Update params for next stage
            params = res.x
            print(f"    Loss: {res.fun:.4f}")

        # Update final state
        self.R_est = axis_angle_to_rotation_matrix(params[:3])
        self.t_est = params[3:6].reshape(3, 1)
        self.delta_est = params[6]
        
        atd, qad, delta_err = self.compute_error()
        print(f"  ✓ Done!")
        print(f"    Spatial: ATD={atd:.2f}cm, QAD={qad:.2f}°")
        print(f"    Temporal: δ_est={self.delta_est*1000:.1f}ms, error={delta_err:.1f}ms")
    
    def reset(self):
        """Reset estimate to ground truth."""
        self.R_est = self.R_gt.copy()
        self.t_est = self.t_gt.copy()
        self.delta_est = 0.0
        self.simulated_delay = 0.0
        print("\n↺ Reset to ground truth")
    
    def run_interactive(self):
        """Run interactive demo."""
        print("\n" + "="*70)
        print("  SST-CALIB INTERACTIVE DEMO")
        print("="*70)
        print("\nControls:")
        print("  [SPACE] - Next frame")
        print("  [P] - Add Error (add noise to estimate)")
        print("  [S] - Spatial-only calibration")
        print("  [J] - Joint spatial-temporal calibration")
        print("  [R] - Reset to ground truth")
        print("  [Q] - Quit")
        print("\nViews:")
        print("  LEFT: Ground truth projection (target)")
        print("  MIDDLE: Current estimate (without δ compensation)")
        print("  RIGHT: Current estimate (with δ compensation)")
        print("="*70)
        
        fig = plt.figure(figsize=(18, 9))
        
        # Image axes
        ax_gt = fig.add_axes([0.01, 0.40, 0.32, 0.55])
        ax_no_delta = fig.add_axes([0.34, 0.40, 0.32, 0.55])
        ax_with_delta = fig.add_axes([0.67, 0.40, 0.32, 0.55])
        
        # Error plot
        ax_error = fig.add_axes([0.1, 0.08, 0.8, 0.25])
        
        # Buttons
        btn_axes = [
            fig.add_axes([0.05, 0.01, 0.12, 0.05]),
            fig.add_axes([0.19, 0.01, 0.12, 0.05]),
            fig.add_axes([0.33, 0.01, 0.15, 0.05]),
            fig.add_axes([0.50, 0.01, 0.18, 0.05]),
            fig.add_axes([0.70, 0.01, 0.12, 0.05]),
        ]
        
        buttons = [
            Button(btn_axes[0], 'Next [SPACE]'),
            Button(btn_axes[1], 'Add error [P]', color='lightsalmon'),
            Button(btn_axes[2], 'Spatial [S]', color='lightyellow'),
            Button(btn_axes[3], 'Joint SST [J]', color='lightgreen'),
            Button(btn_axes[4], 'Reset [R]'),
        ]
        
        def update():
            frame = self.frames[self.frame_idx]
            image = frame['image']
            mask = frame['binary_mask']
            car_pts = frame['car_points_3d']
            vel = self.get_velocity(self.frame_idx)
            
            # Mask overlay
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = [1, 1, 0, 0.4]
            
            atd, qad, delta_err = self.compute_error()
            
            # Update history
            self.step_count += 1
            self.error_history.append(atd)
            self.delta_error_history.append(delta_err)
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
                self.delta_error_history = self.delta_error_history[-100:]
            
            # === Ground Truth ===
            ax_gt.clear()
            ax_gt.imshow(image)
            ax_gt.imshow(overlay)
            pts = self.project_car_points(car_pts, self.R_gt, self.t_gt, 0, vel)
            if len(pts) > 0:
                ax_gt.scatter(pts[:, 0], pts[:, 1], c='lime', s=8, alpha=0.9)
            ax_gt.set_title("GROUND TRUTH\n(Target alignment)", color='green', fontsize=11)
            ax_gt.axis('off')
            
            # === Current estimate WITHOUT delta ===
            ax_no_delta.clear()
            ax_no_delta.imshow(image)
            ax_no_delta.imshow(overlay)
            pts = self.project_car_points(car_pts, self.R_est, self.t_est, 0, vel)
            if len(pts) > 0:
                ax_no_delta.scatter(pts[:, 0], pts[:, 1], c='red', s=8, alpha=0.9)
            color = 'green' if atd < 3 else ('orange' if atd < 8 else 'red')
            ax_no_delta.set_title(f"ESTIMATE\nATD={atd:.1f}cm, QAD={qad:.2f}°", 
                                 color=color, fontsize=11)
            ax_no_delta.axis('off')
            
            # === Current estimate WITH delta ===
            ax_with_delta.clear()
            ax_with_delta.imshow(image)
            ax_with_delta.imshow(overlay)
            pts = self.project_car_points(car_pts, self.R_est, self.t_est, self.delta_est, vel)
            if len(pts) > 0:
                ax_with_delta.scatter(pts[:, 0], pts[:, 1], c='cyan', s=8, alpha=0.9)
            color = 'green' if (atd < 3 and delta_err < 10) else ('orange' if atd < 8 else 'red')
            ax_with_delta.set_title(f"ESTIMATE (δ={self.delta_est*1000:.0f}ms)\n"
                                   f"Temporal err: {delta_err:.0f}ms",
                                   color=color, fontsize=11)
            ax_with_delta.axis('off')
            
            # === Error plot ===
            ax_error.clear()
            x = list(range(len(self.error_history)))
            ax_error.plot(x, self.error_history, 'b-', linewidth=2, label='Spatial (ATD cm)')
            ax_error.plot(x, self.delta_error_history, 'r--', linewidth=2, label='Temporal (ms)')
            ax_error.axhline(y=3, color='green', linestyle=':', alpha=0.5)
            ax_error.axhline(y=10, color='orange', linestyle=':', alpha=0.5)
            ax_error.set_xlabel('Step')
            ax_error.set_ylabel('Error')
            ax_error.set_title(f'Frame {self.frame_idx} | Calibration Error History | '
                              f'Simulated delay: {self.simulated_delay*1000:.0f}ms', fontsize=10)
            ax_error.legend(loc='upper right')
            ax_error.set_ylim(0, max(20, max(self.error_history + self.delta_error_history) * 1.2))
            ax_error.grid(True, alpha=0.3)
            
            fig.canvas.draw_idle()
        
        def on_next(event):
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            # Skip frames with few car points
            attempts = 0
            while self.frames[self.frame_idx]['n_car_points'] < 30 and attempts < len(self.frames):
                self.frame_idx = (self.frame_idx + 1) % len(self.frames)
                attempts += 1
            update()
        
        def on_perturb(event):
            self.perturb_estimate()
            update()
        
        def on_spatial(event):
            self.run_spatial_calibration()
            update()
        
        def on_joint(event):
            self.run_joint_calibration()
            update()
        
        def on_reset(event):
            self.reset()
            update()
        
        def on_key(event):
            if event.key == ' ':
                on_next(None)
            elif event.key == 'p':
                on_perturb(None)
            elif event.key == 's':
                on_spatial(None)
            elif event.key == 'j':
                on_joint(None)
            elif event.key == 'r':
                on_reset(None)
            elif event.key == 'q':
                plt.close(fig)
        
        buttons[0].on_clicked(on_next)
        buttons[1].on_clicked(on_perturb)
        buttons[2].on_clicked(on_spatial)
        buttons[3].on_clicked(on_joint)
        buttons[4].on_clicked(on_reset)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Find first frame with enough car points
        for i, f in enumerate(self.frames):
            if f['n_car_points'] >= 50:
                self.frame_idx = i
                break
        
        update()
        plt.show()
    
    def run_automated_demo(self):
        """Run automated comparison demo."""
        print("\n" + "="*70)
        print("  AUTOMATED DEMO: Calibration Comparison")
        print("="*70)
        
        # Find good frame
        for i, f in enumerate(self.frames):
            if f['n_car_points'] >= 100:
                self.frame_idx = i
                break
        
        frame = self.frames[self.frame_idx]
        vel = self.get_velocity(self.frame_idx)
        
        print(f"\nUsing frame {self.frame_idx} ({frame['n_car_points']} car points)")
        
        results = []
        
        # 1. Ground truth
        self.reset()
        atd, qad, delta_err = self.compute_error()
        results.append(('1. Ground Truth', atd, qad, delta_err, 
                       self.R_gt.copy(), self.t_gt.copy(), 0.0))
        print(f"\n1. Ground Truth: ATD={atd:.2f}cm, QAD={qad:.2f}°")
        
        # 2. After perturbation
        np.random.seed(42)
        self.perturb_estimate(t_noise_cm=5, r_noise_deg=3, delta_noise_ms=100)
        atd, qad, delta_err = self.compute_error()
        results.append((f'2. Error added (δ_sim={self.simulated_delay*1000:.0f}ms)', 
                       atd, qad, delta_err,
                       self.R_est.copy(), self.t_est.copy(), 0.0))
        print(f"2. Error added: ATD={atd:.2f}cm, QAD={qad:.2f}°, δ_err={delta_err:.0f}ms")
        
        # Save perturbed state
        R_perturbed = self.R_est.copy()
        t_perturbed = self.t_est.copy()
        sim_delay = self.simulated_delay
        
        # 3. Spatial-only calibration
        self.run_spatial_calibration()
        atd, qad, delta_err = self.compute_error()
        results.append(('3. Spatial-Only', atd, qad, delta_err,
                       self.R_est.copy(), self.t_est.copy(), 0.0))
        print(f"3. Spatial-Only: ATD={atd:.2f}cm, QAD={qad:.2f}°, δ_err={delta_err:.0f}ms")
        
        # 4. Joint calibration (from perturbed state)
        self.R_est = R_perturbed
        self.t_est = t_perturbed
        self.delta_est = 0.0
        self.simulated_delay = sim_delay
        
        self.run_joint_calibration()
        atd, qad, delta_err = self.compute_error()
        results.append((f'4. Joint SST (δ_est={self.delta_est*1000:.0f}ms)', 
                       atd, qad, delta_err,
                       self.R_est.copy(), self.t_est.copy(), self.delta_est))
        print(f"4. Joint SST: ATD={atd:.2f}cm, QAD={qad:.2f}°, δ_err={delta_err:.0f}ms")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        image = frame['image']
        mask = frame['binary_mask']
        car_pts = frame['car_points_3d']
        
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [1, 1, 0, 0.4]
        
        colors = ['green', 'red', 'orange', 'green']
        
        for ax, (title, atd, qad, delta_err, R, t, delta), color in zip(axes, results, colors):
            ax.imshow(image)
            ax.imshow(overlay)
            
            pts = self.project_car_points(car_pts, R, t, delta, vel)
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c='red', s=8, alpha=0.9)
            
            ax.set_title(f"{title}\nATD={atd:.1f}cm, QAD={qad:.2f}°, δ_err={delta_err:.0f}ms",
                        fontsize=10, color=color)
            ax.axis('off')
        
        plt.suptitle("SST-Calib: Spatial-Only vs Joint Calibration\n"
                    "(Yellow=mask, Red=LIDAR car points)", fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/sst_calib_demo.png', dpi=150)
        plt.show()
        
        # Summary
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)
        print(f"\n{'Method':<40} {'ATD(cm)':<10} {'QAD(°)':<10} {'δ_err(ms)':<10}")
        print("-"*70)
        for title, atd, qad, delta_err, _, _, _ in results:
            print(f"{title:<40} {atd:<10.2f} {qad:<10.2f} {delta_err:<10.1f}")
        print("-"*70)
        print(f"\nSimulated time delay: {sim_delay*1000:.1f}ms")
        print(f"Joint SST estimated: {self.delta_est*1000:.1f}ms")
        print("\n✓ Saved to outputs/sst_calib_demo.png")


def main():
    import sys
    import os
    
    # Add parent directory (scripts/) to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from utils.data_loader import KITTIDataLoader
    from segmentation.image_segmentation import ImageSegmentor
    
    # =====================================================
    BASE_PATH = os.path.join(script_dir, "../../dataset")  # Points to SST_calib/dataset
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    print("Loading...")
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    segmentor = ImageSegmentor(target_classes=['car', 'bus'])
    
    demo = OnlineCalibrationDemo(loader, segmentor)
    
    print("\nModes:")
    print("  1. Interactive")
    print("  2. Automated comparison")
    
    mode = input("\nSelect (1/2): ").strip()
    
    if mode == '1':
        demo.run_interactive()
    else:
        demo.run_automated_demo()


if __name__ == "__main__":
    main()