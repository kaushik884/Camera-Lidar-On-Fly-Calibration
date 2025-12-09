# SST-Calib: Spatial-Temporal LiDAR-Camera Calibration

> **Note:** This is a course project implementation based on the SST-Calib paper. The implementation may not achieve the same accuracy as reported in the original paper and is intended for educational purposes.

Autonomous online calibration of LiDAR-camera extrinsics with temporal synchronization using semantic segmentation.

## Overview

SST-Calib performs joint spatial-temporal calibration between LiDAR and camera sensors without requiring:
- Calibration targets or patterns
- Manual initialization
- Ground truth labels for point detection

**Key Features:**
- **SalsaNext-based LiDAR segmentation** - Detects cars in 3D point clouds
- **DeepLabV3 image segmentation** - Detects cars in camera images
- **Joint optimization** - Estimates rotation, translation, and time delay simultaneously
- **Interactive demo** - Real-time visualization of calibration process

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AFR-Final-Project.git
cd AFR-Final-Project
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Setup

> **Important:** The dataset and pretrained models are **NOT included** in this repository due to size. You must download them separately.

### KITTI Dataset
Download the KITTI raw dataset from [http://www.cvlibs.net/datasets/kitti/raw_data.php](http://www.cvlibs.net/datasets/kitti/raw_data.php)

**Required files:**
- `2011_09_26_drive_0005_sync.zip` (example sequence)
- `2011_09_26_calib.zip` (calibration files)

### SalsaNext Pretrained Model
Download from: [SalsaNext GitHub](https://github.com/TiagoCortinhal/SalsaNext)
- Model weights: `SalsaNext` file
- Configuration: `arch_cfg.yaml`, `data_cfg.yaml`

### Directory Setup
Create the following structure in `SST_calib/dataset/`:

```
AFR-Final-Project/
├── SST_calib/
│   ├── dataset/
│   │   ├── 2011_09_26/
│   │   │   ├── 2011_09_26_drive_0005_sync/
│   │   │   │   ├── image_02/
│   │   │   │   │   └── data/
│   │   │   │   │       ├── 0000000000.png
│   │   │   │   │       ├── 0000000001.png
│   │   │   │   │       └── ...
│   │   │   │   └── velodyne_points/
│   │   │   │       └── data/
│   │   │   │           ├── 0000000000.bin
│   │   │   │           ├── 0000000001.bin
│   │   │   │           └── ...
│   │   │   └── calib_cam_to_cam.txt
│   │   │   └── calib_velo_to_cam.txt
│   │   └── models/
│   │       └── pretrained/
│   │           ├── SalsaNext  (model weights)
│   │           ├── arch_cfg.yaml
│   │           └── data_cfg.yaml
│   └── scripts/
│       ├── calibration.py
│       ├── data_loader.py
│       ├── demo.py
│       └── ...
└── venv/
```

## Usage

### Interactive Demo

Run the calibration demo with visualization:

```bash
cd SST_calib/scripts
../../venv/bin/python ui/demo.py
```

**Controls:**
- `SPACE` - Next frame
- `P` - Add perturbation (simulate calibration error)
- `S` - Run spatial-only calibration
- `J` - Run joint spatial-temporal calibration
- `R` - Reset to ground truth
- `Q` - Quit

**Demo Modes:**
1. **Interactive** - Manual control with keyboard
2. **Automated** - Runs full calibration comparison automatically (Needs Fix)

### Using Individual Modules

```python
from data_loader import KITTIDataLoader
from image_segmentation import ImageSegmentor
from lidar_segmentation import LidarSegmenter

# Load data
loader = KITTIDataLoader('../dataset', '2011_09_26', '0005')
image, point_cloud = loader.load_frame_pair(0)

# Segment image
img_seg = ImageSegmentor(target_classes=['car', 'bus'])
_, car_mask = img_seg.segment(image)

# Segment point cloud
lidar_seg = LidarSegmenter()
car_indices, car_points = lidar_seg.get_car_points(point_cloud)

print(f"Detected {len(car_points)} car points in LiDAR")
print(f"Detected {car_mask.sum()} car pixels in image")
```

## Project Structure

**Modular organization:**

```
SST_calib/scripts/
├── core/                   # Core calibration components
│   ├── calibration.py      # Calibration utilities
│   ├── losses.py           # Loss functions
│   └── optimizer.py        # Optimization routines
├── segmentation/           # Semantic segmentation
│   ├── image_segmentation.py   # DeepLabV3 wrapper
│   └── lidar_segmentation.py   # SalsaNext wrapper
├── models/                 # Neural network models
│   └── salsanext_model.py  # SalsaNext architecture
├── temporal/               # Temporal calibration
│   ├── sst_calib.py        # Main SST-Calib algorithm
│   └── temporal_calibration.py # Time delay estimation
├── ui/                     # User interfaces
│   └── demo.py             # Interactive demo
└── utils/                  # Utilities
    └── data_loader.py      # KITTI dataset loader
```

## Algorithm Overview

1. **Semantic Segmentation**
   - Image: DeepLabV3 detects car pixels
   - LiDAR: SalsaNext detects car points

2. **Correspondence**
   - Bi-directional loss (point-to-image + image-to-point)
   - FOV filtering to match camera view
   - DBSCAN clustering to remove outliers

3. **Optimization**
   - Multi-stage coarse-to-fine optimization
   - Regularization to prevent divergence
   - Joint spatial-temporal estimation

## Citation

Based on the original paper:

```bibtex
@inproceedings{kodaira2022sst,
  title={SST-Calib: Simultaneous spatial-temporal parameter calibration between LiDAR and camera},
  author={Kodaira, Akio and Zhou, Yiyang and Zang, Pengwei and Zhan, Wei and Tomizuka, Masayoshi},
  booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={2896--2902},
  year={2022},
  organization={IEEE}
}
```

**Disclaimer:** This is an independent course project implementation. Results may differ from the original paper. This implementation is for educational purposes and is not affiliated with the original authors.

## Team Members

- Kaushik Deo
- Samara Holmes  
- Nishad Milind Rajhans
- Hemanth Raj Tekumalla
