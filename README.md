# Camera-Only Bird's Eye View Perception

![BEV Perception](https://github.com/anupkumarbochare/camera-only-bev/raw/main/assets/bev_demo.gif)

## Overview

This repository implements a camera-only perception system that generates Bird's Eye View (BEV) maps for autonomous vehicles **without using LiDAR sensors**. The system builds upon and extends the Lift-Splat-Shoot neural architecture, integrating YOLOv11 object detection with DepthAnythingV2 monocular depth estimation across multiple camera perspectives to achieve 360-degree environmental awareness.

Our approach achieves 85% road segmentation accuracy and 85-90% vehicle detection rates compared to LiDAR ground truth, with average positional errors limited to 1.2 meters.

## Key Features

- Complete camera-only BEV generation pipeline that eliminates the need for expensive LiDAR sensors
- Novel integration of YOLOv11 object detection with DepthAnythingV2 for accurate object placement in BEV space
- Custom multi-component loss function (BEVLoss) that evaluates position accuracy, existence detection, and class identification
- Support for various camera configurations (6 or 7 cameras)
- Real-time performance (13 FPS on standard hardware)

## Architecture

The system consists of four main components:
1. **Multi-camera input processing**: Processes images from 6-7 surround-view cameras
2. **Feature extraction with depth estimation**: Extracts visual features using ResNet-50 and estimates depth
3. **3D projection and feature aggregation**: Projects features to 3D space using quaternion-based transformations
4. **BEV semantic map generation**: Generates road segmentation and object detection outputs

![System Architecture](https://github.com/anupkumarbochare/camera-only-bev/raw/main/assets/architecture.png)

## Requirements

```
python >= 3.8
pytorch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.20.0
opencv-python >= 4.5.3
matplotlib >= 3.4.3
```

## Installation

```bash
# Clone the repository
git clone https://github.com/anupkumarbochare/camera-only-bev.git
cd camera-only-bev

# Create a conda environment
conda create -n bev python=3.8
conda activate bev

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py
```

## Dataset Preparation

The system has been tested on the OpenLane-V2 and NuScenes datasets. Follow these steps to prepare the datasets:

### OpenLane-V2

```bash
# Download the dataset
python scripts/download_openlane.py

# Preprocess the dataset
python scripts/preprocess_openlane.py --data_dir /path/to/openlane
```

### NuScenes

```bash
# Download the dataset
python scripts/download_nuscenes.py

# Preprocess the dataset
python scripts/preprocess_nuscenes.py --data_dir /path/to/nuscenes
```

## Training

```bash
# Train on OpenLane-V2
python train.py --config configs/openlane_config.yaml --data_dir /path/to/openlane

# Train on NuScenes
python train.py --config configs/nuscenes_config.yaml --data_dir /path/to/nuscenes

# Resume training from a checkpoint
python train.py --config configs/openlane_config.yaml --resume checkpoints/model_latest.pth
```

## Evaluation

```bash
# Evaluate on OpenLane-V2
python evaluate.py --config configs/openlane_config.yaml --checkpoint checkpoints/model_best.pth --data_dir /path/to/openlane

# Evaluate on NuScenes
python evaluate.py --config configs/nuscenes_config.yaml --checkpoint checkpoints/model_best.pth --data_dir /path/to/nuscenes

# Generate visualization
python evaluate.py --config configs/openlane_config.yaml --checkpoint checkpoints/model_best.pth --visualize
```

## Inference

```bash
# Run inference on a sequence
python inference.py --config configs/openlane_config.yaml --checkpoint checkpoints/model_best.pth --sequence_dir /path/to/sequence

# Run inference on a live camera feed
python live_demo.py --config configs/demo_config.yaml --checkpoint checkpoints/model_best.pth
```

## Results

Our camera-only approach achieves 85.1% segmentation IoU and 82.6% detection AP@0.5, significantly outperforming other camera-only methods while approaching the performance of LiDAR-based systems.

| Method | Seg. IoU (%) | Det. AP@0.5 (%) | Det. AP@0.75 (%) | Pos. Error (m) | Runtime (ms) |
|--------|--------------|-----------------|------------------|---------------|--------------|
| LiDAR-Based | 92.3 | 89.7 | 76.5 | 0.31 | 95 |
| IPM-Based | 63.8 | 42.3 | 21.7 | 2.84 | 25 |
| LSS (Original) | 73.6 | 65.2 | 43.1 | 1.76 | 67 |
| BEVFormer | 81.2 | 72.8 | 51.4 | 1.35 | 120 |
| Tesla-Like | 82.5 | 78.3 | 53.9 | 1.28 | 85 |
| Ours | 85.1 | 82.6 | 56.8 | 1.15 | 78 |

## Visualization

![BEV Visualization](https://github.com/anupkumarbochare/camera-only-bev/raw/main/assets/visualization.png)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{bochare2025camera,
  title={Camera-Only Bird's Eye View Perception: A Neural Approach to LiDAR-Free Environmental Mapping for Autonomous Vehicles},
  author={Bochare, Anupkumar},
  journal={arXiv preprint arXiv:2505.12345},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research was supported by Northeastern University
- We thank the developers of the OpenLane-V2 and NuScenes datasets
- The Lift-Splat-Shoot implementation is based on the work by Philion and Fidler
- DepthAnythingV2 implementation by Yang et al.
- YOLOv11 implementation by Jocher et al.
