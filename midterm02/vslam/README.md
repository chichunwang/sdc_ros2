# 2026 Fall SDC Midterm Competition II - Visual SLAM

Estimate the camera pose of every image from the front camera only. No LiDAR data is given;
LiDAR is used only to build the hidden ground-truth trajectory.

## Code Structure
```
.
├── main.py              # SLAM pipeline: scan-to-submap odometry -> keyframes -> loop closure -> pose graph -> submission.csv
├── process
│   └── build_map.py     # stitch the image point clouds with your poses to check the trajectory visually
└── utils
    ├── ICP.py           # TODO 1: scan-matching ALGO
    ├── loopclosure.py   # TODO 2: loop-closure detection and tf estimation
    ├── io.py            # loaders: point clouds, camera info, IMU
    ├── keyframe.py
    ├── posegraph.py
    ├── submap.py
    └── viz.py
```

## Dataset Format
```
Track1
├── data
│   ├── raw_ply/               # point cloud (x y z + RGB) of each front-camera image (LoGeR), camera frame, named by timestamp
│   ├── images/                # front-camera images
│   ├── camera_info.json       # camera intrinsics (+ IMU-to-camera rotation on Track1)
│   ├── imu.csv                # IMU, Track1 only
│   └── sample_submission.csv  # one row per image: timestamp + 4x4 camera-to-world pose (row-major)
└── result
    ├── submission.csv
    └── map.ply
```
Camera frame = OpenCV axes (x right, y down, z forward). The point clouds are only roughly in metres:
their scale is not exact and can differ slightly between images.

`R_cam_imu` (in `camera_info.json`) rotates IMU-frame vectors into the camera frame. Only the rotation is given
(no translation, no IMU noise parameters). The IMU timestamps are the bag receive times.

## Score
Your camera positions are aligned to the ground truth with a similarity transform (rotation, translation and
scale; Umeyama), then the RMSE of the position errors is computed (ATE, in metres, lower is better).
The scale and the starting pose of your trajectory therefore do not matter. Every image needs a pose.

## Environment Setup
```
conda create -n slam_env python=3.9
conda activate slam_env

conda install numpy pandas scipy tqdm

pip install opencv-python
pip install open3d

# Factor graph optimization
pip install gtsam
pip install small_gicp
```

## Run
1. Finish the TODOs in `utils/ICP.py` and `utils/loopclosure.py`.
2. Run `main.py` to estimate the camera pose of every image -> `Track1/result/submission.csv`.
3. (Optional) Run `python process/build_map.py Track1` to look at the map built from your poses.
4. Submit `submission.csv` to Kaggle.

The IMU (Track1) and the raw images are provided but not used by the starter; how to use them is up to you.

`utils/viz.py` shows the SLAM process live with Open3D (as last year). You may replace it with any viewer you like, e.g. [rerun](https://rerun.io).

Full SLAM / SfM systems (e.g. MASt3R-SLAM, ORB-SLAM3, VGGT-SLAM, COLMAP) must not be used to produce the submission.
