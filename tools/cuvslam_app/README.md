# cuVSLAM Application

A Python application for visual odometry using cuVSLAM library.

## Requirements

Before running the application, set up the environment:
```bash
./create_env.sh
```

## Usage Examples

### KITTI Dataset
Run tracking and generate report:
```bash
./cuvslam_app.py \
    --odometry_mode=multicamera \
    --rectified_stereo_camera=true \
    --async_sba=false \
    --test_config=kitti-vio_gt.cfg \
    --multicam_mode=moderate \
    --use_segments
```

To generate a PDF report in addition to HTML:
```bash
./cuvslam_app.py \
    --odometry_mode=multicamera \
    --rectified_stereo_camera=true \
    --async_sba=false \
    --test_config=kitti-vio_gt.cfg \
    --multicam_mode=moderate \
    --use_segments \
    --pdf
```

### EuroC Dataset
Run tracking and generate report:
```bash
./cuvslam_app.py \
    --odometry_mode=inertial \
    --rectified_stereo_camera=false \
    --async_sba=false \
    --test_config=euroc-vio.cfg \
    --multicam_mode=moderate \
    --use_segments
```

### Custom Video
Run tracking on a video file with one loop:
```bash
./cuvslam_app.py \
    --dataset=/path/to/video \
    --stereo_edex=/path/to/stereo.edex \
    --num_loops=1 \
    --odometry_mode=mono \
    --visualize_rerun \
    --output_dir=/path/to/output/dir \
    --enable_observations_export=true
```

### VIPE Data Refinement
Run data refinement:
```bash
./vipe_evaluation.py \
    --video_dir=path/to/video_dir \
    --output_dir=path/to/output_dir \
    --config_path=path/to/config_file \
    --global_optimizer \
    --save_output_tracker_data
```

### RGBD Mode
Run tracking with depth images using RGBD odometry mode:
```bash
./cuvslam_app.py \
    --dataset=/path/to/rgbd_dataset \
    --odometry_mode=rgbd \
    --output_dir=/path/to/output
```

**RGBD Requirements:**
- The `stereo.edex` file must contain:
  - `depth_id` in camera config (specifies which camera index corresponds to depth)
  - `depth_sequence` array with paths to depth files (similar to image `sequence`)
  - `depth_scale_factor` - conversion factor for depth values (default: 1.0)
    - For PNG uint16: depth_meters = uint16_value * depth_scale_factor
    - For NPY float32: uint16_value = float32_meters / depth_scale_factor
  - Optional: `enable_depth_stereo_tracking` (default: false)
- Depth files can be:
  - `.png` files containing uint16 depth images (will be passed directly to tracker)
  - `.npy` files containing float32 depth arrays in meters (will be converted to uint16 using depth_scale_factor)
  - `.npy` files containing uint16 depth arrays (will be used directly)
- Depth files can be in folders or tar archives (like regular images)
- If using `frame_metadata.jsonl`, it should include `depth` field with paths and timestamps
- **Note**: cuVSLAM expects depth images in uint16 format. Float32 .npy files will be automatically converted.
