# Test util to track, save map, localize using cuvslam API

To track and save poses to a file:
`./bin/cuvslam_api_launcher -dataset=<edex dir> -print_odom_poses=<path> -print_slam_poses=<path>`

To save map:
`./bin/cuvslam_api_launcher -dataset=<edex dir> -output_map=<map dir>`

To localize in map:
`./bin/cuvslam_api_launcher -dataset=<edex dir> -loc_input_map=<map dir> -loc_input_hints=<hint file> -print_loc_poses=<path>`
Additional flags with default values:
`-loc_start_frame=0 -loc_retries=0 -loc_hint_noise=0.0 -localize_forever=false -localize_wait=false -loc_random_rot=false -print_nan_on_failure=false`

Hint file rows format: `timestamp x y z [optional quaternion]`
Float timestamps in seconds and int timestamps in ns are supported. Hints must be sorted by timestamps.
To localize, the util will use the latest hint not later than current frame.

# Run tracker on EuRoC MAV Dataset (OBSOLETE)

```shell script
download.sh # Download .bag files

# Install requirements
sudo apt install python-cv-bridge python-opencv python-rosbag

# You should set CUVSLAM_DATASETS environment variable and
# create $CUVSLAM_DATASETS/euroc folder, then run export:
python3 extract_bag.py

# Run tracker
source cuvslam_vars.sh
./bin/cuvslam_api_launcher -dataset=$CUVSLAM_DATASETS/euroc/MH_01_easy

```
