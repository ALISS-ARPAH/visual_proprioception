# Desctiption
ALISS/ARPA-H Visual Proprioception Package

This repository contains visual proprioception code/algorithm that uses laser beam info for estimating depth/touch for bladder application. 


# Authors 
* **Maintainer:** Mahshid Mansouri (mahshidalsadat.mansouri@vanderbilt.edu)

# Visual Proprioception

ROS 2 package for laser-based visual proprioception with a concentric tube robot (CTR). It detects the green laser spot in endoscopic images, measures spot geometry, estimates laser-to-tissue distance by triangulation, and provides a simple area-threshold-based tissue-contact indication.

## Main nodes

| Executable | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `laser_segmentation_node` | Detects the green laser spot | `/ves_camera/image`, `/ves_camera/camera_info`, `/ves_smoother/right/tip_pose` | `/laser/segmentation_mask`, `/laser/segmentation_overlay`, `/laser/spot/selection_valid` |
| `laser_spot_analysis_node` | Measures segmented spot area and shape | `/laser/segmentation_mask`, `/laser/segmentation_overlay` | `/laser/spot/area_px2`, `/laser/spot/analysis_overlay` (plus other spot metrics) |
| `laser_tissue_touch_node` | Estimates contact using spot area | `/laser/spot/area_px2` | `/tissue_touched` (`std_msgs/Bool`) |
| `laser_tissue_triangulation_node` | Estimates tissue position and distance from the camera and laser rays | Segmentation mask/overlay, camera info, CTR tip pose | `/laser/tissue_point`, `/laser/tissue_distance`, `/laser/triangulation_residual`, `/laser/triangulation_overlay`, `/laser/triangulation_debug_overlay` |
| `laser_spot_monitor_gui` | Displays spot overlay, area history, and touch status; adjusts touch threshold | `/laser/spot/analysis_overlay`, `/laser/spot/area_px2`, `/tissue_touched` | GUI display and touch-threshold parameter updates |
| `laser_depth_area_logger` | Optionally logs paired distance and area measurements | `/laser/tissue_distance`, `/laser/spot/area_px2` | CSV file (default `~/laser_calibration_data/depth_area.csv`) |

**Note:** Topic names above reflect the current development configuration. Confirm the defaults in the node scripts or use `ros2 node info /<node_name>` if topics have been remapped.

## Build

Clone this repository into the `src` directory of your ROS 2 workspace, then build:

```bash
cd ~/git/aliss_core/src
git clone https://github.com/ALISS-ARPAH/visual_proprioception.git
cd ~/git/aliss_core
colcon build --packages-select visual_proprioception
source install/setup.bash
```

The camera and CTR tip-pose publishers must also be running (or provided through a rosbag). Install the Python/ROS dependencies required by the nodes before running.

## Run

Launch the main processing nodes and monitoring GUI:

```bash
ros2 launch visual_proprioception laser_depth_estimation.launch.py
```

For rosbag playback with `/clock`:

```bash
ros2 launch visual_proprioception laser_depth_estimation.launch.py \
  use_sim_time:=true area_threshold_px2:=1500.0
```

To run individual nodes instead:

```bash
ros2 run visual_proprioception laser_segmentation_node
ros2 run visual_proprioception laser_spot_analysis_node
ros2 run visual_proprioception laser_tissue_touch_node
ros2 run visual_proprioception laser_tissue_triangulation_node
ros2 run visual_proprioception laser_spot_monitor_gui
```

Optional depth/area logger (in another terminal with the workspace sourced):

```bash
ros2 run visual_proprioception laser_depth_area_logger
```

## Check outputs

```bash
ros2 topic list | grep -E 'laser|tissue_touched'
ros2 topic echo /laser/spot/area_px2
ros2 topic echo /laser/tissue_distance
ros2 topic echo /tissue_touched
```

## Notes

- The touch output is **an area-threshold heuristic**, not a direct force or contact sensor.
- Triangulation depends on accurate camera calibration, laser geometry, and CTR tip pose. Incorrect frame alignment or nearly parallel rays can make distance estimates invalid.
- This package estimates laser spot geometry and tissue proximity; it does not itself command robot motion.
