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
| `laser_spot_monitor_gui` | Displays spot overlay, area history, touch status and the cut prediction; adjusts touch threshold | `/laser/spot/analysis_overlay`, `/laser/spot/area_px2`, `/tissue_touched`, `/laser/cut/probability`, `/laser/cut/close`, `/laser/cut/ready` | GUI display and touch-threshold parameter updates |
| `laser_cut_predictor_node` | Learned cut predictor: P that firing now gives a successful cut (see [Learned contact model](#learned-contact-model)) | `/ves_camera/image`, `/ves_camera/camera_info`, `/ves_smoother/right/tip_pose` | `/laser/cut/probability`, `/laser/cut/close` (P ≥ 0.6), `/laser/cut/ready` (P ≥ 0.8), `/laser/cut/overlay` |
| `laser_depth_area_logger` | Optionally logs paired distance and area measurements | `/laser/tissue_distance`, `/laser/spot/area_px2` | CSV file (default `~/laser_calibration_data/depth_area.csv`) |
| `laser_contact_model_node` | Same model as `laser_cut_predictor_node` under the older topic names; needs `model_path` | `/ves_camera/image`, `/ves_smoother/right/tip_pose`, `/ves_camera/camera_info` | `/laser/contact/probability`, `/laser/contact/suspected`, `/laser/contact/touched`, `/laser/contact/overlay` |

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

Launch the main processing nodes, the learned cut predictor and the monitoring GUI:

```bash
ros2 launch visual_proprioception laser_depth_estimation.launch.py
```

The GUI then shows the cut prediction (TOO SHALLOW / CLOSE / RIGHT DEPTH and P) above the segmented image. Launch arguments:

| Argument | Default | Meaning |
|---|---|---|
| `use_sim_time` | `false` | `true` for rosbag playback with `/clock` |
| `area_threshold_px2` | `1000.0` | contact threshold of `laser_tissue_touch_node` |
| `cut_predictor` | `true` | also run `laser_cut_predictor_node`; `false` = area-based nodes only, and the GUI banner shows N/A |

For rosbag playback with `/clock`:

```bash
ros2 launch visual_proprioception laser_depth_estimation.launch.py \
  use_sim_time:=true area_threshold_px2:=1500.0
# in another terminal:
ros2 bag play <bag> --clock
```

To run only the cut predictor, without the GUI or the other nodes (e.g. on the robot):

```bash
ros2 launch visual_proprioception laser_cut_predictor.launch.py
# options: model_path:=<model.json> image_topic:=... camera_info_topic:=... tip_pose_topic:=...
#          suspect_threshold:=0.6 confirm_threshold:=0.8 smooth_sec:=0.3 use_sim_time:=true
```

To run individual nodes instead:

```bash
ros2 run visual_proprioception laser_segmentation_node
ros2 run visual_proprioception laser_spot_analysis_node
ros2 run visual_proprioception laser_tissue_touch_node
ros2 run visual_proprioception laser_tissue_triangulation_node
ros2 run visual_proprioception laser_cut_predictor_node
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
ros2 topic echo /laser/cut/probability   # P(successful cut), from laser_cut_predictor_node
ros2 topic echo /laser/cut/ready         # true when P >= 0.8 (right depth)
```

## GUI Monitoring and Contact Threshold Adjustment

The package includes a monitoring GUI (`laser_spot_monitor_gui`) that displays the live endoscopic image with the detected laser spot and fitted ellipse overlaid. The GUI also provides a real-time plot of the laser spot area (in pixels²) and a contact detection indicator showing whether the laser is considered to be touching the tissue.

Users can manually adjust the **contact detection threshold** through an editable field in the GUI. The threshold defines the minimum detected laser spot area required to classify tissue contact. After entering a new threshold and applying the change, the GUI updates the `area_threshold_px2` parameter of the `laser_tissue_touch_node` through ROS2. This allows users to tune contact detection during experiments without restarting the nodes.

By default, the contact threshold is set to **1000 pixels²**, but it can be adjusted based on imaging conditions and experimental observations.

A banner above the segmented image shows the learned cut prediction from `laser_cut_predictor_node` (see [Learned contact model](#learned-contact-model)): whether firing now would give a successful cut, and its probability.

| Banner | Shown when |
|---|---|
| **TOO SHALLOW** (red) | P < 0.6 |
| **CLOSE** (amber) | P from 0.6 up to 0.8 |
| **RIGHT DEPTH** (green) | P ≥ 0.8 |

The state comes from the predictor's `/laser/cut/close` and `/laser/cut/ready` flags, so it follows any thresholds set on that node. The banner shows N/A if the predictor isn't running or has no camera frames.

`laser_depth_estimation.launch.py` starts the predictor along with the other nodes. Pass `cut_predictor:=false` to leave it out.

## Learned contact model

An alternative to the area threshold. For every endoscope frame it outputs **P(the fiber is at a cutting depth)**: close enough to the tissue that firing will cut. The fiber doesn't have to touch; it can be a little above the surface.

The training labels come from the parsed resection recordings and the robot kinematics. The tip position at each firing that cut marks the cutting depth for that spot. Every earlier frame gets its distance from that point along the beam.

The model combines two inputs:

1. **Laser spot features** (`laser_spot_features.py`): spot size, shape and brightness, plus their trend over the last 0.5–1.5 s.
2. **A 192 px image crop around the fiber**, classified by EfficientNet-B0, following the tool–tissue contact classifier of Fan et al.
   - The crop is centred on the laser spot.
   - If the spot isn't visible, it's centred on the fiber tip, projected from `/ves_smoother/right/tip_pose` with the camera calibration.
   - A green blob only counts as the spot if it's within 120 px of the projected tip, which rejects glare at the scope edge and on the other tool.

There are four models:
- `spot`: features only
- `image`: crop only
- `combined`: both
- `temporal`: `combined` plus the change in the crop's CNN features since about 0.4 s earlier. At runtime it reuses the features of recent frames, so the CNN still runs once per frame.

The spot segmentation in `laser_spot_features.py` differs from `laser_segmentation_node`:
- It caps hue at 78, so the blue glow at the fiber tip is not taken as the spot.
- It uses local green contrast, so the spot is still found on red tissue.
- It rejects glare streaks on the tool shaft.

The existing nodes are unchanged.

### Shipped model and files

The best model (combined, depth labels, seed 0) is in [models/cut_depth_combined/](models/cut_depth_combined/) and is installed to `share/visual_proprioception/models/cut_depth_combined/`:

| File | Contents |
|---|---|
| `model.json` | runtime config: features, spot segmentation, crop, thresholds 0.6 / 0.8, smoothing 0.3 s, labels and training settings |
| `model.onnx` | EfficientNet-B0 + spot-feature head, opset 12 (runs in OpenCV 4.6 `cv2.dnn`, no PyTorch) |
| `metrics.json`, `train.log`, `timelines.png` | cross-validated results of the run that produced it (all four models) |
| `dataset.json` | settings and counts of the frame table it was trained on |

Scripts in [scripts/](scripts/):

| Script | Purpose |
|---|---|
| `setup_train_venv.sh`, `requirements-train.txt` | recreate the PyTorch training venv (pinned versions) |
| `reproduce_cut_depth_model.sh` | rebuild the frame table and retrain / re-export the shipped model |
| `data/merge_parts.py`, `data/verify_data.py` | merge Google Drive zip parts into one folder per session and check it is complete |
| `analysis/compare_runs.py` | all runs (laser labels, depth labels × 3 seeds, mm regression) on the same frames, plus paired bootstrap |
| `analysis/per_session.py` | one run's AUC per held-out session, within-approach AUC, P by time before firing |
| `colleague/export_onnx.py` | BladderTouchDynamicCrop checkpoint → ONNX (weights only, `weights_only=True`) |
| `colleague/score_bag.py`, `colleague/evaluate_bag.py` | score a bag with both models on the same frames, then compare them with depth labels from the tip pose |

Results of the shipped model are in [results/](results/README.md):
- videos and per-frame CSVs on four recordings it never saw
- the cross-validated held-out video
- the head-to-head with BladderTouchDynamicCrop on lesion_2

The videos are git-ignored; the CSVs and logs are tracked.

### Workflow

```bash
cd ~/aliss_core/src/visual_proprioception

# 0. Data: one folder per session, <root>/<session>/rosbags/lesion_N/cut_*_rect_parsed/...
#    A Google Drive download split into zip parts (<session>-<stamp>Z-1-NNN/<session>/...) scatters
#    each segment's frames across the parts, so merge them first (hard links, no extra space):
python3 scripts/data/merge_parts.py ~/Resection_Data_Visual_Prop ~/Resection_Data_Visual_Prop_merged
python3 scripts/data/verify_data.py ~/Resection_Data_Visual_Prop_merged

# 1. Training venv (once): PyTorch + ONNX on top of the system ROS 2 / OpenCV
scripts/setup_train_venv.sh ~/aliss_runs/contact_venv

# 2. Frame table + 5-fold leave-one-session-out CV + ONNX export, with the shipped settings.
#    MODELS=combined for the shipped model only; all four take ~45 min on an RTX 5080.
scripts/reproduce_cut_depth_model.sh ~/Resection_Data_Visual_Prop_merged ~/aliss_runs/contact_full
#    The same steps by hand:
#    ros2 run visual_proprioception build_contact_dataset \
#      --data-root ~/Resection_Data_Visual_Prop_merged --out ~/aliss_runs/contact_full/frames_depth.csv
#    ~/aliss_runs/contact_venv/bin/python -m visual_proprioception.contact_train \
#      --frames ~/aliss_runs/contact_full/frames_depth.csv --out-dir ~/aliss_runs/contact_full/runD_seed0 \
#      --models spot image combined temporal --epochs 8 --epoch-frames 16000

# Analysis of the runs in ~/aliss_runs/contact_full (needs frames_depth.csv and the run folders)
~/aliss_runs/contact_venv/bin/python scripts/analysis/compare_runs.py
~/aliss_runs/contact_venv/bin/python scripts/analysis/per_session.py --run runD_seed0 --model combined

# Optional: video of held-out predictions, approaches back to back
ros2 run visual_proprioception render_contact_video \
  --run-dir ~/aliss_runs/contact_full/runD_seed0 --model combined

# Score a recorded bag (raw camera frames) and render a video + per-frame CSV,
# processing frames exactly as the live node does. --bag can also be a folder of split
# .mcap files without metadata.yaml (read in split order, empty files skipped):
MODEL=$(ros2 pkg prefix visual_proprioception)/share/visual_proprioception/models/cut_depth_combined/model.json
ros2 run visual_proprioception score_contact_bag --bag ~/aliss_runs/test_data/cut_1 --model $MODEL

# Compare with the colleague's BladderTouchDynamicCrop model on a bag (training venv;
# score_bag needs ROS sourced). The checkpoint is loaded as weights only:
~/aliss_runs/contact_venv/bin/python scripts/colleague/export_onnx.py <runs/original_crop/best.pt> /tmp/colleague.onnx
~/aliss_runs/contact_venv/bin/python scripts/colleague/score_bag.py <bag> $MODEL /tmp/colleague.onnx both.csv
~/aliss_runs/contact_venv/bin/python scripts/colleague/evaluate_bag.py both.csv

# 3. Run the model live on the robot (system Python; the CNN runs through OpenCV, no PyTorch).
#    Loads the shipped model by default and publishes /laser/cut/{probability,close,ready,overlay}:
ros2 launch visual_proprioception laser_cut_predictor.launch.py
#    or: ros2 run visual_proprioception laser_cut_predictor_node
#    The same processing under the older /laser/contact/* topic names:
ros2 run visual_proprioception laser_contact_model_node --ros-args -p model_path:=$MODEL
```

**Data and labels.** The dataset builder searches `--data-root` for `lesion_*/cut_*_rect_parsed/{resect_start,resect}/<seg>/` folders, each with `endoscope/*.png` and `kinematics.csv`. Lesions are named `<session>/lesion_N`. If the same recording appears twice, identified by identical kinematics timestamps, it is used once. Each approach (`resect_start`) is joined to the cutting segment (`resect`) that follows it.

Labels with `--labels depth` (the default). Here "depth" is the distance along the beam to where the fiber was at laser-on, for a firing that cut:

| Label | Meaning | Rule |
|---|---|---|
| `pre-fire` (1) | at cutting depth | before laser-on, within `--depth-pos` (1 mm) and within 2 mm sideways |
| `approaching` (0) | too shallow | before laser-on, `--depth-neg` (2 mm) or more short |
| `cutting` (not trained on) | laser firing | first 1 s after laser-on |
| `excluded` | not scored | the 1–2 mm band, and everything else |

**Retries.** If the surgeon fires again within 8 s, at least 0.5 mm deeper and within 2 mm sideways, the first firing is treated as too shallow, and its frames are measured to the second firing.

The 1 mm / 2 mm values are a working assumption, not a measured cutting range. Every model is therefore scored at several cutoffs (below).

`--labels laser` gives the earlier labels by time from laser-on (last 0.15 s before firing = contact, more than 0.5 s before = no contact). Those labels are also always written to the `laser_label`/`laser_kind` columns.

**Distance target.** `contact_train --target depth` trains the models to predict the distance in mm instead of P. It ranked worse than the yes/no models (see below), so it isn't the default.

**Evaluation.** `contact_train` holds out whole sessions (bladders) once there are at least 3, then whole lesions, then whole approaches; `--group-by` overrides this. It reports each model next to the current rule (legacy area ≤ 1000 px²), both raw and after a causal mean over the last `--smooth` seconds (default 0.3):
- **AUC at each cutoff c** (0.5, 1, 1.5 and 2 mm): frames within c mm of the cutting depth vs frames c + 1 mm or more short. This is the probability that the model ranks a random closer frame above a random farther one.
- **False alarm:** the share of approaches with any frame flagged while 2 mm or more short.
- **Before firing:** the share of approaches flagged while at cutting depth, before laser-on.
- **By +1 s:** the share flagged at depth or within 1 s after laser-on.
- **Latency:** median time of the first flag relative to laser-on.

Results go to `metrics.json`, and per-frame held-out predictions to `<model>/oof_predictions.csv` and `timelines.png`.

**Runtime.** `laser_cut_predictor_node` is the node for robotic resection. It estimates whether the current probe depth will give a successful cut.

It subscribes to:
- `/ves_camera/image`
- `/ves_camera/camera_info`
- the probe tip pose `/ves_smoother/right/tip_pose` (`PoseWithCovarianceStamped`, in the camera frame)

It publishes:

| Topic | Type | Meaning |
|---|---|---|
| `/laser/cut/probability` | `Float64` | P(firing now cuts), the mean over the last 0.3 s; NaN while no frames arrive |
| `/laser/cut/close` | `Bool` | P ≥ 0.6: close to cutting depth |
| `/laser/cut/ready` | `Bool` | P ≥ 0.8: at cutting depth, firing should cut |
| `/laser/cut/overlay` | `Image` | the spot, crop box, projected tip and P; rendered only while subscribed |

`model_path` defaults to the installed `models/cut_depth_combined/model.json`. `laser_contact_model_node` runs the same code, but needs `model_path` and publishes under the older names: `/laser/contact/probability`, `/laser/contact/suspected` (≥ 0.6) and `/laser/contact/touched` (≥ 0.8). Both nodes take these parameters:

- **Thresholds:** stored in `model.json`; override with `suspect_threshold` and `confirm_threshold`.
- **Smoothing:** `smooth_sec` (default: the model's value, 0.3 s; 0 = off) publishes the mean P over that window.
- **Rectification:** the models were trained on rectified 540×540 frames. Raw `/ves_camera/image` frames (1080×1080 with lens distortion) are undistorted with `CameraInfo`, as `image_proc` does, then resized. Frames that are already rectified 540×540 pass through unchanged.
- **Tip pose:** the node needs `/ves_smoother/right/tip_pose` and `/ves_camera/camera_info` (`tip_pose_topic`, `camera_info_topic`), with a pose within `tip_max_age_sec` (0.1 s) of each frame. Without them the crop falls back to the last seen spot, and the node warns, since the model was trained with the tip.
- **Timeout:** if frames stop for `timeout_sec`, it publishes NaN and False and resets.
- **Overlay:** `/laser/cut/overlay` (or `/laser/contact/overlay`) shows the spot, the crop box and the projected tip.
- **Existing GUI:** add `-r /laser/cut/ready:=/tissue_touched` (or `-r /laser/contact/touched:=/tissue_touched`), and don't run `laser_tissue_touch_node`.
- **Launch file:** `laser_cut_predictor.launch.py` exposes `model_path`, the three input topics, the thresholds, `smooth_sec` and `use_sim_time`.

The combined model takes about 13 ms per 540×540 frame on CPU.

**Training environment.** `~/aliss_runs/contact_venv` (`scripts/setup_train_venv.sh`) was created with `--system-site-packages`, so it still sees rclpy and the apt OpenCV. It has torch 2.11 with CUDA 12.8 for the RTX 5080 (versions pinned in `scripts/requirements-train.txt`). numpy must stay below 2 in this venv, because the apt OpenCV is built against numpy 1.x. Crops are cached next to the frame table (`<frames>.crops216.npy`, about 14 GB for 100k frames). Training uses bfloat16 mixed precision and 16k random frames per epoch.

### Results (all 5 sessions, each bladder held out in turn)

479 approaches, about 100k frames at 30 Hz, 11 lesions. Every row is scored on the same frames with the depth labels; the per-approach columns use P ≥ 0.8. "3 seeds" means three independent training runs; their results agree within ±0.01 AUC and ±1–2 points.

| Model (training labels) | AUC at 0.5 / 1 / 1.5 / 2 mm | False alarm | At depth before firing | By +1 s |
|---|---|---|---|---|
| Current rule (area ≤ 1000 px²) | 0.50 / 0.50 / 0.50 / 0.51 | – | – | – |
| Spot features (depth) | 0.50 / 0.50 / 0.49 / 0.48 | 17% | 7% | 14% |
| Image (old laser-timing labels) | 0.74 / 0.75 / 0.76 / 0.77 | 46% | 73% | 89% |
| Combined (distance target, mm) | 0.75 / 0.76 / 0.77 / 0.78 | 44% | 71% | 81% |
| Image (depth), 3 seeds | 0.78 / 0.79 / 0.80 / 0.81 | 52% | 81% | 91% |
| Temporal (depth), 3 seeds | 0.79 / 0.80 / 0.80 / 0.81 | 51% | 80% | 91% |
| **Combined (depth)** | **0.80 / 0.81 / 0.81 / 0.82** | 47% | 78% | 90% |
| Combined (depth), 0.3 s mean | 0.81 / 0.81 / 0.81 / 0.82 | 26% | 62% | 75% |

- **Depth labels are the biggest gain.** The same image model scores 0.75 at 1 mm when trained on laser-timing labels, and 0.79–0.81 when trained on depth labels. The full 30 Hz data took the image model from 0.68 to about 0.75 under the old laser-timing labels.
- **The best model flags cutting depth before firing in about 80% of approaches,** typically 0.45 s before the surgeon fires.
- **Results barely depend on the cutoff** between 0.5 and 2 mm.
- **Image, combined and temporal are effectively tied.** Temporal minus image at 1 mm is +0.005, +0.005 and −0.003 for the three seeds, each within its 95% bootstrap interval.
- **Predicting millimetres directly is worse:** AUC about 0.77 and a mean error of about 1.6 mm.
- **Spot size and brightness alone carry no usable signal across bladders.**
- **False alarms remain the main gap.** About half of approaches have at least one frame flagged while the fiber is 2 mm or more short; the 0.3 s mean roughly halves that.

**Data notes.**
- The complete export has `kinematics.csv` and every camera frame for every segment; an earlier partial download had about 30%.
- In `bladder_tele_virtuoso_01` the green aiming spot is mostly invisible (detected in about 20% of labelled frames), so its crops come mostly from the projected tip.
