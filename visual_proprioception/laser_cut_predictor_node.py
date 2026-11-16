#!/usr/bin/env python3
"""
ROS2 cut predictor for robotic resection: the probability that firing the laser now cuts.

Runs the shipped cut-depth model (models/cut_depth_combined: EfficientNet-B0 on a
crop around the fiber plus laser-spot features) on every endoscope frame and
publishes the probability that the probe is at cutting depth, i.e. that firing now
results in a successful cut.

Subscribes:
    image_topic        (Image)       /ves_camera/image: raw 1080x1080 or rectified 540x540
    camera_info_topic  (CameraInfo)  /ves_camera/camera_info: rectifies raw frames and
                                     projects the probe tip into the image
    tip_pose_topic     (PoseWithCovarianceStamped)  /ves_smoother/right/tip_pose: laser
                                     probe tip in the camera frame; centres the crop on the
                                     fiber and rejects glare far from it

Publishes:
    /laser/cut/probability  (Float64)  P(firing now cuts), mean over the last 0.3 s;
                                       NaN while no frames arrive (timeout_sec)
    /laser/cut/close        (Bool)     P >= 0.6: close to cutting depth
    /laser/cut/ready        (Bool)     P >= 0.8: at cutting depth, firing should cut
    /laser/cut/overlay      (Image)    debug view (spot, crop box, projected tip, P),
                                       rendered only while someone subscribes

The processing is the same code as laser_contact_model_node, with the same parameters
(model_path, suspect_threshold, confirm_threshold, smooth_sec, tip_max_age_sec,
timeout_sec, *_topic); only the defaults differ. model_path defaults to the model
installed with the package. The /laser/cut/close topic is set by suspected_topic and
/laser/cut/ready by touched_topic.
"""

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import rclpy
from rclpy.executors import ExternalShutdownException

from .laser_contact_model_node import LaserContactModelNode

MODEL = os.path.join('models', 'cut_depth_combined', 'model.json')


def default_model_path():
    """Return the shipped model: the installed share directory, else the source tree."""
    candidates = []
    try:
        candidates.append(os.path.join(get_package_share_directory('visual_proprioception'),
                                       MODEL))
    except PackageNotFoundError:
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', MODEL))
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    return os.path.normpath(candidates[0])


class LaserCutPredictorNode(LaserContactModelNode):
    STATES = ('AT CUT DEPTH', 'CLOSE', 'too far')
    PROB_LABEL = 'P(cut)'

    def __init__(self, **node_kwargs):
        super().__init__('laser_cut_predictor_node', {
            'model_path': default_model_path(),
            'probability_topic': '/laser/cut/probability',
            'depth_topic': '/laser/cut/depth_to_cut_mm',
            'suspected_topic': '/laser/cut/close',
            'touched_topic': '/laser/cut/ready',
            'overlay_topic': '/laser/cut/overlay',
        }, **node_kwargs)


def main(args=None):
    rclpy.init(args=args)
    node = LaserCutPredictorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
