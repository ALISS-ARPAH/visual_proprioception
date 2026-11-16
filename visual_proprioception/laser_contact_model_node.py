#!/usr/bin/env python3
"""
ROS2 learned laser-tissue contact detector.

Runs a model exported by train_contact_model on every endoscope frame:
segment the spot (laser_spot_features), update the short time-window features,
crop around the spot, and score P(contact).

Raw frames are rectified with CameraInfo and resized to 540x540 first (the models
were trained on rectified 540x540 frames); frames that already are rectified
540x540 (CameraInfo without distortion) pass through unchanged.

The fiber tip pose (tip_pose_topic) projected with CameraInfo gates the spot
(blobs far from the tip are glare) and centres the crop when the aiming beam
is not visible, as in training. Set tip_max_age_sec <= 0 to run without it.

Publishes:
    probability_topic (Float64)  P(contact), NaN while the input is stale
                                 (depth models: depth_topic, mm short of cutting depth)
    suspected_topic   (Bool)     P >= suspect threshold: slow down / stop advancing
    touched_topic     (Bool)     P >= confirm threshold: contact confirmed
    overlay_topic     (Image)    debug view, only while someone subscribes

Thresholds default to the values stored in the model (0.6 / 0.8) and can be
overridden with parameters. smooth_sec (default: the model's value, 0 = off)
publishes the mean P over that many seconds instead of the per-frame P. If
frames stop arriving for timeout_sec, the node publishes NaN / False until they
resume, and the time-window state starts over. To feed the existing analysis overlay,
remap touched_topic to /tissue_touched and do not run laser_tissue_touch_node
at the same time.
"""

from collections import deque
import math
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64

from .contact_model import CausalMean, ContactModel
from .laser_spot_features import measure_spot, project_tip, Rectifier, SpotHistory, TipCheck


def stamp_seconds(header):
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


class LaserContactModelNode(Node):
    # Overlay wording for P >= confirm, P >= suspect and below, and the P label.
    STATES = ('CONTACT', 'SUSPECTED', 'no contact')
    PROB_LABEL = 'P(contact)'

    def __init__(self, node_name='laser_contact_model_node', overrides=None, **node_kwargs):
        """Create the node; `overrides` replaces parameter defaults (see the cut predictor)."""
        super().__init__(node_name, **node_kwargs)
        defaults = {
            'image_topic': '/ves_camera/image',
            'model_path': '',
            'probability_topic': '/laser/contact/probability',
            'depth_topic': '/laser/contact/depth_to_cut_mm',
            'suspected_topic': '/laser/contact/suspected',
            'touched_topic': '/laser/contact/touched',
            'overlay_topic': '/laser/contact/overlay',
            'suspect_threshold': -1.0,   # < 0: use the model's value
            'confirm_threshold': -1.0,   # < 0: use the model's value
            'smooth_sec': -1.0,          # < 0: use the model's value; 0: off
            'timeout_sec': 0.5,
            'tip_pose_topic': '/ves_smoother/right/tip_pose',
            'camera_info_topic': '/ves_camera/camera_info',
            'tip_max_age_sec': 0.1,      # pose must be this close to the image stamp; <= 0: off
        }
        defaults.update(overrides or {})
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        model_path = str(self.p('model_path')).strip()
        if not model_path:
            raise RuntimeError('model_path is required (model.json from train_contact_model)')
        self.model = ContactModel(model_path)
        self.history = SpotHistory(**self.model.window)
        smooth = float(self.p('smooth_sec'))
        self.smoother = CausalMean(smooth if smooth >= 0 else self.model.smooth_sec)
        self.bridge = CvBridge()
        self.last_frame_time = None
        self.poses = deque(maxlen=300)   # (stamp, xyz) of recent tip poses
        self.camera_info = None
        self.rectifier = None
        self.tip_check = TipCheck()
        self.wants_tip = 'tip' in self.model.meta.get('crop_center', '')

        depth = self.model.target == 'depth'
        self.prob_pub = self.create_publisher(
            Float64, self.p('depth_topic' if depth else 'probability_topic'), 10)
        self.suspected_pub = self.create_publisher(Bool, self.p('suspected_topic'), 10)
        self.touched_pub = self.create_publisher(Bool, self.p('touched_topic'), 10)
        self.overlay_pub = self.create_publisher(Image, self.p('overlay_topic'), 10)
        # Depth 1: if inference is slower than the camera, score the newest frame.
        self.create_subscription(Image, self.p('image_topic'), self.on_image, QoSProfile(depth=1))
        self.create_subscription(CameraInfo, self.p('camera_info_topic'), self.on_info, 1)
        if float(self.p('tip_max_age_sec')) > 0:
            self.create_subscription(PoseWithCovarianceStamped, self.p('tip_pose_topic'),
                                     self.on_pose, 50)
        self.create_timer(0.1, self.check_timeout)
        suspect, confirm = self.thresholds()
        self.get_logger().info(
            f'Loaded {self.model.kind} contact model from {model_path}; '
            f'suspect >= {suspect:.2f}, confirm >= {confirm:.2f}, '
            f'smoothing {self.smoother.window_s:g} s')

    def p(self, name):
        return self.get_parameter(name).value

    def thresholds(self):
        suspect = float(self.p('suspect_threshold'))
        confirm = float(self.p('confirm_threshold'))
        return (suspect if suspect >= 0 else self.model.thresholds['suspect'],
                confirm if confirm >= 0 else self.model.thresholds['confirm'])

    def on_pose(self, msg):
        p = msg.pose.pose.position
        self.poses.append((stamp_seconds(msg.header), (p.x, p.y, p.z)))

    def on_info(self, msg):
        old = self.camera_info
        if old is None or tuple(old.d) != tuple(msg.d) or tuple(old.p) != tuple(msg.p):
            self.rectifier = Rectifier.from_camera_info(msg)
        self.camera_info = msg

    def tip_pixel(self, t, image):
        """Projected tip for the pose closest in time to the frame, or None."""
        max_age = float(self.p('tip_max_age_sec'))
        if max_age <= 0 or self.camera_info is None or not self.poses:
            return None
        stamp, xyz = min(self.poses, key=lambda e: abs(e[0] - t))
        if abs(stamp - t) > max_age or not self.tip_check.update(t, xyz):
            return None
        info = self.camera_info
        return project_tip(xyz, info.p, info.r, info.width, info.height, image.shape[1])

    def publish(self, prob):
        suspect, confirm = self.thresholds()
        valid = math.isfinite(prob)
        self.prob_pub.publish(Float64(data=prob))
        self.suspected_pub.publish(Bool(data=valid and self.model.at_threshold(prob, suspect)))
        self.touched_pub.publish(Bool(data=valid and self.model.at_threshold(prob, confirm)))

    def on_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.get_logger().warn(f'Bad image: {exc}', throttle_duration_sec=3.0)
            return
        if self.rectifier is not None:
            image = self.rectifier(image)
        elif image.shape[:2] != (540, 540):
            self.get_logger().warn('Waiting for CameraInfo to rectify the raw frames',
                                   throttle_duration_sec=3.0)
            return
        stamp = stamp_seconds(msg.header)
        t = stamp if stamp > 0 else time.monotonic()
        started = time.monotonic()
        tip = self.tip_pixel(t, image)
        if tip is None and self.wants_tip and float(self.p('tip_max_age_sec')) > 0:
            self.get_logger().warn('No usable tip pose for this frame (missing, late, frozen or '
                                   'implausible); using the visible spot only',
                                   throttle_duration_sec=5.0)
        try:
            spot, contour = measure_spot(image, self.model.spot_params, tip)
            feats, center = self.history.update(t, spot, tip)
            prob = self.smoother.update(t, self.model.predict(image, feats, center, t))
        except cv2.error as exc:
            self.get_logger().warn(f'Contact model error: {exc}', throttle_duration_sec=3.0)
            return
        self.last_frame_time = time.monotonic()
        self.publish(prob)
        self.get_logger().debug(f'P(contact)={prob:.3f} in {self.last_frame_time - started:.3f} s')
        if self.overlay_pub.get_subscription_count() > 0:
            self.publish_overlay(msg, image, prob, contour, center, tip)

    def publish_overlay(self, msg, image, prob, contour, center, tip=None):
        canvas = image.copy()
        if tip is not None:
            cv2.drawMarker(canvas, (int(round(tip[0])), int(round(tip[1]))), (255, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)
        if contour is not None:
            cv2.drawContours(canvas, [contour], -1, (0, 0, 255), 2)
        if self.model.kind != 'spot':
            size = self.model.crop_size
            h, w = canvas.shape[:2]
            cu, cv = (w / 2.0, h / 2.0) if center is None else center
            x0, y0 = int(round(cu)) - size // 2, int(round(cv)) - size // 2
            cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (255, 180, 0), 1)
        suspect, confirm = self.thresholds()
        depth = self.model.target == 'depth'
        if self.model.at_threshold(prob, confirm):
            state, color = 'AT DEPTH' if depth else self.STATES[0], (0, 220, 0)
        elif self.model.at_threshold(prob, suspect):
            state, color = 'CLOSE' if depth else self.STATES[1], (0, 200, 255)
        else:
            state, color = 'too far' if depth else self.STATES[2], (0, 0, 255)
        label = (f'{prob:+.1f} mm to cut depth  {state}' if depth
                 else f'{self.PROB_LABEL} {prob:.2f}  {state}')
        for thickness, col in ((5, (0, 0, 0)), (2, color)):
            cv2.putText(canvas, label, (12, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, col, thickness, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        out.header = msg.header
        self.overlay_pub.publish(out)

    def check_timeout(self):
        timeout = float(self.p('timeout_sec'))
        if self.last_frame_time is None or not math.isfinite(timeout) or timeout <= 0:
            return
        if time.monotonic() - self.last_frame_time > timeout:
            self.history.reset()
            self.model.reset()
            self.smoother.reset()
            self.publish(float('nan'))


def main(args=None):
    rclpy.init(args=args)
    node = LaserContactModelNode()
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
