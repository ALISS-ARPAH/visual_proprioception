"""End-to-end test of laser_cut_predictor_node with synthetic camera, calibration and tip pose."""

import math
import os
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64

from visual_proprioception.laser_cut_predictor_node import (default_model_path,
                                                            LaserCutPredictorNode)


def calibration(stamp):
    """540x540 rectified pinhole camera, f = 400 px, centre (270, 270), no distortion."""
    info = CameraInfo(width=540, height=540, distortion_model='plumb_bob')
    info.header.stamp = stamp
    info.k = [400.0, 0.0, 270.0, 0.0, 400.0, 270.0, 0.0, 0.0, 1.0]
    info.d = [0.0] * 5
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [400.0, 0.0, 270.0, 0.0, 0.0, 400.0, 270.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def test_default_model_is_the_shipped_one():
    path = default_model_path()
    assert path.endswith(os.path.join('cut_depth_combined', 'model.json'))
    assert os.path.exists(path)


def test_node_publishes_probability_and_flags(monkeypatch):
    # Keep the test's traffic off the robot network: own domain, this machine only.
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    context = rclpy.context.Context()
    rclpy.init(context=context, domain_id=87)
    node = helper = executor = None
    try:
        node = LaserCutPredictorNode(context=context)
        helper = rclpy.create_node('cut_predictor_test_helper', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        executor.add_node(helper)
        probs, ready, close = [], [], []
        helper.create_subscription(Float64, '/laser/cut/probability',
                                   lambda m: probs.append(m.data), 10)
        helper.create_subscription(Bool, '/laser/cut/ready', lambda m: ready.append(m.data), 10)
        helper.create_subscription(Bool, '/laser/cut/close', lambda m: close.append(m.data), 10)
        info_pub = helper.create_publisher(CameraInfo, '/ves_camera/camera_info', 10)
        pose_pub = helper.create_publisher(PoseWithCovarianceStamped,
                                           '/ves_smoother/right/tip_pose', 10)
        image_pub = helper.create_publisher(Image, '/ves_camera/image', 10)

        rng = np.random.default_rng(0)
        frame = np.full((540, 540, 3), (40, 45, 150), np.uint8)   # red tissue
        frame = cv2.add(frame, rng.integers(0, 8, frame.shape, dtype=np.uint8))
        cv2.circle(frame, (270, 270), 8, (90, 220, 60), -1)       # green aiming spot
        bridge = CvBridge()
        deadline = time.monotonic() + 10.0
        i = 0
        while len(probs) < 5 and time.monotonic() < deadline:
            stamp = helper.get_clock().now().to_msg()
            info_pub.publish(calibration(stamp))
            pose = PoseWithCovarianceStamped()
            pose.header.stamp = stamp
            # 20 mm in front of the camera, on the optical axis (projects to the spot)
            pose.pose.pose.position.x = 1e-5 * (i % 2)
            pose.pose.pose.position.z = 0.02
            pose.pose.pose.orientation.w = 1.0
            pose_pub.publish(pose)
            image = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image.header.stamp = stamp
            image_pub.publish(image)
            i += 1
            end = time.monotonic() + 0.05
            while time.monotonic() < end:
                executor.spin_once(timeout_sec=0.01)
        assert len(probs) >= 5, f'only {len(probs)} probabilities published'
        assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
        assert ready and close
        assert node.model.kind == 'combined'
    finally:
        if executor is not None:
            executor.shutdown()
        for n in (node, helper):
            if n is not None:
                n.destroy_node()
        rclpy.shutdown(context=context)


def test_gui_banner_shows_depth_state_from_predictor_flags():
    from visual_proprioception.laser_spot_monitor_gui import cut_banner
    now = 100.0
    text, bg, _ = cut_banner(0.91, True, True, now - 0.1, now)
    assert 'RIGHT DEPTH' in text and '0.91' in text and bg == '#1e8e3e'
    assert 'CLOSE' in cut_banner(0.7, True, False, now, now)[0]
    assert 'TOO SHALLOW' in cut_banner(0.2, False, False, now, now)[0]
    assert 'running?' in cut_banner(0.9, True, True, now - 5.0, now)[0]     # stale
    assert 'running?' in cut_banner(float('nan'), None, None, None, now)[0]  # never heard
    assert 'no camera frames' in cut_banner(float('nan'), False, False, now, now)[0]
