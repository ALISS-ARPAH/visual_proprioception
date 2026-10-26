#!/usr/bin/env python3
"""ROS2 laser spot geometry diagnostics and calibration data collection.

Spot measurements are published for every mask. Debug overlays are published
ONLY using a segmentation overlay with the matching source image timestamp.
No mask-as-background fallback and no reuse of stale camera frames.
"""
import csv
import math
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64


class LaserSpotAnalysisNode(Node):
    def __init__(self):
        super().__init__('laser_spot_analysis_node')
        self.declare_parameter('mask_topic', '/laser/segmentation_mask')
        self.declare_parameter('overlay_topic', '/laser/segmentation_overlay')
        self.declare_parameter('min_spot_area_px', 30.0)
        self.declare_parameter('ground_truth_distance_m', -1.0)
        self.declare_parameter('csv_path', '')
        self.declare_parameter('publish_debug_overlay', True)
        # Retained for launch-file compatibility. Matching is now by exact
        # source image timestamp rather than approximate arrival time.
        self.declare_parameter('max_overlay_age_sec', 0.05)
        self.declare_parameter('overlay_wait_sec', 0.5)
        self.declare_parameter('max_pending_frames', 100)
        self.declare_parameter('touch_topic', '/tissue_touched')
        self.declare_parameter('touch_status_timeout_sec', 0.5)

        self.bridge = CvBridge()
        self.tissue_touched = False
        self.last_touch_update_ns = None
        self.pending_masks = OrderedDict()   # stamp tuple -> (received_monotonic_ns, values, valid, contour)
        self.pending_overlays = OrderedDict() # stamp tuple -> (received_monotonic_ns, Image)
        self.pubs = {}
        for key in ('area_px2', 'equivalent_diameter_px', 'major_axis_px',
                    'minor_axis_px', 'aspect_ratio', 'circularity',
                    'centroid_u_px', 'centroid_v_px'):
            self.pubs[key] = self.create_publisher(Float64, '/laser/spot/' + key, 10)
        self.debug_pub = self.create_publisher(Image, '/laser/spot/analysis_overlay', 10)
        self.create_subscription(Image, self.get_parameter('overlay_topic').value,
                                 self.on_overlay, 10)
        self.create_subscription(Image, self.get_parameter('mask_topic').value,
                                 self.on_mask, 10)
        self.create_subscription(Bool, self.get_parameter('touch_topic').value,
                                 self.on_touch, 10)
        self.create_timer(0.1, self.prune_pending)

        self.csv_file = None
        self.csv_writer = None
        csv_path = str(self.get_parameter('csv_path').value).strip()
        if csv_path:
            p = Path(os.path.expanduser(csv_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.exists() and p.stat().st_size > 0
            self.csv_file = p.open('a', newline='', buffering=1)
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
                'stamp_sec', 'stamp_nanosec', 'ground_truth_distance_m',
                'valid', 'area_px2', 'equivalent_diameter_px', 'major_axis_px',
                'minor_axis_px', 'aspect_ratio', 'circularity',
                'centroid_u_px', 'centroid_v_px'])
            if not existing:
                self.csv_writer.writeheader()
            self.get_logger().info(f'Writing measurements to {p}')
        self.get_logger().info('Laser spot geometry analysis running (exact timestamp overlay matching)')

    @staticmethod
    def stamp_key(header):
        return (int(header.stamp.sec), int(header.stamp.nanosec))

    def now_ns(self):
        # Use a monotonic clock for arrival/expiration, unaffected by rosbag /clock.
        import time
        return time.monotonic_ns()

    def on_touch(self, msg):
        self.tissue_touched = bool(msg.data)
        self.last_touch_update_ns = self.now_ns()

    def on_overlay(self, msg):
        if not bool(self.get_parameter('publish_debug_overlay').value):
            return
        key = self.stamp_key(msg.header)
        if key in self.pending_masks:
            _, values, valid, contour = self.pending_masks.pop(key)
            self.publish_overlay(msg, values, valid, contour)
        else:
            self.pending_overlays[key] = (self.now_ns(), msg)
            self.pending_overlays.move_to_end(key)
        self.prune_pending()

    def on_mask(self, msg):
        values = {k: float('nan') for k in self.pubs}
        valid = False
        contour = None
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                if area >= float(self.get_parameter('min_spot_area_px').value):
                    moments = cv2.moments(contour)
                    if moments['m00'] > 0:
                        u = moments['m10'] / moments['m00']
                        v = moments['m01'] / moments['m00']
                        perimeter = cv2.arcLength(contour, True)
                        circularity = (4.0 * math.pi * area / (perimeter ** 2)
                                       if perimeter > 0 else float('nan'))
                        major = minor = ratio = float('nan')
                        if len(contour) >= 5:
                            (_, _), (axis1, axis2), _ = cv2.fitEllipse(contour)
                            major = float(max(axis1, axis2))
                            minor = float(min(axis1, axis2))
                            if minor > 1e-6:
                                ratio = major / minor
                        values.update(area_px2=area,
                                      equivalent_diameter_px=2 * math.sqrt(area / math.pi),
                                      major_axis_px=major, minor_axis_px=minor,
                                      aspect_ratio=ratio, circularity=circularity,
                                      centroid_u_px=u, centroid_v_px=v)
                        valid = True
        except (CvBridgeError, ValueError, cv2.error) as exc:
            self.get_logger().warn(f'Invalid segmentation mask: {exc}',
                                   throttle_duration_sec=3.0)

        # Measurement outputs remain independent of image overlay availability.
        for key, publisher in self.pubs.items():
            publisher.publish(Float64(data=float(values[key])))

        if self.csv_writer:
            self.csv_writer.writerow({
                'stamp_sec': msg.header.stamp.sec,
                'stamp_nanosec': msg.header.stamp.nanosec,
                'ground_truth_distance_m': float(self.get_parameter('ground_truth_distance_m').value),
                'valid': int(valid), **values})

        if bool(self.get_parameter('publish_debug_overlay').value):
            key = self.stamp_key(msg.header)
            if key in self.pending_overlays:
                _, overlay_msg = self.pending_overlays.pop(key)
                self.publish_overlay(overlay_msg, values, valid, contour)
            else:
                self.pending_masks[key] = (self.now_ns(), values, valid, contour)
                self.pending_masks.move_to_end(key)
            self.prune_pending()

    def prune_pending(self):
        wait_ns = int(max(0.0, float(self.get_parameter('overlay_wait_sec').value)) * 1e9)
        max_frames = max(1, int(self.get_parameter('max_pending_frames').value))
        now = self.now_ns()
        for cache in (self.pending_masks, self.pending_overlays):
            for key in list(cache.keys()):
                if now - cache[key][0] > wait_ns:
                    del cache[key]
            while len(cache) > max_frames:
                cache.popitem(last=False)

    def publish_overlay(self, overlay_msg, values, valid, contour):
        try:
            canvas = self.bridge.imgmsg_to_cv2(
                overlay_msg, desired_encoding='bgr8').copy()
        except (CvBridgeError, ValueError, cv2.error) as exc:
            self.get_logger().warn(f'Cannot render analysis overlay: {exc}',
                                   throttle_duration_sec=3.0)
            return
        if valid:
            u, v = values['centroid_u_px'], values['centroid_v_px']
            cv2.drawMarker(canvas, (int(round(u)), int(round(v))),
                           (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            if contour is not None and len(contour) >= 5:
                cv2.ellipse(canvas, cv2.fitEllipse(contour), (255, 180, 0), 2)
            lines = [
                f"Area: {values['area_px2']:.1f} px^2",
                f"Eq diameter: {values['equivalent_diameter_px']:.1f} px",
                f"Axes: {values['major_axis_px']:.1f} / {values['minor_axis_px']:.1f} px",
                f"Aspect ratio: {values['aspect_ratio']:.2f}",
                f"Circularity: {values['circularity']:.2f}",
            ]
        else:
            lines = ['Laser spot: N/A']
        for i, line in enumerate(lines):
            pos = (12, 30 + i * 27)
            cv2.putText(canvas, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
        # Bottom-right contact status from the separate touch detection node.
        timeout_ns = int(max(0.0, float(
            self.get_parameter('touch_status_timeout_sec').value)) * 1e9)
        fresh = (self.last_touch_update_ns is not None and
                 self.now_ns() - self.last_touch_update_ns <= timeout_ns)
        if fresh:
            status = 'TRUE' if self.tissue_touched else 'FALSE'
            color = (0, 220, 0) if self.tissue_touched else (0, 0, 255)
        else:
            status = 'N/A'
            color = (0, 200, 255)
        label = f'Tissue touched: {status}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        h, w = canvas.shape[:2]
        x = max(8, w - tw - 18)
        y = max(th + 8, h - 18)
        cv2.putText(canvas, label, (x, y), font, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, y), font, scale,
                    color, thickness, cv2.LINE_AA)

        output = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        output.header = overlay_msg.header
        self.debug_pub.publish(output)

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaserSpotAnalysisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
