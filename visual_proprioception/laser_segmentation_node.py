#!/usr/bin/env python3
"""Ray-guided green laser spot segmentation for ROS 2.

Inputs: camera image, CameraInfo, and right CTR tip pose in camera optical
frame or an explicitly configured equivalent frame.
Only the selected candidate is published in the binary mask. Fail closed: if
pose/camera calibration/selection is unavailable, publish an EMPTY mask.

IMPORTANT: Equivalence is a user-supplied assumption, NOT a TF transform.
Only configure equivalent frames when their coordinate axes and origins match.
Otherwise transform the pose upstream. A projected ray cannot by itself
distinguish a tool reflection that lies on that ray.
"""
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class LaserSegmentationRayGuidedNode(Node):
    def __init__(self):
        super().__init__('laser_segmentation_node')
        params = {
            'image_topic': '/ves_camera/image',
            'camera_info_topic': '/ves_camera/camera_info',
            'tip_pose_topic': '/ves_smoother/right/tip_pose',
            'mask_topic': '/laser/segmentation_mask',
            'overlay_topic': '/laser/segmentation_overlay',
            'selection_valid_topic': '/laser/spot/selection_valid',
            'h_min': 35, 'h_max': 90, 's_min': 12, 's_max': 255,
            'v_min': 95, 'v_max': 255, 'green_excess_min': 9.0,
            'min_area_px': 50.0, 'blur_size': 5, 'kernel_size': 7,
            'laser_origin_tip_m': [0.0, 0.0, 0.004],
            'laser_direction_tip': [0.0, 0.0, 1.0],
            'min_laser_range_m': 0.001, 'max_laser_range_m': 0.20,
            'ray_samples': 120,
            'max_ray_distance_px': 30.0,
            'max_tracking_jump_px': 65.0,
            'track_timeout_sec': 0.7,
            'tracking_weight': 0.35,
            'ambiguity_margin_px': 6.0,
            'max_pose_dt_sec': 0.06,
            'pose_wait_sec': 0.20,
            'max_pending_images': 30,
            'max_pose_history': 100,
            'show_projected_ray': False,
            'equivalent_camera_frames': ['endoscope_optical', 'ves_smoother/camera'],
        }
        for name, value in params.items():
            self.declare_parameter(name, value)
        self.bridge = CvBridge()
        self.mask_pub = self.create_publisher(Image, self.p('mask_topic'), 10)
        self.overlay_pub = self.create_publisher(Image, self.p('overlay_topic'), 10)
        self.valid_pub = self.create_publisher(Bool, self.p('selection_valid_topic'), 10)
        self.camera_info = None
        self.poses = deque(maxlen=max(1, int(self.p('max_pose_history'))))
        self.pending = deque()
        self.last_centroid = None
        self.last_track_stamp_ns = None
        self.create_subscription(CameraInfo, self.p('camera_info_topic'), self.on_info, 10)
        self.create_subscription(PoseWithCovarianceStamped, self.p('tip_pose_topic'), self.on_pose, 30)
        self.create_subscription(Image, self.p('image_topic'), self.on_image, 10)
        self.create_timer(0.025, self.process_pending)
        self.get_logger().info('Ray-guided laser segmentation started; configured equivalent frames are treated as identical (no transform)')

    def p(self, name):
        return self.get_parameter(name).value

    def on_info(self, msg):
        self.camera_info = msg

    def on_pose(self, msg):
        self.poses.append(msg)
        self.process_pending()

    def on_image(self, msg):
        self.pending.append((time.monotonic(), msg))
        max_n = max(1, int(self.p('max_pending_images')))
        while len(self.pending) > max_n:
            _, old = self.pending.popleft()
            self.process_frame(old, None)
        self.process_pending()

    def nearest_pose(self, image_stamp):
        if not self.poses:
            return None
        pose = min(self.poses, key=lambda x: abs(stamp_ns(x.header) - image_stamp))
        if abs(stamp_ns(pose.header) - image_stamp) > float(self.p('max_pose_dt_sec')) * 1e9:
            return None
        return pose

    def process_pending(self):
        wait = max(0.0, float(self.p('pose_wait_sec')))
        while self.pending:
            arrival, image = self.pending[0]
            pose = self.nearest_pose(stamp_ns(image.header))
            if pose is None and time.monotonic() - arrival < wait:
                break
            self.pending.popleft()
            self.process_frame(image, pose)

    def segmentation(self, image):
        blur = max(1, int(self.p('blur_size')))
        if blur % 2 == 0:
            blur += 1
        smoothed = cv2.GaussianBlur(image, (blur, blur), 0)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        lo = np.array([self.p('h_min'), self.p('s_min'), self.p('v_min')], dtype=np.uint8)
        hi = np.array([self.p('h_max'), self.p('s_max'), self.p('v_max')], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        b, g, r = cv2.split(smoothed.astype(np.float32))
        excess = (g - 0.5 * (r + b) >= float(self.p('green_excess_min'))).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, excess)
        kernel_n = max(1, int(self.p('kernel_size')))
        if kernel_n > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_n, kernel_n))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) >= float(self.p('min_area_px'))]

    def project_ray(self, pose, image):
        info = self.camera_info
        if info is None or pose is None:
            return None
        image_frame = image.header.frame_id.strip()
        info_frame = info.header.frame_id.strip()
        pose_frame = pose.header.frame_id.strip()
        equivalent = {str(frame).strip() for frame in self.p('equivalent_camera_frames')}
        equivalent.add(info_frame)
        if not info_frame or not pose_frame or pose_frame not in equivalent:
            self.get_logger().warn(
                f'Pose frame {pose_frame!r} is not CameraInfo frame {info_frame!r} '
                'or a configured equivalent frame; transform pose before running this node',
                throttle_duration_sec=5.0)
            return None
        if image_frame and image_frame not in equivalent:
            self.get_logger().warn(
                f'Image frame {image_frame!r} is not CameraInfo frame {info_frame!r} '
                'or a configured equivalent frame', throttle_duration_sec=5.0)
            return None
        if info.width and info.height and (info.width != image.width or info.height != image.height):
            self.get_logger().warn('CameraInfo/image resolution mismatch', throttle_duration_sec=5.0)
            return None
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            return None
        q = pose.pose.pose.orientation
        qv = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        norm = np.linalg.norm(qv)
        if norm < 1e-9:
            return None
        x, y, z, w = qv / norm
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
        p = pose.pose.pose.position
        tip = np.array([p.x, p.y, p.z], dtype=np.float64)
        origin = tip + R @ np.asarray(self.p('laser_origin_tip_m'), dtype=np.float64)
        direction = R @ np.asarray(self.p('laser_direction_tip'), dtype=np.float64)
        dn = np.linalg.norm(direction)
        if dn < 1e-9:
            return None
        direction /= dn
        near = max(0.0, float(self.p('min_laser_range_m')))
        far = float(self.p('max_laser_range_m'))
        if far <= near:
            return None
        t = np.linspace(near, far, max(2, int(self.p('ray_samples'))))
        xyz = origin[None, :] + t[:, None] * direction[None, :]
        xyz = xyz[xyz[:, 2] > 0.001]
        if len(xyz) < 2:
            return None
        D = np.asarray(info.d, dtype=np.float64)
        model = info.distortion_model
        if model not in ('', 'plumb_bob', 'rational_polynomial'):
            self.get_logger().warn(f'Unsupported distortion model {model!r}', throttle_duration_sec=5.0)
            return None
        if D.size not in (0, 4, 5, 8, 12, 14):
            return None
        try:
            uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, D if D.size else None)
        except cv2.error:
            return None
        uv = uv.reshape(-1, 2)
        uv = uv[np.isfinite(uv).all(axis=1)]
        return uv if len(uv) >= 2 else None

    @staticmethod
    def contour_distance_to_polyline(contour, polyline):
        # Minimum distance from any contour pixel to any sampled laser-ray segment.
        pts = contour.reshape(-1, 2).astype(np.float64)
        starts = polyline[:-1]
        ends = polyline[1:]
        best = float('inf')
        for a, b in zip(starts, ends):
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                continue
            u = np.clip(((pts - a) @ ab) / denom, 0, 1)
            d = np.linalg.norm(pts - (a + u[:, None] * ab), axis=1)
            best = min(best, float(np.min(d)))
        return best

    def select(self, contours, ray, image_stamp):
        if ray is None or not contours:
            return None
        track_ok = (self.last_centroid is not None and
                    self.last_track_stamp_ns is not None and
                    0 <= image_stamp - self.last_track_stamp_ns <=
                    float(self.p('track_timeout_sec')) * 1e9)
        scored = []
        for contour in contours:
            distance = self.contour_distance_to_polyline(contour, ray)
            if distance > float(self.p('max_ray_distance_px')):
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue
            centroid = np.array([moments['m10']/moments['m00'],
                                 moments['m01']/moments['m00']])
            score = distance
            if track_ok:
                jump = float(np.linalg.norm(centroid - self.last_centroid))
                if jump > float(self.p('max_tracking_jump_px')):
                    continue
                score += float(self.p('tracking_weight')) * jump
            scored.append((score, contour, centroid))
        if not scored:
            return None
        scored.sort(key=lambda v: v[0])
        if len(scored) > 1 and scored[1][0] - scored[0][0] < float(self.p('ambiguity_margin_px')):
            return None  # Fail closed if two candidates are similarly plausible.
        return scored[0][1], scored[0][2]

    def process_frame(self, msg, pose):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}', throttle_duration_sec=3.0)
            return
        canvas = image.copy()
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        selected = None
        ray = None
        try:
            contours = self.segmentation(image)
            ray = self.project_ray(pose, msg)
            selected = self.select(contours, ray, stamp_ns(msg.header))
        except (ValueError, TypeError, cv2.error) as exc:
            self.get_logger().warn(f'Segmentation/geometry error: {exc}', throttle_duration_sec=3.0)
        valid = selected is not None
        if valid:
            contour, center = selected
            cv2.drawContours(empty, [contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(canvas, [contour], -1, (0, 0, 255), thickness=2)
            self.last_centroid = center
            self.last_track_stamp_ns = stamp_ns(msg.header)
        elif (self.last_track_stamp_ns is not None and
              (stamp_ns(msg.header) < self.last_track_stamp_ns or
               stamp_ns(msg.header) - self.last_track_stamp_ns >
               float(self.p('track_timeout_sec')) * 1e9)):
            self.last_centroid = None
            self.last_track_stamp_ns = None
        if ray is not None and bool(self.p('show_projected_ray')):
            # Only draw in-bounds ray points to avoid integer overflow in OpenCV.
            visible = ray[(ray[:, 0] > -10000) & (ray[:, 0] < 10000) &
                          (ray[:, 1] > -10000) & (ray[:, 1] < 10000)]
            if len(visible) >= 2:
                cv2.polylines(canvas, [np.round(visible).astype(np.int32)], False,
                              (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Laser spot: {"TRACKED" if valid else "UNKNOWN"}',
                    (12, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if valid else (0, 0, 255), 2, cv2.LINE_AA)
        mask_msg = self.bridge.cv2_to_imgmsg(empty, encoding='mono8')
        overlay_msg = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        mask_msg.header = msg.header
        overlay_msg.header = msg.header
        self.mask_pub.publish(mask_msg)
        self.overlay_pub.publish(overlay_msg)
        self.valid_pub.publish(Bool(data=valid))


def main(args=None):
    rclpy.init(args=args)
    node = LaserSegmentationRayGuidedNode()
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