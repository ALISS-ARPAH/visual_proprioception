#!/usr/bin/env python3
"""Laser tissue triangulation with the segmentation node's exact overlay.

Overlay images are matched to segmentation masks by header timestamp. Each mask
is triangulated against a timestamp-matched tip pose, then its matching
segmentation overlay is annotated with that result or N/A. No contour is redrawn.

Assumptions: CameraInfo intrinsics describe the segmentation image, and the
configured camera frame aliases have an identity transform for the dataset.
Laser extrinsics are in the tip frame and MUST be calibrated for valid depth.
RViz markers show forward camera/laser rays, backward laser extension,
CTR tip, emitter, valid tissue point, and closest-approach residual.
"""
from collections import deque
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA



def stamp_seconds(header):
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


def stamp_key(header):
    return (header.stamp.sec, header.stamp.nanosec)


def quat_rotation(q):
    a = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-10 or not np.all(np.isfinite(a)):
        raise ValueError('Invalid orientation quaternion')
    x, y, z, w = a / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


class LaserTissueTriangulationNode(Node):
    def __init__(self):
        super().__init__('laser_tissue_triangulation_node')
        defaults = {
            'mask_topic': '/laser/segmentation_mask',
            'segmentation_overlay_topic': '/laser/segmentation_overlay',
            'camera_info_topic': '/ves_camera/camera_info',
            'tip_pose_topic': '/ves_smoother/right/tip_pose',
            'camera_frame': 'endoscope_optical',
            'equivalent_camera_frames': ['endoscope_optical', 'ves_smoother/camera'],
            'max_sync_dt_sec': 0.2,
            'max_overlay_sync_dt_sec': 0.001,
            'pose_wait_sec': 0.25,
            'overlay_wait_sec': 0.5,
            'min_spot_area_px': 30.0,
            'laser_origin_tip_m': [0.0, 0.0, 0.004],
            'laser_direction_tip': [0.0, 0.0, 1.0],
            'min_ray_angle_deg': 2.0,
            'max_ray_residual_m': 0.003,
            'min_range_m': 0.001,
            'max_range_m': 0.2,
            'distance_text_corner': 'top_right',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.bridge = CvBridge()
        self.poses = deque(maxlen=250)
        self.info = None
        # Entries are processed once, then removed. Monotonic arrival time
        # ensures pending frames eventually finish even when the bag pauses.
        self.pending = deque(maxlen=100)
        self.overlays = deque(maxlen=100)
        self.point_pub = self.create_publisher(PointStamped, '/laser/tissue_point', 10)
        self.distance_pub = self.create_publisher(Float64, '/laser/tissue_distance', 10)
        self.residual_pub = self.create_publisher(Float64, '/laser/triangulation_residual', 10)
        self.overlay_pub = self.create_publisher(Image, '/laser/triangulation_overlay', 10)
        self.debug_pub = self.create_publisher(Image, '/laser/triangulation_debug_overlay', 10)

        self.ray_marker_pub = self.create_publisher(
            Marker, '/laser/triangulation_rays', 10
        )

        self._last_debug = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('tip_pose_topic').value, self.on_pose, 30)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.on_info, 10)
        self.create_subscription(
            Image, self.get_parameter('mask_topic').value, self.on_mask, 10)
        self.create_subscription(
            Image, self.get_parameter('segmentation_overlay_topic').value,
            self.on_segmentation_overlay, 10)
        self.create_timer(0.03, self.process_pending)
        self.get_logger().info(
            'Ready: annotate /laser/segmentation_overlay with per-mask '
            'triangulated distance or N/A')

    def on_pose(self, msg):
        self.poses.append(msg)
        self.process_pending()

    def on_info(self, msg):
        self.info = msg
        self.process_pending()

    def on_mask(self, msg):
        self.pending.append({
            'mask': msg, 'arrived': time.monotonic(),
            'distance': None, 'processed': False, 'debug': None
        })
        self.process_pending()

    def on_segmentation_overlay(self, msg):
        self.overlays.append((msg, time.monotonic()))
        self.process_pending()

    def matching_overlay(self, mask):
        if not self.overlays:
            return None
        key = stamp_key(mask.header)
        # Prefer the exact image that produced the mask.
        for i, (overlay, _) in enumerate(self.overlays):
            if stamp_key(overlay.header) == key:
                return i, overlay
        tolerance = float(self.get_parameter('max_overlay_sync_dt_sec').value)
        if tolerance <= 0:
            return None
        i, (overlay, _) = min(
            enumerate(self.overlays),
            key=lambda item: abs(stamp_seconds(item[1][0].header) -
                                 stamp_seconds(mask.header)))
        if abs(stamp_seconds(overlay.header) - stamp_seconds(mask.header)) <= tolerance:
            return i, overlay
        return None

    def process_pending(self):
        now = time.monotonic()
        remaining = deque(maxlen=100)
        pose_wait = float(self.get_parameter('pose_wait_sec').value)
        overlay_wait = float(self.get_parameter('overlay_wait_sec').value)

        for entry in self.pending:
            mask = entry['mask']
            if not entry['processed']:
                # An empty mask never needs a tip pose or CameraInfo.
                try:
                    mask_image = self.bridge.imgmsg_to_cv2(
                        mask, desired_encoding='mono8')
                    contours, _ = cv2.findContours(
                        (mask_image > 0).astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    detected = bool(contours) and max(
                        cv2.contourArea(c) for c in contours
                    ) >= float(self.get_parameter('min_spot_area_px').value)
                except (CvBridgeError, ValueError, cv2.error) as exc:
                    detected = False
                    self.get_logger().warn(f'Invalid mask: {exc}', throttle_duration_sec=3.0)

                if not detected:
                    entry['processed'] = True
                else:
                    t = stamp_seconds(mask.header)
                    dt = min((abs(stamp_seconds(p.header) - t)
                              for p in self.poses), default=float('inf'))
                    ready = self.info is not None and dt <= float(
                        self.get_parameter('max_sync_dt_sec').value)
                    if ready:
                        self._last_debug = None
                        entry['distance'] = self.estimate_distance(mask)
                        entry['debug'] = self._last_debug
                        entry['processed'] = True
                    elif now - entry['arrived'] >= pose_wait:
                        self.get_logger().warn(
                            f'N/A: missing CameraInfo or synchronized tip pose '
                            f'(nearest dt={dt:.3f}s)',
                            throttle_duration_sec=3.0)
                        entry['processed'] = True

            if not entry['processed']:
                remaining.append(entry)
                continue

            match = self.matching_overlay(mask)
            if match is not None:
                _, overlay = match
                self.publish_annotated_overlay(overlay, entry['distance'], entry['debug'])
                # Remove this overlay to prevent accidental reuse.
                self.overlays = deque(
                    [(o, arrived) for o, arrived in self.overlays
                     if o is not overlay], maxlen=100)
            elif now - entry['arrived'] < overlay_wait:
                remaining.append(entry)
            else:
                self.get_logger().warn(
                    'No matching segmentation overlay for mask timestamp',
                    throttle_duration_sec=3.0)
        self.pending = remaining

        # If a segmentation overlay arrives without a mask, still publish
        # the original segmentation image plus N/A after a short wait.
        matched_keys = {stamp_key(e['mask'].header) for e in self.pending}
        remaining_overlays = deque(maxlen=100)
        for overlay, arrived in self.overlays:
            if now - arrived >= overlay_wait and stamp_key(overlay.header) not in matched_keys:
                self.publish_annotated_overlay(overlay, None)
            else:
                remaining_overlays.append((overlay, arrived))
        self.overlays = remaining_overlays

    def publish_annotated_overlay(self, overlay_msg, distance_m, debug=None):
        try:
            image = self.bridge.imgmsg_to_cv2(
                overlay_msg, desired_encoding='bgr8').copy()
            label = ('Laser-to-tissue: N/A' if distance_m is None else
                     f'Laser-to-tissue: {distance_m * 1000:.1f} mm')
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            pad, margin = 10, 12
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            height, width = image.shape[:2]
            corner = str(self.get_parameter('distance_text_corner').value)
            if corner not in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                corner = 'top_right'
            left = corner.endswith('left')
            top = corner.startswith('top')
            x = margin if left else max(0, width - tw - 2 * pad - margin)
            y = margin if top else max(0, height - th - baseline - 2 * pad - margin)
            cv2.rectangle(
                image, (x, y),
                (min(width - 1, x + tw + 2 * pad),
                 min(height - 1, y + th + baseline + 2 * pad)),
                (30, 30, 30), -1)
            cv2.putText(image, label, (x + pad, y + pad + th),
                        font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            output = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            output.header = overlay_msg.header
            self.overlay_pub.publish(output)
            self.publish_debug_overlay(image, overlay_msg.header, debug)
        except (CvBridgeError, ValueError, cv2.error) as exc:
            self.get_logger().warn(f'Overlay error: {exc}', throttle_duration_sec=3.0)

    def publish_debug_overlay(self, base_image, header, debug):
        """Publish an independent visualization; never modify the normal overlay."""
        canvas = base_image.copy()
        h, w = canvas.shape[:2]
        if debug is not None:
            u, v = debug['centroid']
            cv2.drawMarker(canvas, (int(round(u)), int(round(v))),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            K, D = debug['K'], debug['D']
            origin, direction = debug['origin'], debug['direction']
            # Independently project the smoother-estimated inner CTR tip.
            tip_xyz = debug['tip_origin']
            if np.isfinite(tip_xyz).all() and tip_xyz[2] > 1e-5:
                tip_px, _ = cv2.projectPoints(
                    tip_xyz.reshape(1, 1, 3).astype(np.float64),
                    np.zeros(3), np.zeros(3), K, D)
                tx, ty = tip_px.reshape(2)
                if np.isfinite([tx, ty]).all() and abs(tx) < 1e6 and abs(ty) < 1e6:
                    tip_uv = (int(round(tx)), int(round(ty)))
                    if 0 <= tip_uv[0] < w and 0 <= tip_uv[1] < h:
                        cv2.circle(canvas, tip_uv, 9, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(canvas, 'CTR tip', (tip_uv[0]+12, tip_uv[1]-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            distances_mm = [0, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
            xyz = np.array([origin + (mm / 1000.0) * direction
                            for mm in distances_mm], dtype=np.float64)
            valid = xyz[:, 2] > 1e-5
            if np.any(valid):
                projected, _ = cv2.projectPoints(
                    xyz[valid], np.zeros(3), np.zeros(3), K, D)
                pixels = projected.reshape(-1, 2)
                # Draw only visible sections; avoid overflow for off-image points.
                previous = None
                for mm, xy in zip(np.array(distances_mm)[valid], pixels):
                    if not np.isfinite(xy).all():
                        previous = None
                        continue
                    x, y = map(float, xy)
                    if abs(x) > 1e6 or abs(y) > 1e6:
                        previous = None
                        continue
                    pt = (int(round(x)), int(round(y)))
                    if previous is not None:
                        cv2.line(canvas, previous, pt, (255, 255, 0), 2, cv2.LINE_AA)
                    if 0 <= pt[0] < w and 0 <= pt[1] < h:
                        #cv2.circle(canvas, pt, 4, (255, 0, 255), -1)
                        cv2.putText(canvas, f'{mm}mm', (pt[0]+7, pt[1]-7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 0), 1, cv2.LINE_AA)
                    previous = pt
            cv2.putText(canvas, 'Yellow cross: spot | Cyan: beam | Blue circle: CTR tip',
                        (12, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, 'Yellow cross: spot | Cyan: beam | Blue circle: CTR tip',
                        (12, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, 'Projection unavailable: missing geometry/calibration',
                        (12, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2, cv2.LINE_AA)
        output = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        output.header = header
        self.debug_pub.publish(output)

    def publish_ray_markers(self, dc, ol, dl, tip_origin, stamp,
                            camera_range=None, laser_range=None,
                            pc=None, pl=None, residual=None, valid=False):
        """Publish 3D camera/laser rays and diagnostic geometry in the optical frame.

        The forward rays are drawn regardless of whether triangulation is valid.
        The backward laser extension is dashed red to expose negative solutions.
        All positions are in meters and in the configured camera frame.
        """
        frame = str(self.get_parameter('camera_frame').value).lstrip('/')
        max_range = float(self.get_parameter('max_range_m').value)
        length = max(0.15, max_range)

        def point(xyz):
            p = Point()
            p.x, p.y, p.z = (float(x) for x in xyz)
            return p

        def color(r, g, b, a=1.0):
            c = ColorRGBA()
            c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
            return c

        def marker(marker_id, ns, marker_type, rgb, scale):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = marker_id
            m.type = marker_type
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.color = color(*rgb)
            m.scale.x = scale
            m.lifetime.sec = 0
            m.lifetime.nanosec = 300000000  # Remove stale visuals if data stops.
            return m

        def line(marker_id, ns, a, b, rgb, width=0.0015):
            m = marker(marker_id, ns, Marker.LINE_LIST, rgb, width)
            m.points = [point(a), point(b)]
            self.ray_marker_pub.publish(m)

        def sphere(marker_id, ns, center, rgb, diameter=0.004):
            m = marker(marker_id, ns, Marker.SPHERE, rgb, diameter)
            m.scale.y = diameter
            m.scale.z = diameter
            m.pose.position = point(center)
            self.ray_marker_pub.publish(m)

        # Yellow: camera bearing through segmented green spot.
        line(0, 'camera_ray', np.zeros(3), length * dc, (1.0, 0.85, 0.0))
        # Cyan: physically forward predicted beam from the laser emitter.
        line(1, 'laser_ray', ol, ol + length * dl, (0.0, 0.85, 1.0))
        # Red dashed: backward extension of the infinite laser line.
        backward = marker(2, 'laser_backward_extension', Marker.LINE_LIST,
                          (1.0, 0.2, 0.2), 0.001)
        for i in range(12):
            a = ol - (i + 0.1) * (length / 12.0) * dl
            b = ol - (i + 0.6) * (length / 12.0) * dl
            backward.points.extend([point(a), point(b)])
        self.ray_marker_pub.publish(backward)

        # Blue: smoother-estimated CTR tip; white: offset laser emitter.
        sphere(3, 'ctr_tip', tip_origin, (0.15, 0.3, 1.0))
        sphere(4, 'laser_emitter', ol, (1.0, 1.0, 1.0), 0.003)

        # Green: validated tissue point. Do not display an invalid solution as tissue.
        if valid and pc is not None and pl is not None:
            sphere(5, 'tissue_point', 0.5 * (pc + pl), (0.1, 1.0, 0.3))

        # Magenta: shortest connecting segment between infinite lines, diagnostic
        # even when the solution lies behind the emitter (negative laser range).
        if pc is not None and pl is not None and np.isfinite(pc).all() and np.isfinite(pl).all():
            line(6, 'closest_approach', pc, pl, (1.0, 0.1, 0.8), 0.002)

    def estimate_distance(self, msg):
        if self.info is None or not self.poses:
            self.get_logger().warn('N/A: missing CameraInfo or tip poses', throttle_duration_sec=3.0)
            return
        t = stamp_seconds(msg.header)
        pose = min(self.poses, key=lambda m: abs(stamp_seconds(m.header)-t))
        dt = abs(stamp_seconds(pose.header)-t)
        if dt > float(self.get_parameter('max_sync_dt_sec').value):
            self.get_logger().warn(f'No synchronized tip pose (nearest dt={dt:.3f}s)', throttle_duration_sec=3.0)
            return
        # The two camera frames were observed to have an identity TF transform.
        # This is a *configured assumption*, not a TF lookup: verify that the
        # transform stays identity throughout the recording before using it.
        frame = str(self.get_parameter('camera_frame').value).lstrip('/')
        equivalent = {str(f).lstrip('/') for f in
                      self.get_parameter('equivalent_camera_frames').value}
        equivalent.add(frame)
        pose_frame = pose.header.frame_id.lstrip('/')
        info_frame = self.info.header.frame_id.lstrip('/')
        if pose_frame not in equivalent or info_frame not in equivalent:
            self.get_logger().warn(
                f'Unexpected camera frames: pose={pose_frame!r}, '
                f'CameraInfo={info_frame!r}; accepted={sorted(equivalent)}',
                throttle_duration_sec=3.0)
            return
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warn(f'Bad mask: {exc}')
            return
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn('N/A: no laser region in mask', throttle_duration_sec=3.0)
            return
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < float(self.get_parameter('min_spot_area_px').value):
            self.get_logger().warn('N/A: laser region below minimum area', throttle_duration_sec=3.0)
            return
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            return
        u, v = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        info = self.info
        if mask.shape[:2] != (info.height, info.width):
            self.get_logger().warn('Mask dimensions do not match CameraInfo', throttle_duration_sec=3.0)
            return
        K = np.asarray(info.k, dtype=float).reshape(3, 3)
        D = np.asarray(info.d, dtype=float)
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            return
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
            self.get_logger().warn('Unsupported camera distortion model', throttle_duration_sec=3.0)
            return
        # Undistort centroid and obtain normalized camera-frame bearing.
        try:
            undistorted = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, D)
        except cv2.error as exc:
            self.get_logger().warn(f'Camera calibration error: {exc}', throttle_duration_sec=3.0)
            return
        xy = undistorted[0, 0]
        dc = np.array([xy[0], xy[1], 1.0])
        dc /= np.linalg.norm(dc)
        tip = pose.pose.pose
        try:
            R = quat_rotation(tip.orientation)
            offset = np.array(self.get_parameter('laser_origin_tip_m').value, dtype=float)
            axis = np.array(self.get_parameter('laser_direction_tip').value, dtype=float)
            if offset.size != 3 or axis.size != 3 or np.linalg.norm(axis) < 1e-10:
                raise ValueError('Laser extrinsics must contain three values; direction must be nonzero')
            tip_origin = np.array([tip.position.x, tip.position.y, tip.position.z])
            ol = tip_origin + R @ offset
            dl = R @ (axis / np.linalg.norm(axis))
            if not np.isfinite(ol).all() or not np.isfinite(dl).all():
                return
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=3.0)
            return
        self._last_debug = {'centroid': (u, v), 'K': K.copy(),
                            'D': D.copy(), 'origin': ol.copy(),
                            'direction': dl.copy(), 'tip_origin': tip_origin.copy()}
        # Camera origin is [0,0,0] in optical frame. Minimize ||lambda*dc - (ol+s*dl)||.
        cos_angle = float(np.clip(np.dot(dc, dl), -1, 1))
        angle = math.degrees(math.acos(abs(cos_angle)))
        # Visualize both rays even when the angle is too small for depth estimation.
        self.publish_ray_markers(dc, ol, dl, tip_origin, msg.header.stamp)
        if angle < float(self.get_parameter('min_ray_angle_deg').value):
            self.get_logger().warn('Camera and laser rays nearly parallel: depth ill-conditioned', throttle_duration_sec=3.0)
            return
        # Diagnostics independent of the assumed laser beam direction:
        # shortest distance from emitter to the infinite camera bearing line.
        # Since the emitter is in front of the camera, its closest point on
        # the forward camera ray is normally the same.
        camera_at_emitter = float(np.dot(ol, dc))
        emitter_to_camera_ray = float(
            np.linalg.norm(ol - camera_at_emitter * dc)
        )

        # Unconstrained closest points on the two infinite lines.
        # Negative ranges are retained for diagnostics but never published.
        A = np.column_stack((dc, -dl))
        lam, s = np.linalg.lstsq(A, ol, rcond=None)[0]
        pc = lam * dc
        pl = ol + s * dl
        residual = float(np.linalg.norm(pc - pl))

        # self.get_logger().info(
        #     f'offset={offset.tolist()}, '
        #     f'axis={axis.tolist()}, '
        #     f'laser_origin={ol.round(5)}, '
        #     f'laser_direction={dl.round(5)}, '
        #     f'camera_range={lam*1000:.2f} mm, '
        #     f'laser_range={s*1000:.2f} mm, '
        #     f'residual={residual*1000:.2f} mm, '
        #     f'angle={angle:.2f} deg, '
        #     f'emitter_to_camera_ray={emitter_to_camera_ray*1000:.2f} mm, '
        #     f'camera_at_emitter={camera_at_emitter*1000:.2f} mm, '
        #     f'pose_sync_dt={dt*1000:.1f} ms'
        # )

        min_range = float(self.get_parameter('min_range_m').value)
        max_range = float(self.get_parameter('max_range_m').value)
        triangulation_valid = (
            lam > 0 and min_range <= s <= max_range
            and residual <= float(self.get_parameter('max_ray_residual_m').value)
        )
        self.publish_ray_markers(
            dc, ol, dl, tip_origin, msg.header.stamp,
            camera_range=lam, laser_range=s, pc=pc, pl=pl,
            residual=residual, valid=triangulation_valid)

        if lam <= 0 or not min_range <= s <= max_range:
            # self.get_logger().warn(
            #     f'N/A: invalid ray ranges '
            #     f'(camera={lam:.3f}m, laser={s:.3f}m)',
            #     throttle_duration_sec=3.0)
            return
        if residual > float(self.get_parameter('max_ray_residual_m').value):
            self.get_logger().warn(
                f'Ray residual {1000*residual:.2f} mm exceeds threshold',
                throttle_duration_sec=3.0)
            return
        point = 0.5*(pc+pl)
        output = PointStamped()
        output.header = msg.header
        output.header.frame_id = frame
        output.point.x, output.point.y, output.point.z = map(float, point)
        self.point_pub.publish(output)
        self.distance_pub.publish(Float64(data=float(s)))
        self.residual_pub.publish(Float64(data=residual))
        return float(s)


def main(args=None):
    rclpy.init(args=args)
    node = LaserTissueTriangulationNode()
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