#!/usr/bin/env python3
"""Image-only green laser segmentation for visualization and ellipse fitting.

No camera calibration, CTR pose, ray-distance, tracking-jump, or ambiguity
checks are used. The largest detected green contour is published.
IMPORTANT: selection_valid means IMAGE DETECTION ONLY, not geometrically
validated laser detection. Do not use it alone for safety-critical contact.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class LaserSegmentationRayGuidedNode(Node):
    def __init__(self):
        super().__init__('laser_segmentation_node')
        defaults = {
            'image_topic': '/ves_camera/image',
            'mask_topic': '/laser/segmentation_mask',
            'overlay_topic': '/laser/segmentation_overlay',
            'selection_valid_topic': '/laser/spot/selection_valid',
            'h_min': 35, 'h_max': 90,
            's_min': 12, 's_max': 255,
            'v_min': 95, 'v_max': 255,
            'green_excess_min': 9.0,
            'min_area_px': 0.0,  # No minimum-area rejection; tiny noise may pass.
            'blur_size': 5,
            'kernel_size': 7,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.bridge = CvBridge()
        self.mask_pub = self.create_publisher(Image, self.p('mask_topic'), 10)
        self.overlay_pub = self.create_publisher(Image, self.p('overlay_topic'), 10)
        self.valid_pub = self.create_publisher(Bool, self.p('selection_valid_topic'), 10)
        self.create_subscription(Image, self.p('image_topic'), self.on_image, 10)
        self.get_logger().info('Image-only laser segmentation enabled: no geometric rejection')

    def p(self, name):
        return self.get_parameter(name).value

    def on_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            blur = max(1, int(self.p('blur_size')))
            if blur % 2 == 0:
                blur += 1
            smoothed = cv2.GaussianBlur(image, (blur, blur), 0)
            hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
            lower = np.array([self.p('h_min'), self.p('s_min'), self.p('v_min')], dtype=np.uint8)
            upper = np.array([self.p('h_max'), self.p('s_max'), self.p('v_max')], dtype=np.uint8)
            binary = cv2.inRange(hsv, lower, upper)
            b, g, r = cv2.split(smoothed.astype(np.float32))
            excess = (g - 0.5 * (r + b) >= float(self.p('green_excess_min'))).astype(np.uint8) * 255
            binary = cv2.bitwise_and(binary, excess)
            kernel_size = max(1, int(self.p('kernel_size')))
            if kernel_size > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Select the largest green region without geometric validation.
            # A contour is required for ellipse fitting; empty frames stay empty.
            contour = max(contours, key=cv2.contourArea) if contours else None
            if contour is not None and cv2.contourArea(contour) < float(self.p('min_area_px')):
                contour = None
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            overlay = image.copy()
            detected = contour is not None and cv2.contourArea(contour) > 0
            if detected:
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)
            cv2.putText(overlay, f'Laser spot: {"DETECTED" if detected else "NOT DETECTED"}',
                        (12, overlay.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0) if detected else (0, 0, 255), 2, cv2.LINE_AA)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            mask_msg.header = msg.header
            overlay_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
            self.overlay_pub.publish(overlay_msg)
            self.valid_pub.publish(Bool(data=detected))
        except (CvBridgeError, ValueError, TypeError, cv2.error) as exc:
            self.get_logger().warn(f'Image segmentation error: {exc}', throttle_duration_sec=3.0)


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