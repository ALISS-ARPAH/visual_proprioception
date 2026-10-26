#!/usr/bin/env python3
"""ROS2 laser spot area-based tissue contact indicator.

Subscribes to /laser/spot/area_px2 (std_msgs/Float64).
Publishes /tissue_touched (std_msgs/Bool).

A small laser spot is treated as a *contact proxy*, not proof of contact.
Invalid/zero/non-finite areas publish False. A watchdog publishes False
if area measurements stop arriving.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64


class LaserTissueTouchNode(Node):
    def __init__(self):
        super().__init__('laser_tissue_touch_node')

        self.declare_parameter('area_topic', '/laser/spot/area_px2')
        self.declare_parameter('touch_topic', '/tissue_touched')
        self.declare_parameter('area_threshold_px2', 1000.0)
        self.declare_parameter('timeout_sec', 0.5)

        area_topic = str(self.get_parameter('area_topic').value)
        touch_topic = str(self.get_parameter('touch_topic').value)
        self.touch_pub = self.create_publisher(Bool, touch_topic, 10)
        self.area_sub = self.create_subscription(
            Float64, area_topic, self.on_area, 10
        )

        self.last_area_time = None
        self.last_touch = False
        self.create_timer(0.1, self.check_timeout)
        self.get_logger().info(
            f'Listening on {area_topic}; publishing Bool on {touch_topic}; '
            f'area threshold = {self.get_parameter("area_threshold_px2").value} px^2'
        )

    def publish_touch(self, touched):
        self.last_touch = bool(touched)
        self.touch_pub.publish(Bool(data=self.last_touch))

    def on_area(self, msg):
        self.last_area_time = time.monotonic()
        area = float(msg.data)
        threshold = float(self.get_parameter('area_threshold_px2').value)

        if not math.isfinite(threshold) or threshold <= 0.0:
            self.get_logger().warn(
                'area_threshold_px2 must be finite and > 0; publishing False',
                throttle_duration_sec=5.0,
            )
            self.publish_touch(False)
            return

        # Missing detections from the upstream analysis node are NaN.
        # Do not interpret missing/zero/invalid area as contact.
        touched = math.isfinite(area) and 0.0 < area <= threshold
        self.publish_touch(touched)

    def check_timeout(self):
        timeout = float(self.get_parameter('timeout_sec').value)
        if not math.isfinite(timeout) or timeout <= 0.0:
            return  # Watchdog disabled.
        if self.last_area_time is None:
            return
        if time.monotonic() - self.last_area_time > timeout:
            # Publish False repeatedly while stale so downstream consumers
            # do not continue treating the last True as current contact.
            self.publish_touch(False)


def main(args=None):
    rclpy.init(args=args)
    node = LaserTissueTouchNode()
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
