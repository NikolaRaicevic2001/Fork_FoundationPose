#!/usr/bin/env python3
"""
Minimal RealSense -> ROS2 bridge, for testing the ArUco pipeline without
the full realsense2_camera driver package installed.

Publishes /camera/color/image_raw + /camera/color/camera_info from the
D455's own color stream. Color only, no depth, no dynamic reconfigure --
just enough for aruco_obstacle_node.py to have something to subscribe to.
"""

import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class RealsenseBridge(Node):

    def __init__(self) -> None:
        super().__init__("realsense_bridge")
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/camera/color/image_raw", 1)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", 1)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.info_msg = CameraInfo()
        self.info_msg.width = intr.width
        self.info_msg.height = intr.height
        self.info_msg.k = [
            intr.fx, 0.0, intr.ppx,
            0.0, intr.fy, intr.ppy,
            0.0, 0.0, 1.0,
        ]
        self.info_msg.d = list(intr.coeffs)
        self.info_msg.distortion_model = "plumb_bob"
        self.info_msg.header.frame_id = "fp_camera_color_optical_frame"

        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)
        self.get_logger().info("RealSense bridge streaming color to ROS2")

    def publish_frame(self) -> None:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return  # transient miss -- skip this tick, don't crash the node
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        frame = np.asanyarray(color_frame.get_data())

        now = self.get_clock().now().to_msg()
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "fp_camera_color_optical_frame"
        self.image_pub.publish(img_msg)

        self.info_msg.header.stamp = now
        self.info_pub.publish(self.info_msg)

    def destroy_node(self) -> None:
        self.pipeline.stop()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RealsenseBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
