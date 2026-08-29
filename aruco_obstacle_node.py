#!/usr/bin/env python3
"""
ArUco obstacle-pose ROS2 node.

Detects the 3 obstacle tags (make_aruco_tags.py in this same clone) on the
SAME camera feed FoundationPose already reads, and publishes each
obstacle's pose as PoseStamped -- one topic per obstacle, mirroring
fp_ros_node.py's own /object_pose convention exactly, so
aruco_tf_broadcaster.py can turn it into TF the same way
fp_tf_broadcaster.py already does for the pushed object.

Pure OpenCV (cv2.aruco), no GPU, no learned model -- runs alongside
FoundationPose's GPU-heavy tracking without competing for it.

Detector parameters are the ones validated live against the D455: the
stock defaults flickered under normal lighting and rejected a tag
outright once it was small in frame (a wider clutter shot, camera farther
back) -- a wider adaptive-threshold search and a much lower minimum
marker size fixed both, at the cost of a few more false candidates that
solvePnP simply fails to decode.
"""

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage

# This environment's opencv is a headless conda-forge build (no GUI
# backend compiled in -- confirmed via cv2.getBuildInformation(), "GUI:
# NONE" -- and no non-headless build exists on conda-forge for linux-64
# at all), so cv2.imshow cannot work here regardless of DISPLAY. Writing
# the frame here instead; image_viewer.py (plain Tkinter, no opencv GUI
# dependency) auto-reloads it from disk for an actual live view.
VIS_PATH = "/tmp/aruco_obstacle_vis.jpg"

# obs_name -> ArUco ID (DICT_4X4_50). Must match make_aruco_tags.py's own
# MARKER_IDS -- this is the one place that mapping is written down on the
# perception side.
MARKER_IDS = {1: "obs_1", 2: "obs_2", 3: "obs_3"}
DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_LENGTH_M = 0.085  # matches the printed tags (make_print_pdfs.py)


def build_detector() -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def marker_object_points(length: float) -> np.ndarray:
    """solvePnP object points in the marker's own local frame: +X right,
    +Y up, +Z out of the tag, origin at its center -- OpenCV's standard
    ArUco convention, matching detectMarkers' own corner order (top-left,
    top-right, bottom-right, bottom-left)."""
    h = length / 2.0
    return np.array([
        [-h, h, 0],
        [h, h, 0],
        [h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


class ArucoObstacleNode(Node):

    def __init__(self) -> None:
        super().__init__("aruco_obstacle_node")
        self.bridge = CvBridge()
        self.latest_cam_K = None
        self.latest_dist = None

        self.detector = build_detector()
        self.obj_points = marker_object_points(MARKER_LENGTH_M)

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        # On by default -- see VIS_PATH above for why this writes to disk
        # (image_viewer.py) rather than using cv2.imshow directly.
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter(
            "visualize").get_parameter_value().bool_value

        rgb_topic = self.get_parameter(
            "rgb_topic").get_parameter_value().string_value
        cam_info_topic = self.get_parameter(
            "camera_info_topic").get_parameter_value().string_value

        self.create_subscription(ROSImage, rgb_topic, self.rgb_callback, 1)
        self.create_subscription(CameraInfo, cam_info_topic,
                                  self.cam_info_callback, 1)

        self.pose_pubs = {
            marker_id: self.create_publisher(
                PoseStamped, f"/obstacle_pose/{name}", 1)
            for marker_id, name in MARKER_IDS.items()
        }

        self.get_logger().info(
            f"ArUco obstacle node ready -- watching for IDs {list(MARKER_IDS)} "
            f"on {rgb_topic}")

    def cam_info_callback(self, msg: CameraInfo) -> None:
        self.latest_cam_K = np.array(msg.k).reshape(3, 3)
        self.latest_dist = np.array(msg.d)

    def rgb_callback(self, msg: ROSImage) -> None:
        if self.latest_cam_K is None:
            self.get_logger().warn("Waiting for camera_info...",
                                    throttle_duration_sec=2.0)
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        vis = None
        if self.visualize:
            vis = frame.copy()
            if rejected:
                cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))

        if ids is not None:
            stamp = msg.header.stamp
            if self.visualize:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in MARKER_IDS:
                    continue  # some other tag in frame, not one of ours
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, corners[i][0], self.latest_cam_K,
                    self.latest_dist)
                if not ok:
                    continue
                self.publish_pose(marker_id, rvec, tvec, stamp)
                if self.visualize:
                    cv2.drawFrameAxes(vis, self.latest_cam_K, self.latest_dist,
                                       rvec, tvec, MARKER_LENGTH_M * 0.75, 3)
                    name = MARKER_IDS[marker_id]
                    corner0 = tuple(corners[i][0][0].astype(int))
                    cv2.putText(vis, name, (corner0[0], corner0[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.visualize:
            n_seen = 0 if ids is None else len(ids)
            n_rej = len(rejected) if rejected else 0
            cv2.putText(vis, f"detected: {n_seen}  rejected: {n_rej}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            try:
                # Temp file keeps the .jpg extension -- imwrite picks its
                # codec from the extension, so ".tmp" alone would fail.
                root, ext = os.path.splitext(VIS_PATH)
                tmp_path = f"{root}.tmp{ext}"
                if not cv2.imwrite(tmp_path, vis):
                    raise IOError(f"cv2.imwrite returned False for {tmp_path}")
                os.replace(tmp_path, VIS_PATH)  # atomic -- viewer never sees a partial write
            except Exception as e:
                # The viewer is a convenience, not the pipeline -- detection
                # and /obstacle_pose publishing above must never go down
                # because a debug frame couldn't be written to disk.
                self.get_logger().warn(f"vis frame write failed: {e}",
                                       throttle_duration_sec=5.0)

    def publish_pose(self, marker_id: int, rvec: np.ndarray, tvec: np.ndarray,
                      stamp) -> None:
        quat_xyzw = Rotation.from_rotvec(rvec.flatten()).as_quat()
        trans = tvec.flatten()

        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = "fp_camera_color_optical_frame"
        out.pose.position.x = float(trans[0])
        out.pose.position.y = float(trans[1])
        out.pose.position.z = float(trans[2])
        out.pose.orientation.x = float(quat_xyzw[0])
        out.pose.orientation.y = float(quat_xyzw[1])
        out.pose.orientation.z = float(quat_xyzw[2])
        out.pose.orientation.w = float(quat_xyzw[3])
        self.pose_pubs[marker_id].publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoObstacleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
