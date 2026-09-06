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

Translation comes from the aligned depth image, not from solvePnP's own
size-based estimate (2026-09-05): solvePnP's translation is only ever as
accurate as MARKER_LENGTH_M matches the real, physically-printed tag --
confirmed live, a re-print at a slightly different size put the
published obstacle position visibly off (while FoundationPose, which
reads real measured depth and assumes no marker size at all, stayed
accurate through the same session). Depth-based translation removes
that whole dependency permanently: the tag's own pixel footprint is
back-projected through the real depth reading at its center, using the
camera's own intrinsics, so a future re-print at yet another size can't
silently reintroduce this. Falls back to solvePnP's translation if no
depth topic has arrived yet or the specific pixel is a hole (common at
a mesh/tag edge) -- rotation is untouched either way: it's a shape fit
across the 4 corners, essentially insensitive to a uniform scale error
in the assumed size, unlike translation.
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
# Still fed to solvePnP (needed for its rotation estimate, and as the
# translation fallback when depth is unavailable) -- no longer the sole
# source of translation, see the module docstring. Re-measure this with
# a ruler (the black/white pattern only, not the white print margin)
# whenever the tags are reprinted, even though depth now absorbs most
# of a mismatch here.
MARKER_LENGTH_M = 0.085
# RealSense's own convention: raw uint16 millimeters. Aligned to the
# color frame, same topic fp_ros_node.py's own depth_callback reads.
DEPTH_TOPIC_DEFAULT = "/camera/aligned_depth_to_color/image_raw"
# Half-width of the pixel window averaged for a depth reading (so a
# single dead/hole pixel at the tag's exact center doesn't silently
# fall back to the less accurate solvePnP estimate for no good reason).
DEPTH_SAMPLE_HALFWIN_PX = 2


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


def depth_at_pixel(
    depth_img: np.ndarray, u: float, v: float, halfwin: int = DEPTH_SAMPLE_HALFWIN_PX,
) -> "float | None":
    """Median valid (nonzero) depth in meters over a small window around
    pixel (u, v). None if the window is entirely holes, or (u, v) falls
    outside the depth image -- aligned depth can be narrower at the
    edges than the RGB frame it's aligned to, so this is a real,
    expected case, not a bug."""
    h, w = depth_img.shape[:2]
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < w and 0 <= iv < h):
        return None
    patch = depth_img[max(0, iv - halfwin):iv + halfwin + 1,
                       max(0, iu - halfwin):iu + halfwin + 1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0  # RealSense mm -> m


class ArucoObstacleNode(Node):

    def __init__(self) -> None:
        super().__init__("aruco_obstacle_node")
        self.bridge = CvBridge()
        self.latest_cam_K = None
        self.latest_dist = None
        self.latest_depth = None

        self.detector = build_detector()
        self.obj_points = marker_object_points(MARKER_LENGTH_M)

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_topic", DEPTH_TOPIC_DEFAULT)
        # On by default -- see VIS_PATH above for why this writes to disk
        # (image_viewer.py) rather than using cv2.imshow directly.
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter(
            "visualize").get_parameter_value().bool_value

        rgb_topic = self.get_parameter(
            "rgb_topic").get_parameter_value().string_value
        cam_info_topic = self.get_parameter(
            "camera_info_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter(
            "depth_topic").get_parameter_value().string_value

        self.create_subscription(ROSImage, rgb_topic, self.rgb_callback, 1)
        self.create_subscription(CameraInfo, cam_info_topic,
                                  self.cam_info_callback, 1)
        self.create_subscription(ROSImage, depth_topic, self.depth_callback, 1)

        self.pose_pubs = {
            marker_id: self.create_publisher(
                PoseStamped, f"/obstacle_pose/{name}", 1)
            for marker_id, name in MARKER_IDS.items()
        }

        self.get_logger().info(
            f"ArUco obstacle node ready -- watching for IDs {list(MARKER_IDS)} "
            f"on {rgb_topic}, depth-backed translation from {depth_topic}")

    def cam_info_callback(self, msg: CameraInfo) -> None:
        self.latest_cam_K = np.array(msg.k).reshape(3, 3)
        self.latest_dist = np.array(msg.d)

    def depth_callback(self, msg: ROSImage) -> None:
        try:
            # "passthrough" keeps the raw uint16 mm values exactly as
            # published -- depth_at_pixel does the mm->m conversion
            # itself, rather than trusting an implicit cv_bridge
            # encoding conversion to also rescale units (it does not).
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

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
                tvec = self._refine_translation_from_depth(corners[i][0], tvec)
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

    def _refine_translation_from_depth(
        self, corners_2d: np.ndarray, tvec: np.ndarray
    ) -> np.ndarray:
        """Replace solvePnP's translation with one backed by measured
        depth at the tag's own pixel footprint -- see the module
        docstring for why. Falls back to solvePnP's own tvec, unchanged,
        if depth isn't available yet or the tag's pixel is a hole."""
        if self.latest_depth is None:
            return tvec
        u, v = corners_2d.mean(axis=0)
        z = depth_at_pixel(self.latest_depth, u, v)
        if z is None:
            return tvec
        fx, fy = self.latest_cam_K[0, 0], self.latest_cam_K[1, 1]
        cx, cy = self.latest_cam_K[0, 2], self.latest_cam_K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([[x], [y], [z]], dtype=np.float64)

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
