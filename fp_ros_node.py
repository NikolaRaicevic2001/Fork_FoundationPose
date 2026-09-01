#!/usr/bin/env python3
"""
FoundationPose ROS2 node.
Subscribes to RGB-D camera images and SAM2 mask,
runs FoundationPose estimation/tracking,
and publishes the object pose as PoseStamped on /object_pose.
"""

import csv
import logging
import os
import time
from datetime import datetime
import numpy as np
import cv2
import torch
import nvdiffrast.torch as dr
import open3d as o3d
import rclpy
import trimesh

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from fp_ros_utils import get_mesh_file
from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    nvdiffrast_render,
    set_logging_format,
    set_seed,
)


class FoundationPoseROS2(Node):

    def __init__(self):
        super().__init__("fp_node")

        # FoundationPose's internals log ~23 INFO lines per tracked frame via the
        # stdlib logger (estimater, predict_pose_refine, make_crop_data_batch).
        # The cost is negligible (~0.09 ms/frame), but the flood buries the lines
        # that matter -- resets, stale-mask warnings, empty masks. Default to
        # WARNING and put the firehose behind -p verbose:=true.
        self.declare_parameter("verbose", False)
        self.verbose = self.get_parameter(
            "verbose").get_parameter_value().bool_value
        set_logging_format(logging.INFO if self.verbose else logging.WARNING)
        set_seed(0)

        # State variables
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_cam_K = None
        self.latest_mask = None
        self.latest_mask_stamp = None  # arrival time of latest mask (staleness)
        self.is_object_registered = False
        self.first = True

        # Constant-velocity SE(3) prior: seed each track step with an
        # extrapolation of the last two poses (pose_last is ob_in_cam).
        self.use_cv_prior = True
        self.pose_last_prev = None
        self.max_translation_step = 0.1  # reject >10cm/frame jumps as glitches

        # Mask-gated tracking: zero depth outside the SAM2 mask before tracking.
        self.use_mask_gating = True
        self.mask_gating_min_pixels = 100  # below this, mask is empty -> skip gating
        self.mask_gating_dilate_px = 15  # grow mask to absorb mask/object lag
        self.mask_gating_max_staleness_sec = 1.2  # skip gating if mask older (SAM2 ~1Hz)

        # Auto-reset parameters for spatial drift detection
        self.use_auto_reset = True
        self.auto_reset_patience = 3  # Consecutive frames required to trigger reset
        self.drift_counter = 0
        # Relaxed from 30 px: in a clean 428 s run the centroid crossed 30 px on
        # 10 frames while the plane criterion stayed flat (tilt < 5 deg, height
        # < 8 mm) -- i.e. those were momentary SAM2 mask lag, not pose error.
        # The plane criterion now covers the real failures, so this one no
        # longer has to be twitchy.
        self.declare_parameter("max_center_dist_px", 50.0)
        self.max_center_dist_px = self.get_parameter(
            "max_center_dist_px").get_parameter_value().double_value

        # Reset mode: centroid distance (translation-only) vs depth residual.
        # Depth residual compares the FP-rendered depth against observed depth
        # over co-visible, non-occluded pixels, so it also catches rotation
        # drift that leaves the centroid in place. Toggle at launch with
        # -p use_depth_residual_reset:=false to fall back to centroid mode.
        self.declare_parameter("use_depth_residual_reset", True)
        self.use_depth_residual_reset = self.get_parameter(
            "use_depth_residual_reset").get_parameter_value().bool_value
        self.declare_parameter("depth_residual_thresh_m", 0.015)
        self.depth_residual_thresh_m = self.get_parameter(
            "depth_residual_thresh_m").get_parameter_value().double_value
        # Min co-visible pixels for a valid residual (else hold, don't count).
        self.depth_residual_min_covisible_px = 200
        # Observed depth this much closer than rendered => an occluder in front.
        self.depth_residual_occlusion_margin_m = 0.03
        # If more than this fraction of co-visible pixels are occluded, the
        # object is heavily occluded => hold, never reset on this frame.
        self.depth_residual_heavy_occ_frac = 0.6

        # Mask IoU vote: FP's rendered silhouette vs SAM2's mask. Catches a
        # pose that is roughly the right distance away but the wrong shape
        # entirely (e.g. locked onto the robot arm instead of the object) --
        # a failure depth residual alone can miss if the arm sits at a
        # similar depth. `measure_all_criteria` already computed this exact
        # number for telemetry; this just lets it vote too.
        # Threshold is deliberately lenient: the reference implementation
        # this fork is built on uses 0.1-0.2 for the same style of check
        # (near-total mismatch only) -- SAM2's mask and FP's rendered
        # silhouette disagree at the boundary even when both are correct
        # (segmentation noise, the pusher stick's own occlusion), so a
        # naive-sounding "50%" would fire on healthy tracking. Toggle at
        # launch with -p use_iou_reset:=false.
        self.declare_parameter("use_iou_reset", True)
        self.use_iou_reset = self.get_parameter(
            "use_iou_reset").get_parameter_value().bool_value
        self.declare_parameter("iou_reset_thresh", 0.2)
        self.iou_reset_thresh = self.get_parameter(
            "iou_reset_thresh").get_parameter_value().double_value

        # ---- Table-plane constraint --------------------------------------
        # Pushing happens on a table, so the object rests on a known plane. That
        # removes exactly the freedoms tracking drifts into when occlusion makes
        # the observation partial: floating off the surface and tilting over.
        # Unlike the centroid / depth / IoU criteria this uses no SAM2 mask and
        # no per-frame depth, so occlusion cannot corrupt it.
        #   off     : disabled
        #   detect  : measure float height + tilt, use them to trigger a reset
        #   correct : additionally snap the pose back onto the plane each frame
        # Toggle at launch with -p plane_mode:=off|detect|correct
        self.declare_parameter("plane_mode", "detect")
        self.plane_mode = self.get_parameter(
            "plane_mode").get_parameter_value().string_value
        if self.plane_mode not in ("off", "detect", "correct"):
            raise ValueError(f"Unknown plane_mode: {self.plane_mode}")
        # Float tolerance is tight: an object resting on a table should not
        # hover. Tilt tolerance is deliberately loose -- for flat/elongated
        # meshes (banana) the reference axis is far less meaningful, so a
        # generous bound avoids false resets.
        # 30 mm: the worst height excursion observed across a clean run was
        # 12.4 mm (sd 1.3 mm), so this keeps ~2.5x margin over normal jitter.
        self.declare_parameter("plane_max_float_m", 0.03)
        self.plane_max_float_m = self.get_parameter(
            "plane_max_float_m").get_parameter_value().double_value
        self.declare_parameter("plane_max_tilt_deg", 25.0)
        self.plane_max_tilt_deg = self.get_parameter(
            "plane_max_tilt_deg").get_parameter_value().double_value
        # Own counter/patience so plane tuning stays independent of the
        # centroid / depth-residual criteria.
        self.declare_parameter("plane_patience", 5)
        self.plane_patience = self.get_parameter(
            "plane_patience").get_parameter_value().integer_value
        self.declare_parameter("plane_fit_thresh_m", 0.01)
        self.plane_fit_thresh_m = self.get_parameter(
            "plane_fit_thresh_m").get_parameter_value().double_value
        # In "correct" mode, which freedoms to snap back. Height is on by
        # default (a resting object cannot hover); tilt is off by default
        # because forcing an axis is the risky half of the constraint.
        self.declare_parameter("plane_correct_float", True)
        self.plane_correct_float = self.get_parameter(
            "plane_correct_float").get_parameter_value().bool_value
        self.declare_parameter("plane_correct_tilt", False)
        self.plane_correct_tilt = self.get_parameter(
            "plane_correct_tilt").get_parameter_value().bool_value

        # ---- Symmetry-aware pose canonicalisation ------------------------
        # A flat, texture-less block looks identical after a 180 deg flip, so
        # registration picks between the two representations essentially at
        # random (observed: 3 of 5 registrations landed on the opposite one).
        # Both are valid descriptions of the same physical placement, but a
        # consumer that reads the axis literally -- the downstream planner --
        # sees the object flip for no reason.
        #
        # So: keep every frame in the SAME representation the rest reference was
        # captured in, by picking whichever symmetry of the mesh minimises the
        # plane tilt. Because the mesh is symmetric under that transform nothing
        # is lost, and if no symmetry improves the tilt the pose is left alone --
        # that is a genuine flip, which the tilt criterion should report rather
        # than have silently rewritten.
        #
        # Deliberately NOT parameterised by naming an axis. An earlier version
        # asked for one ("-z") and picking the wrong one was catastrophic: the
        # named axis lay in the table plane, so its sign test sat on zero and
        # chattered every other frame, flipping the published pose 180 deg and
        # tripping a reset storm. Minimising tilt has no axis to misname and
        # agrees with the reset criterion by construction.
        self.declare_parameter("canonicalize", False)
        self.canonicalize = self.get_parameter(
            "canonicalize").get_parameter_value().bool_value
        # Which of the two representations the downstream consumer wants is a
        # convention, not geometry, so it stays a human decision -- but only a
        # single boolean, applied once at registration.
        self.declare_parameter("canonical_flip", False)
        self.canonical_flip = self.get_parameter(
            "canonical_flip").get_parameter_value().bool_value
        # Only switch representation when the alternative is better by this
        # margin, so a pose sitting near the midpoint cannot chatter.
        self.declare_parameter("canonical_switch_margin_deg", 20.0)
        self.canonical_switch_margin_deg = self.get_parameter(
            "canonical_switch_margin_deg").get_parameter_value().double_value
        self.declare_parameter("symmetry_tol", 0.02)
        self.symmetry_tol = self.get_parameter(
            "symmetry_tol").get_parameter_value().double_value
        self.symmetry_tfs = []  # object-frame 4x4 self-mappings of the mesh
        self.canon_applied = False

        # Plane state, all in the camera frame. Signed distance of a point X is
        # (plane_n . X + plane_d), positive on the camera side of the table.
        self.plane_n = None
        self.plane_d = None
        self.plane_u_ref = None  # object-frame axis that pointed "up" at reg.
        self.plane_rest_offset = 0.0  # lowest-point distance when at rest
        self.plane_violation_counter = 0

        # Refinement iterations
        self.first_est_refine_iter = 5  # Higher quality for first registration
        self.est_refine_iter = 1  # Fast re-init when reset triggered
        self.track_refine_iter = 2  # Per-frame tracking

        # ---- Published-pose smoothing (EMA) ------------------------------
        # Purely a publish-time filter: /object_pose is smoothed, but nothing
        # feeds back into FPModel.pose_last (unlike canonicalize_pose /
        # apply_plane_correction, which deliberately DO write back). Reasons:
        # (1) a discrete symmetry flip is not the kind of thing smoothing
        # should blend across -- canonicalize_pose already runs first, so by
        # the time this sees a pose it should already be in the consistent
        # representation; (2) seeding the tracker's own next-frame guess with
        # a lagged pose risks making FAST motion tracking worse, trading a
        # jitter fix for a drift regression. Every reset-vote/debug-viz/CSV
        # log upstream of this still sees the raw pose -- only the published
        # topic (and the on-screen visualize box) are smoothed.
        # Window is exposed the way it is usually talked about ("EMA over the
        # last N frames") rather than as a raw alpha; alpha = 2/(N+1) is the
        # standard N-period EMA correspondence. N <= 1 disables smoothing.
        self.declare_parameter("pose_ema_window", 3)
        self.pose_ema_window = self.get_parameter(
            "pose_ema_window").get_parameter_value().integer_value
        self.pose_ema_alpha = (2.0 / (self.pose_ema_window + 1)
                               if self.pose_ema_window > 1 else 1.0)
        self.pose_ema_pos = None    # (3,) running mean, world/camera frame
        self.pose_ema_quat = None   # (4,) xyzw, running mean then renormalised

        # FoundationPose library's internal debug level (passed to the model below).
        # Only >= 2 does anything: it dumps point clouds / refiner-vis images to disk,
        # which is slow. Keep at 0 for normal runs; bump manually when deep-debugging.
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.debug = 0
        self.debug_dir = f"{code_dir}/debug"

        # Our node's own real-time visualization (cv2 window). Separate from the
        # library debug above. Toggle at launch with -p visualize:=true.
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter(
            "visualize").get_parameter_value().bool_value
        self.latest_vis_img = None

        # Debug visualization: render the FP mesh silhouette at the tracked pose
        # (the same geometry the refiner compares against internally) and publish
        # it alongside the SAM2 mask so drift can be inspected in rqt.
        # Toggle at launch with -p debug_viz:=false to save the per-frame render.
        self.declare_parameter("debug_viz", True)
        self.debug_viz = self.get_parameter(
            "debug_viz").get_parameter_value().bool_value

        # ---- Per-frame CSV telemetry -------------------------------------
        # EVERY criterion is measured every frame and written out, no matter
        # which one is actually allowed to act. That is the point: thresholds
        # can then be chosen offline from a recorded run ("if tilt > 15 deg had
        # tripped a reset, when would it have fired?") without putting the robot
        # back on the table for each candidate value.
        # Enable with -p log_csv:=true
        self.declare_parameter("log_csv", False)
        self.log_csv = self.get_parameter(
            "log_csv").get_parameter_value().bool_value
        self.declare_parameter("log_dir", f"{code_dir}/logs")
        self.log_dir = self.get_parameter(
            "log_dir").get_parameter_value().string_value
        self.log_file = None
        self.log_writer = None
        self.log_rows = 0
        self.frame_idx = 0
        self.t_start = time.time()
        self.last_reset_reason = ""
        if self.log_csv:
            self._open_log()

        # Processing lock to prevent overlapping timer calls
        self.is_processing = False

        self.bridge = CvBridge()

        # Load object mesh
        mesh_file = get_mesh_file(self)
        self.object_mesh = trimesh.load(mesh_file)
        self.object_mesh.vertices *= 0.001  # Convert mesh from mm to meters
        self.to_origin, extents = trimesh.bounds.oriented_bounds(
            self.object_mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # Subsampled mesh vertices, used to find the object's lowest point above
        # the table each frame. Same convention as the pose published below.
        mesh_v = np.asarray(self.object_mesh.vertices, dtype=np.float64)
        if len(mesh_v) > 5000:
            mesh_v = mesh_v[np.random.default_rng(0).choice(
                len(mesh_v), 5000, replace=False)]
        self.plane_pts = mesh_v
        if self.canonicalize:
            self.symmetry_tfs = self._detect_symmetries()

        # FoundationPose model init
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.FPModel = FoundationPose(
            model_pts=self.object_mesh.vertices,
            model_normals=self.object_mesh.vertex_normals,
            mesh=self.object_mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx,
        )
        self.get_logger().info("FoundationPose model initialized")

        # Camera topic selection via ROS2 parameter
        self.declare_parameter("camera", "realsense")
        camera = self.get_parameter("camera").get_parameter_value().string_value
        self.get_logger().info(f"Using camera: {camera}")

        if camera == "zed":
            rgb_topic = "/zed/zed_node/rgb/image_rect_color"
            depth_topic = "/zed/zed_node/depth/depth_registered"
            cam_info_topic = "/zed/zed_node/rgb/camera_info"
        elif camera == "realsense":
            rgb_topic = "/camera/color/image_raw"
            depth_topic = "/camera/aligned_depth_to_color/image_raw"
            # depth_topic = "/camera/depth/image_rect_raw"
            cam_info_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribers
        self.create_subscription(ROSImage, rgb_topic, self.rgb_callback, 1)
        self.create_subscription(ROSImage, depth_topic, self.depth_callback, 1)
        self.create_subscription(ROSImage, "/sam2_mask", self.mask_callback, 1)
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self.cam_K_callback, 1)
        self.create_subscription(Int32, "/fp_reset", self.reset_callback, 1)

        # Publisher: PoseStamped instead of Pose (adds timestamp)
        self.pose_pub = self.create_publisher(PoseStamped, "/object_pose", 1)

        # Debug-viz publishers: FP silhouette mask + RGB overlay (view in rqt).
        self.render_mask_pub = self.create_publisher(ROSImage,
                                                     "/fp_render_mask", 1)
        self.debug_overlay_pub = self.create_publisher(ROSImage,
                                                       "/fp_debug_overlay", 1)
        # Colorized |z_render - z_obs| heatmap over co-visible pixels.
        self.depth_residual_pub = self.create_publisher(ROSImage,
                                                        "/fp_depth_residual", 1)

        # Timer-driven main loop (runs as fast as GPU allows)
        self.timer = self.create_timer(0.01, self.run_once)
        self.get_logger().info("FoundationPose ROS2 node ready")

    # ---------- callbacks ----------

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.latest_mask_stamp = self.get_clock().now()
        except CvBridgeError as e:
            self.get_logger().error(f"Mask conversion failed: {e}")

    def cam_K_callback(self, data: CameraInfo):
        self.latest_cam_K = np.array(data.k).reshape(3, 3)

    def reset_callback(self, data: Int32):
        if data.data > 0:
            self.get_logger().info("Reset triggered — re-registering object")
            self.is_object_registered = False
        else:
            self.get_logger().info(
                "Reset message received with data <= 0, ignoring")

    # ---------- main loop ----------

    def run_once(self):
        """Called by timer. Runs one registration or tracking step."""
        if self.is_processing:
            return

        if any(x is None for x in [
                self.latest_rgb, self.latest_depth, self.latest_mask,
                self.latest_cam_K
        ]):
            self.get_logger().warn(
                "Waiting for RGB, depth, mask, and camera_info...",
                throttle_duration_sec=2.0)
            return

        self.is_processing = True
        try:
            if not self.is_object_registered:
                self._register()
            else:
                self._track()
        finally:
            self.is_processing = False

    def _register(self):
        """Initial pose estimation using SAM2 mask."""
        self.get_logger().info("Running registration...")
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        mask = self.process_mask(self.latest_mask)
        cam_K = self.latest_cam_K.copy()

        t0 = time.time()
        pose = self.FPModel.register(
            K=cam_K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=self.first_est_refine_iter
            if self.first else self.est_refine_iter,
        )
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(
            f"Registration done in {elapsed_ms:.1f} ms, pose:\n{pose}")
        assert pose.shape == (4, 4), f"Unexpected pose shape: {pose.shape}"
        self.is_object_registered = True
        self.first = False
        self.pose_last_prev = None  # reset CV-prior history after re-registration
        self.pose_ema_pos = None  # reset publish-time smoothing too -- a new
        self.pose_ema_quat = None  # registration must not blend against the old one
        if self.log_csv:
            self._log_row(t_sec=round(time.time() - self.t_start, 4),
                          frame=self.frame_idx,
                          state="register",
                          track_ms=round(elapsed_ms, 2))
            self.frame_idx += 1
        self.plane_violation_counter = 0
        # Fit the table once, from the first (highest-quality) registration.
        # Later re-registrations reuse it: the camera and table do not move, and
        # re-deriving the reference from a mediocre re-registration would bake
        # that error into the constraint.
        if self.plane_mode != "off" and self.plane_n is None:
            self._fit_table_plane(depth, mask, cam_K)
        if self.plane_n is not None and self.plane_u_ref is None:
            self._capture_rest_reference(pose)
            self._report_axis_alignment(pose)
        # Snap to the convention immediately. With a mesh-anchored reference the
        # very first registration can already be the wrong representation, so
        # this must run then too -- not only on re-registrations.
        pose = self.canonicalize_pose(pose)

    def _track(self):
        """Frame-to-frame tracking."""
        rgb = self.process_rgb(self.latest_rgb)
        depth_obs = self.process_depth(self.latest_depth)  # ungated observation
        cam_K = self.latest_cam_K.copy()

        # Mask-gate depth + apply constant-velocity prior before tracking.
        depth = self.gate_depth_by_mask(depth_obs)
        self.apply_cv_prior()

        t0 = time.time()
        pose = self.FPModel.track_one(rgb=rgb,
                                      depth=depth,
                                      K=cam_K,
                                      iteration=self.track_refine_iter)
        elapsed_ms = (time.time() - t0) * 1000
        # Once a second is enough to watch the rate; per-frame is just noise.
        self.get_logger().info(f"Tracking done in {elapsed_ms:.1f} ms",
                               throttle_duration_sec=1.0)

        # Normalise the symmetry representation first, so every measurement and
        # everything published below refers to the same one.
        pose = self.canonicalize_pose(pose)

        # Measure the plane criterion on the RAW pose, before any correction:
        # in correct mode the correction would otherwise erase the very signal
        # the log is meant to capture.
        plane_pre = self.plane_metrics(pose)

        # Snap the pose back onto the table before anything downstream sees it.
        if self.plane_mode == "correct":
            pose = self.apply_plane_correction(pose)
        plane_post = self.plane_metrics(pose)

        # Render the FP depth once and reuse it for the debug viz, the
        # depth-residual reset, and telemetry (never render twice per frame).
        H, W = depth_obs.shape
        rendered_depth = None
        need_render = self.debug_viz or self.log_csv or (
            self.use_auto_reset and self.use_depth_residual_reset)
        if need_render:
            rendered_depth = self.render_fp_depth(pose, cam_K, H, W)

        # Publish debug viz before the reset check so borderline / stuck frames
        # (which do NOT trigger a reset) are still inspectable in rqt.
        if self.debug_viz:
            self.publish_debug_viz(rgb, pose, cam_K, depth_obs, rendered_depth)

        did_reset = self.check_auto_reset(pose, cam_K, depth_obs,
                                          rendered_depth)

        # Smoothing only runs for a frame that will actually publish -- a
        # reset frame must not update the EMA state (see apply_pose_ema's own
        # reasoning). Computed here, before logging, so the CSV can show
        # raw-vs-smoothed side by side instead of only ever the raw pose.
        pose_smoothed = None if did_reset else self.apply_pose_ema(pose)

        if self.log_csv:
            crit = self.measure_all_criteria(pose, cam_K, depth_obs,
                                             rendered_depth)
            trans = pose[:3, 3]
            quat = R.from_matrix(pose[:3, :3]).as_quat()  # xyzw
            if pose_smoothed is not None:
                trans_s = pose_smoothed[:3, 3]
                quat_s = R.from_matrix(pose_smoothed[:3, :3]).as_quat()
            self._log_row(
                t_sec=round(time.time() - self.t_start, 4),
                frame=self.frame_idx,
                state="track",
                track_ms=round(elapsed_ms, 2),
                tx=round(float(trans[0]), 5),
                ty=round(float(trans[1]), 5),
                tz=round(float(trans[2]), 5),
                qx=round(float(quat[0]), 6),
                qy=round(float(quat[1]), 6),
                qz=round(float(quat[2]), 6),
                qw=round(float(quat[3]), 6),
                # Published (post-EMA) pose -- blank on a reset frame, since
                # nothing gets published for one. Compare against tx/ty/tz/
                # qx/qy/qz/qw above to see exactly how much smoothing moved
                # this frame, rather than inferring it from the overlay.
                tx_pub=None if pose_smoothed is None else round(float(trans_s[0]), 5),
                ty_pub=None if pose_smoothed is None else round(float(trans_s[1]), 5),
                tz_pub=None if pose_smoothed is None else round(float(trans_s[2]), 5),
                qx_pub=None if pose_smoothed is None else round(float(quat_s[0]), 6),
                qy_pub=None if pose_smoothed is None else round(float(quat_s[1]), 6),
                qz_pub=None if pose_smoothed is None else round(float(quat_s[2]), 6),
                qw_pub=None if pose_smoothed is None else round(float(quat_s[3]), 6),
                plane_float_m=None if plane_pre is None else round(
                    plane_pre[0], 5),
                plane_tilt_deg=None if plane_pre is None else round(
                    plane_pre[1], 2),
                corr_float_mm=None if (plane_pre is None or plane_post is None)
                else round((plane_post[0] - plane_pre[0]) * 1000, 2),
                corr_tilt_deg=None if (plane_pre is None or plane_post is None)
                else round(plane_post[1] - plane_pre[1], 2),
                canon=int(self.canon_applied),
                drift_counter=self.drift_counter,
                plane_counter=self.plane_violation_counter,
                reset=int(did_reset),
                reset_reason=self.last_reset_reason,
                **crit)
        self.frame_idx += 1

        if did_reset:
            return  # Abort publishing this frame and re-register next frame

        self.update_cv_prior_history()
        # Smoothing is publish-time only -- every check above (reset votes,
        # debug viz, CSV log's raw tx/ty/tz/...) already saw the raw `pose`,
        # on purpose. Reuse the value already computed above rather than
        # calling apply_pose_ema (and updating its state) a second time.
        pose = pose_smoothed
        self.publish_pose(pose)

        if self.visualize:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            vis_img = draw_posed_3d_box(cam_K,
                                        img=vis_img,
                                        ob_in_cam=center_pose,
                                        bbox=self.bbox)
            vis_img = draw_xyz_axis(vis_img,
                                    ob_in_cam=center_pose,
                                    scale=0.1,
                                    K=cam_K,
                                    thickness=3,
                                    transparency=0,
                                    is_input_rgb=True)
            # cv2.imshow("Pose Visualization", vis_img)
            # cv2.waitKey(1)
            self.latest_vis_img = vis_img

    # ---------- helpers ----------

    def publish_pose(self, pose: np.ndarray):
        assert pose.shape == (4, 4), f"Unexpected pose shape: {pose.shape}"
        trans = pose[:3, 3]
        quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"  # adjust if needed
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.pose_pub.publish(msg)

    def process_rgb(self, rgb):
        return rgb

    def process_depth(self, depth):
        depth = depth.copy()
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        if depth.max() > 100:  # mm → m
            depth = depth / 1000.0
        depth[depth < 0.1] = 0
        depth[depth > 4.0] = 0
        return depth

    def process_mask(self, mask):
        return mask.astype(bool)

    def gate_depth_by_mask(self, depth):
        """Zero depth outside the dilated SAM2 mask. Falls back to ungated depth
        if the mask is missing, empty, stale, or shape-mismatched."""
        # missing
        if not self.use_mask_gating or self.latest_mask is None:
            return depth

        # stale
        if self.latest_mask_stamp is not None:
            staleness = (self.get_clock().now() -
                         self.latest_mask_stamp).nanoseconds * 1e-9
            if staleness > self.mask_gating_max_staleness_sec:
                self.get_logger().warn(
                    f"Mask stale ({staleness * 1000:.0f} ms) — skipping gating",
                    throttle_duration_sec=2.0)
                return depth

        # shape-mismatched
        mask = self.latest_mask
        if mask.shape != depth.shape:
            self.get_logger().warn(
                f"Mask shape {mask.shape} != depth {depth.shape} — skipping gating",
                throttle_duration_sec=2.0)
            return depth

        # empty
        mask_bool = mask > 0
        if int(mask_bool.sum()) < self.mask_gating_min_pixels:
            self.get_logger().warn("Mask near-empty — skipping gating",
                                   throttle_duration_sec=2.0)
            return depth

        if self.mask_gating_dilate_px > 0:
            k = 2 * self.mask_gating_dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0

        gated = depth.copy()
        gated[~mask_bool] = 0
        return gated

    # ---------- constant-velocity prior ----------

    def apply_cv_prior(self):
        """Seed FoundationPose.pose_last with a constant-velocity SE(3)
        extrapolation of the last two poses (ob_in_cam, so pred = delta @ curr)."""
        if not self.use_cv_prior:
            return

        pose_last = self.FPModel.pose_last
        if pose_last is None or self.pose_last_prev is None:
            return

        prev = self.pose_last_prev
        curr = pose_last.detach().cpu().numpy().reshape(4, 4)
        delta = curr @ np.linalg.inv(prev)
        pred = delta @ curr

        translation_step = np.linalg.norm(pred[:3, 3] - curr[:3, 3])
        if translation_step > self.max_translation_step:
            self.get_logger().warn(
                f"CV prior jump {translation_step*100:.1f} cm too large — skipping",
                throttle_duration_sec=2.0)
            return

        # Re-orthonormalize rotation against numerical drift.
        u, _, vt = np.linalg.svd(pred[:3, :3])
        pred[:3, :3] = u @ vt

        import torch
        self.FPModel.pose_last = torch.as_tensor(pred,
                                                 dtype=pose_last.dtype,
                                                 device=pose_last.device)

    def update_cv_prior_history(self):
        if not self.use_cv_prior:
            return
        pose_last = self.FPModel.pose_last
        if pose_last is None:
            return
        self.pose_last_prev = pose_last.detach().cpu().numpy().reshape(4, 4)

    # ---------- publish-time smoothing ----------

    def apply_pose_ema(self, pose: np.ndarray) -> np.ndarray:
        """Smooth the pose that actually gets published, over the last
        `pose_ema_window` frames. See this class's own __init__ comment for
        why this never writes back into FPModel.pose_last.

        Quaternion averaging here is the standard small-angle approximation
        (flip to the same hemisphere, blend linearly, renormalise) -- exact
        for identical quaternions, a good approximation for the few-degree
        jitter this is meant to smooth, not a real fix for a discrete flip
        (canonicalize_pose already runs first, so this should never see one)."""
        if self.pose_ema_window <= 1:
            return pose

        pos = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()  # xyzw

        if self.pose_ema_pos is None:
            self.pose_ema_pos = pos.copy()
            self.pose_ema_quat = quat.copy()
        else:
            a = self.pose_ema_alpha
            self.pose_ema_pos = a * pos + (1.0 - a) * self.pose_ema_pos
            # Quaternion double-cover: q and -q are the same rotation, but
            # blend to a near-zero vector if their signs disagree.
            if np.dot(quat, self.pose_ema_quat) < 0.0:
                quat = -quat
            blended = a * quat + (1.0 - a) * self.pose_ema_quat
            norm = np.linalg.norm(blended)
            self.pose_ema_quat = blended / norm if norm > 1e-9 else quat

        smoothed = np.eye(4)
        smoothed[:3, :3] = R.from_quat(self.pose_ema_quat).as_matrix()
        smoothed[:3, 3] = self.pose_ema_pos
        return smoothed

    # ---------- auto-reset ----------

    def check_auto_reset(self,
                         pose: np.ndarray,
                         cam_K: np.ndarray,
                         observed_depth: np.ndarray = None,
                         rendered_depth=None) -> bool:
        """Dispatch to the configured reset criterion. Returns True if a reset
        was triggered. Depth-residual mode catches rotation drift that leaves
        the centroid in place; centroid mode is the translation-only fallback."""

        self.last_reset_reason = ""

        # Plane violation is checked first, independently of
        # use_auto_reset, so it can be A/B tested on its own.
        if self.check_reset_plane(pose):
            return True

        if not self.use_auto_reset:
            return False

        if self.use_depth_residual_reset:
            return self._check_reset_depth_residual(pose, cam_K, rendered_depth,
                                                    observed_depth)

        return self._check_reset_centroid(pose, cam_K)

    def _centroid_distance(self, pose: np.ndarray, cam_K: np.ndarray):
        """Pixel distance between the SAM2 mask centroid and the projected FP
        3D center. Returns None if there is no usable mask or the object is
        behind the camera."""

        if self.latest_mask is None:
            return None

        mask_bool = self.latest_mask > 0
        if int(mask_bool.sum()) < self.mask_gating_min_pixels:
            return None

        ys, xs = np.nonzero(mask_bool)
        mask_u, mask_v = float(np.mean(xs)), float(np.mean(ys))

        t = pose[:3, 3]
        if t[2] <= 0.01:  # behind the camera
            return None

        fp_u = (cam_K[0, 0] * t[0] / t[2]) + cam_K[0, 2]
        fp_v = (cam_K[1, 1] * t[1] / t[2]) + cam_K[1, 2]

        return float(np.hypot(mask_u - fp_u, mask_v - fp_v))

    def _mask_iou(self, rendered_depth, mask_shape):
        """IoU (float) between FP's rendered silhouette and the current SAM2
        mask, or None if either is unavailable/shape-mismatched. Same
        computation `measure_all_criteria` already logs for telemetry --
        this is the one place it also gets to vote."""
        if (rendered_depth is None or self.latest_mask is None
                or mask_shape is None or self.latest_mask.shape != mask_shape):
            return None
        sil = rendered_depth > 0
        mask_bool = self.latest_mask > 0
        union = int(np.logical_or(sil, mask_bool).sum())
        if union == 0:
            return None
        inter = int(np.logical_and(sil, mask_bool).sum())
        return inter / union

    def _check_reset_depth_residual(self, pose: np.ndarray, cam_K: np.ndarray,
                                    rendered_depth, observed_depth) -> bool:
        """Combined reset: depth residual (rotation-sensitive), mask IoU
        (shape-sensitive), OR centroid distance (gross divergence safety net).
        Resets when any votes drift for `auto_reset_patience` frames.

        The occlusion guard silences the DEPTH and IoU votes together (a
        briefly hidden but correct pose reads high residual / low IoU purely
        from the missing pixels, same failure mode for both -- SAM2's mask
        only covers what's visible, while FP's rendered silhouette assumes the
        whole object is there). The centroid net stays active regardless, so a
        lost object that drifted away -- which makes both silhouette-based
        votes read n/a or occluded -- is still caught. Occlusion is only
        trusted when the centroid agrees the object is roughly where FP thinks
        it is."""
        drift_vote = False
        reasons = []

        # (A) Centroid safety net -- catches gross divergence even when the
        # depth residual is n/a (object moved off the rendered silhouette).
        cdist = self._centroid_distance(pose, cam_K)
        centroid_far = cdist is not None and cdist > self.max_center_dist_px
        if centroid_far:
            drift_vote = True
            reasons.append(f"center {cdist:.0f}px")

        # (B) Depth residual -- rotation-sensitive. Computed once and shared
        # with the IoU vote below: both are silhouette-based, so they share
        # the same occlusion read (IoU has no independent signal of its own).
        residual, trust_occlusion = None, False
        if rendered_depth is not None and observed_depth is not None:
            residual, n_covis, n_occ = self.compute_depth_residual(
                rendered_depth, observed_depth)
            occ_frac = (n_occ / n_covis) if n_covis > 0 else 0.0
            heavy_occ = occ_frac > self.depth_residual_heavy_occ_frac
            # Trust "heavy occlusion -> hold" only if the centroid agrees the
            # object is still in place; otherwise it is loss, not occlusion.
            trust_occlusion = heavy_occ and not centroid_far
            if residual is not None:
                if trust_occlusion:
                    pass  # genuinely occluded, do not vote from depth
                elif residual > self.depth_residual_thresh_m:
                    drift_vote = True
                    reasons.append(f"depth {residual*1000:.0f}mm")

        # (C) Mask IoU -- shape-sensitive, catches a pose that is the right
        # distance away but the wrong silhouette (e.g. tracking the robot arm
        # instead of the object -- depth residual alone can miss this if the
        # arm happens to sit at a similar depth).
        if self.use_iou_reset and not trust_occlusion:
            mask_shape = observed_depth.shape if observed_depth is not None else None
            iou = self._mask_iou(rendered_depth, mask_shape)
            if iou is not None and iou < self.iou_reset_thresh:
                drift_vote = True
                reasons.append(f"IoU {iou:.2f}")

        if drift_vote:
            self.drift_counter += 1
        else:
            self.drift_counter = max(0, self.drift_counter - 1)

        if self.drift_counter >= self.auto_reset_patience:
            self.last_reset_reason = "; ".join(reasons)
            self.get_logger().warn(
                f"Tracking lost ({', '.join(reasons)}). Auto-reset triggered.")
            self.is_object_registered = False
            self.drift_counter = 0
            return True

        return False

    def _check_reset_centroid(self, pose: np.ndarray,
                              cam_K: np.ndarray) -> bool:
        """
        Detects spatial drift by comparing the SAM2 mask centroid with the
        2D projection of the FoundationPose 3D center. Triggers a reset if lost.
        Returns True if a reset was triggered, False otherwise.
        """
        dist = self._centroid_distance(pose, cam_K)
        if dist is None:
            return False

        # Update drift counter
        if dist > self.max_center_dist_px:
            self.drift_counter += 1
        else:
            self.drift_counter = max(0, self.drift_counter - 1)

        # Trigger reset if patience is exceeded
        if self.drift_counter >= self.auto_reset_patience:
            self.last_reset_reason = f"center={dist:.0f}px"
            self.get_logger().warn(
                f"Tracking lost (Center dist: {dist:.1f}px). Auto-reset triggered."
            )
            self.is_object_registered = False
            self.drift_counter = 0
            return True

        return False

    # ---------- telemetry ----------

    LOG_COLUMNS = [
        "t_sec", "frame", "state", "track_ms",
        # raw tracked pose -- everything else in this row (reset votes,
        # canon, plane_*) was measured against THIS, not the smoothed one
        "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        # actually published pose, after EMA smoothing (pose_ema_window) --
        # blank on a reset row, since nothing publishes that frame. Diff
        # against tx/ty/tz/qx/qy/qz/qw above to see what smoothing changed
        "tx_pub", "ty_pub", "tz_pub", "qx_pub", "qy_pub", "qz_pub", "qw_pub",
        # table-plane criterion (measured BEFORE any correction)
        "plane_float_m", "plane_tilt_deg",
        # correction actually applied this frame (correct mode only)
        "corr_float_mm", "corr_tilt_deg",
        # centroid criterion
        "centroid_px",
        # silhouette-vs-SAM2 criterion
        "iou", "sil_px", "mask_px", "mask_age_s",
        # depth-residual criterion
        "depth_res_m", "n_covisible", "occ_frac",
        # reset bookkeeping
        "canon", "drift_counter", "plane_counter", "reset", "reset_reason",
    ]

    def _open_log(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.log_dir, f"fp_track_{stamp}.csv")
            self.log_file = open(path, "w", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(self.LOG_COLUMNS)
            self.log_file.flush()
            self.get_logger().info(f"Telemetry log: {path}")
        except Exception as e:
            self.get_logger().warn(f"Could not open telemetry log: {e}")
            self.log_writer = None

    def _log_row(self, **kw):
        """Append one row. Never allowed to disturb tracking."""
        if self.log_writer is None:
            return
        try:
            self.log_writer.writerow(
                ["" if kw.get(c) is None else kw.get(c, "")
                 for c in self.LOG_COLUMNS])
            self.log_rows += 1
            if self.log_rows % 20 == 0:  # survive a kill without losing much
                self.log_file.flush()
        except Exception as e:
            self.get_logger().warn(f"Telemetry write failed: {e}",
                                   throttle_duration_sec=5.0)

    def measure_all_criteria(self, pose, cam_K, observed_depth, rendered_depth):
        """Evaluate every drift criterion, independent of which one is enabled.
        Returns a dict of raw measurements -- no thresholds applied here."""
        out = {
            "centroid_px": None, "iou": None, "sil_px": None,
            "mask_px": None, "mask_age_s": None,
            "depth_res_m": None, "n_covisible": None, "occ_frac": None,
        }

        cdist = self._centroid_distance(pose, cam_K)
        if cdist is not None:
            out["centroid_px"] = round(cdist, 2)

        if self.latest_mask is not None:
            mask_bool = self.latest_mask > 0
            out["mask_px"] = int(mask_bool.sum())
            if self.latest_mask_stamp is not None:
                age = (self.get_clock().now() -
                       self.latest_mask_stamp).nanoseconds * 1e-9
                out["mask_age_s"] = round(age, 3)

        if rendered_depth is not None:
            sil = rendered_depth > 0
            out["sil_px"] = int(sil.sum())
            if (self.latest_mask is not None
                    and self.latest_mask.shape == sil.shape):
                mask_bool = self.latest_mask > 0
                union = int(np.logical_or(sil, mask_bool).sum())
                if union > 0:
                    inter = int(np.logical_and(sil, mask_bool).sum())
                    out["iou"] = round(inter / union, 4)
            if observed_depth is not None:
                res, n_covis, n_occ = self.compute_depth_residual(
                    rendered_depth, observed_depth)
                out["n_covisible"] = n_covis
                out["occ_frac"] = round(n_occ / max(1, n_covis), 4)
                if res is not None:
                    out["depth_res_m"] = round(res, 5)
        return out

    # ---------- symmetry canonicalisation ----------

    def _detect_symmetries(self):
        """Find 180 deg rotations that map the mesh onto itself, tested about the
        oriented-bounding-box axes (the object's own principal directions, which
        need not match the mesh file's axes).

        Symmetry is measured by nearest-neighbour distance rather than voxel
        overlap: mesh vertices are not uniformly sampled, so an occupancy test
        scores vertex placement as much as shape. Observed separation is wide --
        symmetric ~0.01 of the diameter, asymmetric >0.03 -- so the threshold is
        not delicate."""
        verts = np.asarray(self.object_mesh.vertices, dtype=np.float64)
        if len(verts) > 20000:
            verts = verts[np.random.default_rng(0).choice(
                len(verts), 20000, replace=False)]
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        local = verts - center
        diameter = float(np.linalg.norm(local.max(axis=0) - local.min(axis=0)))
        if diameter < 1e-6:
            return []

        tree = cKDTree(local)
        axes = self.to_origin[:3, :3]  # OBB axes expressed in the mesh frame
        found = []
        for i in range(3):
            axis = axes[i] / np.linalg.norm(axes[i])
            rot = 2.0 * np.outer(axis, axis) - np.eye(3)  # 180 deg about axis
            dist, _ = tree.query(local @ rot.T)
            score = float(np.percentile(dist, 95)) / diameter
            self.get_logger().info(
                f"Symmetry check: 180deg about {np.round(axis, 3)} -> "
                f"residual {score:.4f} of diameter"
                f"{'  [SYMMETRY]' if score < self.symmetry_tol else ''}")
            if score < self.symmetry_tol:
                to_c = np.eye(4)
                to_c[:3, 3] = center
                from_c = np.eye(4)
                from_c[:3, 3] = -center
                tf = np.eye(4)
                tf[:3, :3] = rot
                found.append(to_c @ tf @ from_c)  # symmetry about the centroid
        self.get_logger().info(f"Usable symmetries found: {len(found)}")
        return found

    def _apply_symmetry(self, pose: np.ndarray, tf: np.ndarray) -> np.ndarray:
        """Re-express `pose` through a mesh self-symmetry, updating
        FoundationPose's own state too. `pose_last @ S` renders exactly the same
        image (S maps the mesh onto itself), so the tracker carries on in the new
        representation instead of drifting back to the old one next frame."""
        pose_last = self.FPModel.pose_last
        if pose_last is not None:
            updated = pose_last.detach().cpu().numpy().reshape(4, 4) @ tf
            self.FPModel.pose_last = torch.as_tensor(updated,
                                                     dtype=pose_last.dtype,
                                                     device=pose_last.device)
        return pose @ tf

    def _report_axis_alignment(self, pose: np.ndarray):
        """Log how each object axis lines up with the table normal at rest.
        Purely diagnostic, but it is the number that explains which axis is
        physically 'up' for this mesh -- worth having in the log rather than
        inferred by hand."""
        if self.plane_n is None:
            return
        axes = self.to_origin[:3, :3]
        for i in range(3):
            axis = axes[i] / np.linalg.norm(axes[i])
            align = float(np.dot(pose[:3, :3] @ axis, self.plane_n))
            self.get_logger().info(
                f"Axis alignment at rest: object axis {np.round(axis, 3)} "
                f"-> {align:+.3f} along table normal")

    def canonicalize_pose(self, pose: np.ndarray) -> np.ndarray:
        """Keep the pose in the representation the rest reference was captured
        in, by choosing the mesh symmetry that minimises plane tilt.

        Minimising the very quantity the reset criterion measures is what makes
        this safe: canonicalisation can never "fix" a pose into a state the tilt
        check then flags, which is exactly how the earlier named-axis version
        produced frames that were canonicalised and 176 deg tilted at once."""
        self.canon_applied = False
        if (not self.canonicalize or self.plane_n is None
                or self.plane_u_ref is None or not self.symmetry_tfs):
            return pose

        metrics = self.plane_metrics(pose)
        if metrics is None:
            return pose
        best_tilt = metrics[1]

        for tf in self.symmetry_tfs:
            candidate = pose @ tf
            cand_metrics = self.plane_metrics(candidate)
            if cand_metrics is None:
                continue
            if cand_metrics[1] < best_tilt - self.canonical_switch_margin_deg:
                pose = self._apply_symmetry(pose, tf)
                best_tilt = cand_metrics[1]
                self.canon_applied = True

        return pose

    # ---------- table-plane constraint ----------

    def _fit_table_plane(self, depth, mask, cam_K):
        """Fit the table plane by RANSAC on the scene point cloud, excluding the
        object itself."""
        try:
            valid = depth > 0.001
            if mask is not None and mask.shape == depth.shape:
                # Drop the object (generously) so it is not fitted as table.
                k = 2 * 20 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                grown = cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0
                valid &= ~grown

            pts = depth2xyzmap(depth, cam_K)[valid].reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 500:
                self.get_logger().warn(
                    f"Plane fit skipped: only {len(pts)} scene points")
                return
            if len(pts) > 60000:
                pts = pts[np.random.default_rng(0).choice(
                    len(pts), 60000, replace=False)]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            model, inliers = pcd.segment_plane(
                distance_threshold=self.plane_fit_thresh_m,
                ransac_n=3,
                num_iterations=500)
            n = np.asarray(model[:3], dtype=np.float64)
            d = float(model[3])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                self.get_logger().warn("Plane fit produced a degenerate normal")
                return
            n, d = n / norm, d / norm
            # The camera sits above the table, and its origin's signed distance
            # is exactly d -- so flip the plane until d is positive. "Above the
            # table" is then the positive side for every point.
            if d < 0:
                n, d = -n, -d

            self.plane_n, self.plane_d = n, d
            self.get_logger().info(
                f"Table plane fit: normal={np.round(n, 3)}, "
                f"{len(inliers)}/{len(pts)} inliers")
        except Exception as e:  # never let plane setup break registration
            self.get_logger().warn(f"Plane fit failed: {e}")

    def _capture_rest_reference(self, pose):
        """Record what "at rest" looks like: which object-frame axis points
        along the table normal, and the resting height. Defining the reference
        from the pose itself means tilt reads 0 at registration for any mesh --
        no assumption about which axis is "up" -- and a constant bias in the
        mesh or depth image cannot by itself raise a violation."""
        # Where this reference comes from decides whether the published pose is
        # reproducible between runs. Deriving it from the registration pose
        # (R_reg^T @ n) makes tilt read 0 at registration, which is convenient --
        # but registration is exactly the coin flip we are trying to remove, so
        # each run then adopts whichever representation it happened to draw and
        # the published axis flips at random from run to run.
        #
        # With canonicalisation on, anchor it to the MESH instead: the
        # bounding-box axis most aligned with the table normal, keeping the sign
        # the mesh file gives it. The axis choice is identical in either
        # representation (a symmetry maps each axis to plus or minus itself, so
        # the alignment magnitudes are unchanged) and the sign never consults the
        # pose. Every run then converges on the same representation.
        self.plane_u_ref = None
        if self.canonicalize and self.symmetry_tfs:
            axes = self.to_origin[:3, :3]
            aligns = []
            for i in range(3):
                axis = axes[i] / np.linalg.norm(axes[i])
                aligns.append(
                    abs(float(np.dot(pose[:3, :3] @ axis, self.plane_n))))
            best = int(np.argmax(aligns))
            base = axes[best] / np.linalg.norm(axes[best])
            self.plane_u_ref = -base if self.canonical_flip else base
            self.get_logger().info(
                f"Rest reference anchored to mesh axis {np.round(base, 3)} "
                f"(alignments {np.round(aligns, 3)}, flip={self.canonical_flip})")
        else:
            self.plane_u_ref = pose[:3, :3].T @ self.plane_n

        self.plane_rest_offset = 0.0
        metrics = self.plane_metrics(pose)
        if metrics is not None:
            self.plane_rest_offset = metrics[0]
            self.get_logger().info(
                f"Rest reference captured: offset="
                f"{self.plane_rest_offset * 1000:.1f} mm, "
                f"tilt={metrics[1]:.1f} deg")

    def plane_metrics(self, pose: np.ndarray):
        """Return (float_height_m, tilt_deg), or None if the plane is not known.

        float_height is how far the object's centroid sits above its resting
        height (negative = sunk into the table). The centroid, not the lowest
        vertex, is used on purpose: rotating an object moves its lowest vertex,
        which would make the height reading rise and fall with tilt and couple
        the two thresholds together. Measuring the centroid keeps height purely
        translational, so the tilt bound can stay loose (for flat meshes like a
        banana) while the height bound stays tight.
        """
        if self.plane_n is None or self.plane_u_ref is None:
            return None
        rot, trans = pose[:3, :3], pose[:3, 3]
        centroid = rot @ self.plane_pts.mean(axis=0) + trans
        height = float(centroid @ self.plane_n + self.plane_d)
        up_now = rot @ self.plane_u_ref
        cos = float(np.clip(np.dot(up_now, self.plane_n), -1.0, 1.0))
        return height - self.plane_rest_offset, float(np.degrees(np.arccos(cos)))

    def check_reset_plane(self, pose: np.ndarray) -> bool:
        """Reset when the object has been off the table -- floating, sunk, or
        tilted -- for `plane_patience` frames. Uses its own counter so it can be
        tuned and A/B tested independently of the other criteria."""
        if self.plane_mode == "off":
            return False
        metrics = self.plane_metrics(pose)
        if metrics is None:
            return False
        float_h, tilt = metrics

        violated = (abs(float_h) > self.plane_max_float_m
                    or tilt > self.plane_max_tilt_deg)
        if violated:
            self.plane_violation_counter += 1
        else:
            self.plane_violation_counter = max(
                0, self.plane_violation_counter - 1)

        if self.plane_violation_counter >= self.plane_patience:
            self.last_reset_reason = (f"plane float={float_h * 1000:+.0f}mm "
                                      f"tilt={tilt:.0f}deg")
            self.get_logger().warn(
                f"Off the table (float {float_h * 1000:+.0f}mm, "
                f"tilt {tilt:.0f}deg). Auto-reset triggered.")
            self.is_object_registered = False
            self.plane_violation_counter = 0
            return True
        return False

    def apply_plane_correction(self, pose: np.ndarray) -> np.ndarray:
        """Project the pose back onto the table and feed the correction back
        into FoundationPose's own state.

        Writing the result into `pose_last` is the point: tracking is
        frame-to-frame, so an uncorrected drift is the starting guess for the
        next frame and compounds. Re-anchoring every frame means error simply
        cannot accumulate along the constrained freedoms."""
        metrics = self.plane_metrics(pose)
        if metrics is None:
            return pose

        corrected = pose.copy()
        n = self.plane_n

        # 1) Tilt: rotate by the minimal rotation that brings the reference axis
        # back onto the normal, about the object's own center so it does not
        # translate as a side effect.
        if self.plane_correct_tilt:
            up_now = corrected[:3, :3] @ self.plane_u_ref
            axis = np.cross(up_now, n)
            sin_a = float(np.linalg.norm(axis))
            if sin_a > 1e-8:
                angle = float(np.arctan2(sin_a, float(np.dot(up_now, n))))
                rot_c = R.from_rotvec(axis / sin_a * angle).as_matrix()
                center = corrected[:3, 3]
                tf = np.eye(4)
                tf[:3, :3] = rot_c
                tf[:3, 3] = center - rot_c @ center
                corrected = tf @ corrected

        # 2) Height: slide along the normal until the lowest vertex is back at
        # its resting height.
        if self.plane_correct_float:
            metrics = self.plane_metrics(corrected)
            if metrics is not None:
                tf = np.eye(4)
                tf[:3, 3] = -metrics[0] * n
                corrected = tf @ corrected

        # The correction is a left-multiply in the camera frame, so it applies
        # unchanged to FoundationPose's internal pose convention.
        delta = corrected @ np.linalg.inv(pose)
        pose_last = self.FPModel.pose_last
        if pose_last is not None:
            updated = delta @ pose_last.detach().cpu().numpy().reshape(4, 4)
            u, _, vt = np.linalg.svd(updated[:3, :3])  # guard against drift
            updated[:3, :3] = u @ vt
            self.FPModel.pose_last = torch.as_tensor(updated,
                                                     dtype=pose_last.dtype,
                                                     device=pose_last.device)
        return corrected

    # ---------- FP mesh rendering + depth residual ----------

    def render_fp_depth(self, pose: np.ndarray, cam_K: np.ndarray, H: int,
                        W: int):
        """Render the FP mesh at the tracked pose and return its depth map
        (float32, meters; 0 = background). Uses the SAME centered mesh_tensors
        and pose_last convention the refiner compares against internally, so it
        reflects exactly what FoundationPose "sees" as the object. Returns None
        on failure so the tracking loop is never broken by rendering."""
        try:
            ob_in_cam = torch.as_tensor(pose.reshape(1, 4, 4),
                                        device="cuda",
                                        dtype=torch.float)
            _, depth_r, _ = nvdiffrast_render(
                K=cam_K,
                H=H,
                W=W,
                ob_in_cams=ob_in_cam,
                glctx=self.glctx,
                mesh_tensors=self.FPModel.mesh_tensors,
            )
            return depth_r[0].detach().cpu().numpy()  # (H, W), meters; 0 = bg
        except Exception as e:  # rendering must never break the tracking loop
            self.get_logger().warn(f"FP depth render failed: {e}",
                                   throttle_duration_sec=2.0)
            return None

    def compute_depth_residual(self, rendered_depth: np.ndarray,
                               observed_depth: np.ndarray):
        """Median |z_render - z_obs| over co-visible, non-occluded pixels.
        A pixel is co-visible when both rendered and observed depth are valid;
        an occluder is detected when the observation is clearly closer than the
        rendered object surface, and those pixels are excluded from the residual.
        Returns (residual_m or None, n_covisible, n_occluded)."""
        covis = (rendered_depth > 0) & (observed_depth > 0)
        n_covis = int(covis.sum())
        if n_covis < self.depth_residual_min_covisible_px:
            return None, n_covis, 0

        diff = observed_depth[covis] - rendered_depth[covis]
        occluded = diff < -self.depth_residual_occlusion_margin_m
        n_occ = int(occluded.sum())

        inliers = ~occluded
        if int(inliers.sum()) < self.depth_residual_min_covisible_px:
            return None, n_covis, n_occ

        residual = float(np.median(np.abs(diff[inliers])))
        return residual, n_covis, n_occ

    # ---------- debug visualization ----------

    def publish_debug_viz(self, rgb: np.ndarray, pose: np.ndarray,
                          cam_K: np.ndarray, observed_depth: np.ndarray,
                          rendered_depth):
        """Publish the FP silhouette (/fp_render_mask), an RGB overlay
        (/fp_debug_overlay) comparing the FP silhouette (green) with the SAM2
        mask (red), and a colorized depth-residual heatmap (/fp_depth_residual).
        Overlays centroid distance, mask IoU, and depth residual for diagnosis."""
        if rendered_depth is None:
            return

        H, W = rendered_depth.shape
        fp_sil = (rendered_depth > 0).astype(np.uint8) * 255
        fp_bool = fp_sil > 0

        # /fp_render_mask — raw FP silhouette
        try:
            self.render_mask_pub.publish(
                self.bridge.cv2_to_imgmsg(fp_sil, encoding="mono8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"render_mask publish failed: {e}",
                                   throttle_duration_sec=2.0)

        # /fp_debug_overlay — RGB with FP silhouette + SAM2 mask overlaid
        overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

        # Translucent green fill for the FP silhouette
        if fp_bool.any():
            tint = overlay.copy()
            tint[fp_bool] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.75, tint, 0.25, 0)
            cnts, _ = cv2.findContours(fp_sil, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

        # SAM2 mask outline in red + centroid distance / IoU text
        if self.latest_mask is not None and self.latest_mask.shape == (H, W):
            sam_bool = self.latest_mask > 0
            sam_u8 = sam_bool.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(sam_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)

            # FP projected 3D center (blue) — matches the centroid reset metric
            t = pose[:3, 3]
            if t[2] > 0.01:
                fp_u = int((cam_K[0, 0] * t[0] / t[2]) + cam_K[0, 2])
                fp_v = int((cam_K[1, 1] * t[1] / t[2]) + cam_K[1, 2])
                cv2.circle(overlay, (fp_u, fp_v), 5, (255, 0, 0), -1)

                if sam_bool.any():
                    ys, xs = np.nonzero(sam_bool)
                    sam_u, sam_v = int(np.mean(xs)), int(np.mean(ys))
                    cv2.circle(overlay, (sam_u, sam_v), 5, (0, 0, 255), -1)
                    cv2.line(overlay, (fp_u, fp_v), (sam_u, sam_v),
                             (255, 255, 0), 1)
                    dist = float(np.hypot(fp_u - sam_u, fp_v - sam_v))
                    inter = int(np.logical_and(fp_bool, sam_bool).sum())
                    union = int(np.logical_or(fp_bool, sam_bool).sum())
                    iou = inter / union if union > 0 else 0.0
                    cv2.putText(overlay,
                                f"center_dist={dist:.1f}px  IoU={iou:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

        # Depth residual text (independent of SAM2). Always print covis / occ%,
        # even on n/a, so the two failure causes are distinguishable:
        #   - low covis  : n_covis below the min (few co-visible pixels at all)
        #   - heavy excl : n_covis fine but most pixels excluded as "occluded"
        #                  (broad exclusion -> a pose error masquerading as occ)
        residual, n_covis, n_occ = self.compute_depth_residual(
            rendered_depth, observed_depth)
        occ_frac = n_occ / max(1, n_covis)
        if residual is not None:
            txt = (f"depth_res={residual*1000:.1f}mm  "
                   f"occ={occ_frac*100:.0f}%  covis={n_covis}")
        else:
            txt = (f"depth_res=n/a  covis={n_covis}  occ={occ_frac*100:.0f}%  "
                   f"(min covis={self.depth_residual_min_covisible_px})")
        cv2.putText(overlay, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        # Table-plane diagnostics (independent of SAM2 and of depth noise)
        plane = self.plane_metrics(pose)
        if plane is not None:
            cv2.putText(
                overlay,
                f"float={plane[0] * 1000:+.0f}mm  tilt={plane[1]:.0f}deg",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2,
                cv2.LINE_AA)

        try:
            self.debug_overlay_pub.publish(
                self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"debug_overlay publish failed: {e}",
                                   throttle_duration_sec=2.0)

        # /fp_depth_residual — colorized |z_render - z_obs| over co-visible px.
        # Non co-visible = black; excluded occluders = white. Display is capped
        # at 3x the reset threshold so the useful range is well spread.
        covis = (rendered_depth > 0) & (observed_depth > 0)
        diff = observed_depth - rendered_depth
        cap = max(self.depth_residual_thresh_m * 3.0, 1e-6)
        norm = np.clip(np.abs(diff) / cap, 0.0, 1.0)
        heat = np.zeros((H, W), np.uint8)
        heat[covis] = (norm[covis] * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_color[~covis] = (0, 0, 0)
        occluded = covis & (diff < -self.depth_residual_occlusion_margin_m)
        heat_color[occluded] = (255, 255, 255)
        try:
            self.depth_residual_pub.publish(
                self.bridge.cv2_to_imgmsg(heat_color, encoding="bgr8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"depth_residual publish failed: {e}",
                                   throttle_duration_sec=2.0)


def main(args=None):
    import threading
    rclpy.init(args=args)
    node = FoundationPoseROS2()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if node.visualize and node.latest_vis_img is not None:
                cv2.imshow("Pose Visualization", node.latest_vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
