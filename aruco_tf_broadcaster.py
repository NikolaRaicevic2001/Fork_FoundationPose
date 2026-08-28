#!/usr/bin/env python3
"""
TF broadcaster for the ArUco obstacle-pose pipeline.

Mirrors fp_tf_broadcaster.py's shape, but simpler: the camera extrinsics
(robot_base -> camera) are already on the TF tree once fp_tf_broadcaster.py
is running (real robot+camera mount calibration -- out of scope here), so
this only adds, per obstacle:

  1. dynamic:  camera -> obs_N_tag         (from /obstacle_pose/obs_N)
  2. static:   obs_N_tag -> obs_N_center   (the tag-to-box offset)

The offset is translation-only and IDENTICAL for all three obstacles,
despite obs_3's tag being glued on rotated 90deg from the other two on
the physical box: box_clutter_real.xml's geom `size` ordering was chosen
specifically to absorb that difference on the model side, so this side
never has to special-case obs_3.

Ros2Interface then looks up xarm_device -> obs_N_center exactly the way
it already looks up xarm_device -> fp_object_pose for the pushed block --
TF composes the whole chain (extrinsics + detected tag pose + fixed
offset) automatically.
"""

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# The tag sits centered on the box's top face, so the box center is
# straight down from the tag by half the box height (box_clutter_real.xml:
# geom size [.., .., 0.0298], i.e. half-height 0.0298m).
BOX_HALF_HEIGHT_M = 0.0298
OBSTACLES = ("obs_1", "obs_2", "obs_3")


class ArucoTFBroadcaster(Node):

    def __init__(self) -> None:
        super().__init__("aruco_tf_broadcaster")

        self.declare_parameter("camera_frame", "fp_camera_color_optical_frame")
        self.camera_frame = self.get_parameter("camera_frame").value

        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # StaticTransformBroadcaster.sendTransform, called once per
        # obstacle with a single TransformStamped each time, OVERWRITES
        # rather than accumulates -- only the last call's transform ends
        # up on /tf_static. All three must go out together, in one call.
        self.static_broadcaster.sendTransform(
            [self._tag_offset_transform(name) for name in OBSTACLES]
        )
        for name in OBSTACLES:
            self.create_subscription(
                PoseStamped, f"/obstacle_pose/{name}",
                lambda msg, name=name: self._pose_callback(msg, name), 1)

        self.get_logger().info(
            f"Bridging /obstacle_pose/{{{','.join(OBSTACLES)}}} -> TF, "
            f"off {self.camera_frame}")

    def _tag_offset_transform(self, name: str) -> TransformStamped:
        """obs_N_tag -> obs_N_center, static."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = f"{name}_tag"
        t.child_frame_id = f"{name}_center"
        t.transform.translation.z = -BOX_HALF_HEIGHT_M
        t.transform.rotation.w = 1.0  # identity -- see module docstring
        return t

    def _pose_callback(self, msg: PoseStamped, name: str) -> None:
        """camera -> obs_N_tag, re-broadcast from /obstacle_pose/obs_N."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = f"{name}_tag"
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main() -> None:
    rclpy.init()
    node = ArucoTFBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
