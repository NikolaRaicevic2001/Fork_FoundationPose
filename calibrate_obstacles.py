#!/usr/bin/env python3
"""
One-time obstacle-pose calibration: look up xarm_device -> obs_N_center
for each obstacle over a short window, average out detector jitter, and
write the result to a JSON file that run_real.py (a different Python
environment -- no rclpy there) reads at startup.

TF composes the whole chain automatically: extrinsics (fp_tf_broadcaster.py's
robot_base -> camera, from the real robot+camera mount calibration) + the
ArUco-detected tag pose + the fixed tag-to-box offset. This script only
ever reads xarm_device -> obs_N_center -- it has no idea, and does not
need to, that the chain has three hops underneath.

    python calibrate_obstacles.py [--out /tmp/obstacle_calibration.json]
        [--window-s 1.5] [--world-frame xarm_device]
"""

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

OBSTACLES = ("obs_1", "obs_2", "obs_3")


def se2_from_transform(t) -> np.ndarray:
    """[x, y, yaw] from a TransformStamped -- yaw about the table normal
    (z), same projection _lookup_object_se2 uses for the pushed block."""
    x = t.transform.translation.x
    y = t.transform.translation.y
    q = t.transform.rotation
    yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
    return np.array([x, y, yaw])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/obstacle_calibration.json")
    p.add_argument("--window-s", type=float, default=1.5,
                   help="collect samples for this long per obstacle, "
                        "then average -- damps detector jitter")
    p.add_argument("--world-frame", default="xarm_device")
    args = p.parse_args()

    rclpy.init()
    node = Node("obstacle_calibration")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa: F841 -- keeps the sub alive

    result = {}
    for name in OBSTACLES:
        target = f"{name}_center"
        samples = []
        t_end = time.time() + args.window_s
        print(f"[calibrate] {name}: sampling for {args.window_s}s...")
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                tf = buffer.lookup_transform(
                    args.world_frame, target, rclpy.time.Time())
                samples.append(se2_from_transform(tf))
            except (LookupException, ConnectivityException,
                    ExtrapolationException):
                pass

        if not samples:
            print(f"[calibrate] {name}: NO SAMPLES -- is the tag in view? "
                  f"Skipping, run_real.py will fall back to the MJCF default.")
            continue

        arr = np.stack(samples)
        # Yaw averaged via unit vectors (mean of raw angles breaks across
        # the +-pi wrap boundary; this doesn't).
        mean_xy = arr[:, :2].mean(axis=0)
        mean_yaw = np.arctan2(np.sin(arr[:, 2]).mean(), np.cos(arr[:, 2]).mean())
        std_xy = arr[:, :2].std(axis=0)
        result[name] = [float(mean_xy[0]), float(mean_xy[1]), float(mean_yaw)]
        print(f"[calibrate] {name}: n={len(samples)}  "
              f"xy=({mean_xy[0]:.4f}, {mean_xy[1]:.4f})  "
              f"yaw={np.degrees(mean_yaw):.1f}deg  "
              f"xy_std=({std_xy[0]*1000:.1f}, {std_xy[1]*1000:.1f})mm")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[calibrate] wrote {len(result)}/{len(OBSTACLES)} obstacles to {args.out}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
