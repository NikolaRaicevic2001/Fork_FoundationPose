#!/usr/bin/env python3
"""
SAM2 ROS2 node.
Subscribes to an RGB camera topic, runs SAM2 segmentation/tracking,
and publishes the object mask on /sam2_mask.

Default prompt method is "user_select": on the first frame (and after each
reset) a matplotlib window pops up and you click once on the object to track.

The "mesh" / "text" prompt methods are kept but their heavy dependencies
(Grounding DINO, GPT-4o via mesh_to_bbox) are imported lazily inside the
relevant methods, so the node runs without them as long as you use
"user_select" / "hardcoded".
"""

import numpy as np
import rclpy
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Tuple
from cv_bridge import CvBridge
from PIL import Image

from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image as ROSImage

from sam2_model import SAM2Model

import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def rgb_to_pil(rgb_image: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb_image)


def get_user_point(rgb_image: np.ndarray, title: str) -> Tuple[int, int]:
    # Get prompt as click via a matplotlib window
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.imshow(rgb_image)
    plt.axis("off")
    points = plt.ginput(1)  # get one click
    plt.close()

    x, y = int(points[0][0]), int(points[0][1])
    return x, y


def draw_prompts(image: np.ndarray, prompts: dict) -> np.ndarray:
    _H, _W, C = image.shape
    assert C == 3, f"{C}"
    image = image.copy()

    RED = [255, 0, 0]
    BLUE = [0, 0, 255]

    if prompts["box"] is not None:
        x_min, y_min, x_max, y_max = (
            int(prompts["box"][0]),
            int(prompts["box"][1]),
            int(prompts["box"][2]),
            int(prompts["box"][3]),
        )
        BOX_THICKNESS = 2
        BOX_COLOR = BLUE
        image[y_min:y_min + BOX_THICKNESS, x_min:x_max] = BOX_COLOR  # Top
        image[y_max - BOX_THICKNESS:y_max, x_min:x_max] = BOX_COLOR  # Bottom
        image[y_min:y_max, x_min:x_min + BOX_THICKNESS] = BOX_COLOR  # Left
        image[y_min:y_max, x_max - BOX_THICKNESS:x_max] = BOX_COLOR  # Right

    if prompts["points"] is not None:
        points = prompts["points"]
        labels = prompts["labels"]
        N_points = points.shape[0]
        for i in range(N_points):
            point = points[i]
            label = labels[i]
            x, y = int(point[0]), int(point[1])
            POSITIVE_COLOR = BLUE
            NEGATIVE_COLOR = RED
            if label == 1:
                image[y - 5:y + 5, x - 5:x + 5] = POSITIVE_COLOR
            elif label == 0:
                image[y - 5:y + 5, x - 5:x + 5] = NEGATIVE_COLOR
            else:
                raise ValueError(f"Unknown label: {label}")
    return image


class SAM2RosNode(Node):

    def __init__(self):
        super().__init__("sam2_ros_node")

        # SAM2 model wrapper (device selection etc. handled inside)
        self.sam2_model = SAM2Model()

        self.rgb_image: Optional[np.ndarray] = None
        self.is_mask_initialized = False
        self.prompts: Optional[dict] = None
        self.bridge = CvBridge()

        # Cache generated text prompt (only used by mesh/text methods)
        self.cached_generated_text_prompt: Optional[str] = None

        # Camera selection
        self.declare_parameter("camera", "realsense")
        camera = self.get_parameter("camera").get_parameter_value().string_value
        self.get_logger().info(f"Using camera: {camera}")
        if camera == "zed":
            image_sub_topic = "/zed/zed_node/rgb/image_rect_color"
        elif camera == "realsense":
            image_sub_topic = "/camera/color/image_raw"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Prompt method selection (default: click to select)
        self.declare_parameter("prompt_method", "user_select")
        self.prompt_method = (self.get_parameter(
            "prompt_method").get_parameter_value().string_value)
        self.get_logger().info(f"Using prompt method: {self.prompt_method}")

        # Subscriptions
        self.create_subscription(ROSImage, image_sub_topic, self.image_callback,
                                 1)
        self.create_subscription(Int32, "/sam2_reset", self.reset_callback, 1)

        # Publishers
        self.mask_pub = self.create_publisher(ROSImage, "/sam2_mask", 1)
        self.mask_with_prompt_pub = self.create_publisher(
            ROSImage, "/sam2_mask_with_prompt", 1)
        self.num_mask_pixels_pub = self.create_publisher(
            Int32, "/sam2_num_mask_pixels", 1)

        # Loop frequency selection (default: 5.0 Hz)
        self.declare_parameter("frequency", 5.0)
        freq = self.get_parameter("frequency").get_parameter_value().double_value
        timer_period = 1.0 / freq
        self.get_logger().info(f"SAM2 ROS2 node ready, running at {freq} Hz...")

        # Main loop triggered by timer
        self.timer = self.create_timer(timer_period, self.run_once)
        self.get_logger().info("SAM2 ROS2 node ready, waiting for images...")

    # ---------- callbacks ----------

    def image_callback(self, data):
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

    def reset_callback(self, data: Int32):
        if data.data > 0:
            self.get_logger().info(
                "Reset triggered — will re-prompt and re-init")
            self.is_mask_initialized = False
        else:
            self.get_logger().info("Reset message with data <= 0, ignoring")

    # ---------- prompt generation ----------

    def generate_sam_prompts_from_mesh(self, rgb_image: np.ndarray,
                                       mesh_filepath: Path) -> Optional[dict]:
        # Lazy import: only needed for the mesh method (GPT-4o + Grounding DINO)
        from mesh_to_bbox import mesh_to_description

        if self.cached_generated_text_prompt is None:
            self.get_logger().info("Generating text prompt from mesh...")
            assert mesh_filepath.exists(), f"{mesh_filepath}"
            _, generated_text_prompt = mesh_to_description(
                mesh_filepath=mesh_filepath)
            self.get_logger().info(
                f"Generated text prompt: {generated_text_prompt}")
            self.cached_generated_text_prompt = generated_text_prompt
        else:
            self.get_logger().info(
                f"Using cached text prompt: {self.cached_generated_text_prompt}"
            )

        return self.generate_sam_prompts_from_text(
            rgb_image=rgb_image, text_prompt=self.cached_generated_text_prompt)

    def generate_sam_prompts_from_text(self, rgb_image: np.ndarray,
                                       text_prompt: str) -> Optional[dict]:
        # Lazy import: only needed for the text method (Grounding DINO)
        from mesh_to_bbox import generate_bbox

        pil_image = rgb_to_pil(rgb_image)
        try:
            bboxes, _, _ = generate_bbox(
                image=pil_image,
                text_prompt=text_prompt,
                grounding_model="gdino",
                gdino_1_5_api_token=None,
            )
            assert bboxes.shape == (1, 4), f"{bboxes.shape}"
            return {"points": None, "labels": None, "box": bboxes[0]}
        except ValueError as e:
            self.get_logger().error(f"No object found via text prompt: {e}")
            return None

    def get_user_prompt(
        self,
        rgb_image: np.ndarray,
        use_negative_prompt: bool = False,
        use_2_points: bool = False,
    ) -> dict:
        x, y = get_user_point(rgb_image=rgb_image,
                              title="Click on the object to track")
        self.get_logger().info(f"Clicked point: ({x}, {y})")

        if use_negative_prompt:
            neg_x, neg_y = get_user_point(
                rgb_image=rgb_image,
                title="Click a NEGATIVE point (background)")
            points = np.array([[x, y], [neg_x, neg_y]], dtype=np.float32)
            labels = np.array([1, 0], dtype=np.int32)
        elif use_2_points:
            x2, y2 = get_user_point(rgb_image=rgb_image,
                                    title="Click a SECOND point on the object")
            points = np.array([[x, y], [x2, y2]], dtype=np.float32)
            labels = np.array([1, 1], dtype=np.int32)
        else:
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

        return {"points": points, "labels": labels, "box": None}

    def generate_sam_prompts(self, rgb_image: np.ndarray) -> Optional[dict]:
        method = self.prompt_method

        if method == "mesh":
            self.declare_parameter("mesh_file", "")
            mesh_file = (self.get_parameter(
                "mesh_file").get_parameter_value().string_value)
            assert mesh_file, "mesh_file parameter required for mesh prompt method"
            prompts = self.generate_sam_prompts_from_mesh(
                rgb_image=rgb_image, mesh_filepath=Path(mesh_file))
        elif method == "text":
            self.declare_parameter("text_prompt", "red cup")
            text_prompt = (self.get_parameter(
                "text_prompt").get_parameter_value().string_value)
            self.get_logger().info(f"Using text prompt: {text_prompt}")
            prompts = self.generate_sam_prompts_from_text(
                rgb_image=rgb_image, text_prompt=text_prompt)
        elif method == "hardcoded":
            prompts = self.sam2_model.get_hardcoded_prompts()
        elif method in (
                "user_select",
                "user_select_with_negative",
                "user_select_with_2_points",
        ):
            prompts = self.get_user_prompt(
                rgb_image=rgb_image,
                use_negative_prompt=(method == "user_select_with_negative"),
                use_2_points=(method == "user_select_with_2_points"),
            )
        else:
            raise ValueError(f"Unknown prompt_method: {method}")

        if prompts is not None:
            self.validate_sam_prompts(prompts)
        return prompts

    @staticmethod
    def validate_sam_prompts(prompts: dict) -> dict:
        assert (prompts["points"] is None) == (prompts["labels"]
                                               is None), f"{prompts}"
        if prompts["points"] is not None:
            N = prompts["points"].shape[0]
            assert prompts["points"].shape == (N, 2), f"{prompts}"
            assert prompts["labels"].shape == (N,), f"{prompts}"
        if prompts["box"] is not None:
            assert prompts["box"].shape == (4,), f"{prompts}"
        return prompts

    # ---------- main loop ----------

    def run_once(self):
        if self.rgb_image is None:
            self.get_logger().warn("Waiting for the first image...",
                                   throttle_duration_sec=2.0)
            return

        if not self.is_mask_initialized:
            self._initialize()
        else:
            self._track()

    def _initialize(self):
        first_rgb_image = self.rgb_image.copy()
        self.prompts = self.generate_sam_prompts(rgb_image=first_rgb_image)

        if self.prompts is None:
            self.get_logger().error(
                "prompts is None (no object found). Will retry.")
            self.is_mask_initialized = False
            return

        self.sam2_model.predict(rgb_image=first_rgb_image,
                                first=True,
                                prompts=self.prompts)
        self.is_mask_initialized = True
        self.get_logger().info("Mask initialized")

    def _track(self):
        new_rgb_image = self.rgb_image.copy()
        mask = self.sam2_model.predict(rgb_image=new_rgb_image,
                                       first=False,
                                       prompts=None)
        assert mask.shape == new_rgb_image.shape, f"{mask.shape} != {new_rgb_image.shape}"

        num_mask_pixels = int((mask[..., 0] > 0).sum())
        MIN_MASK_PIXELS = 0
        if num_mask_pixels <= MIN_MASK_PIXELS:
            self.get_logger().warn(
                f"Mask is empty (num_pixels={num_mask_pixels}), re-init")
            self.is_mask_initialized = False
        else:
            self.is_mask_initialized = True

        self.num_mask_pixels_pub.publish(Int32(data=num_mask_pixels))

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="rgb8")
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(mask_msg)

        # Mask overlaid with the prompt (for debugging/visualization)
        if self.prompts is not None:
            mask_with_prompt = draw_prompts(image=mask.copy(),
                                            prompts=self.prompts)
            mwp_msg = self.bridge.cv2_to_imgmsg(mask_with_prompt,
                                                encoding="rgb8")
            mwp_msg.header.stamp = self.get_clock().now().to_msg()
            self.mask_with_prompt_pub.publish(mwp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SAM2RosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
