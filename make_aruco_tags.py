"""Generate 3 printable ArUco tags for the box_clutter_real obstacles.

Each tag gets a distinct ID from the same dictionary, so the perception
node can tell the 3 identical boxes apart by reading the ID off the tag --
the ID is decoded directly from the tag's black/white grid, not something
we choose per print beyond picking which of the dictionary's IDs to use.

Also saves an annotated preview of each tag showing its own local
+X/+Y/+Z, per OpenCV's ArUco pose convention (see the note printed at the
end) -- this convention is IDENTICAL for every ID in the dictionary; only
the printed black/white pattern (which encodes the ID) differs.

Needs opencv-contrib-python (the "aruco" submodule lives there, not in
plain opencv-python): pip install opencv-contrib-python

    python make_aruco_tags.py

Writes, for each of obs_1/obs_2/obs_3:
  aruco_tags/obs_N_id<ID>_print.png      -- clean, print this, stick it down
  aruco_tags/obs_N_id<ID>_preview.png    -- same tag with X/Y/Z labeled
"""

import os

import cv2
import numpy as np

# obs_name -> marker ID. The mapping itself is arbitrary -- change these if
# you like -- but whatever you pick here is what the perception node's own
# config must use too, since that's the only place "ID 2 means obs_2" is
# ever written down.
MARKER_IDS = {"obs_1": 1, "obs_2": 2, "obs_3": 3}

DICTIONARY = cv2.aruco.DICT_4X4_50  # 4x4-bit pattern, 50 possible IDs
MARKER_PIXELS = 800  # print resolution of the marker pattern itself
MARGIN_PIXELS = 40  # thin white margin around it, for detection contrast

OUT_DIR = "aruco_tags"


def _make_marker(aruco_dict, marker_id: int, side_pixels: int) -> np.ndarray:
    """The clean marker image: black-bordered pattern only, no margin."""
    try:
        return cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_pixels)
    except AttributeError:
        # OpenCV < 4.7 named this drawMarker; same output either way.
        return cv2.aruco.drawMarker(aruco_dict, marker_id, side_pixels)


def _add_margin(img: np.ndarray, margin: int) -> np.ndarray:
    """White border for print contrast -- not part of the encoded pattern."""
    return cv2.copyMakeBorder(
        img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
    )


def _make_preview(marker_img: np.ndarray) -> np.ndarray:
    """The same tag with its own +X (red) / +Y (green) / +Z (blue) labeled,
    drawn on the marker's CANONICAL as-generated orientation. If you
    physically rotate the tag when sticking it on the box, rotate this
    mental picture along with it -- the axes are fixed to the pattern, not
    to the page.
    """
    preview = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    h, w = preview.shape[:2]
    cx, cy = w // 2, h // 2
    arrow_len = w // 3

    # +X: right, red (BGR: 0,0,255)
    cv2.arrowedLine(preview, (cx, cy), (cx + arrow_len, cy), (0, 0, 255), 6,
                     tipLength=0.2)
    cv2.putText(preview, "+X", (cx + arrow_len + 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # +Y: up on the page == decreasing pixel row, green (BGR: 0,255,0)
    cv2.arrowedLine(preview, (cx, cy), (cx, cy - arrow_len), (0, 255, 0), 6,
                     tipLength=0.2)
    cv2.putText(preview, "+Y", (cx + 10, cy - arrow_len - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # +Z: out of the page, toward the viewer -- no arrow for that, a dot.
    cv2.circle(preview, (cx, cy), 14, (255, 0, 0), -1)
    cv2.putText(preview, "+Z (toward viewer)", (cx + 20, cy + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return preview


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)

    print("Dictionary: DICT_4X4_50 (4x4-bit pattern, 50 possible IDs)\n")
    for obs_name, marker_id in MARKER_IDS.items():
        marker = _make_marker(aruco_dict, marker_id, MARKER_PIXELS)

        print_img = _add_margin(marker, MARGIN_PIXELS)
        print_path = os.path.join(OUT_DIR, f"{obs_name}_id{marker_id}_print.png")
        cv2.imwrite(print_path, print_img)

        preview_img = _make_preview(marker)
        preview_path = os.path.join(
            OUT_DIR, f"{obs_name}_id{marker_id}_preview.png"
        )
        cv2.imwrite(preview_path, preview_img)

        print(f"{obs_name}  ->  marker ID {marker_id}  (DICT_4X4_50)")
        print(f"  print this:             {print_path}")
        print(f"  orientation reference:  {preview_path}")
        print()

    print("On every tag, holding the PRINTED image right-side up as generated:")
    print("  +X -> right")
    print("  +Y -> up")
    print("  +Z -> out of the page, toward whoever is looking at it")
    print("Same convention on all three -- only the black/white pattern")
    print("(which encodes the ID) differs between tags.")


if __name__ == "__main__":
    main()
