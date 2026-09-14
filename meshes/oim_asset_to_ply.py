"""Write a FoundationPose mesh from an Object-Informed-Manipulation-MJX asset.

    python meshes/oim_asset_to_ply.py <oim>/oim/models/xarm6_pusht_tabletop_real/assets \
        banana 0.0184 sugar_box 0.0247 coca_cola 0.0400 coffee_cup 0.0500 cup 0.0600

One `<name> <half_height>` pair per object; `half_height` is the entry's
value in `oim/objects/library.py`. Reads `<assets>/<name>_centered.obj` and
writes `meshes/<name>/<name>.ply` (plus `<name>.png` when the OBJ is photo
textured), so `--object <name>` on the planner side and `mesh_file` on
this side name the same thing.

FRAME. The OBJ is in the planner's body frame in metres: plan bounding box
centred on the origin, underside at z = 0. The PLY is that frame in
MILLIMETRES (fp_ros_node multiplies by 0.001) with the origin lifted to
`half_height` above the underside -- which is where the planner puts the
block body, and for a flat print is mid-thickness, the convention the
letter blocks already use. So the pose FoundationPose publishes IS the
body pose, `fp_origin_offset` stays (0, 0), and the interface's flip
symmetry logic keeps working. (For a tall object such as the bottle the
origin is NOT mid-height: it sits at the push height the planner uses,
0.04 m up a 0.215 m bottle. That is deliberate.)

COLOUR. The texture is looked for as `<assets>/<name>_texture.png` (what
`prepare_objects.py` writes), then `<assets>/../assets_original/<name>/
texture_map.png` (the YCB scans, whose UVs the centred OBJ keeps), then
whatever the OBJ's own material names. A photo is carried over as
`texture_u`/`texture_v` per vertex plus a `comment TextureFile
<name>.png` -- the layout the YCB meshes here ship in and what trimesh
reads back as a TextureVisuals with an image -- downsampled to at most
1024 px a side. A flat paint (every pixel the same) is written as
per-vertex `red green blue` instead, like the letters.
"""

import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


TEXTURE_MAX_PX = 1024


def _flat_colour(image: Image.Image):
    """The one RGB of a flat paint texture, else None."""
    rgb = image.convert("RGB")
    extrema = rgb.getextrema()
    if any(lo != hi for lo, hi in extrema):
        return None
    return tuple(int(lo) for lo, _ in extrema)


def _find_texture(assets_dir: str, name: str, mesh: trimesh.Trimesh):
    """The object's texture image, by the search order in the module note."""
    candidates = [
        os.path.join(assets_dir, f"{name}_texture.png"),
        os.path.join(assets_dir, "..", "assets_original", name,
                     "texture_map.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return Image.open(path), os.path.normpath(path)
    image = getattr(getattr(mesh.visual, "material", None), "image", None)
    return image, "OBJ material"


def write(assets_dir: str, name: str, half_height: float) -> str:
    """Write meshes/<name>/<name>.ply (+ .png) from <assets_dir>/<name>_centered.obj."""
    mesh = trimesh.load(
        os.path.join(assets_dir, f"{name}_centered.obj"),
        force="mesh", process=False,
    )
    lo, hi = mesh.bounds
    v = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
    v[:, 2] -= half_height * 1000.0
    n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    image, source = _find_texture(assets_dir, name, mesh)
    uv = getattr(mesh.visual, "uv", None)
    colour = _flat_colour(image) if image is not None else (200, 200, 200)
    textured = image is not None and colour is None and uv is not None
    if textured and max(image.size) > TEXTURE_MAX_PX:
        scale = TEXTURE_MAX_PX / max(image.size)
        image = image.resize(
            (round(image.size[0] * scale), round(image.size[1] * scale)),
            Image.LANCZOS)

    out_dir = os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.ply")

    header = ["ply", "format ascii 1.0",
              "comment made by meshes/oim_asset_to_ply.py from "
              f"{name}_centered.obj; mm; origin {half_height * 1000:.1f} mm "
              "above the underside, plan bbox centred"]
    if textured:
        image.convert("RGB").save(os.path.join(out_dir, f"{name}.png"))
        header.append(f"comment TextureFile {name}.png")
    header += [f"element vertex {len(v)}",
               "property float x", "property float y", "property float z",
               "property float nx", "property float ny", "property float nz"]
    if textured:
        header += ["property float texture_u", "property float texture_v"]
    else:
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    header += [f"element face {len(f)}",
               "property list uchar int vertex_indices", "end_header"]

    with open(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        for i in range(len(v)):
            row = (f"{v[i, 0]:.4f} {v[i, 1]:.4f} {v[i, 2]:.4f} "
                   f"{n[i, 0]:.5f} {n[i, 1]:.5f} {n[i, 2]:.5f}")
            if textured:
                row += f" {uv[i, 0]:.6f} {uv[i, 1]:.6f}"
            else:
                row += f" {colour[0]} {colour[1]} {colour[2]}"
            fh.write(row + "\n")
        for tri in f:
            fh.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    ext = (hi - lo) * 1000.0
    print(f"{name:11s} {len(v):5d} verts {len(f):5d} faces  "
          f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm  "
          f"origin z {half_height * 1000:.1f} mm  "
          f"{'texture ' + name + '.png ' + str(image.size) + ' from ' + source if textured else 'colour ' + str(colour)}"
          f"  -> {os.path.relpath(path, HERE)}")
    return path


def main() -> None:
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2:
        sys.exit(__doc__)
    assets_dir = sys.argv[1]
    pairs = sys.argv[2:]
    for name, hh in zip(pairs[::2], pairs[1::2], strict=True):
        write(assets_dir, name, float(hh))


if __name__ == "__main__":
    main()
