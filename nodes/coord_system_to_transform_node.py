"""
Coord System CAD to Transform.

Reads an AutoForm coordinate-system IGS (3 axis lines sharing an origin, loaded as a CAD_MODEL
via "Load CAD" / "Load CAD from Path") and emits translation + rotation (degrees) that reproduce
that frame's placement using the CADabra "Transform CAD" node.

  - "normal"  : the coordinate system's own placement = LOCAL frame -> PRODUCT/WORLD.
  - "inverse" : PRODUCT/WORLD -> LOCAL frame (the exact inverse rigid transform).

Use case (the reason this exists): a CAD exported in a local operation frame (e.g. Draw-20) gets
that operation's product->local transform applied again downstream, so it's transformed twice.
Feed the local coordinate system ("... as seen by product") here, wire the outputs into
Transform CAD, and route the geometry through it to move it between the local and product frames,
cancelling the double transform. "normal" maps local->product (so a later product->local lands it
once); "inverse" is the opposite direction.

The rotation outputs match Transform CAD's exact convention: it applies M(p) = Rx·Ry·Rz·p + t
(rotations about the world X, then Y, then Z axes in degrees; translation last, in world coords).
"""

import logging
import math
import os

from comfy_api.latest import io

from .cad_common import get_occ_shape
from .cad_nodes import parse_axis_igs   # proven parser, reused when the source .igs is available

log = logging.getLogger("cadabra")


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        raise ValueError("zero-length axis vector in coordinate system")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _axes_from_shape(shape):
    """Extract (origin, x_dir, y_dir, z_dir) from a 3-line coordinate-system shape. Mirrors
    parse_axis_igs: the three lines share the origin (start), edge order is X, Y, Z."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods

    segs, seen = [], set()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            s = curve.Value(first); e = curve.Value(last)
            seg = ((s.X(), s.Y(), s.Z()), (e.X(), e.Y(), e.Z()))
            key = tuple(round(c, 6) for c in seg[0] + seg[1])
            if key not in seen:
                seen.add(key); segs.append(seg)
        exp.Next()

    if len(segs) < 3:
        raise ValueError(f"Expected 3 axis lines in the coordinate-system CAD, found {len(segs)}")
    origin = segs[0][0]
    ends = [segs[0][1], segs[1][1], segs[2][1]]
    dirs = [tuple(ends[i][k] - origin[k] for k in range(3)) for i in range(3)]
    return origin, dirs[0], dirs[1], dirs[2]


def _frame_matrix(x_dir, y_dir, z_dir):
    """Orthonormal right-handed rotation matrix whose COLUMNS are the frame axes X, Y, Z (in
    product coords). Z is primary, X the reference (matches CADabra build_axis_transform /
    gp_Ax3(origin, z_dir, x_dir)); axis lengths in the IGS are ignored (directions only)."""
    z = _normalize(z_dir)
    x = x_dir
    d = _dot(x, z)
    x = _normalize((x[0] - d * z[0], x[1] - d * z[1], x[2] - d * z[2]))   # X orthogonal to Z
    y = _cross(z, x)                                                      # right-handed
    return [[x[0], y[0], z[0]],
            [x[1], y[1], z[1]],
            [x[2], y[2], z[2]]]


def _transpose(R):
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _matvec(R, v):
    return tuple(R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3))


def _euler_xyz_deg(R):
    """Angles (rx, ry, rz) in degrees such that R == Rx(rx)·Ry(ry)·Rz(rz), the exact rotation
    composition of CADabra's Transform CAD."""
    r02 = max(-1.0, min(1.0, R[0][2]))
    ry = math.asin(r02)
    if abs(math.cos(ry)) > 1e-7:
        rx = math.atan2(-R[1][2], R[2][2])
        rz = math.atan2(-R[0][1], R[0][0])
    else:                                       # gimbal lock (ry = ±90°)
        rx = math.atan2(R[2][1], R[1][1])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


class CoordSystemCADToTransform(io.ComfyNode):
    """Coordinate-system CAD (3 axis lines) -> translation + rotation for the Transform CAD node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CoordSystemCADToTransform",
            display_name="Coord System CAD to Transform",
            category="CADabra/Transform",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.Combo.Input(
                    "mode", options=["normal", "inverse"], default="normal",
                    tooltip="normal = the coordinate system's placement (LOCAL frame -> PRODUCT/WORLD). "
                            "inverse = PRODUCT/WORLD -> LOCAL frame. Wire the outputs into Transform CAD.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="translate_x"),
                io.Float.Output(display_name="translate_y"),
                io.Float.Output(display_name="translate_z"),
                io.Float.Output(display_name="rotate_x"),
                io.Float.Output(display_name="rotate_y"),
                io.Float.Output(display_name="rotate_z"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, mode="normal"):
        # Prefer the proven IGS parser when the source .igs path survived on the CAD_MODEL;
        # otherwise read the axes straight from the loaded shape (works however it was loaded).
        origin = xdir = ydir = zdir = None
        fp = cad_model.get("file_path") if isinstance(cad_model, dict) else None
        if fp and os.path.isfile(fp) and fp.lower().endswith((".igs", ".iges")):
            data = parse_axis_igs(fp)
            if data:
                origin, xdir, ydir, zdir = data["origin"], data["x_dir"], data["y_dir"], data["z_dir"]
        if origin is None:
            origin, xdir, ydir, zdir = _axes_from_shape(get_occ_shape(cad_model))

        R = _frame_matrix(xdir, ydir, zdir)                 # LOCAL -> PRODUCT (columns = frame axes)
        O = tuple(float(c) for c in origin)

        if mode == "inverse":
            Rout = _transpose(R)                            # PRODUCT -> LOCAL
            inv_t = _matvec(Rout, O)
            tout = (-inv_t[0], -inv_t[1], -inv_t[2])
        else:
            Rout = R                                        # LOCAL -> PRODUCT
            tout = O

        rx, ry, rz = _euler_xyz_deg(Rout)
        info = (
            f"mode = {mode}\n"
            f"frame origin (product) = ({O[0]:.5f}, {O[1]:.5f}, {O[2]:.5f})\n"
            f"-> translate = ({tout[0]:.5f}, {tout[1]:.5f}, {tout[2]:.5f})\n"
            f"-> rotate_xyz (deg) = ({rx:.5f}, {ry:.5f}, {rz:.5f})\n"
            f"(feeds Transform CAD; M(p) = Rx·Ry·Rz·p + t)"
        )
        log.info("Coord System CAD to Transform: %s", info.replace("\n", " | "))
        return io.NodeOutput(float(tout[0]), float(tout[1]), float(tout[2]),
                             float(rx), float(ry), float(rz), info)


NODE_CLASS_MAPPINGS = {
    "CoordSystemCADToTransform": CoordSystemCADToTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordSystemCADToTransform": "Coord System CAD to Transform",
}
