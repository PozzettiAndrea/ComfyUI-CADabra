from __future__ import annotations
import logging
from OCC.Core.BRep import BRep_Builder, BRep_Tool

log = logging.getLogger("cadabra")
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform
)
from OCC.Core.BRepProj import BRepProj_Projection
from OCC.Core.BRepTools import breptools
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Wire, topods
from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Ax3, gp_Vec, gp_Trsf
from OCC.Core.Geom import Geom_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
)
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
import math

from comfy_api.latest import io

def _get_occ_shape(cad_model):
    """Get OCC shape from CAD_MODEL dict (loads from brep_path)."""
    try:
        from .utils.brep_cache import get_occ_shape
    except ImportError:
        from .utils.brep_cache import get_occ_shape
    return get_occ_shape(cad_model)


def _get_surface_type_names():
    """Get surface type names dict - called at runtime when OCC is available."""
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
    )
    return {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier",
        GeomAbs_BSplineSurface: "B-Spline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_OtherSurface: "Other",
    }


def _get_curve_type_names():
    """Get curve type names dict - called at runtime when OCC is available."""
    from OCC.Core.GeomAbs import (
        GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
        GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
    )
    return {
        GeomAbs_Line: "Line",
        GeomAbs_Circle: "Circle",
        GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola",
        GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "Bezier",
        GeomAbs_BSplineCurve: "B-Spline",
        GeomAbs_OtherCurve: "Other",
    }


def _get_face_diagnostics(face, plane_normal_vec):
    """
    Get detailed diagnostic info about a face for debugging projection failures.

    Args:
        face: TopoDS_Face to analyze
        plane_normal_vec: gp_Vec representing the projection plane normal

    Returns:
        dict with diagnostic info
    """
    info = {}

    try:
        # Surface type
        adaptor = BRepAdaptor_Surface(face)
        surface_type = adaptor.GetType()
        info["surface_type"] = _get_surface_type_names().get(surface_type, f"Unknown({surface_type})")

        # Bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            info["bbox"] = {
                "min": (round(xmin, 4), round(ymin, 4), round(zmin, 4)),
                "max": (round(xmax, 4), round(ymax, 4), round(zmax, 4)),
                "size": (round(xmax - xmin, 4), round(ymax - ymin, 4), round(zmax - zmin, 4))
            }

        # Face normal at center (approximate)
        try:
            u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

            pnt = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            adaptor.D1(u_mid, v_mid, pnt, d1u, d1v)
            normal = d1u.Crossed(d1v)

            if normal.Magnitude() > 1e-10:
                normal.Normalize()
                info["normal"] = (round(normal.X(), 4), round(normal.Y(), 4), round(normal.Z(), 4))

                # Calculate angle to projection plane normal
                axis_vec = gp_Vec(plane_normal_vec)
                if axis_vec.Magnitude() > 1e-10:
                    axis_vec.Normalize()
                    dot = abs(normal.Dot(axis_vec))
                    angle_deg = math.degrees(math.acos(min(dot, 1.0)))
                    info["angle_to_axis"] = round(angle_deg, 1)
                    info["is_parallel"] = angle_deg < 5.0  # Within 5 degrees of parallel
        except Exception:
            info["normal"] = "Could not compute"

        # Edge info
        edge_types = []
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            try:
                curve_adaptor = BRepAdaptor_Curve(edge)
                curve_type = curve_adaptor.GetType()
                edge_types.append(_get_curve_type_names().get(curve_type, f"Unknown({curve_type})"))
            except Exception:
                edge_types.append("Error")
            edge_explorer.Next()

        info["edge_count"] = len(edge_types)
        info["edge_types"] = edge_types

    except Exception as e:
        info["error"] = str(e)

    return info


def _print_face_diagnostics(face_id, face, plane_normal_vec, failure_reason, extra_info=None):
    """Log detailed diagnostics for a failed face projection."""
    log.warning(f"=== Face {face_id} Projection Failed ===")
    log.warning(f"Reason: {failure_reason}")

    diag = _get_face_diagnostics(face, plane_normal_vec)

    log.warning(f"Surface type: {diag.get('surface_type', 'Unknown')}")

    if "bbox" in diag:
        bbox = diag["bbox"]
        log.warning(f"Bounding box: min={bbox['min']}, max={bbox['max']}")
        log.warning(f"Size: {bbox['size']}")

    if "normal" in diag:
        log.warning(f"Normal at center: {diag['normal']}")
        if "angle_to_axis" in diag:
            n = plane_normal_vec
            log.warning(
                f"Angle to plane normal ({n.X():.3f},{n.Y():.3f},{n.Z():.3f}): "
                f"{diag['angle_to_axis']}deg"
            )
            if diag.get("is_parallel"):
                log.warning("Face is nearly parallel to projection direction!")

    log.warning(f"Edge count: {diag.get('edge_count', 'Unknown')}")
    if "edge_types" in diag:
        type_counts = {}
        for t in diag["edge_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
        log.warning(f"Edge types: {type_counts}")

    if extra_info:
        log.warning(f"Extra: {extra_info}")

    log.warning("================================")


def _create_projection_face(origin, normal_dir):
    """
    Create an infinite planar face for projection.

    Args:
        origin: gp_Pnt - a point on the projection plane
        normal_dir: gp_Dir - the plane normal direction

    Returns:
        TopoDS_Face representing the projection plane
    """
    plane = gp_Pln(origin, normal_dir)
    face_builder = BRepBuilderAPI_MakeFace(plane, -1e6, 1e6, -1e6, 1e6)
    return face_builder.Face()


def _build_plane_axes(normal_dir):
    """
    Build an orthonormal (U, V, N) frame for a plane given its normal.

    Chooses a stable reference vector to avoid degeneracy when the normal
    is close to the world Z axis.

    Args:
        normal_dir: gp_Dir - the plane normal

    Returns:
        (gp_Dir, gp_Dir, gp_Dir) - (U, V, N) where U and V lie in the plane
    """
    n_vec = gp_Vec(normal_dir)
    # Pick a reference not parallel to normal
    if abs(n_vec.Dot(gp_Vec(0, 0, 1))) < 0.9:
        ref = gp_Vec(0, 0, 1)
    else:
        ref = gp_Vec(1, 0, 0)

    u_vec = ref.Crossed(n_vec)
    u_vec.Normalize()
    v_vec = n_vec.Crossed(u_vec)
    v_vec.Normalize()

    return gp_Dir(u_vec), gp_Dir(v_vec), normal_dir


def _project_wire_to_plane(wire, proj_face, proj_dir):
    """
    Project a wire onto a plane using exact curve projection.

    Projects the entire wire at once (not edge-by-edge) to let OCC
    handle edge connectivity internally.

    Args:
        wire: TopoDS_Wire to project
        proj_face: TopoDS_Face (the plane to project onto)
        proj_dir: gp_Dir (projection direction)

    Returns:
        tuple: (projected shape or None, dict with projection stats)
    """
    stats = {
        "success": False,
        "reason": None,
    }

    try:
        # Project entire wire at once - let OCC handle edge connectivity
        projector = BRepProj_Projection(wire, proj_face, proj_dir)

        if projector.IsDone():
            # Get complete result (could be single wire or compound of wires)
            result = projector.Shape()
            stats["success"] = True
            return result, stats
        else:
            stats["reason"] = "BRepProj_Projection.IsDone() returned False"
            return None, stats

    except Exception as e:
        stats["reason"] = str(e)
        return None, stats


def _resolve_plane(plane, plane_origin_x, plane_origin_y, plane_origin_z,
                   plane_normal_x, plane_normal_y, plane_normal_z):
    """
    Return (origin: gp_Pnt, normal: gp_Dir) for the selected plane mode.
    """
    if plane == "XY":
        return gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)
    elif plane == "XZ":
        return gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)
    elif plane == "YZ":
        return gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)
    else:  # Custom
        nx, ny, nz = plane_normal_x, plane_normal_y, plane_normal_z
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if mag < 1e-12:
            raise ValueError(
                "Custom plane normal must be non-zero. "
                f"Got ({nx}, {ny}, {nz})."
            )
        return (
            gp_Pnt(plane_origin_x, plane_origin_y, plane_origin_z),
            gp_Dir(nx, ny, nz),  # gp_Dir normalises automatically
        )


class CADProjectFaces2D(io.ComfyNode):
    """
    Project CAD faces onto an arbitrary 2D plane with EXACT curve geometry.

    Uses OCC's mathematical curve projection - no sampling/approximation.
    A 3D NURBS curve projects to a 2D NURBS curve, arcs to arcs, etc.

    Supports projection onto the standard XY / XZ / YZ planes as well as a
    fully custom plane defined by an origin point and a normal vector.

    Output maintains mapping to original 3D faces for later use
    (e.g., height map generation, spatial queries).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADProjectFaces2D",
            display_name="CAD Project Faces to 2D",
            category="CADabra/Analysis",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="3D CAD model to project"),
                io.Combo.Input("plane", options=["XY", "XZ", "YZ", "Custom"], default="XY",
                               tooltip=(
                                   "Projection plane. XY projects along Z, XZ along Y, "
                                   "YZ along X. Custom uses the origin/normal parameters."
                               )),
                io.Float.Input("plane_origin_x", default=0.0, step=0.1,
                               tooltip="X coordinate of a point on the custom plane", optional=True),
                io.Float.Input("plane_origin_y", default=0.0, step=0.1,
                               tooltip="Y coordinate of a point on the custom plane", optional=True),
                io.Float.Input("plane_origin_z", default=0.0, step=0.1,
                               tooltip="Z coordinate of a point on the custom plane", optional=True),
                io.Float.Input("plane_normal_x", default=0.0, step=0.01,
                               tooltip="X component of the custom plane normal", optional=True),
                io.Float.Input("plane_normal_y", default=0.0, step=0.01,
                               tooltip="Y component of the custom plane normal", optional=True),
                io.Float.Input("plane_normal_z", default=1.0, step=0.01,
                               tooltip="Z component of the custom plane normal (default 1 = XY plane)",
                               optional=True),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="projected_2d"),
                io.Custom("FACE_MAPPING").Output(display_name="face_mapping"),
            ],
        )

    @classmethod
    def execute(
        cls,
        cad_model,
        plane="XY",
        plane_origin_x=0.0,
        plane_origin_y=0.0,
        plane_origin_z=0.0,
        plane_normal_x=0.0,
        plane_normal_y=0.0,
        plane_normal_z=1.0,
    ):
        """
        Project all faces to a 2D plane with exact curve geometry.

        Returns:
            projected_2d: CAD_MODEL with 2D planar faces (exact curves)
            face_mapping: dict mapping 2D face index to original 3D face
        """
        shape = _get_occ_shape(cad_model)

        try:
            from .utils.brep_cache import save_shape
        except ImportError:
            from .utils.brep_cache import save_shape

        # ------ resolve plane origin / normal ------
        origin, normal_dir = _resolve_plane(
            plane, plane_origin_x, plane_origin_y, plane_origin_z,
            plane_normal_x, plane_normal_y, plane_normal_z,
        )
        normal_vec = gp_Vec(normal_dir)

        # Projection direction is along the plane normal (towards the plane).
        proj_dir = normal_dir

        # ------ projection target face ------
        proj_face = _create_projection_face(origin, normal_dir)

        # ------ build the (U, V, N) frame used to flatten results to XY ------
        u_dir, v_dir, _ = _build_plane_axes(normal_dir)
        target_ax3 = gp_Ax3(origin, normal_dir, u_dir)

        # We will transform the projected compound so that the custom plane
        # maps to the world XY plane.  This gives downstream nodes a true 2D
        # result regardless of which plane was chosen.
        needs_reframe = True
        xy_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0))

        # Optimisation: skip the reframe when already projecting onto XY at origin
        if plane == "XY":
            needs_reframe = False

        projected_faces = []
        projected_wires = []  # Store wires for faces that fail face creation
        face_mapping = {}
        original_faces = []
        original_face_brep_paths = []

        # Iterate over all faces
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        success_count = 0

        while explorer.More():
            face_3d = topods.Face(explorer.Current())
            original_faces.append(face_3d)
            original_face_brep_paths.append(save_shape(face_3d, f"orig_face_{face_id}"))

            # Get outer wire of the face
            outer_wire = breptools.OuterWire(face_3d)

            if outer_wire is None:
                _print_face_diagnostics(face_id, face_3d, normal_vec, "No outer wire found")
                explorer.Next()
                face_id += 1
                continue

            # Collect all wires: outer + inner (holes)
            all_wires = []
            wire_explorer = TopExp_Explorer(face_3d, TopAbs_WIRE)
            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                is_outer = wire.IsSame(outer_wire)
                all_wires.append((wire, is_outer))
                wire_explorer.Next()

            num_inner_wires = sum(1 for w, is_outer in all_wires if not is_outer)
            if num_inner_wires > 0:
                log.info(f"Face {face_id}: {len(all_wires)} wires ({num_inner_wires} holes)")

            # Project outer wire first
            projected_outer, proj_stats = _project_wire_to_plane(outer_wire, proj_face, proj_dir)

            if projected_outer is not None:
                # Try to create a planar face from the projected outer wire
                try:
                    # Extract the outer wire from projected result
                    outer_wire_2d = None
                    wire_exp = TopExp_Explorer(projected_outer, TopAbs_WIRE)
                    if wire_exp.More():
                        outer_wire_2d = topods.Wire(wire_exp.Current())

                    if outer_wire_2d is not None:
                        # Create face from outer wire
                        face_builder = BRepBuilderAPI_MakeFace(outer_wire_2d, True)

                        if face_builder.IsDone():
                            # Project and add inner wires (holes)
                            holes_added = 0
                            for wire, is_outer in all_wires:
                                if is_outer:
                                    continue  # Skip outer wire, already used

                                # Project inner wire
                                projected_inner, _ = _project_wire_to_plane(wire, proj_face, proj_dir)
                                if projected_inner is not None:
                                    inner_wire_exp = TopExp_Explorer(projected_inner, TopAbs_WIRE)
                                    if inner_wire_exp.More():
                                        inner_wire_2d = topods.Wire(inner_wire_exp.Current())
                                        face_builder.Add(inner_wire_2d)  # Add hole
                                        holes_added += 1

                            if holes_added > 0:
                                log.info(f"Face {face_id}: Added {holes_added} holes to projected face")

                            face_2d = face_builder.Face()
                            projected_faces.append(face_2d)
                            face_mapping[len(projected_faces) - 1] = {
                                "face_3d_brep": original_face_brep_paths[face_id],
                                "original_index": face_id
                            }
                            success_count += 1
                        else:
                            # Face creation failed, store projected shape instead
                            projected_wires.append((projected_outer, face_3d, face_id))
                            _print_face_diagnostics(
                                face_id, face_3d, normal_vec,
                                "Wire projected but face creation failed")
                    else:
                        projected_wires.append((projected_outer, face_3d, face_id))
                        _print_face_diagnostics(
                            face_id, face_3d, normal_vec,
                            "Could not extract wire from projected result")
                except Exception as e:
                    # Face creation failed, store projected shape instead
                    projected_wires.append((projected_outer, face_3d, face_id))
                    _print_face_diagnostics(
                        face_id, face_3d, normal_vec,
                        f"Wire projected but face creation threw exception: {e}")
            else:
                reason = proj_stats.get("reason", "Unknown")
                _print_face_diagnostics(
                    face_id, face_3d, normal_vec,
                    f"Wire projection failed: {reason}")

            explorer.Next()
            face_id += 1

        # Combine all projected geometry into a compound
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        # Add successfully created faces
        for face_2d in projected_faces:
            builder.Add(compound, face_2d)

        # Add wires that couldn't become faces
        for wire_data in projected_wires:
            wire_or_shape = wire_data[0]
            builder.Add(compound, wire_or_shape)

        # ------ reframe to world-XY when using non-XY planes ------
        if needs_reframe:
            trsf = gp_Trsf()
            trsf.SetTransformation(target_ax3, xy_ax3)
            transformer = BRepBuilderAPI_Transform(compound, trsf, True)
            compound = transformer.Shape()

        total_projected = len(projected_faces) + len(projected_wires)
        log.info(f"Projected {total_projected} items from {face_id} original faces")
        log.info(f"  - {len(projected_faces)} complete faces")
        log.info(f"  - {len(projected_wires)} wires (face creation failed)")
        log.info(
            f"Plane: {plane}  origin=({origin.X():.3f},{origin.Y():.3f},{origin.Z():.3f})  "
            f"normal=({normal_dir.X():.4f},{normal_dir.Y():.4f},{normal_dir.Z():.4f})"
        )

        # Build result - save compound to brep file
        brep_path = save_shape(compound, "projected")
        projected_cad = {
            "brep_path": brep_path,
            "format": "brep",
            "projection_plane": plane,
            "plane_origin": (origin.X(), origin.Y(), origin.Z()),
            "plane_normal": (normal_dir.X(), normal_dir.Y(), normal_dir.Z()),
            "original_face_breps": original_face_brep_paths,
        }

        # Face mapping includes the 3D face BREP paths for later use
        face_mapping_result = {
            "mapping": face_mapping,
            "original_face_breps": original_face_brep_paths,
            "plane": plane,
            "plane_origin": (origin.X(), origin.Y(), origin.Z()),
            "plane_normal": (normal_dir.X(), normal_dir.Y(), normal_dir.Z()),
        }

        return io.NodeOutput(projected_cad, face_mapping_result)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADProjectFaces2D": CADProjectFaces2D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADProjectFaces2D": "CAD Project Faces to 2D",
}
