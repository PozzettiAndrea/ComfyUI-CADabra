# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
CAD Face Projection Node for CADabra

Projects 3D CAD faces onto a 2D plane (e.g., XY plane for Z-direction view).
Creates 2D CAD geometry with EXACT curve boundaries (not sampled polylines),
maintaining a mapping back to the original 3D faces.

Uses OCC's BRepProj_Projection for mathematically exact curve projection:
- 3D NURBS curve → 2D NURBS curve
- 3D arc → 2D arc/ellipse
- 3D line → 2D line

Use cases:
- Fast height map generation for stampable parts
- 2D spatial queries for face lookup
- DXF/SVG export of projected outlines
"""

# Check for OCC availability
try:
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCC.Core.BRepProj import BRepProj_Projection
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Wire, topods
    from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Ax3, gp_Vec
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
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    print("[CADabra] Warning: pythonocc-core not available. CAD Face Projection will be disabled.")


def _get_occ_shape(cad_model):
    """Get OCC shape from CAD_MODEL dict."""
    shape = cad_model.get("occ_shape")
    if shape is None:
        raise RuntimeError("CAD model has no OCC shape")
    return shape


# Surface type names for diagnostics
SURFACE_TYPE_NAMES = {
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

# Curve type names for diagnostics
CURVE_TYPE_NAMES = {
    GeomAbs_Line: "Line",
    GeomAbs_Circle: "Circle",
    GeomAbs_Ellipse: "Ellipse",
    GeomAbs_Hyperbola: "Hyperbola",
    GeomAbs_Parabola: "Parabola",
    GeomAbs_BezierCurve: "Bezier",
    GeomAbs_BSplineCurve: "B-Spline",
    GeomAbs_OtherCurve: "Other",
}


def _get_face_diagnostics(face, proj_axis):
    """
    Get detailed diagnostic info about a face for debugging projection failures.

    Args:
        face: TopoDS_Face to analyze
        proj_axis: 'X', 'Y', or 'Z' - the projection axis

    Returns:
        dict with diagnostic info
    """
    info = {}

    try:
        # Surface type
        adaptor = BRepAdaptor_Surface(face)
        surface_type = adaptor.GetType()
        info["surface_type"] = SURFACE_TYPE_NAMES.get(surface_type, f"Unknown({surface_type})")

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

                # Calculate angle to projection axis
                if proj_axis == 'Z':
                    axis_vec = gp_Vec(0, 0, 1)
                elif proj_axis == 'X':
                    axis_vec = gp_Vec(1, 0, 0)
                else:  # Y
                    axis_vec = gp_Vec(0, 1, 0)

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
                edge_types.append(CURVE_TYPE_NAMES.get(curve_type, f"Unknown({curve_type})"))
            except Exception:
                edge_types.append("Error")
            edge_explorer.Next()

        info["edge_count"] = len(edge_types)
        info["edge_types"] = edge_types

    except Exception as e:
        info["error"] = str(e)

    return info


def _print_face_diagnostics(face_id, face, proj_axis, failure_reason, extra_info=None):
    """Print detailed diagnostics for a failed face projection."""
    print(f"\n[CADabra] === Face {face_id} Projection Failed ===")
    print(f"[CADabra] Reason: {failure_reason}")

    diag = _get_face_diagnostics(face, proj_axis)

    print(f"[CADabra] Surface type: {diag.get('surface_type', 'Unknown')}")

    if "bbox" in diag:
        bbox = diag["bbox"]
        print(f"[CADabra] Bounding box: min={bbox['min']}, max={bbox['max']}")
        print(f"[CADabra] Size: {bbox['size']}")

    if "normal" in diag:
        print(f"[CADabra] Normal at center: {diag['normal']}")
        if "angle_to_axis" in diag:
            print(f"[CADabra] Angle to {proj_axis}-axis: {diag['angle_to_axis']}°")
            if diag.get("is_parallel"):
                print(f"[CADabra] WARNING: Face is nearly parallel to projection axis!")

    print(f"[CADabra] Edge count: {diag.get('edge_count', 'Unknown')}")
    if "edge_types" in diag:
        type_counts = {}
        for t in diag["edge_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"[CADabra] Edge types: {type_counts}")

    if extra_info:
        print(f"[CADabra] Extra: {extra_info}")

    print(f"[CADabra] ================================\n")


def _create_projection_face(axis):
    """
    Create an infinite planar face for projection.

    Args:
        axis: 'Z', 'X', or 'Y' - the axis we're projecting along

    Returns:
        TopoDS_Face representing the projection plane
    """
    # Create plane at origin, perpendicular to axis
    if axis == 'Z':
        plane = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    elif axis == 'X':
        plane = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    elif axis == 'Y':
        plane = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Create a large face on the plane for projection
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    face_builder = BRepBuilderAPI_MakeFace(plane, -1e6, 1e6, -1e6, 1e6)
    return face_builder.Face()


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


class CADProjectFacesXY:
    """
    Project CAD faces onto a 2D plane with EXACT curve geometry.

    Uses OCC's mathematical curve projection - no sampling/approximation.
    A 3D NURBS curve projects to a 2D NURBS curve, arcs to arcs, etc.

    Output maintains mapping to original 3D faces for later use
    (e.g., height map generation, spatial queries).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL", {
                    "tooltip": "3D CAD model to project"
                }),
                "direction": (["Z-", "Z+", "X-", "X+", "Y-", "Y+"], {
                    "default": "Z-",
                    "tooltip": "Viewing direction (e.g., Z- = looking down from above)"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "FACE_MAPPING")
    RETURN_NAMES = ("projected_2d", "face_mapping")
    FUNCTION = "project_faces"
    CATEGORY = "CADabra/Analysis"

    def project_faces(self, cad_model, direction="Z-"):
        """
        Project all faces to 2D plane with exact curve geometry.

        Returns:
            projected_2d: CAD_MODEL with 2D planar faces (exact curves)
            face_mapping: dict mapping 2D face index to original 3D face
        """
        if not HAS_OCC:
            raise ImportError(
                "pythonocc-core is required for CAD Face Projection. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )

        shape = _get_occ_shape(cad_model)

        # Parse direction
        axis = direction[0].upper()  # 'Z', 'X', or 'Y'
        sign = -1 if direction[1] == '-' else 1

        # Create projection direction vector
        if axis == 'Z':
            proj_dir = gp_Dir(0, 0, sign)
        elif axis == 'X':
            proj_dir = gp_Dir(sign, 0, 0)
        elif axis == 'Y':
            proj_dir = gp_Dir(0, sign, 0)

        # Create projection plane face
        proj_face = _create_projection_face(axis)

        projected_faces = []
        projected_wires = []  # Store wires for faces that fail face creation
        face_mapping = {}
        original_faces = []

        # Iterate over all faces
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        success_count = 0

        while explorer.More():
            face_3d = topods.Face(explorer.Current())
            original_faces.append(face_3d)

            # Get outer wire of the face
            outer_wire = breptools.OuterWire(face_3d)

            if outer_wire is None:
                _print_face_diagnostics(face_id, face_3d, axis, "No outer wire found")
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
                print(f"[CADabra] Face {face_id}: {len(all_wires)} wires ({num_inner_wires} holes)")

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
                                print(f"[CADabra] Face {face_id}: Added {holes_added} holes to projected face")

                            face_2d = face_builder.Face()
                            projected_faces.append(face_2d)
                            face_mapping[len(projected_faces) - 1] = {
                                "face_3d": face_3d,
                                "original_index": face_id
                            }
                            success_count += 1
                        else:
                            # Face creation failed, store projected shape instead
                            projected_wires.append((projected_outer, face_3d, face_id))
                            _print_face_diagnostics(face_id, face_3d, axis, "Wire projected but face creation failed")
                    else:
                        projected_wires.append((projected_outer, face_3d, face_id))
                        _print_face_diagnostics(face_id, face_3d, axis, "Could not extract wire from projected result")
                except Exception as e:
                    # Face creation failed, store projected shape instead
                    projected_wires.append((projected_outer, face_3d, face_id))
                    _print_face_diagnostics(face_id, face_3d, axis, f"Wire projected but face creation threw exception: {e}")
            else:
                reason = proj_stats.get("reason", "Unknown")
                _print_face_diagnostics(face_id, face_3d, axis, f"Wire projection failed: {reason}")

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

        total_projected = len(projected_faces) + len(projected_wires)
        print(f"[CADabra] Projected {total_projected} items from {face_id} original faces")
        print(f"[CADabra]   - {len(projected_faces)} complete faces")
        print(f"[CADabra]   - {len(projected_wires)} wires (face creation failed)")
        print(f"[CADabra] Direction: {direction} (exact curve projection)")

        # Build result
        projected_cad = {
            "occ_shape": compound,
            "format": "occ",
            "projection_axis": axis,
            "original_faces": original_faces,
        }

        # Face mapping includes the 3D faces for later use
        face_mapping_result = {
            "mapping": face_mapping,
            "original_faces": original_faces,
            "axis": axis,
            "direction": direction,
        }

        return (projected_cad, face_mapping_result)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADProjectFacesXY": CADProjectFacesXY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADProjectFacesXY": "CAD Project Faces to 2D",
}
