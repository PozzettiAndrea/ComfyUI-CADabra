"""
CAD Edge Viewer (VTK) node for ComfyUI-CADabra
Uses VTK.js with Trackball interactor for free 3D rotation.
Exports edges to VTP format for native VTK.js rendering.
"""

import os
import time
import json
import folder_paths
import vtk

from ..utils.occ_logging import logger

# OCC imports for edge analysis
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

# Curve type to integer mapping for VTP cell data
CURVE_TYPE_TO_INT = {
    GeomAbs_Line: 0,
    GeomAbs_Circle: 1,
    GeomAbs_Ellipse: 2,
    GeomAbs_Hyperbola: 3,
    GeomAbs_Parabola: 4,
    GeomAbs_BezierCurve: 5,
    GeomAbs_BSplineCurve: 6,
    GeomAbs_OtherCurve: 7,
}

CURVE_TYPE_NAMES = {
    0: "Line",
    1: "Circle",
    2: "Ellipse",
    3: "Hyperbola",
    4: "Parabola",
    5: "Bezier",
    6: "BSpline",
    7: "Other",
}


class CADEdgeViewerVTK:
    """
    CAD Edge Viewer with VTK.js Trackball interactor.

    Exports edges to VTP (VTK PolyData) format and displays in a VTK.js viewer
    with free 3D rotation (trackball controls - no gimbal lock).

    Edge metadata is stored as CellData arrays in the VTP file:
    - edge_id: Integer edge index
    - edge_type: Integer type (0=Line, 1=Circle, 2=Ellipse, etc.)
    - is_free: Boolean (1=boundary edge, 0=shared edge)
    - length: Float edge length
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL", {
                    "tooltip": "CAD model to analyze"
                }),
                "linear_deflection": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Linear deflection for edge discretization (lower = more points)"
                }),
                "visualization_mode": (["normal", "edge_type", "length"], {
                    "default": "normal",
                    "tooltip": "Color mode: normal (free=red), edge_type heatmap, or length heatmap"
                }),
                "selected_edge_id": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000,
                    "tooltip": "Currently selected edge ID (click edge in viewer to update)"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("cad_model", "vtp_filepath", "report", "spline_data_file", "selected_edge_id")
    OUTPUT_NODE = True
    FUNCTION = "analyze_and_export"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Interactive edge viewer with VTK.js Trackball controls.

    Features FREE 3D ROTATION (no gimbal lock) - rotate your model in any direction!

    Controls:
    - Left mouse: Rotate (free 3D rotation)
    - Middle mouse / Shift+Left: Pan
    - Right mouse / Scroll: Zoom

    Edge data stored in VTP format with metadata for:
    - Edge type (Line, Circle, BSpline, etc.)
    - Length
    - Free/shared boundary status
    """

    def analyze_and_export(self, cad_model, linear_deflection=0.1, visualization_mode="normal", selected_edge_id=-1):
        """Analyze edges and export to VTP format."""
        # Get OCC shape
        occ_shape = cad_model.get("occ_shape") or cad_model.get("shape")
        if occ_shape is None:
            raise RuntimeError("CAD model has no OCC shape")

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_edge_vtk_{timestamp}"
        vtp_filename = f"{base_filename}.vtp"
        vtp_path = os.path.join(output_dir, vtp_filename)
        spline_json_filename = f"{base_filename}_spline.json"
        spline_json_path = os.path.join(output_dir, spline_json_filename)

        # Count entities
        num_faces = sum(1 for _ in self._iter_occ(occ_shape, TopAbs_FACE))
        num_edges = sum(1 for _ in self._iter_occ(occ_shape, TopAbs_EDGE))
        num_vertices = sum(1 for _ in self._iter_occ(occ_shape, TopAbs_VERTEX))

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(occ_shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = 0, 0, 0, 0, 0, 0

        bounds_min = [xmin, ymin, zmin]
        bounds_max = [xmax, ymax, zmax]

        logger.info(f"[CADEdgeViewerVTK] Analyzing {num_edges} edges, {num_faces} faces, {num_vertices} vertices")

        # Build indexed maps
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ_shape, TopAbs_EDGE, edge_map)

        faces_list = list(self._iter_occ(occ_shape, TopAbs_FACE))

        # Build edge-to-faces mapping for free/shared detection
        edge_to_faces = self._build_edge_to_faces_map(occ_shape, edge_map, faces_list)

        # Create VTK structures
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # CellData arrays
        edge_ids = vtk.vtkIntArray()
        edge_ids.SetName("edge_id")
        edge_ids.SetNumberOfComponents(1)

        edge_types = vtk.vtkIntArray()
        edge_types.SetName("edge_type")
        edge_types.SetNumberOfComponents(1)

        is_free_arr = vtk.vtkIntArray()
        is_free_arr.SetName("is_free")
        is_free_arr.SetNumberOfComponents(1)

        lengths = vtk.vtkFloatArray()
        lengths.SetName("length")
        lengths.SetNumberOfComponents(1)

        # Statistics
        edge_type_counts = {}
        length_values = []
        free_edge_count = 0
        point_offset = 0

        # Spline data collection (indexed by edge_id as string)
        spline_data = {}

        # Process each edge
        for edge_idx in range(1, edge_map.Size() + 1):
            edge = topods.Edge(edge_map.FindKey(edge_idx))
            zero_idx = edge_idx - 1  # Convert to 0-indexed

            try:
                curve = BRepAdaptor_Curve(edge)

                # Get edge type
                curve_type = curve.GetType()
                edge_type_int = CURVE_TYPE_TO_INT.get(curve_type, 7)
                edge_type_name = CURVE_TYPE_NAMES.get(edge_type_int, "Unknown")
                edge_type_counts[edge_type_name] = edge_type_counts.get(edge_type_name, 0) + 1

                # Extract spline data if applicable (BSpline or Bezier)
                if curve_type in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
                    edge_spline_data = self._extract_spline_data(curve, curve_type)
                    if edge_spline_data:
                        spline_data[str(zero_idx)] = edge_spline_data

                # Get length
                try:
                    props = GProp_GProps()
                    brepgprop.LinearProperties(edge, props)
                    edge_length = props.Mass()
                except Exception:
                    edge_length = 0.0

                if edge_length > 0:
                    length_values.append(edge_length)

                # Is free edge?
                face_indices = edge_to_faces.get(edge_idx, set())
                is_free = 1 if len(face_indices) == 1 else 0
                if is_free:
                    free_edge_count += 1

                # Discretize edge
                edge_points = self._discretize_edge(curve, linear_deflection)

                if len(edge_points) >= 2:
                    # Add points to VTK
                    for pt in edge_points:
                        points.InsertNextPoint(pt[0], pt[1], pt[2])

                    # Create polyline cell
                    polyline = vtk.vtkPolyLine()
                    polyline.GetPointIds().SetNumberOfIds(len(edge_points))
                    for i in range(len(edge_points)):
                        polyline.GetPointIds().SetId(i, point_offset + i)
                    lines.InsertNextCell(polyline)

                    # Add cell data
                    edge_ids.InsertNextValue(zero_idx)
                    edge_types.InsertNextValue(edge_type_int)
                    is_free_arr.InsertNextValue(is_free)
                    lengths.InsertNextValue(edge_length)

                    point_offset += len(edge_points)

            except Exception as e:
                logger.warning(f"[CADEdgeViewerVTK] Error processing edge {edge_idx}: {e}")
                continue

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetCellData().AddArray(edge_ids)
        polydata.GetCellData().AddArray(edge_types)
        polydata.GetCellData().AddArray(is_free_arr)
        polydata.GetCellData().AddArray(lengths)

        # Write VTP file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_path)
        writer.SetInputData(polydata)
        writer.SetDataModeToAscii()  # ASCII for easier debugging
        writer.Write()

        logger.info(f"[CADEdgeViewerVTK] VTP saved: {vtp_filename} ({lines.GetNumberOfCells()} edges)")

        # Write spline data JSON (only if there are spline edges)
        if spline_data:
            with open(spline_json_path, 'w') as f:
                json.dump(spline_data, f, indent=2)
            logger.info(f"[CADEdgeViewerVTK] Spline data saved: {spline_json_filename} ({len(spline_data)} spline edges)")
        else:
            spline_json_filename = None  # No spline edges in model
            spline_json_path = ""  # Empty path for output

        # Compute statistics
        length_stats = self._compute_stats(length_values) if length_values else None

        # Build report
        report = self._build_report(
            num_edges, num_vertices, num_faces,
            bounds_min, bounds_max,
            edge_type_counts, length_stats, free_edge_count
        )

        return {
            "ui": {
                "vtp_file": [vtp_filename],
                "spline_data_file": [spline_json_filename],  # None if no spline edges
                "num_edges": [num_edges],
                "num_faces": [num_faces],
                "bounds_min": [bounds_min],
                "bounds_max": [bounds_max],
                "edge_type_counts": [edge_type_counts],
                "visualization_mode": [visualization_mode],
                "free_edge_count": [free_edge_count],
                "selected_edge_id": [selected_edge_id],
            },
            "result": (cad_model, vtp_path, report, spline_json_path, selected_edge_id)
        }

    def _iter_occ(self, shape, topology_type):
        """Helper to iterate over OCC TopExp_Explorer results."""
        explorer = TopExp_Explorer(shape, topology_type)
        while explorer.More():
            yield explorer.Current()
            explorer.Next()

    def _build_edge_to_faces_map(self, occ_shape, edge_map, faces_list):
        """Build mapping from edge index to list of adjacent face indices."""
        edge_to_faces = {}

        for face_idx, face in enumerate(faces_list):
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_index = edge_map.FindIndex(edge)
                if edge_index > 0:
                    if edge_index not in edge_to_faces:
                        edge_to_faces[edge_index] = set()
                    edge_to_faces[edge_index].add(face_idx)
                edge_explorer.Next()

        return edge_to_faces

    def _discretize_edge(self, curve, deflection):
        """Discretize edge curve into points for visualization."""
        points = []

        try:
            discretizer = GCPnts_UniformDeflection(curve, deflection)

            if discretizer.IsDone() and discretizer.NbPoints() >= 2:
                for i in range(1, discretizer.NbPoints() + 1):
                    pnt = discretizer.Value(i)
                    points.append([pnt.X(), pnt.Y(), pnt.Z()])
        except Exception:
            # Fallback: just use start and end points
            try:
                start_pnt = curve.Value(curve.FirstParameter())
                end_pnt = curve.Value(curve.LastParameter())
                points = [
                    [start_pnt.X(), start_pnt.Y(), start_pnt.Z()],
                    [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]
                ]
            except Exception:
                pass

        return points

    def _check_planarity(self, geom_curve):
        """Calculate planarity deviation using SVD best-fit plane.

        Args:
            geom_curve: Geom_BSplineCurve or Geom_BezierCurve handle

        Returns: {
            "max_deviation": float,  # Max distance from best-fit plane
            "curve_length": float,   # Arc length of curve
            "deviation_percent": float,  # max_deviation / curve_length * 100
            "plane": "XY" | "XZ" | "YZ" | "arbitrary" | "line" | "point" | None,
            "normal": [nx, ny, nz] | None
        }
        """
        try:
            import numpy as np
            from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
            from OCC.Core.GCPnts import GCPnts_AbscissaPoint

            n_poles = geom_curve.NbPoles()

            # Calculate curve length
            adaptor = GeomAdaptor_Curve(geom_curve)
            curve_length = GCPnts_AbscissaPoint.Length(adaptor)

            if n_poles < 2:
                return {"max_deviation": 0.0, "curve_length": curve_length,
                        "deviation_percent": 0.0, "plane": "point", "normal": None}
            if n_poles < 3:
                return {"max_deviation": 0.0, "curve_length": curve_length,
                        "deviation_percent": 0.0, "plane": "line", "normal": None}

            # Build numpy array from control poles
            points = np.array([
                [geom_curve.Pole(i).X(),
                 geom_curve.Pole(i).Y(),
                 geom_curve.Pole(i).Z()]
                for i in range(1, n_poles + 1)
            ])

            # SVD plane fitting
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Smallest singular value's eigenvector

            # Max deviation = max distance from any point to the plane
            deviations = np.abs(centered @ normal)
            max_dev = float(np.max(deviations))

            # Calculate percentage deviation
            deviation_percent = (max_dev / curve_length * 100) if curve_length > 0 else 0.0

            # Identify axis-aligned plane
            nx, ny, nz = normal
            abs_n = [abs(nx), abs(ny), abs(nz)]
            if abs_n[2] > 0.99:
                plane_name = "XY"
            elif abs_n[1] > 0.99:
                plane_name = "XZ"
            elif abs_n[0] > 0.99:
                plane_name = "YZ"
            else:
                plane_name = "arbitrary"

            return {
                "max_deviation": round(max_dev, 6),
                "curve_length": round(curve_length, 6),
                "deviation_percent": round(deviation_percent, 4),
                "plane": plane_name,
                "normal": [round(float(nx), 6), round(float(ny), 6), round(float(nz), 6)]
            }

        except Exception as e:
            logger.warning(f"[CADEdgeViewerVTK] Planarity check failed: {e}")
            return {"max_deviation": None, "curve_length": None,
                    "deviation_percent": None, "plane": None, "normal": None}

    def _extract_spline_data(self, curve, curve_type):
        """Extract B-spline or Bezier curve parameters.

        Returns dict with degree, control_points, knots, weights, etc.
        Returns None for non-spline curve types.
        """
        try:
            if curve_type == GeomAbs_BSplineCurve:
                bspline = curve.BSpline()
                is_rational = bspline.IsRational()

                # Extract control points and weights
                control_points = []
                for i in range(1, bspline.NbPoles() + 1):
                    pole = bspline.Pole(i)
                    weight = bspline.Weight(i) if is_rational else 1.0
                    control_points.append({
                        "x": round(pole.X(), 6),
                        "y": round(pole.Y(), 6),
                        "z": round(pole.Z(), 6),
                        "w": round(weight, 6)
                    })

                # Extract knot vector (flat with multiplicities)
                knots_flat = []
                knots_unique = []
                multiplicities = []
                for i in range(1, bspline.NbKnots() + 1):
                    knot = round(bspline.Knot(i), 6)
                    mult = bspline.Multiplicity(i)
                    knots_unique.append(knot)
                    multiplicities.append(mult)
                    knots_flat.extend([knot] * mult)

                # Check planarity
                planarity = self._check_planarity(bspline)

                return {
                    "type": "bspline",
                    "degree": bspline.Degree(),
                    "is_rational": is_rational,
                    "num_poles": bspline.NbPoles(),
                    "num_knots": bspline.NbKnots(),
                    "control_points": control_points,
                    "knots_flat": knots_flat,
                    "knots_unique": knots_unique,
                    "multiplicities": multiplicities,
                    "param_range": [
                        round(curve.FirstParameter(), 6),
                        round(curve.LastParameter(), 6)
                    ],
                    "planarity": planarity
                }

            elif curve_type == GeomAbs_BezierCurve:
                bezier = curve.Bezier()
                degree = bezier.Degree()
                is_rational = bezier.IsRational()

                # Extract control points and weights
                control_points = []
                for i in range(1, bezier.NbPoles() + 1):
                    pole = bezier.Pole(i)
                    weight = bezier.Weight(i) if is_rational else 1.0
                    control_points.append({
                        "x": round(pole.X(), 6),
                        "y": round(pole.Y(), 6),
                        "z": round(pole.Z(), 6),
                        "w": round(weight, 6)
                    })

                # Bezier has implicit knot vector: [0,...,0, 1,...,1]
                knots_flat = [0.0] * (degree + 1) + [1.0] * (degree + 1)

                # Check planarity
                planarity = self._check_planarity(bezier)

                return {
                    "type": "bezier",
                    "degree": degree,
                    "is_rational": is_rational,
                    "num_poles": bezier.NbPoles(),
                    "num_knots": 2,  # Bezier always has 2 unique knots: 0 and 1
                    "control_points": control_points,
                    "knots_flat": knots_flat,
                    "knots_unique": [0.0, 1.0],
                    "multiplicities": [degree + 1, degree + 1],
                    "param_range": [
                        round(curve.FirstParameter(), 6),
                        round(curve.LastParameter(), 6)
                    ],
                    "planarity": planarity
                }

        except Exception as e:
            logger.warning(f"[CADEdgeViewerVTK] Failed to extract spline data: {e}")
            return None

        return None

    def _compute_stats(self, values):
        """Compute min/max/avg statistics."""
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "total": sum(values)
        }

    def _build_report(self, num_edges, num_vertices, num_faces,
                      bounds_min, bounds_max,
                      edge_type_counts, length_stats, free_edge_count):
        """Build text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CAD EDGE VIEWER (VTK) REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Topology summary
        lines.append("--- TOPOLOGY SUMMARY ---")
        lines.append(f"Edges: {num_edges} | Vertices: {num_vertices} | Faces: {num_faces}")
        lines.append(f"Free edges (boundary): {free_edge_count}")
        lines.append(f"Shared edges (internal): {num_edges - free_edge_count}")
        lines.append("")

        # Bounding box
        lines.append("--- BOUNDING BOX ---")
        lines.append(f"Min: ({bounds_min[0]:.2f}, {bounds_min[1]:.2f}, {bounds_min[2]:.2f})")
        lines.append(f"Max: ({bounds_max[0]:.2f}, {bounds_max[1]:.2f}, {bounds_max[2]:.2f})")
        lines.append("")

        # Edge types
        lines.append("--- EDGE TYPES ---")
        for edge_type, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / num_edges if num_edges > 0 else 0
            lines.append(f"{edge_type}: {count} ({pct:.1f}%)")
        lines.append("")

        # Length statistics
        if length_stats:
            lines.append("--- LENGTH STATISTICS ---")
            lines.append(f"Min: {length_stats['min']:.4g}")
            lines.append(f"Max: {length_stats['max']:.4g}")
            lines.append(f"Avg: {length_stats['avg']:.4g}")
            lines.append(f"Total: {length_stats['total']:.4g}")
            lines.append("")

        lines.append("--- VIEWER CONTROLS ---")
        lines.append("Left mouse: FREE 3D rotation (no gimbal lock!)")
        lines.append("Middle/Shift+Left: Pan")
        lines.append("Right/Scroll: Zoom")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADEdgeViewerVTK": CADEdgeViewerVTK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADEdgeViewerVTK": "CAD Edge Viewer (VTK)",
}
