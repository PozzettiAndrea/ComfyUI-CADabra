"""
CAD Edge Analysis nodes for ComfyUI-CADabra
Wireframe/edge inspection with geometric and topological analysis.
"""

import json
import os
import time
import folder_paths

from ..utils.occ_logging import log_operation

# OCC imports for edge analysis
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformDeflection, GCPnts_AbscissaPoint
from OCC.Core.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Pnt, gp_Vec

# Curve type names (consistent with cad_project_faces.py)
CURVE_TYPE_NAMES = {
    GeomAbs_Line: "Line",
    GeomAbs_Circle: "Circle",
    GeomAbs_Ellipse: "Ellipse",
    GeomAbs_Hyperbola: "Hyperbola",
    GeomAbs_Parabola: "Parabola",
    GeomAbs_BezierCurve: "Bezier",
    GeomAbs_BSplineCurve: "BSpline",
    GeomAbs_OtherCurve: "Other",
}


class CADEdgeAnalysis:
    """
    CAD edge analysis node for wireframe inspection.

    Analyzes all edges in a CAD model and extracts:
    - Geometric data: edge type, length, curvature, start/end points, tangents
    - Topological data: adjacent faces, connected edges at vertices, boundary status

    Outputs a JSON file with per-edge metadata for visualization.
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
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("cad_model", "edge_json_filepath", "report")
    OUTPUT_NODE = True
    FUNCTION = "analyze_edges"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Analyze edges in a CAD model for wireframe inspection.

    Outputs:
    - edge_json_filepath: Path to JSON file with per-edge analysis
    - report: Text summary with edge statistics and type distribution

    Edge data includes: type (Line, Circle, BSpline, etc.), length,
    curvature, start/end vertices, tangent vectors, adjacent faces,
    and connected edges at each vertex.
    """

    def analyze_edges(self, cad_model, linear_deflection=0.1):
        """Analyze all edges in the CAD model."""
        # Get OCC shape
        occ_shape = cad_model.get("occ_shape") or cad_model.get("shape")
        if occ_shape is None:
            raise RuntimeError("CAD model has no OCC shape")

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_edge_analysis_{timestamp}"
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(output_dir, json_filename)

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

        print(f"[CADabra] Edge Analysis: {num_edges} edges, {num_faces} faces, {num_vertices} vertices")

        # Build indexed maps
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ_shape, TopAbs_EDGE, edge_map)

        vertex_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ_shape, TopAbs_VERTEX, vertex_map)

        faces_list = list(self._iter_occ(occ_shape, TopAbs_FACE))

        # Build edge-to-faces mapping
        print(f"[CADabra] Building edge-to-faces map...")
        edge_to_faces = self._build_edge_to_faces_map(occ_shape, edge_map, faces_list)

        # Build vertex-to-edges mapping
        print(f"[CADabra] Building vertex-to-edges map...")
        vertex_to_edges = self._build_vertex_to_edges_map(occ_shape, edge_map, vertex_map)

        # Analyze each edge
        edge_data = []
        with log_operation("CAD Edge Analysis", edges=num_edges):
            for edge_idx in range(1, edge_map.Size() + 1):
                edge = topods.Edge(edge_map.FindKey(edge_idx))
                edge_info = self._analyze_edge(
                    edge, edge_idx - 1,  # Convert to 0-indexed
                    edge_to_faces, vertex_to_edges, vertex_map,
                    linear_deflection
                )
                if edge_info:
                    edge_data.append(edge_info)

        # Compute statistics
        edge_type_counts = {}
        lengths = []
        free_edge_count = 0

        for edge_info in edge_data:
            edge_type = edge_info.get("edge_type", "Unknown")
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

            if edge_info.get("length") is not None:
                lengths.append(edge_info["length"])

            if edge_info.get("is_free", False):
                free_edge_count += 1

        edge_stats = self._compute_stats(lengths) if lengths else None
        if edge_stats:
            edge_stats["total"] = sum(lengths)

        print(f"[CADabra] Edge types: {edge_type_counts}")
        print(f"[CADabra] Free edges: {free_edge_count}, Shared edges: {len(edge_data) - free_edge_count}")

        # Build JSON output
        analysis_data = {
            "timestamp": timestamp,
            "num_edges": len(edge_data),
            "num_vertices": num_vertices,
            "num_faces": num_faces,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "linear_deflection": linear_deflection,
            "edge_type_counts": edge_type_counts,
            "edge_stats": edge_stats,
            "free_edge_count": free_edge_count,
            "shared_edge_count": len(edge_data) - free_edge_count,
            "edges": edge_data
        }

        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"[CADabra] Edge analysis saved: {json_filename}")

        # Build text report
        report = self._build_report(
            len(edge_data), num_vertices, num_faces,
            bounds_min, bounds_max,
            edge_type_counts, edge_stats,
            free_edge_count
        )

        return {"ui": {"text": (report,)}, "result": (cad_model, json_path, report)}

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

    def _build_vertex_to_edges_map(self, occ_shape, edge_map, vertex_map):
        """Build mapping from vertex index to list of connected edge indices."""
        vertex_to_edges = {}

        for edge_idx in range(1, edge_map.Size() + 1):
            edge = topods.Edge(edge_map.FindKey(edge_idx))

            # Get vertices of this edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                vertex_index = vertex_map.FindIndex(vertex)
                if vertex_index > 0:
                    if vertex_index not in vertex_to_edges:
                        vertex_to_edges[vertex_index] = set()
                    vertex_to_edges[vertex_index].add(edge_idx)
                vertex_explorer.Next()

        return vertex_to_edges

    def _analyze_edge(self, edge, edge_idx, edge_to_faces, vertex_to_edges, vertex_map, deflection):
        """Analyze a single edge and extract all metadata."""
        edge_info = {
            "edge_id": edge_idx,
        }

        try:
            # Get curve adaptor
            curve = BRepAdaptor_Curve(edge)

            # Edge type
            curve_type = curve.GetType()
            edge_info["edge_type"] = CURVE_TYPE_NAMES.get(curve_type, f"Unknown({curve_type})")

            # Edge length using BRepGProp
            try:
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                edge_info["length"] = props.Mass()
            except Exception:
                # Fallback: compute from points
                edge_info["length"] = None

            # Get parameter range
            first_param = curve.FirstParameter()
            last_param = curve.LastParameter()

            # Start and end vertices
            try:
                start_pnt = curve.Value(first_param)
                end_pnt = curve.Value(last_param)
                edge_info["vertices"] = {
                    "start": [start_pnt.X(), start_pnt.Y(), start_pnt.Z()],
                    "end": [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]
                }
            except Exception:
                edge_info["vertices"] = None

            # Tangent vectors at start and end
            try:
                pnt_start = gp_Pnt()
                tan_start = gp_Vec()
                curve.D1(first_param, pnt_start, tan_start)
                if tan_start.Magnitude() > 1e-10:
                    tan_start.Normalize()

                pnt_end = gp_Pnt()
                tan_end = gp_Vec()
                curve.D1(last_param, pnt_end, tan_end)
                if tan_end.Magnitude() > 1e-10:
                    tan_end.Normalize()

                edge_info["tangents"] = {
                    "start": [tan_start.X(), tan_start.Y(), tan_start.Z()],
                    "end": [tan_end.X(), tan_end.Y(), tan_end.Z()]
                }
            except Exception:
                edge_info["tangents"] = None

            # Curvature sampling
            edge_info["curvature"] = self._sample_curvature(curve, first_param, last_param)

            # Additional properties for specific curve types
            if curve_type == GeomAbs_Circle:
                try:
                    circle = curve.Circle()
                    edge_info["radius"] = circle.Radius()
                    center = circle.Location()
                    edge_info["center"] = [center.X(), center.Y(), center.Z()]
                except Exception:
                    pass
            elif curve_type == GeomAbs_Ellipse:
                try:
                    ellipse = curve.Ellipse()
                    edge_info["major_radius"] = ellipse.MajorRadius()
                    edge_info["minor_radius"] = ellipse.MinorRadius()
                    center = ellipse.Location()
                    edge_info["center"] = [center.X(), center.Y(), center.Z()]
                except Exception:
                    pass

            # Adjacent faces (convert edge index to 1-indexed for lookup)
            face_indices = edge_to_faces.get(edge_idx + 1, set())
            edge_info["adjacent_faces"] = sorted(list(face_indices))
            edge_info["is_free"] = len(face_indices) == 1

            # Connected edges at vertices
            connected_edges = {"start": [], "end": []}
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            vertex_positions = []

            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                vertex_index = vertex_map.FindIndex(vertex)
                if vertex_index > 0:
                    # Get position for matching
                    pnt = BRep_Tool.Pnt(topods.Vertex(vertex))
                    vertex_positions.append([pnt.X(), pnt.Y(), pnt.Z()])

                    # Get connected edges (excluding this edge)
                    connected = vertex_to_edges.get(vertex_index, set())
                    other_edges = [e - 1 for e in connected if e != edge_idx + 1]  # Convert to 0-indexed

                    # Determine if this is start or end vertex
                    if edge_info.get("vertices"):
                        start_v = edge_info["vertices"]["start"]
                        dist_to_start = sum((a - b) ** 2 for a, b in zip([pnt.X(), pnt.Y(), pnt.Z()], start_v))
                        if dist_to_start < 1e-10:
                            connected_edges["start"] = other_edges
                        else:
                            connected_edges["end"] = other_edges

                vertex_explorer.Next()

            edge_info["connected_edges"] = connected_edges

            # Discretize edge for visualization
            edge_info["points"] = self._discretize_edge(curve, deflection)

        except Exception as e:
            edge_info["error"] = str(e)

        return edge_info

    def _sample_curvature(self, curve, first_param, last_param, num_samples=10):
        """Sample curvature along the edge."""
        curvatures = []

        try:
            for i in range(num_samples):
                t = first_param + (last_param - first_param) * i / (num_samples - 1)

                # Get first and second derivatives
                pnt = gp_Pnt()
                d1 = gp_Vec()
                d2 = gp_Vec()

                try:
                    curve.D2(t, pnt, d1, d2)

                    # Curvature = |d1 x d2| / |d1|^3
                    cross = d1.Crossed(d2)
                    d1_mag = d1.Magnitude()
                    if d1_mag > 1e-10:
                        curvature = cross.Magnitude() / (d1_mag ** 3)
                        curvatures.append(curvature)
                except Exception:
                    continue

            if curvatures:
                return {
                    "min": min(curvatures),
                    "max": max(curvatures),
                    "avg": sum(curvatures) / len(curvatures)
                }
        except Exception:
            pass

        return {"min": 0, "max": 0, "avg": 0}

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

    def _compute_stats(self, values):
        """Compute min/max/avg statistics."""
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }

    def _build_report(self, num_edges, num_vertices, num_faces,
                      bounds_min, bounds_max,
                      edge_type_counts, edge_stats, free_edge_count):
        """Build text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CAD EDGE ANALYSIS REPORT")
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
        if edge_stats:
            lines.append("--- LENGTH STATISTICS ---")
            lines.append(f"Min: {edge_stats['min']:.4g}")
            lines.append(f"Max: {edge_stats['max']:.4g}")
            lines.append(f"Avg: {edge_stats['avg']:.4g}")
            lines.append(f"Total: {edge_stats['total']:.4g}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class CADEdgeViewer:
    """
    Interactive edge viewer node.

    Takes the edge analysis JSON file and displays an interactive 3D viewer
    with edge selection, highlighting, and info panel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edge_json_filepath": ("STRING", {
                    "tooltip": "Path to edge analysis JSON from CADEdgeAnalysis"
                }),
                "visualization_mode": (["normal", "edge_type_heatmap", "length_heatmap"], {
                    "default": "normal",
                    "tooltip": "Color mode: normal (free=red, shared=black), edge type heatmap, or length heatmap"
                }),
                "selected_edge_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Edge ID to pass to downstream nodes (set to the ID shown when you click an edge)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("edge_json_filepath", "selected_edge_id")
    OUTPUT_NODE = True
    FUNCTION = "show_viewer"
    CATEGORY = "CADabra/Visualization"
    DESCRIPTION = """
    Interactive 3D viewer for edge analysis results.

    Features:
    - Click on edges to select and highlight them
    - View edge type, length, curvature, and connectivity
    - Adjacent faces and connected edges shown in info panel
    - Multiple visualization modes for edge coloring

    Outputs:
    - edge_json_filepath: Pass-through for chaining to detail analyzer
    - selected_edge_id: The edge ID input (set this to the ID you clicked)
    """

    def show_viewer(self, edge_json_filepath, visualization_mode="normal", selected_edge_id=0):
        """Display interactive edge viewer."""
        # Extract filename from path
        edge_filename = os.path.basename(edge_json_filepath)

        # Load JSON to get metadata for the info panel
        try:
            with open(edge_json_filepath, 'r') as f:
                analysis_data = json.load(f)

            num_edges = analysis_data.get("num_edges", 0)
            num_vertices = analysis_data.get("num_vertices", 0)
            num_faces = analysis_data.get("num_faces", 0)
            bounds_min = analysis_data.get("bounds_min", [0, 0, 0])
            bounds_max = analysis_data.get("bounds_max", [0, 0, 0])
            edge_type_counts = analysis_data.get("edge_type_counts", {})
            edge_stats = analysis_data.get("edge_stats", {})
            free_edge_count = analysis_data.get("free_edge_count", 0)
        except Exception as e:
            print(f"[CADabra] Warning: Could not load edge analysis JSON: {e}")
            num_edges = num_vertices = num_faces = 0
            bounds_min = bounds_max = [0, 0, 0]
            edge_type_counts = {}
            edge_stats = {}
            free_edge_count = 0

        return {
            "ui": {
                "edge_file": [edge_filename],
                "format": ["json"],
                "num_edges": [num_edges],
                "num_vertices": [num_vertices],
                "num_faces": [num_faces],
                "bounds_min": [bounds_min],
                "bounds_max": [bounds_max],
                "edge_type_counts": [edge_type_counts],
                "edge_stats": [edge_stats],
                "free_edge_count": [free_edge_count],
                "visualization_mode": [visualization_mode],
                "selected_edge_id": [selected_edge_id],
            },
            "result": (edge_json_filepath, selected_edge_id)
        }


class CADEdgeDetailAnalyzer:
    """
    Detailed edge analyzer with 2D visualization and full parameter display.

    Takes an edge ID and displays the edge curve in a 2D canvas with all
    geometric parameters shown below (radius for circles, control points
    for splines, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edge_json_filepath": ("STRING", {
                    "tooltip": "Path to edge analysis JSON from CADEdgeAnalysis"
                }),
                "edge_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Edge ID to analyze in detail"
                }),
                "projection_plane": (["auto", "XY", "XZ", "YZ"], {
                    "default": "auto",
                    "tooltip": "Plane to project edge onto for 2D view"
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "analyze_edge_detail"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Detailed 2D analysis of a single edge.

    Features:
    - 2D plot of the edge curve (projected onto best-fit plane)
    - Full parameter display based on edge type:
      - Line: direction, length
      - Circle: center, radius, arc angles
      - Ellipse: center, major/minor radii
      - BSpline/Bezier: control points, degree
    - Curvature visualization
    """

    def analyze_edge_detail(self, edge_json_filepath, edge_id=0, projection_plane="auto"):
        """Analyze a single edge in detail."""
        # Load JSON
        try:
            with open(edge_json_filepath, 'r') as f:
                analysis_data = json.load(f)
        except Exception as e:
            print(f"[CADabra] Error loading edge JSON: {e}")
            return {"ui": {"error": [str(e)]}}

        # Find the edge
        edges = analysis_data.get("edges", [])
        edge_data = None
        for edge in edges:
            if edge.get("edge_id") == edge_id:
                edge_data = edge
                break

        if edge_data is None:
            return {"ui": {"error": [f"Edge ID {edge_id} not found"]}}

        # Determine best projection plane if auto
        if projection_plane == "auto":
            projection_plane = self._find_best_projection_plane(edge_data)

        # Project points to 2D
        points_2d = self._project_points(edge_data.get("points", []), projection_plane)

        # Build detailed parameter info
        params = self._extract_detailed_params(edge_data)

        return {
            "ui": {
                "edge_id": [edge_id],
                "edge_type": [edge_data.get("edge_type", "Unknown")],
                "length": [edge_data.get("length")],
                "is_free": [edge_data.get("is_free", False)],
                "projection_plane": [projection_plane],
                "points_2d": [points_2d],
                "points_3d": [edge_data.get("points", [])],
                "vertices": [edge_data.get("vertices")],
                "tangents": [edge_data.get("tangents")],
                "curvature": [edge_data.get("curvature")],
                "params": [params],
                "adjacent_faces": [edge_data.get("adjacent_faces", [])],
                "connected_edges": [edge_data.get("connected_edges", {})],
            }
        }

    def _find_best_projection_plane(self, edge_data):
        """Find the best projection plane based on edge extent."""
        points = edge_data.get("points", [])
        if len(points) < 2:
            return "XY"

        # Compute extent in each axis
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        extent_x = max(xs) - min(xs) if xs else 0
        extent_y = max(ys) - min(ys) if ys else 0
        extent_z = max(zs) - min(zs) if zs else 0

        # Choose plane with smallest extent (most flat)
        if extent_z <= extent_x and extent_z <= extent_y:
            return "XY"  # Flat in Z, view from Z
        elif extent_y <= extent_x and extent_y <= extent_z:
            return "XZ"  # Flat in Y, view from Y
        else:
            return "YZ"  # Flat in X, view from X

    def _project_points(self, points, plane):
        """Project 3D points onto a 2D plane."""
        points_2d = []
        for p in points:
            if plane == "XY":
                points_2d.append([p[0], p[1]])
            elif plane == "XZ":
                points_2d.append([p[0], p[2]])
            elif plane == "YZ":
                points_2d.append([p[1], p[2]])
        return points_2d

    def _extract_detailed_params(self, edge_data):
        """Extract detailed parameters based on edge type."""
        params = {}
        edge_type = edge_data.get("edge_type", "Unknown")

        # Common params
        if edge_data.get("length") is not None:
            params["length"] = edge_data["length"]

        if edge_data.get("vertices"):
            params["start_point"] = edge_data["vertices"]["start"]
            params["end_point"] = edge_data["vertices"]["end"]

        if edge_data.get("tangents"):
            params["start_tangent"] = edge_data["tangents"]["start"]
            params["end_tangent"] = edge_data["tangents"]["end"]

        if edge_data.get("curvature"):
            params["curvature_min"] = edge_data["curvature"]["min"]
            params["curvature_max"] = edge_data["curvature"]["max"]
            params["curvature_avg"] = edge_data["curvature"]["avg"]

        # Type-specific params
        if edge_type == "Circle":
            if edge_data.get("radius") is not None:
                params["radius"] = edge_data["radius"]
            if edge_data.get("center") is not None:
                params["center"] = edge_data["center"]

        elif edge_type == "Ellipse":
            if edge_data.get("major_radius") is not None:
                params["major_radius"] = edge_data["major_radius"]
            if edge_data.get("minor_radius") is not None:
                params["minor_radius"] = edge_data["minor_radius"]
            if edge_data.get("center") is not None:
                params["center"] = edge_data["center"]

        elif edge_type == "Line":
            # Compute direction vector
            if edge_data.get("vertices"):
                start = edge_data["vertices"]["start"]
                end = edge_data["vertices"]["end"]
                direction = [end[i] - start[i] for i in range(3)]
                mag = sum(d**2 for d in direction) ** 0.5
                if mag > 1e-10:
                    direction = [d / mag for d in direction]
                params["direction"] = direction

        return params


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADEdgeAnalysis": CADEdgeAnalysis,
    "CADEdgeViewer": CADEdgeViewer,
    "CADEdgeDetailAnalyzer": CADEdgeDetailAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADEdgeAnalysis": "CAD Edge Analysis",
    "CADEdgeViewer": "CAD Edge Viewer",
    "CADEdgeDetailAnalyzer": "CAD Edge Detail Analyzer",
}
