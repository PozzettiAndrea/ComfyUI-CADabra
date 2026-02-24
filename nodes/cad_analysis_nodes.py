from __future__ import annotations

import json
import logging
import os
import time
import folder_paths

from .utils.occ_logging import log_operation
from OCC.Core.TopExp import TopExp_Explorer, topexp

log = logging.getLogger("cadabra")
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_WIRE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
)
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
import vtk

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
        GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_OtherSurface: "Other",
    }


class CADFaceAnalysis:
    """
    Unified CAD face analysis node.

    Combines face-level analysis (surface types, curvature, adjacency) with
    topology diagnostics (sewing complexity, free edges, problematic faces).

    Outputs file paths for VTP mesh and JSON metadata, plus a combined text report.
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
                    "tooltip": "Linear deflection for tessellation (lower = finer mesh)"
                }),
                "angular_deflection": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Angular deflection in radians for tessellation"
                }),
                "tolerance": ("FLOAT", {
                    "default": 1.0,
                    "min": 1e-6,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Tolerance for detecting tiny faces and sewing analysis"
                }),
                "extract_control_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract NURBS/BSpline control points (may be slow for complex models)"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cad_model", "vtp_filepath", "json_filepath", "info", "top_30_by_edges", "top_30_smallest_area")
    OUTPUT_NODE = True
    FUNCTION = "analyze_cad"
    CATEGORY = "CADabra/Analysis"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, False, True, True)  # report is concatenated, others are lists
    DESCRIPTION = """
    Unified CAD analysis node combining face inspection and topology diagnostics.

    Outputs:
    - vtp_filepath: Path to tessellated mesh with face IDs (for viewer)
    - json_filepath: Path to JSON metadata with per-face analysis
    - report: Combined text report with topology summary, surface types,
      sewing complexity, and face statistics
    - top_30_by_edges: Top 30 faces by edge count (sorted descending), format: "id:edges, ..."
    - top_30_smallest_area: Top 30 smallest faces by area (sorted ascending), format: "id:area, ..."
    """

    def analyze_cad(self, cad_model, linear_deflection=0.1, angular_deflection=0.5,
                    tolerance=1.0, extract_control_points=True):
        """
        Analyze CAD model(s) and extract comprehensive per-face metadata.
        Supports batch processing when INPUT_IS_LIST=True.
        """
        # Extract scalar values from lists (INPUT_IS_LIST behavior)
        linear_deflection_val = linear_deflection[0] if isinstance(linear_deflection, list) else linear_deflection
        angular_deflection_val = angular_deflection[0] if isinstance(angular_deflection, list) else angular_deflection
        tolerance_val = tolerance[0] if isinstance(tolerance, list) else tolerance
        extract_control_points_val = extract_control_points[0] if isinstance(extract_control_points, list) else extract_control_points

        # Ensure cad_model is a list
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        # Collect results for batch
        all_vtp_paths = []
        all_json_paths = []
        all_reports = []
        all_top_30_edges = []
        all_top_30_smallest = []

        for model_idx, cm in enumerate(cad_models):
            # Get filename for this model
            file_path = cm.get("file_path", f"model_{model_idx}")
            filename = os.path.basename(file_path)

            log.info(f"Analyzing model {model_idx + 1}/{len(cad_models)}: {filename}")

            vtp_path, json_path, report, top_30_edges, top_30_smallest = self._analyze_single(
                cm, linear_deflection_val, angular_deflection_val, tolerance_val, extract_control_points_val, filename
            )

            all_vtp_paths.append(vtp_path)
            all_json_paths.append(json_path)
            all_reports.append(report)
            all_top_30_edges.append(top_30_edges)
            all_top_30_smallest.append(top_30_smallest)

        # Concatenate reports with newlines
        combined_report = "\n\n".join(all_reports)

        return {"ui": {"text": (combined_report,)}, "result": (cad_models, all_vtp_paths, all_json_paths, combined_report, all_top_30_edges, all_top_30_smallest)}

    def _analyze_single(self, cad_model, linear_deflection, angular_deflection, tolerance, extract_control_points, filename):
        """
        Analyze a single CAD model and return results.
        """
        # Get OCC shape from brep_path
        try:
            from .utils.brep_cache import get_occ_shape
        except ImportError:
            from .utils.brep_cache import get_occ_shape
        occ_shape = get_occ_shape(cad_model)

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_analysis_{timestamp}"
        json_filename = f"{base_filename}_analysis.json"
        json_path = os.path.join(output_dir, json_filename)

        # Count entities
        num_volumes = sum(1 for _ in self._iter_occ(occ_shape, TopAbs_SOLID))
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
        extents = [xmax - xmin, ymax - ymin, zmax - zmin]

        log.info(f"OCC Analysis: {num_faces} faces, {num_edges} edges, {num_volumes} volumes")

        # Analyze each face
        face_data = []
        faces_list = list(self._iter_occ(occ_shape, TopAbs_FACE))

        with log_operation("CAD Face Analysis", faces=num_faces):
            for face_idx, face in enumerate(faces_list):
                face_info = self._analyze_face_occ(face, face_idx, extract_control_points)
                face_data.append(face_info)

        # Build face adjacency graph and count free edges
        log.info("Building face adjacency graph...")
        with log_operation("CAD Adjacency Graph", faces=num_faces):
            adjacency, free_edge_count, edge_to_faces = self._build_adjacency_with_free_edges(occ_shape, faces_list)

        # Add adjacency info to each face
        for face_info in face_data:
            face_info["adjacent_faces"] = adjacency.get(face_info["face_id"], [])

        # Count connected components via BFS
        component_count = self._count_connected_components(adjacency, len(faces_list))

        # Compute adjacency stats
        total_adjacencies = sum(len(v) for v in adjacency.values()) // 2
        avg_neighbors = sum(len(v) for v in adjacency.values()) / len(adjacency) if adjacency else 0
        isolated_faces = sum(1 for v in adjacency.values() if len(v) == 0)

        log.info(f"Adjacency: {total_adjacencies} shared edges, avg {avg_neighbors:.1f} neighbors/face, {isolated_faces} isolated")
        log.info(f"Topology: {component_count} components, {free_edge_count} free edges")

        # Compute surface type distribution
        surface_type_counts = {}
        for face_info in face_data:
            st = face_info.get("surface_type", "Unknown")
            surface_type_counts[st] = surface_type_counts.get(st, 0) + 1

        # Compute face statistics
        edges_per_face = [f.get("edge_count", 0) for f in face_data]
        face_areas = [f.get("area", 0) for f in face_data if f.get("area") is not None]

        edge_stats = self._compute_stats(edges_per_face)
        area_stats = self._compute_stats(face_areas) if face_areas else None

        # Detect problematic faces
        high_edge_faces = [(i, e) for i, e in enumerate(edges_per_face) if e > 100]
        tol_squared = tolerance * tolerance
        tiny_faces = [(i, a) for i, a in enumerate(face_areas) if a < tol_squared] if face_areas else []

        # Sewing complexity analysis
        sewing_complexity = self._analyze_sewing_complexity(
            free_edge_count, component_count, num_faces
        )

        # Extract edge geometry for visualization
        log.info("Extracting edge geometry...")
        edges_geometry = self._extract_edge_geometry(occ_shape, edge_to_faces, linear_deflection)
        free_edges_count = sum(1 for e in edges_geometry if e["is_free"])
        shared_edges_count = len(edges_geometry) - free_edges_count
        log.info(f"Extracted {len(edges_geometry)} edges ({free_edges_count} free, {shared_edges_count} shared)")

        # Build JSON metadata
        analysis_data = {
            "timestamp": timestamp,
            "filename": filename,
            "num_faces": num_faces,
            "num_edges": num_edges,
            "num_vertices": num_vertices,
            "num_volumes": num_volumes,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "extents": extents,
            "linear_deflection": linear_deflection,
            "analysis_mode": "occ",
            "sewing_complexity": sewing_complexity,
            "adjacency_stats": {
                "total_shared_edges": total_adjacencies,
                "avg_neighbors": round(avg_neighbors, 2),
                "isolated_faces": isolated_faces,
            },
            "surface_type_counts": surface_type_counts,
            "edge_stats": edge_stats,
            "area_stats": area_stats,
            "faces": face_data,
            "edges": edges_geometry
        }

        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        log.info(f"OCC Analysis complete: {json_filename}")

        # Tessellate shape for mesh export
        log.info(f"Tessellating with linear_deflection={linear_deflection}, angular_deflection={angular_deflection}")
        mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()

        # Build face edge counts dict for VTP export
        face_edge_counts = {f["face_id"]: f.get("edge_count", 0) for f in face_data}

        # Export mesh with face IDs
        mesh_filename = self._export_mesh_with_face_ids(
            occ_shape, base_filename, output_dir, linear_deflection, face_edge_counts
        )

        # Build combined text report
        report = self._build_report(
            num_faces, num_edges, num_vertices, num_volumes,
            component_count, free_edge_count,
            bounds_min, bounds_max, extents,
            sewing_complexity, surface_type_counts,
            edge_stats, area_stats, high_edge_faces, tiny_faces,
            total_adjacencies, avg_neighbors, isolated_faces,
            tolerance, filename
        )

        # Return file paths and report
        vtp_path = os.path.join(output_dir, mesh_filename)

        # Build top 30 faces by edge count (sorted descending)
        faces_with_edges = [(i, e) for i, e in enumerate(edges_per_face)]
        faces_with_edges.sort(key=lambda x: x[1], reverse=True)
        top_30_edges = faces_with_edges[:30]
        top_30_by_edges = ", ".join(f"{face_idx}:{edge_count}" for face_idx, edge_count in top_30_edges)

        # Build top 30 smallest faces by area (sorted ascending)
        faces_with_area = [(i, a) for i, a in enumerate(face_areas)]
        faces_with_area.sort(key=lambda x: x[1])
        top_30_smallest = faces_with_area[:30]
        top_30_smallest_area = ", ".join(f"{face_idx}:{area:.6g}" for face_idx, area in top_30_smallest)

        return vtp_path, json_path, report, top_30_by_edges, top_30_smallest_area

    def _iter_occ(self, shape, topology_type):
        """Helper to iterate over OCC TopExp_Explorer results."""
        explorer = TopExp_Explorer(shape, topology_type)
        while explorer.More():
            yield explorer.Current()
            explorer.Next()

    def _analyze_face_occ(self, face, face_idx, extract_control_points):
        """Analyze a single face using pure OCC."""
        face_info = {
            "face_id": face_idx,
        }

        try:
            # Get surface type
            adaptor = BRepAdaptor_Surface(topods.Face(face))
            surface_type = adaptor.GetType()
            face_info["surface_type"] = _get_surface_type_names().get(surface_type, f"Unknown({surface_type})")

            # Get bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)
            if not bbox.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                face_info["bbox_min"] = [xmin, ymin, zmin]
                face_info["bbox_max"] = [xmax, ymax, zmax]
                face_info["centroid"] = [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2
                ]

            # Get area and center of mass
            try:
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                face_info["area"] = props.Mass()

                com = props.CentreOfMass()
                face_info["centroid"] = [com.X(), com.Y(), com.Z()]
            except Exception:
                pass

            # Get normal at center
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
                    face_info["normal"] = [normal.X(), normal.Y(), normal.Z()]
            except Exception:
                pass

            # Count edges for this face
            edge_count = 0
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge_count += 1
                edge_explorer.Next()
            face_info["edge_count"] = edge_count

            # Curvature analysis based on surface type
            if surface_type == GeomAbs_Plane:
                face_info["curvature_type"] = "Planar"
                face_info["is_planar"] = True
            elif surface_type in (GeomAbs_Cylinder, GeomAbs_Cone):
                face_info["curvature_type"] = "Cylindrical/Conical"
                face_info["is_planar"] = False
            elif surface_type == GeomAbs_Sphere:
                face_info["curvature_type"] = "Spherical"
                face_info["is_planar"] = False
            elif surface_type in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
                face_info["curvature_type"] = "Freeform/NURBS"
                face_info["is_planar"] = False
            else:
                face_info["curvature_type"] = "Other"
                face_info["is_planar"] = None

        except Exception as e:
            face_info["error"] = str(e)

        return face_info

    def _build_adjacency_with_free_edges(self, occ_shape, faces_list):
        """Build face adjacency graph and count free edges."""
        if len(faces_list) == 0:
            return {}, 0, {}

        # Build indexed map of all edges
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ_shape, TopAbs_EDGE, edge_map)

        # Build edge to faces mapping
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

        # Build adjacency and count free edges
        adjacency = {i: set() for i in range(len(faces_list))}
        free_edge_count = 0

        for edge_index, face_indices in edge_to_faces.items():
            if len(face_indices) == 1:
                free_edge_count += 1
            face_list = list(face_indices)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adjacency[face_list[i]].add(face_list[j])
                    adjacency[face_list[j]].add(face_list[i])

        return {k: sorted(list(v)) for k, v in adjacency.items()}, free_edge_count, edge_to_faces

    def _count_connected_components(self, adjacency, num_faces):
        """Count connected components via BFS."""
        visited = set()
        component_count = 0

        for start_face in range(num_faces):
            if start_face in visited:
                continue
            queue = [start_face]
            visited.add(start_face)
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component_count += 1

        return component_count

    def _extract_edge_geometry(self, occ_shape, edge_to_faces, deflection=0.1):
        """Extract discretized edge curves with free/shared classification."""
        edges_data = []

        # Build indexed map of all edges
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(occ_shape, TopAbs_EDGE, edge_map)

        # Iterate over edges we found in edge_to_faces (from face exploration)
        for edge_index, face_set in edge_to_faces.items():
            try:
                edge = topods.Edge(edge_map.FindKey(edge_index))

                # Discretize the edge curve
                curve = BRepAdaptor_Curve(edge)
                discretizer = GCPnts_UniformDeflection(curve, deflection)

                if discretizer.IsDone() and discretizer.NbPoints() >= 2:
                    points = []
                    for j in range(1, discretizer.NbPoints() + 1):
                        pnt = discretizer.Value(j)
                        points.append([pnt.X(), pnt.Y(), pnt.Z()])

                    # Check if free edge (only 1 face) or shared (2+ faces)
                    is_free = (len(face_set) == 1)

                    edges_data.append({
                        "points": points,
                        "is_free": is_free
                    })
            except Exception as e:
                # Skip problematic edges (degenerate, etc.)
                continue

        return edges_data

    def _compute_stats(self, values):
        """Compute min/max/avg statistics."""
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }

    def _analyze_sewing_complexity(self, free_edge_count, component_count, num_faces):
        """Analyze sewing complexity based on free edges."""
        if free_edge_count == 0:
            return {
                "free_edges": 0,
                "connected_components": component_count,
                "complexity_level": "none",
                "edge_pairs_to_compare": 0,
                "all_disconnected": False
            }

        edge_pairs = free_edge_count * (free_edge_count - 1) // 2
        all_disconnected = (component_count == num_faces and num_faces > 1)

        if free_edge_count > 50000:
            complexity_level = "danger"
        elif free_edge_count > 10000:
            complexity_level = "warning"
        elif free_edge_count > 1000:
            complexity_level = "moderate"
        else:
            complexity_level = "low"

        return {
            "free_edges": free_edge_count,
            "connected_components": component_count,
            "complexity_level": complexity_level,
            "edge_pairs_to_compare": edge_pairs,
            "all_disconnected": all_disconnected
        }

    def _build_report(self, num_faces, num_edges, num_vertices, num_volumes,
                      component_count, free_edge_count,
                      bounds_min, bounds_max, extents,
                      sewing_complexity, surface_type_counts,
                      edge_stats, area_stats, high_edge_faces, tiny_faces,
                      total_adjacencies, avg_neighbors, isolated_faces,
                      tolerance, filename="unknown"):
        """Build combined text report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"CAD ANALYSIS REPORT: {filename}")
        lines.append("=" * 60)
        lines.append("")

        # Topology summary
        lines.append("--- TOPOLOGY SUMMARY ---")
        lines.append(f"Faces: {num_faces} | Edges: {num_edges} | Vertices: {num_vertices}")
        lines.append(f"Volumes: {num_volumes}")
        lines.append(f"Connected components: {component_count}")
        lines.append(f"Free edges: {free_edge_count}")
        lines.append("")

        # Bounding box
        lines.append("--- BOUNDING BOX ---")
        lines.append(f"Min: ({bounds_min[0]:.2f}, {bounds_min[1]:.2f}, {bounds_min[2]:.2f})")
        lines.append(f"Max: ({bounds_max[0]:.2f}, {bounds_max[1]:.2f}, {bounds_max[2]:.2f})")
        lines.append(f"Extents: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}")
        lines.append("")

        # Sewing complexity
        lines.append("--- SEWING COMPLEXITY ---")
        lines.append(f"Complexity: {sewing_complexity['complexity_level'].upper()}")
        lines.append(f"Free edges: {sewing_complexity['free_edges']}")
        lines.append(f"Edge pairs to compare: {sewing_complexity['edge_pairs_to_compare']:,}")
        if sewing_complexity.get('all_disconnected'):
            lines.append("WARNING: Every face is disconnected (typical IGES import)")
        if sewing_complexity['complexity_level'] == 'danger':
            lines.append("DANGER: >50k free edges - sewing will likely hang!")
            lines.append("SUGGESTION: Use CADSewFacesParallel with timeout")
        elif sewing_complexity['complexity_level'] == 'warning':
            lines.append("WARNING: Very slow sewing expected")
            lines.append("SUGGESTION: Consider CADSewFacesParallel with timeout")
        lines.append("")

        # Surface types
        lines.append("--- SURFACE TYPES ---")
        for stype, count in sorted(surface_type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"{stype}: {count} faces")
        lines.append("")

        # Face statistics
        lines.append("--- FACE STATISTICS ---")
        if edge_stats:
            lines.append(f"Edges per face: {edge_stats['min']} - {edge_stats['max']} (avg: {edge_stats['avg']:.1f})")
        if area_stats:
            lines.append(f"Area range: {area_stats['min']:.6g} - {area_stats['max']:.6g} (avg: {area_stats['avg']:.6g})")
        lines.append(f"Tiny faces (< {tolerance}²): {len(tiny_faces)}")
        lines.append(f"Problematic faces (>100 edges): {len(high_edge_faces)}")

        if high_edge_faces:
            lines.append("  High-edge faces suggest fragmented trim curves")
            for face_idx, ec in sorted(high_edge_faces, key=lambda x: -x[1])[:3]:
                lines.append(f"    Face {face_idx}: {ec} edges")

        if tiny_faces:
            lines.append("  Tiny faces can cause sewing issues")
        lines.append("")

        # Adjacency
        lines.append("--- ADJACENCY ---")
        lines.append(f"Total shared edges: {total_adjacencies}")
        lines.append(f"Avg neighbors per face: {avg_neighbors:.1f}")
        lines.append(f"Isolated faces: {isolated_faces}")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _export_mesh_with_face_ids(self, occ_shape, base_filename, output_dir, linear_deflection, face_edge_counts=None):
        """Export tessellated mesh with face IDs and edge counts."""
        if face_edge_counts is None:
            face_edge_counts = {}

        # Extract triangulation from each face
        all_vertices = []
        all_indices = []
        face_ranges = []

        vertex_offset = 0
        triangle_start = 0

        for face_idx, face in enumerate(self._iter_occ(occ_shape, TopAbs_FACE)):
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(topods.Face(face), loc)

            if triangulation is None:
                face_ranges.append({
                    "face_id": face_idx,
                    "triangle_start": triangle_start,
                    "triangle_count": 0
                })
                continue

            trsf = loc.Transformation()
            num_nodes = triangulation.NbNodes()
            num_triangles = triangulation.NbTriangles()

            # Extract vertices
            for i in range(1, num_nodes + 1):
                pnt = triangulation.Node(i).Transformed(trsf)
                all_vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])

            # Extract triangles
            for i in range(1, num_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                all_indices.extend([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset
                ])

            face_ranges.append({
                "face_id": face_idx,
                "triangle_start": triangle_start,
                "triangle_count": num_triangles
            })

            vertex_offset += num_nodes
            triangle_start += num_triangles

        total_vertices = len(all_vertices) // 3
        total_triangles = len(all_indices) // 3
        log.info(f"Mesh: {total_vertices} vertices, {total_triangles} triangles")

        return self._export_vtp(
            all_vertices, all_indices, face_ranges,
            base_filename, output_dir, face_edge_counts
        )

    def _export_vtp(self, vertices, indices, face_ranges, base_filename, output_dir, face_edge_counts=None):
        """Export mesh as VTP with FaceID and EdgeCount cell data."""
        if face_edge_counts is None:
            face_edge_counts = {}

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        face_ids = vtk.vtkIntArray()
        face_ids.SetName("FaceID")
        edge_counts = vtk.vtkIntArray()
        edge_counts.SetName("EdgeCount")

        # Add all vertices
        for i in range(0, len(vertices), 3):
            points.InsertNextPoint(vertices[i], vertices[i+1], vertices[i+2])

        # Build face ID lookup
        triangle_to_face = {}
        for fr in face_ranges:
            for t in range(fr["triangle_start"], fr["triangle_start"] + fr["triangle_count"]):
                triangle_to_face[t] = fr["face_id"]

        # Add all triangles with face IDs
        for tri_idx in range(0, len(indices), 3):
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, indices[tri_idx])
            triangle.GetPointIds().SetId(1, indices[tri_idx + 1])
            triangle.GetPointIds().SetId(2, indices[tri_idx + 2])
            polys.InsertNextCell(triangle)

            face_id = triangle_to_face.get(tri_idx // 3, -1)
            face_ids.InsertNextValue(face_id)
            edge_counts.InsertNextValue(face_edge_counts.get(face_id, 0))

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)
        polydata.GetCellData().AddArray(face_ids)
        polydata.GetCellData().AddArray(edge_counts)

        # Write VTP
        vtp_filename = f"{base_filename}.vtp"
        vtp_path = os.path.join(output_dir, vtp_filename)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_path)
        writer.SetInputData(polydata)
        writer.SetDataModeToAscii()
        writer.Write()

        file_size = os.path.getsize(vtp_path)
        log.info(f"Exported VTP mesh: {vtp_filename} ({file_size:,} bytes)")
        return vtp_filename

    def _export_mesh_json(self, vertices, indices, face_ranges, base_filename, output_dir, face_edge_counts=None):
        """Export mesh as JSON (fallback if VTK not available)."""
        if face_edge_counts is None:
            face_edge_counts = {}
        mesh_data = {
            "vertices": vertices,
            "indices": indices,
            "face_ranges": face_ranges,
            "face_edge_counts": face_edge_counts
        }

        json_filename = f"{base_filename}_mesh.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(mesh_data, f)

        file_size = os.path.getsize(json_path)
        log.info(f"Exported mesh JSON: {json_filename} ({file_size:,} bytes)")
        return json_filename


class CADAnalysisViewer:
    """
    Interactive analysis viewer node.

    Takes VTP and JSON file paths as input (from CADFaceAnalysis) and displays
    an interactive 3D viewer with face selection, neighbor highlighting, and
    face info panel.

    Supports batch processing with index-based navigation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "Index of model to display in batch"
                }),
                "vtp_filepath": ("STRING", {
                    "tooltip": "Path to VTP mesh file from CADFaceAnalysis"
                }),
                "json_filepath": ("STRING", {
                    "tooltip": "Path to JSON analysis file from CADFaceAnalysis"
                }),
                "visualization_mode": (["normal", "edge_count_heatmap", "connectivity_inspection"], {
                    "default": "normal",
                    "tooltip": "Color mode: normal, edge count heatmap, or connectivity inspection (neighbor count)"
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_viewer"
    CATEGORY = "CADabra/Visualization"
    INPUT_IS_LIST = True
    DESCRIPTION = """
    Interactive 3D viewer for CAD analysis results.

    Features:
    - Click on faces to select and highlight them
    - View surface type, area, curvature, and adjacency
    - Neighbor highlighting shows connected faces
    - Edge count heatmap mode for topology inspection
    """

    def show_viewer(self, index, vtp_filepath, json_filepath, visualization_mode="normal"):
        """Display interactive viewer with analysis data. Supports batch processing."""
        # Extract scalar values from lists (INPUT_IS_LIST behavior)
        index_val = index[0] if isinstance(index, list) else index
        visualization_mode_val = visualization_mode[0] if isinstance(visualization_mode, list) else visualization_mode

        # Ensure file paths are lists
        vtp_paths = vtp_filepath if isinstance(vtp_filepath, list) else [vtp_filepath]
        json_paths = json_filepath if isinstance(json_filepath, list) else [json_filepath]

        # Get batch size and clamp index
        batch_size = len(vtp_paths)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current files from batch
        current_vtp = vtp_paths[actual_index]
        current_json = json_paths[actual_index]

        # Extract just the filename from the path for the viewer
        vtp_filename = os.path.basename(current_vtp)
        json_filename = os.path.basename(current_json)

        log.info(f"Analysis Viewer: showing {actual_index + 1}/{batch_size}: {vtp_filename}")

        # Load JSON to get metadata for the info panel
        try:
            with open(current_json, 'r') as f:
                analysis_data = json.load(f)

            num_faces = analysis_data.get("num_faces", 0)
            num_edges = analysis_data.get("num_edges", 0)
            num_volumes = analysis_data.get("num_volumes", 0)
            bounds_min = analysis_data.get("bounds_min", [0, 0, 0])
            bounds_max = analysis_data.get("bounds_max", [0, 0, 0])
            extents = analysis_data.get("extents", [0, 0, 0])
            linear_deflection = analysis_data.get("linear_deflection", 0.1)
        except Exception as e:
            log.warning(f"Could not load analysis JSON: {e}")
            num_faces = num_edges = num_volumes = 0
            bounds_min = bounds_max = extents = [0, 0, 0]
            linear_deflection = 0.1

        return {
            "ui": {
                "mesh_file": [vtp_filename],
                "analysis_file": [json_filename],
                "format": ["occ"],
                "original_format": ["analysis"],
                "num_volumes": [num_volumes],
                "num_faces": [num_faces],
                "num_edges": [num_edges],
                "bounds_min": [bounds_min],
                "bounds_max": [bounds_max],
                "extents": [extents],
                "linear_deflection": [linear_deflection],
                "visualization_mode": [visualization_mode_val],
                "batch_size": [batch_size],
                "current_index": [actual_index],
            }
        }


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADFaceAnalysis": CADFaceAnalysis,
    "CADAnalysisViewer": CADAnalysisViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADFaceAnalysis": "CAD Face Analysis",
    "CADAnalysisViewer": "CAD Analysis Viewer",
}
