from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_BezierCurve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Circ, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BRepTools import BRepTools_ReShape
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from .utils.occ_logging import logger


def _sample_curve_points(edge: TopoDS_Edge, n_samples: int = 20) -> np.ndarray:
    """
    Sample uniformly distributed points along an edge curve.

    Returns:
        numpy array of shape (n_samples, 3)
    """
    adaptor = BRepAdaptor_Curve(edge)
    u_start = adaptor.FirstParameter()
    u_end = adaptor.LastParameter()

    # Use uniform abscissa for even spacing along curve length
    try:
        uniform = GCPnts_UniformAbscissa(adaptor, n_samples)
        if uniform.IsDone() and uniform.NbPoints() >= 3:
            points = []
            for i in range(1, uniform.NbPoints() + 1):
                u = uniform.Parameter(i)
                pnt = adaptor.Value(u)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])
            return np.array(points)
    except Exception:
        pass

    # Fallback: uniform parametric spacing
    params = np.linspace(u_start, u_end, n_samples)
    points = []
    for u in params:
        pnt = adaptor.Value(u)
        points.append([pnt.X(), pnt.Y(), pnt.Z()])
    return np.array(points)


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a plane to 3D points using SVD.

    Returns:
        (centroid, normal, max_deviation)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD decomposition
    U, S, Vt = np.linalg.svd(centered)

    # Normal is the eigenvector with smallest singular value (last row of Vt)
    normal = Vt[-1]

    # Max deviation from plane
    deviations = np.abs(centered @ normal)
    max_dev = np.max(deviations)

    return centroid, normal, max_dev


def _fit_circle_2d(points_2d: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit a circle to 2D points using algebraic method (Kasa).

    Returns:
        (center_x, center_y, radius, max_deviation)
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    n = len(x)

    # Build system: A * [a, b, c]^T = B
    # where circle is: (x-a)^2 + (y-b)^2 = r^2
    # Rewritten: x^2 + y^2 = 2ax + 2by + (r^2 - a^2 - b^2)
    # Let c = r^2 - a^2 - b^2, then: 2ax + 2by + c = x^2 + y^2

    A = np.column_stack([2*x, 2*y, np.ones(n)])
    B = x**2 + y**2

    try:
        # Solve least squares
        result, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        a, b, c = result

        # Radius: r^2 = c + a^2 + b^2
        r_sq = c + a**2 + b**2
        if r_sq <= 0:
            return 0, 0, 0, float('inf')
        r = np.sqrt(r_sq)

        # Compute max deviation
        distances = np.sqrt((x - a)**2 + (y - b)**2)
        deviations = np.abs(distances - r)
        max_dev = np.max(deviations)

        return a, b, r, max_dev
    except Exception:
        return 0, 0, 0, float('inf')


def _project_to_plane(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points onto a plane and return 2D coordinates.

    Returns:
        (points_2d, axis_u, axis_v) where axis_u and axis_v are the local 2D axes
    """
    # Create orthonormal basis on the plane
    normal = normal / np.linalg.norm(normal)

    # Find a vector not parallel to normal
    if abs(normal[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])

    # Create orthonormal basis
    axis_u = np.cross(normal, ref)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    # Project points
    centered = points - centroid
    points_2d = np.column_stack([
        centered @ axis_u,
        centered @ axis_v
    ])

    return points_2d, axis_u, axis_v


def _detect_circle_3d(points: np.ndarray, tolerance: float) -> Optional[Dict[str, Any]]:
    """
    Detect if 3D points lie on a circle.

    Returns:
        Circle info dict or None if not a circle
    """
    if len(points) < 3:
        return None

    # First check if points are planar
    centroid, normal, plane_dev = _fit_plane_svd(points)

    # Project to 2D
    points_2d, axis_u, axis_v = _project_to_plane(points, centroid, normal)

    # Fit circle in 2D
    cx_2d, cy_2d, radius, circle_dev = _fit_circle_2d(points_2d)

    # Total deviation includes planarity
    total_dev = max(plane_dev, circle_dev)

    if total_dev > tolerance:
        return None

    # Convert center back to 3D
    center_3d = centroid + cx_2d * axis_u + cy_2d * axis_v

    return {
        "center": center_3d.tolist(),
        "radius": float(radius),
        "normal": normal.tolist(),
        "deviation": float(total_dev)
    }


def _detect_line_3d(points: np.ndarray, tolerance: float) -> Optional[Dict[str, Any]]:
    """
    Detect if 3D points are collinear (lie on a line).

    Returns:
        Line info dict or None if not a line
    """
    if len(points) < 2:
        return None

    # Use SVD to check collinearity
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        U, S, Vt = np.linalg.svd(centered)
    except Exception:
        return None

    # For a line, the two smaller singular values should be ~0
    # S is sorted descending, so S[1] and S[2] should be small
    if len(S) < 2:
        return None

    # Max deviation from line is related to S[1]
    # More precisely, compute distance from line through first/last points
    start = points[0]
    end = points[-1]
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-10:
        return None

    direction = direction / length

    # Distance of each point from the line
    deviations = []
    for pt in points:
        v = pt - start
        proj = np.dot(v, direction) * direction
        perp = v - proj
        deviations.append(np.linalg.norm(perp))

    max_dev = max(deviations)

    if max_dev > tolerance:
        return None

    return {
        "start": start.tolist(),
        "end": end.tolist(),
        "deviation": float(max_dev)
    }


def _create_circle_edge(circle_info: Dict, edge: TopoDS_Edge) -> Optional[TopoDS_Edge]:
    """
    Create a new edge with circle geometry, ensuring correct arc direction.

    Uses midpoint sampling to determine which way the arc should go,
    preventing the "long way around" problem.
    """
    from math import atan2, pi

    try:
        adaptor = BRepAdaptor_Curve(edge)
        u_start = adaptor.FirstParameter()
        u_end = adaptor.LastParameter()

        # Get start, mid, end points from original curve
        start_pnt = adaptor.Value(u_start)
        mid_pnt = adaptor.Value((u_start + u_end) / 2)
        end_pnt = adaptor.Value(u_end)

        # Create circle geometry
        center = gp_Pnt(*circle_info["center"])
        normal = gp_Dir(*circle_info["normal"])
        radius = circle_info["radius"]

        axis = gp_Ax2(center, normal)
        circle = gp_Circ(axis, radius)

        # Get circle's local coordinate system
        x_dir = axis.XDirection()
        y_dir = axis.YDirection()

        def point_to_angle(pnt):
            """Convert 3D point to angle on circle plane."""
            v = gp_Vec(center, pnt)
            x = v.Dot(gp_Vec(x_dir))
            y = v.Dot(gp_Vec(y_dir))
            return atan2(y, x)

        def normalize_angle(a):
            """Normalize angle to [0, 2*pi)."""
            while a < 0:
                a += 2 * pi
            while a >= 2 * pi:
                a -= 2 * pi
            return a

        angle_start = normalize_angle(point_to_angle(start_pnt))
        angle_mid = normalize_angle(point_to_angle(mid_pnt))
        angle_end = normalize_angle(point_to_angle(end_pnt))

        from math import degrees
        logger.info(f"[CircleEdge] Angles: start={degrees(angle_start):.1f}°, mid={degrees(angle_mid):.1f}°, end={degrees(angle_end):.1f}°, R={radius:.1f}")

        # Check for degenerate arc (all points at same angle, or wrapping around)
        angle_tolerance = 0.01  # ~0.5 degrees
        angle_diff = abs(angle_start - angle_end)
        if angle_diff < angle_tolerance or angle_diff > (2 * pi - angle_tolerance):
            logger.warning(f"[CircleEdge] Degenerate arc (start~end, diff={degrees(angle_diff):.2f}°), skipping replacement")
            return None

        # Determine correct arc direction by checking if midpoint is on the path
        # We want the arc that passes through the midpoint

        # Calculate arc span going from start to end (counterclockwise)
        if angle_end >= angle_start:
            ccw_span = angle_end - angle_start
        else:
            ccw_span = (2 * pi - angle_start) + angle_end

        # Check if midpoint is within the counterclockwise arc
        if angle_end >= angle_start:
            mid_in_ccw = angle_start <= angle_mid <= angle_end
        else:
            mid_in_ccw = angle_mid >= angle_start or angle_mid <= angle_end

        logger.info(f"[CircleEdge] CCW span={degrees(ccw_span):.1f}°, mid_in_ccw={mid_in_ccw}")

        if mid_in_ccw:
            # Counterclockwise arc is correct
            param_start = angle_start
            param_end = angle_start + ccw_span
        else:
            # Clockwise arc is correct (go the other way)
            cw_span = 2 * pi - ccw_span
            param_start = angle_start
            param_end = angle_start - cw_span

        arc_span = abs(param_end - param_start)
        logger.info(f"[CircleEdge] Final arc: {degrees(param_start):.1f}° to {degrees(param_end):.1f}° (span={degrees(arc_span):.1f}°)")

        # Create edge with explicit parameter range
        builder = BRepBuilderAPI_MakeEdge(circle, param_start, param_end)

        if builder.IsDone():
            return builder.Edge()

        # Fallback: try with points if parameter method fails
        logger.warning(f"Parameter-based circle edge failed, trying point-based")
        builder = BRepBuilderAPI_MakeEdge(circle, start_pnt, end_pnt)
        if builder.IsDone():
            return builder.Edge()

        return None
    except Exception as e:
        logger.warning(f"Failed to create circle edge: {e}")
        return None


def _create_line_edge(line_info: Dict) -> Optional[TopoDS_Edge]:
    """
    Create a new edge with line geometry.
    """
    try:
        start_pnt = gp_Pnt(*line_info["start"])
        end_pnt = gp_Pnt(*line_info["end"])

        builder = BRepBuilderAPI_MakeEdge(start_pnt, end_pnt)

        if builder.IsDone():
            return builder.Edge()

        return None
    except Exception as e:
        logger.warning(f"Failed to create line edge: {e}")
        return None

class CADPrimitiveReconstruction:
    """
    Detect B-spline edges that are geometrically circles or lines,
    and optionally replace them with true OCC primitives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "circle_tolerance": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-10,
                    "max": 1.0,
                    "step": 1e-5,
                    "tooltip": "Max deviation for circle detection (smaller = stricter)"
                }),
                "line_tolerance": ("FLOAT", {
                    "default": 1e-6,
                    "min": 1e-10,
                    "max": 1.0,
                    "step": 1e-7,
                    "tooltip": "Max deviation for line detection (smaller = stricter)"
                }),
            },
            "optional": {
                "replace_circles": ("BOOLEAN", {"default": True}),
                "replace_lines": ("BOOLEAN", {"default": True}),
                "n_samples": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 100,
                    "tooltip": "Number of points to sample for detection"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING")
    RETURN_NAMES = ("cad_model", "info")
    FUNCTION = "reconstruct_primitives"
    CATEGORY = "CADabra/Processing"

    def reconstruct_primitives(
        self,
        cad_model: Dict,
        circle_tolerance: float = 1e-4,
        line_tolerance: float = 1e-6,
        replace_circles: bool = True,
        replace_lines: bool = True,
        n_samples: int = 20
    ) -> Tuple[Dict, str]:
        """
        Detect and optionally replace B-spline primitives.
        """
        logger.info(f"[PrimitiveReconstruction] Received cad_model keys: {list(cad_model.keys())}")
        # Get OCC shape from brep_path
        try:
            from .utils.brep_cache import get_occ_shape
        except ImportError:
            from .utils.brep_cache import get_occ_shape
        try:
            shape = get_occ_shape(cad_model)
        except Exception as e:
            logger.error(f"[PrimitiveReconstruction] Failed to load shape: {e}")
            return cad_model, json.dumps({"error": str(e)})
        logger.info(f"[PrimitiveReconstruction] Shape type: {type(shape)}, is None: {shape is None}")
        if shape is None or not isinstance(shape, TopoDS_Shape):
            return cad_model, json.dumps({"error": "Invalid CAD model"})

        # Results tracking
        detected_circles = []
        detected_lines = []
        replacements = []

        # Collect all edges
        edges = []
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_id = 0
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            edges.append((edge_id, edge))
            edge_id += 1
            explorer.Next()

        total_edges = len(edges)
        logger.info(f"[PrimitiveReconstruction] Analyzing {total_edges} edges...")

        # Prepare reshaper for replacements
        reshaper = BRepTools_ReShape()

        for edge_id, edge in edges:
            # Get curve type
            adaptor = BRepAdaptor_Curve(edge)
            curve_type = adaptor.GetType()

            # Only process B-splines and Beziers
            if curve_type not in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
                continue

            # Sample points on the curve
            points = _sample_curve_points(edge, n_samples)

            if len(points) < 3:
                continue

            # Try circle detection first
            circle_info = _detect_circle_3d(points, circle_tolerance)
            if circle_info:
                circle_info["edge_id"] = edge_id
                circle_info["replaced"] = False

                if replace_circles:
                    new_edge = _create_circle_edge(circle_info, edge)
                    if new_edge:
                        reshaper.Replace(edge, new_edge)
                        circle_info["replaced"] = True
                        replacements.append(("circle", edge_id))

                detected_circles.append(circle_info)
                continue

            # Try line detection
            line_info = _detect_line_3d(points, line_tolerance)
            if line_info:
                line_info["edge_id"] = edge_id
                line_info["replaced"] = False

                if replace_lines:
                    new_edge = _create_line_edge(line_info)
                    if new_edge:
                        reshaper.Replace(edge, new_edge)
                        line_info["replaced"] = True
                        replacements.append(("line", edge_id))

                detected_lines.append(line_info)

        # Apply all replacements
        if replacements:
            new_shape = reshaper.Apply(shape)
            logger.info(f"[PrimitiveReconstruction] Applied {len(replacements)} replacements")
        else:
            new_shape = shape

        # Build report
        circles_replaced = sum(1 for c in detected_circles if c.get("replaced"))
        lines_replaced = sum(1 for l in detected_lines if l.get("replaced"))

        report = {
            "summary": {
                "total_edges": total_edges,
                "circles_detected": len(detected_circles),
                "circles_replaced": circles_replaced,
                "lines_detected": len(detected_lines),
                "lines_replaced": lines_replaced
            },
            "detected_circles": detected_circles,
            "detected_lines": detected_lines
        }

        # Create human-readable summary
        summary_lines = [
            f"Primitive Reconstruction Report",
            f"================================",
            f"Total edges analyzed: {total_edges}",
            f"",
            f"Circles: {len(detected_circles)} detected, {circles_replaced} replaced",
            f"Lines: {len(detected_lines)} detected, {lines_replaced} replaced",
        ]

        if detected_circles:
            summary_lines.append("")
            summary_lines.append("Detected Circles:")
            for c in detected_circles[:10]:  # Limit output
                status = "[x] replaced" if c.get("replaced") else "detected only"
                summary_lines.append(
                    f"  Edge #{c['edge_id']}: R={c['radius']:.3f}, "
                    f"center=({c['center'][0]:.2f}, {c['center'][1]:.2f}, {c['center'][2]:.2f}), "
                    f"dev={c['deviation']:.2e} [{status}]"
                )
            if len(detected_circles) > 10:
                summary_lines.append(f"  ... and {len(detected_circles) - 10} more")

        if detected_lines:
            summary_lines.append("")
            summary_lines.append("Detected Lines:")
            for l in detected_lines[:10]:
                status = "[x] replaced" if l.get("replaced") else "detected only"
                summary_lines.append(
                    f"  Edge #{l['edge_id']}: dev={l['deviation']:.2e} [{status}]"
                )
            if len(detected_lines) > 10:
                summary_lines.append(f"  ... and {len(detected_lines) - 10} more")

        report_text = "\n".join(summary_lines)

        # Create new CAD model with new shape (saves to brep_path)
        try:
            from .utils.brep_cache import make_cad_model
        except ImportError:
            from .utils.brep_cache import make_cad_model
        new_cad_model = make_cad_model(new_shape, cad_model, "primitive_recon")

        return new_cad_model, report_text


# Node registration
NODE_CLASS_MAPPINGS = {
    "CADPrimitiveReconstruction": CADPrimitiveReconstruction
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADPrimitiveReconstruction": "CAD Primitive Reconstruction"
}
