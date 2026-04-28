"""
CAD Curve Plotter node for ComfyUI-CADabra
2D sketchpad visualization of B-spline/Bezier curves with control points.
"""

import os
import json
import numpy as np
import folder_paths
from comfy_api.latest import io
from .utils.occ_logging import logger


class CADCurvePlotter(io.ComfyNode):
    """
    2D Curve Plotter for B-spline and Bezier curves.

    Displays curves on a sketchpad-style background with:
    - Grid lines
    - Curve plot
    - Control polygon
    - Control points with indices
    - Curve parameters

    Only plots planar curves (deviation <= 1%). Refuses 3D curves.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADCurvePlotter",
            display_name="CAD Curve Plotter",
            category="CADabra/Visualization",
            is_output_node=True,
            description="""
    2D Curve Plotter for B-spline/Bezier edges.

    Displays on a sketchpad background:
    - The curve
    - Control polygon (dashed)
    - Control points with indices
    - Curve parameters (degree, knots, planarity)

    Only works with planar curves (deviation <= 1%).
    """,
            inputs=[
                io.String.Input("spline_data_file", default="",
                                tooltip="Path to spline data JSON file (from Edge Viewer VTK)"),
                io.Int.Input("edge_id", default=0, min=0, max=10000,
                             tooltip="Edge ID to plot (visible in Edge Viewer)"),
                io.Int.Input("num_samples", default=100, min=20, max=500,
                             tooltip="Number of points to sample along curve"),
            ],
            outputs=[
                io.String.Output(display_name="plot_data"),
            ],
        )

    @classmethod
    def execute(cls, spline_data_file, edge_id, num_samples=100):
        """Load spline data and prepare 2D plot."""

        # Load spline data JSON
        if not spline_data_file or not os.path.exists(spline_data_file):
            return _error_result(f"Spline data file not found: {spline_data_file}")

        try:
            with open(spline_data_file, 'r') as f:
                spline_data = json.load(f)
        except Exception as e:
            return _error_result(f"Failed to load spline data: {e}")

        # Find the specified edge
        edge_key = str(edge_id)
        if edge_key not in spline_data:
            available = list(spline_data.keys())[:10]
            return _error_result(
                f"Edge {edge_id} not found in spline data. "
                f"Available: {available}{'...' if len(spline_data) > 10 else ''}"
            )

        edge_data = spline_data[edge_key]

        # Check if it's a spline type
        curve_type = edge_data.get("type", "unknown")
        if curve_type not in ("bspline", "bezier"):
            return _error_result(
                f"Edge {edge_id} is type '{curve_type}', not a B-spline or Bezier curve"
            )

        # Check planarity
        planarity = edge_data.get("planarity", {})
        deviation_pct = planarity.get("deviation_percent", 100)

        if deviation_pct is None or deviation_pct > 1.0:
            return _error_result(
                f"Cannot plot: curve is 3D (deviation: {deviation_pct:.2f}%)\n"
                f"Only planar curves (<=1% deviation) can be plotted."
            )

        # Get control points
        control_points_3d = edge_data.get("control_points", [])
        if not control_points_3d:
            return _error_result("No control points found")

        # Project to 2D
        plane = planarity.get("plane", "XY")
        normal = planarity.get("normal", [0, 0, 1])
        control_points_2d = _project_to_2d(control_points_3d, plane, normal)

        # Get curve parameters
        degree = edge_data.get("degree", 3)
        knots = edge_data.get("knots_flat", [])
        is_rational = edge_data.get("is_rational", False)
        weights = [pt.get("w", 1.0) for pt in control_points_3d] if is_rational else None

        # Sample curve points
        curve_points_2d = _sample_curve(
            control_points_2d, degree, knots, weights, num_samples
        )

        # Calculate bounds for scaling
        all_points = control_points_2d + curve_points_2d
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)

        # Add padding
        pad_x = (max_x - min_x) * 0.1 or 1.0
        pad_y = (max_y - min_y) * 0.1 or 1.0

        # Build result
        result = {
            "success": True,
            "edge_id": edge_id,
            "curve_type": curve_type,
            "degree": degree,
            "is_rational": is_rational,
            "num_poles": len(control_points_2d),
            "planarity": {
                "plane": plane,
                "deviation_percent": deviation_pct
            },
            "knots": knots,
            "weights": [pt.get("w", 1.0) for pt in control_points_3d] if is_rational else None,
            "control_points": control_points_2d,
            "curve_points": curve_points_2d,
            "bounds": {
                "min_x": min_x - pad_x,
                "max_x": max_x + pad_x,
                "min_y": min_y - pad_y,
                "max_y": max_y + pad_y
            }
        }

        result_json = json.dumps(result)

        return io.NodeOutput(result_json, ui={"plot_data": [result_json]})


def _error_result(message):
    """Return error result."""
    result = {
        "success": False,
        "error": message
    }
    result_json = json.dumps(result)
    return io.NodeOutput(result_json, ui={"plot_data": [result_json]})


def _project_to_2d(control_points_3d, plane, normal):
    """Project 3D control points to 2D based on plane orientation."""
    points_2d = []

    for pt in control_points_3d:
        x, y, z = pt["x"], pt["y"], pt["z"]

        if plane == "XY":
            points_2d.append([x, y])
        elif plane == "XZ":
            points_2d.append([x, z])
        elif plane == "YZ":
            points_2d.append([y, z])
        else:
            # Arbitrary plane - rotate to align normal with Z
            points_2d.append(_rotate_to_xy(x, y, z, normal))

    return points_2d


def _rotate_to_xy(x, y, z, normal):
    """Rotate point to align plane normal with Z axis."""
    # Rodrigues rotation to align normal with [0,0,1]
    n = np.array(normal)
    n = n / np.linalg.norm(n)

    target = np.array([0, 0, 1])

    # If already aligned
    if np.allclose(n, target) or np.allclose(n, -target):
        return [x, y]

    # Rotation axis (cross product)
    axis = np.cross(n, target)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    cos_angle = np.dot(n, target)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    # Rodrigues rotation formula
    p = np.array([x, y, z])
    k = axis
    p_rot = (p * np.cos(angle) +
             np.cross(k, p) * np.sin(angle) +
             k * np.dot(k, p) * (1 - np.cos(angle)))

    return [float(p_rot[0]), float(p_rot[1])]


def _sample_curve(control_points, degree, knots, weights, num_samples):
    """Sample B-spline curve using De Boor algorithm."""
    if not knots or len(knots) < 2:
        # Bezier - use Bernstein polynomials
        return _sample_bezier(control_points, degree, num_samples)

    # B-spline - use De Boor
    n = len(control_points)
    p = degree

    # Parameter range
    t_min = knots[p]
    t_max = knots[n]

    curve_points = []
    for i in range(num_samples):
        t = t_min + (t_max - t_min) * i / (num_samples - 1)
        # Clamp to avoid numerical issues at boundaries
        t = max(t_min, min(t, t_max - 1e-10))
        pt = _de_boor(t, control_points, knots, p, weights)
        curve_points.append(pt)

    return curve_points


def _de_boor(t, control_points, knots, p, weights=None):
    """De Boor algorithm for B-spline evaluation."""
    n = len(control_points)

    # Find knot span
    k = p
    while k < n and knots[k + 1] <= t:
        k += 1
    k = min(k, n - 1)

    # Copy relevant control points
    if weights:
        # NURBS - work in homogeneous coordinates
        d = [[control_points[j][0] * weights[j],
              control_points[j][1] * weights[j],
              weights[j]]
             for j in range(k - p, k + 1)]
    else:
        d = [list(control_points[j]) for j in range(k - p, k + 1)]

    # De Boor recursion
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            idx = k - p + j
            alpha_denom = knots[idx + p + 1 - r] - knots[idx]
            if abs(alpha_denom) < 1e-10:
                alpha = 0
            else:
                alpha = (t - knots[idx]) / alpha_denom

            for dim in range(len(d[j])):
                d[j][dim] = (1 - alpha) * d[j - 1][dim] + alpha * d[j][dim]

    if weights:
        # Convert from homogeneous
        w = d[p][2]
        if abs(w) < 1e-10:
            w = 1e-10
        return [d[p][0] / w, d[p][1] / w]
    else:
        return d[p]


def _sample_bezier(control_points, degree, num_samples):
    """Sample Bezier curve using De Casteljau."""
    curve_points = []
    n = len(control_points)

    for i in range(num_samples):
        t = i / (num_samples - 1)

        # De Casteljau
        pts = [list(p) for p in control_points]
        for r in range(1, n):
            for j in range(n - r):
                pts[j][0] = (1 - t) * pts[j][0] + t * pts[j + 1][0]
                pts[j][1] = (1 - t) * pts[j][1] + t * pts[j + 1][1]

        curve_points.append(pts[0])

    return curve_points


# Node registration
NODE_CLASS_MAPPINGS = {
    "CADCurvePlotter": CADCurvePlotter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADCurvePlotter": "CAD Curve Plotter"
}
