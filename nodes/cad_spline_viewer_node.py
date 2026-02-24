"""
CAD Spline Viewer node for ComfyUI-CADabra
Interactive visualization of B-spline, Bezier, and NURBS surface parameters.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import folder_paths

from .utils.occ_logging import logger

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
)
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface


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

# Detailed surface type info for education
SURFACE_TYPE_INFO = {
    "BSpline": {
        "full_name": "B-Spline Surface",
        "description": "Non-Uniform Rational B-Spline (NURBS) when weights != 1, otherwise B-spline. Control points have LOCAL influence defined by basis functions.",
        "has_control_points": True,
        "has_knots": True,
        "has_weights": True,  # When rational
    },
    "Bezier": {
        "full_name": "Bezier Surface",
        "description": "Polynomial surface where ALL control points influence the ENTIRE surface. Degree = number of control points - 1.",
        "has_control_points": True,
        "has_knots": False,  # Implicit
        "has_weights": True,  # When rational
    },
    "Plane": {
        "full_name": "Planar Surface",
        "description": "Flat infinite plane defined by a point and normal direction.",
        "has_control_points": False,
        "has_knots": False,
        "has_weights": False,
    },
    "Cylinder": {
        "full_name": "Cylindrical Surface",
        "description": "Surface at constant distance from an axis. Defined by axis and radius.",
        "has_control_points": False,
        "has_knots": False,
        "has_weights": False,
    },
    "Sphere": {
        "full_name": "Spherical Surface",
        "description": "Surface at constant distance from a center point. Defined by center and radius.",
        "has_control_points": False,
        "has_knots": False,
        "has_weights": False,
    },
    "Cone": {
        "full_name": "Conical Surface",
        "description": "Surface formed by lines through a point (apex). Defined by axis, apex, and half-angle.",
        "has_control_points": False,
        "has_knots": False,
        "has_weights": False,
    },
    "Torus": {
        "full_name": "Toroidal Surface",
        "description": "Donut-shaped surface. Defined by axis, major radius, and minor radius.",
        "has_control_points": False,
        "has_knots": False,
        "has_weights": False,
    },
}


def bspline_basis(i, p, u, knots):
    """
    Compute B-spline basis function N_{i,p}(u) using Cox-de Boor recursion.

    Args:
        i: Basis function index (0-indexed)
        p: Degree
        u: Parameter value
        knots: Flat knot vector (list or array)

    Returns:
        float: Basis function value at u
    """
    if p == 0:
        # Base case
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        elif u == knots[i + 1] == knots[-1]:  # Handle endpoint
            return 1.0
        return 0.0

    # Recursive case
    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]

    term1 = 0.0
    term2 = 0.0

    if denom1 != 0:
        term1 = (u - knots[i]) / denom1 * bspline_basis(i, p - 1, u, knots)

    if denom2 != 0:
        term2 = (knots[i + p + 1] - u) / denom2 * bspline_basis(i + 1, p - 1, u, knots)

    return term1 + term2


def compute_bspline_basis_grid(num_ctrl, degree, knots, param_values):
    """
    Compute basis function values for all control points at all parameter values.

    Args:
        num_ctrl: Number of control points
        degree: Degree of the B-spline
        knots: Flat knot vector
        param_values: Array of parameter values to evaluate

    Returns:
        numpy array of shape (num_ctrl, num_params)
    """
    num_params = len(param_values)
    basis_values = np.zeros((num_ctrl, num_params))

    for i in range(num_ctrl):
        for j, u in enumerate(param_values):
            basis_values[i, j] = bspline_basis(i, degree, u, knots)

    return basis_values


class CADSplineViewer:
    """
    Interactive spline/NURBS visualization node.

    Extracts and visualizes control points, weights, knot vectors, and basis
    function influence regions from B-spline and Bezier surfaces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL", {
                    "tooltip": "CAD model to analyze"
                }),
                "face_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Index of the face to visualize (0-based)"
                }),
            },
            "optional": {
                "linear_deflection": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh quality - lower = finer mesh"
                }),
                "compute_influence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Precompute basis function values for influence visualization"
                }),
                "influence_resolution": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Resolution for influence grid computation"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("cad_model", "json_filepath", "surface_info")
    OUTPUT_NODE = True
    FUNCTION = "analyze_spline"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Interactive visualization of B-spline, Bezier, and NURBS surface parameters.

    For spline surfaces, shows:
    - Control points with coordinates and weights
    - Knot vectors with multiplicities
    - Control net (grid of control points)
    - Basis function influence regions

    Click on control points to see their influence on the surface.
    """

    def analyze_spline(self, cad_model, face_index=0, linear_deflection=0.1,
                       compute_influence=True, influence_resolution=20):
        """Analyze a face's spline parameters."""

        # Get OCC shape from brep_path
        from .utils.brep_cache import get_occ_shape
        occ_shape = get_occ_shape(cad_model)

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_spline_{timestamp}"
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(output_dir, json_filename)

        # Get all faces
        faces = []
        explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
        while explorer.More():
            faces.append(topods.Face(explorer.Current()))
            explorer.Next()

        if not faces:
            raise RuntimeError("CAD model has no faces")

        if face_index >= len(faces):
            raise RuntimeError(f"Face index {face_index} out of range (model has {len(faces)} faces)")

        face = faces[face_index]

        # Get surface adaptor
        adaptor = BRepAdaptor_Surface(face)
        surface_type = adaptor.GetType()
        surface_type_name = _get_surface_type_names().get(surface_type, "Unknown")

        # Mesh the face
        BRepMesh_IncrementalMesh(face, linear_deflection)

        # Extract mesh data
        mesh_data = self._extract_mesh(face)

        # Build output data
        output_data = {
            "face_index": face_index,
            "total_faces": len(faces),
            "surface_type": surface_type_name,
            "surface_info": SURFACE_TYPE_INFO.get(surface_type_name, {}),
            "mesh": mesh_data,
            "timestamp": timestamp,
        }

        # Extract spline parameters if applicable
        if surface_type == GeomAbs_BSplineSurface:
            spline_data = self._extract_bspline_data(adaptor, compute_influence, influence_resolution)
            output_data.update(spline_data)

        elif surface_type == GeomAbs_BezierSurface:
            bezier_data = self._extract_bezier_data(adaptor, compute_influence, influence_resolution)
            output_data.update(bezier_data)

        else:
            # For analytic surfaces, extract basic parameters
            analytic_data = self._extract_analytic_data(adaptor)
            output_data.update(analytic_data)

        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Build info string
        info_lines = [
            f"Face {face_index}/{len(faces) - 1}",
            f"Type: {surface_type_name}",
        ]

        if "degree_u" in output_data:
            info_lines.append(f"Degree: {output_data['degree_u']} x {output_data['degree_v']}")
        if "num_ctrl_pts_u" in output_data:
            info_lines.append(f"Control Points: {output_data['num_ctrl_pts_u']} x {output_data['num_ctrl_pts_v']}")
        if "is_rational" in output_data:
            info_lines.append(f"Rational: {'Yes' if output_data['is_rational'] else 'No'}")

        surface_info = "\n".join(info_lines)

        logger.info(f"[CADSplineViewer] Analyzed face {face_index}: {surface_type_name}")

        return {
            "ui": {
                "spline_file": [json_filename],
                "surface_type": [surface_type_name],
                "face_index": [face_index],
                "total_faces": [len(faces)],
            },
            "result": (cad_model, json_filename, surface_info)
        }

    def _extract_mesh(self, face):
        """Extract triangulated mesh from a face."""
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is None:
            return {"vertices": [], "indices": [], "uv_coords": []}

        trsf = location.Transformation()

        # Extract vertices and UV coordinates
        vertices = []
        uv_coords = []
        num_nodes = triangulation.NbNodes()

        for i in range(1, num_nodes + 1):
            pnt = triangulation.Node(i).Transformed(trsf)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

            # Get UV parameters if available
            if triangulation.HasUVNodes():
                uv = triangulation.UVNode(i)
                uv_coords.append([uv.X(), uv.Y()])

        # Extract triangles
        indices = []
        num_triangles = triangulation.NbTriangles()

        for i in range(1, num_triangles + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            # Convert to 0-indexed
            indices.append([n1 - 1, n2 - 1, n3 - 1])

        return {
            "vertices": vertices,
            "indices": indices,
            "uv_coords": uv_coords,
            "num_vertices": len(vertices),
            "num_triangles": len(indices),
        }

    def _extract_bspline_data(self, adaptor, compute_influence, influence_resolution):
        """Extract B-spline surface parameters."""
        bspline = adaptor.BSpline()

        # Basic info
        degree_u = bspline.UDegree()
        degree_v = bspline.VDegree()
        num_ctrl_u = bspline.NbUPoles()
        num_ctrl_v = bspline.NbVPoles()
        is_u_rational = bspline.IsURational()
        is_v_rational = bspline.IsVRational()
        is_rational = is_u_rational or is_v_rational

        # Extract control points
        control_points = []
        for i in range(1, num_ctrl_u + 1):
            for j in range(1, num_ctrl_v + 1):
                pole = bspline.Pole(i, j)
                weight = bspline.Weight(i, j) if is_rational else 1.0
                control_points.append({
                    "i": i - 1,  # 0-indexed for JS
                    "j": j - 1,
                    "x": pole.X(),
                    "y": pole.Y(),
                    "z": pole.Z(),
                    "weight": weight,
                })

        # Extract knot vectors (flat)
        knots_u = []
        knots_v = []

        # Get flat knot sequence (with multiplicities expanded)
        for i in range(1, bspline.NbUKnots() + 1):
            knot = bspline.UKnot(i)
            mult = bspline.UMultiplicity(i)
            knots_u.extend([knot] * mult)

        for i in range(1, bspline.NbVKnots() + 1):
            knot = bspline.VKnot(i)
            mult = bspline.VMultiplicity(i)
            knots_v.extend([knot] * mult)

        # Extract unique knots with multiplicities
        unique_knots_u = []
        mults_u = []
        for i in range(1, bspline.NbUKnots() + 1):
            unique_knots_u.append(bspline.UKnot(i))
            mults_u.append(bspline.UMultiplicity(i))

        unique_knots_v = []
        mults_v = []
        for i in range(1, bspline.NbVKnots() + 1):
            unique_knots_v.append(bspline.VKnot(i))
            mults_v.append(bspline.VMultiplicity(i))

        # Parameter bounds
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

        data = {
            "degree_u": degree_u,
            "degree_v": degree_v,
            "num_ctrl_pts_u": num_ctrl_u,
            "num_ctrl_pts_v": num_ctrl_v,
            "is_rational": is_rational,
            "is_u_rational": is_u_rational,
            "is_v_rational": is_v_rational,
            "control_points": control_points,
            "knots_u_flat": knots_u,
            "knots_v_flat": knots_v,
            "knots_u": unique_knots_u,
            "knots_v": unique_knots_v,
            "multiplicities_u": mults_u,
            "multiplicities_v": mults_v,
            "u_range": [u_min, u_max],
            "v_range": [v_min, v_max],
        }

        # Compute influence grid if requested
        if compute_influence:
            influence_data = self._compute_bspline_influence(
                num_ctrl_u, num_ctrl_v, degree_u, degree_v,
                knots_u, knots_v, u_min, u_max, v_min, v_max,
                influence_resolution
            )
            data["influence_grid"] = influence_data

        return data

    def _extract_bezier_data(self, adaptor, compute_influence, influence_resolution):
        """Extract Bezier surface parameters."""
        bezier = adaptor.Bezier()

        # Basic info
        degree_u = bezier.UDegree()
        degree_v = bezier.VDegree()
        num_ctrl_u = degree_u + 1
        num_ctrl_v = degree_v + 1
        is_u_rational = bezier.IsURational()
        is_v_rational = bezier.IsVRational()
        is_rational = is_u_rational or is_v_rational

        # Extract control points
        control_points = []
        for i in range(1, num_ctrl_u + 1):
            for j in range(1, num_ctrl_v + 1):
                pole = bezier.Pole(i, j)
                weight = bezier.Weight(i, j) if is_rational else 1.0
                control_points.append({
                    "i": i - 1,
                    "j": j - 1,
                    "x": pole.X(),
                    "y": pole.Y(),
                    "z": pole.Z(),
                    "weight": weight,
                })

        # Bezier has implicit knot vector [0, 0, ..., 0, 1, 1, ..., 1]
        knots_u = [0.0] * (degree_u + 1) + [1.0] * (degree_u + 1)
        knots_v = [0.0] * (degree_v + 1) + [1.0] * (degree_v + 1)

        # Parameter bounds
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

        data = {
            "degree_u": degree_u,
            "degree_v": degree_v,
            "num_ctrl_pts_u": num_ctrl_u,
            "num_ctrl_pts_v": num_ctrl_v,
            "is_rational": is_rational,
            "is_u_rational": is_u_rational,
            "is_v_rational": is_v_rational,
            "control_points": control_points,
            "knots_u_flat": knots_u,
            "knots_v_flat": knots_v,
            "knots_u": [0.0, 1.0],
            "knots_v": [0.0, 1.0],
            "multiplicities_u": [degree_u + 1, degree_u + 1],
            "multiplicities_v": [degree_v + 1, degree_v + 1],
            "u_range": [u_min, u_max],
            "v_range": [v_min, v_max],
        }

        # Compute influence grid if requested
        if compute_influence:
            influence_data = self._compute_bspline_influence(
                num_ctrl_u, num_ctrl_v, degree_u, degree_v,
                knots_u, knots_v, u_min, u_max, v_min, v_max,
                influence_resolution
            )
            data["influence_grid"] = influence_data

        return data

    def _extract_analytic_data(self, adaptor):
        """Extract parameters for analytic surfaces (plane, cylinder, etc.)."""
        surface_type = adaptor.GetType()
        data = {"analytic_params": {}}

        if surface_type == GeomAbs_Plane:
            plane = adaptor.Plane()
            loc = plane.Location()
            axis = plane.Axis().Direction()
            data["analytic_params"] = {
                "location": [loc.X(), loc.Y(), loc.Z()],
                "normal": [axis.X(), axis.Y(), axis.Z()],
            }

        elif surface_type == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            loc = cyl.Location()
            axis = cyl.Axis().Direction()
            data["analytic_params"] = {
                "location": [loc.X(), loc.Y(), loc.Z()],
                "axis": [axis.X(), axis.Y(), axis.Z()],
                "radius": cyl.Radius(),
            }

        elif surface_type == GeomAbs_Cone:
            cone = adaptor.Cone()
            apex = cone.Apex()
            axis = cone.Axis().Direction()
            data["analytic_params"] = {
                "apex": [apex.X(), apex.Y(), apex.Z()],
                "axis": [axis.X(), axis.Y(), axis.Z()],
                "semi_angle": cone.SemiAngle(),
                "ref_radius": cone.RefRadius(),
            }

        elif surface_type == GeomAbs_Sphere:
            sphere = adaptor.Sphere()
            center = sphere.Location()
            data["analytic_params"] = {
                "center": [center.X(), center.Y(), center.Z()],
                "radius": sphere.Radius(),
            }

        elif surface_type == GeomAbs_Torus:
            torus = adaptor.Torus()
            loc = torus.Location()
            axis = torus.Axis().Direction()
            data["analytic_params"] = {
                "location": [loc.X(), loc.Y(), loc.Z()],
                "axis": [axis.X(), axis.Y(), axis.Z()],
                "major_radius": torus.MajorRadius(),
                "minor_radius": torus.MinorRadius(),
            }

        return data

    def _compute_bspline_influence(self, num_ctrl_u, num_ctrl_v, degree_u, degree_v,
                                    knots_u, knots_v, u_min, u_max, v_min, v_max,
                                    resolution):
        """
        Compute basis function influence values on a grid.

        Returns a dict mapping control point indices "i_j" to a 2D array of
        influence values on a resolution x resolution grid.
        """
        # Create parameter grid
        u_values = np.linspace(u_min, u_max, resolution)
        v_values = np.linspace(v_min, v_max, resolution)

        # Compute basis functions for U and V directions
        basis_u = compute_bspline_basis_grid(num_ctrl_u, degree_u, knots_u, u_values)
        basis_v = compute_bspline_basis_grid(num_ctrl_v, degree_v, knots_v, v_values)

        # Store influence data
        influence = {
            "resolution": resolution,
            "u_values": u_values.tolist(),
            "v_values": v_values.tolist(),
            "control_point_influence": {},
        }

        # For each control point, compute the tensor product of basis functions
        for i in range(num_ctrl_u):
            for j in range(num_ctrl_v):
                # Tensor product: N_i(u) * N_j(v)
                grid = np.outer(basis_u[i], basis_v[j])
                key = f"{i}_{j}"
                influence["control_point_influence"][key] = grid.tolist()

        return influence


# Node registration
NODE_CLASS_MAPPINGS = {
    "CADSplineViewer": CADSplineViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADSplineViewer": "CAD Spline Viewer",
}
