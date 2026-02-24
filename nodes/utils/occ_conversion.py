import logging
import numpy as np
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax2, gp_Ax3

log = logging.getLogger("cadabra")
from OCC.Core.Geom import (
    Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface,
    Geom_ConicalSurface, Geom_BSplineSurface
)
from OCC.Core.GeomAPI import (
    GeomAPI_PointsToBSplineSurface, GeomAPI_ProjectPointOnSurf
)
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face
from OCC.Core.BRep import BRep_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

def check_occ():
    """Check if OCC is available."""
    if not HAS_OCC:
        raise ImportError(
            "OpenCASCADE not available. Install pythonocc-core or use cadquery."
        )


def plane_to_occ(axis, distance, center=None):
    """
    Convert plane parameters to OCC Geom_Plane.

    Args:
        axis: Normal vector [nx, ny, nz]
        distance: Distance from origin along normal
        center: Optional center point [x, y, z]. If None, computed from axis*distance

    Returns:
        Geom_Plane object
    """
    check_occ()

    axis = np.array(axis, dtype=np.float64).flatten()  # Ensure 1D float array
    axis = axis / np.linalg.norm(axis)  # Normalize

    if center is None:
        center = axis * distance
    else:
        center = np.array(center, dtype=np.float64).flatten()  # Also flatten center

    normal = gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
    point = gp_Pnt(float(center[0]), float(center[1]), float(center[2]))

    return Geom_Plane(point, normal)


def sphere_to_occ(center, radius):
    """
    Convert sphere parameters to OCC Geom_SphericalSurface.

    Args:
        center: Center point [x, y, z]
        radius: Sphere radius

    Returns:
        Geom_SphericalSurface object
    """
    check_occ()

    center = np.array(center)
    ax = gp_Ax3(
        gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
        gp_Dir(0, 0, 1)  # Z-up orientation
    )

    return Geom_SphericalSurface(ax, float(radius))


def cylinder_to_occ(axis, center, radius):
    """
    Convert cylinder parameters to OCC Geom_CylindricalSurface.

    Args:
        axis: Cylinder axis direction [ax, ay, az]
        center: Point on cylinder axis [x, y, z]
        radius: Cylinder radius

    Returns:
        Geom_CylindricalSurface object
    """
    check_occ()

    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    center = np.array(center)

    ax = gp_Ax3(
        gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
        gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
    )

    return Geom_CylindricalSurface(ax, float(radius))


def cone_to_occ(apex, axis, half_angle):
    """
    Convert cone parameters to OCC Geom_ConicalSurface.

    Args:
        apex: Cone apex point [x, y, z]
        axis: Cone axis direction [ax, ay, az]
        half_angle: Cone half-angle in radians

    Returns:
        Geom_ConicalSurface object
    """
    check_occ()

    apex = np.array(apex)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    ax = gp_Ax3(
        gp_Pnt(float(apex[0]), float(apex[1]), float(apex[2])),
        gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
    )

    # OCC uses reference radius at distance 1 from apex
    # For half_angle theta, ref_radius = tan(theta)
    ref_radius = np.tan(half_angle)

    return Geom_ConicalSurface(ax, float(half_angle), float(ref_radius))


# =============================================================================
# Bounds computation for primitive surfaces
# =============================================================================

def compute_surface_bounds_from_points(surface, points, margin=0.2):
    """
    Compute UV bounds for a surface based on point cloud extent.

    For planes and other primitive surfaces, uses a simple approach based on
    the spatial extent of the point cloud to avoid huge UV ranges.

    Args:
        surface: OCC Geom_Surface object (Geom_Plane, etc.)
        points: Nx3 numpy array of points
        margin: Fractional margin to add around bounds (default 0.2 = 20%)

    Returns:
        (umin, umax, vmin, vmax) tuple, or None if computation fails
    """
    check_occ()

    points = np.array(points, dtype=np.float64)
    if len(points) == 0:
        return None

    # Compute bounding box extent of the point cloud
    # Use this as a reasonable UV range (centered at origin)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))

    if max_extent < 1e-6:
        return None

    # Use half-extent with margin as UV bounds (centered at 0)
    half_extent = max_extent * (1.0 + margin) / 2.0

    return (-half_extent, half_extent, -half_extent, half_extent)


# =============================================================================
# Mesh boundary extraction for trimmed faces
# =============================================================================

def mesh_boundary_to_wire(mesh):
    """
    Extract boundary edges from trimesh and create OCC wire using polyline edges.

    Args:
        mesh: trimesh.Trimesh object (e.g., grid mesh from surface fitting)

    Returns:
        TopoDS_Wire or None if no boundary found
    """
    check_occ()

    try:
        from trimesh.grouping import group_rows
    except ImportError:
        log.warning(" trimesh.grouping not available")
        return None

    # Find boundary edges (edges that appear only once - not shared between faces)
    edges_sorted = mesh.edges_sorted
    boundary_indices = group_rows(edges_sorted, require_count=1)
    if len(boundary_indices) == 0:
        log.warning(" No boundary edges found")
        return None

    boundary_edges = edges_sorted[boundary_indices]

    # Connect edges into ordered loop
    loop_vertices = _connect_edges_to_loop(boundary_edges)
    if loop_vertices is None or len(loop_vertices) < 3:
        log.warning(" Could not form closed loop from boundary edges")
        return None

    # Get 3D coordinates
    loop_points = mesh.vertices[loop_vertices]

    # Create wire from polyline edges (line segments between consecutive points)
    wire_maker = BRepBuilderAPI_MakeWire()

    for i in range(len(loop_points)):
        p1 = loop_points[i]
        p2 = loop_points[(i + 1) % len(loop_points)]  # Wrap to close loop

        pt1 = gp_Pnt(float(p1[0]), float(p1[1]), float(p1[2]))
        pt2 = gp_Pnt(float(p2[0]), float(p2[1]), float(p2[2]))

        edge_maker = BRepBuilderAPI_MakeEdge(pt1, pt2)
        if edge_maker.IsDone():
            wire_maker.Add(edge_maker.Edge())

    if not wire_maker.IsDone():
        log.warning(" Wire creation failed")
        return None

    log.info(f" Created boundary wire with {len(loop_points)} vertices")
    return wire_maker.Wire()


def _connect_edges_to_loop(boundary_edges):
    """
    Connect disconnected edges into a single ordered vertex loop.

    Args:
        boundary_edges: Nx2 array of vertex index pairs

    Returns:
        List of vertex indices forming a connected loop, or None if failed
    """
    if len(boundary_edges) == 0:
        return None

    # Build adjacency: vertex -> list of (edge_idx, other_vertex)
    vertex_to_edges = {}
    for i, (v1, v2) in enumerate(boundary_edges):
        vertex_to_edges.setdefault(v1, []).append((i, v2))
        vertex_to_edges.setdefault(v2, []).append((i, v1))

    # Walk the boundary starting from first edge
    used = set()
    start_vertex = boundary_edges[0][0]
    loop = [start_vertex]
    current = boundary_edges[0][1]
    used.add(0)

    # Walk until we return to start or run out of edges
    while current != start_vertex and len(used) < len(boundary_edges):
        loop.append(current)
        found = False
        for edge_idx, next_v in vertex_to_edges.get(current, []):
            if edge_idx not in used:
                used.add(edge_idx)
                current = next_v
                found = True
                break
        if not found:
            break

    return loop if len(loop) >= 3 else None


# =============================================================================
# B-spline to OCC conversion
# =============================================================================

def geomdl_to_occ_bspline(surf):
    """
    Convert geomdl BSpline.Surface to OCC Geom_BSplineSurface.

    Args:
        surf: geomdl BSpline.Surface object

    Returns:
        Geom_BSplineSurface object
    """
    check_occ()

    # Get control points
    ctrl_pts = np.array(surf.ctrlpts2d)
    n_u, n_v = ctrl_pts.shape[:2]

    # Create OCC control point array (1-indexed)
    poles = TColgp_Array2OfPnt(1, n_u, 1, n_v)
    for i in range(n_u):
        for j in range(n_v):
            pt = ctrl_pts[i, j]
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

    # Get knot vectors
    kv_u = np.array(surf.knotvector_u)
    kv_v = np.array(surf.knotvector_v)

    # Convert to unique knots and multiplicities
    def knots_to_occ(kv, degree):
        """Convert flat knot vector to unique knots and multiplicities."""
        unique_knots = []
        mults = []
        prev = None
        for k in kv:
            if prev is None or k != prev:
                unique_knots.append(k)
                mults.append(1)
            else:
                mults[-1] += 1
            prev = k
        return unique_knots, mults

    u_knots, u_mults = knots_to_occ(kv_u, surf.degree_u)
    v_knots, v_mults = knots_to_occ(kv_v, surf.degree_v)

    # Create OCC arrays
    u_knots_arr = TColStd_Array1OfReal(1, len(u_knots))
    u_mults_arr = TColStd_Array1OfInteger(1, len(u_mults))
    for i, (k, m) in enumerate(zip(u_knots, u_mults)):
        u_knots_arr.SetValue(i + 1, float(k))
        u_mults_arr.SetValue(i + 1, int(m))

    v_knots_arr = TColStd_Array1OfReal(1, len(v_knots))
    v_mults_arr = TColStd_Array1OfInteger(1, len(v_mults))
    for i, (k, m) in enumerate(zip(v_knots, v_mults)):
        v_knots_arr.SetValue(i + 1, float(k))
        v_mults_arr.SetValue(i + 1, int(m))

    # Create B-spline surface
    return Geom_BSplineSurface(
        poles,
        u_knots_arr, v_knots_arr,
        u_mults_arr, v_mults_arr,
        surf.degree_u, surf.degree_v
    )


def nurbsdiff_to_occ_bspline(nurbs_result):
    """
    Convert NURBSDiff result to OCC Geom_BSplineSurface.

    Args:
        nurbs_result: Dict from fit_nurbs_nurbsdiff containing control_points, degree

    Returns:
        Geom_BSplineSurface object
    """
    check_occ()

    ctrl_pts = nurbs_result["control_points"][0]  # (n_u, n_v, 4) with weights
    n_u, n_v = ctrl_pts.shape[:2]
    degree_u = nurbs_result["degree_u"]
    degree_v = nurbs_result["degree_v"]

    # Create OCC control point array
    poles = TColgp_Array2OfPnt(1, n_u, 1, n_v)
    for i in range(n_u):
        for j in range(n_v):
            pt = ctrl_pts[i, j, :3]
            poles.SetValue(i + 1, j + 1,
                          gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

    # Create uniform knot vectors
    def make_uniform_knots(n_ctrl, degree):
        """Create uniform clamped knot vector."""
        n_knots = n_ctrl + degree + 1
        knots = []
        for i in range(n_knots):
            if i < degree + 1:
                knots.append(0.0)
            elif i >= n_knots - degree - 1:
                knots.append(1.0)
            else:
                knots.append((i - degree) / (n_knots - 2 * degree - 1))
        return knots

    kv_u = make_uniform_knots(n_u, degree_u)
    kv_v = make_uniform_knots(n_v, degree_v)

    # Convert to unique knots and multiplicities
    def knots_to_occ(kv):
        unique_knots = []
        mults = []
        prev = None
        for k in kv:
            if prev is None or abs(k - prev) > 1e-10:
                unique_knots.append(k)
                mults.append(1)
            else:
                mults[-1] += 1
            prev = k
        return unique_knots, mults

    u_knots, u_mults = knots_to_occ(kv_u)
    v_knots, v_mults = knots_to_occ(kv_v)

    # Create OCC arrays
    u_knots_arr = TColStd_Array1OfReal(1, len(u_knots))
    u_mults_arr = TColStd_Array1OfInteger(1, len(u_mults))
    for i, (k, m) in enumerate(zip(u_knots, u_mults)):
        u_knots_arr.SetValue(i + 1, float(k))
        u_mults_arr.SetValue(i + 1, int(m))

    v_knots_arr = TColStd_Array1OfReal(1, len(v_knots))
    v_mults_arr = TColStd_Array1OfInteger(1, len(v_mults))
    for i, (k, m) in enumerate(zip(v_knots, v_mults)):
        v_knots_arr.SetValue(i + 1, float(k))
        v_mults_arr.SetValue(i + 1, int(m))

    return Geom_BSplineSurface(
        poles,
        u_knots_arr, v_knots_arr,
        u_mults_arr, v_mults_arr,
        degree_u, degree_v
    )


# =============================================================================
# Surface params to OCC
# =============================================================================

def surface_params_to_occ(surface_result):
    """
    Convert surface fitting result to OCC geometry.

    Args:
        surface_result: Dict with 'type' and 'params' from surface fitting

    Returns:
        OCC Geom_Surface object
    """
    surface_type = surface_result["type"]
    params = surface_result.get("params")

    if surface_type == "plane":
        axis, distance = params
        # Get center from mesh if available
        if "mesh" in surface_result:
            center = surface_result["mesh"].centroid
        else:
            center = np.array(axis) * distance
        return plane_to_occ(axis, distance, center)

    elif surface_type == "sphere":
        center, radius = params
        return sphere_to_occ(center, radius)

    elif surface_type == "cylinder":
        axis, center, radius = params
        return cylinder_to_occ(axis, center, radius)

    elif surface_type == "cone":
        apex, axis, half_angle = params
        return cone_to_occ(apex, axis, half_angle)

    elif surface_type == "open_spline":
        # B-spline needs to be fit separately
        raise ValueError("open_spline requires B-spline fitting first")

    else:
        raise ValueError(f"Unknown surface type: {surface_type}")


# =============================================================================
# Compound building and export
# =============================================================================

def build_compound(surfaces, bounds_list=None, boundary_wires=None):
    """
    Build OCC TopoDS_Compound from list of surfaces.

    Args:
        surfaces: List of OCC Geom_Surface objects
        bounds_list: Optional list of (umin, umax, vmin, vmax) tuples per surface.
                    If None or entry is None, uses default bounds based on surface type.
        boundary_wires: Optional list of TopoDS_Wire objects for trimmed faces.
                       If provided, wire is used to create a trimmed face instead of UV bounds.

    Returns:
        TopoDS_Compound
    """
    check_occ()

    if bounds_list is None:
        bounds_list = [None] * len(surfaces)

    if boundary_wires is None:
        boundary_wires = [None] * len(surfaces)

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for i, surf in enumerate(surfaces):
        bounds = bounds_list[i] if i < len(bounds_list) else None
        wire = boundary_wires[i] if i < len(boundary_wires) else None

        try:
            face_maker = None
            face = None

            # Priority 1: Use boundary wire for trimmed face on surface
            if wire is not None:
                try:
                    # Create face from surface bounded by wire
                    face_maker = BRepBuilderAPI_MakeFace(surf, wire, True)
                    if face_maker.IsDone():
                        face = face_maker.Face()
                        log.info(f"build_compound: Surface {i}: created trimmed face from surface + wire")
                except Exception as e:
                    log.info(f"build_compound: Surface {i}: wire-based face failed ({e}), falling back")
                    face = None

            # Priority 2: Use UV bounds
            if face is None and bounds is not None:
                umin, umax, vmin, vmax = bounds
                face_maker = BRepBuilderAPI_MakeFace(surf, umin, umax, vmin, vmax, 1e-6)
                if face_maker.IsDone():
                    face = face_maker.Face()

            # Priority 3: Default based on surface type
            if face is None:
                try:
                    if surf.IsKind(Geom_Plane.get_type_descriptor()):
                        # Use reasonable default bounds for planes (10 units)
                        face_maker = BRepBuilderAPI_MakeFace(surf, -5, 5, -5, 5, 1e-6)
                    else:
                        # B-splines and other bounded surfaces work with just tolerance
                        face_maker = BRepBuilderAPI_MakeFace(surf, 1e-6)

                    if face_maker.IsDone():
                        face = face_maker.Face()
                except Exception:
                    # Fallback if type check fails
                    face_maker = BRepBuilderAPI_MakeFace(surf, 1e-6)
                    if face_maker.IsDone():
                        face = face_maker.Face()

            if face is not None:
                builder.Add(compound, face)
            else:
                log.warning(f"Face maker failed for surface {i}")

        except Exception as e:
            log.warning(f"Could not create face from surface {i}: {e}")

    return compound


def export_step(compound, filename):
    """
    Export OCC compound to STEP file.

    Args:
        compound: TopoDS_Compound or TopoDS_Shape
        filename: Output STEP file path

    Returns:
        True if successful
    """
    check_occ()

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(filename)

    return status == IFSelect_RetDone
