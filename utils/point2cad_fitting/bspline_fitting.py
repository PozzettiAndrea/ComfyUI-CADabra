"""
B-spline/NURBS surface fitting with multiple backends.

Backends:
- geomdl: Lightweight, requires UV parameterization (from INR encoder)
- nurbsdiff: Direct point cloud fitting via Chamfer distance (requires pytorch3d)
"""
import numpy as np
import torch
import trimesh

# Check available backends
HAS_NURBSDIFF = False
HAS_GEOMDL = False

try:
    from geomdl.fitting import approximate_surface
    from geomdl import BSpline
    HAS_GEOMDL = True
except ImportError:
    pass

try:
    from nurbsdiff.surface_eval import SurfEval
    from pytorch3d.loss import chamfer_distance
    HAS_NURBSDIFF = True
except ImportError:
    pass


def fit_bspline_geomdl(
    points: np.ndarray,
    uv_coords: np.ndarray,
    ctrl_pts_u: int = 8,
    ctrl_pts_v: int = 8,
    degree: int = 3,
    grid_size: int = 50,
):
    """
    Fit B-spline using geomdl (requires UV parameterization).

    Args:
        points: (N, 3) point cloud in world coordinates
        uv_coords: (N, 2) UV parameterization from INR encoder
        ctrl_pts_u: Number of control points in U direction
        ctrl_pts_v: Number of control points in V direction
        degree: B-spline degree (2-5)
        grid_size: Resolution for UV grid interpolation

    Returns:
        geomdl BSpline.Surface object
    """
    if not HAS_GEOMDL:
        raise ImportError("geomdl not installed. Run: pip install geomdl")

    from scipy.interpolate import griddata

    # Ensure numpy arrays
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(uv_coords):
        uv_coords = uv_coords.detach().cpu().numpy()

    # Create regular UV grid
    u_min, u_max = uv_coords[:, 0].min(), uv_coords[:, 0].max()
    v_min, v_max = uv_coords[:, 1].min(), uv_coords[:, 1].max()

    u_vals = np.linspace(u_min, u_max, grid_size)
    v_vals = np.linspace(v_min, v_max, grid_size)
    grid_u, grid_v = np.meshgrid(u_vals, v_vals, indexing='ij')

    # Interpolate XYZ at grid UV locations
    grid_points = griddata(uv_coords, points, (grid_u, grid_v), method='linear')

    # Handle NaN from extrapolation - fill with nearest neighbor
    if np.isnan(grid_points).any():
        grid_points_nearest = griddata(uv_coords, points, (grid_u, grid_v), method='nearest')
        nan_mask = np.isnan(grid_points)
        grid_points[nan_mask] = grid_points_nearest[nan_mask]

    # Flatten for geomdl (row-major order expected)
    flat_points = grid_points.reshape(-1, 3).tolist()

    # Fit B-spline using least squares approximation
    surf = approximate_surface(
        flat_points,
        size_u=grid_size,
        size_v=grid_size,
        degree_u=degree,
        degree_v=degree,
        ctrlpts_size_u=ctrl_pts_u,
        ctrlpts_size_v=ctrl_pts_v,
    )

    return surf


def fit_nurbs_nurbsdiff(
    points: torch.Tensor,
    ctrl_pts_u: int = 8,
    ctrl_pts_v: int = 8,
    degree: int = 3,
    num_iters: int = 500,
    lr: float = 0.01,
    device: str = "cuda",
):
    """
    Fit NURBS directly to unordered point cloud using NURBSDiff.

    Uses Chamfer distance optimization - no UV parameterization needed.

    Args:
        points: (N, 3) unordered point cloud
        ctrl_pts_u: Number of control points in U direction
        ctrl_pts_v: Number of control points in V direction
        degree: NURBS degree
        num_iters: Optimization iterations
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        Dict with control_points, degree, knots, and surface_points
    """
    if not HAS_NURBSDIFF:
        raise ImportError("nurbsdiff and pytorch3d required. Run: pip install nurbsdiff pytorch3d")

    if not torch.is_tensor(points):
        points = torch.from_numpy(points.astype(np.float32))
    points = points.to(device)

    # Initialize control points from point cloud bounds
    bounds_min = points.min(0).values
    bounds_max = points.max(0).values

    # Create initial control point grid spanning the bounding box
    u_vals = torch.linspace(0, 1, ctrl_pts_u, device=device)
    v_vals = torch.linspace(0, 1, ctrl_pts_v, device=device)
    grid_u, grid_v = torch.meshgrid(u_vals, v_vals, indexing='ij')

    # Initialize as flat grid at mean Z
    ctrl_pts = torch.zeros(1, ctrl_pts_u, ctrl_pts_v, 4, device=device)
    ctrl_pts[..., 0] = bounds_min[0] + grid_u * (bounds_max[0] - bounds_min[0])
    ctrl_pts[..., 1] = bounds_min[1] + grid_v * (bounds_max[1] - bounds_min[1])
    ctrl_pts[..., 2] = points[:, 2].mean()
    ctrl_pts[..., 3] = 1.0  # Rational weights

    ctrl_pts = torch.nn.Parameter(ctrl_pts)

    # Create NURBS evaluator
    out_dim = 50  # Evaluation resolution
    surf_eval = SurfEval(
        ctrl_pts_u, ctrl_pts_v,
        degree=degree,
        dimension=3,
        out_dim_u=out_dim,
        out_dim_v=out_dim
    ).to(device)

    optimizer = torch.optim.Adam([ctrl_pts], lr=lr)

    for i in range(num_iters):
        optimizer.zero_grad()

        # Evaluate surface
        surf_pts = surf_eval(ctrl_pts)  # (1, out_dim, out_dim, 3)
        surf_pts_flat = surf_pts.reshape(1, -1, 3)

        # Chamfer distance to target point cloud
        loss, _ = chamfer_distance(surf_pts_flat, points.unsqueeze(0))

        loss.backward()
        optimizer.step()

    return {
        "control_points": ctrl_pts.detach().cpu(),
        "degree_u": degree,
        "degree_v": degree,
        "surface_points": surf_pts.detach().cpu(),
    }


def fit_bspline_surface(
    points,
    uv_coords=None,
    ctrl_pts_u: int = 8,
    ctrl_pts_v: int = 8,
    degree: int = 3,
    backend: str = "auto",
    device: str = "cuda",
    **kwargs,
):
    """
    Unified interface for B-spline surface fitting.

    Args:
        points: (N, 3) point cloud
        uv_coords: (N, 2) UV parameterization (required for geomdl backend)
        ctrl_pts_u: Control points in U direction
        ctrl_pts_v: Control points in V direction
        degree: B-spline degree
        backend: "auto", "geomdl", or "nurbsdiff"
        device: Device for nurbsdiff backend

    Returns:
        geomdl BSpline.Surface or dict with NURBS data
    """
    if backend == "auto":
        # Prefer nurbsdiff if available (no UV needed), else geomdl
        if HAS_NURBSDIFF:
            backend = "nurbsdiff"
        elif HAS_GEOMDL:
            backend = "geomdl"
        else:
            raise ImportError("No B-spline backend available. Install geomdl or nurbsdiff+pytorch3d")

    if backend == "nurbsdiff":
        return fit_nurbs_nurbsdiff(
            points, ctrl_pts_u, ctrl_pts_v, degree,
            device=device, **kwargs
        )
    elif backend == "geomdl":
        if uv_coords is None:
            raise ValueError("geomdl backend requires uv_coords. Use INR encoder or set backend='nurbsdiff'")
        return fit_bspline_geomdl(
            points, uv_coords, ctrl_pts_u, ctrl_pts_v, degree, **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'geomdl', or 'nurbsdiff'")


def sample_bspline_mesh(surf, mesh_dim: int = 100):
    """
    Sample a mesh from a geomdl B-spline surface.

    Args:
        surf: geomdl BSpline.Surface object
        mesh_dim: Grid resolution for mesh

    Returns:
        trimesh.Trimesh object
    """
    # Set evaluation delta (tuple for u and v directions)
    surf.delta = (1.0 / mesh_dim, 1.0 / mesh_dim)
    surf.evaluate()

    # Get evaluated points
    vertices = np.array(surf.evalpts)

    # Get actual sample size from geomdl
    pts_u, pts_v = surf.sample_size

    # Build faces for regular grid
    faces = []
    for i in range(pts_u - 1):
        for j in range(pts_v - 1):
            idx = i * pts_v + j
            # Two triangles per quad
            faces.append([idx, idx + pts_v, idx + 1])
            faces.append([idx + 1, idx + pts_v, idx + pts_v + 1])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def sample_nurbsdiff_mesh(nurbs_result, mesh_dim: int = 100):
    """
    Sample a mesh from NURBSDiff result.

    Args:
        nurbs_result: Dict from fit_nurbs_nurbsdiff
        mesh_dim: Grid resolution (uses pre-evaluated points if available)

    Returns:
        trimesh.Trimesh object
    """
    # Use pre-evaluated surface points
    surf_pts = nurbs_result["surface_points"][0]  # (out_dim_u, out_dim_v, 3)
    pts_u, pts_v = surf_pts.shape[:2]

    vertices = surf_pts.reshape(-1, 3).numpy()

    # Build faces for regular grid
    faces = []
    for i in range(pts_u - 1):
        for j in range(pts_v - 1):
            idx = i * pts_v + j
            faces.append([idx, idx + pts_v, idx + 1])
            faces.append([idx + 1, idx + pts_v, idx + pts_v + 1])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
