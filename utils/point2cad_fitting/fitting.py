"""
Complete surface fitting pipeline for Point2CAD.
Combines primitive fitting and SplineINR for freeform surfaces.
Adapted from Point2CAD: https://github.com/prs-eth/point2cad
"""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .primitives import Fit
from .spline_inr import fit_one_inr_spline, sample_inr_mesh
from .utils import project_to_plane


# ============================================================================
# Mesh generation functions from Point2CAD fitting_utils.py
# ============================================================================

def up_sample_points_torch_memory_efficient(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two nearest neighbors and finds the centroid.
    """
    for _ in range(times):
        indices = []
        N = min(points.shape[0], 100)
        for i in range(points.shape[0] // N):
            diff_ = torch.sum(
                (
                    torch.unsqueeze(points[i * N : (i + 1) * N], 1)
                    - torch.unsqueeze(points, 0)
                )
                ** 2,
                2,
            )
            _, diff_indices = torch.topk(diff_, 5, 1, largest=False)
            indices.append(diff_indices)
        indices = torch.cat(indices, 0)
        neighbors = points[indices[:, 0:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def create_grid(input_pts, grid_points, size_u, size_v, thres=0.02, device="cuda"):
    """Create grid mask based on proximity to input points."""
    grid_points = torch.from_numpy(grid_points.astype(np.float32)).to(device)
    input_pts = torch.from_numpy(input_pts.astype(np.float32)).to(device)
    try:
        grid_points = grid_points.reshape((size_u + 2, size_v + 2, 3))
    except:
        grid_points = grid_points.reshape((size_u, size_v, 3))

    grid_points.permute(2, 0, 1)
    grid_points = torch.unsqueeze(grid_points, 0)

    filter_kernel = np.array(
        [[[0.25, 0.25], [0.25, 0.25]], [[0, 0], [0, 0]], [[0.0, 0.0], [0.0, 0.0]]]
    ).astype(np.float32)
    filter_kernel = np.stack([filter_kernel, np.roll(filter_kernel, 1, 0), np.roll(filter_kernel, 2, 0)])
    filter_kernel = torch.from_numpy(filter_kernel).to(device)
    grid_mean_points = F.conv2d(grid_points.permute(0, 3, 1, 2), filter_kernel, padding=0)
    grid_mean_points = grid_mean_points.permute(0, 2, 3, 1)
    try:
        grid_mean_points = grid_mean_points.reshape(((size_u + 1) * (size_v + 1), 3))
    except:
        grid_mean_points = grid_mean_points.reshape(((size_u - 1) * (size_v - 1), 3))

    diff = []
    for i in range(grid_mean_points.shape[0]):
        diff.append(
            torch.sum(
                (
                    torch.unsqueeze(grid_mean_points[i : i + 1], 1)
                    - torch.unsqueeze(input_pts, 0)
                )
                ** 2,
                2,
            )
        )
    diff = torch.cat(diff, 0)
    diff = torch.sqrt(diff)
    indices = torch.min(diff, 1)[0] < thres
    try:
        mask_grid = indices.reshape(((size_u + 1), (size_v + 1)))
    except:
        mask_grid = indices.reshape(((size_u - 1), (size_v - 1)))
    return mask_grid, diff, filter_kernel, grid_mean_points


def tessellate_points_fast(vertices, size_u, size_v, mask=None):
    """
    Given grid points, return a tessellation using triangles.
    If mask is given, those grids are avoided.
    """
    def index_to_id(i, j, size_v):
        return i * size_v + j

    triangles = []
    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None:
                if mask[i, j] == 0:
                    continue
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
            ]
            triangles.append(tri)
            tri = [
                index_to_id(i, j, size_v),
                index_to_id(i + 1, j + 1, size_v),
                index_to_id(i, j + 1, size_v),
            ]
            triangles.append(tri)

    if len(triangles) == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.stack(vertices, 0),
        faces=np.array(triangles)
    )
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    return mesh


def bit_mapping_points_torch(input_pts, output_points, thres, size_u, size_v, device="cuda"):
    """Create trimmed mesh from sampled points based on proximity to input."""
    mask, diff, filter_kernel, grid_mean_points = create_grid(
        input_pts, output_points, size_u, size_v, thres=thres, device=device
    )
    mesh = tessellate_points_fast(output_points, size_u, size_v, mask=mask.cpu().numpy())
    return mesh


def visualize_basic_mesh(shape_type, in_points, pred, epsilon=0.1, device="cuda"):
    """
    Create mesh for basic primitive shapes (from Point2CAD).
    """
    if shape_type == "plane":
        part_points = up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        e = epsilon if epsilon else 0.02
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["plane_new_points"]), e, 120, 120, device=device
        )

    elif shape_type == "sphere":
        part_points = up_sample_points_torch_memory_efficient(in_points, 2).data.cpu().numpy()
        e = epsilon if epsilon else 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["sphere_new_points"]), e, 100, 100, device=device
        )

    elif shape_type == "cylinder":
        part_points = up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        e = epsilon if epsilon else 0.03
        pred_mesh = bit_mapping_points_torch(
            part_points, np.array(pred["cylinder_new_points"]), e, 200, 60, device=device
        )

    elif shape_type == "cone":
        part_points = up_sample_points_torch_memory_efficient(in_points, 3).data.cpu().numpy()
        e = epsilon if epsilon else 0.03
        try:
            N = np.array(pred["cone_new_points"]).shape[0] // 51
            pred_mesh = bit_mapping_points_torch(
                part_points, np.array(pred["cone_new_points"]), e, N, 51, device=device
            )
        except:
            pred_mesh = None
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return pred_mesh


# ============================================================================
# Original fitting functions
# ============================================================================


def fit_basic_primitives(pts):
    """
    Fit basic primitives (plane, sphere, cylinder, cone) to points.

    Args:
        pts: Tensor of shape (N, 3)

    Returns:
        Dict with fitted parameters and errors for each primitive type
    """
    if pts.shape[0] < 20:
        raise ValueError("Too few points for primitive fitting (need >= 20)")

    fitting = Fit()
    recon = {}

    # Plane
    axis, distance = fitting.fit_plane_torch(pts)
    new_points = project_to_plane(pts, axis, distance.item())
    plane_err = torch.linalg.norm(new_points - pts, dim=-1).mean()
    recon["plane_params"] = (axis.data.cpu().numpy().tolist(), distance.data.cpu().numpy().tolist())
    # Compute extent from actual point cloud to ensure plane covers all points
    pts_np = pts.data.cpu().numpy()
    pts_extent = np.max(pts_np.max(axis=0) - pts_np.min(axis=0)) * 0.6  # 60% of max extent
    recon["plane_new_points"] = fitting.sample_plane(
        distance.item(), axis.data.cpu().numpy(),
        torch.mean(new_points, 0).data.cpu().numpy(),
        extent=pts_extent
    ).tolist()
    recon["plane_err"] = plane_err.data.cpu().numpy().tolist()

    # Sphere
    center, radius = fitting.fit_sphere_torch(pts)
    sphere_err = (torch.linalg.norm(pts - center, dim=-1) - radius).abs().mean()
    new_points, _ = fitting.sample_sphere(radius.item(), center.data.cpu().numpy())
    recon["sphere_params"] = (center.data.cpu().numpy().tolist(), radius.item())
    recon["sphere_new_points"] = new_points.tolist()
    recon["sphere_err"] = sphere_err.data.cpu().numpy().tolist()

    # Cylinder
    axis, center, radius = fitting.fit_cylinder(pts)
    new_points, _ = fitting.sample_cylinder_trim(radius, center, axis, pts.data.cpu().numpy())
    # Compute error as distance to cylinder axis minus radius
    from .primitives import distance_line_point
    cylinder_err = np.abs(distance_line_point(center, axis, pts.detach().cpu().numpy()) - radius).mean()
    recon["cylinder_params"] = (axis.tolist(), center.tolist(), radius.tolist() if hasattr(radius, 'tolist') else radius)
    recon["cylinder_new_points"] = new_points.tolist()
    recon["cylinder_err"] = float(cylinder_err)

    # Cone
    apex, axis, theta, cone_err, failure = fitting.fit_cone(pts)
    if not failure and apex is not None:
        # Reject degenerate cones (half-angle >= 87.6 degrees is essentially a plane)
        if theta >= 1.53:
            recon["cone_failure"] = True
            recon["cone_err"] = float('inf')
        else:
            new_points, _ = fitting.sample_cone_trim(apex, axis, theta, pts.data.cpu().numpy())
            if new_points is not None:
                recon["cone_params"] = (apex.tolist(), axis.tolist(), float(theta))
                recon["cone_new_points"] = new_points.tolist()
                recon["cone_err"] = float(cone_err)
                recon["cone_failure"] = False
            else:
                recon["cone_failure"] = True
                recon["cone_err"] = float('inf')
    else:
        recon["cone_failure"] = True
        recon["cone_err"] = float('inf')

    return recon


def fit_inrs(pts, device="cuda", num_attempts=1, seed=42, progress_bar=False, early_exit_threshold=0.002):
    """
    Fit SplineINR to points with progressive closure configurations.

    Optimization: Try open spline first (most common for CAD), and only
    try other configurations if the error is above threshold.

    Args:
        pts: Tensor of shape (N, 3)
        device: 'cuda' or 'cpu'
        num_attempts: Number of random restarts
        seed: Random seed
        progress_bar: Show progress
        early_exit_threshold: If error < this, skip remaining configs

    Returns:
        Dict with best fitted INR model and error
    """
    out_inr = None

    # Try open spline first (False, False) - most common for CAD surfaces
    try:
        cur_inr = fit_one_inr_spline(
            pts,
            is_u_closed=False,
            is_v_closed=False,
            num_fit_steps=1000,
            device=device,
            seed=seed,
            progress_bar=progress_bar,
        )
        out_inr = cur_inr

        # Early exit if open spline fits well enough
        if cur_inr['err'] < early_exit_threshold:
            out_inr["mesh_uv"] = sample_inr_mesh(out_inr, mesh_dim=100, uv_margin=0.2)
            return out_inr
    except Exception as e:
        print(f"      [INR] Failed (open): {e}")

    # Only try other configs if open spline wasn't good enough
    # Reversed order: try most-closed first (if it fits well, exit early)
    other_configs = [(True, True), (True, False), (False, True)]
    for is_u_closed, is_v_closed in other_configs:
        try:
            cur_inr = fit_one_inr_spline(
                pts,
                is_u_closed=is_u_closed,
                is_v_closed=is_v_closed,
                num_fit_steps=1000,
                device=device,
                seed=seed,
                progress_bar=progress_bar,
            )
            if out_inr is None or cur_inr['err'] < out_inr['err']:
                out_inr = cur_inr
                # Early exit if this config fits well enough
                if out_inr['err'] < early_exit_threshold:
                    break
        except Exception as e:
            print(f"      [INR] Failed (u={is_u_closed}, v={is_v_closed}): {e}")
            continue

    if out_inr is not None:
        out_inr["mesh_uv"] = sample_inr_mesh(out_inr, mesh_dim=100, uv_margin=0.2)

    return out_inr


def process_one_surface(points, segment_id, device="cuda", seed=42, progress_bar=False):
    """
    Complete surface fitting pipeline for a single segment.

    1. Fits basic primitives (plane, sphere, cylinder, cone)
    2. Fits SplineINR for freeform surfaces
    3. Selects best fit based on error with preference for primitives

    Args:
        points: Tensor or ndarray of shape (N, 3) in WORLD coordinates
        segment_id: ID of the segment (for linking back to segmentation)
        device: 'cuda' or 'cpu'
        seed: Random seed
        progress_bar: Show progress

    Returns:
        Dict with:
            - segment_id: ID linking to segmentation labels
            - type: 'plane', 'sphere', 'cylinder', 'cone', or 'open_spline'
            - params: Fitted parameters (in NORMALIZED space)
            - normalization: {offset, scale} for transforming to world coords
            - err: Fitting error (in normalized space)
            - For INR: inr_model, uv_bounds, mesh (pre-sampled in world coords)
    """
    if len(points) < 20:
        return None

    if not torch.is_tensor(points):
        points = torch.from_numpy(points.astype(np.float32))
    points = points.to(device)

    # Input points are ALREADY NORMALIZED by the Surface Fitting node
    # Fit primitives on normalized points
    recon_basic = fit_basic_primitives(points)

    # Collect primitive errors
    plane_err = recon_basic["plane_err"]
    sphere_err = recon_basic["sphere_err"]
    cylinder_err = recon_basic["cylinder_err"]
    cone_err = recon_basic.get("cone_err", float('inf'))
    if recon_basic.get("cone_failure", True):
        cone_err = float('inf')

    # Check if primitive is good enough to skip INR (saves ~40s per segment)
    SKIP_INR_THRESHOLD = 0.002
    best_prim_err = min(plane_err, sphere_err, cylinder_err, cone_err)
    if best_prim_err < SKIP_INR_THRESHOLD:
        recon_inr = None  # Skip expensive INR training
    else:
        # Fit INR (on the same normalized points)
        recon_inr = fit_inrs(points, device=device, seed=seed, progress_bar=progress_bar)

    # Get INR error for reporting (always show it)
    inr_err_raw = recon_inr["err"] if recon_inr else None
    # For selection: use INR only if it's a reasonable fit (relaxed threshold)
    inr_err_for_selection = recon_inr["err"] if recon_inr and recon_inr.get("err", float('inf')) < 0.5 else float('inf')

    # Select best shape with preference for primitives
    all_errors = np.array([plane_err, sphere_err, cylinder_err, cone_err, inr_err_for_selection])
    min_idx = np.argmin(all_errors)

    # Prefer primitives if error is similar to INR
    preference_thres = 0.008
    preference_increment = 0.001
    if min_idx == 4:  # INR was best
        prim_min = np.min([plane_err, sphere_err, cylinder_err, cone_err])
        if prim_min < preference_thres or prim_min < inr_err_for_selection + preference_increment:
            min_idx = np.argmin([plane_err, sphere_err, cylinder_err, cone_err])

    # Prefer plane over cone when errors are close (from original Point2CAD)
    # Cone has more degrees of freedom and can overfit to planar surfaces
    if min_idx == 3:  # Cone was selected
        if np.abs(plane_err - cone_err) <= 1e-5:
            min_idx = 0  # Use plane instead

    # Build result with new clean structure
    shape_names = ["plane", "sphere", "cylinder", "cone", "open_spline"]

    result = {
        "segment_id": segment_id,
        "type": shape_names[min_idx],
        # Store all errors for reporting (always show INR error if available)
        "all_errors": {
            "plane": float(plane_err),
            "sphere": float(sphere_err),
            "cylinder": float(cylinder_err),
            "cone": float(cone_err) if cone_err != float('inf') else None,
            "open_spline": float(inr_err_raw) if inr_err_raw is not None else None,
        },
    }

    # Generate mesh for ALL surface types (in NORMALIZED space)
    # Surface Fitting node will denormalize back to world coordinates
    if min_idx == 0:  # Plane
        result["params"] = recon_basic["plane_params"]
        result["err"] = float(plane_err)
        mesh = visualize_basic_mesh("plane", points, recon_basic, epsilon=0.1, device=device)
        if mesh is not None:
            result["mesh"] = mesh

    elif min_idx == 1:  # Sphere
        result["params"] = recon_basic["sphere_params"]
        result["err"] = float(sphere_err)
        mesh = visualize_basic_mesh("sphere", points, recon_basic, epsilon=0.1, device=device)
        if mesh is not None:
            result["mesh"] = mesh

    elif min_idx == 2:  # Cylinder
        result["params"] = recon_basic["cylinder_params"]
        result["err"] = float(cylinder_err)
        mesh = visualize_basic_mesh("cylinder", points, recon_basic, epsilon=0.1, device=device)
        if mesh is not None:
            result["mesh"] = mesh

    elif min_idx == 3:  # Cone
        result["params"] = recon_basic["cone_params"]
        result["err"] = float(cone_err)
        mesh = visualize_basic_mesh("cone", points, recon_basic, epsilon=0.1, device=device)
        if mesh is not None:
            result["mesh"] = mesh

    else:  # INR
        result["params"] = None
        result["err"] = float(inr_err_raw) if inr_err_raw else None
        if recon_inr:
            result["inr_model"] = recon_inr["model"]
            result["uv_bounds"] = {
                "min": recon_inr["uv_bb_min"],
                "max": recon_inr["uv_bb_max"],
            }
            result["is_u_closed"] = recon_inr["is_u_closed"]
            result["is_v_closed"] = recon_inr["is_v_closed"]
            # Pre-sampled mesh in normalized space
            # Surface Fitting node will denormalize to world coordinates
            if "mesh_uv" in recon_inr:
                result["mesh"] = recon_inr["mesh_uv"]

    return result
