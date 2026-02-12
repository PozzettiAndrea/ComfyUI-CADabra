"""
NeurCADRecon Utility Functions

Core utilities for gradient computation and mesh extraction.
Adapted from NeurCADRecon (SIGGRAPH 2024) paper by Dong et al.
"""

import numpy as np
import torch
from torch.autograd import grad
from tqdm import tqdm

try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False
    try:
        from skimage import measure
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    """Compute gradient of outputs with respect to inputs.

    Args:
        inputs: Input tensor (requires grad)
        outputs: Output tensor from network
        create_graph: Whether to create graph for higher-order derivatives
        retain_graph: Whether to retain computation graph

    Returns:
        Gradient tensor with same shape as inputs
    """
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True
    )[0]
    return points_grad


def get_cuda_ifavailable(torch_obj, device=None):
    """Move tensor to CUDA if available."""
    if torch.cuda.is_available():
        return torch_obj.cuda(device=device)
    else:
        return torch_obj


def scale_pc_to_unit_sphere(points, cp=None, s=None):
    """Scale point cloud to unit sphere centered at origin.

    Args:
        points: Point cloud array (N, 3)
        cp: Optional center point (computed if None)
        s: Optional scale factor (computed if None)

    Returns:
        scaled_points: Normalized point cloud
        cp: Center point used
        s: Scale factor used
    """
    if cp is None:
        cp = points.mean(axis=0)
    points = points - cp[None, :]
    if s is None:
        s = np.linalg.norm(points, axis=-1).max(-1)
    points = points / s
    return points, cp, s


def get_3d_grid(resolution=100, bbox=1.2 * np.array([[-1, 1], [-1, 1], [-1, 1]]),
                device=None, eps=0.1, dtype=np.float32):
    """Generate 3D grid points for marching cubes.

    Args:
        resolution: Grid resolution along shortest axis
        bbox: Bounding box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        device: Target device for tensor
        eps: Padding around bounding box
        dtype: Data type for coordinates

    Returns:
        Dictionary with grid_points, xyz coordinates, and axis info
    """
    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])

    if shortest_axis == 0:
        x = np.linspace(bbox[0, 0] - eps, bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif shortest_axis == 1:
        y = np.linspace(bbox[1, 0] - eps, bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif shortest_axis == 2:
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype))
    grid_points = torch.tensor(
        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T,
        dtype=torch.float32
    )

    return {
        "grid_points": grid_points,
        "shortest_axis_length": length,
        "xyz": [x, y, z],
        "shortest_axis_index": shortest_axis
    }


def implicit2mesh(decoder, mods, grid_res, translate=[0., 0., 0.], scale=1,
                  get_mesh=True, device=None, bbox=np.array([[-1, 1], [-1, 1], [-1, 1]]),
                  batch_size=10000, verbose=True):
    """Extract mesh from implicit SDF network using marching cubes.

    Args:
        decoder: SDF decoder network
        mods: Modulation parameters (None for standard decoder)
        grid_res: Resolution of marching cubes grid
        translate: Translation to apply to output vertices
        scale: Scale factor for output vertices
        get_mesh: Whether to return trimesh object
        device: Computation device
        bbox: Bounding box for evaluation
        batch_size: Points to evaluate per batch
        verbose: Whether to print progress

    Returns:
        trimesh.Trimesh object representing the extracted surface
    """
    if not HAS_MCUBES and not HAS_SKIMAGE:
        raise ImportError("Either mcubes or scikit-image is required for mesh extraction")

    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh output")

    mesh = None
    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"]

    # Evaluate SDF on grid points
    z = []
    iterator = torch.split(pnts, batch_size, dim=0)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating SDF on grid")

    for point in iterator:
        point = get_cuda_ifavailable(point, device=device)
        with torch.no_grad():
            z.append(decoder(point.type(torch.float32), mods).detach().squeeze(0).cpu().numpy())

    z = np.concatenate(z, axis=0).reshape(
        grid_dict['xyz'][1].shape[0],
        grid_dict['xyz'][0].shape[0],
        grid_dict['xyz'][2].shape[0]
    ).transpose([1, 0, 2]).astype(np.float64)

    if verbose:
        print(f"SDF range: [{z.min():.4f}, {z.max():.4f}]")

    # Run marching cubes
    thresh = 0.0
    if np.sum(z > 0.0) < np.sum(z < 0.0):
        thresh = -thresh

    if HAS_MCUBES:
        verts, faces = mcubes.marching_cubes(z, 0)
        # mcubes returns vertices in grid index space, need to scale by cell_width
        verts = verts * cell_width
    else:
        verts, faces, _, _ = measure.marching_cubes(
            volume=z, level=thresh,
            spacing=(cell_width, cell_width, cell_width),
            method='lewiner'
        )

    # Transform from grid space to normalized coordinate space
    # Add the grid origin offset
    verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])

    # Transform back to world coordinates
    # Original normalization was: normalized = (world - cp) / scale
    # So inverse is: world = normalized * scale + cp
    # We receive: translate = -cp, scale = 1/original_scale
    # So: world = verts * (1/scale) - translate = verts * original_scale + cp
    verts = verts * (1 / scale) - translate

    if get_mesh:
        mesh = trimesh.Trimesh(verts, faces, validate=True)

    return mesh


def normalize_mesh(mesh, file_out=None):
    """Normalize mesh to fit in unit cube [-0.5, 0.5].

    Args:
        mesh: trimesh.Trimesh object
        file_out: Optional output file path

    Returns:
        Normalized trimesh.Trimesh object
    """
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return mesh

    # Translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation_matrix = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation_matrix)

    # Scale to unit cube
    scale = 1.0 / bounds.max()
    scale_matrix = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_matrix)

    if file_out is not None:
        mesh.export(file_out)

    return mesh


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def center_and_scale(points, cp=None, scale=None):
    """Center and scale point cloud to unit sphere.

    Args:
        points: Point cloud array (B, N, 3) or (N, 3)
        cp: Optional center point
        scale: Optional scale factor

    Returns:
        points: Centered and scaled points
        cp: Center point used
        scale: Scale factor used
    """
    if points.ndim == 2:
        # Single point cloud (N, 3)
        if cp is None:
            cp = points.mean(axis=0)
        points = points - cp[None, :]
        if scale is None:
            scale = np.linalg.norm(points, axis=-1).max()
        points = points / scale
    else:
        # Batch of point clouds (B, N, 3)
        if cp is None:
            cp = points.mean(axis=1)
        points = points - cp[:, None, :]
        if scale is None:
            scale = np.linalg.norm(points, axis=-1).max(-1)
        points = points / scale[:, None, None]

    return points, cp, scale
