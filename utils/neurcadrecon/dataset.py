"""
NeurCADRecon Dataset Module

Dataset class for point cloud loading and sampling during training.
Adapted from NeurCADRecon (SIGGRAPH 2024) paper by Dong et al.

This version is modified to work with numpy arrays directly (for ComfyUI integration)
rather than requiring Open3D file loading.
"""

import numpy as np
import scipy.spatial as spatial
import torch.utils.data as data

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class ReconDataset(data.Dataset):
    """Dataset for NeurCADRecon training from point clouds.

    Generates training samples including:
    - Manifold points (points on surface)
    - Non-manifold points (random points in volume)
    - Near-surface points (for Morse loss)

    Args:
        points: Numpy array of point cloud (N, 3) or path to .ply file
        n_points: Number of points to sample per batch
        n_samples: Number of training samples (iterations)
        res: Grid resolution (unused, kept for compatibility)
        sample_type: Type of non-manifold sampling ('grid' or 'gaussian')
        grid_range: Range for uniform sampling of non-manifold points
        normals: Optional normals array (N, 3)
    """

    def __init__(self, points, n_points=20000, n_samples=128, res=128,
                 sample_type='gaussian', grid_range=1.1, normals=None):
        self.n_points = n_points
        self.n_samples = n_samples
        self.grid_range = grid_range

        # Load points from various input types
        self.points, self.mnfld_n = self._load_points(points, normals)

        # Compute bounding box
        self.bbox = np.array([
            np.min(self.points, axis=0),
            np.max(self.points, axis=0)
        ]).transpose()

        if HAS_TRIMESH:
            self.bbox_trimesh = trimesh.PointCloud(self.points).bounding_box.copy()
        else:
            self.bbox_trimesh = None

        self.point_idxs = np.arange(self.points.shape[0], dtype=np.int32)

        # Compute sigma for near-surface sampling
        self.sample_gaussian_noise_around_shape()

    def _load_points(self, points, normals=None):
        """Load points from various input types.

        Args:
            points: Can be:
                - numpy array (N, 3)
                - path to .ply file
                - trimesh.Trimesh (will sample surface)
                - trimesh.PointCloud

        Returns:
            points: Normalized points array (N, 3)
            normals: Normals array (N, 3) or zeros if unavailable
        """
        if isinstance(points, str):
            # Load from file
            if HAS_TRIMESH:
                mesh = trimesh.load(points, process=False)
                if isinstance(mesh, trimesh.Trimesh):
                    # Sample points from mesh surface
                    pts, face_idx = trimesh.sample.sample_surface(mesh, 50000)
                    pts = pts.astype(np.float32)
                    nrm = mesh.face_normals[face_idx].astype(np.float32)
                elif isinstance(mesh, trimesh.PointCloud):
                    pts = np.asarray(mesh.vertices, dtype=np.float32)
                    nrm = np.zeros_like(pts)
                else:
                    raise ValueError(f"Unsupported file content type: {type(mesh)}")
            else:
                raise ImportError("trimesh required for file loading")

        elif HAS_TRIMESH and isinstance(points, trimesh.Trimesh):
            # Sample from mesh surface
            pts, face_idx = trimesh.sample.sample_surface(points, 50000)
            pts = pts.astype(np.float32)
            nrm = points.face_normals[face_idx].astype(np.float32)

        elif HAS_TRIMESH and isinstance(points, trimesh.PointCloud):
            pts = np.asarray(points.vertices, dtype=np.float32)
            nrm = np.zeros_like(pts)

        elif isinstance(points, np.ndarray):
            pts = points.astype(np.float32)
            if normals is not None:
                nrm = normals.astype(np.float32)
            else:
                nrm = np.zeros_like(pts)

        else:
            raise ValueError(f"Unsupported points type: {type(points)}")

        # Center and scale to unit sphere
        self.cp = pts.mean(axis=0)
        pts = pts - self.cp[None, :]
        self.scale = np.abs(pts).max()
        pts = pts / self.scale

        return pts, nrm

    def sample_gaussian_noise_around_shape(self):
        """Compute local sigma for near-surface point sampling.

        Uses k-NN to estimate local point density and set appropriate
        noise scale for near-surface sampling.
        """
        kd_tree = spatial.KDTree(self.points)
        # Query 51 nearest neighbors (including self)
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        # Use distance to 50th neighbor as sigma
        sigmas = dist[:, -1:]
        self.sigmas = sigmas

    def __getitem__(self, index):
        """Get a training sample.

        Returns dict with:
            points: Manifold points (n_points, 3)
            mnfld_n: Manifold normals (n_points, 3)
            nonmnfld_points: Random volume points (n_points, 3)
            near_points: Near-surface points (n_points, 3)
        """
        # Random permutation for sampling
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:self.n_points]

        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        manifold_normals = self.mnfld_n[mnfld_idx]  # (n_points, 3)

        # Random uniform points in volume
        nonmnfld_points = np.random.uniform(
            -self.grid_range, self.grid_range,
            size=(self.n_points, 3)
        ).astype(np.float32)

        # Near-surface points: manifold + noise
        near_points = (
            manifold_points +
            self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0], manifold_points.shape[1])
        ).astype(np.float32)

        return {
            'points': manifold_points,
            'mnfld_n': manifold_normals,
            'nonmnfld_points': nonmnfld_points,
            'near_points': near_points
        }

    def get_train_data(self, batch_size):
        """Get training data batch (alternative interface).

        Args:
            batch_size: Number of points to sample

        Returns:
            manifold_points: Surface points
            near_points: Near-surface points
            all_points: All manifold points (for reference)
        """
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:batch_size]

        manifold_points = self.points[mnfld_idx]
        near_points = (
            manifold_points +
            self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0], manifold_points.shape[1])
        ).astype(np.float32)

        return manifold_points, near_points, self.points

    def gen_new_data(self, dense_pts):
        """Update dataset with new/denser point cloud.

        Args:
            dense_pts: New point cloud array
        """
        self.points = dense_pts
        kd_tree = spatial.KDTree(self.points)
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas

    def __len__(self):
        return self.n_samples

    def get_normalization_params(self):
        """Get normalization parameters for mesh reconstruction.

        Returns:
            cp: Center point used for normalization
            scale: Scale factor used for normalization
            bbox: Bounding box of normalized points
        """
        return self.cp, self.scale, self.bbox


def create_dataset_from_points(points, n_points=20000, n_samples=10000,
                                grid_range=1.1, normals=None):
    """Convenience function to create dataset from numpy points.

    Args:
        points: Point cloud as numpy array (N, 3) or trimesh object
        n_points: Points per training sample
        n_samples: Number of training iterations
        grid_range: Range for volume sampling
        normals: Optional normals array

    Returns:
        ReconDataset instance
    """
    return ReconDataset(
        points=points,
        n_points=n_points,
        n_samples=n_samples,
        grid_range=grid_range,
        normals=normals
    )
