"""
GPU-accelerated Mean Shift Clustering for BGPSeg

Standalone implementation adapted from BGPSeg/util/loss_util.py
Used for clustering primitive embeddings into instance segments.
"""

import torch
import numpy as np
import math
from torch import sqrt, exp
from typing import Tuple, List


class MeanShiftGPU:
    """
    GPU-accelerated Mean Shift clustering implementation.

    Mean shift is a non-parametric clustering algorithm that finds modes
    (local maxima) in the feature space density. Used in BGPSeg to cluster
    per-point embeddings into primitive instances.

    Args:
        bandwidth: Kernel bandwidth - controls cluster granularity.
                   Higher values = fewer, larger clusters.
                   Default 1.31 from BGPSeg config.
        batch_size: Batch size for GPU processing (memory vs speed tradeoff)
        max_iter: Maximum iterations for mode finding
        eps: Convergence threshold
        cluster_eps: Distance threshold for cluster assignment
        check_converge: Whether to check convergence (slower but more accurate)
    """

    def __init__(
        self,
        bandwidth: float = 1.31,
        batch_size: int = 700,
        max_iter: int = 10,
        eps: float = 1e-5,
        cluster_eps: float = 1e-1,
        check_converge: bool = False
    ):
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eps = eps
        self.cluster_eps = cluster_eps
        self.check_converge = check_converge

    def _distance_batch(self, a: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between a and each element in B."""
        return sqrt(((a[None, :] - B[:, None]) ** 2)).sum(2)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt(((a - b) ** 2).sum())

    def _gaussian(self, dist: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Gaussian kernel for weighting neighbors."""
        return exp(-0.5 * ((dist / bandwidth)) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

    def _cluster_points(self, points: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        """
        Assign converged points to clusters based on proximity.

        After mean shift converges, points that converged to similar
        locations are grouped into the same cluster.
        """
        cluster_ids = []
        cluster_centers = []
        cluster_idx = 0

        for i, point in enumerate(points):
            if len(cluster_ids) == 0:
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                assigned = False
                for j, center in enumerate(cluster_centers):
                    dist = self._distance(point, center)
                    if dist < self.cluster_eps:
                        cluster_ids.append(j)
                        assigned = True
                        break
                if not assigned:
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1

        return cluster_ids, cluster_centers

    def fit(self, data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Perform mean shift clustering on input data.

        Args:
            data: (N, D) tensor of feature vectors to cluster

        Returns:
            labels: (N,) array of cluster assignments
            centers: (K, D) array of cluster centers
            X: (N, D) tensor of converged feature positions
        """
        with torch.no_grad():
            n = len(data)

            # Move to GPU if not already
            if not data.is_cuda:
                data_gpu = data.cuda()
                X = data_gpu.clone()
            else:
                X = data.clone()

            # Mean shift iterations
            for _ in range(self.max_iter):
                max_dis = 0
                for i in range(0, n, self.batch_size):
                    s = slice(i, min(n, i + self.batch_size))

                    if self.check_converge:
                        dis = self._distance_batch(X, X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch
                        weight = self._gaussian(dis, self.bandwidth)
                    else:
                        weight = self._gaussian(
                            self._distance_batch(X, X[s]),
                            self.bandwidth
                        )

                    # Update positions towards weighted mean
                    num = (weight[:, :, None] * X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:, None]

                # Check convergence
                if self.check_converge and max_dis < self.eps:
                    break

            # Cluster the converged points
            points = X.cpu().data.numpy()
            labels, centers = self._cluster_points(points)

            labels = np.array(labels)
            centers = np.array(centers)

            return labels, centers, X


def mean_shift_gpu(
    x: torch.Tensor,
    offset: torch.Tensor,
    bandwidth: float = 1.31,
    batch_size: int = 700
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Batch-aware mean shift clustering for multiple point clouds.

    Args:
        x: (N, D) Features to cluster
        offset: (B,) Batch offsets indicating boundaries between samples
        bandwidth: Mean shift kernel bandwidth
        batch_size: GPU batch size

    Returns:
        IDX: (N,) Cluster labels per point
        X_fea: Converged feature tensor
    """
    N, c = x.shape
    IDX = np.zeros(N, dtype=int)

    ms = MeanShiftGPU(
        bandwidth=bandwidth,
        batch_size=batch_size,
        max_iter=10,
        eps=1e-5,
        check_converge=False
    )

    for i in range(len(offset)):
        if i == 0:
            pred = x[0:offset[i]]
        else:
            pred = x[offset[i - 1]:offset[i]]

        labels, centers, X_fea = ms.fit(pred)

        if i == 0:
            IDX[0:offset[i]] = labels
        else:
            IDX[offset[i - 1]:offset[i]] = labels

    return IDX, X_fea


def cluster_embeddings(
    embeddings: torch.Tensor,
    bandwidth: float = 1.31,
    batch_size: int = 700,
    max_iter: int = 10
) -> Tuple[np.ndarray, int]:
    """
    Convenience function for single-sample clustering.

    Args:
        embeddings: (N, D) tensor of per-point embeddings
        bandwidth: Mean shift bandwidth parameter
        batch_size: GPU batch size
        max_iter: Maximum iterations

    Returns:
        labels: (N,) Cluster ID per point
        num_clusters: Number of clusters found
    """
    ms = MeanShiftGPU(
        bandwidth=bandwidth,
        batch_size=batch_size,
        max_iter=max_iter,
        check_converge=False
    )

    labels, centers, _ = ms.fit(embeddings)
    num_clusters = len(centers)

    return labels, num_clusters
