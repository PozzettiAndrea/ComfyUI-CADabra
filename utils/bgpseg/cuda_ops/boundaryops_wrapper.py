"""
BoundaryOps Python Wrapper Functions

Provides the Python interface for boundary-guided CUDA operations.
Adapted from BGPSeg/lib/boundaryops/functions/boundaryops.py
"""

import torch
from torch.autograd import Function
from typing import Tuple


class BoundaryQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset, edges, boundary, cuda_module):
        """
        Boundary-guided neighbor query.

        Samples neighbors while prioritizing points near primitive boundaries,
        which improves segmentation quality at instance edges.

        Args:
            nsample: Number of neighbors to sample
            xyz: (n, 3) Source point coordinates
            new_xyz: (m, 3) Query point coordinates
            offset: (b,) Source batch offsets
            new_offset: (b,) Query batch offsets
            edges: (n, k) Edge connectivity for boundary sampling
            boundary: (n,) Boundary labels (0=non-boundary, 1=boundary)
            cuda_module: CUDA module

        Returns:
            idx: (m, nsample) Neighbor indices
            dist: (m, nsample) Neighbor distances
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        b = edges.shape[1]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        cuda_module.boundaryquery_cuda(
            m, nsample, b, xyz, new_xyz, offset, new_offset, idx, dist2, edges, boundary
        )
        return idx, torch.sqrt(dist2)


def boundaryquery(nsample, xyz, new_xyz, offset, new_offset, edges, boundary, cuda_module):
    """Boundary query wrapper."""
    return BoundaryQuery.apply(
        nsample, xyz, new_xyz, offset, new_offset, edges, boundary, cuda_module
    )
