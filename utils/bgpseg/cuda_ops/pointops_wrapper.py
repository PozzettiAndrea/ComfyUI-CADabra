"""
PointOps Python Wrapper Functions

Provides the Python interface for CUDA point cloud operations.
Adapted from BGPSeg/lib/pointops/functions/pointops.py
"""

import torch
from torch.autograd import Function
from typing import Tuple, Optional


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset, cuda_module):
        """
        Furthest point sampling.

        Args:
            xyz: (n, 3) Point coordinates
            offset: (b,) Batch offsets
            new_offset: (b,) Target batch offsets after sampling

        Returns:
            idx: (m,) Indices of sampled points
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        cuda_module.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx


def furthestsampling(xyz, offset, new_offset, cuda_module):
    """Furthest point sampling wrapper."""
    return FurthestSampling.apply(xyz, offset, new_offset, cuda_module)


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset, cuda_module):
        """
        K-nearest neighbor query.

        Args:
            nsample: Number of neighbors
            xyz: (n, 3) Source point coordinates
            new_xyz: (m, 3) Query point coordinates
            offset: (b,) Source batch offsets
            new_offset: (b,) Query batch offsets

        Returns:
            idx: (m, nsample) Neighbor indices
            dist: (m, nsample) Neighbor distances
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        cuda_module.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, torch.sqrt(dist2)


def knnquery(nsample, xyz, new_xyz, offset, new_offset, cuda_module):
    """KNN query wrapper."""
    return KNNQuery.apply(nsample, xyz, new_xyz, offset, new_offset, cuda_module)


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx, cuda_module):
        """
        Group features by indices.

        Args:
            input: (n, c) Features
            idx: (m, nsample) Indices

        Returns:
            output: (m, nsample, c) Grouped features
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        cuda_module.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        ctx.cuda_module = cuda_module
        return output

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.n
        idx, = ctx.saved_tensors
        cuda_module = ctx.cuda_module
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        cuda_module.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None, None


def grouping(input, idx, cuda_module):
    """Grouping wrapper."""
    return Grouping.apply(input, idx, cuda_module)


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz, cuda_module):
    """
    Query neighbors and group features.

    Args:
        nsample: Number of neighbors
        xyz: (n, 3) Source point coordinates
        new_xyz: (m, 3) Query point coordinates
        feat: (n, c) Source features
        idx: Optional precomputed indices
        offset: (b,) Source batch offsets
        new_offset: (b,) Query batch offsets
        use_xyz: Whether to include xyz in output
        cuda_module: CUDA module

    Returns:
        new_feat: (m, nsample, c+3) or (m, nsample, c) Grouped features
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset, cuda_module)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat


def interpolation(xyz, new_xyz, feat, offset, new_offset, k, cuda_module):
    """
    Feature interpolation using inverse distance weighting.

    Args:
        xyz: (m, 3) Source point coordinates
        new_xyz: (n, 3) Target point coordinates
        feat: (m, c) Source features
        offset: (b,) Source batch offsets
        new_offset: (b,) Target batch offsets
        k: Number of neighbors for interpolation
        cuda_module: CUDA module

    Returns:
        new_feat: (n, c) Interpolated features
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset, cuda_module)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        # Handle potential index out of bounds
        if idx[:, i].long().max() >= feat.shape[0]:
            idx[idx >= feat.shape[0]] = feat.shape[0] - 1
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)

    return new_feat
