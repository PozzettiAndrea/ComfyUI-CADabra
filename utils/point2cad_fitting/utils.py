"""
Utility functions for Point2CAD fitting.
Adapted from Point2CAD: https://github.com/prs-eth/point2cad
"""

import numpy as np
import torch
from torch.autograd import Function

EPS = np.finfo(np.float32).eps


def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)


def guard_exp(x, max_value=75, min_value=-75):
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def rotation_matrix_a_to_b(A, B):
    """Rotation matrix from vector A to vector B. B = R @ A"""
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def get_rotation_matrix(theta):
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    return R


def regular_parameterization(grid_u, grid_v):
    nx, ny = (grid_u, grid_v)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    parameters = np.concatenate([xv, yv], 1)
    return parameters


def get_rng(device, seed=None, seed_increment=0):
    if seed is None:
        if device == "cpu":
            rng = torch.random.default_generator
        else:
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device("cuda", device)
            device_idx = device.index
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            rng = torch.cuda.default_generators[device_idx]
    else:
        rng = torch.Generator(device)
        rng.manual_seed(seed + seed_increment)
    return rng


# Custom SVD for stable gradients
def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    device = S.device
    S = torch.eye(N, device=device) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1
    device = S.device
    eps = torch.ones((N, N), device=device) * 10 ** (-6)
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)
    K_neg = sign_diff * max_diff
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus
    ones = torch.ones((N, N), device=device)
    rm_diag = ones - torch.eye(N, device=device)
    K = K_neg * K_pos * rm_diag
    return K


class CustomSVD(Function):
    @staticmethod
    def forward(ctx, input):
        U, S, V = torch.svd(input, some=True)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input


customsvd = CustomSVD.apply


class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """Differentiable least squares: solve Ax = Y"""
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            raise RuntimeError("Infinity in least squares")

        if cols == torch.linalg.matrix_rank(A):
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            AtA = A.transpose(1, 0) @ A
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols, device=A.device)
            Y_dash = A.transpose(1, 0) @ Y
            x = self.lstsq(A_dash, Y_dash, 1)
        return x


def best_lambda(A):
    """Find smallest lambda that makes A + lambda*I invertible."""
    lamb = 1e-6
    cols = A.shape[0]
    for i in range(7):
        A_dash = A + lamb * torch.eye(cols, device=A.device)
        if cols == torch.linalg.matrix_rank(A_dash):
            break
        else:
            lamb *= 10
    return lamb


def project_to_plane(points, a, d):
    """Project points onto plane with normal a and distance d from origin."""
    a = a.reshape((3, 1))
    a = a / torch.norm(a, 2)
    projections = points - ((points @ a).permute(1, 0) * a).permute(1, 0)
    projections = projections + a.transpose(1, 0) * d
    return projections
