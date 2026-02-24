"""
Primitive fitting for Point2CAD.
Fits plane, sphere, cylinder, cone to point clouds.
Adapted from Point2CAD: https://github.com/prs-eth/point2cad
"""

import numpy as np
import torch
from scipy import optimize
from scipy.optimize import minimize

from .utils import (
    LeastSquares, customsvd, guard_sqrt, project_to_plane,
    rotation_matrix_a_to_b, get_rotation_matrix, regular_parameterization, EPS
)


class Fit:
    """Primitive fitting and sampling for plane, sphere, cylinder, cone."""

    def __init__(self):
        LS = LeastSquares()
        self.lstsq = LS.lstsq

    def fit_plane_torch(self, points, normals=None, weights=None):
        """Fit a plane to points. Returns (normal, distance)."""
        if weights is None:
            weights = torch.ones_like(points)[:, :1]
        weights_sum = torch.sum(weights) + EPS
        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum
        weighted_X = weights * X
        U, s, V = customsvd(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))
        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
        return a, d

    def fit_sphere_torch(self, points, normals=None, weights=None):
        """Fit a sphere to points. Returns (center, radius)."""
        if weights is None:
            weights = torch.ones_like(points)[:, :1]
        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (-points + torch.sum(points * weights, 0) / sum_weights)
        dot_points = weights * torch.sum(points * points, 1, keepdim=True)
        normalization = torch.sum(dot_points) / sum_weights
        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y
        center = -self.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1)) / sum_weights
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    def fit_cylinder(self, points, normals=None, weights=None):
        """Fit a cylinder to points. Returns (axis, center, radius)."""
        w_fit, C_fit, r_fit, _ = fitcylinder(points.detach().cpu().numpy())
        return w_fit, C_fit, r_fit

    def fit_cone(self, points, normals=None, weights=None):
        """Fit a cone to points. Returns (apex, axis, theta, error, failure)."""
        return fitcone(points.detach().cpu().numpy())

    def sample_plane(self, d, n, mean, extent=None):
        """Sample points on a fitted plane.

        Args:
            d: Distance from origin
            n: Normal vector
            mean: Center point
            extent: Optional size of the plane (if None, uses 0.75 for normalized data)
        """
        regular_parameters = regular_parameterization(120, 120)
        n = n.reshape(3)
        r1, r2 = np.random.random(), np.random.random()
        a = (d - r1 * n[1] - r2 * n[2]) / (n[0] + EPS)
        x = np.array([a, r1, r2]) - d * n
        x = x / np.linalg.norm(x)
        n = n.reshape((1, 3))
        y = np.cross(x, n)
        y = y / np.linalg.norm(y)
        param = 1 - 2 * np.array(regular_parameters)
        # Scale based on extent if provided, otherwise use 0.75 (original Point2CAD)
        scale = extent if extent is not None else 0.75
        param = param * scale
        gridded_points = param[:, 0:1] * x + param[:, 1:2] * y
        gridded_points = gridded_points + mean
        return gridded_points

    def sample_sphere(self, radius, center, N=1000):
        """Sample points on a fitted sphere."""
        center = center.reshape((1, 3))
        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        lam = np.linspace(-radius + 1e-7, radius - 1e-7, 100)
        radii = np.sqrt(radius**2 - lam**2)
        circle = np.concatenate([circle] * lam.shape[0], 0)
        spread_radii = np.repeat(radii, d_theta, 0)
        new_circle = circle * spread_radii.reshape((-1, 1))
        height = np.repeat(lam, d_theta, 0)
        points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
        points = points - np.mean(points, 0)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points + center
        return points, normals

    def sample_cylinder_trim(self, radius, center, axis, points, N=1000):
        """Sample points on a fitted cylinder, trimmed to input extent."""
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))
        d_theta, d_height = 60, 100
        R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])
        points = points - center
        projection = points @ axis
        min_proj = np.squeeze(projection[np.argmin(projection)]) - 0.1
        max_proj = np.squeeze(projection[np.argmax(projection)]) + 0.1
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius
        normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
        height = np.repeat(height, d_theta, axis=0)
        points = np.concatenate([circle, height], 1)
        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T
        return points, normals

    def sample_cone_trim(self, apex, axis, theta, points):
        """Sample points on a fitted cone, trimmed to input extent."""
        if apex is None:
            return None, None
        c = apex.reshape((3))
        a = axis.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        proj_max = np.max(proj) + 0.2 * np.abs(np.max(proj))
        proj_min = np.min(proj) - 0.2 * np.abs(np.min(proj))

        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        d = np.array([x, 1, 1])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d
        p = p.reshape((3, 1))

        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        sampled_points, normals = [], []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        rel_unit_vector = (p - c) / np.linalg.norm(p - c)
        rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta) + EPS)
        rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta) + EPS)

        for j in range(100):
            p_ = rel_unit_vector_min + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j
            d_points, d_normals = [], []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(rotate_point - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a)
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])
            sampled_points += d_points
            normals += d_normals

        sampled_points = np.stack(sampled_points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)
        proj = (sampled_points - c.reshape((1, 3))) @ a
        proj = proj[:, 0]
        indices = np.logical_and(proj < proj_max, proj > proj_min)
        return sampled_points[indices], normals[indices]


# Cylinder fitting using Eberly's method
def fitcylinder(data, guess_angles=None):
    """Fit cylinder to points. Returns (axis, center, radius, error)."""
    def direction(theta, phi):
        return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])

    def projection_matrix(w):
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n
        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

    def G(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))
        u = sum(np.dot(Y, Y) for Y in Ys) / n
        v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))
        return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

    def C(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))
        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

    def r(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)
        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    Xs, t = preprocess_data(data)
    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    best_fit, best_score = None, float("inf")
    for sp in start_points:
        fitted = minimize(lambda x: G(direction(x[0], x[1]), Xs), sp, method="Powell", tol=1e-6)
        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])
    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun


# Cone fitting
def distance_line_point(anchor, direction, point):
    """Distance from point to line defined by anchor and direction."""
    direction = direction / np.linalg.norm(direction)
    v = point - anchor
    t = np.dot(v, direction)
    projection = anchor + t[:, np.newaxis] * direction
    return np.linalg.norm(point - projection, axis=-1)


class Cone:
    def __init__(self, theta, axis, vertex):
        self.vertex = vertex
        self.axis = axis / np.linalg.norm(axis)
        self.theta = theta

    def distance_to_point(self, point):
        a = distance_line_point(self.vertex, self.axis, point)
        k = a * np.tan(self.theta)
        b = k + np.abs(np.dot((point - self.vertex), self.axis))
        l = b * np.sin(self.theta)
        d = a / np.cos(self.theta) - l
        return np.abs(d)


def fitcone(points, weights=None):
    """Fit cone to points. Returns (apex, axis, theta, error, failure)."""
    initial_guesses = [
        Cone(0.0, np.array([1.0, 0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 1.0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 0, 1.0]), np.zeros(3)),
    ]

    def cone_fit_residuals(cone_params, points, weights):
        cone = Cone(cone_params[0], cone_params[1:4], cone_params[4:7])
        distances = cone.distance_to_point(points)
        if weights is None:
            return distances
        return distances * np.sqrt(weights)

    best_fit, best_score = None, float("inf")
    for initial_guess in initial_guesses:
        x0 = np.concatenate([np.array([initial_guess.theta]), initial_guess.axis, initial_guess.vertex])
        results = optimize.least_squares(cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10)
        if results.success and results.fun.sum() < best_score:
            best_score = results.fun.sum()
            best_fit = results

    if best_fit is None:
        return None, None, None, None, True

    apex = best_fit.x[4:7]
    axis = best_fit.x[1:4]
    theta = best_fit.x[0]
    err = best_fit.fun.mean()
    return apex, axis, theta, err, False


# ============================================================================
# Pure NumPy fitting functions (no torch dependency)
# ============================================================================

def fit_plane_numpy(points):
    """
    Fit plane to points using SVD (pure numpy).

    Args:
        points: ndarray of shape (N, 3)

    Returns:
        normal: ndarray of shape (3,) - plane normal vector
        distance: float - signed distance from origin to plane
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Smallest singular value corresponds to normal direction
    distance = np.dot(normal, centroid)
    return normal, distance


def fit_sphere_numpy(points):
    """
    Fit sphere to points using algebraic least-squares (pure numpy).

    Solves the linearized sphere equation:
    |p - c|^2 = r^2  =>  2*c.dot(p) - |c|^2 + r^2 = |p|^2

    Args:
        points: ndarray of shape (N, 3)

    Returns:
        center: ndarray of shape (3,) - sphere center
        radius: float - sphere radius
    """
    # Build design matrix: [2*x, 2*y, 2*z, 1] for each point
    A = np.hstack([2 * points, np.ones((len(points), 1))])
    # Right-hand side: |p|^2 for each point
    b = np.sum(points**2, axis=1)
    # Solve least-squares: result = [cx, cy, cz, r^2 - |c|^2]
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = result[:3]
    # Recover radius from: result[3] = r^2 - |c|^2
    radius_sq = result[3] + np.dot(center, center)
    radius = np.sqrt(max(radius_sq, 1e-6))  # Clamp to avoid negative sqrt
    return center, radius
