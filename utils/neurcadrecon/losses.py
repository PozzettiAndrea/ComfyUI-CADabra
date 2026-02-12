"""
NeurCADRecon Loss Functions

Morse loss with Gaussian curvature regularization for CAD surface reconstruction.
Adapted from NeurCADRecon (SIGGRAPH 2024) paper by Dong et al.
"""

import torch
import torch.nn as nn

from .utils import gradient


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    """Compute eikonal loss that penalizes when ||grad(f)|| != 1.

    The eikonal equation ensures the network output is a valid signed distance function.

    Args:
        nonmnfld_grad: Gradients at off-manifold points (B, N, 3)
        mnfld_grad: Gradients at manifold points (B, N, 3)
        eikonal_type: 'abs' for L1, 'square' for L2

    Returns:
        Eikonal loss value
    """
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad
    else:
        return torch.tensor(0.0)

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term


def latent_rg_loss(latent_reg, device):
    """Compute VAE latent representation regularization loss."""
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)
    return reg_loss


def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
    """Compute Gaussian curvature loss (Morse loss).

    This is the key innovation of NeurCADRecon - it enforces zero Gaussian
    curvature which is a characteristic property of CAD surfaces (developable surfaces).

    Args:
        nonmnfld_hessian_term: Hessian matrix at points (B, N, 3, 3)
        morse_nonmnfld_grad: Gradient at points (B, N, 3)

    Returns:
        Gaussian curvature loss value
    """
    device = morse_nonmnfld_grad.device

    # Construct augmented Hessian matrix for Gaussian curvature computation
    # [H, grad]
    # [grad^T, 0]
    nonmnfld_hessian_term = torch.cat(
        (nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]),
        dim=-1
    )

    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
        device=device
    )
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)

    # Gaussian curvature = -det(augmented_H) / ||grad||^4
    morse_nonmnfld = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
        nonmnfld_hessian_term
    )

    morse_nonmnfld = morse_nonmnfld.abs()
    morse_loss = morse_nonmnfld.mean()

    return morse_loss


class MorseLoss(nn.Module):
    """Combined loss function for NeurCADRecon training.

    Combines multiple loss terms:
    - SDF term: Points on manifold should have SDF = 0
    - Inter term: Exponential penalty for off-manifold points near zero
    - Normal term: Optional normal supervision
    - Eikonal term: ||grad(SDF)|| = 1
    - Morse term: Gaussian curvature regularization (key for CAD surfaces)

    Args:
        weights: Loss weights [sdf, inter, normal, eikonal, div, morse]
        loss_type: Loss configuration type
        div_decay: Decay schedule for morse weight ('none', 'linear', 'quintic', 'step')
        div_type: Loss type for divergence term
        bidirectional_morse: Apply morse loss on both manifold and non-manifold points
        udf: Whether using unsigned distance function
    """

    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', bidirectional_morse=True, udf=False):
        super().__init__()
        if weights is None:
            weights = [7e3, 6e2, 1e2, 5e1, 0, 10]
        self.weights = weights  # [sdf, inter, normal, eikonal, div, morse]
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.bidirectional_morse = bidirectional_morse
        self.udf = udf

    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None):
        """Compute combined loss.

        Args:
            output_pred: Network output dictionary
            mnfld_points: Points on manifold (B, N, 3)
            nonmnfld_points: Points off manifold (B, N, 3)
            mnfld_n_gt: Ground truth normals at manifold points (optional)
            near_points: Points near manifold for morse loss (optional)

        Returns:
            loss_dict: Dictionary of loss values
            mnfld_grad: Gradients at manifold points
        """
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        # Get predictions
        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]

        # Initialize loss terms
        div_loss = torch.tensor([0.0], device=device)
        morse_loss = torch.tensor([0.0], device=device)
        curv_term = torch.tensor([0.0], device=device)
        normal_term = torch.tensor([0.0], device=device)
        min_surf_loss = torch.tensor([0.0], device=device)

        # Compute gradients for manifold points
        if manifold_pred is not None:
            mnfld_grad = gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        # Compute gradients for non-manifold points
        nonmnfld_grad = gradient(nonmnfld_points, non_manifold_pred)

        # Determine points for morse loss
        morse_nonmnfld_points = None
        morse_nonmnfld_grad = None

        if self.use_morse and near_points is not None:
            morse_nonmnfld_points = near_points
            morse_nonmnfld_grad = gradient(near_points, output_pred['near_points_pred'])
        elif self.use_morse and near_points is None:
            morse_nonmnfld_points = nonmnfld_points
            morse_nonmnfld_grad = nonmnfld_grad

        # Compute Morse (Gaussian curvature) loss
        if self.use_morse:
            # Compute Hessian for non-manifold/near points
            nonmnfld_dx = gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 0])
            nonmnfld_dy = gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 1])

            # Compute Hessian for manifold points
            mnfld_dx = gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = gradient(mnfld_points, mnfld_grad[:, :, 1])

            if dims == 3:
                nonmnfld_dz = gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 2])
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

                mnfld_dz = gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy), dim=-1)
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            morse_mnfld = torch.tensor([0.0], device=device)
            if self.div_type == 'l1':
                morse_loss = gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad)

                if self.bidirectional_morse:
                    morse_mnfld = gaussian_curvature(mnfld_hessian_term, mnfld_grad)

                morse_loss = 0.5 * (morse_loss + morse_mnfld)

        # Latent regularization for multi-shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)

        # SDF term: points on manifold should have SDF = 0
        sdf_term = torch.abs(manifold_pred).mean()

        # Eikonal term: ||grad|| = 1
        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # Inter term: exponential penalty for off-manifold points near zero
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # Compute total loss based on loss type
        if self.loss_type == 'siren_wo_n_w_morse':
            self.weights[2] = 0  # No normal supervision
            loss = (self.weights[0] * sdf_term +
                    self.weights[1] * inter_term +
                    self.weights[3] * eikonal_term +
                    self.weights[5] * morse_loss)
        else:
            raise ValueError(f"Unrecognized loss type: {self.loss_type}")

        return {
            "loss": loss,
            'sdf_term': sdf_term,
            'inter_term': inter_term,
            'latent_reg_term': latent_reg_term,
            'eikonal_term': eikonal_term,
            'normals_loss': normal_term,
            'div_loss': div_loss,
            'curv_loss': curv_term.mean() if torch.is_tensor(curv_term) else curv_term,
            'morse_term': morse_loss,
            'min_surf_loss': min_surf_loss
        }, mnfld_grad

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        """Update morse loss weight during training.

        Supports various annealing schedules for the morse weight.

        Args:
            current_iteration: Current training iteration
            n_iterations: Total number of iterations
            params: Decay parameters (start_weight, *optional middle, end_weight)
        """
        if params is None:
            return

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip(
                [params[0], *params[1:-1][1::2], params[-1]],
                [0, *params[1:-1][::2], 1]
            ))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Annealing functions
        if self.div_decay == 'linear':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass


# Loss weight presets for different reconstruction needs
LOSS_PRESETS = {
    'balanced': [7e3, 6e2, 1e2, 5e1, 0, 10],
    'sharp_edges': [7e3, 6e2, 1e2, 5e1, 0, 50],  # Higher morse weight
    'smooth_surfaces': [7e3, 6e2, 1e2, 5e1, 0, 2],  # Lower morse weight
}


def get_loss_weights(preset_name):
    """Get loss weights for a given preset.

    Args:
        preset_name: One of 'balanced', 'sharp_edges', 'smooth_surfaces'

    Returns:
        List of loss weights [sdf, inter, normal, eikonal, div, morse]
    """
    if preset_name not in LOSS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(LOSS_PRESETS.keys())}")
    return LOSS_PRESETS[preset_name].copy()
