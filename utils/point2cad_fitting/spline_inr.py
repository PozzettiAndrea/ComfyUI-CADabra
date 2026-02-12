"""
SplineINR - Neural Implicit Representation for freeform surfaces.
Trains on-the-fly for each surface segment (1000 steps, ~1-2 sec).
Adapted from Point2CAD: https://github.com/prs-eth/point2cad
"""

import numpy as np
import torch
import trimesh
import warnings
from tqdm import tqdm

from .layers import PositionalEncoding, ResBlock, SirenLayer, SirenWithResblock
from .utils import get_rng


class Mapping(torch.nn.Module):
    """Base mapping network using SIREN or ResBlock architecture."""

    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=32,
        num_hidden_layers=0,
        block_type="residual",
        resblock_posenc_numfreqs=0,
        resblock_zeroinit_posenc=True,
        resblock_act_type="silu",
        resblock_batchnorms=True,
        resblock_shortcut=False,
        resblock_channels_fraction=0.5,
        sirenblock_omega_first=10,
        sirenblock_omega_other=10,
        sirenblock_act_type="sinc",
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype

        if block_type == "residual":
            posenc = PositionalEncoding(resblock_posenc_numfreqs, True, dtype)
            dim_in_real = dim_in * posenc.dim_multiplier
            layers = [
                posenc,
                ResBlock(dim_in_real, dim_hidden, batchnorms=resblock_batchnorms, act_type=resblock_act_type, shortcut=False),
            ]
            layers += [ResBlock(dim_hidden, dim_hidden, batchnorms=resblock_batchnorms, act_type=resblock_act_type, shortcut=resblock_shortcut)] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]
            if resblock_zeroinit_posenc:
                with torch.no_grad():
                    layers[1].linear.weight *= 0.01

        elif block_type == "siren":
            layers = [SirenLayer(dim_in, dim_hidden, is_first=True, omega=sirenblock_omega_first, act_type=sirenblock_act_type)]
            layers += [SirenLayer(dim_hidden, dim_hidden, is_first=False, omega=sirenblock_omega_other, act_type=sirenblock_act_type)] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]

        elif block_type == "combined":
            layers = [SirenWithResblock(dim_in, dim_hidden, sirenblock_is_first=True, sirenblock_omega=sirenblock_omega_first, sirenblock_act_type=sirenblock_act_type, resblock_batchnorms=resblock_batchnorms, resblock_act_type=resblock_act_type, resblock_shortcut=resblock_shortcut, resblock_channels_fraction=resblock_channels_fraction)]
            layers += [SirenWithResblock(dim_hidden, dim_hidden, sirenblock_is_first=False, sirenblock_omega=sirenblock_omega_other, sirenblock_act_type=sirenblock_act_type, resblock_batchnorms=resblock_batchnorms, resblock_act_type=resblock_act_type, resblock_shortcut=resblock_shortcut, resblock_channels_fraction=resblock_channels_fraction)] * num_hidden_layers
            layers += [torch.nn.Linear(dim_hidden, dim_out)]

        else:
            raise ValueError(f"Unknown block_type={block_type}")

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def convert_encoder_output_to_uv(x, is_u_closed, is_v_closed):
    """Convert encoder output to UV coordinates."""
    xu, xv = x.chunk(2, dim=1)
    if is_u_closed:
        xu = torch.atan2(xu[:, [0]], xu[:, [1]]) / np.pi
    else:
        xu = torch.tanh(xu[:, [0]])
    if is_v_closed:
        xv = torch.atan2(xv[:, [0]], xv[:, [1]]) / np.pi
    else:
        xv = torch.tanh(xv[:, [0]])
    return torch.cat((xu, xv), dim=1)


def convert_uv_to_decoder_input(x, is_u_closed, is_v_closed, open_replicate=True):
    """Convert UV coordinates to decoder input."""
    if is_u_closed:
        xu_closed_rad = x[:, [0]] * np.pi
        xu_0, xu_1 = xu_closed_rad.cos(), xu_closed_rad.sin()
    else:
        xu_open = x[:, [0]]
        xu_0 = xu_open
        xu_1 = xu_open if open_replicate else torch.zeros_like(xu_open)
    if is_v_closed:
        xv_closed_rad = x[:, [1]] * np.pi
        xv_0, xv_1 = xv_closed_rad.cos(), xv_closed_rad.sin()
    else:
        xv_open = x[:, [1]]
        xv_0 = xv_open
        xv_1 = xv_open if open_replicate else torch.zeros_like(xv_open)
    return torch.cat((xu_0, xu_1, xv_0, xv_1), dim=1)


class Map3DtoUV(Mapping):
    """Encoder: 3D points -> UV coordinates."""
    def __init__(self, is_u_closed, is_v_closed, **kwargs):
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        super().__init__(3, 4, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return convert_encoder_output_to_uv(x, self.is_u_closed, self.is_v_closed)


class MapUVto3D(Mapping):
    """Decoder: UV coordinates -> 3D points."""
    def __init__(self, is_u_closed, is_v_closed, **kwargs):
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        super().__init__(4, 3, **kwargs)

    def forward(self, x):
        if not torch.is_tensor(x) or x.dim() not in (1, 2) or x.dtype != self.dtype:
            raise ValueError("Invalid input")
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_batch_dim_unsqueezed = True
        else:
            is_batch_dim_unsqueezed = False
        x = convert_uv_to_decoder_input(x, self.is_u_closed, self.is_v_closed)
        x = super().forward(x)
        if is_batch_dim_unsqueezed:
            x = x.squeeze(0)
        return x


class SplineINR(torch.nn.Module):
    """
    Neural Implicit Representation for freeform surfaces.
    Encoder maps 3D -> UV, Decoder maps UV -> 3D.
    """
    def __init__(self, is_u_closed=False, is_v_closed=False, **kwargs):
        super().__init__()
        self.encoder = Map3DtoUV(is_u_closed, is_v_closed, **kwargs)
        self.decoder = MapUVto3D(is_u_closed, is_v_closed, **kwargs)

    def forward(self, x):
        uv = self.encoder(x)
        x_hat = self.decoder(uv)
        return x_hat, uv


def fit_one_inr_spline(
    points,
    is_u_closed=False,
    is_v_closed=False,
    num_fit_steps=1000,
    lr=1e-1,
    device="cuda",
    seed=None,
    progress_bar=False,
):
    """
    Fit SplineINR to a point cloud segment.

    Args:
        points: Tensor of shape (N, 3)
        is_u_closed: Whether surface is closed in U direction
        is_v_closed: Whether surface is closed in V direction
        num_fit_steps: Training iterations (default 1000)
        lr: Learning rate
        device: 'cuda' or 'cpu'
        seed: Random seed
        progress_bar: Show progress

    Returns:
        Dict with model, error, and UV bounding box
    """
    if not torch.is_tensor(points):
        raise ValueError("Input must be a torch tensor")
    if points.dim() != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        raise ValueError("Points must be (N, 3) with N >= 3")

    dtype = torch.float32
    if device != "cpu" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, fitting on CPU")
        device = "cpu"

    # ComfyUI runs with inference_mode(True) globally - we need to escape it for training
    # Wrap EVERYTHING in inference_mode(False) to ensure all tensors are normal tensors
    with torch.inference_mode(False):
        # Clone the input to escape inference mode
        points = points.clone()

        # Create model with Point2CAD defaults
        model = SplineINR(
            is_u_closed=is_u_closed,
            is_v_closed=is_v_closed,
            dim_hidden=64,
            num_hidden_layers=0,
            block_type="combined",
            resblock_posenc_numfreqs=0,
            resblock_zeroinit_posenc=True,
            resblock_act_type="silu",
            resblock_batchnorms=False,
            resblock_shortcut=False,
            resblock_channels_fraction=0.5,
            sirenblock_omega_first=10,
            sirenblock_omega_other=10,
            sirenblock_act_type="sinc",
            dtype=dtype,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.L1Loss()

        # Per-segment normalization (like original Point2CAD)
        # This ensures the INR trains on data centered at 0 with unit scale
        points_raw = points.to(device=device, dtype=dtype)
        points_mean = points_raw.mean(dim=0)
        points_std = points_raw.std(dim=0)
        points_scale = points_std.max()  # Use max std for uniform scaling
        points_norm = (points_raw - points_mean) / points_scale

        # Split train/val
        num_points = points_norm.shape[0]
        num_points_val = max(1, num_points // 10)
        rng = get_rng(device, seed=seed)
        permutation = torch.randperm(num_points, device=device, generator=rng)
        points_norm = points_norm[permutation]
        points_train = points_norm[:-num_points_val]
        points_val = points_norm[-num_points_val:]

        model = model.to(device)
        model.train()

        pbar = tqdm(range(num_fit_steps), disable=not progress_bar, desc="SplineINR")
        with torch.enable_grad():
            for step in pbar:
                # Learning rate decay
                new_lr = lr * (0.001 ** (step / num_fit_steps))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                optimizer.zero_grad()
                x_hat, _ = model(points_train)
                loss = loss_fn(x_hat, points_train)
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.5f}")

        # Evaluate (still inside inference_mode(False) to access tensors)
        model.eval()
        with torch.no_grad():
            uv = model.encoder(points_norm)
            x_hat, _ = model(points_val)
            val_err = torch.nn.functional.mse_loss(x_hat, points_val).item()
            err = torch.sqrt(((points_norm - model(points_norm)[0]) ** 2).sum(-1)).mean().item()

        uv_bb_min = uv.min(dim=0).values.cpu().detach()
        uv_bb_max = uv.max(dim=0).values.cpu().detach()
        model = model.cpu()
        # Store normalization params for mesh denormalization
        points_mean = points_mean.cpu().detach()
        points_scale = points_scale.cpu().detach().item()

    return {
        "is_u_closed": is_u_closed,
        "is_v_closed": is_v_closed,
        "points3d_offset": points_mean,
        "points3d_scale": points_scale,
        "val_err_l2": val_err,
        "is_good_fit": val_err < 1e-4,
        "uv_bb_min": uv_bb_min,
        "uv_bb_max": uv_bb_max,
        "model": model,
        "err": err,
    }


def sample_inr_mesh(fit_out, mesh_dim=100, uv_margin=0.2):
    """
    Sample a mesh from fitted SplineINR.

    Args:
        fit_out: Output dict from fit_one_inr_spline
        mesh_dim: Grid resolution for mesh
        uv_margin: Margin around UV bounding box

    Returns:
        trimesh.Trimesh object
    """
    uv_bb_sz = fit_out["uv_bb_max"] - fit_out["uv_bb_min"]
    uv_bb_margin = uv_bb_sz * uv_margin
    uv_min = fit_out["uv_bb_min"] - uv_bb_margin
    uv_max = fit_out["uv_bb_max"] + uv_bb_margin

    if fit_out["is_u_closed"]:
        uv_min[0] = max(uv_min[0], -1)
        uv_max[0] = min(uv_max[0], 1)
    if fit_out["is_v_closed"]:
        uv_min[1] = max(uv_min[1], -1)
        uv_max[1] = min(uv_max[1], 1)

    model = fit_out["model"]
    device = next(model.parameters()).device

    u, v = torch.meshgrid(
        torch.linspace(uv_min[0].item(), uv_max[0].item(), mesh_dim, device=device),
        torch.linspace(uv_min[1].item(), uv_max[1].item(), mesh_dim, device=device),
        indexing="xy",
    )
    uv = torch.stack((u, v), dim=2)

    model.eval()
    with torch.no_grad():
        points = model.decoder(uv.reshape(-1, 2))

    # Denormalize
    points3d_scale = fit_out["points3d_scale"]
    points3d_offset = fit_out["points3d_offset"]
    if not torch.is_tensor(points3d_scale):
        points3d_scale = torch.tensor(points3d_scale, device=device)
    if not torch.is_tensor(points3d_offset):
        points3d_offset = torch.tensor(points3d_offset, device=device)
    points = points * points3d_scale + points3d_offset

    # Build faces
    faces = []
    for i in range(mesh_dim - 1):
        for j in range(mesh_dim - 1):
            faces.append([i * mesh_dim + j, (i + 1) * mesh_dim + j, i * mesh_dim + j + 1])
            faces.append([i * mesh_dim + j + 1, (i + 1) * mesh_dim + j, (i + 1) * mesh_dim + j + 1])

    return trimesh.Trimesh(points.cpu().numpy(), np.array(faces))
