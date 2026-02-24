"""
NeurCADRecon Network Module

SIREN-based neural network for signed distance function (SDF) representation.
Adapted from NeurCADRecon (SIGGRAPH 2024) paper by Dong et al.

Paper: NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces
"""

import numpy as np
import torch
import torch.nn as nn


class Sine(nn.Module):
    """SIREN sine activation with frequency scaling (omega_0 = 30)."""

    def forward(self, input):
        # See SIREN paper sec. 3.2 for discussion of factor 30
        return torch.sin(30 * input)


class AbsLayer(nn.Module):
    """Absolute value layer for UDF output."""

    def __init__(self):
        super(AbsLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)


def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Decoder(nn.Module):
    """SDF/UDF decoder wrapper with optional absolute value output."""

    def __init__(self, udf=False):
        super(Decoder, self).__init__()
        self.nl = nn.Identity() if not udf else AbsLayer()

    def forward(self, *args, **kwargs):
        res = self.fc_block(*args, **kwargs)
        res = self.nl(res)
        return res


class Modulator(nn.Module):
    """Modulator network for latent code conditioning (optional VAE mode)."""

    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)


class FCBlock(nn.Module):
    """Fully connected neural network block with SIREN initialization.

    Supports modulation for hypernetwork-style conditioning.
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 sphere_init_params=[1.6, 1.0]):
        super().__init__()

        self.first_layer_init = None
        self.sphere_init_params = sphere_init_params
        self.init_type = init_type

        nl_dict = {
            'sine': Sine(),
            'relu': nn.ReLU(inplace=True),
            'softplus': nn.Softplus(beta=100),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        nl = nl_dict[nonlinearity]

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)

        # Apply initialization based on init_type
        if init_type == 'siren':
            self.net.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)
        elif init_type == 'geometric_sine':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_geom_sine_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)
        elif init_type == 'mfgi':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_mfgi_init)
            self.net[1].apply(second_layer_mfgi_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)
        elif init_type == 'geometric_relu':
            self.net.apply(geom_relu_init)
            self.net[-1].apply(geom_relu_last_layers_init)

    def forward(self, coords, mods=None):
        mods = cast_tuple(mods, len(self.net))
        x = coords

        for layer, mod in zip(self.net, mods):
            x = layer(x)
            if exists(mod):
                if mod.shape[1] != 1:
                    mod = mod[:, None, :]
                x = x * mod
        if mods[0] is not None:
            x = self.net[-1](x)  # last layer

        return x


class Network(nn.Module):
    """Main NeurCADRecon network for SDF learning.

    Args:
        latent_size: Size of latent code (0 for no latent conditioning)
        in_dim: Input dimension (3 for 3D coordinates)
        decoder_hidden_dim: Hidden layer width
        nl: Nonlinearity type ('sine', 'relu', etc.)
        decoder_n_hidden_layers: Number of hidden layers
        init_type: Initialization type ('siren', 'geometric_sine', 'mfgi')
        sphere_init_params: Parameters for sphere initialization [radius, scaling]
        udf: Whether to output unsigned distance (UDF) instead of SDF
    """

    def __init__(self, latent_size=0, in_dim=3, decoder_hidden_dim=256, nl='sine',
                 decoder_n_hidden_layers=4, init_type='siren',
                 sphere_init_params=[1.6, 1.0], udf=False):
        super().__init__()
        self.latent_size = latent_size
        self.init_type = init_type

        # Modulator for latent conditioning (optional)
        self.modulator = Modulator(
            dim_in=latent_size,
            dim_hidden=decoder_hidden_dim,
            num_layers=decoder_n_hidden_layers + 1
        ) if latent_size > 0 else None

        # Main decoder
        self.decoder = Decoder(udf=udf)
        self.decoder.fc_block = FCBlock(
            in_dim, 1,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nl,
            init_type=init_type,
            sphere_init_params=sphere_init_params
        )

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, near_points=None,
                only_nonmnfld=False, latent=None):
        """Forward pass for training.

        Args:
            non_mnfld_pnts: Points not on manifold (for eikonal loss)
            mnfld_pnts: Points on manifold (surface points)
            near_points: Points near surface (for morse loss)
            only_nonmnfld: Only compute non-manifold predictions
            latent: Optional latent code for conditioning

        Returns:
            Dictionary with SDF predictions for each point type
        """
        # Handle latent modulation
        if self.latent_size > 0 and latent is not None:
            mods = self.modulator(latent)
            latent_reg = 1e-3 * latent.norm(-1).mean()
        else:
            latent_reg = None
            mods = None

        # Compute predictions
        if mnfld_pnts is not None and not only_nonmnfld:
            manifold_pnts_pred = self.decoder(mnfld_pnts, mods)
        else:
            manifold_pnts_pred = None

        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts, mods)

        near_points_pred = None
        if near_points is not None:
            near_points_pred = self.decoder(near_points, mods)

        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
            'near_points_pred': near_points_pred,
            "latent_reg": latent_reg,
        }


# ===================== SIREN Initialization Functions =====================

def sine_init(m):
    """SIREN weight initialization for hidden layers."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    """SIREN weight initialization for first layer."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


# ===================== Geometric Sine Initialization =====================

def geom_sine_init(m):
    """Geometric initialization for sine networks."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def first_layer_geom_sine_init(m):
    """Geometric initialization for first layer of sine networks."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    """Geometric initialization for second-to-last layer."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            assert m.weight.shape == (num_output, num_output)
            m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.weight.data /= 30
            m.bias.data /= 30


def last_layer_geom_sine_init(m):
    """Geometric initialization for last layer."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (1, num_input)
            assert m.bias.shape == (1,)
            m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.bias.data = torch.zeros(1) + num_input


# ===================== Multi-Frequency Geometric Initialization =====================

# Number of periods of sine for each section of the output vector
_periods = [1, 30]
_portion_per_period = np.array([0.25, 0.75])


def first_layer_mfgi_init(m):
    """Multi-frequency geometric initialization for first layer."""
    global _periods, _portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            num_output = m.weight.size(0)
            num_per_period = (_portion_per_period * num_output).astype(int)
            assert len(_periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(_periods)):
                period = _periods[i]
                num = num_per_period[i]
                scale = 30 / period
                weights.append(torch.zeros(num, num_input).uniform_(
                    -np.sqrt(3 / num_input) / scale,
                    np.sqrt(3 / num_input) / scale
                ))
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new


def second_layer_mfgi_init(m):
    """Multi-frequency geometric initialization for second layer."""
    global _portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (num_input, num_input)
            num_per_period = (_portion_per_period * num_input).astype(int)
            k = num_per_period[0]
            W1_new = torch.zeros(num_input, num_input).uniform_(
                -np.sqrt(3 / num_input),
                np.sqrt(3 / num_input) / 30
            ) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new


# ===================== Geometric ReLU Initialization =====================

def geom_relu_init(m):
    """Geometric initialization for ReLU networks (IGR style)."""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            out_dims = m.out_features
            m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
            m.bias.data = torch.zeros_like(m.bias.data)


def geom_relu_last_layers_init(m):
    """Geometric initialization for last layer of ReLU networks."""
    radius_init = 1
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
            m.bias.data = torch.Tensor([-radius_init])
