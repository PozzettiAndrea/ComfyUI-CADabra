"""
SECAD-Net neural network architecture.

Paper: SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .sdfs import sdfExtrusion, transform_points, add_latent


class Encoder(nn.Module):
    """
    Encoder network that converts 64³ voxel grids to 256D latent codes.

    Architecture: 5 Conv3D layers with stride=2, LeakyReLU activation.
    """

    def __init__(self, ef_dim=32):
        super(Encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 8, 4, stride=1, padding=0, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, inputs):
        """
        Args:
            inputs: Voxel grid tensor [B, 1, 64, 64, 64]
        Returns:
            Latent code tensor [B, 256]
        """
        d = inputs
        for i in range(1, 6):
            d = getattr(self, f"conv_{i}")(d)
            d = F.leaky_relu(d, negative_slope=0.01)
        d = d.view(-1, self.ef_dim * 8)
        d = F.leaky_relu(d, negative_slope=0.01)
        return d


class Decoder(nn.Module):
    """
    Decoder network that converts latent codes to primitive parameters.

    Each primitive has 8 parameters:
    - 4 values: Quaternion (rotation)
    - 3 values: Translation (XYZ position)
    - 1 value: Extrusion height
    """

    def __init__(self, ef_dim=32, num_primitives=4):
        super(Decoder, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = ef_dim

        self.num_primitive_parameters_aggregated = 4 + 3 + 1  # quat + trans + height
        self.primitive_linear = nn.Linear(
            self.feature_dim * 8,
            int(self.num_primitives * self.num_primitive_parameters_aggregated),
            bias=True,
        )
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

    def forward(self, feature):
        """
        Args:
            feature: Latent code tensor [B, 256]
        Returns:
            Primitive parameters tensor [B, 8, num_primitives]
        """
        shapes = self.primitive_linear(feature)
        para_3d = shapes[..., : self.num_primitives * (4 + 3 + 1)].view(
            -1, (4 + 3 + 1), int(self.num_primitives)
        )  # B,C,P
        return para_3d


class SketchHead(nn.Module):
    """
    Neural network head that computes 2D sketch occupancy from point coordinates.

    Architecture: 3-layer MLP with Softplus activation.
    """

    def __init__(self, d_in, dims):
        super().__init__()
        dims = [d_in] + dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(
                    lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001
                )
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
            else:
                x = x.clamp(-1, 1)
        return x


class Generator(nn.Module):
    """
    Generator network that computes occupancy fields from primitive parameters.

    Uses neural sketch heads to compute 2D profiles, then extrudes to 3D.
    """

    def __init__(self, num_primitives=4, sharpness=150, test=False):
        super(Generator, self).__init__()
        self.num_primitives = num_primitives
        self.sharpness = sharpness
        self.test = test

        D_IN = 2
        LATENT_SIZE = 256

        for i in range(num_primitives):
            setattr(
                self,
                'sketch_head_' + str(i),
                SketchHead(d_in=D_IN + LATENT_SIZE, dims=[512, 512, 512]),
            )

    def forward(self, sample_point_coordinates, primitive_parameters, code):
        """
        Args:
            sample_point_coordinates: Query points [B, N, 3]
            primitive_parameters: Parameters from decoder [B, 8, num_primitives]
            code: Latent code [B, 256]
        Returns:
            union_occupancies: Combined occupancy [B, N]
            total_2d_occ: 2D sketch occupancies [B, N, num_primitives]
            transformed_points: Points in primitive-local coordinates
        """
        B, N = sample_point_coordinates.shape[:2]
        primitive_parameters = primitive_parameters.transpose(2, 1)
        B, K, param_dim = primitive_parameters.shape

        boxes = primitive_parameters[..., :8]

        transformed_points = transform_points(
            boxes[..., :4], boxes[..., 4:7], sample_point_coordinates
        )  # [B, N, K, 3]

        latent_points = []
        for i in range(self.num_primitives):
            latent_point = add_latent(transformed_points[..., i, :2], code).float()
            latent_points.append(latent_point)

        sdfs_2d = [
            getattr(self, f'sketch_head_{i}')(latent_points[i]).reshape(B, N, -1).float()
            for i in range(self.num_primitives)
        ]
        sdfs_2d = torch.cat(sdfs_2d, dim=-1)

        total_2d_occ = torch.sigmoid(-1 * sdfs_2d * self.sharpness)

        box_ext = sdfExtrusion(sdfs_2d, boxes[..., 7], transformed_points).squeeze(-1)
        primitive_sdf = box_ext
        primitive_occupancies = torch.sigmoid(-1 * primitive_sdf * self.sharpness)

        with torch.no_grad():
            weights = torch.softmax(primitive_occupancies * 20, dim=-1)

        union_occupancies = torch.sum(weights * primitive_occupancies, dim=-1)

        return union_occupancies, total_2d_occ, transformed_points
