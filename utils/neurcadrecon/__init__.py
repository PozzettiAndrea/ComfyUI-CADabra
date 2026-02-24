"""
NeurCADRecon Integration for ComfyUI-CADabra

Paper: NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces
Authors: Qiujie Dong et al. (SIGGRAPH 2024)

SIREN-based SDF network with Morse loss for sharp edge reconstruction.
This module provides neural implicit surface reconstruction optimized for
CAD surfaces with developable (zero Gaussian curvature) properties.
"""

from .network import Network, FCBlock, Sine
from .losses import MorseLoss, eikonal_loss, gaussian_curvature, get_loss_weights, LOSS_PRESETS
from .utils import gradient, implicit2mesh, scale_pc_to_unit_sphere, count_parameters
from .dataset import ReconDataset, create_dataset_from_points
from .training import (
    TrainingConfig,
    train_neurcadrecon,
    extract_mesh,
    save_checkpoint,
    load_checkpoint,
    get_default_checkpoint_dir,
)

__all__ = [
    # Network
    'Network',
    'FCBlock',
    'Sine',
    # Losses
    'MorseLoss',
    'eikonal_loss',
    'gaussian_curvature',
    'get_loss_weights',
    'LOSS_PRESETS',
    # Utilities
    'gradient',
    'implicit2mesh',
    'scale_pc_to_unit_sphere',
    'count_parameters',
    # Dataset
    'ReconDataset',
    'create_dataset_from_points',
    # Training
    'TrainingConfig',
    'train_neurcadrecon',
    'extract_mesh',
    'save_checkpoint',
    'load_checkpoint',
    'get_default_checkpoint_dir',
]
