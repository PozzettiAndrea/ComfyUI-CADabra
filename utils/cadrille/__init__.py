"""
Cadrille Utilities for ComfyUI-CADabra

Multi-modal CAD reconstruction from point clouds, images, or text.
Paper: "Cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning"
"""

from .model import Cadrille, FourierPointEncoder, FourierEmbedder
from .inference import (
    generate_code,
    prepare_point_cloud_input,
    prepare_image_input,
    prepare_text_input,
    sample_points_fps,
    normalize_point_cloud,
    render_mesh_views,
)

__all__ = [
    'Cadrille',
    'FourierPointEncoder',
    'FourierEmbedder',
    'generate_code',
    'prepare_point_cloud_input',
    'prepare_image_input',
    'prepare_text_input',
    'sample_points_fps',
    'normalize_point_cloud',
    'render_mesh_views',
]
