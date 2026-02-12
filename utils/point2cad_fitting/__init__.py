"""
Point2CAD Surface Fitting Module
Adapted from https://github.com/prs-eth/point2cad

Provides:
- Fit: Analytical primitive fitting (plane, sphere, cylinder, cone)
- SplineINR: Neural implicit representation for freeform surfaces
- process_one_surface: Complete fitting pipeline per segment
- Surface sampling utilities for mesh generation
"""

from .primitives import Fit
from .spline_inr import SplineINR, fit_one_inr_spline, sample_inr_mesh
from .fitting import process_one_surface, fit_basic_primitives, fit_inrs
from .surface_sampling import (
    sample_plane_mesh,
    sample_sphere_mesh,
    sample_cylinder_mesh,
    sample_cone_mesh,
    sample_surface_mesh,
    sample_all_surfaces,
)
__all__ = [
    'Fit',
    'SplineINR',
    'fit_one_inr_spline',
    'sample_inr_mesh',
    'process_one_surface',
    'fit_basic_primitives',
    'fit_inrs',
    # Surface sampling
    'sample_plane_mesh',
    'sample_sphere_mesh',
    'sample_cylinder_mesh',
    'sample_cone_mesh',
    'sample_surface_mesh',
    'sample_all_surfaces',
]
