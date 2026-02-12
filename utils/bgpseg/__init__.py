"""
BGPSeg - Boundary-Guided Primitive Instance Segmentation

Standalone implementation for ComfyUI-CADabra.
Based on: "BGPSeg: Boundary-Guided Primitive Instance Segmentation of Point Clouds"
Paper: IEEE Transactions on Image Processing (2025)
Authors: Fang et al.

This module provides:
- BoundaryNet: Predicts boundary points in point clouds
- BGPSeg (BGFE): Boundary-guided feature extraction for primitive segmentation
- MeanShift_GPU: GPU-accelerated mean-shift clustering
- Inference utilities for two-stage segmentation pipeline
"""

# Lazy imports to avoid loading heavy dependencies at startup
_cuda_ops_loaded = False
_models_loaded = False


def load_cuda_ops():
    """
    Lazily load and compile CUDA operations.
    Call this before using models that require pointops/boundaryops.
    """
    global _cuda_ops_loaded
    if _cuda_ops_loaded:
        return True

    try:
        from .cuda_ops import build_ops
        success = build_ops()
        _cuda_ops_loaded = success
        return success
    except Exception as e:
        print(f"[BGPSeg] Failed to load CUDA ops: {e}")
        return False


def get_models_dir():
    """Get the directory where BGPSeg model weights are stored."""
    from pathlib import Path
    current_dir = Path(__file__).parent.parent.parent  # ComfyUI-CADabra/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "bgpseg"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


__all__ = [
    'load_cuda_ops',
    'get_models_dir',
]
