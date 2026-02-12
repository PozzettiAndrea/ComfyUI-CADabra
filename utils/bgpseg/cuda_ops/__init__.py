"""
BGPSeg CUDA Operations

This module provides JIT-compiled CUDA extensions for:
- pointops: KNN query, furthest point sampling, grouping, interpolation
- boundaryops: Boundary-guided neighbor query

Extensions are compiled on first use and cached for subsequent runs.
"""

import os
from pathlib import Path

# Cache for loaded modules
_pointops_module = None
_boundaryops_module = None
_build_attempted = False


def get_cuda_ops_dir() -> Path:
    """Get the directory containing CUDA source files."""
    return Path(__file__).parent


def build_ops() -> bool:
    """
    Build CUDA extensions if not already built.
    Returns True if successful, False otherwise.
    """
    global _build_attempted
    if _build_attempted:
        return _pointops_module is not None

    _build_attempted = True

    try:
        import torch
        if not torch.cuda.is_available():
            print("[BGPSeg] CUDA not available, cannot build CUDA extensions")
            return False

        # Try to load pre-built extensions first
        try:
            get_pointops()
            get_boundaryops()
            print("[BGPSeg] CUDA extensions loaded successfully")
            return True
        except Exception as e:
            print(f"[BGPSeg] Building CUDA extensions (first run may take a few minutes)...")
            print(f"[BGPSeg] Build error details: {e}")
            return False

    except Exception as e:
        print(f"[BGPSeg] Failed to build CUDA extensions: {e}")
        return False


def get_pointops():
    """
    Get the pointops CUDA module.
    Builds it via JIT compilation if not already available.
    """
    global _pointops_module
    if _pointops_module is not None:
        return _pointops_module

    import torch
    from torch.utils.cpp_extension import load

    cuda_ops_dir = get_cuda_ops_dir()
    pointops_src = cuda_ops_dir / "pointops" / "src"

    sources = [
        str(pointops_src / "pointops_api.cpp"),
        str(pointops_src / "knnquery" / "knnquery_cuda.cpp"),
        str(pointops_src / "knnquery" / "knnquery_cuda_kernel.cu"),
        str(pointops_src / "sampling" / "sampling_cuda.cpp"),
        str(pointops_src / "sampling" / "sampling_cuda_kernel.cu"),
        str(pointops_src / "grouping" / "grouping_cuda.cpp"),
        str(pointops_src / "grouping" / "grouping_cuda_kernel.cu"),
        str(pointops_src / "interpolation" / "interpolation_cuda.cpp"),
        str(pointops_src / "interpolation" / "interpolation_cuda_kernel.cu"),
        str(pointops_src / "subtraction" / "subtraction_cuda.cpp"),
        str(pointops_src / "subtraction" / "subtraction_cuda_kernel.cu"),
        str(pointops_src / "aggregation" / "aggregation_cuda.cpp"),
        str(pointops_src / "aggregation" / "aggregation_cuda_kernel.cu"),
    ]

    # Build directory for compiled extensions
    build_dir = cuda_ops_dir / "build" / "pointops"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[BGPSeg] JIT compiling pointops_cuda...")
    _pointops_module = load(
        name="pointops_cuda",
        sources=sources,
        build_directory=str(build_dir),
        extra_cflags=["-g"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )

    # Create a wrapper module with the expected interface
    class PointOpsWrapper:
        def __init__(self, module):
            self._module = module

        def furthestsampling(self, xyz, offset, new_offset):
            """Furthest point sampling."""
            from .pointops_wrapper import furthestsampling
            return furthestsampling(xyz, offset, new_offset, self._module)

        def knnquery(self, nsample, xyz, new_xyz, offset, new_offset):
            """K-nearest neighbor query."""
            from .pointops_wrapper import knnquery
            return knnquery(nsample, xyz, new_xyz, offset, new_offset, self._module)

        def queryandgroup(self, nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
            """Query and group features."""
            from .pointops_wrapper import queryandgroup
            return queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz, self._module)

        def interpolation(self, xyz, new_xyz, feat, offset, new_offset, k=3):
            """Feature interpolation."""
            from .pointops_wrapper import interpolation
            return interpolation(xyz, new_xyz, feat, offset, new_offset, k, self._module)

    _pointops_module = PointOpsWrapper(_pointops_module)
    print("[BGPSeg] pointops_cuda compiled successfully")

    return _pointops_module


def get_boundaryops():
    """
    Get the boundaryops CUDA module.
    Builds it via JIT compilation if not already available.
    """
    global _boundaryops_module
    if _boundaryops_module is not None:
        return _boundaryops_module

    import torch
    from torch.utils.cpp_extension import load

    cuda_ops_dir = get_cuda_ops_dir()
    boundaryops_src = cuda_ops_dir / "boundaryops" / "src"

    sources = [
        str(boundaryops_src / "boundaryops_api.cpp"),
        str(boundaryops_src / "boundaryquery" / "boundaryquery_cuda.cpp"),
        str(boundaryops_src / "boundaryquery" / "boundaryquery_cuda_kernel.cu"),
    ]

    # Build directory for compiled extensions
    build_dir = cuda_ops_dir / "build" / "boundaryops"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[BGPSeg] JIT compiling boundaryops_cuda...")
    _boundaryops_module = load(
        name="boundaryops_cuda",
        sources=sources,
        build_directory=str(build_dir),
        extra_cflags=["-g"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )

    # Create a wrapper module with the expected interface
    class BoundaryOpsWrapper:
        def __init__(self, module):
            self._module = module

        def boundaryquery(self, nsample, xyz, new_xyz, offset, new_offset, edges, boundary):
            """Boundary-guided neighbor query."""
            from .boundaryops_wrapper import boundaryquery
            return boundaryquery(nsample, xyz, new_xyz, offset, new_offset, edges, boundary, self._module)

    _boundaryops_module = BoundaryOpsWrapper(_boundaryops_module)
    print("[BGPSeg] boundaryops_cuda compiled successfully")

    return _boundaryops_module
