# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
Common utilities shared across CAD nodes.
"""

import os
import sys

# Check for OCC availability
try:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SHELL, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import TopoDS_Compound, topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_FixSmallFace, ShapeFix_Wireframe
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

# Check for scipy (used for KDTree in gap estimation)
try:
    from scipy.spatial import KDTree
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _progress_bar(completed, total, elapsed, width=30, prefix=""):
    """Print a tqdm-style progress bar."""
    if total == 0:
        return
    pct = completed / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    sys.stdout.write(f"\r{prefix}|{bar}| {completed}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def get_occ_shape(cad_model):
    """Get OCC shape from CAD_MODEL dict."""
    shape = cad_model.get("occ_shape")
    if shape is None:
        raise RuntimeError("CAD model has no OCC shape")
    return shape


def make_cad_model(occ_shape, original_cad_model=None):
    """Create new CAD_MODEL dict with OCC shape."""
    result = {
        "occ_shape": occ_shape,
        "format": "occ",
    }
    # Preserve original file path and metadata if available
    if original_cad_model:
        if "file_path" in original_cad_model:
            result["file_path"] = original_cad_model["file_path"]
        if "metadata" in original_cad_model:
            result["metadata"] = original_cad_model["metadata"]
    return result


def count_shape_entities(shape, entity_type):
    """Count entities of a given type in a shape."""
    if not HAS_OCC:
        return 0
    count = 0
    explorer = TopExp_Explorer(shape, entity_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count
