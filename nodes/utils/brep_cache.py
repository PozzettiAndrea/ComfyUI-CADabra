"""
BREP file cache utilities for CAD_MODEL.

CAD_MODEL stores shapes as file paths to BREP files rather than OCC objects.
This avoids pickle issues when crossing process boundaries (isolated workers).
"""

import os
import hashlib
import tempfile

# Global cache directory - set during initialization
_CACHE_DIR = None


def get_cache_dir():
    """Get the BREP cache directory, creating if needed."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # Default to temp directory if not set
        _CACHE_DIR = os.path.join(tempfile.gettempdir(), "cadabra_brep_cache")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def set_cache_dir(path):
    """Set the BREP cache directory."""
    global _CACHE_DIR
    _CACHE_DIR = path
    os.makedirs(_CACHE_DIR, exist_ok=True)


def save_shape(shape, name_hint="shape"):
    """
    Save OCC shape to BREP file, return path.

    Args:
        shape: OCC TopoDS_Shape
        name_hint: Optional name for the file

    Returns:
        Path to the saved BREP file
    """
    from OCC.Core.BRepTools import breptools

    # Generate unique filename
    import uuid
    uid = uuid.uuid4().hex[:12]
    filename = f"{name_hint}_{uid}.brep"
    filepath = os.path.join(get_cache_dir(), filename)

    # Save shape
    success = breptools.Write(shape, filepath)
    if not success:
        raise RuntimeError(f"Failed to write BREP file: {filepath}")

    return filepath


def load_shape(brep_path):
    """
    Load OCC shape from BREP file.

    Args:
        brep_path: Path to BREP file

    Returns:
        OCC TopoDS_Shape
    """
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Shape

    if not os.path.exists(brep_path):
        raise FileNotFoundError(f"BREP file not found: {brep_path}")

    builder = BRep_Builder()
    shape = TopoDS_Shape()
    success = breptools.Read(shape, brep_path, builder)

    if not success:
        raise RuntimeError(f"Failed to read BREP file: {brep_path}")

    return shape


def get_occ_shape(cad_model):
    """
    Get OCC shape from CAD_MODEL dict.

    CAD_MODEL should contain 'brep_path' key pointing to a BREP file.

    Args:
        cad_model: dict with 'brep_path' key

    Returns:
        OCC TopoDS_Shape
    """
    brep_path = cad_model.get("brep_path")
    if brep_path is None:
        raise RuntimeError("CAD model has no brep_path")
    return load_shape(brep_path)


def make_cad_model(shape, original_cad_model=None, name_hint="shape"):
    """
    Create CAD_MODEL dict from OCC shape.

    Saves the shape to a BREP file and returns a dict with the path.

    Args:
        shape: OCC TopoDS_Shape
        original_cad_model: Optional original CAD_MODEL to preserve metadata from
        name_hint: Name hint for the BREP file

    Returns:
        dict with 'brep_path' and other metadata
    """
    brep_path = save_shape(shape, name_hint)

    result = {
        "brep_path": brep_path,
        "format": "brep",
    }

    # Preserve metadata from original
    if original_cad_model:
        if "file_path" in original_cad_model:
            result["file_path"] = original_cad_model["file_path"]
        if "metadata" in original_cad_model:
            result["metadata"] = original_cad_model["metadata"]

    return result
