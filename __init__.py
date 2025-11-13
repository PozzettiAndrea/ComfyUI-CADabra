"""
ComfyUI-CADabra - CAD Processing Custom Nodes

Provides CAD file loading, meshing, and ML-based surface reconstruction nodes.
"""

import sys

# Only run initialization when loaded by ComfyUI, not during pytest
# This prevents import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    try:
        import gmsh
        print("[CADabra] Gmsh found - CAD nodes available")
    except ImportError:
        print("[CADabra] WARNING: Gmsh not found")
        print("[CADabra] Install with: pip install gmsh")
    except OSError as e:
        print("[CADabra] WARNING: Gmsh installation incomplete")
        print(f"[CADabra] Missing system library: {e}")
        print("[CADabra] Install system dependencies: apt-get install libglu1-mesa")

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # During testing, don't import nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
