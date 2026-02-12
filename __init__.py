"""
ComfyUI-CADabra - CAD Processing Custom Nodes

Provides CAD file loading, meshing, and ML-based surface reconstruction nodes.
"""

import sys

# Note: Windows DLL path fix for OpenCASCADE is done in prestartup_script.py
# which runs BEFORE PyTorch loads (critical for DLL loading order)

# Only run initialization when loaded by ComfyUI, not during pytest
# This prevents import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    # Import all nodes from the nodes module
    # This includes both CAD nodes and mesh reconstruction nodes
    # Note: gmsh is initialized lazily by individual nodes when needed
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # During testing, don't import nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Set web directory for JavaScript extensions (CAD viewer widgets)
# This tells ComfyUI where to find our JavaScript files and HTML viewers
# Files will be served at /extensions/ComfyUI-CADabra/*
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
