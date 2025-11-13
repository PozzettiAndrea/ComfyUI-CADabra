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
        # Initialize gmsh once in main thread to avoid signal handler issues
        # in ComfyUI's worker threads
        if not gmsh.is_initialized():
            try:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output
                print("[CADabra] ✓ Gmsh initialized (main thread)")
            except ValueError as e:
                # Signal handler setup failed - continue anyway if that's the only issue
                if "signal only works in main thread" in str(e):
                    print("[CADabra] ⚠ Gmsh initialized without signal handlers")
                else:
                    raise
    except ImportError:
        print("[CADabra] WARNING: Gmsh not found")
        print("[CADabra] Install with: pip install gmsh")
    except OSError as e:
        print("[CADabra] WARNING: Gmsh installation incomplete")
        print(f"[CADabra] Missing system library: {e}")
        print("[CADabra] Install system dependencies: apt-get install libglu1-mesa libxft2")

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
