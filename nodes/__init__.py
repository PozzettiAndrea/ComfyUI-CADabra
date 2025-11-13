"""
CADabra Nodes Module
Exports all node class mappings from different node modules
"""

# Import existing CAD nodes (Gmsh-based)
from .cad_nodes import (
    NODE_CLASS_MAPPINGS as CAD_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CAD_NODE_DISPLAY_NAME_MAPPINGS
)

# Import new mesh reconstruction nodes
from .mesh_recon_nodes import (
    NODE_CLASS_MAPPINGS as MESH_RECON_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MESH_RECON_NODE_DISPLAY_NAME_MAPPINGS
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **CAD_NODE_CLASS_MAPPINGS,
    **MESH_RECON_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CAD_NODE_DISPLAY_NAME_MAPPINGS,
    **MESH_RECON_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
