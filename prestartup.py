"""
Pre-startup script for ComfyUI-CADabra
"""

print("""
╔═══════════════════════════════════════╗
║     ComfyUI-CADabra Loaded           ║
║     CAD Processing & ML Nodes         ║
╚═══════════════════════════════════════╝
""")

try:
    import gmsh
    print("[CADabra] ✓ Gmsh initialized")
except ImportError:
    print("[CADabra] ✗ Gmsh not found - install with: pip install gmsh")
