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
except OSError as e:
    # System library missing (e.g., libGLU.so.1, libXft.so.2)
    if "libGLU" in str(e) or "libXft" in str(e):
        print(f"[CADabra] ✗ Gmsh system dependency missing: {e}")
        print("[CADabra] Attempting to install system dependencies...")

        try:
            from .install import install_system_deps
            if install_system_deps():
                print("[CADabra] Please restart ComfyUI for changes to take effect")
            else:
                print("[CADabra] Auto-installation failed")
                print("[CADabra] Run manually: sudo apt-get update && sudo apt-get install -y libglu1-mesa libxft2")
        except Exception as install_error:
            print(f"[CADabra] ✗ Could not auto-install: {install_error}")
            print("[CADabra] Run manually: sudo apt-get update && sudo apt-get install -y libglu1-mesa libxft2")
    else:
        print(f"[CADabra] ✗ Gmsh error: {e}")
