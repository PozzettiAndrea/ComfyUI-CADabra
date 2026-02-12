"""
CADabra PreStartup Script
Copies example CAD files and workflows to ComfyUI on startup.
Also sets up Windows DLL paths for OpenCASCADE BEFORE PyTorch loads.
"""
import os
import sys
import shutil
import ctypes

# macOS: Import open3d BEFORE PyTorch to avoid segfault
# Open3D and PyTorch have C++ ABI incompatibilities that cause crashes.
# The required import order is VERSION AND PLATFORM DEPENDENT:
#   - macOS: open3d must load BEFORE torch (tested, crashes otherwise)
#   - Windows/Linux: may require the OPPOSITE order (torch before open3d)
# See: https://github.com/isl-org/Open3D/issues/5597
#      https://github.com/pytorch/pytorch/issues/74547
# DO NOT add open3d preloading for Windows without testing - it may cause crashes!
if sys.platform == 'darwin':
    try:
        import open3d
        print("[CADabra] OK - Open3D pre-initialized (before PyTorch)")
    except ImportError:
        pass  # open3d not installed, that's fine

# Linux: Preload correct TBB and libstdc++ from conda environment
# This prevents conflicts with older bundled versions from packages like
# pymeshlab, open3d, cgal that may load before occt_rt
if sys.platform == 'linux':
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        lib_dir = os.path.join(conda_prefix, 'lib')
        for lib_name in ['libtbb.so.12', 'libstdc++.so.6']:
            lib_path = os.path.join(lib_dir, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

# Windows DLL path fix for OpenCASCADE (pythonocc-core)
# CRITICAL: Must happen in prestartup_script.py BEFORE PyTorch loads
if sys.platform == 'win32':
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        dll_dir = os.path.join(conda_prefix, 'Library', 'bin')
        if os.path.isdir(dll_dir):
            # Method 1: Use add_dll_directory (Python 3.8+)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_dir)
            # Method 2: Also prepend to PATH as fallback
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

    # Preload correct TBB version BEFORE any package can load an incompatible one
    # (pymeshlab bundles an older tbb12.dll that breaks embree4.dll)
    try:
        import site
        for sp in site.getsitepackages():
            tbb_path = os.path.join(sp, 'occt_rt', 'libs', 'tbb12.dll')
            if os.path.exists(tbb_path):
                ctypes.CDLL(tbb_path)
                break
    except:
        pass

    # Force early OCC import to load DLLs before PyTorch can interfere
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        print("[CADabra] OK - OpenCASCADE pre-initialized (before PyTorch)")
    except Exception as e:
        print(f"[CADabra] WARNING - OpenCASCADE not available: {e}")

def setup_example_assets():
    """Automatically copy all CAD files to ComfyUI input/cad folder and STL files to input/3d folder."""
    try:
        import folder_paths

        # Get paths
        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(custom_node_dir, "assets")

        # Create subdirectories
        cad_folder = os.path.join(input_folder, "cad")
        non_cad_folder = os.path.join(input_folder, "3d")
        os.makedirs(cad_folder, exist_ok=True)
        os.makedirs(non_cad_folder, exist_ok=True)

        # Define file extensions for categorization
        NON_CAD_EXTENSIONS = {'.stl', '.obj', '.ply'}
        CAD_EXTENSIONS = {'.stp', '.step', '.igs', '.iges', '.sldprt', '.brep', '.iges', '.xyz'}

        # Scan assets directory recursively for all files
        if os.path.exists(assets_dir):
            for root, dirs, files in os.walk(assets_dir):
                for filename in files:
                    source_file = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()

                    # Determine destination based on file extension
                    dest_folder = None
                    folder_name = None

                    if file_ext in NON_CAD_EXTENSIONS:
                        dest_folder = non_cad_folder
                        folder_name = "input/3d"
                    elif file_ext in CAD_EXTENSIONS:
                        dest_folder = cad_folder
                        folder_name = "input/cad"
                    else:
                        # Skip files with unknown extensions
                        continue

                    dest_file = os.path.join(dest_folder, filename)

                    # Copy if destination doesn't exist
                    if not os.path.exists(dest_file):
                        shutil.copy2(source_file, dest_file)
                        print(f"[CADabra] Copied {filename} to {folder_name} folder")
                    else:
                        print(f"[CADabra] {filename} already exists in {folder_name} folder")
        else:
            print(f"[CADabra] Warning: assets directory not found at {assets_dir}")

    except Exception as e:
        print(f"[CADabra] Error setting up example assets: {e}")

# Run on import
setup_example_assets()