"""ComfyUI-CADabra Prestartup Script."""
import logging

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer, get_three_dir

log = logging.getLogger("cadabra")

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy example assets from subdirectories
copy_files(SCRIPT_DIR / "assets" / "cad", COMFYUI_DIR / "input" / "cad", "**/*")
copy_files(SCRIPT_DIR / "assets" / "3d", COMFYUI_DIR / "input" / "3d", "**/*")

# Copy viewers
viewers = [
    "cad_analysis", "cad_curve", "cad_edge", "cad_edge_detail",
    "cad_edge_vtk", "cad_hierarchy", "cad_occ", "cad_roi", "cad_spline",
    "cad_dual",
]
for viewer in viewers:
    try:
        copy_viewer(viewer, SCRIPT_DIR / "web")
    except Exception as e:
        log.warning(f"{e}")

# Copy Three.js (used directly by CAD viewers via <script> tags)
copy_files(get_three_dir(), SCRIPT_DIR / "web" / "three")

# Copy non-viewer JS widgets
from comfy_3d_viewers import get_nodes_dir
for name in (
    "cadrille_inference.js", "mask_analyzer.js", "cad_preview_batch.js", "load_cad_upload.js",
    "cad_mesh_info.js", "cad_filename_info.js", "cad_load_glob_info.js",
    "cad_detect_degenerate_faces_info.js", "cad_free_edges_info.js",
):
    copy_files(get_nodes_dir(), SCRIPT_DIR / "web" / "js", pattern=name)

