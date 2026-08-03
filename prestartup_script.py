"""ComfyUI-CADabra Prestartup Script."""
import logging
import os as _dbg_os
import sys as _dbg_sys

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer, get_three_dir

log = logging.getLogger("cadabra")

# TEMP DEBUG (remove after copy_files silent-0 mystery resolved):
# Wrap both example-asset copies so we can see why input/cad/ ends up
# missing on Comfy Desktop --dev runs.
def _dbg_call_copy(src_arg, dst_arg, pattern="*"):
    print(f"[CADabra PRESTARTUP DEBUG] __file__={__file__}", flush=True)
    print(f"[CADabra PRESTARTUP DEBUG] cwd={_dbg_os.getcwd()}", flush=True)
    _script_dir = Path(__file__).resolve().parent
    src_abs = _script_dir / src_arg
    print(f"[CADabra PRESTARTUP DEBUG] resolved-src-would-be={src_abs} "
          f"exists={src_abs.exists()} is_dir={src_abs.is_dir()} "
          f"list_head={sorted(p.name for p in src_abs.glob('*'))[:5] if src_abs.exists() else 'N/A'}",
          flush=True)
    try:
        import folder_paths as _dbg_fp
        _dbg_input = _dbg_fp.get_input_directory()
    except Exception as _e:
        _dbg_input = f"<folder_paths import err: {_e}>"
    print(f"[CADabra PRESTARTUP DEBUG] folder_paths.get_input_directory()={_dbg_input}",
          flush=True)
    print(f"[CADabra PRESTARTUP DEBUG] calling copy_files({src_arg!r}, {dst_arg!r}, {pattern!r})",
          flush=True)
    try:
        _n = copy_files(src_arg, dst_arg, pattern)
        print(f"[CADabra PRESTARTUP DEBUG]   -> returned {_n}", flush=True)
    except Exception as _e:
        print(f"[CADabra PRESTARTUP DEBUG]   -> RAISED {type(_e).__name__}: {_e}",
              flush=True)
        raise
    # Post-check what actually landed on disk
    try:
        import folder_paths as _dbg_fp2
        _base = Path(_dbg_fp2.get_input_directory())
        _sub = dst_arg.split('/', 1)[1] if '/' in dst_arg else ''
        _final = _base / _sub if _sub else _base
        print(f"[CADabra PRESTARTUP DEBUG]   post-check dst={_final} "
              f"exists={_final.exists()} "
              f"n_files={len(list(_final.glob('*'))) if _final.exists() else 'N/A'}",
              flush=True)
    except Exception as _e:
        print(f"[CADabra PRESTARTUP DEBUG]   post-check err: {_e}", flush=True)

print(f"[CADabra PRESTARTUP DEBUG] === CADabra prestartup starting (pid={_dbg_os.getpid()}) ===",
      flush=True)

setup_env()
print("[CADabra PRESTARTUP DEBUG] setup_env() returned OK", flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent  # still used for viewer/web paths

# Copy example assets into ComfyUI's effective input dir.
# `"input/cad"` etc. are resolved by copy_files via folder_paths so they
# land where CAD_Load actually reads from — correct on vanilla ComfyUI,
# Comfy Desktop (inputDir override), --input-directory, etc.
_dbg_call_copy("assets/cad", "input/cad", "**/*")
_dbg_call_copy("assets/3d",  "input/3d",  "**/*")

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
    "cad_detect_degenerate_faces_info.js", "cad_detect_free_edges_info.js",
):
    copy_files(get_nodes_dir(), SCRIPT_DIR / "web" / "js", pattern=name)

