"""ComfyUI-CADabra Prestartup Script."""
import logging

import shutil
from pathlib import Path
import folder_paths
from comfy_env import setup_env

log = logging.getLogger("cadabra")

setup_env()

# The CONFIGURED input directory, never the code-tree one. ComfyUI Desktop
# (--base-directory) and --input-directory both relocate it, and the load
# nodes only ever scan folder_paths.get_input_directory(). main.py runs
# apply_custom_paths() before prestartup scripts, so this is already resolved.
INPUT = Path(folder_paths.get_input_directory())


def copy_files(src: Path, dst: Path, pattern: str = "*") -> int:
    """Copy bundled assets into a directory. Returns files written.

    Seeds rather than syncs: an existing file is left alone, so a user's
    edited demo asset survives every relaunch. Raises if `src` is missing --
    a typo'd asset directory is a packaging bug, and silence is how it stays
    one.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"asset directory not found: {src}")
    written = 0
    for f in src.glob(pattern):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        written += 1
    return written


SCRIPT_DIR = Path(__file__).resolve().parent  # still used for viewer/web paths

# Copy example assets into ComfyUI's effective input dir.
# `"input/cad"` etc. are resolved by copy_files via folder_paths so they
# land where CAD_Load actually reads from — correct on vanilla ComfyUI,
# Comfy Desktop (inputDir override), --input-directory, etc.
copy_files("assets/cad", "input/cad", "**/*")
copy_files("assets/3d",  "input/3d",  "**/*")


# Copy Three.js (used directly by CAD viewers via <script> tags)
copy_files(get_three_dir(), SCRIPT_DIR / "web" / "three")

# Copy non-viewer JS widgets
for name in (
    "cadrille_inference.js", "mask_analyzer.js", "cad_preview_batch.js", "load_cad_upload.js",
    "cad_mesh_info.js", "cad_filename_info.js", "cad_load_glob_info.js",
    "cad_detect_degenerate_faces_info.js", "cad_detect_free_edges_info.js",
):
    copy_files(get_nodes_dir(), SCRIPT_DIR / "web" / "js", pattern=name)
