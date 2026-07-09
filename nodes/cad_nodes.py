from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
import numpy as np
import folder_paths
import trimesh
from comfy_api.latest import io
from .utils.occ_logging import log_operation

log = logging.getLogger("CADabra")  # use the configured logger (occ_logging) so these surface


def _progress_bar(completed, total, elapsed, width=30, prefix=""):
    """Print a tqdm-style progress bar."""
    if total == 0:
        return
    pct = completed / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    # Use \r to overwrite the line
    sys.stdout.write(f"\r{prefix}|{bar}| {completed}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _load_occ_shape_to_gmsh(occ_shape, model_name="cad_model"):
    """
    Convert OCC shape to GMSH model for meshing.
    This is the ONLY place where we need the OCC->GMSH conversion.
    """
    import gmsh
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone

    # Write OCC shape to temp STEP file
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        temp_path = f.name

    try:
        writer = STEPControl_Writer()
        writer.Transfer(occ_shape, STEPControl_AsIs)
        status = writer.Write(temp_path)
        if status != IFSelect_RetDone:
            raise RuntimeError("Failed to write temporary STEP file for meshing")

        # Ensure GMSH is initialized
        if not gmsh.is_initialized():
            try:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
            except ValueError as e:
                if "signal only works in main thread" not in str(e):
                    raise

        # Clear and import into GMSH
        gmsh.model.remove()
        gmsh.model.add(model_name)
        gmsh.model.occ.importShapes(temp_path)
        gmsh.model.occ.synchronize()

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _iter_occ_explorer(shape, topology_type):
    """
    Helper to iterate over OCC TopExp_Explorer results.

    Args:
        shape: TopoDS_Shape to explore
        topology_type: TopAbs type (e.g., TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE)

    Yields:
        TopoDS_Shape objects of the specified type
    """
    from OCC.Core.TopExp import TopExp_Explorer
    explorer = TopExp_Explorer(shape, topology_type)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


# IGES unit flag to string mapping (per IGES spec)
_IGES_UNIT_FLAGS = {
    1: "inches",
    2: "millimeters",
    3: "feet",
    4: "miles",
    5: "meters",
    6: "kilometers",
    7: "mils",
    8: "microns",
    9: "centimeters",
    10: "microinches",
}

# prevent IGESControl_Reader destructor crash (free(): invalid pointer)
_keep_alive = []


def _extract_iges_metadata(reader):
    """
    Extract metadata from IGES Global Section.

    Args:
        reader: IGESControl_Reader after ReadFile()

    Returns:
        dict with metadata fields, or empty dict if extraction fails
    """
    try:
        from OCC.Core.IGESData import IGESData_IGESModel

        ws = reader.WS()
        model = ws.Model()

        # Downcast to IGESData_IGESModel to access GlobalSection
        iges_model = IGESData_IGESModel.DownCast(model)
        if not iges_model:
            log.warning(" Could not downcast to IGESData_IGESModel")
            return {}

        gs = iges_model.GlobalSection()

        # Helper to safely get string from HAsciiString
        def get_string(ascii_obj):
            if ascii_obj:
                try:
                    return ascii_obj.ToCString()
                except Exception:
                    return ""
            return ""

        unit_flag = gs.UnitFlag()

        metadata = {
            "resolution": gs.Resolution(),
            "unit_flag": unit_flag,
            "units": _IGES_UNIT_FLAGS.get(unit_flag, f"unknown({unit_flag})"),
            "author": get_string(gs.AuthorName()),
            "filename": get_string(gs.FileName()),
            "company": get_string(gs.CompanyName()),
            "iges_version": gs.IGESVersion(),
            "drafting_standard": gs.DraftingStandard(),
        }

        return metadata
    except Exception as e:
        log.warning(f" Could not extract IGES metadata: {e}")
        return {}


# --- BREP Caching Helpers ---

def _load_brep(path):
    """Load BREP file, return OCC shape."""
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Shape
    builder = BRep_Builder()
    shape = TopoDS_Shape()
    with log_operation("BREP Read", file=os.path.basename(path)):
        success = breptools.Read(shape, path, builder)
    if not success:
        raise RuntimeError(f"Failed to read BREP file: {path}")
    return shape


def _save_brep(shape, path):
    """Save OCC shape to BREP file."""
    from OCC.Core.BRepTools import breptools
    success = breptools.Write(shape, path)
    if not success:
        raise RuntimeError(f"Failed to write BREP file: {path}")


def _load_step_or_iges(path, ext, filename):
    """
    Load STEP or IGES file, return (shape, metadata).

    Args:
        path: Full path to the CAD file
        ext: File extension (e.g., '.step', '.iges')
        filename: Just the filename for logging

    Returns:
        Tuple of (occ_shape, metadata_dict)
    """
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static

    metadata = {}

    if ext in ['.step', '.stp']:
        reader = STEPControl_Reader()
        with log_operation("STEP ReadFile", file=filename):
            status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {path}")
        with log_operation("STEP TransferRoots", file=filename):
            reader.TransferRoots()
        occ_shape = reader.OneShape()

    elif ext in ['.iges', '.igs']:
        # Preserve analytic curves (circles, ellipses) - don't convert to B-splines
        Interface_Static.SetIVal("read.iges.bspline.continuity", 0)
        Interface_Static.SetIVal("read.surfacecurve.mode", 0)  # 0 = use 3D curve, preserve type

        reader = IGESControl_Reader()
        with log_operation("IGES ReadFile", file=filename):
            status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read IGES file: {path}")

        metadata = _extract_iges_metadata(reader)

        with log_operation("IGES TransferRoots", file=filename):
            reader.TransferRoots()
        occ_shape = reader.OneShape()

        # WORKAROUND: IGESControl_Reader's C++ destructor crashes with
        # "free(): invalid pointer" after TransferRoots on certain files.
        # Keep the reader alive to prevent the destructor from running.
        # Memory impact is negligible since IGES loads are cached to BREP.
        _keep_alive.append(reader)
    else:
        raise RuntimeError(f"Unsupported format for _load_step_or_iges: {ext}")

    return occ_shape, metadata


def _load_cache_metadata(path):
    """Load metadata from JSON sidecar file."""
    import json
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache_metadata(metadata, path):
    """Save metadata to JSON sidecar file."""
    import json
    with open(path, 'w') as f:
        json.dump(metadata, f)


_CAD_EXTENSIONS = ['.step', '.stp', '.iges', '.igs', '.brep']
_NO_CAD_FILES = "(no CAD files found in input/cad)"

# Scan sources mirroring GeometryPack's Load Mesh: recursive input/cad (values
# input-relative, e.g. "cad/foo.step") plus top-level files in the input root
# (bare names). Shared by the initial snapshot (_get_cad_files) and the comfy-env
# live-refresh marker so both produce identical option values.
_CAD_SOURCES = [
    {"dir": "cad", "recursive": True, "rel_to_input": True},
    {"dir": "", "recursive": False, "rel_to_input": False},
]


def _scan_cad_sources(base, sources):
    """Scan input sources for CAD files. Mirrors comfy-env's parent-side scanner
    so the snapshot and live refresh agree. Returns sorted, de-duplicated values."""
    names, seen = [], set()
    for src in sources:
        subdir = src.get("dir", "") or ""
        recursive = bool(src.get("recursive", False))
        rel_to_input = bool(src.get("rel_to_input", False))
        root = os.path.join(base, subdir) if subdir else base
        try:
            if recursive:
                walked = (
                    (r, fn)
                    for r, _d, files in os.walk(root)
                    for fn in files
                )
                for r, fn in walked:
                    if os.path.splitext(fn)[1].lower() not in _CAD_EXTENSIONS:
                        continue
                    rel = os.path.relpath(os.path.join(r, fn),
                                          base if rel_to_input else root)
                    val = rel.replace(os.sep, "/")
                    if val not in seen:
                        seen.add(val)
                        names.append(val)
            else:
                for fn in os.listdir(root):
                    if not os.path.isfile(os.path.join(root, fn)):
                        continue
                    if os.path.splitext(fn)[1].lower() not in _CAD_EXTENSIONS:
                        continue
                    val = (os.path.join(subdir, fn).replace(os.sep, "/")
                           if rel_to_input and subdir else fn)
                    if val not in seen:
                        seen.add(val)
                        names.append(val)
        except Exception:
            continue
    names.sort()
    return names


def _get_cad_files():
    """Get available CAD files (input/cad recursively + input root top-level)."""
    base = folder_paths.get_input_directory()
    cad_files = _scan_cad_sources(base, _CAD_SOURCES)
    if not cad_files:
        cad_files = [_NO_CAD_FILES]
    return cad_files


def _resolve_cad_path(file_path):
    """Resolve a CAD file path the way GeometryPack's Load Mesh does: try
    input/cad/<p>, then input/<p>, then as an absolute path. Returns the resolved
    absolute path, or raises FileNotFoundError listing every location searched."""
    if not file_path or not str(file_path).strip():
        raise ValueError("CAD file path cannot be empty")
    base = folder_paths.get_input_directory()
    searched = []
    for candidate in (
        os.path.join(base, 'cad', file_path),
        os.path.join(base, file_path),
        file_path,
    ):
        searched.append(candidate)
        if os.path.isfile(candidate):
            return candidate
    msg = f"CAD file not found: '{file_path}'\nSearched in:"
    for p in searched:
        msg += f"\n  - {p}"
    raise FileNotFoundError(msg)


def _load_cad_model(cad_file_path):
    """Load a resolved CAD file into a CAD_MODEL dict, with BREP caching for
    STEP/IGES. Shared by CAD_Load (combo) and CAD_Load_Path (string path)."""
    import hashlib

    name = os.path.basename(cad_file_path)
    ext = os.path.splitext(cad_file_path)[1].lower()
    metadata = {}

    # BREP cache for STEP/IGES files (avoids expensive TransferRoots on reload)
    if ext in ['.step', '.stp', '.iges', '.igs']:
        cache_dir = os.path.join(os.path.dirname(cad_file_path), '.brep_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Auto-apply the AutoForm draw-direction tipping from a sibling <stem>_axis.igs
        # (IGES only), matching "Load CAD From Glob" / load_subprocess.py so every loader
        # produces the same tipped geometry. The axis file's mtime is folded into the cache
        # key so any pre-existing (un-tipped) cache is invalidated and a changed axis re-tips.
        axis_path = None
        cache_extra = ""
        if ext in ['.iges', '.igs'] and not cad_file_path.endswith('_axis.igs'):
            cand = os.path.splitext(cad_file_path)[0] + '_axis.igs'
            if os.path.exists(cand):
                axis_path = cand
                cache_extra = f"_axis{os.path.getmtime(cand)}"

        # Cache key = absolute path + modification time (auto-invalidates on change)
        mtime = os.path.getmtime(cad_file_path)
        cache_key = hashlib.md5(f"{cad_file_path}_{mtime}{cache_extra}".encode()).hexdigest()
        cache_brep = os.path.join(cache_dir, f"{cache_key}.brep")
        cache_meta = os.path.join(cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_brep):
            # FAST PATH: Use cached BREP (already tipped if an axis applied when cached)
            log.info(f" Using cache: {name}")
            metadata = _load_cache_metadata(cache_meta)
            brep_path = cache_brep
        else:
            # SLOW PATH: Load original, (optionally) tip via _axis.igs, create cache
            log.info(f" Loading CAD file: {name}")
            occ_shape, metadata = _load_step_or_iges(cad_file_path, ext, name)
            if axis_path:
                axis_data = parse_axis_igs(axis_path)
                if axis_data:
                    trsf = build_axis_transform(axis_data)
                    if trsf:
                        occ_shape = apply_transform_to_shape(occ_shape, trsf)
                        log.info(f" Applied axis transform from {os.path.basename(axis_path)}")
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata["axis_transform_applied"] = True
            _save_brep(occ_shape, cache_brep)
            if metadata:
                _save_cache_metadata(metadata, cache_meta)
            log.info(f" Cached to BREP: {name}")
            brep_path = cache_brep
    else:
        # BREP files - use directly (no caching needed, already native format)
        log.info(f" Using BREP file: {name}")
        brep_path = cad_file_path

    # Build CAD_MODEL return dict with brep_path (not occ_shape) to avoid pickle
    # issues when crossing process boundaries.
    cad_data = {
        "brep_path": brep_path,
        "file_path": cad_file_path,
        "format": ext,
    }
    if metadata:
        cad_data["metadata"] = metadata

    log.info(f" Ready: {name}")
    return cad_data


class CAD_Load(io.ComfyNode):
    """Load CAD files (STEP, IGES, BREP) using OpenCASCADE with BREP caching."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CAD_Load",
            display_name="Load CAD",
            category="CADabra",
            inputs=[
                io.Combo.Input("filename", options=_get_cad_files(),
                               tooltip="Select CAD file from input/cad folder",
                               extra_dict={
                                   # comfy-env: re-scan input/cad (and input root)
                                   # live so newly uploaded files appear without a
                                   # restart. Mirrors GeometryPack's Load Mesh.
                                   "comfy_env_dynamic_dir": "cad",
                                   "comfy_env_sources": _CAD_SOURCES,
                                   "comfy_env_exts": _CAD_EXTENSIONS,
                                   "comfy_env_placeholder": _NO_CAD_FILES,
                               }),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        """Accept any filename (execute() resolves/raises). This skips the combo
        "Value not in list" check so just-uploaded CAD files -- added to the widget
        client-side but not in the cached schema options -- still validate."""
        return True

    @classmethod
    def execute(cls, filename):
        # Check if placeholder (no files found)
        if filename.startswith("(no CAD files found"):
            raise FileNotFoundError(
                "No CAD files found in input/cad folder. "
                "Please add .step, .stp, .iges, .igs, or .brep files to ComfyUI/input/cad/"
            )

        cad_file_path = _resolve_cad_path(filename)
        return io.NodeOutput(_load_cad_model(cad_file_path))


class CAD_Load_Path(io.ComfyNode):
    """Load a CAD file (STEP, IGES, BREP) by typing its path, instead of picking
    from the dropdown. Mirrors GeometryPack's Load Mesh (Path)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CAD_Load_Path",
            display_name="Load CAD (Path)",
            category="CADabra",
            inputs=[
                io.String.Input(
                    "file_path", default="", multiline=False,
                    tooltip="Path to a CAD file. Resolved against input/cad, then "
                            "the input folder, then as an absolute path."),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file_path):
        """Force re-execution when the resolved file changes on disk."""
        try:
            return os.path.getmtime(_resolve_cad_path(file_path))
        except Exception:
            return file_path

    @classmethod
    def execute(cls, file_path):
        cad_file_path = _resolve_cad_path(file_path)
        return io.NodeOutput(_load_cad_model(cad_file_path))



def _mesh_diagnostics(tm, occ_shape=None):
    """Watertightness diagnostics for the CAD_Mesh info output.

    open_edges = mesh edges used by only one triangle (cracks/holes in the output).
    free_brep_edges = edges in the INPUT B-rep bounded by <2 faces (genuine open
    boundaries in the CAD itself) -- these are the usual source of a non-watertight
    output: the tessellation faithfully reproduces them and no vertex weld can close them.
    """
    d = {"watertight": None, "open_edges": None, "free_brep_edges": None}
    try:
        d["watertight"] = bool(tm.is_watertight)
    except Exception:
        pass
    try:
        import trimesh.grouping as _g
        d["open_edges"] = int(len(tm.edges[_g.group_rows(tm.edges_sorted, require_count=1)]))
    except Exception:
        pass
    if occ_shape is not None:
        try:
            from OCC.Core.TopExp import topexp
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
            from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
            m = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(occ_shape, TopAbs_EDGE, TopAbs_FACE, m)
            d["free_brep_edges"] = int(sum(1 for i in range(1, m.Size() + 1)
                                           if m.FindFromIndex(i).Size() < 2))
        except Exception:
            pass
    return d


def _mesh_info_str(backend, tm, metadata, diag):
    """Human-readable summary shown in the CAD_Mesh 'info' output / text box."""
    L = [f"CAD Mesh [{backend}]",
         f"  output: {len(tm.vertices):,} verts, {len(tm.faces):,} faces"]
    if metadata.get("num_cad_faces"):
        L.append(f"  CAD B-rep faces: {metadata['num_cad_faces']:,}")
    ld = metadata.get("linear_deflection")
    if ld is not None:
        L.append(f"  linear_deflection: {ld}  angular: {metadata.get('angular_deflection')}")
    if metadata.get("size_mode"):
        L.append(f"  size_mode: {metadata.get('size_mode')}  target: {metadata.get('target_size')}")
    wt, oe, fe = diag.get("watertight"), diag.get("open_edges"), diag.get("free_brep_edges")
    L.append(f"  watertight: {wt}" + (f"   (open edges: {oe})" if oe else ""))
    if fe is not None:
        L.append(f"  input B-rep free edges: {fe}"
                 + ("   <- these become the open edges (heal/sew the CAD to close)"
                    if fe else "   (closed solid)"))
    if metadata.get("has_cad_face_ids"):
        L.append("  cad_face_id: stored per triangle")
    return "\n".join(L)


def _flatten_dynamic(d):
    """Flatten DynamicCombo values into a flat kwargs dict.

    A DynamicCombo delivers its selected option's child widgets either nested as
    a sub-dict OR (in current ComfyUI) flattened with a 'comboName.' prefix, e.g.
    {'size_mode': 'Absolute', 'size_mode.target_size': 0.5, ...}. We recurse into
    sub-dicts AND strip any dotted prefix so children arrive under their bare
    parameter name ('size_mode.target_size' -> 'target_size'); otherwise they land
    in **kwargs and the node silently uses defaults (the target_size bug)."""
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten_dynamic(v))
        else:
            flat[k.rsplit(".", 1)[-1]] = v  # 'size_mode.target_size' -> 'target_size'
    return flat


class CAD_Mesh(io.ComfyNode):
    """Unified CAD-to-mesh node. Supports BRepIncremental (OCC) and GMSH backends."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CAD_Mesh",
            display_name="CAD Mesh",
            category="CADabra",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.DynamicCombo.Input("backend", tooltip="Tessellation engine. BRepIncremental (OCC): fast, robust chordal tessellation — one triangulation per CAD face, so every triangle keeps its exact source face (ideal for cad_face_id segmentation). GMSH: slower, but full control over algorithms, element-size fields, quads, and 3D volume meshes.", options=[
                    io.DynamicCombo.Option("BRepIncremental", [
                        io.Float.Input("linear_deflection", default=0.1, min=0.001, max=100.0, step=0.01,
                                       tooltip="Max chordal deviation — the largest allowed gap between a triangle and the true surface — in the SAME units as the CAD geometry (STEP is usually millimetres). NOT a fraction of the bounding box. Smaller = finer mesh / more triangles. Rule of thumb: ~0.1-1% of the part's size (e.g. a 100mm part: 0.1 -> very fine, 1.0 -> coarse). Planar faces are exact regardless of this value. If 'relative' is on, it is instead interpreted per-edge (see that tooltip)."),
                        io.Float.Input("angular_deflection", default=0.5, min=0.01, max=1.57, step=0.01,
                                       tooltip="Max angle between adjacent triangle normals on CURVED faces, in RADIANS (always absolute — NOT affected by 'relative'). Quick conversions (deg = rad x 57.3): 0.1 rad ~ 6 deg, 0.2 ~ 11 deg, 0.5 ~ 29 deg, 1.0 ~ 57 deg. Smaller = smoother curves / more triangles; typical 0.1-0.5. Sampled independently of linear_deflection — the mesher refines until BOTH limits are met, so lower both for a fine mesh on cylinders/spheres/fillets."),
                        io.Boolean.Input("merge_vertices", default=True,
                                         tooltip="Weld coincident vertices along shared CAD edges into one vertex -> watertight, connected (manifold) mesh. Off = each CAD face is tessellated independently with duplicated boundary vertices (faces stay 'loose'; breaks adjacency but gives strictly per-face geometry). This is a post-process, not an OCC mesher parameter."),
                        io.Boolean.Input("relative", default=False, advanced=True,
                                         tooltip="Applies to the LINEAR deflection ONLY (the angular deflection is always absolute radians). When ON, linear_deflection is interpreted relative to each EDGE's length — short edges get proportionally finer tessellation — relative to the edge, NOT the bounding box. Default OFF = absolute model units."),
                        io.Boolean.Input("parallel", default=False, advanced=True,
                                         tooltip="Tessellate independent CAD faces concurrently across CPU cores. Faster on large multi-face parts; geometry is identical."),
                        io.Float.Input("min_size", default=0.0, min=0.0, max=100.0, step=0.001, advanced=True,
                                       tooltip="Lower bound on triangle edge length, in model units. Prevents slivers / over-refinement on tiny features. 0 = no minimum (OCC picks one automatically)."),
                        io.Boolean.Input("internal_vertices_mode", default=True, advanced=True,
                                         tooltip="Insert vertices in the face INTERIOR (not just along boundaries/curvature) for better-shaped triangles. OFF = boundary/curvature-driven only -> fewer, more elongated interior triangles (faster, coarser interior). OCC IMeshTools 'InternalVerticesMode'."),
                        io.Boolean.Input("control_surface_deflection", default=True, advanced=True,
                                         tooltip="Enforce the deflection check across the surface INTERIOR, not only along edges. OFF = faster, but curved face interiors may bow out beyond linear_deflection. OCC IMeshTools 'ControlSurfaceDeflection'."),
                    ]),
                    io.DynamicCombo.Option("GMSH", [
                        io.DynamicCombo.Input("output_dim", tooltip="Mesh dimensionality. '2D Surface' tessellates only the outer faces into triangles — what you want for rendering, 3D printing, and segmentation. '3D Volume' additionally fills the solid interior with tetrahedra (for FEM/CFD); surfaces are meshed first, then the volume. 3D is much slower and only valid on watertight solids.", options=[
                            io.DynamicCombo.Option("2D Surface", []),
                            io.DynamicCombo.Option("3D Volume", [
                                io.Combo.Input("algorithm_3d", options=["Delaunay", "Frontal (Netgen)", "HXT", "MMG3D"],
                                               default="Delaunay", tooltip="Tetrahedral (volume) mesher, used only for 3D Volume. Delaunay: robust general-purpose default. Frontal (Netgen): advancing-front, better-shaped tets near boundaries, slower. HXT: massively-parallel Delaunay — fastest for very large volumes. MMG3D: anisotropic / quality remeshing."),
                            ]),
                        ]),
                        io.Combo.Input("algorithm_2d", options=[
                            "Delaunay", "Automatic", "MeshAdapt", "Frontal-Delaunay",
                            "BAMG", "Frontal-Delaunay for Quads",
                            "Packing of Parallelograms", "Quasi-structured Quad"
                        ], tooltip="Surface (triangle) mesher — always runs, since surfaces are meshed before any 3D step. Delaunay: fast, robust, isotropic (good default). Automatic: gmsh chooses per surface. MeshAdapt: most robust on dirty/complex/near-degenerate CAD. Frontal-Delaunay: best triangle quality/uniformity (recommended for clean isotropic meshes). BAMG: anisotropic. 'Frontal-Delaunay for Quads' / 'Packing of Parallelograms' / 'Quasi-structured Quad' aim for quad-dominant or structured layouts — pair with 'recombine_to_quads' for full quads."),
                        io.DynamicCombo.Input("size_mode", tooltip="How 'target_size' is interpreted. Absolute = edge length in model units. Relative to Bounds = fraction of the bounding-box diagonal. Curvature Adaptive = size driven purely by local curvature (target_size is ignored).", options=[
                            io.DynamicCombo.Option("Absolute", [
                                io.Float.Input("target_size", default=1.0, min=0.0001, max=1000.0, step=0.001,
                                               tooltip="Target triangle edge length in the SAME units as the CAD (STEP is usually mm). Element size is then clamped to [target x min_size_factor, target x max_size_factor]. Smaller = finer / more triangles. E.g. on a ~600mm-diagonal part: 1.0 -> very fine (~50k tris), 5 -> medium, 20 -> coarse."),
                                io.Float.Input("size_factor", default=1.0, min=0.01, max=100.0, step=0.1,
                                               tooltip="Global multiplier applied to ALL mesh sizes (gmsh Mesh.MeshSizeFactor), on top of target_size. 0.5 = everything twice as fine; 2.0 = twice as coarse. Quick global density knob."),
                                io.Float.Input("min_size_factor", default=1.0, min=0.001, max=1.0, step=0.01,
                                               tooltip="Floor on element size as a fraction of target: min = target x this. 1.0 = nothing smaller than target (uniform). Lower (e.g. 0.1) lets tight features/curves refine below target."),
                                io.Float.Input("max_size_factor", default=10.0, min=1.0, max=100.0, step=0.1,
                                               tooltip="Ceiling on element size as a multiple of target: max = target x this. 1.0 = uniform (cap at target). Higher (e.g. 10) lets flat regions use bigger triangles -> far fewer total elements."),
                            ]),
                            io.DynamicCombo.Option("Relative to Bounds", [
                                io.Float.Input("target_size", default=1.0, min=0.0001, max=1000.0, step=0.001,
                                               tooltip="Target edge length as a FRACTION of the bounding-box diagonal (unit/scale-independent across parts). 0.01 = 1% of the diagonal (fine), 0.05 = 5% (medium). Clamped to [x min_size_factor, x max_size_factor]. Use this for a consistent triangle count regardless of the part's absolute size."),
                                io.Float.Input("size_factor", default=1.0, min=0.01, max=100.0, step=0.1,
                                               tooltip="Global multiplier applied to ALL mesh sizes (gmsh Mesh.MeshSizeFactor), on top of target_size. 0.5 = twice as fine; 2.0 = twice as coarse."),
                                io.Float.Input("min_size_factor", default=1.0, min=0.001, max=1.0, step=0.01,
                                               tooltip="Floor on element size as a fraction of target (target = fraction-of-diagonal here). 1.0 = uniform; lower lets fine features refine further."),
                                io.Float.Input("max_size_factor", default=10.0, min=1.0, max=100.0, step=0.1,
                                               tooltip="Ceiling on element size as a multiple of target. 1.0 = uniform; higher lets flat regions coarsen for fewer elements."),
                            ]),
                            io.DynamicCombo.Option("Curvature Adaptive", [
                                io.Int.Input("elements_per_2pi", default=20, min=4, max=100, step=1,
                                             tooltip="Curvature-driven sizing: number of elements placed around a FULL circle (2*PI) of curvature. 20 ~= 18 deg per segment; 40 = smoother curves / more triangles on cylinders, holes and fillets. Flat faces stay coarse. (gmsh MinimumCirclePoints.) target_size is not used in this mode."),
                                io.Boolean.Input("extend_from_boundary", default=True,
                                                 tooltip="Propagate the (fine) curvature-driven size from curved boundaries into adjacent flatter regions so the size transitions smoothly. Off = flat interiors can be much coarser right next to fine curved edges (sharper size jumps)."),
                                io.Float.Input("max_edge_length", default=0.0, min=0.0, max=10000.0, step=0.1,
                                               tooltip="Upper bound on edge length in model units (mm). Curves still refine finer than this per 'elements_per_2pi', but FLAT faces are capped here instead of getting giant triangles. This is the key knob: curvature-guided where it matters, uniform-ish everywhere else. 0 = no cap (flat faces can be very coarse, up to 10x the bbox diagonal)."),
                                io.Float.Input("min_edge_length", default=0.0, min=0.0, max=10000.0, step=0.001, advanced=True,
                                               tooltip="Lower bound on edge length in model units (mm). Stops over-refinement on tiny high-curvature features (e.g. small fillets) from exploding the triangle count. 0 = auto (~0.0001 x bbox diagonal)."),
                            ]),
                        ]),
                        io.Combo.Input("element_order", options=["1", "2"],
                                       tooltip="Polynomial order per element. 1 = linear, straight-sided triangles/tets — use for rendering, 3D printing, segmentation. 2 = quadratic (adds mid-edge nodes -> curved elements) — only for FEM accuracy; ~doubles the node count and is NOT useful for visualization."),
                        io.Int.Input("smoothing_steps", default=0, min=0, max=10, step=1,
                                     tooltip="Laplacian smoothing iterations applied DURING generation (gmsh Mesh.Smoothing): each pass nudges interior nodes toward their neighbours' average to improve triangle shape. Node movement only, no topology change. 0 = off; 1-5 is plenty. Cheap."),
                        io.Boolean.Input("optimize", default=False,
                                         tooltip="After generation, run a quality optimiser that relocates nodes and swaps edges/faces to remove badly-shaped or inverted elements (slivers). 2D = a mild Laplace2D pass; 3D Volume = the Netgen optimiser (the impactful one for tet quality). Off = use the raw mesher output."),
                        io.Int.Input("optimization_passes", default=1, min=0, max=10, step=1, advanced=True,
                                     tooltip="How many times the optimiser (see 'optimize') runs back-to-back after generation — each pass further improves element quality with diminishing returns; 1-2 is almost always enough, >3 rarely helps. ONLY has an effect when 'optimize' is ON — ignored otherwise."),
                        io.Boolean.Input("recombine_to_quads", default=False,
                                         tooltip="After triangulation, merge adjacent triangle pairs into quadrilaterals (gmsh Mesh.RecombineAll) -> a quad-dominant (or all-quad) mesh. Pair with a quad-oriented 2D algorithm ('Frontal-Delaunay for Quads' / 'Quasi-structured Quad') for the cleanest quads."),
                        io.Combo.Input("subdivision", options=["None", "All Triangles", "All Quadrangles", "Barycentric"],
                                       tooltip="Uniform refinement applied AFTER meshing (gmsh Mesh.SubdivisionAlgorithm). None = leave as generated. All Triangles = split each triangle 1->4. All Quadrangles = convert to an all-quad mesh. Barycentric = split via face centroids. Each multiplies the element count."),
                        io.Int.Input("num_threads", default=0, min=0, max=256, step=1, advanced=True,
                                     tooltip="CPU threads for meshing. gmsh is OpenMP-parallel but defaults to SINGLE-threaded, so this is usually free speed. 0 = use all cores. Biggest gains: 2D meshing of many-surface CAD (each surface on a thread — these parts have hundreds of faces) and the HXT 3D algorithm. Sequential Frontal meshing benefits little; results may differ slightly run-to-run when threaded."),
                        io.String.Input("gmsh_options", multiline=True, default="{}",
                                        tooltip="Escape hatch: raw gmsh options as a JSON object, applied verbatim AFTER the settings above (numbers via setNumber, strings via setString), overriding them. e.g. {\"Mesh.MeshSizeFromCurvature\": 1, \"Mesh.AngleToleranceFacetOverlap\": 0.1}. See the gmsh reference manual for option names."),
                    ]),
                ]),
                io.Boolean.Input("extract_face_ids", default=True, optional=True,
                                 tooltip="Tag every triangle with the index of the CAD B-rep face it came from, as the 'cad_face_id' face attribute (one int per triangle). OCC tessellates each TopoDS_Face separately, so this is exact and free — the basis for per-face segmentation/colouring. IDs follow the face enumeration order: stable for a given shape, but renumbered for a different/edited model."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
                io.Custom("MESH_METADATA").Output(display_name="metadata"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, backend, extract_face_ids=True):
        params = _flatten_dynamic(backend)
        selected = params.pop("backend")
        if selected == "BRepIncremental":
            return cls._mesh_brep(cad_model, extract_face_ids=extract_face_ids, **params)
        elif selected == "GMSH":
            return cls._mesh_gmsh(cad_model, extract_face_ids=extract_face_ids, **params)
        else:
            raise ValueError(f"Unknown backend: {selected}")

    # -------------------------------------------------------------------------
    # BRepIncremental backend
    # -------------------------------------------------------------------------
    @classmethod
    def _mesh_brep(cls, cad_model, linear_deflection=0.1, angular_deflection=0.5,
                   relative=False, parallel=False, min_size=0.0,
                   internal_vertices_mode=True, control_surface_deflection=True,
                   merge_vertices=True, extract_face_ids=True, **_kwargs):
        import trimesh
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.IMeshTools import IMeshTools_Parameters
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.TopoDS import topods

        from .utils.brep_cache import get_occ_shape
        occ_shape = get_occ_shape(cad_model)

        log.info(f"CAD_Mesh [BRepIncremental]: linear_deflection={linear_deflection}, angular_deflection={angular_deflection}, "
                 f"relative={relative}, parallel={parallel}, min_size={min_size}")

        # 1. Mesh the shape with BRepMesh using IMeshTools_Parameters
        params = IMeshTools_Parameters()
        params.Deflection = linear_deflection
        params.Angle = angular_deflection
        params.Relative = relative
        params.InParallel = parallel
        params.MinSize = min_size
        # Set defensively: these exist on OCC 7.x but guard in case of API drift.
        for _attr, _val in (("InternalVerticesMode", internal_vertices_mode),
                            ("ControlSurfaceDeflection", control_surface_deflection)):
            if hasattr(params, _attr):
                setattr(params, _attr, _val)

        with log_operation("BRepMesh", linear_deflection=linear_deflection, angular_deflection=angular_deflection,
                           relative=relative, parallel=parallel, min_size=min_size):
            mesher = BRepMesh_IncrementalMesh(occ_shape, params)
            mesher.Perform()

        if not mesher.IsDone():
            raise RuntimeError("BRepMesh tessellation failed")

        # 2. Collect triangulations from all faces
        all_verts = []
        all_faces = []
        cad_face_ids = []
        vertex_offset = 0
        cad_face_idx = 0

        explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)

            if tri is not None:
                trsf = loc.Transformation()

                # Extract vertices with transformation
                for i in range(1, tri.NbNodes() + 1):
                    pnt = tri.Node(i)
                    pnt.Transform(trsf)
                    all_verts.append([pnt.X(), pnt.Y(), pnt.Z()])

                # Extract triangles with offset, handle face orientation
                is_reversed = face.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    triangle = tri.Triangle(i)
                    n1, n2, n3 = triangle.Get()
                    if is_reversed:
                        all_faces.append([
                            vertex_offset + n1 - 1,
                            vertex_offset + n3 - 1,
                            vertex_offset + n2 - 1
                        ])
                    else:
                        all_faces.append([
                            vertex_offset + n1 - 1,
                            vertex_offset + n2 - 1,
                            vertex_offset + n3 - 1
                        ])
                    cad_face_ids.append(cad_face_idx)

                vertex_offset += tri.NbNodes()

            cad_face_idx += 1
            explorer.Next()

        if len(all_verts) == 0:
            raise RuntimeError("Tessellation produced no vertices")

        # 3. Create trimesh
        tm = trimesh.Trimesh(
            vertices=np.array(all_verts, dtype=np.float32),
            faces=np.array(all_faces, dtype=np.int32)
        )

        verts_before = len(tm.vertices)

        # 4. Optionally merge duplicate vertices
        if merge_vertices:
            tm.merge_vertices()

        # 5. Store CAD face IDs if requested
        if extract_face_ids and cad_face_ids:
            cad_face_ids_array = np.array(cad_face_ids, dtype=np.int32)
            tm.face_attributes['cad_face_id'] = cad_face_ids_array
            tm.metadata['has_cad_face_ids'] = True
            tm.metadata['num_cad_faces'] = cad_face_idx
        else:
            tm.metadata['has_cad_face_ids'] = False

        # 6. Build metadata
        file_path = cad_model.get("file_path", "model")
        metadata = {
            "backend": "BRepIncremental",
            "num_vertices": len(tm.vertices),
            "num_faces": len(tm.faces),
            "num_cad_faces": cad_face_idx,
            "linear_deflection": linear_deflection,
            "angular_deflection": angular_deflection,
            "relative": relative,
            "parallel": parallel,
            "min_size": min_size,
            "merged": merge_vertices,
            "vertices_before_merge": verts_before,
            "has_cad_face_ids": extract_face_ids,
            "file_path": file_path,
        }

        tm.metadata['file_path'] = file_path

        log.info(f"CAD_Mesh [BRepIncremental]: {len(tm.vertices)} verts, {len(tm.faces)} faces "
                 f"from {cad_face_idx} CAD faces (merged={merge_vertices}, before={verts_before})")

        diag = _mesh_diagnostics(tm, occ_shape)
        metadata.update(diag)
        info = _mesh_info_str("BRepIncremental", tm, metadata, diag)
        log.info("[CAD_Mesh] %s", info.replace("\n", " | "))
        return io.NodeOutput(tm, metadata, info, ui={"text": [info]})

    # -------------------------------------------------------------------------
    # GMSH backend
    # -------------------------------------------------------------------------
    @classmethod
    def _mesh_gmsh(cls, cad_model, output_dim="2D Surface", algorithm_2d="Delaunay",
                   algorithm_3d="Delaunay", size_mode="Absolute",
                   target_size=1.0, size_factor=1.0,
                   min_size_factor=1.0, max_size_factor=10.0,
                   elements_per_2pi=20, extend_from_boundary=True,
                   max_edge_length=0.0, min_edge_length=0.0,
                   smoothing_steps=0, optimize=False, optimization_passes=1,
                   recombine_to_quads=False, subdivision="None",
                   element_order="1", extract_face_ids=True, num_threads=0, gmsh_options="{}", **_kwargs):
        try:
            import gmsh
            import json
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        # --- DEBUG: exactly what reached the mesher (catches nested-DynamicCombo plumbing bugs) ---
        log.info("[GMSH] launching meshing — mode=%s | 2D algo=%s | 3D algo=%s | target_size=%s | order=%s | "
                 "optimize=%s (passes=%d) | smoothing=%d | recombine=%s | subdivision=%s | threads=%s | face_ids=%s",
                 size_mode, algorithm_2d, (algorithm_3d if output_dim == "3D Volume" else "n/a (2D surface)"),
                 target_size, element_order, ("yes" if optimize else "no"), optimization_passes,
                 smoothing_steps, ("yes" if recombine_to_quads else "no"), subdivision,
                 (num_threads if num_threads else "all"),
                 ("yes" if extract_face_ids else "no"))
        log.info("[GMSH] >>> _mesh_gmsh called")
        log.info("[GMSH] size: size_mode=%r target_size=%r size_factor=%r min_size_factor=%r max_size_factor=%r",
                 size_mode, target_size, size_factor, min_size_factor, max_size_factor)
        log.info("[GMSH] curv: elements_per_2pi=%r extend_from_boundary=%r max_edge_length=%r min_edge_length=%r",
                 elements_per_2pi, extend_from_boundary, max_edge_length, min_edge_length)
        log.info("[GMSH] algo: output_dim=%r algorithm_2d=%r algorithm_3d=%r element_order=%r",
                 output_dim, algorithm_2d, algorithm_3d, element_order)
        log.info("[GMSH] post: smoothing_steps=%r optimize=%r optimization_passes=%r recombine=%r subdivision=%r extract_face_ids=%r",
                 smoothing_steps, optimize, optimization_passes, recombine_to_quads, subdivision, extract_face_ids)
        if _kwargs:
            log.warning("[GMSH] UNCONSUMED kwargs (likely a widget-name/plumbing mismatch): %r", _kwargs)

        # 2D algorithm mappings (from Gmsh docs)
        algo_2d_map = {
            "Automatic": 2,
            "MeshAdapt": 1,
            "Delaunay": 5,
            "Frontal-Delaunay": 6,
            "BAMG": 7,
            "Frontal-Delaunay for Quads": 8,
            "Packing of Parallelograms": 9,
            "Quasi-structured Quad": 11,
        }

        # 3D algorithm mappings (from Gmsh docs)
        algo_3d_map = {
            "Delaunay": 1,
            "Frontal (Netgen)": 4,
            "MMG3D": 7,
            "HXT": 10,
        }

        subdivision_map = {
            "None": 0,
            "All Triangles": 1,
            "All Quadrangles": 2,
            "Barycentric": 3,
        }

        try:
            # Load OCC shape from brep_path
            from .utils.brep_cache import load_shape
            occ_shape = load_shape(cad_model.get("brep_path"))
            _load_occ_shape_to_gmsh(occ_shape, "mesh_model")

            # Get model bounding box for size calculations
            try:
                bbox = gmsh.model.getBoundingBox(-1, -1)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox
                diagonal = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
                log.info(f" Model diagonal: {diagonal:.2f}")
            except Exception as e:
                log.warning(f" Could not get bounding box ({e})")
                diagonal = 1.0  # Fallback

            # Calculate actual sizes based on mode
            if size_mode == "Relative to Bounds":
                actual_target_size = diagonal * target_size
                actual_min_size = actual_target_size * min_size_factor
                actual_max_size = actual_target_size * max_size_factor
                log.info(f" Relative sizing: target={target_size:.3f} * diagonal -> {actual_target_size:.4f}")

                gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", actual_min_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_max_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            elif size_mode == "Absolute":
                # Check if target size is dangerously small (< 0.1% of diagonal)
                size_ratio = target_size / diagonal
                if size_ratio < 0.001:
                    raise ValueError(
                        f"Target size {target_size} is too small for this model (diagonal={diagonal:.1f}). "
                        f"Size is {size_ratio*100:.3f}% of diagonal - this would create an extremely dense mesh "
                        f"and likely hang. Minimum recommended size is {diagonal * 0.001:.1f} (0.1% of diagonal). "
                        f"Consider using 'Relative to Bounds' mode or increasing target_size."
                    )

                actual_target_size = target_size
                actual_min_size = actual_target_size * min_size_factor
                actual_max_size = actual_target_size * max_size_factor
                log.info(f" Absolute sizing: target={actual_target_size}")

                gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", actual_min_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_max_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            elif size_mode == "Curvature Adaptive":
                # Curvature drives the size; the Min/Max clamps bound it. A user-set
                # max_edge_length caps flat faces (curves still refine finer); else
                # the historical diagonal*10 (effectively uncapped) is used.
                _cmax = max_edge_length if (max_edge_length and max_edge_length > 0) else diagonal * 10.0
                _cmin = min_edge_length if (min_edge_length and min_edge_length > 0) else diagonal * 0.0001
                log.info(f" Curvature adaptive: elements_per_2pi={elements_per_2pi}, extend_from_boundary={extend_from_boundary}, "
                         f"max_edge={_cmax:.4f} ({'user' if max_edge_length>0 else 'auto'}), "
                         f"min_edge={_cmin:.4f} ({'user' if min_edge_length>0 else 'auto'})")

                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
                gmsh.option.setNumber("Mesh.MinimumCirclePoints", elements_per_2pi)

                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", _cmin)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", _cmax)

                if extend_from_boundary:
                    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
                else:
                    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

                gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)

                actual_target_size = 0
                actual_min_size = 0
                actual_max_size = 0
            else:
                # Fallback
                actual_target_size = diagonal * target_size
                actual_min_size = actual_target_size * min_size_factor
                actual_max_size = actual_target_size * max_size_factor
                gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", actual_min_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_max_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            # --- DEBUG: the EFFECTIVE size options gmsh will actually use ---
            log.info("[GMSH] sizing branch=%r diagonal=%.6f -> actual_target=%.6f actual_min=%.6f actual_max=%.6f",
                     size_mode, diagonal, actual_target_size, actual_min_size, actual_max_size)
            for _k in ("Mesh.MeshSizeFactor", "Mesh.MeshSizeMin", "Mesh.MeshSizeMax",
                       "Mesh.CharacteristicLengthMin", "Mesh.CharacteristicLengthMax",
                       "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeFromPoints",
                       "Mesh.MeshSizeExtendFromBoundary", "Mesh.MinimumCirclePoints"):
                try:
                    log.info("[GMSH]   effective %s = %s", _k, gmsh.option.getNumber(_k))
                except Exception as _e:
                    log.info("[GMSH]   effective %s = <unavailable: %s>", _k, _e)

            # Set 2D algorithm (always used - surfaces are meshed first even for 3D)
            gmsh.option.setNumber("Mesh.Algorithm", algo_2d_map.get(algorithm_2d, 2))

            # Set 3D algorithm only if generating 3D mesh
            is_3d = output_dim == "3D Volume"
            if is_3d:
                gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d_map.get(algorithm_3d, 1))

            # Threads (gmsh defaults to single-threaded). 0 = all cores.
            import os as _os
            _nthreads = int(num_threads) if num_threads and int(num_threads) > 0 else (_os.cpu_count() or 1)
            gmsh.option.setNumber("General.NumThreads", _nthreads)
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", _nthreads)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", _nthreads)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", _nthreads)
            log.info("[GMSH] threads = %d (num_threads input=%s, cores=%s)", _nthreads, num_threads, _os.cpu_count())

            # Set element order
            gmsh.option.setNumber("Mesh.ElementOrder", int(element_order))

            # Set smoothing steps
            gmsh.option.setNumber("Mesh.Smoothing", smoothing_steps)

            # Set recombination
            if recombine_to_quads:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
            else:
                gmsh.option.setNumber("Mesh.RecombineAll", 0)

            # Apply custom gmsh options from JSON
            if gmsh_options and gmsh_options.strip() != "{}":
                try:
                    options_dict = json.loads(gmsh_options)
                    for key, value in options_dict.items():
                        if isinstance(value, (int, float)):
                            gmsh.option.setNumber(key, value)
                            log.info(f" Applied gmsh option: {key} = {value}")
                        elif isinstance(value, str):
                            gmsh.option.setString(key, value)
                            log.info(f" Applied gmsh option: {key} = '{value}'")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in gmsh_options: {e}")

            # Set up Gmsh logging to file (named after CAD file)
            cad_file_path = cad_model.get("file_path", "")
            if cad_file_path:
                cad_name = os.path.splitext(os.path.basename(cad_file_path))[0]
            else:
                cad_name = "unknown"
            log_file = os.path.join(folder_paths.get_output_directory(), f"gmsh_mesh_{cad_name}.log")
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 99)
            gmsh.logger.start()

            log.info(f" Starting mesh generation (log: {log_file})...")

            # Generate mesh
            dim = 3 if is_3d else 2
            with log_operation("GMSH mesh.generate", dim=dim, algorithm_2d=algorithm_2d):
                gmsh.model.mesh.generate(dim)

            # --- DEBUG: resulting element counts (so a no-op change is obvious in the log) ---
            try:
                _e2 = gmsh.model.mesh.getElements(2)[1]
                _ntri = sum(len(a) for a in _e2) if _e2 else 0
                _nnodes = len(gmsh.model.mesh.getNodes()[0])
                log.info("[GMSH] generated: %d surface elements, %d nodes (dim=%d)", _ntri, _nnodes, dim)
            except Exception as _e:
                log.info("[GMSH] could not count elements: %s", _e)

            # Write Gmsh logs to file
            logs = gmsh.logger.get()
            gmsh.logger.stop()

            with open(log_file, 'w') as f:
                f.write(f"=== Gmsh Mesh Generation Log ===\n")
                f.write(f"Diagonal: {diagonal:.2f}\n")
                f.write(f"Size mode: {size_mode}\n")
                f.write(f"Algorithm 2D: {algorithm_2d}\n\n")
                for line in logs:
                    f.write(line + '\n')

            log.info(f" Mesh generation complete. Log saved to: {log_file}")

            # Apply mesh optimization if requested
            if optimize and optimization_passes > 0:
                for i in range(optimization_passes):
                    gmsh.model.mesh.optimize("Laplace2D" if not is_3d else "Netgen")
                log.info(f" Applied {optimization_passes} optimization passes")

            # Apply subdivision if requested
            if subdivision != "None":
                subdivision_algo = subdivision_map[subdivision]
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", subdivision_algo)
                gmsh.model.mesh.refine()

            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)

            # Helper function to determine nodes per element
            def get_nodes_per_elem(elem_type):
                elem_type_map = {
                    2: 3,   # 3-node triangle
                    3: 4,   # 4-node quadrangle
                    4: 4,   # 4-node tetrahedron
                    5: 8,   # 8-node hexahedron
                    9: 6,   # 6-node triangle (2nd order)
                    10: 9,  # 9-node quadrangle (2nd order)
                    11: 10, # 10-node tetrahedron (2nd order)
                }
                return elem_type_map.get(elem_type, None)

            # Helper function to triangulate a face
            def triangulate_face(face, is_3d_volume):
                triangles = []
                if len(face) == 3:
                    triangles.append(list(face))
                elif len(face) == 4:
                    triangles.append([face[0], face[1], face[2]])
                    triangles.append([face[0], face[2], face[3]])
                elif len(face) > 4:
                    if not is_3d_volume:
                        triangles.append([face[0], face[1], face[2]])
                        if len(face) == 9:
                            triangles.append([face[0], face[2], face[3]])
                    else:
                        triangles.append([face[0], face[1], face[2]])
                        if len(face) >= 4:
                            triangles.append([face[0], face[1], face[3]])
                            triangles.append([face[0], face[2], face[3]])
                            triangles.append([face[1], face[2], face[3]])
                return triangles

            # Extract face IDs if requested (only for 2D surface meshing)
            cad_face_ids = []
            triangulated_faces = []
            original_face_count = 0

            if extract_face_ids and not is_3d:
                surface_entities = gmsh.model.getEntities(dim=2)
                log.info(f" Extracting face IDs from {len(surface_entities)} CAD surfaces")

                for cad_face_idx, (dim, tag) in enumerate(surface_entities):
                    surf_elem_types, surf_elem_tags, surf_elem_node_tags = gmsh.model.mesh.getElements(dim, tag)

                    for i, elem_type in enumerate(surf_elem_types):
                        elem_nodes = surf_elem_node_tags[i]
                        nodes_per_elem = get_nodes_per_elem(elem_type)
                        if nodes_per_elem is None:
                            nodes_per_elem = len(elem_nodes) // len(surf_elem_tags[i])

                        faces = elem_nodes.reshape(-1, nodes_per_elem) - 1
                        original_face_count += len(faces)

                        for face in faces:
                            tris = triangulate_face(face, is_3d)
                            for tri in tris:
                                triangulated_faces.append(tri)
                                cad_face_ids.append(cad_face_idx)

                log.info(f" Extracted {len(cad_face_ids)} triangles with CAD face IDs")

            else:
                # Standard extraction (no face IDs, or 3D volume meshing)
                if is_3d:
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
                else:
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

                if len(elem_node_tags) == 0:
                    raise RuntimeError("Mesh generation produced no elements")

                for i, elem_type in enumerate(elem_types):
                    elem_nodes = elem_node_tags[i]
                    nodes_per_elem = get_nodes_per_elem(elem_type)
                    if nodes_per_elem is None:
                        nodes_per_elem = len(elem_nodes) // len(elem_tags[i])

                    faces = elem_nodes.reshape(-1, nodes_per_elem) - 1
                    original_face_count += len(faces)

                    for face in faces:
                        tris = triangulate_face(face, is_3d)
                        triangulated_faces.extend(tris)

            if len(triangulated_faces) == 0:
                raise RuntimeError("Mesh generation produced no elements")

            triangulated_faces = np.array(triangulated_faces, dtype=np.int32)

            if original_face_count != len(triangulated_faces):
                log.info(f"Converted {original_face_count} elements to {len(triangulated_faces)} triangles")

            # Create trimesh object
            tm = trimesh.Trimesh(
                vertices=vertices,
                faces=triangulated_faces
            )

            # Store CAD face IDs as face attribute if extracted
            if cad_face_ids:
                cad_face_ids_array = np.array(cad_face_ids, dtype=np.int32)
                tm.face_attributes['cad_face_id'] = cad_face_ids_array
                tm.metadata['has_cad_face_ids'] = True
                tm.metadata['num_cad_faces'] = len(surface_entities)
                log.info(f" Stored cad_face_id attribute ({len(surface_entities)} CAD faces)")
            else:
                tm.metadata['has_cad_face_ids'] = False

            # Store CADabra-specific metadata in trimesh.metadata
            tm.metadata['cadabra_type'] = "3D" if is_3d else "2D"
            tm.metadata['output_dim'] = output_dim
            tm.metadata['size_mode'] = size_mode
            tm.metadata['target_size'] = target_size
            tm.metadata['actual_target_size'] = actual_target_size
            tm.metadata['actual_min_size'] = actual_min_size
            tm.metadata['actual_max_size'] = actual_max_size
            tm.metadata['size_factor'] = size_factor
            tm.metadata['element_order'] = int(element_order)
            tm.metadata['algorithm_2d'] = algorithm_2d
            tm.metadata['algorithm_3d'] = algorithm_3d if is_3d else None
            tm.metadata['curvature_adaptive'] = size_mode == "Curvature Adaptive"
            tm.metadata['smoothing_steps'] = smoothing_steps
            tm.metadata['recombine_to_quads'] = recombine_to_quads

            # Create separate metadata dict for explicit output
            mesh_type_str = "3D" if is_3d else "2D"
            metadata = {
                'backend': 'GMSH',
                'type': mesh_type_str,
                'output_dim': output_dim,
                'size_mode': size_mode,
                'target_size': target_size,
                'actual_target_size': actual_target_size,
                'actual_min_size': actual_min_size,
                'actual_max_size': actual_max_size,
                'size_factor': size_factor,
                'element_order': int(element_order),
                'algorithm_2d': algorithm_2d,
                'algorithm_3d': algorithm_3d if is_3d else None,
                'curvature_adaptive': size_mode == "Curvature Adaptive",
                'elements_per_2pi': elements_per_2pi if size_mode == "Curvature Adaptive" else None,
                'smoothing_steps': smoothing_steps,
                'optimized': optimize,
                'optimization_passes': optimization_passes if optimize else 0,
                'recombine_to_quads': recombine_to_quads,
                'vertex_count': len(vertices),
                'face_count': len(triangulated_faces),
                'original_element_count': original_face_count,
                'has_cad_face_ids': len(cad_face_ids) > 0,
                'num_cad_faces': tm.metadata.get('num_cad_faces', 0),
            }

            log.info(f" CAD_Mesh [GMSH]: {mesh_type_str} mesh, "
                     f"{len(vertices)} vertices, {len(triangulated_faces)} triangular faces, "
                     f"size: {actual_target_size:.4f} ({size_mode}), algo: {algorithm_2d}")

            diag = _mesh_diagnostics(tm, occ_shape)
            metadata.update(diag)
            metadata.setdefault("size_mode", size_mode)
            metadata.setdefault("target_size", actual_target_size)
            info = _mesh_info_str("GMSH", tm, metadata, diag)
            log.info("[CAD_Mesh] %s", info.replace("\n", " | "))
            return io.NodeOutput(tm, metadata, info, ui={"text": [info]})

        except Exception as e:
            raise RuntimeError(f"GMSH mesh generation failed: {str(e)}")


# =============================================================================
# Axis Transform Helper Functions (for AutoForm IGES files with _axis.igs)
# =============================================================================

def parse_axis_igs(filepath):
    """
    Parse an IGES axis file containing 3 lines that define a coordinate system.

    AutoForm exports coordinate systems as IGES files with 3 Line entities.
    Each line shares the same start point (origin) and the end points define X, Y, Z directions.

    Uses OpenCASCADE to properly read the IGES file instead of regex parsing.

    Args:
        filepath: Path to the _axis.igs file

    Returns:
        dict with 'origin', 'x_dir', 'y_dir', 'z_dir' as (x, y, z) tuples,
        or None if parsing fails
    """
    try:
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods
        import math

        # Read IGES file using OpenCASCADE
        reader = IGESControl_Reader()
        status = reader.ReadFile(str(filepath))

        if status != IFSelect_RetDone:
            log.info(f" Failed to read axis file: {filepath}")
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        # Extract all edges (lines) from the shape
        lines_data = []
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)

        while explorer.More():
            edge = topods.Edge(explorer.Current())

            # Get the curve and its parameter range
            curve, first, last = BRep_Tool.Curve(edge)
            if curve is not None:
                # Get start and end points
                start_pnt = curve.Value(first)
                end_pnt = curve.Value(last)

                lines_data.append({
                    'start': (start_pnt.X(), start_pnt.Y(), start_pnt.Z()),
                    'end': (end_pnt.X(), end_pnt.Y(), end_pnt.Z())
                })

            explorer.Next()

        if len(lines_data) < 3:
            log.warning(f" Expected 3 lines in axis file, found {len(lines_data)}")
            if len(lines_data) > 0:
                log.info(f"   Found lines: {lines_data}")
            return None

        # All lines should share the same origin (start point)
        origin = lines_data[0]['start']
        x_end = lines_data[0]['end']
        y_end = lines_data[1]['end']
        z_end = lines_data[2]['end']

        # Compute direction vectors
        x_dir = (x_end[0] - origin[0], x_end[1] - origin[1], x_end[2] - origin[2])
        y_dir = (y_end[0] - origin[0], y_end[1] - origin[1], y_end[2] - origin[2])
        z_dir = (z_end[0] - origin[0], z_end[1] - origin[1], z_end[2] - origin[2])

        log.info(f" Axis parsed via OCC: origin={origin}")
        log.info(f"   X-dir: {x_dir}")
        log.info(f"   Y-dir: {y_dir}")
        log.info(f"   Z-dir: {z_dir}")

        return {
            'origin': origin,
            'x_dir': x_dir,
            'y_dir': y_dir,
            'z_dir': z_dir
        }

    except Exception as e:
        log.error(f"Error parsing axis file {filepath}", exc_info=True)
        return None


def build_axis_transform(axis_data):
    """
    Build an OpenCASCADE transformation matrix from parsed axis data.

    The transformation moves geometry from the coordinate system defined in the axis file
    to the world coordinate system (origin at 0,0,0 with standard X,Y,Z axes).

    Args:
        axis_data: dict from parse_axis_igs() with 'origin', 'x_dir', 'y_dir', 'z_dir'

    Returns:
        gp_Trsf transformation, or None if creation fails
    """
    try:
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir, gp_Trsf, gp_Vec
        import math

        origin = axis_data['origin']
        x_dir = axis_data['x_dir']
        z_dir = axis_data['z_dir']

        # Normalize direction vectors
        def normalize(v):
            length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            if length < 1e-10:
                return None
            return (v[0]/length, v[1]/length, v[2]/length)

        x_norm = normalize(x_dir)
        z_norm = normalize(z_dir)

        if x_norm is None or z_norm is None:
            log.warning(" Axis vectors have zero length")
            return None

        # Create source coordinate system (from the axis file)
        source_origin = gp_Pnt(origin[0], origin[1], origin[2])
        source_z = gp_Dir(z_norm[0], z_norm[1], z_norm[2])
        source_x = gp_Dir(x_norm[0], x_norm[1], x_norm[2])
        source_ax3 = gp_Ax3(source_origin, source_z, source_x)

        # Target is world coordinates (origin at 0,0,0 with standard axes)
        target_ax3 = gp_Ax3()  # Default: origin=(0,0,0), Z=(0,0,1), X=(1,0,0)

        # Build transformation: source -> target
        trsf = gp_Trsf()
        trsf.SetDisplacement(source_ax3, target_ax3)

        return trsf

    except Exception as e:
        log.info(f" Error building axis transform: {e}")
        return None


def apply_transform_to_shape(shape, trsf):
    """
    Apply a transformation to an OpenCASCADE shape.

    Args:
        shape: TopoDS_Shape to transform
        trsf: gp_Trsf transformation

    Returns:
        Transformed TopoDS_Shape, or original shape if transformation fails
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

        transformer = BRepBuilderAPI_Transform(shape, trsf, True)  # True = copy geometry
        if transformer.IsDone():
            return transformer.Shape()
        else:
            log.warning(" Shape transformation failed")
            return shape
    except Exception as e:
        log.info(f" Error applying transform: {e}")
        return shape


# Minimum file size to consider as a valid CAD file (filters out axis files, etc.)
MIN_CAD_FILE_SIZE_KB = 5



class CAD_Load_From_Glob(io.ComfyNode):
    """
    Load multiple CAD files matching a glob pattern (batch loading).

    The glob_pattern can be:
    - Relative to ComfyUI's input directory (e.g., '**/*.step', 'my_parts/*.stp')
    - An absolute path glob (e.g., '/data/cad/**/*.step')

    Supports single_core (sequential) or parallel execution mode.
    Parallel mode uses subprocesses for crash isolation and OS-level timeout.
    """

    SUPPORTED_EXTENSIONS = ['.step', '.stp', '.iges', '.igs', '.brep']

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CAD_Load_From_Glob",
            display_name="Load CAD From Glob",
            category="CADabra/Loading",
            inputs=[
                io.String.Input("glob_pattern", default="**/*.step",
                                tooltip="Glob pattern for CAD files. Relative patterns resolve from ComfyUI input dir. Absolute paths are used as-is."),
                io.DynamicCombo.Input("execution_mode", tooltip="single_core: sequential loading. parallel: subprocess workers with timeout", options=[
                    io.DynamicCombo.Option("single_core", []),
                    io.DynamicCombo.Option("parallel", [
                        io.Int.Input("num_workers", default=4, min=1, max=32,
                                     tooltip="Number of parallel subprocesses (parallel mode only)"),
                        io.Int.Input("timeout", default=60, min=10, max=600,
                                     tooltip="Timeout per file in seconds (parallel mode only)"),
                    ]),
                ]),
                io.Int.Input("start_index", default=0, min=0, max=100000,
                             tooltip="Skip first N files"),
                io.Int.Input("max_cads", default=-1, min=-1, max=100000,
                             tooltip="Load up to N files (-1 = unlimited)"),
                io.Boolean.Input("recursive", default=True, optional=True,
                                 tooltip="Search subdirectories recursively"),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_models", is_output_list=True,
                                              tooltip="Batch of loaded CAD models"),
            ],
        )

    @classmethod
    def _discover_cad_files(cls, glob_pattern, start_index, max_cads, recursive):
        """Discover CAD files matching a glob pattern. Returns list of cad_files."""
        from pathlib import Path
        import glob as glob_module

        # If pattern starts with / or drive letter, treat as absolute
        # Otherwise, resolve relative to ComfyUI input directory
        if os.path.isabs(glob_pattern):
            base_path = Path("/")
            pattern = glob_pattern
        else:
            base_path = Path(folder_paths.get_input_directory())
            pattern = str(base_path / glob_pattern)

        log.info(f" Glob pattern: {pattern}")

        # Search for CAD files using glob
        if recursive:
            files = sorted(Path(p) for p in glob_module.glob(pattern, recursive=True))
        else:
            files = sorted(Path(p) for p in glob_module.glob(pattern))

        # Filter for valid CAD extensions and exclude small files
        valid_extensions = set(cls.SUPPORTED_EXTENSIONS)
        min_size_bytes = MIN_CAD_FILE_SIZE_KB * 1024

        cad_files = []
        skipped_small = 0
        skipped_axis = 0

        for f in files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in valid_extensions:
                continue
            if '_axis' in f.stem.lower():
                skipped_axis += 1
                continue
            try:
                if f.stat().st_size < min_size_bytes:
                    skipped_small += 1
                    continue
            except OSError:
                continue
            cad_files.append(f)

        if skipped_small > 0 or skipped_axis > 0:
            log.info(f" Skipped {skipped_axis} axis files and {skipped_small} small files (< {MIN_CAD_FILE_SIZE_KB}KB)")

        cad_files.sort()

        if len(cad_files) == 0:
            raise FileNotFoundError(
                f"No CAD files found matching pattern: {pattern}\n"
                f"Recursive: {recursive}\n"
                f"Valid extensions: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        log.info(f" Found {len(cad_files)} CAD files")

        # Apply start_index and max_cads
        if start_index > 0:
            if start_index >= len(cad_files):
                raise ValueError(f"start_index ({start_index}) is >= number of CAD files ({len(cad_files)})")
            cad_files = cad_files[start_index:]
            log.info(f" Skipping first {start_index} files")

        if max_cads > 0:
            cad_files = cad_files[:max_cads]
            log.info(f" Loading up to {max_cads} CAD files")

        return cad_files

    @classmethod
    def _load_sequential(cls, cad_files):
        """Load CAD files sequentially (single_core mode)."""
        try:
            import gmsh
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        # Check if gmsh is initialized
        if not gmsh.is_initialized():
            try:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
            except ValueError as e:
                if "signal only works in main thread" not in str(e):
                    raise

        loaded_cads = []
        for i, cad_file in enumerate(cad_files):
            cad_file_path = str(cad_file)
            ext = cad_file.suffix.lower()

            try:
                # Clear any previous model
                gmsh.model.remove()
                gmsh.model.add(f"cadabra_batch_{i}")

                # Import CAD file
                gmsh.model.occ.importShapes(cad_file_path)
                gmsh.model.occ.synchronize()

                # Also load with pythonocc for OCC-based operations
                occ_shape = None
                axis_applied = False
                try:
                    from OCC.Core.STEPControl import STEPControl_Reader
                    from OCC.Core.IGESControl import IGESControl_Reader
                    from OCC.Core.BRepTools import breptools
                    from OCC.Core.BRep import BRep_Builder
                    from OCC.Core.TopoDS import TopoDS_Shape
                    from OCC.Core.IFSelect import IFSelect_RetDone

                    if ext in ['.step', '.stp']:
                        reader = STEPControl_Reader()
                        status = reader.ReadFile(cad_file_path)
                        if status == IFSelect_RetDone:
                            reader.TransferRoots()
                            occ_shape = reader.OneShape()
                    elif ext in ['.iges', '.igs']:
                        reader = IGESControl_Reader()
                        status = reader.ReadFile(cad_file_path)
                        if status == IFSelect_RetDone:
                            reader.TransferRoots()
                            occ_shape = reader.OneShape()
                    elif ext == '.brep':
                        builder = BRep_Builder()
                        shape = TopoDS_Shape()
                        breptools.Read(shape, cad_file_path, builder)
                        occ_shape = shape

                    # Auto-apply axis transform if matching _axis.igs file exists
                    if occ_shape is not None and ext in ['.iges', '.igs']:
                        axis_path = cad_file.with_name(cad_file.stem + '_axis.igs')
                        if axis_path.exists():
                            axis_data = parse_axis_igs(str(axis_path))
                            if axis_data:
                                trsf = build_axis_transform(axis_data)
                                if trsf:
                                    occ_shape = apply_transform_to_shape(occ_shape, trsf)
                                    axis_applied = True
                                    log.info(f"   Applied axis transform from {axis_path.name}")

                except Exception as occ_error:
                    log.warning(f" Could not load OCC shape for {cad_file.name}: {occ_error}")

                # Save shape to brep file and store path
                brep_path = None
                if occ_shape is not None:
                    try:
                        from .utils.brep_cache import save_shape
                    except ImportError:
                        from .utils.brep_cache import save_shape
                    brep_path = save_shape(occ_shape, f"batch_{i}")

                cad_data = {
                    "file_path": cad_file_path,
                    "brep_path": brep_path,
                    "format": ext,
                    "model_name": f"cadabra_batch_{i}",
                    "axis_transform_applied": axis_applied
                }

                loaded_cads.append(cad_data)
                log.info(f" [{i+1}/{len(cad_files)}] Loaded {cad_file.name}")

            except Exception as e:
                log.warning(f" Failed to load {cad_file.name}: {e}")
                continue

        return loaded_cads

    @classmethod
    def _load_parallel(cls, cad_files, num_workers, timeout):
        """Load CAD files in parallel using subprocesses (parallel mode)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder
        import subprocess
        import tempfile
        import time
        import json
        import sys

        log.info(f"Parallel: Loading {len(cad_files)} files with {num_workers} workers, timeout={timeout}s")

        # Create temp directory for output BREP files
        temp_dir = tempfile.mkdtemp(prefix="cadabra_load_")

        # Create persistent log directory in ComfyUI output folder
        from datetime import datetime
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs", f"load_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        log.info(f"Parallel: Logs: {log_dir}")

        # Path to subprocess script
        script_path = os.path.join(os.path.dirname(__file__), "load_subprocess.py")

        # Prepare work items
        work_items = []
        for idx, f in enumerate(cad_files):
            output_brep = os.path.join(temp_dir, f"output_{idx}.brep")
            result_file = os.path.join(temp_dir, f"result_{idx}.json")
            log_file = os.path.join(log_dir, f"{f.stem}.log")
            work_items.append({
                "idx": idx,
                "input_file": str(f),
                "output_brep": output_brep,
                "result_file": result_file,
                "log_file": log_file,
            })

        def run_load_subprocess(item):
            """Run loading in subprocess with timeout."""
            cmd = [
                sys.executable,
                script_path,
                item["input_file"],
                item["output_brep"],
                f"--result-file={item['result_file']}",
                f"--log-file={item['log_file']}"
            ]
            item_start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5
                )
                elapsed = time.time() - item_start
                if os.path.exists(item["result_file"]):
                    with open(item["result_file"], 'r') as f:
                        return {**json.load(f), "file_path": item["input_file"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                elif result.returncode == 0:
                    return {"success": True, "file_path": item["input_file"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                else:
                    return {"success": False, "error": result.stderr or "Unknown error", "file_path": item["input_file"], "log_file": item["log_file"], "elapsed": elapsed}
            except subprocess.TimeoutExpired:
                elapsed = time.time() - item_start
                return {"success": False, "error": f"Timed out after {timeout}s", "file_path": item["input_file"], "log_file": item["log_file"], "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - item_start
                return {"success": False, "error": str(e), "file_path": item["input_file"], "log_file": item["log_file"], "elapsed": elapsed}

        # Process in parallel using ThreadPoolExecutor
        results = [None] * len(work_items)
        start_time = time.time()
        loaded_cads = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(run_load_subprocess, item): i
                    for i, item in enumerate(work_items)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {"success": False, "error": str(e), "file_path": work_items[idx]["input_file"]}
                    completed += 1
                    elapsed = time.time() - start_time
                    _progress_bar(completed, len(work_items), elapsed, prefix="Loading: ")

            # Build CAD_MODEL dicts from results
            failed = 0
            SLOW_THRESHOLD = 10.0

            for i, result in enumerate(results):
                elapsed_val = result.get('elapsed', 0) if result else 0
                elapsed_str = f"{elapsed_val:.1f}s"
                filename = os.path.basename(result.get("file_path", work_items[i]["input_file"])) if result else os.path.basename(work_items[i]["input_file"])
                log_file = result.get("log_file", work_items[i]["log_file"]) if result else work_items[i]["log_file"]
                log_basename = os.path.basename(log_file)
                show_log = elapsed_val >= SLOW_THRESHOLD

                if result is None or not result.get("success", False):
                    failed += 1
                    err = result.get("error", "Unknown") if result else "No result"
                    log.info(f"Parallel: {filename}: FAILED ({elapsed_str}) - {err} [log: {log_basename}]")
                    continue

                output_brep = result.get("output_brep", work_items[i]["output_brep"])
                if os.path.exists(output_brep):
                    # Use the output BREP file directly as brep_path
                    cad_data = {
                        "brep_path": output_brep,
                        "file_path": result["file_path"],
                        "format": "brep",
                    }
                    if "metadata" in result:
                        cad_data["metadata"] = result["metadata"]

                    loaded_cads.append(cad_data)
                    num_faces = result.get("num_faces", "?")
                    if show_log:
                        log.info(f"Parallel: {filename}: {num_faces} faces ({elapsed_str}) [log: {log_basename}]")
                    else:
                        log.info(f"Parallel: {filename}: {num_faces} faces ({elapsed_str})")
                else:
                    failed += 1
                    log.info(f"Parallel: {filename}: FAILED ({elapsed_str}) - output file missing [log: {log_basename}]")

            log.info(f"Parallel: Successfully loaded {len(loaded_cads)} CAD files ({failed} failed)")

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                log.warning(f"Parallel: Could not clean temp dir: {e}")

        return loaded_cads

    @classmethod
    def execute(cls, glob_pattern, execution_mode, start_index, max_cads, recursive=True):
        """Main dispatch method - routes to sequential or parallel loading."""
        # Extract DynamicCombo params
        if isinstance(execution_mode, dict):
            mode = execution_mode["execution_mode"]
            num_workers = execution_mode.get("num_workers", 4)
            timeout = execution_mode.get("timeout", 60)
        else:
            mode = execution_mode
            num_workers = 4
            timeout = 60

        # Discover CAD files
        cad_files = cls._discover_cad_files(glob_pattern, start_index, max_cads, recursive)

        # Dispatch based on execution mode
        if mode == "parallel":
            loaded_cads = cls._load_parallel(cad_files, num_workers, timeout)
        else:
            loaded_cads = cls._load_sequential(cad_files)

        if len(loaded_cads) == 0:
            raise ValueError(f"Failed to load any CAD files matching pattern: {glob_pattern}")

        log.info(f" Successfully loaded {len(loaded_cads)} CAD files")

        loaded_names = [os.path.basename(cad["file_path"]) for cad in loaded_cads]
        info_lines = [
            f"Pattern: {glob_pattern}",
            f"Mode: {mode}",
            f"Matched: {len(cad_files)}  Loaded: {len(loaded_cads)}  Failed: {len(cad_files) - len(loaded_cads)}",
            "",
        ] + loaded_names
        info = "\n".join(info_lines)

        return io.NodeOutput(loaded_cads, ui={"text": [info]})



class CAD_Quality_Metrics(io.ComfyNode):
    """Analyze CAD model topology and quality metrics"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CAD_Quality_Metrics",
            display_name="CAD Quality Metrics",
            category="CADabra/Analysis",
            description="Analyze CAD topology: watertightness, connectedness, face quality, boundaries",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.Boolean.Input("check_watertightness", default=True, optional=True,
                                 tooltip="Check if model is closed/manifold"),
                io.Boolean.Input("check_degeneracy", default=True, optional=True,
                                 tooltip="Check for degenerate faces (zero area, collapsed edges)"),
                io.Boolean.Input("check_boundaries", default=True, optional=True,
                                 tooltip="Analyze face boundaries and loops"),
                io.Float.Input("area_tolerance", default=1e-6, min=1e-12, max=1.0, step=1e-6, optional=True,
                               tooltip="Minimum area threshold for degenerate face detection"),
            ],
            outputs=[
                io.String.Output(display_name="metrics_dict", tooltip="Metrics as JSON string"),
                io.String.Output(display_name="info", tooltip="Human-readable report"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, check_watertightness=True, check_degeneracy=True,
                check_boundaries=True, area_tolerance=1e-6):
        try:
            import gmsh
            import json
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        metrics = {}

        try:
            # Get all entities from the model
            entities = gmsh.model.getEntities()

            # Count entities by dimension
            volumes = [e for e in entities if e[0] == 3]
            faces = [e for e in entities if e[0] == 2]
            edges = [e for e in entities if e[0] == 1]
            vertices = [e for e in entities if e[0] == 0]

            metrics['topology'] = {
                'num_volumes': len(volumes),
                'num_faces': len(faces),
                'num_edges': len(edges),
                'num_vertices': len(vertices)
            }

            log.info(f" Topology: {len(volumes)} volumes, {len(faces)} faces, "
                  f"{len(edges)} edges, {len(vertices)} vertices")

            # Get bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            metrics['bounding_box'] = {
                'min': [xmin, ymin, zmin],
                'max': [xmax, ymax, zmax],
                'extents': [xmax - xmin, ymax - ymin, zmax - zmin],
                'volume': (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            }

            # Connectedness analysis - count number of separate shells
            if len(volumes) > 0:
                metrics['connectedness'] = {
                    'num_shells': len(volumes),
                    'is_single_solid': len(volumes) == 1
                }
            else:
                # For surface models without volumes
                metrics['connectedness'] = {
                    'num_shells': 'N/A (no volumes)',
                    'is_single_solid': False
                }

            # Watertightness check
            if check_watertightness and len(faces) > 0:
                # Check for orphaned edges (edges not shared by exactly 2 faces)
                orphaned_edges = []
                boundary_edges = []

                for edge in edges:
                    dim, tag = edge
                    # Get adjacent faces
                    upward, _ = gmsh.model.getAdjacencies(dim, tag)

                    if len(upward) < 2:
                        boundary_edges.append(tag)
                    elif len(upward) == 0:
                        orphaned_edges.append(tag)

                metrics['watertightness'] = {
                    'num_boundary_edges': len(boundary_edges),
                    'num_orphaned_edges': len(orphaned_edges),
                    'is_watertight': len(boundary_edges) == 0 and len(orphaned_edges) == 0,
                }

                if len(boundary_edges) > 0:
                    log.info(f" Found {len(boundary_edges)} boundary edges (model has openings)")

            # Face degeneracy check
            if check_degeneracy and len(faces) > 0:
                degenerate_faces = []
                zero_area_faces = []

                for face in faces:
                    dim, tag = face
                    try:
                        # Get face mass properties
                        mass = gmsh.model.occ.getMass(dim, tag)

                        # For a surface (dim=2), mass is actually area
                        if mass < area_tolerance:
                            zero_area_faces.append(tag)
                            degenerate_faces.append({
                                'tag': tag,
                                'area': mass,
                                'reason': 'zero_area'
                            })
                    except Exception as e:
                        log.debug("getMass failed for face tag=%s: %s", tag, e)
                        # If getMass fails, face might be degenerate
                        degenerate_faces.append({
                            'tag': tag,
                            'area': None,
                            'reason': 'mass_computation_failed'
                        })

                metrics['degeneracy'] = {
                    'num_degenerate_faces': len(degenerate_faces),
                    'num_zero_area_faces': len(zero_area_faces),
                    'has_degenerate_faces': len(degenerate_faces) > 0,
                    'area_tolerance': area_tolerance
                }

                if len(degenerate_faces) > 0:
                    log.info(f" Found {len(degenerate_faces)} degenerate faces")
                    metrics['degeneracy']['degenerate_face_details'] = degenerate_faces[:10]  # Limit output

            # Boundary/loop analysis
            if check_boundaries and len(faces) > 0:
                face_loop_counts = []
                faces_with_multiple_loops = []

                for face in faces[:100]:  # Sample first 100 faces to avoid performance issues
                    dim, tag = face
                    try:
                        # Get boundary edges of face
                        downward, _ = gmsh.model.getAdjacencies(dim, tag)

                        # Approximate loop count (proper implementation would trace edge connectivity)
                        # This is a simplified heuristic
                        num_boundary_edges = len(downward)
                        face_loop_counts.append(num_boundary_edges)

                        # Faces with many edges likely have holes
                        if num_boundary_edges > 10:
                            faces_with_multiple_loops.append({
                                'tag': tag,
                                'boundary_edges': num_boundary_edges
                            })
                    except Exception as e:
                        log.debug("Boundary edge analysis failed for face tag=%s: %s", tag, e)

                avg_boundary_edges = np.mean(face_loop_counts) if face_loop_counts else 0

                metrics['boundaries'] = {
                    'avg_boundary_edges_per_face': float(avg_boundary_edges),
                    'num_faces_with_many_edges': len(faces_with_multiple_loops),
                    'faces_analyzed': min(len(faces), 100)
                }

                if len(faces_with_multiple_loops) > 0:
                    log.info(f" Found {len(faces_with_multiple_loops)} faces with complex boundaries")

            # Face duplication check (compare centroids and areas)
            if len(faces) > 1 and len(faces) < 1000:  # Only for reasonably sized models
                centroids = []
                areas = []

                for face in faces:
                    dim, tag = face
                    try:
                        mass = gmsh.model.occ.getMass(dim, tag)
                        com = gmsh.model.occ.getCenterOfMass(dim, tag)
                        centroids.append(com)
                        areas.append(mass)
                    except Exception as e:
                        log.debug("Failed to get mass/centroid for entity (%s, %s): %s", dim, tag, e)

                # Find potential duplicates (same centroid + area within tolerance)
                duplicates = []
                if len(centroids) > 0:
                    centroids = np.array(centroids)
                    areas = np.array(areas)

                    for i in range(len(centroids)):
                        for j in range(i + 1, len(centroids)):
                            dist = np.linalg.norm(centroids[i] - centroids[j])
                            area_diff = abs(areas[i] - areas[j])

                            if dist < 1e-6 and area_diff < area_tolerance:
                                duplicates.append({
                                    'face1': faces[i][1],
                                    'face2': faces[j][1],
                                    'centroid_distance': float(dist),
                                    'area_difference': float(area_diff)
                                })

                metrics['duplication'] = {
                    'num_potential_duplicate_pairs': len(duplicates),
                    'has_duplicates': len(duplicates) > 0
                }

                if len(duplicates) > 0:
                    log.info(f" Found {len(duplicates)} potential duplicate face pairs")
                    metrics['duplication']['duplicate_pairs'] = duplicates[:5]  # Limit output

            # Generate report
            report_lines = ["=" * 60]
            report_lines.append("CAD QUALITY ANALYSIS REPORT")
            report_lines.append("=" * 60)
            report_lines.append("")

            # Topology
            report_lines.append("TOPOLOGY:")
            report_lines.append(f"  Volumes:  {metrics['topology']['num_volumes']}")
            report_lines.append(f"  Faces:    {metrics['topology']['num_faces']}")
            report_lines.append(f"  Edges:    {metrics['topology']['num_edges']}")
            report_lines.append(f"  Vertices: {metrics['topology']['num_vertices']}")
            report_lines.append("")

            # Bounding box
            bb = metrics['bounding_box']
            report_lines.append("BOUNDING BOX:")
            report_lines.append(f"  Min: ({bb['min'][0]:.3f}, {bb['min'][1]:.3f}, {bb['min'][2]:.3f})")
            report_lines.append(f"  Max: ({bb['max'][0]:.3f}, {bb['max'][1]:.3f}, {bb['max'][2]:.3f})")
            report_lines.append(f"  Extents: {bb['extents'][0]:.3f} × {bb['extents'][1]:.3f} × {bb['extents'][2]:.3f}")
            report_lines.append("")

            # Connectedness
            report_lines.append("CONNECTEDNESS:")
            conn = metrics.get('connectedness', {})
            report_lines.append(f"  Number of shells: {conn.get('num_shells', 'N/A')}")
            report_lines.append(f"  Single solid: {'Yes' if conn.get('is_single_solid', False) else 'No'}")
            report_lines.append("")

            # Watertightness
            if 'watertightness' in metrics:
                wt = metrics['watertightness']
                status = "[x] WATERTIGHT" if wt['is_watertight'] else "[ ] NOT WATERTIGHT"
                report_lines.append(f"WATERTIGHTNESS: {status}")
                report_lines.append(f"  Boundary edges: {wt['num_boundary_edges']}")
                report_lines.append(f"  Orphaned edges: {wt['num_orphaned_edges']}")
                report_lines.append("")

            # Degeneracy
            if 'degeneracy' in metrics:
                deg = metrics['degeneracy']
                status = "[x] NO DEGENERATE FACES" if not deg['has_degenerate_faces'] else "[ ] DEGENERATE FACES FOUND"
                report_lines.append(f"DEGENERACY: {status}")
                report_lines.append(f"  Degenerate faces: {deg['num_degenerate_faces']}")
                report_lines.append(f"  Zero-area faces: {deg['num_zero_area_faces']}")
                report_lines.append(f"  Area tolerance: {deg['area_tolerance']:.2e}")
                report_lines.append("")

            # Boundaries
            if 'boundaries' in metrics:
                bnd = metrics['boundaries']
                report_lines.append("BOUNDARIES:")
                report_lines.append(f"  Avg boundary edges/face: {bnd['avg_boundary_edges_per_face']:.2f}")
                report_lines.append(f"  Faces with complex boundaries: {bnd['num_faces_with_many_edges']}")
                report_lines.append(f"  (Analyzed {bnd['faces_analyzed']} faces)")
                report_lines.append("")

            # Duplication
            if 'duplication' in metrics:
                dup = metrics['duplication']
                status = "[x] NO DUPLICATES" if not dup['has_duplicates'] else "[ ] DUPLICATE FACES FOUND"
                report_lines.append(f"DUPLICATION: {status}")
                report_lines.append(f"  Potential duplicate pairs: {dup['num_potential_duplicate_pairs']}")
                report_lines.append("")

            report_lines.append("=" * 60)

            report = "\n".join(report_lines)
            metrics_json = json.dumps(metrics, indent=2)

            log.info(f" Quality analysis complete")

            return io.NodeOutput(metrics_json, report)

        except Exception as e:
            raise RuntimeError(f"Quality analysis failed: {str(e)}")



class PreviewCADOCC(io.ComfyNode):
    """Preview CAD model with VTP tessellation and VTK.js viewer."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewCADOCC",
            display_name="Preview CAD (OCC)",
            category="CADabra/Visualization",
            is_output_node=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.Float.Input("linear_deflection", default=0.1, min=0.001, max=1.0, step=0.01,
                               tooltip="Tessellation quality (smaller = finer mesh)"),
                io.Float.Input("angular_deflection", default=0.5, min=0.1, max=1.0, step=0.1,
                               tooltip="Angular deflection in radians"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, cad_model, linear_deflection=0.1, angular_deflection=0.5):
        """Export CAD to VTP mesh for preview."""
        import folder_paths

        # Get OCC shape from brep_path
        try:
            from .utils.brep_cache import get_occ_shape
        except ImportError:
            from .utils.brep_cache import get_occ_shape
        occ_shape = get_occ_shape(cad_model)

        # Get model info
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.TopoDS import topods

        num_volumes = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_SOLID))
        num_faces = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_FACE))
        num_edges = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_EDGE))

        # Bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(occ_shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = 0, 0, 0, 0, 0, 0

        # Tessellate
        log.info(f" Tessellating (linear: {linear_deflection}, angular: {angular_deflection})")
        with log_operation("BRepMesh", linear_deflection=linear_deflection, angular_deflection=angular_deflection):
            mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection)
            mesh.Perform()

        # Extract mesh with face IDs
        all_vertices = []
        all_indices = []
        face_ranges = []
        vertex_offset = 0
        triangle_start = 0

        explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
        face_idx = 0
        while explorer.More():
            face = topods.Face(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)

            if triangulation is not None:
                trsf = location.Transformation()
                num_nodes = triangulation.NbNodes()
                num_triangles = triangulation.NbTriangles()

                # Extract vertices
                for i in range(1, num_nodes + 1):
                    pnt = triangulation.Node(i).Transformed(trsf)
                    all_vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])

                # Extract triangles
                for i in range(1, num_triangles + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    all_indices.extend([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])

                face_ranges.append({"face_id": face_idx, "triangle_start": triangle_start, "triangle_count": num_triangles})
                vertex_offset += num_nodes
                triangle_start += num_triangles
            else:
                face_ranges.append({"face_id": face_idx, "triangle_start": triangle_start, "triangle_count": 0})

            face_idx += 1
            explorer.Next()

        total_vertices = len(all_vertices) // 3
        total_triangles = len(all_indices) // 3
        log.info(f" Mesh: {total_vertices} vertices, {total_triangles} triangles")

        # Export VTP
        output_dir = folder_paths.get_output_directory()
        base_name = f"preview_{uuid.uuid4().hex[:8]}"

        try:
            import vtk
            vtp_filename = cls._export_vtp(all_vertices, all_indices, face_ranges, base_name, output_dir)
        except ImportError:
            raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

        log.info(f" Entities: {num_volumes} volumes, {num_faces} faces, {num_edges} edges")

        return io.NodeOutput(ui={
            "mesh_file": [vtp_filename],
            "format": ["vtp"],
            "original_format": [cad_model.get("format", "unknown")],
            "num_volumes": [num_volumes],
            "num_faces": [num_faces],
            "num_edges": [num_edges],
            "bounds_min": [[xmin, ymin, zmin]],
            "bounds_max": [[xmax, ymax, zmax]],
            "extents": [[xmax - xmin, ymax - ymin, zmax - zmin]],
            "linear_deflection": [linear_deflection],
        })

    @staticmethod
    def _export_vtp(vertices, indices, face_ranges, base_filename, output_dir):
        """Export mesh as VTP with FaceID cell data."""
        import vtk

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        face_ids = vtk.vtkIntArray()
        face_ids.SetName("FaceID")

        # Add vertices
        for i in range(0, len(vertices), 3):
            points.InsertNextPoint(vertices[i], vertices[i+1], vertices[i+2])

        # Build face ID lookup
        triangle_to_face = {}
        for fr in face_ranges:
            for t in range(fr["triangle_start"], fr["triangle_start"] + fr["triangle_count"]):
                triangle_to_face[t] = fr["face_id"]

        # Add triangles with face IDs
        for tri_idx in range(0, len(indices), 3):
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, indices[tri_idx])
            triangle.GetPointIds().SetId(1, indices[tri_idx + 1])
            triangle.GetPointIds().SetId(2, indices[tri_idx + 2])
            polys.InsertNextCell(triangle)
            face_ids.InsertNextValue(triangle_to_face.get(tri_idx // 3, -1))

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)
        polydata.GetCellData().AddArray(face_ids)

        # Write VTP
        vtp_filename = f"{base_filename}.vtp"
        vtp_path = os.path.join(output_dir, vtp_filename)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_path)
        writer.SetInputData(polydata)
        writer.SetDataModeToAscii()
        writer.Write()

        file_size = os.path.getsize(vtp_path)
        log.info(f" Exported VTP: {vtp_filename} ({file_size:,} bytes)")
        return vtp_filename


def tessellate_occ_shape(occ_shape, linear_deflection=0.1, angular_deflection=0.5):
    """
    Tessellate an OCC shape using BRepMesh and return vertices/faces arrays.

    Args:
        occ_shape: OpenCASCADE TopoDS_Shape object
        linear_deflection: Max distance between mesh and true surface
        angular_deflection: Max angle deviation in radians

    Returns:
        tuple: (vertices array, faces array)
    """
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    # Perform tessellation
    with log_operation("BRepMesh", linear_deflection=linear_deflection, angular_deflection=angular_deflection):
        mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection)
        mesh.Perform()

    if not mesh.IsDone():
        raise RuntimeError("BRepMesh tessellation failed")

    # Extract triangulated mesh from all faces
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is not None:
            trsf = location.Transformation()

            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(trsf)
                all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                all_faces.append([
                    vertex_offset + n1 - 1,
                    vertex_offset + n2 - 1,
                    vertex_offset + n3 - 1
                ])

            vertex_offset += triangulation.NbNodes()

        explorer.Next()

    return np.array(all_vertices, dtype=np.float32), np.array(all_faces, dtype=np.int32)


def trimesh_to_glb(trimesh_obj, output_path):
    """
    Export a trimesh object to GLB format.

    Args:
        trimesh_obj: trimesh.Trimesh object
        output_path: Path to save the GLB file

    Returns:
        tuple: (vertex_count, face_count)
    """
    from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
    from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
    from pygltflib import FLOAT, UNSIGNED_INT
    from pygltflib import TRIANGLES

    vertices_array = np.array(trimesh_obj.vertices, dtype=np.float32)
    faces_array = np.array(trimesh_obj.faces, dtype=np.int32)

    indices_flat = faces_array.flatten().astype(np.uint32)
    vertices_bytes = vertices_array.tobytes()
    indices_bytes = indices_flat.tobytes()
    buffer_data = vertices_bytes + indices_bytes

    gltf = GLTF2()
    gltf.scenes = [Scene(nodes=[0])]
    gltf.nodes = [Node(mesh=0)]
    gltf.meshes = [Mesh(primitives=[Primitive(
        attributes={"POSITION": 0},
        indices=1,
        mode=TRIANGLES
    )])]

    gltf.buffers = [Buffer(byteLength=len(buffer_data))]

    vertices_offset = 0
    indices_offset = len(vertices_bytes)

    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=vertices_offset, byteLength=len(vertices_bytes), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=indices_offset, byteLength=len(indices_bytes), target=ELEMENT_ARRAY_BUFFER)
    ]

    gltf.accessors = [
        Accessor(bufferView=0, componentType=FLOAT, count=len(vertices_array), type="VEC3",
                min=vertices_array.min(axis=0).tolist(), max=vertices_array.max(axis=0).tolist()),
        Accessor(bufferView=1, componentType=UNSIGNED_INT, count=len(indices_flat), type="SCALAR")
    ]

    gltf.set_binary_blob(buffer_data)
    gltf.save(output_path)

    return len(vertices_array), len(faces_array)



class PreviewCADBatch(io.ComfyNode):
    """Preview batch of CAD models or TRIMESH objects with VTK.js viewer and index navigation.

    Accepts either CAD_MODEL or TRIMESH as input (but not both).
    - If CAD_MODEL: Tessellates using BRepMesh and exports to VTP
    - If TRIMESH: Converts to VTP for viewing
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewCADBatch",
            display_name="Preview CAD Batch",
            category="CADabra/Visualization",
            is_output_node=True,
            is_input_list=True,
            inputs=[
                io.Int.Input("index", default=0, min=0, max=100),
                io.Custom("CAD_MODEL").Input("cad_model", optional=True),
                io.Custom("TRIMESH").Input("trimesh", optional=True),
                io.Float.Input("linear_deflection", default=0.1, min=0.001, max=1.0, step=0.01, optional=True,
                               tooltip="Tessellation quality (smaller = finer mesh)"),
                io.Float.Input("angular_deflection", default=0.5, min=0.1, max=1.0, step=0.1, optional=True,
                               tooltip="Angular deflection in radians"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, index, cad_model=None, trimesh=None, linear_deflection=None, angular_deflection=None):
        """Preview CAD model or TRIMESH from batch with navigation controls."""
        import folder_paths

        # Extract values from lists (ComfyUI passes inputs as lists when INPUT_IS_LIST=True)
        index_val = index[0] if isinstance(index, list) else index
        linear_deflection_val = (linear_deflection[0] if isinstance(linear_deflection, list) else linear_deflection) if linear_deflection else 0.1
        angular_deflection_val = (angular_deflection[0] if isinstance(angular_deflection, list) else angular_deflection) if angular_deflection else 0.5

        # Determine input type and validate
        has_cad = cad_model is not None and len(cad_model) > 0
        has_trimesh = trimesh is not None and len(trimesh) > 0

        if has_cad and has_trimesh:
            raise ValueError("Cannot provide both cad_model and trimesh inputs. Please connect only one.")

        if not has_cad and not has_trimesh:
            raise ValueError("Must provide either cad_model or trimesh input.")

        # Route to appropriate handler
        if has_trimesh:
            return cls._preview_trimesh_batch(trimesh, index_val)
        else:
            return cls._preview_cad_batch(cad_model, index_val, linear_deflection_val, angular_deflection_val)

    @classmethod
    def _preview_trimesh_batch(cls, trimesh_list, index_val):
        """Handle TRIMESH input - convert to VTP."""
        import folder_paths

        # Validate batch
        batch_size = len(trimesh_list)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current mesh from batch
        current_mesh = trimesh_list[actual_index]

        log.info(f"Batch: TRIMESH mode - Batch size: {batch_size}, showing index: {actual_index + 1}/{batch_size}")

        # Generate unique filename
        base_name = f"preview_batch_{uuid.uuid4().hex[:8]}"
        output_dir = folder_paths.get_output_directory()

        # Get mesh bounds
        bounds = current_mesh.bounds
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]

        # Export to VTP
        try:
            import vtk
            vtp_filename = cls._export_trimesh_vtp(current_mesh, base_name, output_dir)
        except ImportError:
            raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

        mesh_vertex_count = len(current_mesh.vertices)
        mesh_face_count = len(current_mesh.faces)
        log.info(f"Batch: Exported TRIMESH to VTP: {vtp_filename} ({mesh_vertex_count} vertices, {mesh_face_count} faces)")

        # Build UI data dictionary
        ui_data = {
            "mesh_file": [vtp_filename],
            "format": ["vtp"],
            "original_format": ["trimesh"],
            "num_volumes": [0],
            "num_faces": [mesh_face_count],
            "num_edges": [0],
            "bounds_min": [[xmin, ymin, zmin]],
            "bounds_max": [[xmax, ymax, zmax]],
            "extents": [[xmax - xmin, ymax - ymin, zmax - zmin]],
            "linear_deflection": [0],
            "batch_size": [batch_size],
            "current_index": [actual_index],
        }

        return io.NodeOutput(ui=ui_data)

    @staticmethod
    def _export_trimesh_vtp(mesh, base_filename, output_dir):
        """Export trimesh to VTP format."""
        import vtk

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        # Add vertices
        for v in mesh.vertices:
            points.InsertNextPoint(v[0], v[1], v[2])

        # Add triangles
        for face in mesh.faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, int(face[0]))
            triangle.GetPointIds().SetId(1, int(face[1]))
            triangle.GetPointIds().SetId(2, int(face[2]))
            polys.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        vtp_filename = f"{base_filename}.vtp"
        vtp_filepath = os.path.join(output_dir, vtp_filename)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_filepath)
        writer.SetInputData(polydata)
        writer.Write()

        return vtp_filename

    @classmethod
    def _preview_cad_batch(cls, cad_model, index_val, linear_deflection_val, angular_deflection_val):
        """Handle CAD_MODEL input - tessellate and export to VTP."""
        import folder_paths

        # Validate batch
        batch_size = len(cad_model)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current model from batch
        current_model = cad_model[actual_index]

        log.info(f"Batch: CAD_MODEL mode - Batch size: {batch_size}, showing index: {actual_index + 1}/{batch_size}")

        # Generate unique filenames
        base_name = f"preview_batch_{uuid.uuid4().hex[:8]}"
        output_dir = folder_paths.get_output_directory()

        try:
            # Get OCC shape from brep_path
            try:
                from .utils.brep_cache import get_occ_shape
            except ImportError:
                from .utils.brep_cache import get_occ_shape
            occ_shape = get_occ_shape(current_model)

            # Get model info for metadata using OCC
            from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods

            num_volumes = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_SOLID))
            num_faces = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_FACE))
            num_edges = sum(1 for _ in _iter_occ_explorer(occ_shape, TopAbs_EDGE))

            # Get bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(occ_shape, bbox)
            if not bbox.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            else:
                xmin, ymin, zmin, xmax, ymax, zmax = 0, 0, 0, 0, 0, 0

            # Tessellate
            log.info(f"Batch: Tessellating (linear: {linear_deflection_val}, angular: {angular_deflection_val})")
            with log_operation("BRepMesh", linear_deflection=linear_deflection_val, angular_deflection=angular_deflection_val):
                mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection_val, False, angular_deflection_val)
                mesh.Perform()

            # Extract mesh with face IDs
            all_vertices = []
            all_indices = []
            face_ranges = []
            vertex_offset = 0
            triangle_start = 0

            explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
            face_idx = 0
            while explorer.More():
                face = topods.Face(explorer.Current())
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)

                if triangulation is not None:
                    trsf = location.Transformation()
                    num_nodes = triangulation.NbNodes()
                    num_triangles = triangulation.NbTriangles()

                    # Extract vertices
                    for i in range(1, num_nodes + 1):
                        pnt = triangulation.Node(i).Transformed(trsf)
                        all_vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])

                    # Extract triangles
                    for i in range(1, num_triangles + 1):
                        tri = triangulation.Triangle(i)
                        n1, n2, n3 = tri.Get()
                        all_indices.extend([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])

                    face_ranges.append({"face_id": face_idx, "triangle_start": triangle_start, "triangle_count": num_triangles})
                    vertex_offset += num_nodes
                    triangle_start += num_triangles
                else:
                    face_ranges.append({"face_id": face_idx, "triangle_start": triangle_start, "triangle_count": 0})

                face_idx += 1
                explorer.Next()

            total_vertices = len(all_vertices) // 3
            total_triangles = len(all_indices) // 3
            log.info(f"Batch: Mesh: {total_vertices} vertices, {total_triangles} triangles")

            # Export VTP
            try:
                import vtk
                vtp_filename = cls._export_vtp(all_vertices, all_indices, face_ranges, base_name, output_dir)
            except ImportError:
                raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

            log.info(f"Batch: Entities: {num_volumes} volumes, {num_faces} faces, {num_edges} edges")

            # Build UI data dictionary
            ui_data = {
                "mesh_file": [vtp_filename],
                "format": ["vtp"],
                "original_format": [current_model.get("format", "unknown")],
                "num_volumes": [num_volumes],
                "num_faces": [num_faces],
                "num_edges": [num_edges],
                "bounds_min": [[xmin, ymin, zmin]],
                "bounds_max": [[xmax, ymax, zmax]],
                "extents": [[xmax - xmin, ymax - ymin, zmax - zmin]],
                "linear_deflection": [linear_deflection_val],
                "batch_size": [batch_size],
                "current_index": [actual_index],
            }

            return io.NodeOutput(ui=ui_data)

        except Exception as e:
            raise RuntimeError(f"Failed to export CAD for batch preview: {str(e)}")

    @staticmethod
    def _export_vtp(vertices, indices, face_ranges, base_filename, output_dir):
        """Export mesh as VTP with FaceID cell data."""
        import vtk

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        face_ids = vtk.vtkIntArray()
        face_ids.SetName("FaceID")

        # Add vertices
        for i in range(0, len(vertices), 3):
            points.InsertNextPoint(vertices[i], vertices[i+1], vertices[i+2])

        # Build face ID lookup
        triangle_to_face = {}
        for fr in face_ranges:
            for t in range(fr["triangle_start"], fr["triangle_start"] + fr["triangle_count"]):
                triangle_to_face[t] = fr["face_id"]

        # Add triangles with face IDs
        for tri_idx in range(0, len(indices), 3):
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, indices[tri_idx])
            triangle.GetPointIds().SetId(1, indices[tri_idx + 1])
            triangle.GetPointIds().SetId(2, indices[tri_idx + 2])
            polys.InsertNextCell(triangle)

            face_id = triangle_to_face.get(tri_idx // 3, -1)
            face_ids.InsertNextValue(face_id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)
        polydata.GetCellData().AddArray(face_ids)

        vtp_filename = f"{base_filename}.vtp"
        vtp_filepath = os.path.join(output_dir, vtp_filename)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_filepath)
        writer.SetInputData(polydata)
        writer.Write()

        return vtp_filename

# Node registration
NODE_CLASS_MAPPINGS = {
    "CAD_Load": CAD_Load,
    "CAD_Load_Path": CAD_Load_Path,
    "CAD_Load_From_Glob": CAD_Load_From_Glob,
    "CAD_Mesh": CAD_Mesh,
    "CAD_Quality_Metrics": CAD_Quality_Metrics,
    "PreviewCADOCC": PreviewCADOCC,
    "PreviewCADBatch": PreviewCADBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_Load": "Load CAD",
    "CAD_Load_Path": "Load CAD (Path)",
    "CAD_Load_From_Glob": "Load CADs from Glob",
    "CAD_Mesh": "Mesh CAD",
    "CAD_Quality_Metrics": "CAD Quality Metrics",
    "PreviewCADOCC": "Preview CAD",
    "PreviewCADBatch": "Preview CAD Batch",
}
