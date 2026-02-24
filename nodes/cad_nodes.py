"""
ComfyUI-CADabra Nodes
CAD file loading, meshing, and ML-based processing nodes
"""

import os
import sys
import tempfile
import uuid
import numpy as np
import torch
import folder_paths
import trimesh

from ..utils.occ_logging import log_operation


def _progress_bar(completed, total, elapsed, width=30, prefix=""):
    """Print a tqdm-style progress bar."""
    if total == 0:
        return
    pct = completed / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
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


def _extract_iges_metadata(reader):
    """
    Extract metadata from IGES Global Section.

    Args:
        reader: IGESControl_Reader after ReadFile() but before TransferRoots()

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
            print("[CADabra] Warning: Could not downcast to IGESData_IGESModel")
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
        print(f"[CADabra] Warning: Could not extract IGES metadata: {e}")
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

        # Extract IGES metadata from Global Section before transfer
        metadata = _extract_iges_metadata(reader)

        with log_operation("IGES TransferRoots", file=filename):
            reader.TransferRoots()
        occ_shape = reader.OneShape()
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


class CAD_Load:
    """Load CAD files (STEP, IGES, BREP) using OpenCASCADE with BREP caching."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get CAD files from input/cad directory
        input_dir = folder_paths.get_input_directory()
        cad_dir = os.path.join(input_dir, 'cad')

        cad_files = []
        if os.path.exists(cad_dir) and os.path.isdir(cad_dir):
            cad_extensions = ['.step', '.stp', '.iges', '.igs', '.brep']
            for filename in os.listdir(cad_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in cad_extensions:
                    cad_files.append(filename)
            cad_files.sort()

        # If no files found, provide a placeholder
        if not cad_files:
            cad_files = ["(no CAD files found in input/cad)"]

        return {
            "required": {
                "filename": (cad_files, {
                    "tooltip": "Select CAD file from input/cad folder"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL",)
    FUNCTION = "load_cad"
    CATEGORY = "CADabra"

    def load_cad(self, filename):
        import hashlib

        # Check if placeholder (no files found)
        if filename.startswith("(no CAD files found"):
            raise FileNotFoundError(
                "No CAD files found in input/cad folder. "
                "Please add .step, .stp, .iges, .igs, or .brep files to ComfyUI/input/cad/"
            )

        # Build path to the CAD file
        input_dir = folder_paths.get_input_directory()
        cad_dir = os.path.join(input_dir, 'cad')
        cad_file_path = os.path.join(cad_dir, filename)

        # Verify file exists
        if not os.path.exists(cad_file_path):
            raise FileNotFoundError(f"CAD file not found: {cad_file_path}")

        ext = os.path.splitext(cad_file_path)[1].lower()
        metadata = {}

        # BREP cache for STEP/IGES files (avoids expensive TransferRoots on reload)
        if ext in ['.step', '.stp', '.iges', '.igs']:
            cache_dir = os.path.join(cad_dir, '.brep_cache')
            os.makedirs(cache_dir, exist_ok=True)

            # Cache key = filename + modification time (auto-invalidates on file change)
            mtime = os.path.getmtime(cad_file_path)
            cache_key = hashlib.md5(f"{filename}_{mtime}".encode()).hexdigest()
            cache_brep = os.path.join(cache_dir, f"{cache_key}.brep")
            cache_meta = os.path.join(cache_dir, f"{cache_key}.json")

            if os.path.exists(cache_brep):
                # FAST PATH: Load from BREP cache
                print(f"[CADabra] Loading from cache: {filename}")
                occ_shape = _load_brep(cache_brep)
                metadata = _load_cache_metadata(cache_meta)
            else:
                # SLOW PATH: Load original, create cache
                print(f"[CADabra] Loading CAD file: {filename}")
                occ_shape, metadata = _load_step_or_iges(cad_file_path, ext, filename)
                _save_brep(occ_shape, cache_brep)
                if metadata:
                    _save_cache_metadata(metadata, cache_meta)
                print(f"[CADabra] Cached to BREP: {filename}")
        else:
            # BREP files - load directly (no caching needed, already native format)
            print(f"[CADabra] Loading BREP file: {filename}")
            occ_shape = _load_brep(cad_file_path)

        if occ_shape is None:
            raise RuntimeError(f"Failed to load CAD shape from: {cad_file_path}")

        # Build CAD_MODEL return dict
        cad_data = {
            "occ_shape": occ_shape,
            "file_path": cad_file_path,
            "format": ext,
        }

        if metadata:
            cad_data["metadata"] = metadata

        print(f"[CADabra] Loaded: {filename}")
        return (cad_data,)


class CAD_Mesh_Gmsh:
    """Generate mesh from CAD model using Gmsh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "mesh_type": (["2D", "3D"],),
                "element_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Element size (interpretation depends on size_mode)"
                }),
                "size_mode": (["Relative to Bounds", "Absolute"],{
                    "tooltip": "Relative: element_size is fraction of model diagonal (0.05 = 5%). Absolute: element_size in model units (mm, m, etc.)"
                }),
                "algorithm": (["Automatic", "Delaunay", "Frontal"],),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESH_METADATA")
    FUNCTION = "generate_mesh"
    CATEGORY = "CADabra"

    def generate_mesh(self, cad_model, mesh_type, element_size, size_mode, algorithm):
        try:
            import gmsh
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        algo_map = {"Automatic": 1, "Delaunay": 5, "Frontal": 6}

        try:
            # Load OCC shape into GMSH for meshing
            occ_shape = cad_model.get("occ_shape")
            if occ_shape is None:
                raise RuntimeError("CAD model has no OCC shape")
            _load_occ_shape_to_gmsh(occ_shape, "mesh_model")

            # Calculate actual element size based on mode
            if size_mode == "Relative to Bounds":
                # Get model bounding box to compute diagonal
                try:
                    bbox = gmsh.model.getBoundingBox(-1, -1)  # Get overall bounding box
                    xmin, ymin, zmin, xmax, ymax, zmax = bbox
                    diagonal = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
                    actual_element_size = diagonal * element_size
                    print(f"[CADabra] Model bounds: [{xmin:.2f}, {ymin:.2f}, {zmin:.2f}] to [{xmax:.2f}, {ymax:.2f}, {zmax:.2f}]")
                    print(f"[CADabra] Diagonal: {diagonal:.2f}, Relative size: {element_size:.3f} -> Actual: {actual_element_size:.4f}")
                except Exception as e:
                    print(f"[CADabra] Warning: Could not get bounding box ({e}), using absolute size")
                    actual_element_size = element_size
            else:
                actual_element_size = element_size
                print(f"[CADabra] Using absolute element size: {actual_element_size}")

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_element_size)
            gmsh.option.setNumber("Mesh.Algorithm", algo_map[algorithm])

            dim = 2 if mesh_type == "2D" else 3
            with log_operation("GMSH mesh.generate", dim=dim, element_size=actual_element_size, algorithm=algorithm):
                gmsh.model.mesh.generate(dim)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)

            if mesh_type == "2D":
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
            else:
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)

            if len(elem_node_tags) == 0:
                raise RuntimeError("Mesh generation produced no elements")

            faces = elem_node_tags[0].reshape(-1, 3 if mesh_type == "2D" else 4) - 1

            # Convert faces to triangles if needed (for 3D meshes with tets, extract surface)
            if mesh_type == "3D" and faces.shape[1] == 4:
                # For tetrahedra, extract triangular surface faces
                # Each tet has 4 triangular faces
                triangles = []
                for tet in faces:
                    triangles.append([tet[0], tet[1], tet[2]])
                    triangles.append([tet[0], tet[1], tet[3]])
                    triangles.append([tet[0], tet[2], tet[3]])
                    triangles.append([tet[1], tet[2], tet[3]])
                faces = np.array(triangles)
                print(f"  Converted 3D tets to surface triangles: {len(faces)} faces")

            # Create trimesh object
            tm = trimesh.Trimesh(
                vertices=vertices,
                faces=faces
            )

            # Store CADabra-specific metadata in trimesh.metadata
            tm.metadata['cadabra_type'] = mesh_type
            tm.metadata['element_size'] = actual_element_size
            tm.metadata['element_size_input'] = element_size
            tm.metadata['size_mode'] = size_mode
            tm.metadata['algorithm'] = algorithm

            # Create separate metadata dict for explicit output
            metadata = {
                'type': mesh_type,
                'element_size': actual_element_size,
                'element_size_input': element_size,
                'size_mode': size_mode,
                'algorithm': algorithm,
                'vertex_count': len(vertices),
                'face_count': len(faces),
            }

            print(f"[CADabra] Generated {mesh_type} mesh: {len(vertices)} vertices, {len(faces)} elements (size: {actual_element_size:.4f})")
            # Don't finalize gmsh - keep it alive for next operation
            return (tm, metadata)

        except Exception as e:
            # Don't finalize gmsh - keep it alive for next operation
            raise RuntimeError(f"Mesh generation failed: {str(e)}")


class CAD_Mesh_Gmsh_Advanced:
    """Advanced mesh generation with size fields, refinement, and extended algorithms"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "output_dim": (["2D Surface", "3D Volume"], {
                    "tooltip": "2D Surface: mesh only surfaces (triangles/quads). 3D Volume: mesh interior with tetrahedra/hexahedra"
                }),
                "algorithm_2d": ([
                    "Delaunay",
                    "Automatic",
                    "MeshAdapt",
                    "Frontal-Delaunay",
                    "BAMG",
                    "Frontal-Delaunay for Quads",
                    "Packing of Parallelograms",
                    "Quasi-structured Quad"
                ], {
                    "tooltip": "2D surface meshing algorithm (always used - surfaces meshed first even for 3D). Automatic: Delaunay for flat, MeshAdapt for curved. MeshAdapt: most robust for complex curves. Delaunay: fastest for large meshes. Frontal-Delaunay: highest quality triangles. BAMG: anisotropic. Quad variants: for quad meshes"
                }),
                "size_mode": (["Absolute", "Relative to Bounds", "Curvature Adaptive"], {
                    "tooltip": "Relative: sizes as fraction of model diagonal. Absolute: sizes in model units. Curvature: adapts element size to geometry curvature"
                }),
            },
            "optional": {
                # 3D Algorithm - FIRST in optional so it appears right after algorithm_2d (shown/hidden by JS)
                "algorithm_3d": (["Delaunay", "Frontal (Netgen)", "HXT", "MMG3D"], {
                    "default": "Delaunay",
                    "tooltip": "3D volume meshing algorithm. Delaunay: most robust, supports size fields. HXT: fastest (parallel) for large meshes. Frontal (Netgen): high quality near boundaries. MMG3D: anisotropic tetrahedra (experimental)"
                }),

                # Target size - in optional because it's hidden in Curvature Adaptive mode
                "target_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0001,
                    "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Target element size. For Relative mode: 0.05 = 5% of model diagonal"
                }),

                # Size controls
                "size_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Global multiplier for all mesh sizes (Mesh.MeshSizeFactor). Values <1 make mesh finer, >1 coarser"
                }),
                "min_size_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum size as fraction of target (e.g., 0.1 = min is 10% of target size)"
                }),
                "max_size_factor": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Maximum size as multiple of target (e.g., 5.0 = max is 5x target size)"
                }),

                # Curvature Adaptive Settings - only for Curvature Adaptive mode
                "elements_per_2pi": ("INT", {
                    "default": 20,
                    "min": 4,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of mesh elements per 2*PI radians of curvature. Higher = finer mesh on curved surfaces"
                }),
                "extend_from_boundary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "(3D Volume + Curvature Adaptive only) Extend curvature-based size field from boundary surfaces into volume interior"
                }),

                # Quality Settings
                "smoothing_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Elliptic smoother iterations for mesh regularity (Mesh.Smoothing). 0 = disabled"
                }),
                "optimize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run mesh optimization after generation to improve element quality"
                }),
                "optimization_passes": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of optimization passes. More passes = better quality but slower"
                }),

                # Recombination
                "recombine_to_quads": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Recombine triangles into quadrangles where possible (Mesh.RecombineAll)"
                }),
                "subdivision": (["None", "All Triangles", "All Quadrangles", "Barycentric"], {
                    "tooltip": "Mesh refinement by subdivision (splits elements). All Triangles: refine into smaller triangles. All Quads: convert to quad-dominant mesh. Barycentric: split by adding centroid vertex. This increases mesh density, not quality"
                }),

                # Advanced
                "element_order": (["1", "2"], {
                    "tooltip": "Element order. 1 = linear elements. 2 = quadratic elements with mid-edge nodes"
                }),
                "extract_face_ids": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract CAD face IDs for each mesh triangle. Stores which CAD surface each mesh element came from as 'cad_face_id' face attribute"
                }),
                "gmsh_options": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Advanced Gmsh options as JSON, e.g. {\"Mesh.OptimizeNetgen\": 1}"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESH_METADATA")
    FUNCTION = "generate_mesh"
    CATEGORY = "CADabra/Advanced"

    def generate_mesh(self, cad_model, output_dim, algorithm_2d, size_mode,
                     target_size=1.0, algorithm_3d="Delaunay", size_factor=1.0,
                     min_size_factor=1.0, max_size_factor=10.0,
                     elements_per_2pi=20, extend_from_boundary=True,
                     smoothing_steps=0, optimize=False, optimization_passes=1,
                     recombine_to_quads=False, subdivision="None",
                     element_order="1", extract_face_ids=True, gmsh_options="{}"):
        try:
            import gmsh
            import json
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        # Corrected 2D algorithm mappings (from Gmsh docs)
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

        # Corrected 3D algorithm mappings (from Gmsh docs)
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
            # Load OCC shape into GMSH for meshing
            occ_shape = cad_model.get("occ_shape")
            if occ_shape is None:
                raise RuntimeError("CAD model has no OCC shape")
            _load_occ_shape_to_gmsh(occ_shape, "mesh_model_advanced")

            # Get model bounding box for size calculations
            try:
                bbox = gmsh.model.getBoundingBox(-1, -1)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox
                diagonal = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
                print(f"[CADabra] Model diagonal: {diagonal:.2f}")
            except Exception as e:
                print(f"[CADabra] Warning: Could not get bounding box ({e})")
                diagonal = 1.0  # Fallback

            # Calculate actual sizes based on mode
            if size_mode == "Relative to Bounds":
                actual_target_size = diagonal * target_size
                actual_min_size = actual_target_size * min_size_factor
                actual_max_size = actual_target_size * max_size_factor
                print(f"[CADabra] Relative sizing: target={target_size:.3f} * diagonal -> {actual_target_size:.4f}")

                # Apply global size factor (Mesh.MeshSizeFactor)
                gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
                # Apply size constraints
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", actual_min_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_max_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            elif size_mode == "Absolute":
                # Check if target size is dangerously small (< 0.1% of diagonal)
                size_ratio = target_size / diagonal
                if size_ratio < 0.001:  # 0.1%
                    raise ValueError(
                        f"Target size {target_size} is too small for this model (diagonal={diagonal:.1f}). "
                        f"Size is {size_ratio*100:.3f}% of diagonal - this would create an extremely dense mesh "
                        f"and likely hang. Minimum recommended size is {diagonal * 0.001:.1f} (0.1% of diagonal). "
                        f"Consider using 'Relative to Bounds' mode or increasing target_size."
                    )

                actual_target_size = target_size
                actual_min_size = actual_target_size * min_size_factor
                actual_max_size = actual_target_size * max_size_factor
                print(f"[CADabra] Absolute sizing: target={actual_target_size}")

                # Apply global size factor (Mesh.MeshSizeFactor)
                gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
                # Apply size constraints
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", actual_min_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", actual_max_size)
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            elif size_mode == "Curvature Adaptive":
                # Curvature mode: size is determined ONLY by geometry curvature
                # Do NOT use target_size - let Gmsh determine sizes automatically
                print(f"[CADabra] Curvature adaptive: elements_per_2pi={elements_per_2pi}, extend_from_boundary={extend_from_boundary}")

                # Enable curvature-based sizing
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
                gmsh.option.setNumber("Mesh.MinimumCirclePoints", elements_per_2pi)

                # Set very permissive size bounds - let curvature control everything
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", diagonal * 0.0001)  # Very small min
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", diagonal * 10.0)    # Very large max

                if extend_from_boundary:
                    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
                else:
                    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

                # Don't use size factor in curvature mode
                gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)

                # Set placeholder values for metadata (not actually used)
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

            # Set 2D algorithm (always used - surfaces are meshed first even for 3D)
            gmsh.option.setNumber("Mesh.Algorithm", algo_2d_map.get(algorithm_2d, 2))

            # Set 3D algorithm only if generating 3D mesh
            is_3d = output_dim == "3D Volume"
            if is_3d:
                gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d_map.get(algorithm_3d, 1))

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
                            print(f"[CADabra] Applied gmsh option: {key} = {value}")
                        elif isinstance(value, str):
                            gmsh.option.setString(key, value)
                            print(f"[CADabra] Applied gmsh option: {key} = '{value}'")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in gmsh_options: {e}")

            # Set up Gmsh logging to file (named after CAD file)
            cad_file_path = cad_model.get("file_path", "")
            if cad_file_path:
                cad_name = os.path.splitext(os.path.basename(cad_file_path))[0]
            else:
                cad_name = "unknown"
            log_file = os.path.join(folder_paths.get_output_directory(), f"gmsh_mesh_{cad_name}.log")
            gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
            gmsh.option.setNumber("General.Verbosity", 99)  # Maximum verbosity
            gmsh.logger.start()  # Start capturing logs

            print(f"[CADabra] Starting mesh generation (log: {log_file})...")

            # Generate mesh
            dim = 3 if is_3d else 2
            with log_operation("GMSH mesh.generate", dim=dim, algorithm_2d=algorithm_2d):
                gmsh.model.mesh.generate(dim)

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

            print(f"[CADabra] Mesh generation complete. Log saved to: {log_file}")

            # Apply mesh optimization if requested
            if optimize and optimization_passes > 0:
                for i in range(optimization_passes):
                    gmsh.model.mesh.optimize("Laplace2D" if not is_3d else "Netgen")
                print(f"[CADabra] Applied {optimization_passes} optimization passes")

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

            # Helper function to triangulate a face and return the count of triangles created
            def triangulate_face(face, is_3d_volume):
                triangles = []
                if len(face) == 3:
                    triangles.append(list(face))
                elif len(face) == 4:
                    triangles.append([face[0], face[1], face[2]])
                    triangles.append([face[0], face[2], face[3]])
                elif len(face) > 4:
                    if not is_3d_volume:
                        # 2D higher-order: extract corner nodes
                        triangles.append([face[0], face[1], face[2]])
                        if len(face) == 9:  # Higher-order quad
                            triangles.append([face[0], face[2], face[3]])
                    else:
                        # 3D volume: extract tet surface faces
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
                # For 2D surface meshing: get elements per CAD surface entity
                surface_entities = gmsh.model.getEntities(dim=2)
                print(f"[CADabra] Extracting face IDs from {len(surface_entities)} CAD surfaces")

                for cad_face_idx, (dim, tag) in enumerate(surface_entities):
                    # Get mesh elements for this specific CAD surface
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

                print(f"[CADabra] Extracted {len(cad_face_ids)} triangles with CAD face IDs")

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
                print(f"  Converted {original_face_count} elements to {len(triangulated_faces)} triangles")

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
                print(f"[CADabra] Stored cad_face_id attribute ({len(surface_entities)} CAD faces)")
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

            print(f"[CADabra] Generated {mesh_type_str} mesh (Advanced): "
                  f"{len(vertices)} vertices, {len(triangulated_faces)} triangular faces, "
                  f"size: {actual_target_size:.4f} ({size_mode}), algo: {algorithm_2d}")

            return (tm, metadata)

        except Exception as e:
            raise RuntimeError(f"Advanced mesh generation failed: {str(e)}")


class ML_SurfaceRecon:
    """ML-based surface reconstruction (stub for future implementation)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "model_name": ([
                    "hustvl/surface-recon",
                    "threedvision/point2surf",
                    "zju3dv/patchnetsurface",
                    "kaist-3d/point2shape"
                ],),
                "smoothness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SURFACE",)
    FUNCTION = "reconstruct_surface"
    CATEGORY = "CADabra/ML"

    def reconstruct_surface(self, mesh, model_name, smoothness):
        print(f"[CADabra] ML_SurfaceRecon stub called with model: {model_name}")
        print(f"[CADabra] This is a placeholder - implement with actual HuggingFace model")

        # Stub: return input mesh as-is
        surface_data = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "model_used": model_name,
            "smoothness": smoothness,
            "status": "stub_implementation"
        }

        return (surface_data,)


class ML_FeatureDetection:
    """ML-based CAD feature detection (stub for future implementation)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "model_name": ([
                    "autodesk/feature-detection-cad",
                    "hustvl/cad-features"
                ],),
                "confidence_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FEATURES",)
    FUNCTION = "detect_features"
    CATEGORY = "CADabra/ML"

    def detect_features(self, mesh, model_name, confidence_threshold):
        print(f"[CADabra] ML_FeatureDetection stub called with model: {model_name}")
        print(f"[CADabra] This is a placeholder - implement with actual HuggingFace model")

        # Stub: return empty features
        features_data = {
            "features": [],
            "model_used": model_name,
            "confidence_threshold": confidence_threshold,
            "status": "stub_implementation"
        }

        return (features_data,)


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
            print(f"[CADabra] Failed to read axis file: {filepath}")
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
            print(f"[CADabra] Warning: Expected 3 lines in axis file, found {len(lines_data)}")
            if len(lines_data) > 0:
                print(f"[CADabra]   Found lines: {lines_data}")
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

        print(f"[CADabra] Axis parsed via OCC: origin={origin}")
        print(f"[CADabra]   X-dir: {x_dir}")
        print(f"[CADabra]   Y-dir: {y_dir}")
        print(f"[CADabra]   Z-dir: {z_dir}")

        return {
            'origin': origin,
            'x_dir': x_dir,
            'y_dir': y_dir,
            'z_dir': z_dir
        }

    except Exception as e:
        print(f"[CADabra] Error parsing axis file {filepath}: {e}")
        import traceback
        traceback.print_exc()
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
            print("[CADabra] Warning: Axis vectors have zero length")
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
        print(f"[CADabra] Error building axis transform: {e}")
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
            print("[CADabra] Warning: Shape transformation failed")
            return shape
    except Exception as e:
        print(f"[CADabra] Error applying transform: {e}")
        return shape


# Minimum file size to consider as a valid CAD file (filters out axis files, etc.)
MIN_CAD_FILE_SIZE_KB = 5


class CAD_Load_From_Folder:
    """
    Load multiple CAD files from a folder (batch loading).

    Supports single_core (sequential) or parallel execution mode.
    Parallel mode uses subprocesses for crash isolation and OS-level timeout.
    """

    # Supported CAD file extensions
    SUPPORTED_EXTENSIONS = ['.step', '.stp', '.iges', '.igs', '.brep']

    @classmethod
    def _get_cad_folders(cls):
        """Build list of folders: input, output, and all their subfolders recursively."""
        cad_folders = []
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()

        def add_subfolders(base_name, base_path, depth=0):
            """Recursively add subfolders up to depth 3."""
            if depth > 3:
                return
            cad_folders.append(base_name)
            if os.path.isdir(base_path):
                try:
                    for item in sorted(os.listdir(base_path)):
                        subfolder_path = os.path.join(base_path, item)
                        if os.path.isdir(subfolder_path) and not item.startswith('.'):
                            add_subfolders(f"{base_name}/{item}", subfolder_path, depth + 1)
                except OSError:
                    pass

        add_subfolders("input", input_dir)
        add_subfolders("output", output_dir)

        return cad_folders if cad_folders else ["input"]

    @classmethod
    def INPUT_TYPES(cls):
        cad_folders = cls._get_cad_folders()
        return {
            "required": {
                "folder": (cad_folders, {
                    "tooltip": "CAD folder to load from"
                }),
                "pattern": ("STRING", {
                    "default": "*.step",
                    "tooltip": "File pattern (e.g., *.step, *.stp, part_*.igs)"
                }),
                "execution_mode": (["single_core", "parallel"], {
                    "default": "single_core",
                    "tooltip": "single_core: sequential loading. parallel: subprocess workers with timeout"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Skip first N files"
                }),
                "max_cads": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000,
                    "tooltip": "Load up to N files (-1 = unlimited)"
                }),
            },
            "optional": {
                "num_workers": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Number of parallel subprocesses (parallel mode only)"
                }),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 600,
                    "tooltip": "Timeout per file in seconds (parallel mode only)"
                }),
                "recursive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Search subdirectories recursively"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL",)
    RETURN_NAMES = ("cad_models",)
    OUTPUT_TOOLTIPS = ("Batch of loaded CAD models",)
    FUNCTION = "load_from_folder"
    CATEGORY = "CADabra/Loading"
    OUTPUT_IS_LIST = (True,)

    def _discover_cad_files(self, folder, pattern, start_index, max_cads, recursive):
        """Discover CAD files in a folder. Returns (cad_files, full_folder_path)."""
        from pathlib import Path

        # Resolve folder path based on format
        full_folder_path = None
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()

        if folder == "input":
            full_folder_path = Path(input_dir)
        elif folder == "output":
            full_folder_path = Path(output_dir)
        elif folder.startswith("input/"):
            # e.g., "input/cad" or "input/cad/subfolder"
            relative_path = folder[6:]  # Remove "input/" prefix
            full_folder_path = Path(os.path.join(input_dir, relative_path))
        elif folder.startswith("output/"):
            # e.g., "output/cad" or "output/cad/subfolder"
            relative_path = folder[7:]  # Remove "output/" prefix
            full_folder_path = Path(os.path.join(output_dir, relative_path))
        else:
            # Legacy: try in ComfyUI input folder first, then as absolute path
            input_path = Path(os.path.join(input_dir, folder))
            if input_path.exists() and input_path.is_dir():
                full_folder_path = input_path
            else:
                full_folder_path = Path(folder)

        if not full_folder_path.exists() or not full_folder_path.is_dir():
            raise ValueError(f"Folder not found: '{folder}' (resolved to: {full_folder_path})")

        print(f"[CADabra] Loading from folder: {full_folder_path}")

        # Search for CAD files
        if recursive:
            files = list(full_folder_path.rglob(pattern))
        else:
            files = list(full_folder_path.glob(pattern))

        # Filter for valid CAD extensions and exclude small files
        valid_extensions = set(self.SUPPORTED_EXTENSIONS)
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
            print(f"[CADabra] Skipped {skipped_axis} axis files and {skipped_small} small files (< {MIN_CAD_FILE_SIZE_KB}KB)")

        cad_files.sort()

        if len(cad_files) == 0:
            raise FileNotFoundError(
                f"No CAD files found in {full_folder_path}\n"
                f"Pattern: {pattern}\n"
                f"Recursive: {recursive}\n"
                f"Valid extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        print(f"[CADabra] Found {len(cad_files)} CAD files")

        # Apply start_index and max_cads
        if start_index > 0:
            if start_index >= len(cad_files):
                raise ValueError(f"start_index ({start_index}) is >= number of CAD files ({len(cad_files)})")
            cad_files = cad_files[start_index:]
            print(f"[CADabra] Skipping first {start_index} files")

        if max_cads > 0:
            cad_files = cad_files[:max_cads]
            print(f"[CADabra] Loading up to {max_cads} CAD files")

        return cad_files, full_folder_path

    def _load_sequential(self, cad_files):
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
                                    print(f"[CADabra]   Applied axis transform from {axis_path.name}")

                except Exception as occ_error:
                    print(f"[CADabra] Warning: Could not load OCC shape for {cad_file.name}: {occ_error}")

                cad_data = {
                    "file_path": cad_file_path,
                    "occ_shape": occ_shape,
                    "format": ext,
                    "model_name": f"cadabra_batch_{i}",
                    "axis_transform_applied": axis_applied
                }

                loaded_cads.append(cad_data)
                print(f"[CADabra] [{i+1}/{len(cad_files)}] Loaded {cad_file.name}")

            except Exception as e:
                print(f"[CADabra] Warning: Failed to load {cad_file.name}: {e}")
                continue

        return loaded_cads

    def _load_parallel(self, cad_files, num_workers, timeout):
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

        print(f"[CADabra Parallel] Loading {len(cad_files)} files with {num_workers} workers, timeout={timeout}s")

        # Create temp directory for output BREP files
        temp_dir = tempfile.mkdtemp(prefix="cadabra_load_")

        # Create persistent log directory in ComfyUI output folder
        from datetime import datetime
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs", f"load_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"[CADabra Parallel] Logs: {log_dir}")

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
                    print(f"[CADabra Parallel] {filename}: FAILED ({elapsed_str}) - {err} [log: {log_basename}]")
                    continue

                output_brep = result.get("output_brep", work_items[i]["output_brep"])
                if os.path.exists(output_brep):
                    occ_shape = TopoDS_Shape()
                    builder = BRep_Builder()
                    breptools.Read(occ_shape, str(output_brep), builder)

                    cad_data = {
                        "occ_shape": occ_shape,
                        "file_path": result["file_path"],
                        "format": result.get("format", "unknown"),
                    }
                    if "metadata" in result:
                        cad_data["metadata"] = result["metadata"]

                    loaded_cads.append(cad_data)
                    num_faces = result.get("num_faces", "?")
                    if show_log:
                        print(f"[CADabra Parallel] {filename}: {num_faces} faces ({elapsed_str}) [log: {log_basename}]")
                    else:
                        print(f"[CADabra Parallel] {filename}: {num_faces} faces ({elapsed_str})")
                else:
                    failed += 1
                    print(f"[CADabra Parallel] {filename}: FAILED ({elapsed_str}) - output file missing [log: {log_basename}]")

            print(f"[CADabra Parallel] Successfully loaded {len(loaded_cads)} CAD files ({failed} failed)")

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[CADabra Parallel] Warning: Could not clean temp dir: {e}")

        return loaded_cads

    def load_from_folder(self, folder, pattern, execution_mode, start_index, max_cads, num_workers=4, timeout=60, recursive=True):
        """Main dispatch method - routes to sequential or parallel loading."""
        # Discover CAD files
        cad_files, full_folder_path = self._discover_cad_files(folder, pattern, start_index, max_cads, recursive)

        # Dispatch based on execution mode
        if execution_mode == "parallel":
            loaded_cads = self._load_parallel(cad_files, num_workers, timeout)
        else:
            loaded_cads = self._load_sequential(cad_files)

        if len(loaded_cads) == 0:
            raise ValueError(f"Failed to load any CAD files from folder: {full_folder_path}")

        print(f"[CADabra] Successfully loaded {len(loaded_cads)} CAD files")

        return (loaded_cads,)


class CAD_Quality_Metrics:
    """Analyze CAD model topology and quality metrics"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
            },
            "optional": {
                "check_watertightness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check if model is closed/manifold"
                }),
                "check_degeneracy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check for degenerate faces (zero area, collapsed edges)"
                }),
                "check_boundaries": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Analyze face boundaries and loops"
                }),
                "area_tolerance": ("FLOAT", {
                    "default": 1e-6,
                    "min": 1e-12,
                    "max": 1.0,
                    "step": 1e-6,
                    "tooltip": "Minimum area threshold for degenerate face detection"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metrics_dict", "report")
    OUTPUT_TOOLTIPS = ("Metrics as JSON string", "Human-readable report")
    FUNCTION = "analyze_quality"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = "Analyze CAD topology: watertightness, connectedness, face quality, boundaries"

    def analyze_quality(self, cad_model, check_watertightness=True, check_degeneracy=True,
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

            print(f"[CADabra] Topology: {len(volumes)} volumes, {len(faces)} faces, "
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
                    print(f"[CADabra] Found {len(boundary_edges)} boundary edges (model has openings)")

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
                    except:
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
                    print(f"[CADabra] Found {len(degenerate_faces)} degenerate faces")
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
                    except:
                        pass

                avg_boundary_edges = np.mean(face_loop_counts) if face_loop_counts else 0

                metrics['boundaries'] = {
                    'avg_boundary_edges_per_face': float(avg_boundary_edges),
                    'num_faces_with_many_edges': len(faces_with_multiple_loops),
                    'faces_analyzed': min(len(faces), 100)
                }

                if len(faces_with_multiple_loops) > 0:
                    print(f"[CADabra] Found {len(faces_with_multiple_loops)} faces with complex boundaries")

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
                    except:
                        pass

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
                    print(f"[CADabra] Found {len(duplicates)} potential duplicate face pairs")
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
                status = "✓ WATERTIGHT" if wt['is_watertight'] else "✗ NOT WATERTIGHT"
                report_lines.append(f"WATERTIGHTNESS: {status}")
                report_lines.append(f"  Boundary edges: {wt['num_boundary_edges']}")
                report_lines.append(f"  Orphaned edges: {wt['num_orphaned_edges']}")
                report_lines.append("")

            # Degeneracy
            if 'degeneracy' in metrics:
                deg = metrics['degeneracy']
                status = "✓ NO DEGENERATE FACES" if not deg['has_degenerate_faces'] else "✗ DEGENERATE FACES FOUND"
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
                status = "✓ NO DUPLICATES" if not dup['has_duplicates'] else "✗ DUPLICATE FACES FOUND"
                report_lines.append(f"DUPLICATION: {status}")
                report_lines.append(f"  Potential duplicate pairs: {dup['num_potential_duplicate_pairs']}")
                report_lines.append("")

            report_lines.append("=" * 60)

            report = "\n".join(report_lines)
            metrics_json = json.dumps(metrics, indent=2)

            print(f"[CADabra] Quality analysis complete")

            return (metrics_json, report)

        except Exception as e:
            raise RuntimeError(f"Quality analysis failed: {str(e)}")


class PreviewCADOCC:
    """Preview CAD model with VTP tessellation and VTK.js viewer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "linear_deflection": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01,
                                                "tooltip": "Tessellation quality (smaller = finer mesh)"}),
                "angular_deflection": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                                                 "tooltip": "Angular deflection in radians"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_cad"
    CATEGORY = "CADabra/Visualization"

    def preview_cad(self, cad_model, linear_deflection=0.1, angular_deflection=0.5):
        """Export CAD to VTP mesh for preview."""
        import folder_paths

        # Get OCC shape
        occ_shape = cad_model.get("occ_shape")
        if occ_shape is None:
            raise ValueError("CAD model does not contain occ_shape")

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
        print(f"[CADabra] Tessellating (linear: {linear_deflection}, angular: {angular_deflection})")
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
        print(f"[CADabra] Mesh: {total_vertices} vertices, {total_triangles} triangles")

        # Export VTP
        output_dir = folder_paths.get_output_directory()
        base_name = f"preview_{uuid.uuid4().hex[:8]}"

        try:
            import vtk
            vtp_filename = self._export_vtp(all_vertices, all_indices, face_ranges, base_name, output_dir)
        except ImportError:
            raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

        print(f"[CADabra] Entities: {num_volumes} volumes, {num_faces} faces, {num_edges} edges")

        return {"ui": {
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
        }}

    def _export_vtp(self, vertices, indices, face_ranges, base_filename, output_dir):
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
        print(f"[CADabra] Exported VTP: {vtp_filename} ({file_size:,} bytes)")
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


class CAD_Remesh_OCC:
    """Tessellate CAD model using BRepMesh and return TRIMESH."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "linear_deflection": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0, "step": 0.01}),
                "angular_deflection": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.57, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    FUNCTION = "remesh"
    CATEGORY = "CADabra"

    def remesh(self, cad_model, linear_deflection, angular_deflection):
        import trimesh

        occ_shape = cad_model.get("occ_shape")
        if occ_shape is None:
            raise ValueError("CAD model does not contain occ_shape")

        print(f"[CAD_Remesh_OCC] Tessellating with linear_deflection={linear_deflection}, angular_deflection={angular_deflection}")

        vertices, faces = tessellate_occ_shape(occ_shape, linear_deflection, angular_deflection)

        if len(vertices) == 0:
            raise RuntimeError("Tessellation produced no vertices")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"[CAD_Remesh_OCC] Created mesh: {len(vertices)} vertices, {len(faces)} faces")

        return (mesh,)


class CAD_Remesh_OCC_Advanced:
    """
    Advanced tessellation of CAD model using BRepMesh.

    Supports single_core (sequential) or parallel execution mode.
    Parallel mode uses subprocesses for crash isolation and OS-level timeout.

    Adds vertex merging option (for connected meshes from sewn shapes)
    and handles face orientation for consistent normals.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "linear_deflection": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Max distance between mesh and actual surface (smaller = finer mesh)"
                }),
                "angular_deflection": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 1.57,
                    "step": 0.01,
                    "tooltip": "Angular deflection in radians (controls curvature sampling)"
                }),
                "merge_vertices": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge duplicate vertices on shared edges (produces connected mesh from sewn shapes)"
                }),
                "extract_face_ids": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Store CAD face index for each mesh triangle as 'cad_face_id' face attribute"
                }),
                "output_folder": ("STRING", {
                    "default": "cadabra_meshes",
                    "tooltip": "Folder name in output directory for VTP files (parallel mode)"
                }),
            },
            "optional": {
                "execution_mode": (["single_core", "parallel"], {
                    "default": "single_core",
                    "tooltip": "single_core: process sequentially. parallel: subprocess workers with timeout"
                }),
                "num_workers": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Number of parallel subprocesses (parallel mode only)"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600,
                    "tooltip": "Timeout per model in seconds (parallel mode only)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESH_METADATA", "STRING")
    RETURN_NAMES = ("meshes", "metadata", "vtp_paths")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "remesh_advanced"
    CATEGORY = "CADabra"

    def _remesh_single(self, cad_model, linear_deflection, angular_deflection, merge_vertices, extract_face_ids):
        import trimesh
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.TopoDS import topods

        occ_shape = cad_model.get("occ_shape")
        if occ_shape is None:
            raise ValueError("CAD model does not contain occ_shape")

        print(f"[CAD_Remesh_OCC_Advanced] Tessellating with linear_deflection={linear_deflection}, angular_deflection={angular_deflection}")

        # 1. Mesh the shape with BRepMesh
        with log_operation("BRepMesh", linear_deflection=linear_deflection, angular_deflection=angular_deflection):
            mesher = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection)
            mesher.Perform()

        if not mesher.IsDone():
            raise RuntimeError("BRepMesh tessellation failed")

        # 2. Collect triangulations from all faces
        all_verts = []
        all_faces = []
        cad_face_ids = []  # Track which CAD face each triangle came from
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
                        # Flip winding order for reversed faces
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
        mesh = trimesh.Trimesh(
            vertices=np.array(all_verts, dtype=np.float32),
            faces=np.array(all_faces, dtype=np.int32)
        )

        verts_before = len(mesh.vertices)

        # 4. Optionally merge duplicate vertices
        if merge_vertices:
            mesh.merge_vertices()

        # 5. Store CAD face IDs if requested
        if extract_face_ids and cad_face_ids:
            cad_face_ids_array = np.array(cad_face_ids, dtype=np.int32)
            mesh.face_attributes['cad_face_id'] = cad_face_ids_array
            mesh.metadata['has_cad_face_ids'] = True
            mesh.metadata['num_cad_faces'] = cad_face_idx

        # 6. Build metadata
        file_path = cad_model.get("file_path", "model")
        metadata = {
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "num_cad_faces": cad_face_idx,
            "backend": "OCC BRepMesh",
            "linear_deflection": linear_deflection,
            "angular_deflection": angular_deflection,
            "merged": merge_vertices,
            "vertices_before_merge": verts_before,
            "has_cad_face_ids": extract_face_ids,
            "file_path": file_path,
        }

        mesh.metadata['file_path'] = file_path

        print(f"[CAD_Remesh_OCC_Advanced] Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces from {cad_face_idx} CAD faces (merged={merge_vertices}, before={verts_before})")

        return mesh, metadata

    def _remesh_sequential(self, cad_models, linear_deflection, angular_deflection, merge_vertices, extract_face_ids):
        """Process CAD models sequentially (single_core mode)."""
        meshes = []
        metadata_list = []

        for i, cm in enumerate(cad_models):
            try:
                mesh, metadata = self._remesh_single(cm, linear_deflection, angular_deflection, merge_vertices, extract_face_ids)
                meshes.append(mesh)
                metadata_list.append(metadata)
                filename = os.path.basename(cm.get("file_path", f"model_{i}"))
                print(f"[CAD_Remesh_OCC_Advanced] [{i+1}/{len(cad_models)}] {filename}: {len(mesh.vertices)} verts, {len(mesh.faces)} tris")
            except Exception as e:
                filename = os.path.basename(cm.get("file_path", f"model_{i}"))
                print(f"[CAD_Remesh_OCC_Advanced] [{i+1}/{len(cad_models)}] {filename}: FAILED - {e}")
                continue

        return meshes, metadata_list

    def _remesh_parallel(self, cad_models, num_workers, timeout, linear_deflection, angular_deflection, extract_face_ids, output_folder):
        """Process CAD models in parallel using subprocesses (parallel mode)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from OCC.Core.BRepTools import breptools
        import subprocess
        import tempfile
        import time
        import json
        import sys
        import trimesh
        import pyvista as pv

        print(f"[CADabra Parallel] Processing {len(cad_models)} models with {num_workers} workers, timeout={timeout}s")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cadabra_mesh_")

        # Path to subprocess script
        script_path = os.path.join(os.path.dirname(__file__), "mesh_subprocess.py")

        # Prepare work items
        work_items = []
        skipped = 0
        for idx, cm in enumerate(cad_models):
            occ_shape = cm.get("occ_shape")
            file_path = cm.get("file_path", f"model_{idx}")
            if occ_shape is None:
                skipped += 1
                continue

            input_brep = os.path.join(temp_dir, f"input_{idx}.brep")
            output_json = os.path.join(temp_dir, f"output_{idx}.json")
            result_file = os.path.join(temp_dir, f"result_{idx}.json")
            breptools.Write(occ_shape, input_brep)

            work_items.append({
                "idx": idx,
                "input_brep": input_brep,
                "output_json": output_json,
                "result_file": result_file,
                "file_path": file_path,
            })

        if skipped > 0:
            print(f"[CADabra Parallel] Warning: Skipped {skipped} models without OCC shape")

        if len(work_items) == 0:
            raise ValueError("No valid CAD models to process")

        def run_mesh_subprocess(item):
            """Run meshing in subprocess with timeout."""
            cmd = [
                sys.executable,
                script_path,
                item["input_brep"],
                item["output_json"],
                str(linear_deflection),
                str(angular_deflection),
                f"--result-file={item['result_file']}",
                "--no-merge-vertices",
            ]
            if extract_face_ids:
                cmd.append("--extract-face-ids")
            else:
                cmd.append("--no-extract-face-ids")

            item_start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5
                )
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(f"  {line}")
                elapsed = time.time() - item_start
                if os.path.exists(item["result_file"]):
                    with open(item["result_file"], 'r') as f:
                        return {**json.load(f), "file_path": item["file_path"], "output_json": item["output_json"], "elapsed": elapsed}
                elif result.returncode == 0:
                    return {"success": True, "file_path": item["file_path"], "output_json": item["output_json"], "elapsed": elapsed}
                else:
                    return {"success": False, "error": result.stderr or "Unknown error", "file_path": item["file_path"], "elapsed": elapsed}
            except subprocess.TimeoutExpired:
                elapsed = time.time() - item_start
                return {"success": False, "error": f"Timed out after {timeout}s", "file_path": item["file_path"], "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - item_start
                return {"success": False, "error": str(e), "file_path": item["file_path"], "elapsed": elapsed}

        # Process in parallel
        results = [None] * len(work_items)
        start_time = time.time()
        meshes = []
        metadata_list = []
        vtp_paths = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(run_mesh_subprocess, item): i
                    for i, item in enumerate(work_items)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {"success": False, "error": str(e), "file_path": work_items[idx]["file_path"]}
                    completed += 1
                    elapsed = time.time() - start_time
                    _progress_bar(completed, len(work_items), elapsed, prefix="Meshing: ")

            # Build trimesh objects from results
            failed = 0
            output_dir = folder_paths.get_output_directory()
            vtp_folder = os.path.join(output_dir, output_folder)
            os.makedirs(vtp_folder, exist_ok=True)

            for i, result in enumerate(results):
                elapsed_str = f"{result.get('elapsed', 0):.1f}s" if result else "?"
                filename = os.path.basename(result.get("file_path", work_items[i]["file_path"])) if result else os.path.basename(work_items[i]["file_path"])

                if result is None or not result.get("success", False):
                    failed += 1
                    err = result.get("error", "Unknown") if result else "No result"
                    print(f"[CADabra Parallel] {filename}: FAILED ({elapsed_str}) - {err}")
                    continue

                output_json = result.get("output_json", work_items[i]["output_json"])
                if os.path.exists(output_json):
                    with open(output_json, 'r') as f:
                        mesh_data = json.load(f)

                    mesh = trimesh.Trimesh(
                        vertices=np.array(mesh_data["vertices"], dtype=np.float32),
                        faces=np.array(mesh_data["faces"], dtype=np.int32)
                    )

                    # Remove degenerate faces
                    num_faces_before = len(mesh.faces)
                    nondegen_mask = mesh.nondegenerate_faces()
                    if not nondegen_mask.all():
                        if "cad_face_ids" in mesh_data:
                            mesh_data["cad_face_ids"] = [mesh_data["cad_face_ids"][j] for j, keep in enumerate(nondegen_mask) if keep]
                        mesh.update_faces(nondegen_mask)
                        num_removed = num_faces_before - len(mesh.faces)
                        print(f"  [Degenerate] Removed {num_removed} degenerate faces from {filename}")

                    # Store CAD face IDs if extracted
                    if extract_face_ids and "cad_face_ids" in mesh_data:
                        mesh.face_attributes['cad_face_id'] = np.array(mesh_data["cad_face_ids"], dtype=np.int32)
                        mesh.metadata['has_cad_face_ids'] = True
                        mesh.metadata['num_cad_faces'] = result.get("num_cad_faces", 0)

                    mesh.metadata['file_path'] = result["file_path"]

                    meshes.append(mesh)
                    metadata_list.append({
                        "num_vertices": len(mesh.vertices),
                        "num_faces": len(mesh.faces),
                        "num_cad_faces": result.get("num_cad_faces", 0),
                        "backend": "OCC BRepMesh (Subprocess)",
                        "linear_deflection": linear_deflection,
                        "angular_deflection": angular_deflection,
                        "has_cad_face_ids": extract_face_ids,
                        "file_path": result["file_path"],
                    })

                    # Save VTP
                    base_name = os.path.splitext(os.path.basename(result["file_path"]))[0]
                    vtp_path = os.path.join(vtp_folder, f"{base_name}.vtp")
                    verts = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
                    pv_mesh = pv.PolyData(verts, pv_faces)
                    if 'cad_face_id' in mesh.face_attributes:
                        pv_mesh.cell_data['cad_face_id'] = mesh.face_attributes['cad_face_id']
                    pv_mesh.save(vtp_path)
                    vtp_paths.append(vtp_path)

                    num_verts = result.get("num_vertices", len(mesh.vertices))
                    num_tris = result.get("num_faces", len(mesh.faces))
                    print(f"[CADabra Parallel] {filename}: {num_verts} verts, {num_tris} tris ({elapsed_str})")
                else:
                    failed += 1
                    print(f"[CADabra Parallel] {filename}: FAILED ({elapsed_str}) - output file missing")

            print(f"[CADabra Parallel] Completed: {len(meshes)} meshes from {len(cad_models)} models ({failed} failed)")

        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[CADabra Parallel] Warning: Could not clean temp dir: {e}")

        return meshes, metadata_list, vtp_paths

    def remesh_advanced(self, cad_model, linear_deflection, angular_deflection, merge_vertices,
                        extract_face_ids, output_folder, execution_mode="single_core", num_workers=4, timeout=120):
        """Main dispatch method - routes to sequential or parallel processing."""
        # Extract scalar values from lists (INPUT_IS_LIST behavior)
        execution_mode_val = execution_mode[0] if isinstance(execution_mode, list) else execution_mode
        num_workers_val = num_workers[0] if isinstance(num_workers, list) else num_workers
        timeout_val = timeout[0] if isinstance(timeout, list) else timeout
        linear_val = linear_deflection[0] if isinstance(linear_deflection, list) else linear_deflection
        angular_val = angular_deflection[0] if isinstance(angular_deflection, list) else angular_deflection
        merge_val = merge_vertices[0] if isinstance(merge_vertices, list) else merge_vertices
        extract_val = extract_face_ids[0] if isinstance(extract_face_ids, list) else extract_face_ids
        output_folder_val = output_folder[0] if isinstance(output_folder, list) else output_folder

        # Ensure cad_model is a list
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        if execution_mode_val == "parallel":
            meshes, metadata_list, vtp_paths = self._remesh_parallel(
                cad_models, num_workers_val, timeout_val, linear_val, angular_val, extract_val, output_folder_val
            )
        else:
            meshes, metadata_list = self._remesh_sequential(cad_models, linear_val, angular_val, merge_val, extract_val)
            vtp_paths = [""] * len(meshes)  # Empty paths for single_core mode

        if len(meshes) == 0:
            raise ValueError("Failed to process any CAD models")

        return (meshes, metadata_list, vtp_paths)


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


class PreviewCADBatch:
    """Preview batch of CAD models or TRIMESH objects with VTK.js viewer and index navigation.

    Accepts either CAD_MODEL or TRIMESH as input (but not both).
    - If CAD_MODEL: Tessellates using BRepMesh and exports to VTP
    - If TRIMESH: Converts to VTP for viewing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                "cad_model": ("CAD_MODEL",),
                "trimesh": ("TRIMESH",),
                "linear_deflection": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01,
                                                "tooltip": "Tessellation quality (smaller = finer mesh)"}),
                "angular_deflection": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                                                 "tooltip": "Angular deflection in radians"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_cad_batch"
    CATEGORY = "CADabra/Visualization"
    INPUT_IS_LIST = True

    def preview_cad_batch(self, index, cad_model=None, trimesh=None, linear_deflection=None, angular_deflection=None):
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
            return self._preview_trimesh_batch(trimesh, index_val)
        else:
            return self._preview_cad_batch(cad_model, index_val, linear_deflection_val, angular_deflection_val)

    def _preview_trimesh_batch(self, trimesh_list, index_val):
        """Handle TRIMESH input - convert to VTP."""
        import folder_paths

        # Validate batch
        batch_size = len(trimesh_list)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current mesh from batch
        current_mesh = trimesh_list[actual_index]

        print(f"[CADabra Batch] TRIMESH mode - Batch size: {batch_size}, showing index: {actual_index + 1}/{batch_size}")

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
            vtp_filename = self._export_trimesh_vtp(current_mesh, base_name, output_dir)
        except ImportError:
            raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

        mesh_vertex_count = len(current_mesh.vertices)
        mesh_face_count = len(current_mesh.faces)
        print(f"[CADabra Batch] Exported TRIMESH to VTP: {vtp_filename} ({mesh_vertex_count} vertices, {mesh_face_count} faces)")

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

        return {"ui": ui_data}

    def _export_trimesh_vtp(self, mesh, base_filename, output_dir):
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

    def _preview_cad_batch(self, cad_model, index_val, linear_deflection_val, angular_deflection_val):
        """Handle CAD_MODEL input - tessellate and export to VTP."""
        import folder_paths

        # Validate batch
        batch_size = len(cad_model)
        actual_index = max(0, min(index_val, batch_size - 1))

        # Select current model from batch
        current_model = cad_model[actual_index]

        print(f"[CADabra Batch] CAD_MODEL mode - Batch size: {batch_size}, showing index: {actual_index + 1}/{batch_size}")

        # Generate unique filenames
        base_name = f"preview_batch_{uuid.uuid4().hex[:8]}"
        output_dir = folder_paths.get_output_directory()

        try:
            # Get OCC shape - required for all operations
            occ_shape = current_model.get("occ_shape")
            if occ_shape is None:
                raise ValueError("CAD model does not contain occ_shape")

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
            print(f"[CADabra Batch] Tessellating (linear: {linear_deflection_val}, angular: {angular_deflection_val})")
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
            print(f"[CADabra Batch] Mesh: {total_vertices} vertices, {total_triangles} triangles")

            # Export VTP
            try:
                import vtk
                vtp_filename = self._export_vtp(all_vertices, all_indices, face_ranges, base_name, output_dir)
            except ImportError:
                raise RuntimeError("VTK not available. Please install vtk: pip install vtk")

            print(f"[CADabra Batch] Entities: {num_volumes} volumes, {num_faces} faces, {num_edges} edges")

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

            return {"ui": ui_data}

        except Exception as e:
            raise RuntimeError(f"Failed to export CAD for batch preview: {str(e)}")

    def _export_vtp(self, vertices, indices, face_ranges, base_filename, output_dir):
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


class CAD_Convert_Format:
    """Convert between CAD formats using pythonocc-core"""

    def __init__(self):
        """Initialize and check for pythonocc availability"""
        self.pythonocc_available = False
        try:
            from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
            from OCC.Core.IGESControl import IGESControl_Reader, IGESControl_Writer
            from OCC.Core.BRepTools import BRepTools
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.StlAPI import StlAPI_Writer
            from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
            self.pythonocc_available = True
        except ImportError:
            pass  # Will provide helpful error message in convert() method

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"default": ""}),
                "output_format": (["STEP", "IGES", "BREP", "STL"],),
            },
            "optional": {
                "stl_quality": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "keep_original_name": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "CAD_MODEL")
    FUNCTION = "convert"
    CATEGORY = "CADabra/Conversion"

    def convert(self, input_file, output_format, stl_quality=0.1, keep_original_name=True):
        """Convert CAD file between formats"""

        # Check if pythonocc is available
        if not self.pythonocc_available:
            raise ImportError(
                "pythonocc-core is required for CAD format conversion.\n\n"
                "Install with:\n"
                "  conda install -c conda-forge pythonocc-core=7.9.0\n\n"
                "See INSTALL.md for detailed installation instructions."
            )

        # Import pythonocc modules
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IGESControl import IGESControl_Reader, IGESControl_Writer
        from OCC.Core.BRepTools import BRepTools, breptools
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
        from OCC.Core.IFSelect import IFSelect_ReturnStatus
        import folder_paths

        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Detect input format
        input_ext = os.path.splitext(input_file)[1].lower()

        # Generate output filename
        if keep_original_name:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
        else:
            base_name = f"converted_{uuid.uuid4().hex[:8]}"

        output_ext = output_format.lower()
        if output_ext == "step":
            output_ext = "stp"
        output_filename = f"{base_name}.{output_ext}"

        # Use ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        output_file = os.path.join(output_dir, output_filename)

        try:
            # Step 1: Read input file
            print(f"[CADabra] Reading {input_ext} file: {os.path.basename(input_file)}")
            shape = None

            if input_ext in ['.step', '.stp']:
                reader = STEPControl_Reader()
                status = reader.ReadFile(input_file)
                if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                    raise RuntimeError(f"Failed to read STEP file: {input_file}")
                reader.TransferRoots()
                shape = reader.OneShape()

            elif input_ext in ['.iges', '.igs']:
                reader = IGESControl_Reader()
                status = reader.ReadFile(input_file)
                if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                    raise RuntimeError(f"Failed to read IGES file: {input_file}")
                reader.TransferRoots()
                shape = reader.OneShape()

            elif input_ext == '.brep':
                builder = BRep_Builder()
                shape = TopoDS_Shape()
                breptools.Read(shape, input_file, builder)

            elif input_ext == '.stl':
                raise NotImplementedError(
                    "STL to B-rep conversion requires mesh-to-solid reconstruction.\n"
                    "This feature is not yet implemented. Use STL as final export format only."
                )

            else:
                raise ValueError(
                    f"Unsupported input format: {input_ext}\n\n"
                    f"Supported formats: .step, .stp, .iges, .igs, .brep\n\n"
                    f"For proprietary formats (CATIA, SolidWorks, etc.), "
                    f"please export to STEP from your CAD software first."
                )

            if shape is None or shape.IsNull():
                raise RuntimeError("Failed to load shape from file")

            # Step 2: Write output file
            print(f"[CADabra] Converting to {output_format}...")

            if output_format == "STEP":
                writer = STEPControl_Writer()
                writer.Transfer(shape, STEPControl_AsIs)
                status = writer.Write(output_file)
                if status != IFSelect_ReturnStatus.IFSelect_RetDone:
                    raise RuntimeError("Failed to write STEP file")

            elif output_format == "IGES":
                writer = IGESControl_Writer()
                writer.AddShape(shape)
                writer.ComputeModel()
                if not writer.Write(output_file):
                    raise RuntimeError("Failed to write IGES file")

            elif output_format == "BREP":
                if not breptools.Write(shape, output_file):
                    raise RuntimeError("Failed to write BREP file")

            elif output_format == "STL":
                writer = StlAPI_Writer()
                writer.SetASCIIMode(False)  # Binary STL
                # stl_quality is linear deflection (smaller = higher quality)
                writer.Write(shape, output_file, stl_quality)

            # Verify output file was created
            if not os.path.exists(output_file):
                raise RuntimeError("Output file was not created")

            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"[CADabra] ✓ Conversion successful: {output_filename} ({file_size:.1f} KB)")

            # Create CAD_MODEL data structure for chaining
            cad_model = {
                "file_path": output_file,
                "format": output_ext,
                "original_file": input_file,
                "conversion": f"{input_ext} → {output_format}"
            }

            return (output_file, cad_model)

        except Exception as e:
            raise RuntimeError(f"CAD conversion failed: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "CAD_Load": CAD_Load,
    "CAD_Load_From_Folder": CAD_Load_From_Folder,
    "CAD_Mesh_Gmsh": CAD_Mesh_Gmsh,
    "CAD_Mesh_Gmsh_Advanced": CAD_Mesh_Gmsh_Advanced,
    "CAD_Convert_Format": CAD_Convert_Format,
    "CAD_Quality_Metrics": CAD_Quality_Metrics,
    "ML_SurfaceRecon": ML_SurfaceRecon,
    "ML_FeatureDetection": ML_FeatureDetection,
    "PreviewCADOCC": PreviewCADOCC,
    "PreviewCADBatch": PreviewCADBatch,
    "CAD_Remesh_OCC": CAD_Remesh_OCC,
    "CAD_Remesh_OCC_Advanced": CAD_Remesh_OCC_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_Load": "Load CAD",
    "CAD_Load_From_Folder": "Load CADs from Folder",
    "CAD_Mesh_Gmsh": "Generate Mesh (Gmsh)",
    "CAD_Mesh_Gmsh_Advanced": "Advanced Mesh Generation (Gmsh)",
    "CAD_Convert_Format": "Convert CAD Format",
    "CAD_Quality_Metrics": "CAD Quality Metrics",
    "ML_SurfaceRecon": "ML Surface Reconstruction",
    "ML_FeatureDetection": "ML Feature Detection",
    "PreviewCADOCC": "Preview CAD",
    "PreviewCADBatch": "Preview CAD Batch",
    "CAD_Remesh_OCC": "CAD to Mesh (OCC)",
    "CAD_Remesh_OCC_Advanced": "CAD to Mesh (OCC) Advanced",
}
