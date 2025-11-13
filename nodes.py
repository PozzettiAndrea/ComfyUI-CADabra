"""
ComfyUI-CADabra Nodes
CAD file loading, meshing, and ML-based processing nodes
"""

import os
import tempfile
import uuid
import numpy as np
import torch
import folder_paths


class CAD_Load_Gmsh:
    """Load CAD files (STEP, IGES, BREP) using Gmsh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_file_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CAD_MODEL",)
    FUNCTION = "load_cad"
    CATEGORY = "CADabra"

    def load_cad(self, cad_file_path):
        try:
            import gmsh
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        # Search for CAD file in multiple locations
        search_paths = []

        # 1. Try the provided path as-is (absolute or relative)
        search_paths.append(cad_file_path)

        # 2. Try in ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        search_paths.append(os.path.join(input_dir, cad_file_path))

        # 3. Try in ComfyUI input/cad subdirectory (following convention like input/3d)
        cad_input_dir = os.path.join(input_dir, 'cad')
        search_paths.append(os.path.join(cad_input_dir, cad_file_path))

        # Find the first existing file
        found_path = None
        for path in search_paths:
            if os.path.exists(path) and os.path.isfile(path):
                found_path = path
                print(f"[CADabra] Found CAD file at: {path}")
                break

        if not found_path:
            raise FileNotFoundError(
                f"CAD file not found: {cad_file_path}\n"
                f"Searched in:\n" +
                "\n".join(f"  - {p}" for p in search_paths)
            )

        cad_file_path = found_path
        ext = os.path.splitext(cad_file_path)[1].lower()
        if ext not in ['.step', '.stp', '.iges', '.igs', '.brep']:
            raise ValueError(f"Unsupported file format: {ext}")

        # Gmsh is already initialized in __init__.py (main thread)
        # Check if initialized, if not, try to initialize (with signal handler workaround)
        if not gmsh.is_initialized():
            try:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
            except ValueError as e:
                # Signal handler issue in worker thread - gmsh still works
                if "signal only works in main thread" not in str(e):
                    raise

        try:
            # Clear any previous model
            gmsh.model.remove()
            gmsh.model.add("cadabra_model")

            if ext in ['.step', '.stp']:
                gmsh.model.occ.importShapes(cad_file_path)
            elif ext in ['.iges', '.igs']:
                gmsh.model.occ.importShapes(cad_file_path)
            elif ext == '.brep':
                gmsh.model.occ.importShapes(cad_file_path)

            gmsh.model.occ.synchronize()

            cad_data = {
                "file_path": cad_file_path,
                "gmsh_model": gmsh.model,
                "format": ext
            }

            print(f"[CADabra] Loaded CAD file: {os.path.basename(cad_file_path)}")
            return (cad_data,)

        except Exception as e:
            # Don't finalize gmsh - keep it alive for next operation
            raise RuntimeError(f"Failed to load CAD file: {str(e)}")


class CAD_Mesh_Gmsh:
    """Generate mesh from CAD model using Gmsh"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "mesh_type": (["2D", "3D"],),
                "element_size": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "algorithm": (["Automatic", "Delaunay", "Frontal"],),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate_mesh"
    CATEGORY = "CADabra"

    def generate_mesh(self, cad_model, mesh_type, element_size, algorithm):
        try:
            import gmsh
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        algo_map = {"Automatic": 1, "Delaunay": 5, "Frontal": 6}

        try:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)
            gmsh.option.setNumber("Mesh.Algorithm", algo_map[algorithm])

            if mesh_type == "2D":
                gmsh.model.mesh.generate(2)
            else:
                gmsh.model.mesh.generate(3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)

            if mesh_type == "2D":
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
            else:
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)

            if len(elem_node_tags) == 0:
                raise RuntimeError("Mesh generation produced no elements")

            faces = elem_node_tags[0].reshape(-1, 3 if mesh_type == "2D" else 4) - 1

            mesh_data = {
                "vertices": vertices,
                "faces": faces,
                "type": mesh_type,
                "element_size": element_size
            }

            print(f"[CADabra] Generated {mesh_type} mesh: {len(vertices)} vertices, {len(faces)} elements")
            # Don't finalize gmsh - keep it alive for next operation
            return (mesh_data,)

        except Exception as e:
            # Don't finalize gmsh - keep it alive for next operation
            raise RuntimeError(f"Mesh generation failed: {str(e)}")


class ML_SurfaceRecon:
    """ML-based surface reconstruction (stub for future implementation)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
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
            "vertices": mesh["vertices"],
            "faces": mesh["faces"],
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
                "mesh": ("MESH",),
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


class PreviewCADOCC:
    """Preview CAD with OpenCascade.js viewer (B-rep, no tessellation)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_cad_occ"
    CATEGORY = "CADabra/Visualization"

    def preview_cad_occ(self, cad_model):
        """Export CAD to STEP and prepare for OpenCascade.js preview."""
        try:
            import gmsh
            import folder_paths
        except ImportError as e:
            raise ImportError(f"Required module not found: {e}")

        # Generate unique filename for temp STEP file
        filename = f"preview_occ_{uuid.uuid4().hex[:8]}.step"

        # Use ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        try:
            # The cad_model dict contains gmsh_model which is already initialized
            gmsh_model = cad_model.get("gmsh_model")
            if gmsh_model is None:
                raise ValueError("CAD model does not contain gmsh_model")

            # Export to STEP format for OpenCascade.js
            gmsh.write(filepath)

            # Get model info for metadata
            entities = gmsh.model.getEntities()
            num_volumes = len([e for e in entities if e[0] == 3])
            num_faces = len([e for e in entities if e[0] == 2])
            num_edges = len([e for e in entities if e[0] == 1])

            # Get bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)

            print(f"[CADabra] Exported CAD to STEP: {filename}")
            print(f"[CADabra] Entities: {num_volumes} volumes, {num_faces} faces, {num_edges} edges")

            # Return UI data dictionary (all values must be lists!)
            return {
                "ui": {
                    "cad_file": [filename],
                    "format": ["step"],
                    "original_format": [cad_model.get("format", "unknown")],
                    "num_volumes": [num_volumes],
                    "num_faces": [num_faces],
                    "num_edges": [num_edges],
                    "bounds_min": [[xmin, ymin, zmin]],
                    "bounds_max": [[xmax, ymax, zmax]],
                    "extents": [[xmax - xmin, ymax - ymin, zmax - zmin]],
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to export CAD for preview: {str(e)}")


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
    "CAD_Load_Gmsh": CAD_Load_Gmsh,
    "CAD_Mesh_Gmsh": CAD_Mesh_Gmsh,
    "CAD_Convert_Format": CAD_Convert_Format,
    "ML_SurfaceRecon": ML_SurfaceRecon,
    "ML_FeatureDetection": ML_FeatureDetection,
    "PreviewCADOCC": PreviewCADOCC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_Load_Gmsh": "Load CAD (Gmsh)",
    "CAD_Mesh_Gmsh": "Generate Mesh (Gmsh)",
    "CAD_Convert_Format": "Convert CAD Format",
    "ML_SurfaceRecon": "ML Surface Reconstruction",
    "ML_FeatureDetection": "ML Feature Detection",
    "PreviewCADOCC": "Preview CAD (OpenCascade.js)",
}
