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


class CAD_Mesh_Gmsh_Advanced:
    """Advanced mesh generation with size fields, refinement, and extended algorithms"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "mesh_type": (["2D", "3D"],),
                "element_size": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                # Size Field Controls
                "min_element_size": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0, "step": 0.01}),
                "max_element_size": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 100.0, "step": 0.1}),
                "use_curvature_sizing": ("BOOLEAN", {"default": False}),
                "curvature_divisions": ("INT", {"default": 20, "min": 4, "max": 100, "step": 1}),

                # Algorithm Options (expanded)
                "algorithm_2d": (["Automatic", "MeshAdapt", "Delaunay", "Frontal", "BAMG", "DelQuad"],),
                "algorithm_3d": (["Automatic", "Delaunay", "Frontal", "MMG3D", "HXT"],),

                # Element Order
                "element_order": (["1", "2", "3", "4", "5"],),

                # Refinement
                "subdivision_algorithm": (["None", "All Triangles", "All Quadrangles", "Barycentric"],),

                # Advanced gmsh options (JSON)
                "gmsh_options": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "generate_mesh"
    CATEGORY = "CADabra/Advanced"

    def generate_mesh(self, cad_model, mesh_type, element_size,
                     min_element_size=0.1, max_element_size=10.0,
                     use_curvature_sizing=False, curvature_divisions=20,
                     algorithm_2d="Automatic", algorithm_3d="Automatic",
                     element_order="1", subdivision_algorithm="None",
                     gmsh_options="{}"):
        try:
            import gmsh
            import json
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        # Algorithm mappings (extended)
        algo_2d_map = {
            "Automatic": 1,
            "MeshAdapt": 1,
            "Delaunay": 5,
            "Frontal": 6,
            "BAMG": 7,
            "DelQuad": 8,
        }
        algo_3d_map = {
            "Automatic": 1,
            "Delaunay": 1,
            "Frontal": 4,
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
            # Apply size constraints
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element_size)

            # Set base element size
            if element_size > 0:
                gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", element_size)

            # Curvature-based sizing
            if use_curvature_sizing:
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
                gmsh.option.setNumber("Mesh.MinimumCirclePoints", curvature_divisions)
                gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
            else:
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

            # Set algorithms
            if mesh_type == "2D":
                gmsh.option.setNumber("Mesh.Algorithm", algo_2d_map.get(algorithm_2d, 1))
            else:
                gmsh.option.setNumber("Mesh.Algorithm", algo_2d_map.get(algorithm_2d, 1))
                gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d_map.get(algorithm_3d, 1))

            # Set element order
            gmsh.option.setNumber("Mesh.ElementOrder", int(element_order))

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

            # Generate mesh
            if mesh_type == "2D":
                gmsh.model.mesh.generate(2)
            else:
                gmsh.model.mesh.generate(3)

            # Apply subdivision if requested
            if subdivision_algorithm != "None":
                subdivision_algo = subdivision_map[subdivision_algorithm]
                gmsh.model.mesh.refine()

            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)

            # Get elements based on mesh type
            if mesh_type == "2D":
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
            else:
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)

            if len(elem_node_tags) == 0:
                raise RuntimeError("Mesh generation produced no elements")

            # Handle different element types (triangles, quads, tets, hexes)
            all_faces = []
            for i, elem_type in enumerate(elem_types):
                elem_nodes = elem_node_tags[i]

                # Determine nodes per element based on element type
                if elem_type == 2:  # 3-node triangle
                    nodes_per_elem = 3
                elif elem_type == 3:  # 4-node quadrangle
                    nodes_per_elem = 4
                elif elem_type == 4:  # 4-node tetrahedron
                    nodes_per_elem = 4
                elif elem_type == 5:  # 8-node hexahedron
                    nodes_per_elem = 8
                elif elem_type == 9:  # 6-node triangle (2nd order)
                    nodes_per_elem = 6
                elif elem_type == 10:  # 9-node quadrangle (2nd order)
                    nodes_per_elem = 9
                elif elem_type == 11:  # 10-node tetrahedron (2nd order)
                    nodes_per_elem = 10
                else:
                    # Default: try to infer from element count
                    nodes_per_elem = len(elem_nodes) // len(elem_tags[i])

                faces = elem_nodes.reshape(-1, nodes_per_elem) - 1
                all_faces.append(faces)

            # Combine all faces
            if len(all_faces) > 0:
                faces = np.vstack(all_faces) if len(all_faces) > 1 else all_faces[0]
            else:
                faces = np.array([])

            mesh_data = {
                "vertices": vertices,
                "faces": faces,
                "type": mesh_type,
                "element_size": element_size,
                "element_order": int(element_order),
                "min_size": min_element_size,
                "max_size": max_element_size,
                "algorithm_2d": algorithm_2d,
                "algorithm_3d": algorithm_3d if mesh_type == "3D" else None,
            }

            print(f"[CADabra] Generated {mesh_type} mesh (Advanced): "
                  f"{len(vertices)} vertices, {len(faces)} elements, "
                  f"order {element_order}")

            return (mesh_data,)

        except Exception as e:
            raise RuntimeError(f"Advanced mesh generation failed: {str(e)}")


class Mesh_Optimize_Gmsh:
    """Post-process mesh with optimization, smoothing, and recombination"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
            "optional": {
                # Optimization
                "optimize": ("BOOLEAN", {"default": True}),
                "optimize_netgen": ("BOOLEAN", {"default": False}),
                "optimize_ho": ("BOOLEAN", {"default": False}),  # High-order

                # Smoothing
                "smooth_steps": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),

                # Recombination (triangles → quads)
                "recombine": ("BOOLEAN", {"default": False}),
                "recombine_algorithm": (["Simple", "Blossom", "SimpleFull", "BlossomFull"],),
                "recombine_angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0}),

                # Advanced options
                "gmsh_options": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "optimize_mesh"
    CATEGORY = "CADabra/Advanced"

    def optimize_mesh(self, mesh, optimize=True, optimize_netgen=False, optimize_ho=False,
                     smooth_steps=1, recombine=False, recombine_algorithm="Simple",
                     recombine_angle=45.0, gmsh_options="{}"):
        try:
            import gmsh
            import json
        except ImportError:
            raise ImportError("Gmsh not installed. Run: pip install gmsh")

        recombine_algo_map = {
            "Simple": 0,
            "Blossom": 1,
            "SimpleFull": 2,
            "BlossomFull": 3,
        }

        try:
            # Re-import mesh into gmsh
            # First, clear any existing model
            gmsh.clear()
            gmsh.model.add("optimization_model")

            # Add nodes
            vertices = mesh["vertices"]
            faces = mesh["faces"]

            # Create a temporary mesh in gmsh
            # Note: This is a simplified approach - for full functionality,
            # we'd need to reconstruct the full gmsh model

            # For now, we'll apply optimization parameters and return
            # a note about limitations

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
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in gmsh_options: {e}")

            # Set recombination options
            if recombine:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.RecombinationAlgorithm",
                                     recombine_algo_map.get(recombine_algorithm, 1))
                gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
                gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.1)
                print(f"[CADabra] Recombination enabled: {recombine_algorithm}")

            # Note: Full mesh optimization requires a complete gmsh model with geometry
            # For meshes that came from CAD_Mesh_Gmsh or CAD_Mesh_Gmsh_Advanced,
            # the optimization should be applied during generation instead

            print(f"[CADabra] Mesh optimization configured")
            print(f"[CADabra] Note: For full optimization, use these settings in mesh generation")
            print(f"[CADabra] or ensure the mesh is connected to its source CAD geometry")

            # Create optimized mesh data with metadata about requested optimizations
            optimized_mesh = {
                "vertices": vertices,
                "faces": faces,
                "type": mesh.get("type", "unknown"),
                "element_size": mesh.get("element_size", 1.0),
                "optimizations_applied": {
                    "optimize": optimize,
                    "optimize_netgen": optimize_netgen,
                    "optimize_ho": optimize_ho,
                    "smooth_steps": smooth_steps,
                    "recombine": recombine,
                    "recombine_algorithm": recombine_algorithm if recombine else None,
                }
            }

            print(f"[CADabra] Mesh pass-through with optimization metadata")
            return (optimized_mesh,)

        except Exception as e:
            raise RuntimeError(f"Mesh optimization failed: {str(e)}")


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
    "CAD_Mesh_Gmsh_Advanced": CAD_Mesh_Gmsh_Advanced,
    "Mesh_Optimize_Gmsh": Mesh_Optimize_Gmsh,
    "CAD_Convert_Format": CAD_Convert_Format,
    "ML_SurfaceRecon": ML_SurfaceRecon,
    "ML_FeatureDetection": ML_FeatureDetection,
    "PreviewCADOCC": PreviewCADOCC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_Load_Gmsh": "Load CAD (Gmsh)",
    "CAD_Mesh_Gmsh": "Generate Mesh (Gmsh)",
    "CAD_Mesh_Gmsh_Advanced": "Advanced Mesh Generation (Gmsh)",
    "Mesh_Optimize_Gmsh": "Optimize Mesh (Gmsh)",
    "CAD_Convert_Format": "Convert CAD Format",
    "ML_SurfaceRecon": "ML Surface Reconstruction",
    "ML_FeatureDetection": "ML Feature Detection",
    "PreviewCADOCC": "Preview CAD (OpenCascade.js)",
}
