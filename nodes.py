"""
ComfyUI-CADabra Nodes
CAD file loading, meshing, and ML-based processing nodes
"""

import os
import tempfile
import numpy as np
import torch


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

        if not os.path.exists(cad_file_path):
            raise FileNotFoundError(f"CAD file not found: {cad_file_path}")

        ext = os.path.splitext(cad_file_path)[1].lower()
        if ext not in ['.step', '.stp', '.iges', '.igs', '.brep']:
            raise ValueError(f"Unsupported file format: {ext}")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
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
            gmsh.finalize()
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
            gmsh.finalize()
            return (mesh_data,)

        except Exception as e:
            gmsh.finalize()
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


class CAD_Viewer:
    """Interactive 3D CAD/Mesh viewer"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
            "optional": {
                "wireframe": ("BOOLEAN", {"default": False}),
                "color": ("STRING", {"default": "#888888"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_cad"
    CATEGORY = "CADabra"

    def view_cad(self, mesh, wireframe=False, color="#888888"):
        vertices = mesh["vertices"]
        faces = mesh["faces"]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; }}
                canvas {{ width: 100%; height: 100%; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
        </head>
        <body>
            <script>
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                const vertices = {vertices.flatten().tolist()};
                const faces = {faces.flatten().tolist()};

                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geometry.setIndex(faces);
                geometry.computeVertexNormals();

                const material = new THREE.MeshPhongMaterial({{
                    color: "{color}",
                    wireframe: {str(wireframe).lower()},
                    side: THREE.DoubleSide
                }});

                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);

                const light = new THREE.DirectionalLight(0xffffff, 1);
                light.position.set(5, 5, 5);
                scene.add(light);
                scene.add(new THREE.AmbientLight(0x404040));

                camera.position.z = 10;

                const controls = new THREE.OrbitControls(camera, renderer.domElement);

                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();

                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            temp_path = f.name

        print(f"[CADabra] Viewer HTML generated: {temp_path}")
        return {"ui": {"html": html}}


# Node registration
NODE_CLASS_MAPPINGS = {
    "CAD_Load_Gmsh": CAD_Load_Gmsh,
    "CAD_Mesh_Gmsh": CAD_Mesh_Gmsh,
    "ML_SurfaceRecon": ML_SurfaceRecon,
    "ML_FeatureDetection": ML_FeatureDetection,
    "CAD_Viewer": CAD_Viewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAD_Load_Gmsh": "Load CAD (Gmsh)",
    "CAD_Mesh_Gmsh": "Generate Mesh (Gmsh)",
    "ML_SurfaceRecon": "ML Surface Reconstruction",
    "ML_FeatureDetection": "ML Feature Detection",
    "CAD_Viewer": "CAD Viewer",
}
