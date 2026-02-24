"""
CAD ROI Selector Node
Interactive node for selecting a Region of Interest on a CAD model.
"""

import os
import numpy as np
import folder_paths

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from .cad_common import get_occ_shape
from ..utils.occ_logging import log_operation


def generate_face_colors(num_faces, seed=42):
    """Generate distinct random colors for each face using golden ratio hue distribution."""
    import random
    import colorsys

    rng = random.Random(seed)

    # Use golden ratio for maximally distinct hues
    golden_ratio = 0.618033988749895

    # Start with random hue, then use golden ratio to space them out
    hue = rng.random()

    colors = []
    for i in range(num_faces):
        # Vary saturation and lightness slightly for more visual interest
        saturation = 0.6 + rng.random() * 0.3  # 0.6-0.9
        lightness = 0.45 + rng.random() * 0.2   # 0.45-0.65

        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append([r, g, b])

        # Advance hue by golden ratio for maximum separation
        hue = (hue + golden_ratio) % 1.0

    # Shuffle to avoid adjacent faces having similar colors due to iteration order
    rng.shuffle(colors)

    return colors


class CADROISelector:
    """
    Interactive node for selecting a Region of Interest on a CAD model.
    Displays the model in an iframe and allows click-drag rectangle selection.
    Outputs ROI in world coordinates (X1,Y1,X2,Y2 format).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "linear_deflection": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Tessellation linear deflection (lower = finer mesh)"
                }),
                "angular_deflection": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 3.14,
                    "step": 0.01,
                    "tooltip": "Tessellation angular deflection in radians"
                }),
                "roi_value": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Selected ROI (X1,Y1,X2,Y2) - updated by clicking on viewer"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("roi",)
    FUNCTION = "select_roi"
    CATEGORY = "CADabra/Analysis"
    OUTPUT_NODE = True

    def select_roi(self, cad_model, linear_deflection, angular_deflection, roi_value=""):
        print(f"[CADabra ROI] select_roi called with roi_value='{roi_value}'")
        shape = get_occ_shape(cad_model)

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        if bbox.IsVoid():
            raise RuntimeError("CAD model has no geometry (empty bounding box)")
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        # Tessellate with BRepMesh
        with log_operation("BRepMesh ROI Selector", linear_deflection=linear_deflection, angular_deflection=angular_deflection):
            mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
            mesh.Perform()

        if not mesh.IsDone():
            raise RuntimeError("BRepMesh tessellation failed")

        # Extract triangulated mesh from all faces WITH face IDs
        all_vertices = []
        all_faces = []
        vertex_colors = []  # RGB per vertex
        triangle_face_ids = []  # CAD face ID per triangle (for click detection)
        face_bboxes = []  # Bounding box per CAD face
        vertex_offset = 0
        face_id = 0

        # First pass: count faces for color generation
        face_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_count += 1
            explorer.Next()

        # Generate colors for each face
        colors = generate_face_colors(face_count)

        # Second pass: extract mesh with face IDs
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)

            # Get per-face bbox
            face_bbox = Bnd_Box()
            brepbndlib.Add(face, face_bbox)
            if not face_bbox.IsVoid():
                fxmin, fymin, fzmin, fxmax, fymax, fzmax = face_bbox.Get()
                face_bboxes.append({
                    "id": face_id,
                    "min": [fxmin, fymin, fzmin],
                    "max": [fxmax, fymax, fzmax]
                })
            else:
                face_bboxes.append({
                    "id": face_id,
                    "min": [0, 0, 0],
                    "max": [0, 0, 0]
                })

            if triangulation is not None:
                trsf = location.Transformation()
                color = colors[face_id]

                for i in range(1, triangulation.NbNodes() + 1):
                    pnt = triangulation.Node(i)
                    pnt.Transform(trsf)
                    all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                    vertex_colors.append(color)

                for i in range(1, triangulation.NbTriangles() + 1):
                    triangle = triangulation.Triangle(i)
                    n1, n2, n3 = triangle.Get()
                    all_faces.append([
                        vertex_offset + n1 - 1,
                        vertex_offset + n2 - 1,
                        vertex_offset + n3 - 1
                    ])
                    triangle_face_ids.append(face_id)  # Track which CAD face this triangle belongs to

                vertex_offset += triangulation.NbNodes()

            face_id += 1
            explorer.Next()

        if len(all_vertices) == 0:
            raise RuntimeError("Tessellation produced no vertices")

        vertices_array = np.array(all_vertices, dtype=np.float32)
        faces_array = np.array(all_faces, dtype=np.int32)
        colors_array = np.array(vertex_colors, dtype=np.float32)

        mesh_vertex_count = len(vertices_array)
        mesh_face_count = len(faces_array)

        print(f"[CADabra] ROI Selector: {mesh_vertex_count} vertices, {mesh_face_count} triangles, {face_count} CAD faces")

        # Export to GLB for viewer (with vertex colors)
        glb_filename = f"roi_selector_{id(cad_model)}.glb"
        glb_filepath = os.path.join(folder_paths.get_temp_directory(), glb_filename)

        self._export_glb_with_colors(vertices_array, faces_array, colors_array, glb_filepath)

        file_size_kb = os.path.getsize(glb_filepath) / 1024
        print(f"[CADabra] ROI Selector mesh exported: {glb_filename} ({file_size_kb:.1f} KB)")

        # Build UI data
        ui_data = {
            "glb_file": [glb_filename],
            "bounds_min": [[xmin, ymin, zmin]],
            "bounds_max": [[xmax, ymax, zmax]],
            "current_roi": [roi_value if roi_value else ""],
            "mesh_vertex_count": [mesh_vertex_count],
            "mesh_face_count": [mesh_face_count],
            "face_bboxes": [face_bboxes],  # Per-face bounding boxes
            "triangle_face_ids": [triangle_face_ids],  # Map triangle index -> CAD face ID
        }

        print(f"[CADabra ROI] Returning roi_value='{roi_value}'")

        return {"ui": ui_data, "result": (roi_value,)}

    def _export_glb(self, vertices_array, faces_array, filepath):
        """Export mesh to GLB format using pygltflib."""
        from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
        from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
        from pygltflib import FLOAT, UNSIGNED_INT
        from pygltflib import TRIANGLES

        # Flatten indices to uint32
        indices_flat = faces_array.flatten().astype(np.uint32)

        # Create binary buffers
        vertices_bytes = vertices_array.tobytes()
        indices_bytes = indices_flat.tobytes()

        # Concatenate binary data
        buffer_data = vertices_bytes + indices_bytes

        # Create glTF structure
        gltf = GLTF2()
        gltf.scenes = [Scene(nodes=[0])]
        gltf.nodes = [Node(mesh=0)]
        gltf.meshes = [Mesh(primitives=[Primitive(
            attributes={"POSITION": 0},
            indices=1,
            mode=TRIANGLES
        )])]

        # Buffer (contains all binary data)
        gltf.buffers = [Buffer(byteLength=len(buffer_data))]

        # BufferViews (slices of the buffer)
        vertices_offset = 0
        indices_offset = len(vertices_bytes)

        gltf.bufferViews = [
            BufferView(buffer=0, byteOffset=vertices_offset, byteLength=len(vertices_bytes), target=ARRAY_BUFFER),
            BufferView(buffer=0, byteOffset=indices_offset, byteLength=len(indices_bytes), target=ELEMENT_ARRAY_BUFFER)
        ]

        # Accessors (describe the data)
        gltf.accessors = [
            Accessor(bufferView=0, componentType=FLOAT, count=len(vertices_array), type="VEC3",
                    min=vertices_array.min(axis=0).tolist(), max=vertices_array.max(axis=0).tolist()),
            Accessor(bufferView=1, componentType=UNSIGNED_INT, count=len(indices_flat), type="SCALAR")
        ]

        # Set binary data and save
        gltf.set_binary_blob(buffer_data)
        gltf.save(filepath)

    def _export_glb_with_colors(self, vertices_array, faces_array, colors_array, filepath):
        """Export mesh to GLB format with vertex colors using pygltflib."""
        from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
        from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
        from pygltflib import FLOAT, UNSIGNED_INT
        from pygltflib import TRIANGLES

        # Flatten indices to uint32
        indices_flat = faces_array.flatten().astype(np.uint32)

        # Create binary buffers
        vertices_bytes = vertices_array.tobytes()
        colors_bytes = colors_array.tobytes()
        indices_bytes = indices_flat.tobytes()

        # Concatenate binary data
        buffer_data = vertices_bytes + colors_bytes + indices_bytes

        # Create glTF structure
        gltf = GLTF2()
        gltf.scenes = [Scene(nodes=[0])]
        gltf.nodes = [Node(mesh=0)]
        gltf.meshes = [Mesh(primitives=[Primitive(
            attributes={"POSITION": 0, "COLOR_0": 1},
            indices=2,
            mode=TRIANGLES
        )])]

        # Buffer (contains all binary data)
        gltf.buffers = [Buffer(byteLength=len(buffer_data))]

        # BufferViews (slices of the buffer)
        vertices_offset = 0
        colors_offset = len(vertices_bytes)
        indices_offset = colors_offset + len(colors_bytes)

        gltf.bufferViews = [
            BufferView(buffer=0, byteOffset=vertices_offset, byteLength=len(vertices_bytes), target=ARRAY_BUFFER),
            BufferView(buffer=0, byteOffset=colors_offset, byteLength=len(colors_bytes), target=ARRAY_BUFFER),
            BufferView(buffer=0, byteOffset=indices_offset, byteLength=len(indices_bytes), target=ELEMENT_ARRAY_BUFFER)
        ]

        # Accessors (describe the data)
        gltf.accessors = [
            Accessor(bufferView=0, componentType=FLOAT, count=len(vertices_array), type="VEC3",
                    min=vertices_array.min(axis=0).tolist(), max=vertices_array.max(axis=0).tolist()),
            Accessor(bufferView=1, componentType=FLOAT, count=len(colors_array), type="VEC3"),
            Accessor(bufferView=2, componentType=UNSIGNED_INT, count=len(indices_flat), type="SCALAR")
        ]

        # Set binary data and save
        gltf.set_binary_blob(buffer_data)
        gltf.save(filepath)


NODE_CLASS_MAPPINGS = {
    "CADROISelector": CADROISelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADROISelector": "CAD ROI Selector",
}
