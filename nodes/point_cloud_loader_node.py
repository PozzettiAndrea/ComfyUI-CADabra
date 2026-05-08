import logging
import trimesh
import os
import folder_paths

from comfy_api.latest import io

log = logging.getLogger("cadabra")

class LoadPointCloudXYZ(io.ComfyNode):
    """
    Loads a point cloud from an XYZ file.
    """
    @classmethod
    def define_schema(cls):
        # define_schema runs in comfy-env's metadata-scan subprocess at
        # registration time, before any user has dropped files into
        # input/cad/. Listing an unguarded path crashes the entire scan
        # with FileNotFoundError -> 0 nodes registered for the package.
        # Mirrors the defensive pattern in cad_nodes.py:_get_cad_files.
        input_dir = os.path.join(folder_paths.get_input_directory(), 'cad')
        xyz_files = []
        if os.path.isdir(input_dir):
            for filename in os.listdir(input_dir):
                if filename.lower().endswith(".xyz"):
                    xyz_files.append(filename)
            xyz_files.sort()

        if not xyz_files:
            xyz_files = ["(no .xyz files found in input folder)"]

        return io.Schema(
            node_id="LoadPointCloudXYZ",
            display_name="Load Point Cloud (XYZ)",
            category="CADabra/Point2CAD",
            inputs=[
                io.Combo.Input("filename", options=xyz_files,
                               tooltip="Select .xyz file from ComfyUI/input folder"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="point_cloud_mesh"),
            ],
        )

    @classmethod
    def execute(cls, filename):
        if filename.startswith("(no .xyz files found"):
            raise FileNotFoundError(
                "No .xyz files found in ComfyUI/input folder. "
                "Please add .xyz files to ComfyUI/input/"
            )

        file_path = os.path.join(folder_paths.get_input_directory(), 'cad', filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        log.info(f" Loading point cloud from: {file_path}")
        try:
            # trimesh.load can directly load .xyz files
            mesh = trimesh.load(file_path, file_type='xyz')
            log.info(f" Loaded {len(mesh.vertices)} points.")
            return io.NodeOutput(mesh)
        except Exception as e:
            raise RuntimeError(f"Failed to load .xyz file {file_path}: {e}")

NODE_CLASS_MAPPINGS = {
    "LoadPointCloudXYZ": LoadPointCloudXYZ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPointCloudXYZ": "Load Point Cloud (XYZ)",
}
