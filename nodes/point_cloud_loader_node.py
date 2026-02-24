import logging
import trimesh
import os
import folder_paths

log = logging.getLogger("cadabra")

class LoadPointCloudXYZ:
    """
    Loads a point cloud from an XYZ file.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = os.path.join(folder_paths.get_input_directory(), 'cad')
        xyz_files = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".xyz"):
                xyz_files.append(filename)
        xyz_files.sort()

        if not xyz_files:
            xyz_files = ["(no .xyz files found in input folder)"]

        return {
            "required": {
                "filename": (xyz_files, {
                    "tooltip": "Select .xyz file from ComfyUI/input folder"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("point_cloud_mesh",)
    FUNCTION = "load_xyz"
    CATEGORY = "CADabra/Point2CAD"

    def load_xyz(self, filename):
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
            return (mesh,)
        except Exception as e:
            raise RuntimeError(f"Failed to load .xyz file {file_path}: {e}")

NODE_CLASS_MAPPINGS = {
    "LoadPointCloudXYZ": LoadPointCloudXYZ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPointCloudXYZ": "Load Point Cloud (XYZ)",
}
