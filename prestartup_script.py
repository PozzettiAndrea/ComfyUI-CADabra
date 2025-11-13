"""
CADabra PreStartup Script
Copies example CAD files to ComfyUI's input folder on startup.
"""
import os
import shutil

def setup_example_assets():
    """Copy example CAD files to ComfyUI input folder if not already present."""
    try:
        import folder_paths

        # Get paths
        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # List of example CAD files to copy (only Gmsh-supported formats)
        example_files = [
            "spiral_wind_turbine.stp",  # STEP format - wind turbine model
            "bellow_pipe.igs",           # IGES format - bellows pipe model
            # Note: Impeller.SLDPRT is skipped - SolidWorks format not supported by Gmsh
        ]

        # Copy each example file
        for filename in example_files:
            source_file = os.path.join(custom_node_dir, "assets", filename)
            dest_file = os.path.join(input_folder, filename)

            # Copy if source exists and destination doesn't
            if os.path.exists(source_file):
                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    print(f"[CADabra] Copied {filename} to input folder")
                else:
                    print(f"[CADabra] {filename} already exists in input folder")
            else:
                print(f"[CADabra] Warning: assets/{filename} not found")

    except Exception as e:
        print(f"[CADabra] Error setting up example assets: {e}")

# Run on import
setup_example_assets()
