from importlib import import_module

_MODULES = [
    ".cad_nodes",
    ".mesh_recon_nodes",
    ".cad_analysis_nodes",
    ".cad_inspect_nodes",
    ".cad_edge_analysis_nodes",
    ".point_cloud_loader_node",
    ".cad_utility_nodes",
    ".cad_project_faces",
    ".raytracer_node",
    ".cad_roi_selector_node",
    ".cad_transform_nodes",
    ".cad_hierarchy_node",
    ".cad_spline_viewer_node",
    ".cad_edge_viewer_vtk_node",
    ".cad_curve_plotter_node",
    ".cad_curvature_viewer_node",
    ".cad_primitive_reconstruction_node",
    ".cad_preview_dual_node",
    ".coord_system_to_transform_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod_name in _MODULES:
    _mod = import_module(_mod_name, package=__name__)
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
