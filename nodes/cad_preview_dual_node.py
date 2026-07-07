"""
Preview CAD Dual — side-by-side / overlay preview of TWO CAD models.

Modeled on GeometryPack's "Preview Mesh Dual" (shares the identical vtk-gltf.js + js/utils viewer
stack; the dual viewer HTML was copied to web/viewer_cad_dual.html). Each CAD_MODEL is tessellated
to a triangle mesh via the shared tessellate_occ_shape(); then:
  - side_by_side : each mesh exported separately (STL) -> two synchronized viewports
  - overlay      : both meshes combined into one VTP carrying a per-vertex ``mesh_id`` scalar
                   (0 = cad_1, 1 = cad_2). The companion JS auto-colours overlay by ``mesh_id``
                   through a diverging colormap, so the two CADs render in DIFFERENT COLOURS.
"""

import logging
import os
import uuid

import numpy as np
from comfy_api.latest import io

from .cad_common import get_occ_shape
from .cad_nodes import tessellate_occ_shape

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    import tempfile
    _OUTPUT_DIR = tempfile.gettempdir()

log = logging.getLogger("cadabra")


def _tessellate(cad_model, linear_deflection, angular_deflection):
    """CAD_MODEL -> (verts float32 [N,3], faces int32 [M,3])."""
    occ = get_occ_shape(cad_model)
    verts, faces = tessellate_occ_shape(occ, linear_deflection, angular_deflection)
    if verts is None or len(verts) == 0:
        raise RuntimeError("Tessellation produced no vertices (empty/invalid CAD).")
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def _write_stl(verts, faces, path):
    import trimesh
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path, file_type="stl")


def _write_vtp(verts, faces, path, mesh_id=None):
    """Write points + triangle polys as VTP, with an optional named point-data ``mesh_id`` scalar.
    Uses vtk directly (same dependency CADabra's PreviewCADOCC._export_vtp uses)."""
    import vtk
    from vtk.util import numpy_support

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(verts, np.float32), deep=True))

    n = len(faces)
    conn = np.hstack([np.full((n, 1), 3, np.int64), faces.astype(np.int64)]).ravel()
    polys = vtk.vtkCellArray()
    try:
        polys.SetCells(n, numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn), deep=True))
    except Exception:                                   # very old VTK — fall back to per-triangle
        for f in faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(f[0])); tri.GetPointIds().SetId(1, int(f[1])); tri.GetPointIds().SetId(2, int(f[2]))
            polys.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    if mesh_id is not None:
        arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh_id, np.float32), deep=True)
        arr.SetName("mesh_id")
        polydata.GetPointData().AddArray(arr)
        polydata.GetPointData().SetActiveScalars("mesh_id")

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    writer.Write()


class PreviewCADDual(io.ComfyNode):
    """Preview two CAD models side-by-side or overlaid (overlay colours them distinctly)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewCADDual",
            display_name="Preview CAD Dual",
            category="CADabra/Visualization",
            is_output_node=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_1"),
                io.Custom("CAD_MODEL").Input("cad_2"),
                io.Combo.Input("layout", options=["overlay", "side_by_side"], default="overlay", optional=True),
                io.Float.Input("opacity_1", default=1.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("opacity_2", default=0.6, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("linear_deflection", default=0.1, min=1e-4, max=1e6, step=0.05, optional=True,
                               tooltip="Tessellation quality: max distance between mesh and true surface (smaller = finer)."),
                io.Float.Input("angular_deflection", default=0.5, min=0.01, max=3.14, step=0.05, optional=True,
                               tooltip="Tessellation quality: max angle deviation (radians)."),
            ],
        )

    @classmethod
    def execute(cls, cad_1, cad_2, layout="overlay", opacity_1=1.0, opacity_2=0.6,
                linear_deflection=0.1, angular_deflection=0.5):
        v1, f1 = _tessellate(cad_1, linear_deflection, angular_deflection)
        v2, f2 = _tessellate(cad_2, linear_deflection, angular_deflection)
        log.info("Preview CAD Dual: cad_1 %d verts/%d tris, cad_2 %d verts/%d tris, layout=%s",
                 len(v1), len(f1), len(v2), len(f2), layout)

        ext1 = (v1.max(0) - v1.min(0)).tolist()
        ext2 = (v2.max(0) - v2.min(0)).tolist()
        pid = uuid.uuid4().hex[:8]

        if layout == "side_by_side":
            fn1 = f"preview_cad_dual_1_{pid}.stl"
            fn2 = f"preview_cad_dual_2_{pid}.stl"
            _write_stl(v1, f1, os.path.join(_OUTPUT_DIR, fn1))
            _write_stl(v2, f2, os.path.join(_OUTPUT_DIR, fn2))
            ui = {
                "layout": [layout],
                "mesh_1_file": [fn1], "mesh_2_file": [fn2],
                "vertex_count_1": [len(v1)], "vertex_count_2": [len(v2)],
                "face_count_1": [len(f1)], "face_count_2": [len(f2)],
                "extents_1": [ext1], "extents_2": [ext2],
                "is_watertight_1": [False], "is_watertight_2": [False],
                "opacity_1": [float(opacity_1)], "opacity_2": [float(opacity_2)],
                "common_fields": [[]],
            }
        else:  # overlay — combine + bake mesh_id so the two shapes are coloured distinctly
            verts = np.vstack([v1, v2]).astype(np.float32)
            faces = np.vstack([f1, f2 + len(v1)]).astype(np.int32)
            mesh_id = np.concatenate([np.zeros(len(v1), np.float32), np.ones(len(v2), np.float32)])
            fn = f"preview_cad_dual_overlay_{pid}.vtp"
            _write_vtp(verts, faces, os.path.join(_OUTPUT_DIR, fn), mesh_id=mesh_id)
            ui = {
                "layout": [layout],
                "mesh_file": [fn],
                "vertex_count_1": [len(v1)], "vertex_count_2": [len(v2)],
                "face_count_1": [len(f1)], "face_count_2": [len(f2)],
                "opacity_1": [float(opacity_1)], "opacity_2": [float(opacity_2)],
                "common_fields": [["mesh_id"]],
            }

        log.info("Preview CAD Dual ready")
        return io.NodeOutput(ui=ui)


NODE_CLASS_MAPPINGS = {
    "PreviewCADDual": PreviewCADDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewCADDual": "Preview CAD Dual",
}
