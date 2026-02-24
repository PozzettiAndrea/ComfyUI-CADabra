"""
CAD Curvature node for ComfyUI-CADabra
Computes surface curvature directly from CAD geometry using BRepLProp_SLProps.
Outputs a TRIMESH with curvature stored in vertex_attributes for visualization
with ComfyUI-GeometryPack's Preview Mesh (VTK) node.
"""

import numpy as np
import trimesh

from ..utils.occ_logging import logger

# OCC imports for curvature computation
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Precision import precision


class CADCurvature:
    """
    Compute surface curvature from CAD geometry.

    Computes Gaussian, Mean, Min, and Max principal curvatures directly from
    CAD surfaces using BRepLProp_SLProps. Outputs a TRIMESH with curvature
    values stored in vertex_attributes.

    Use with GeometryPack's "Preview Mesh (VTK)" node for visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL", {
                    "tooltip": "CAD model to analyze"
                }),
                "linear_deflection": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Linear deflection for tessellation (lower = finer mesh)"
                }),
                "angular_deflection": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Angular deflection in radians for tessellation"
                }),
            },
            "optional": {
                "clamp_percentile": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Percentile clamping for outliers (0=no clamping, 1=P1-P99)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "CAD_MODEL", "STRING")
    RETURN_NAMES = ("mesh", "cad_model", "report")
    FUNCTION = "compute_curvature"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Compute surface curvature from CAD geometry.

    Outputs a TRIMESH with curvature in vertex_attributes:
    - gaussian: K = k1 * k2 (positive on spheres, negative on saddles)
    - mean: H = (k1 + k2) / 2
    - min_curvature: k1 (minimum principal curvature)
    - max_curvature: k2 (maximum principal curvature)
    - face_id: CAD face index for each vertex

    Connect to GeometryPack's "Preview Mesh (VTK)" for 3D visualization.
    """

    def compute_curvature(self, cad_model, linear_deflection, angular_deflection,
                          clamp_percentile=1.0):
        """Compute curvature and return as TRIMESH with vertex attributes."""
        # Get OCC shape
        occ_shape = cad_model.get("occ_shape") or cad_model.get("shape")
        if occ_shape is None:
            raise RuntimeError("CAD model has no OCC shape")

        # Get bounding box for report
        bbox = Bnd_Box()
        brepbndlib.Add(occ_shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = 0, 0, 0, 0, 0, 0
        bounds_min = [xmin, ymin, zmin]
        bounds_max = [xmax, ymax, zmax]

        # Count faces
        num_faces = sum(1 for _ in self._iter_occ(occ_shape, TopAbs_FACE))
        logger.info(f"[CADCurvature] Processing {num_faces} faces")

        # Tessellate the shape
        logger.info(f"[CADCurvature] Tessellating with deflection={linear_deflection}")
        mesh = BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()

        # Collect all vertices, faces, and curvatures
        all_vertices = []
        all_faces = []
        all_gaussian = []
        all_mean = []
        all_min = []
        all_max = []
        all_face_ids = []  # Per-vertex face ID

        undefined_count = 0
        vertex_offset = 0

        # Process each CAD face
        for face_idx, face in enumerate(self._iter_occ(occ_shape, TopAbs_FACE)):
            occ_face = topods.Face(face)
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(occ_face, loc)

            if triangulation is None:
                logger.warning(f"[CADCurvature] Face {face_idx}: no triangulation")
                continue

            trsf = loc.Transformation()
            num_nodes = triangulation.NbNodes()
            num_tris = triangulation.NbTriangles()

            # Get surface adaptor for curvature computation
            adaptor = BRepAdaptor_Surface(occ_face)
            props = BRepLProp_SLProps(adaptor, 2, precision.PConfusion())

            # Process vertices
            for i in range(1, num_nodes + 1):
                # Get 3D point
                pnt = triangulation.Node(i).Transformed(trsf)
                all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

                # Get UV parameters
                uv = triangulation.UVNode(i)
                u, v = uv.X(), uv.Y()

                # Compute curvature at this point
                props.SetParameters(u, v)

                if props.IsCurvatureDefined():
                    g = props.GaussianCurvature()
                    m = props.MeanCurvature()
                    k_min = props.MinCurvature()
                    k_max = props.MaxCurvature()

                    # Handle numerical issues
                    if np.isnan(g) or np.isinf(g):
                        g = 0.0
                    if np.isnan(m) or np.isinf(m):
                        m = 0.0
                    if np.isnan(k_min) or np.isinf(k_min):
                        k_min = 0.0
                    if np.isnan(k_max) or np.isinf(k_max):
                        k_max = 0.0
                else:
                    g = m = k_min = k_max = 0.0
                    undefined_count += 1

                all_gaussian.append(g)
                all_mean.append(m)
                all_min.append(k_min)
                all_max.append(k_max)
                all_face_ids.append(face_idx)

            # Process triangles (convert to 0-indexed)
            for i in range(1, num_tris + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                all_faces.append([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset
                ])

            vertex_offset += num_nodes

        # Convert to numpy arrays
        vertices = np.array(all_vertices, dtype=np.float64)
        faces = np.array(all_faces, dtype=np.int32)
        gaussian = np.array(all_gaussian, dtype=np.float32)
        mean = np.array(all_mean, dtype=np.float32)
        min_curv = np.array(all_min, dtype=np.float32)
        max_curv = np.array(all_max, dtype=np.float32)
        face_ids = np.array(all_face_ids, dtype=np.float32)

        logger.info(f"[CADCurvature] Processed {len(vertices)} vertices, {len(faces)} triangles")
        logger.info(f"[CADCurvature] Undefined curvature at {undefined_count} vertices")

        # Compute statistics
        curvature_stats = {}
        for name, values in [
            ("gaussian", gaussian),
            ("mean", mean),
            ("min", min_curv),
            ("max", max_curv),
        ]:
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                raw_min, raw_max = valid.min(), valid.max()
                if clamp_percentile > 0:
                    p_low = np.percentile(valid, clamp_percentile)
                    p_high = np.percentile(valid, 100 - clamp_percentile)
                else:
                    p_low, p_high = raw_min, raw_max

                curvature_stats[name] = {
                    "raw_min": float(raw_min),
                    "raw_max": float(raw_max),
                    "clamped_min": float(p_low),
                    "clamped_max": float(p_high),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                }

        # Create trimesh object
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Store curvatures as vertex attributes (for GeometryPack VTK viewer)
        tm.vertex_attributes['gaussian'] = gaussian
        tm.vertex_attributes['mean'] = mean
        tm.vertex_attributes['min_curvature'] = min_curv
        tm.vertex_attributes['max_curvature'] = max_curv
        tm.vertex_attributes['face_id'] = face_ids

        # Store stats in metadata
        tm.metadata['curvature_stats'] = curvature_stats
        tm.metadata['num_cad_faces'] = num_faces
        tm.metadata['undefined_count'] = undefined_count

        # Build report
        report = self._build_report(
            num_faces, len(vertices), len(faces),
            undefined_count, bounds_min, bounds_max,
            curvature_stats, clamp_percentile
        )

        logger.info(f"[CADCurvature] Output TRIMESH with vertex_attributes: {list(tm.vertex_attributes.keys())}")

        return (tm, cad_model, report)

    def _iter_occ(self, shape, topology_type):
        """Helper to iterate over OCC TopExp_Explorer results."""
        explorer = TopExp_Explorer(shape, topology_type)
        while explorer.More():
            yield explorer.Current()
            explorer.Next()

    def _build_report(self, num_faces, num_vertices, num_triangles,
                      undefined_count, bounds_min, bounds_max,
                      curvature_stats, clamp_percentile):
        """Build text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("CAD CURVATURE REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append("--- MESH STATISTICS ---")
        lines.append(f"CAD Faces: {num_faces}")
        lines.append(f"Vertices: {num_vertices:,}")
        lines.append(f"Triangles: {num_triangles:,}")
        lines.append(f"Undefined curvature: {undefined_count} vertices")
        lines.append("")

        lines.append("--- BOUNDING BOX ---")
        lines.append(f"Min: ({bounds_min[0]:.2f}, {bounds_min[1]:.2f}, {bounds_min[2]:.2f})")
        lines.append(f"Max: ({bounds_max[0]:.2f}, {bounds_max[1]:.2f}, {bounds_max[2]:.2f})")
        lines.append("")

        lines.append("--- CURVATURE STATISTICS ---")
        lines.append(f"Percentile clamping: P{clamp_percentile}-P{100-clamp_percentile}")
        lines.append("")

        for name in ["gaussian", "mean", "min", "max"]:
            stats = curvature_stats.get(name, {})
            if stats:
                lines.append(f"{name.upper()}:")
                lines.append(f"  Raw:     [{stats['raw_min']:.4g}, {stats['raw_max']:.4g}]")
                lines.append(f"  Clamped: [{stats['clamped_min']:.4g}, {stats['clamped_max']:.4g}]")
                lines.append(f"  Mean: {stats['mean']:.4g}, Std: {stats['std']:.4g}")
        lines.append("")

        lines.append("--- VERTEX ATTRIBUTES ---")
        lines.append("gaussian, mean, min_curvature, max_curvature, face_id")
        lines.append("")
        lines.append("Use GeometryPack 'Preview Mesh (VTK)' to visualize.")
        lines.append("=" * 60)

        return "\n".join(lines)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADCurvature": CADCurvature,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADCurvature": "CAD Curvature",
}
