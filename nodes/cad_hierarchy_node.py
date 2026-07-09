from __future__ import annotations

import json
import os
import time
import folder_paths
from comfy_api.latest import io
from .utils.occ_logging import logger

from OCC.Core.TopAbs import (
    TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID,
    TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE,
    TopAbs_EDGE, TopAbs_VERTEX
)
from OCC.Core.TopoDS import TopoDS_Iterator, topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GCPnts import GCPnts_UniformDeflection
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
import vtk

def _get_shape_type_names():
    """Get shape type names dict - called at runtime when OCC is available."""
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID,
        TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE,
        TopAbs_EDGE, TopAbs_VERTEX
    )
    return {
        TopAbs_COMPOUND: "COMPOUND",
        TopAbs_COMPSOLID: "COMPSOLID",
        TopAbs_SOLID: "SOLID",
        TopAbs_SHELL: "SHELL",
        TopAbs_FACE: "FACE",
        TopAbs_WIRE: "WIRE",
        TopAbs_EDGE: "EDGE",
        TopAbs_VERTEX: "VERTEX",
    }


def _get_surface_type_names():
    """Get surface type names dict - called at runtime when OCC is available."""
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
    )
    return {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier",
        GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_OtherSurface: "Other",
    }


def _get_curve_type_names():
    """Get curve type names dict - called at runtime when OCC is available."""
    from OCC.Core.GeomAbs import (
        GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
        GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
    )
    return {
        GeomAbs_Line: "Line",
        GeomAbs_Circle: "Circle",
        GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola",
        GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "Bezier",
        GeomAbs_BSplineCurve: "BSpline",
        GeomAbs_OtherCurve: "Other",
    }

class CADHierarchyTree(io.ComfyNode):
    """
    Visualizes the topological structure of a CAD model as a collapsible tree.

    Traverses the TopoDS_Shape hierarchy (COMPOUND -> SOLID -> SHELL -> FACE -> WIRE -> EDGE -> VERTEX)
    and outputs a JSON tree structure with entity counts and type information.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADHierarchyTree",
            display_name="CAD Hierarchy Tree",
            category="CADabra/Analysis",
            is_output_node=True,
            description="""
    Visualizes the topological structure of a CAD model as a collapsible tree.

    Shows the hierarchy: COMPOUND -> SOLID -> SHELL -> FACE -> WIRE -> EDGE -> VERTEX
    with surface types for faces and curve types for edges.

    Outputs:
    - json_filepath: Path to JSON file with full tree structure
    - summary_text: Quick summary of entity counts
    """,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model to analyze"),
                io.Int.Input("max_depth", default=-1, min=-1, max=20, step=1,
                             tooltip="Maximum depth to traverse (-1 = unlimited)", optional=True),
                io.Boolean.Input("include_properties", default=True,
                                 tooltip="Include surface/curve types for faces and edges", optional=True),
                io.Boolean.Input("collapse_vertices", default=True,
                                 tooltip="Collapse vertices into a count instead of listing each one", optional=True),
                io.Float.Input("linear_deflection", default=0.1, min=0.001, max=10.0, step=0.01,
                               tooltip="Tessellation linear deflection for the 3D preview mesh (lower = finer)", optional=True),
                io.Float.Input("angular_deflection", default=0.5, min=0.01, max=3.14, step=0.01,
                               tooltip="Tessellation angular deflection for the 3D preview mesh, in radians", optional=True),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
                io.String.Output(display_name="json_filepath"),
                io.String.Output(display_name="summary_text"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, max_depth=-1, include_properties=True, collapse_vertices=True,
                linear_deflection=0.1, angular_deflection=0.5):
        """Build the hierarchy tree from a CAD model."""

        # Get OCC shape from brep_path
        try:
            from .utils.brep_cache import get_occ_shape
        except ImportError:
            from .utils.brep_cache import get_occ_shape
        occ_shape = get_occ_shape(cad_model)

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_hierarchy_{timestamp}"
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(output_dir, json_filename)

        # Initialize counters
        counters = {
            "COMPOUND": 0,
            "COMPSOLID": 0,
            "SOLID": 0,
            "SHELL": 0,
            "FACE": 0,
            "WIRE": 0,
            "EDGE": 0,
            "VERTEX": 0,
        }
        surface_types = {}
        curve_types = {}
        node_index = [0]

        # Ordered shape lists, appended as a side effect of the tree DFS below so that
        # a FACE/EDGE/VERTEX node's "geom_id" is guaranteed to be its own index into the
        # matching list here -- tessellation then just walks these lists directly instead
        # of re-exploring the shape, so tree geom_id and mesh/edge ids can never drift apart.
        face_shapes = []
        edge_shapes = []
        vertex_shapes = []

        # Build tree recursively
        tree = _build_tree_node(
            occ_shape,
            depth=0,
            max_depth=max_depth,
            include_properties=include_properties,
            collapse_vertices=collapse_vertices,
            counters=counters,
            surface_types=surface_types,
            curve_types=curve_types,
            node_index=node_index,
            face_shapes=face_shapes,
            edge_shapes=edge_shapes,
            vertex_shapes=vertex_shapes,
        )

        # Tessellate for the 3D preview viewport. face_id/edge_id in these VTPs are
        # just the index into face_shapes/edge_shapes -- i.e. exactly the "geom_id"
        # already stamped on the matching tree node above.
        mesh_filename = None
        edges_filename = None
        if face_shapes:
            BRepMesh_IncrementalMesh(occ_shape, linear_deflection, False, angular_deflection)
            mesh_filename = _export_face_mesh_vtp(face_shapes, base_filename, output_dir)
        if edge_shapes:
            edges_filename = _export_edge_vtp(edge_shapes, base_filename, output_dir, linear_deflection)

        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
        bbox = Bnd_Box()
        brepbndlib.Add(occ_shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            bounds_min = [xmin, ymin, zmin]
            bounds_max = [xmax, ymax, zmax]

        # Build summary
        summary = {
            "total_compounds": counters["COMPOUND"],
            "total_compsolids": counters["COMPSOLID"],
            "total_solids": counters["SOLID"],
            "total_shells": counters["SHELL"],
            "total_faces": counters["FACE"],
            "total_wires": counters["WIRE"],
            "total_edges": counters["EDGE"],
            "total_vertices": counters["VERTEX"],
            "surface_types": surface_types,
            "curve_types": curve_types,
        }

        # Full output
        output_data = {
            "root": tree,
            "summary": summary,
            "timestamp": timestamp,
        }

        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Build summary text
        summary_lines = [
            f"CAD Hierarchy Summary:",
            f"  Compounds: {counters['COMPOUND']}",
            f"  Solids: {counters['SOLID']}",
            f"  Shells: {counters['SHELL']}",
            f"  Faces: {counters['FACE']}",
            f"  Wires: {counters['WIRE']}",
            f"  Edges: {counters['EDGE']}",
            f"  Vertices: {counters['VERTEX']}",
            f"",
            f"Surface Types:",
        ]
        for st, count in sorted(surface_types.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {st}: {count}")

        summary_lines.append("")
        summary_lines.append("Curve Types:")
        for ct, count in sorted(curve_types.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {ct}: {count}")

        summary_text = "\n".join(summary_lines)

        logger.info(f"[CADHierarchyTree] Built hierarchy with {counters['FACE']} faces, {counters['EDGE']} edges")

        return io.NodeOutput(
            cad_model, json_filename, summary_text,
            ui={
                "hierarchy_file": [json_filename],
                "num_faces": [counters["FACE"]],
                "num_edges": [counters["EDGE"]],
                "num_solids": [counters["SOLID"]],
                "mesh_file": [mesh_filename or ""],
                "edges_file": [edges_filename or ""],
                "bounds_min": [bounds_min],
                "bounds_max": [bounds_max],
            },
        )


def _build_tree_node(shape, depth, max_depth, include_properties, collapse_vertices,
                     counters, surface_types, curve_types, node_index,
                     face_shapes, edge_shapes, vertex_shapes):
    """Recursively build a tree node for a shape."""

    shape_type = shape.ShapeType()
    type_name = _get_shape_type_names().get(shape_type, "UNKNOWN")

    # Update counter
    if type_name in counters:
        counters[type_name] += 1

    # Create node
    node = {
        "type": type_name,
        "index": node_index[0],
        "children": [],
    }
    node_index[0] += 1

    # Add properties for faces and edges
    if include_properties:
        if shape_type == TopAbs_FACE:
            try:
                adaptor = BRepAdaptor_Surface(topods.Face(shape))
                surface_type = adaptor.GetType()
                surface_name = _get_surface_type_names().get(surface_type, "Unknown")
                node["properties"] = {"surface_type": surface_name}

                # Update surface type counter
                surface_types[surface_name] = surface_types.get(surface_name, 0) + 1
            except Exception:
                node["properties"] = {"surface_type": "Error"}

        elif shape_type == TopAbs_EDGE:
            try:
                adaptor = BRepAdaptor_Curve(topods.Edge(shape))
                curve_type = adaptor.GetType()
                curve_name = _get_curve_type_names().get(curve_type, "Unknown")
                node["properties"] = {"curve_type": curve_name}

                # Update curve type counter
                curve_types[curve_name] = curve_types.get(curve_name, 0) + 1
            except Exception:
                node["properties"] = {"curve_type": "Error"}

    # Stamp a geom_id = index into the matching ordered shape list, appended right here
    # so it can never drift from the tessellation step, which walks these same lists.
    if shape_type == TopAbs_FACE:
        node["geom_id"] = len(face_shapes)
        face_shapes.append(shape)
    elif shape_type == TopAbs_EDGE:
        node["geom_id"] = len(edge_shapes)
        edge_shapes.append(shape)
    elif shape_type == TopAbs_VERTEX:
        node["geom_id"] = len(vertex_shapes)
        vertex_shapes.append(shape)

    # Check depth limit
    if max_depth >= 0 and depth >= max_depth:
        # Count children without recursing
        child_count = 0
        iterator = TopoDS_Iterator(shape)
        while iterator.More():
            child_count += 1
            iterator.Next()
        if child_count > 0:
            node["children_count"] = child_count
            node["truncated"] = True
        return node

    # Special handling for vertices - collapse if requested
    if collapse_vertices and shape_type == TopAbs_WIRE:
        # For wires, only show edges, count vertices
        vertex_count = 0
        iterator = TopoDS_Iterator(shape)
        while iterator.More():
            child = iterator.Value()
            child_type = child.ShapeType()
            if child_type == TopAbs_VERTEX:
                vertex_count += 1
                counters["VERTEX"] += 1
            else:
                child_node = _build_tree_node(
                    child, depth + 1, max_depth, include_properties, collapse_vertices,
                    counters, surface_types, curve_types, node_index,
                    face_shapes, edge_shapes, vertex_shapes
                )
                node["children"].append(child_node)
            iterator.Next()
        if vertex_count > 0:
            node["vertex_count"] = vertex_count
        return node

    # Recurse into children
    iterator = TopoDS_Iterator(shape)
    while iterator.More():
        child = iterator.Value()

        # Skip vertices if collapsing
        if collapse_vertices and child.ShapeType() == TopAbs_VERTEX:
            counters["VERTEX"] += 1
            iterator.Next()
            continue

        child_node = _build_tree_node(
            child, depth + 1, max_depth, include_properties, collapse_vertices,
            counters, surface_types, curve_types, node_index,
            face_shapes, edge_shapes, vertex_shapes
        )
        node["children"].append(child_node)
        iterator.Next()

    return node


def _export_face_mesh_vtp(face_shapes, base_filename, output_dir):
    """Tessellate an ordered list of TopoDS_Face (BRepMesh_IncrementalMesh already
    run on the parent shape) into a VTP with a "FaceID" cell-data array, where
    FaceID is simply the face's index in face_shapes -- i.e. its tree geom_id."""
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    face_ids = vtk.vtkIntArray()
    face_ids.SetName("FaceID")

    vertex_offset = 0
    for face_id, face in enumerate(face_shapes):
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(topods.Face(face), loc)
        if triangulation is None:
            continue

        trsf = loc.Transformation()
        num_nodes = triangulation.NbNodes()
        for i in range(1, num_nodes + 1):
            pnt = triangulation.Node(i).Transformed(trsf)
            points.InsertNextPoint(pnt.X(), pnt.Y(), pnt.Z())

        for i in range(1, triangulation.NbTriangles() + 1):
            n1, n2, n3 = triangulation.Triangle(i).Get()
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, vertex_offset + n1 - 1)
            triangle.GetPointIds().SetId(1, vertex_offset + n2 - 1)
            triangle.GetPointIds().SetId(2, vertex_offset + n3 - 1)
            polys.InsertNextCell(triangle)
            face_ids.InsertNextValue(face_id)

        vertex_offset += num_nodes

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    polydata.GetCellData().AddArray(face_ids)

    mesh_filename = f"{base_filename}_mesh.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.join(output_dir, mesh_filename))
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()
    writer.Write()

    logger.info(f"[CADHierarchyTree] Mesh exported: {mesh_filename} "
                f"({points.GetNumberOfPoints()} verts, {polys.GetNumberOfCells()} tris)")
    return mesh_filename


def _discretize_edge(curve, deflection):
    """Discretize an edge curve into points for visualization."""
    points = []
    try:
        discretizer = GCPnts_UniformDeflection(curve, deflection)
        if discretizer.IsDone() and discretizer.NbPoints() >= 2:
            for i in range(1, discretizer.NbPoints() + 1):
                pnt = discretizer.Value(i)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])
    except Exception:
        try:
            start_pnt = curve.Value(curve.FirstParameter())
            end_pnt = curve.Value(curve.LastParameter())
            points = [
                [start_pnt.X(), start_pnt.Y(), start_pnt.Z()],
                [end_pnt.X(), end_pnt.Y(), end_pnt.Z()],
            ]
        except Exception:
            pass
    return points


def _export_edge_vtp(edge_shapes, base_filename, output_dir, linear_deflection):
    """Discretize an ordered list of TopoDS_Edge into a VTP of polylines (one
    vtkPolyLine cell per edge) with an "edge_id" cell-data array = the edge's
    index in edge_shapes -- i.e. its tree geom_id."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    edge_ids = vtk.vtkIntArray()
    edge_ids.SetName("edge_id")

    point_offset = 0
    for edge_id, edge in enumerate(edge_shapes):
        try:
            curve = BRepAdaptor_Curve(topods.Edge(edge))
            edge_points = _discretize_edge(curve, linear_deflection)
        except Exception:
            edge_points = []

        if len(edge_points) < 2:
            continue

        for pt in edge_points:
            points.InsertNextPoint(pt[0], pt[1], pt[2])

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(edge_points))
        for i in range(len(edge_points)):
            polyline.GetPointIds().SetId(i, point_offset + i)
        lines.InsertNextCell(polyline)
        edge_ids.InsertNextValue(edge_id)

        point_offset += len(edge_points)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().AddArray(edge_ids)

    edges_filename = f"{base_filename}_edges.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(os.path.join(output_dir, edges_filename))
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()
    writer.Write()

    logger.info(f"[CADHierarchyTree] Edges exported: {edges_filename} ({lines.GetNumberOfCells()} edges)")
    return edges_filename


# Node registration
NODE_CLASS_MAPPINGS = {
    "CADHierarchyTree": CADHierarchyTree,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADHierarchyTree": "CAD Hierarchy Tree",
}
