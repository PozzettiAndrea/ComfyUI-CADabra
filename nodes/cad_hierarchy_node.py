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
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
                io.String.Output(display_name="json_filepath"),
                io.String.Output(display_name="summary_text"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, max_depth=-1, include_properties=True, collapse_vertices=True):
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
        )

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
            },
        )


def _build_tree_node(shape, depth, max_depth, include_properties, collapse_vertices,
                     counters, surface_types, curve_types, node_index):
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
                    counters, surface_types, curve_types, node_index
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
            counters, surface_types, curve_types, node_index
        )
        node["children"].append(child_node)
        iterator.Next()

    return node


# Node registration
NODE_CLASS_MAPPINGS = {
    "CADHierarchyTree": CADHierarchyTree,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADHierarchyTree": "CAD Hierarchy Tree",
}
