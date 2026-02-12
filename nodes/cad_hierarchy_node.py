"""
CAD Hierarchy Tree node for ComfyUI-CADabra
Visualizes the topological structure of a CAD model as a collapsible tree.
"""

import json
import os
import time
import folder_paths

from ..utils.occ_logging import logger

# OCC imports
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

# Shape type names
SHAPE_TYPE_NAMES = {
    TopAbs_COMPOUND: "COMPOUND",
    TopAbs_COMPSOLID: "COMPSOLID",
    TopAbs_SOLID: "SOLID",
    TopAbs_SHELL: "SHELL",
    TopAbs_FACE: "FACE",
    TopAbs_WIRE: "WIRE",
    TopAbs_EDGE: "EDGE",
    TopAbs_VERTEX: "VERTEX",
}

# Surface type names
SURFACE_TYPE_NAMES = {
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

# Curve type names
CURVE_TYPE_NAMES = {
    GeomAbs_Line: "Line",
    GeomAbs_Circle: "Circle",
    GeomAbs_Ellipse: "Ellipse",
    GeomAbs_Hyperbola: "Hyperbola",
    GeomAbs_Parabola: "Parabola",
    GeomAbs_BezierCurve: "Bezier",
    GeomAbs_BSplineCurve: "BSpline",
    GeomAbs_OtherCurve: "Other",
}


class CADHierarchyTree:
    """
    Visualizes the topological structure of a CAD model as a collapsible tree.

    Traverses the TopoDS_Shape hierarchy (COMPOUND → SOLID → SHELL → FACE → WIRE → EDGE → VERTEX)
    and outputs a JSON tree structure with entity counts and type information.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL", {
                    "tooltip": "CAD model to analyze"
                }),
            },
            "optional": {
                "max_depth": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Maximum depth to traverse (-1 = unlimited)"
                }),
                "include_properties": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include surface/curve types for faces and edges"
                }),
                "collapse_vertices": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Collapse vertices into a count instead of listing each one"
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("cad_model", "json_filepath", "summary_text")
    OUTPUT_NODE = True
    FUNCTION = "build_hierarchy"
    CATEGORY = "CADabra/Analysis"
    DESCRIPTION = """
    Visualizes the topological structure of a CAD model as a collapsible tree.

    Shows the hierarchy: COMPOUND → SOLID → SHELL → FACE → WIRE → EDGE → VERTEX
    with surface types for faces and curve types for edges.

    Outputs:
    - json_filepath: Path to JSON file with full tree structure
    - summary_text: Quick summary of entity counts
    """

    def build_hierarchy(self, cad_model, max_depth=-1, include_properties=True, collapse_vertices=True):
        """Build the hierarchy tree from a CAD model."""

        # Get OCC shape
        occ_shape = cad_model.get("occ_shape") or cad_model.get("shape")
        if occ_shape is None:
            raise RuntimeError("CAD model has no OCC shape")

        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time() * 1000)
        base_filename = f"cad_hierarchy_{timestamp}"
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(output_dir, json_filename)

        # Initialize counters
        self.counters = {
            "COMPOUND": 0,
            "COMPSOLID": 0,
            "SOLID": 0,
            "SHELL": 0,
            "FACE": 0,
            "WIRE": 0,
            "EDGE": 0,
            "VERTEX": 0,
        }
        self.surface_types = {}
        self.curve_types = {}
        self.node_index = 0

        # Build tree recursively
        tree = self._build_tree_node(
            occ_shape,
            depth=0,
            max_depth=max_depth,
            include_properties=include_properties,
            collapse_vertices=collapse_vertices
        )

        # Build summary
        summary = {
            "total_compounds": self.counters["COMPOUND"],
            "total_compsolids": self.counters["COMPSOLID"],
            "total_solids": self.counters["SOLID"],
            "total_shells": self.counters["SHELL"],
            "total_faces": self.counters["FACE"],
            "total_wires": self.counters["WIRE"],
            "total_edges": self.counters["EDGE"],
            "total_vertices": self.counters["VERTEX"],
            "surface_types": self.surface_types,
            "curve_types": self.curve_types,
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
            f"  Compounds: {self.counters['COMPOUND']}",
            f"  Solids: {self.counters['SOLID']}",
            f"  Shells: {self.counters['SHELL']}",
            f"  Faces: {self.counters['FACE']}",
            f"  Wires: {self.counters['WIRE']}",
            f"  Edges: {self.counters['EDGE']}",
            f"  Vertices: {self.counters['VERTEX']}",
            f"",
            f"Surface Types:",
        ]
        for st, count in sorted(self.surface_types.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {st}: {count}")

        summary_lines.append("")
        summary_lines.append("Curve Types:")
        for ct, count in sorted(self.curve_types.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {ct}: {count}")

        summary_text = "\n".join(summary_lines)

        logger.info(f"[CADHierarchyTree] Built hierarchy with {self.counters['FACE']} faces, {self.counters['EDGE']} edges")

        return {
            "ui": {
                "hierarchy_file": [json_filename],
                "num_faces": [self.counters["FACE"]],
                "num_edges": [self.counters["EDGE"]],
                "num_solids": [self.counters["SOLID"]],
            },
            "result": (cad_model, json_filename, summary_text)
        }

    def _build_tree_node(self, shape, depth, max_depth, include_properties, collapse_vertices):
        """Recursively build a tree node for a shape."""

        shape_type = shape.ShapeType()
        type_name = SHAPE_TYPE_NAMES.get(shape_type, "UNKNOWN")

        # Update counter
        if type_name in self.counters:
            self.counters[type_name] += 1

        # Create node
        node = {
            "type": type_name,
            "index": self.node_index,
            "children": [],
        }
        self.node_index += 1

        # Add properties for faces and edges
        if include_properties:
            if shape_type == TopAbs_FACE:
                try:
                    adaptor = BRepAdaptor_Surface(topods.Face(shape))
                    surface_type = adaptor.GetType()
                    surface_name = SURFACE_TYPE_NAMES.get(surface_type, "Unknown")
                    node["properties"] = {"surface_type": surface_name}

                    # Update surface type counter
                    self.surface_types[surface_name] = self.surface_types.get(surface_name, 0) + 1
                except Exception:
                    node["properties"] = {"surface_type": "Error"}

            elif shape_type == TopAbs_EDGE:
                try:
                    adaptor = BRepAdaptor_Curve(topods.Edge(shape))
                    curve_type = adaptor.GetType()
                    curve_name = CURVE_TYPE_NAMES.get(curve_type, "Unknown")
                    node["properties"] = {"curve_type": curve_name}

                    # Update curve type counter
                    self.curve_types[curve_name] = self.curve_types.get(curve_name, 0) + 1
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
                    self.counters["VERTEX"] += 1
                else:
                    child_node = self._build_tree_node(
                        child, depth + 1, max_depth, include_properties, collapse_vertices
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
                self.counters["VERTEX"] += 1
                iterator.Next()
                continue

            child_node = self._build_tree_node(
                child, depth + 1, max_depth, include_properties, collapse_vertices
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
