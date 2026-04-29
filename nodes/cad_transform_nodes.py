"""CADabra Transform Nodes - Rotate/Translate/Scale CAD shapes"""
from __future__ import annotations
import math

from OCC.Core.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from comfy_api.latest import io


class CADTransform(io.ComfyNode):
    """Apply rotation/translation/scale to CAD shapes."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADTransform",
            display_name="Transform CAD",
            category="CADabra/Transform",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.Float.Input("translate_x", default=0.0, step=1.0),
                io.Float.Input("translate_y", default=0.0, step=1.0),
                io.Float.Input("translate_z", default=0.0, step=1.0),
                io.Float.Input("rotate_x", default=0.0, min=-360.0, max=360.0, step=5.0,
                               tooltip="Rotation around X axis (degrees)"),
                io.Float.Input("rotate_y", default=0.0, min=-360.0, max=360.0, step=5.0,
                               tooltip="Rotation around Y axis (degrees)"),
                io.Float.Input("rotate_z", default=0.0, min=-360.0, max=360.0, step=5.0,
                               tooltip="Rotation around Z axis (degrees)"),
                io.Float.Input("scale", default=1.0, min=0.001, max=1000.0, step=0.1),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, translate_x, translate_y, translate_z,
                rotate_x, rotate_y, rotate_z, scale):
        # Get OCC shape from brep_path
        from .utils.brep_cache import get_occ_shape, make_cad_model
        shape = get_occ_shape(cad_model)

        # Build combined transformation
        trsf = gp_Trsf()

        # Translation
        if translate_x != 0 or translate_y != 0 or translate_z != 0:
            trsf.SetTranslation(gp_Vec(translate_x, translate_y, translate_z))

        # Rotations (degrees -> radians)
        origin = gp_Pnt(0, 0, 0)
        if rotate_x != 0:
            rot = gp_Trsf()
            rot.SetRotation(gp_Ax1(origin, gp_Dir(1, 0, 0)), math.radians(rotate_x))
            trsf = trsf * rot
        if rotate_y != 0:
            rot = gp_Trsf()
            rot.SetRotation(gp_Ax1(origin, gp_Dir(0, 1, 0)), math.radians(rotate_y))
            trsf = trsf * rot
        if rotate_z != 0:
            rot = gp_Trsf()
            rot.SetRotation(gp_Ax1(origin, gp_Dir(0, 0, 1)), math.radians(rotate_z))
            trsf = trsf * rot

        # Scale
        if scale != 1.0:
            scale_trsf = gp_Trsf()
            scale_trsf.SetScale(origin, scale)
            trsf = trsf * scale_trsf

        # Apply transform
        transformer = BRepBuilderAPI_Transform(shape, trsf, True)
        if not transformer.IsDone():
            raise RuntimeError("Transform failed")

        result = make_cad_model(transformer.Shape(), cad_model, "transformed")

        return io.NodeOutput(result)


class CADBoundingBox(io.ComfyNode):
    """Get bounding box of a CAD shape."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADBoundingBox",
            display_name="CAD Bounding Box",
            category="CADabra/Transform",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
            ],
            outputs=[
                io.Float.Output(display_name="x_min"),
                io.Float.Output(display_name="x_max"),
                io.Float.Output(display_name="y_min"),
                io.Float.Output(display_name="y_max"),
                io.Float.Output(display_name="z_min"),
                io.Float.Output(display_name="z_max"),
                io.Float.Output(display_name="width"),
                io.Float.Output(display_name="height"),
                io.Float.Output(display_name="depth"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model):
        # Get OCC shape from brep_path
        from .utils.brep_cache import get_occ_shape
        shape = get_occ_shape(cad_model)

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()

        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min

        info = (
            f"X: [{x_min:.3f}, {x_max:.3f}] (width: {width:.3f})\n"
            f"Y: [{y_min:.3f}, {y_max:.3f}] (height: {height:.3f})\n"
            f"Z: [{z_min:.3f}, {z_max:.3f}] (depth: {depth:.3f})"
        )

        return io.NodeOutput(x_min, x_max, y_min, y_max, z_min, z_max, width, height, depth, info)


class FloatMath(io.ComfyNode):
    """Simple float math operations (negate, add, multiply, etc.)"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FloatMath",
            display_name="Float Math",
            category="CADabra/Utils",
            inputs=[
                io.Float.Input("value", default=0.0, force_input=True),
                io.Combo.Input("operation", options=["negate", "add", "subtract", "multiply", "divide", "abs"]),
                io.Float.Input("operand", default=1.0, optional=True),
            ],
            outputs=[
                io.Float.Output(display_name="result"),
            ],
        )

    @classmethod
    def execute(cls, value, operation, operand=1.0):
        if operation == "negate":
            return io.NodeOutput(-value)
        elif operation == "add":
            return io.NodeOutput(value + operand)
        elif operation == "subtract":
            return io.NodeOutput(value - operand)
        elif operation == "multiply":
            return io.NodeOutput(value * operand)
        elif operation == "divide":
            return io.NodeOutput(value / operand if operand != 0 else 0.0)
        elif operation == "abs":
            return io.NodeOutput(abs(value))
        return io.NodeOutput(value)


NODE_CLASS_MAPPINGS = {
    "CADTransform": CADTransform,
    "CADBoundingBox": CADBoundingBox,
    "FloatMath": FloatMath,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CADTransform": "Transform CAD",
    "CADBoundingBox": "CAD Bounding Box",
    "FloatMath": "Float Math",
}
