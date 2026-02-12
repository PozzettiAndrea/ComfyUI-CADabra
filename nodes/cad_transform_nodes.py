"""CADabra Transform Nodes - Rotate/Translate/Scale CAD shapes"""
import math
from OCC.Core.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


class CADTransform:
    """Apply rotation/translation/scale to CAD shapes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "translate_x": ("FLOAT", {"default": 0.0, "step": 1.0}),
                "translate_y": ("FLOAT", {"default": 0.0, "step": 1.0}),
                "translate_z": ("FLOAT", {"default": 0.0, "step": 1.0}),
                "rotate_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0,
                                       "tooltip": "Rotation around X axis (degrees)"}),
                "rotate_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0,
                                       "tooltip": "Rotation around Y axis (degrees)"}),
                "rotate_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0,
                                       "tooltip": "Rotation around Z axis (degrees)"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("CAD_MODEL",)
    RETURN_NAMES = ("cad_model",)
    FUNCTION = "transform"
    CATEGORY = "CADabra/Transform"

    def transform(self, cad_model, translate_x, translate_y, translate_z,
                  rotate_x, rotate_y, rotate_z, scale):
        shape = cad_model.get("occ_shape")
        if shape is None:
            raise RuntimeError("CAD model has no OCC shape")

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

        result = {
            "occ_shape": transformer.Shape(),
            "format": "occ",
        }
        if "file_path" in cad_model:
            result["file_path"] = cad_model["file_path"]

        return (result,)


class CADBoundingBox:
    """Get bounding box of a CAD shape."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "width", "height", "depth", "info")
    FUNCTION = "get_bbox"
    CATEGORY = "CADabra/Transform"

    def get_bbox(self, cad_model):
        shape = cad_model.get("occ_shape")
        if shape is None:
            raise RuntimeError("CAD model has no OCC shape")

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

        return (x_min, x_max, y_min, y_max, z_min, z_max, width, height, depth, info)


class FloatMath:
    """Simple float math operations (negate, add, multiply, etc.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "operation": (["negate", "add", "subtract", "multiply", "divide", "abs"],),
            },
            "optional": {
                "operand": ("FLOAT", {"default": 1.0}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "compute"
    CATEGORY = "CADabra/Utils"

    def compute(self, value, operation, operand=1.0):
        if operation == "negate":
            return (-value,)
        elif operation == "add":
            return (value + operand,)
        elif operation == "subtract":
            return (value - operand,)
        elif operation == "multiply":
            return (value * operand,)
        elif operation == "divide":
            return (value / operand if operand != 0 else 0.0,)
        elif operation == "abs":
            return (abs(value),)
        return (value,)


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
