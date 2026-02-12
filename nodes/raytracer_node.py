"""
CADabra Raytracer Node
Uses occt_rt for BVH raytracing on BREP files
Outputs individual MASK channels (raw/unnormalized)
"""

import os
import numpy as np
import torch

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from occt_rt import Raytracer


class CADRaytracerBVH:
    """
    Executes BVH raytracing on a CAD model using occt_rt.
    Outputs all channels as separate MASK outputs (raw/unnormalized).
    Use MaskNormalizer to normalize values to 0-1 range.
    Use MasksToRGB to combine 3 masks into an RGB image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cad_model": ("CAD_MODEL",),
                "resolution": ("INT", {"default": 500, "min": 32, "max": 4096}),
                "deflection": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.01,
                                         "tooltip": "Tessellation deflection (smaller = finer mesh)"}),
            },
            "optional": {
                "backend": (["embree", "occt", "embree_simd4", "embree_simd8"], {"default": "embree"}),
                "output_mode": (["full", "normals", "basic"], {"default": "full",
                               "tooltip": "full=all outputs, normals=no curvatures, basic=depth+faceid only"}),
                "roi": ("STRING", {
                    "default": "",
                    "tooltip": "ROI bounds (X1,Y1,X2,Y2) - connect from ROI Selector to crop render area"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("height", "normal_x", "normal_y", "normal_z", "position_x", "position_y", "position_z",
                    "face_id", "curv_gauss", "curv_mean", "curv_min", "curv_max", "hit_mask")
    FUNCTION = "run_raytracer"
    CATEGORY = "CADabra/Analysis"

    def run_raytracer(self, cad_model, resolution, deflection,
                      backend="embree", output_mode="full", roi=""):
        # Get shape from CAD_MODEL
        shape = cad_model.get("occ_shape")
        if shape is None:
            raise ValueError("CAD model does not contain occ_shape")

        file_path = cad_model.get("file_path", "unknown")
        print(f"[CADabra] Raytracing CAD model: {os.path.basename(file_path)}")

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        print(f"[CADabra] Bounding box: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}] Z[{zmin:.2f}, {zmax:.2f}]")

        # Override with ROI if provided
        if roi and roi.strip():
            try:
                parts = [float(x) for x in roi.split(",")]
                if len(parts) == 4:
                    xmin, ymin, xmax, ymax = parts
                    print(f"[CADabra] Using ROI bounds: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}]")
            except ValueError:
                print(f"[CADabra] Warning: Invalid ROI format '{roi}', using full bounds")

        # Create raytracer
        print(f"[CADabra] Creating raytracer (backend={backend}, deflection={deflection})")
        rt = Raytracer(shape, deflection=deflection, backend=backend)
        print(f"[CADabra] Raytracer loaded: {rt.num_faces} faces")

        # Render orthographic (top-down Z view)
        offset = zmax + 10.0  # Start rays above the shape
        print(f"[CADabra] Rendering {resolution}x{resolution} orthographic view (output_mode={output_mode})")

        results = rt.render_orthographic(
            resolution=(resolution, resolution),
            bounds=(xmin, ymin, xmax, ymax),
            axis='z',
            offset=offset,
            output_mode=output_mode,
        )

        # Extract depth (Z heights)
        depth = results["depth"]
        face_ids = results["face_ids"]

        # Create hit mask for valid pixels
        hits = ~np.isnan(depth)
        hit_mask = hits.astype(np.float32)

        H, W = depth.shape

        # Height (raw Z values, NaN where no hit)
        height = depth.copy().astype(np.float32)

        # Position (world coordinates)
        xs = np.linspace(xmin, xmax, W)
        ys = np.linspace(ymax, ymin, H)  # Flipped Y for image coords
        xx, yy = np.meshgrid(xs, ys)
        position_x = xx.astype(np.float32)
        position_y = yy.astype(np.float32)
        position_z = depth.copy().astype(np.float32)

        # Set position to 0 where no hit (optional, could keep as-is)
        position_x[~hits] = 0.0
        position_y[~hits] = 0.0
        position_z[~hits] = 0.0

        # Face IDs (set to -1 where no hit)
        face_id = face_ids.astype(np.float32)
        face_id[~hits] = -1

        # Get normals (if available)
        if "normals" in results:
            normals = results["normals"].astype(np.float32)
            normal_x = normals[:, :, 0]
            normal_y = normals[:, :, 1]
            normal_z = normals[:, :, 2]
            normal_x[~hits] = 0.0
            normal_y[~hits] = 0.0
            normal_z[~hits] = 0.0
        else:
            # Default to up-facing normals
            normal_x = np.zeros((H, W), dtype=np.float32)
            normal_y = np.zeros((H, W), dtype=np.float32)
            normal_z = np.ones((H, W), dtype=np.float32)
            normal_z[~hits] = 0.0

        # Get curvatures (if available)
        if "gauss_curvatures" in results:
            curv_gauss = results["gauss_curvatures"].astype(np.float32)
            curv_mean = results["mean_curvatures"].astype(np.float32)
            curv_min = results["min_curvatures"].astype(np.float32)
            curv_max = results["max_curvatures"].astype(np.float32)
            curv_gauss[~hits] = 0.0
            curv_mean[~hits] = 0.0
            curv_min[~hits] = 0.0
            curv_max[~hits] = 0.0
        else:
            curv_gauss = np.zeros((H, W), dtype=np.float32)
            curv_mean = np.zeros((H, W), dtype=np.float32)
            curv_min = np.zeros((H, W), dtype=np.float32)
            curv_max = np.zeros((H, W), dtype=np.float32)

        # Convert to tensors - MASK format: (B, H, W)
        def to_mask_tensor(arr):
            return torch.from_numpy(arr).unsqueeze(0).float()

        print(f"[CADabra] Raytracing complete. Output shape: ({H}, {W})")

        return (
            to_mask_tensor(height),
            to_mask_tensor(normal_x),
            to_mask_tensor(normal_y),
            to_mask_tensor(normal_z),
            to_mask_tensor(position_x),
            to_mask_tensor(position_y),
            to_mask_tensor(position_z),
            to_mask_tensor(face_id),
            to_mask_tensor(curv_gauss),
            to_mask_tensor(curv_mean),
            to_mask_tensor(curv_min),
            to_mask_tensor(curv_max),
            to_mask_tensor(hit_mask),
        )


NODE_CLASS_MAPPINGS = {
    "CADRaytracerBVH": CADRaytracerBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADRaytracerBVH": "CAD Raytracer (BVH)",
}
