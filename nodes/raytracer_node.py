"""
CADabra Raytracer Node
Uses occt_rt for BVH raytracing on BREP files
Outputs a .npz with named channels (raw/unnormalized)
"""
from __future__ import annotations

import logging
import os
import uuid
import numpy as np
import folder_paths

from comfy_api.latest import io

log = logging.getLogger("cadabra")

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from occt_rt import Raytracer


# Channel names for each output mode
_CHANNELS_FULL = [
    "height", "normal_x", "normal_y", "normal_z",
    "position_x", "position_y", "position_z",
    "face_id", "curv_gauss", "curv_mean", "curv_min", "curv_max",
    "curv_xx", "curv_yy", "curv_xy",
    "hit_mask",
]
_CHANNELS_NORMALS = [
    "height", "normal_x", "normal_y", "normal_z",
    "position_x", "position_y", "position_z",
    "face_id", "hit_mask",
]
_CHANNELS_BASIC = ["height", "face_id", "hit_mask"]


class CADRaytracerBVH(io.ComfyNode):
    """
    Executes BVH raytracing on a CAD model using occt_rt.
    Saves a .npz with named channels (raw/unnormalized) to ComfyUI output folder.
    Use Multiband's LoadMultibandImage to reload if needed.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADRaytracerBVH",
            display_name="CAD Raytracer (BVH)",
            category="CADabra/Analysis",
            is_output_node=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
                io.Int.Input("resolution", default=500, min=32, max=4096),
                io.Float.Input("deflection", default=0.02, min=0.001, max=10.0, step=0.01,
                               tooltip="Tessellation deflection (smaller = finer mesh)"),
                io.Combo.Input("backend", options=["embree", "occt", "embree_simd4", "embree_simd8"],
                               default="embree", optional=True),
                io.Combo.Input("output_mode", options=["full", "normals", "basic"],
                               default="full",
                               tooltip="full=all outputs, normals=no curvatures, basic=depth+faceid only",
                               optional=True),
                io.String.Input("roi", default="",
                                tooltip="ROI bounds (X1,Y1,X2,Y2) - connect from ROI Selector to crop render area",
                                optional=True),
                io.String.Input("npz_name", default="",
                                tooltip="Output .npz base name. Leave BLANK for batches/ROIs: the file is "
                                        "auto-named 'raycast_<hash of inputs>.npz' so identical inputs reuse the "
                                        "same stable file (lets downstream Load Multiband / Z-Height cache instead "
                                        "of re-running). Set a name only for a single render -> '<name>.npz' exactly.",
                                optional=True),
            ],
            outputs=[
                io.String.Output(display_name="npz_path"),
            ],
        )

    @staticmethod
    def _render_key(cad_model, resolution, deflection, backend, output_mode, roi):
        """Stable hash of everything that affects the rendered .npz (NOT the file name)."""
        import hashlib
        h = hashlib.sha256()
        cm = cad_model or {}
        bp = cm.get("brep_path") or cm.get("file_path") or ""
        h.update(str(bp).encode())
        try:
            h.update(str(os.path.getmtime(bp)).encode())
        except Exception:
            pass
        for x in (int(resolution), float(deflection), str(backend), str(output_mode), str(roi or "")):
            h.update(repr(x).encode())
        return h.hexdigest()

    @classmethod
    def fingerprint_inputs(cls, cad_model, resolution, deflection,
                           backend="embree", output_mode="full", roi="", npz_name=""):
        """Content-based cache key so the (expensive) raytrace is skipped when nothing changed,
        even if an upstream node re-executes and hands us a fresh-but-identical cad_model/roi."""
        return cls._render_key(cad_model, resolution, deflection, backend, output_mode, roi) + "|" + str(npz_name or "")

    @classmethod
    def execute(cls, cad_model, resolution, deflection,
                backend="embree", output_mode="full", roi="", npz_name=""):
        # Deterministic, idempotent output: the .npz name encodes every render input, so the
        # SAME inputs always map to the SAME file. If it already exists we skip the (expensive)
        # raytrace entirely and return it untouched -> its bytes never change -> Load Multiband's
        # content-hash stays stable -> the whole downstream chain caches instead of re-running.
        out_dir = folder_paths.get_output_directory()
        _name = (npz_name or "").strip()
        prefix = (_name + "_") if _name else "raycast_"
        key = cls._render_key(cad_model, resolution, deflection, backend, output_mode, roi)
        npz_path = os.path.join(out_dir, f"{prefix}{key[:12]}.npz")
        if os.path.exists(npz_path):
            log.info(f" Reusing cached raytrace (inputs unchanged): {npz_path}")
            return io.NodeOutput(npz_path)

        from .utils.brep_cache import get_occ_shape

        shape = get_occ_shape(cad_model)

        file_path = cad_model.get("file_path", "unknown")
        log.info(f" Raytracing CAD model: {os.path.basename(file_path)}")

        # Get bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        log.info(f" Bounding box: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}] Z[{zmin:.2f}, {zmax:.2f}]")

        # Override with ROI if provided
        if roi and roi.strip():
            try:
                parts = [float(x) for x in roi.split(",")]
                if len(parts) == 4:
                    xmin, ymin, xmax, ymax = parts
                    log.info(f" Using ROI bounds: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}]")
            except ValueError:
                log.warning(f" Invalid ROI format '{roi}', using full bounds")

        # Create raytracer
        log.info(f" Creating raytracer (backend={backend}, deflection={deflection})")
        rt = Raytracer(shape, deflection=deflection, backend=backend)
        log.info(f" Raytracer loaded: {rt.num_faces} faces")

        # Render orthographic (top-down Z view)
        offset = zmax + 10.0
        log.info(f" Rendering {resolution}x{resolution} orthographic view (output_mode={output_mode})")

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

        # Height (raw Z values)
        height = depth.copy().astype(np.float32)
        height[~hits] = 0.0

        # Face IDs (set to -1 where no hit)
        face_id = face_ids.astype(np.float32)
        face_id[~hits] = -1

        # Build channel list based on output_mode
        channels = []

        if output_mode == "basic":
            channels = [height, face_id, hit_mask]
            channel_names = list(_CHANNELS_BASIC)
        else:
            # Position (world coordinates)
            xs = np.linspace(xmin, xmax, W)
            ys = np.linspace(ymax, ymin, H)  # Flipped Y for image coords
            xx, yy = np.meshgrid(xs, ys)
            position_x = xx.astype(np.float32)
            position_y = yy.astype(np.float32)
            position_z = depth.copy().astype(np.float32)
            position_x[~hits] = 0.0
            position_y[~hits] = 0.0
            position_z[~hits] = 0.0

            # Normals
            if "normals" in results:
                normals = results["normals"].astype(np.float32)
                normal_x = normals[:, :, 0]
                normal_y = normals[:, :, 1]
                normal_z = normals[:, :, 2]
                normal_x[~hits] = 0.0
                normal_y[~hits] = 0.0
                normal_z[~hits] = 0.0
            else:
                normal_x = np.zeros((H, W), dtype=np.float32)
                normal_y = np.zeros((H, W), dtype=np.float32)
                normal_z = np.ones((H, W), dtype=np.float32)
                normal_z[~hits] = 0.0

            if output_mode == "normals":
                channels = [height, normal_x, normal_y, normal_z,
                            position_x, position_y, position_z,
                            face_id, hit_mask]
                channel_names = list(_CHANNELS_NORMALS)
            else:
                # Full mode - include curvatures
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

                # Analytic height-field Hessian d2Z/dX2, d2Z/dY2, d2Z/dXdY (occt_rt >= 1.3.1).
                # These feed the decode-ready full-Hessian curvature image
                # (R=curv_xy, G=curv_xx, B=curv_yy).
                if "curv_xx" in results:
                    curv_xx = results["curv_xx"].astype(np.float32)
                    curv_yy = results["curv_yy"].astype(np.float32)
                    curv_xy = results["curv_xy"].astype(np.float32)
                    curv_xx[~hits] = 0.0
                    curv_yy[~hits] = 0.0
                    curv_xy[~hits] = 0.0
                else:
                    log.warning(" occt_rt build lacks curv_xx/yy/xy (need >= 1.3.1); writing zeros")
                    curv_xx = np.zeros((H, W), dtype=np.float32)
                    curv_yy = np.zeros((H, W), dtype=np.float32)
                    curv_xy = np.zeros((H, W), dtype=np.float32)

                channels = [height, normal_x, normal_y, normal_z,
                            position_x, position_y, position_z,
                            face_id, curv_gauss, curv_mean, curv_min, curv_max,
                            curv_xx, curv_yy, curv_xy,
                            hit_mask]
                channel_names = list(_CHANNELS_FULL)

        # Stack into (1, C, H, W) and save as .npz
        stacked = np.stack(channels, axis=0)[np.newaxis, ...].astype(np.float32)

        np.savez_compressed(npz_path, samples=stacked,                # npz_path computed at top
                            channel_names=np.array(channel_names, dtype=object))

        log.info(f" Raytracing complete. {len(channels)} channels, shape: ({H}, {W})")
        log.info(f" Saved to {npz_path}")

        return io.NodeOutput(npz_path)


NODE_CLASS_MAPPINGS = {
    "CADRaytracerBVH": CADRaytracerBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADRaytracerBVH": "CAD Raytracer (BVH)",
}
