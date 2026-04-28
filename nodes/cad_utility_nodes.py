# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
CAD Utility Nodes for CADabra

Provides utility operations for CAD models:
- Count faces
- Extract faces only (remove edges/wires)
- Sew disconnected faces
- Count disconnected components

NOTE: CAD_MODEL now stores OCC shapes directly (no GMSH model).
All operations work on OCC shapes, preserving topology.
"""

import logging
import os
import sys

from comfy_api.latest import io
from .utils.occ_logging import log_operation

log = logging.getLogger("cadabra")


def _progress_bar(completed, total, elapsed, width=30, prefix=""):
    """Print a tqdm-style progress bar."""
    if total == 0:
        return
    pct = completed / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    # Use \r to overwrite the line
    sys.stdout.write(f"\r{prefix}|{bar}| {completed}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SHELL, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_FixSmallFace, ShapeFix_Wireframe



def _get_occ_shape(cad_model):
    """Get OCC shape from CAD_MODEL dict (loads from brep_path)."""
    from .utils.brep_cache import get_occ_shape
    return get_occ_shape(cad_model)


def _make_cad_model(occ_shape, original_cad_model=None, name_hint="shape"):
    """Create new CAD_MODEL dict (saves shape to brep_path)."""
    from .utils.brep_cache import make_cad_model
    return make_cad_model(occ_shape, original_cad_model, name_hint)


class CADExtractFaces(io.ComfyNode):
    """Extract only faces from a CAD model, removing edges, wires, and solids."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADExtractFaces",
            display_name="Extract Faces",
            category="CADabra/Utility",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model to extract faces from"),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="faces_only"),
            ],
        )

    @classmethod
    def execute(cls, cad_model):
        """Extract only faces using OCC."""
        shape = _get_occ_shape(cad_model)

        # Create new compound with only faces
        face_compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(face_compound)

        face_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            builder.Add(face_compound, face)
            face_count += 1
            explorer.Next()

        log.info(f"Extracted {face_count} faces from CAD model")

        # Return new CAD_MODEL with OCC shape (no STEP round-trip!)
        result = _make_cad_model(face_compound, cad_model)

        return io.NodeOutput(result)


class CADSewFaces(io.ComfyNode):
    """
    Sew disconnected faces together into a watertight shell.

    Uses OCC's BRepBuilderAPI_Sewing to join faces that share edges
    within the specified tolerance.

    Supports batch processing with optional parallel execution for
    OS-level timeout and crash isolation.

    WARNING: Using too high a tolerance (e.g., 10+) can create degenerate edges
    where both endpoints are merged to the same vertex. This will cause mesh
    connectivity issues. Recommended: use tolerance <= 1.0 for most models.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADSewFaces",
            display_name="CAD Sew Faces",
            category="CADabra/Utility",
            is_input_list=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model(s) with disconnected faces"),
                io.Float.Input("tolerance", default=1.0, min=1e-6, max=1000.0, step=0.1,
                    tooltip="Maximum distance between edges/vertices to consider them coincident (in model units, e.g. mm)"),
                io.DynamicCombo.Input("execution_mode",
                    tooltip="single_core: sequential in main process. parallel: subprocesses with timeout and crash isolation",
                    options=[
                        io.DynamicCombo.Option("single_core", []),
                        io.DynamicCombo.Option("parallel", [
                            io.Int.Input("num_workers", default=4, min=1, max=32,
                                tooltip="Number of parallel subprocesses (parallel mode only)"),
                            io.Int.Input("timeout", default=120, min=10, max=600,
                                tooltip="Timeout per model in seconds (parallel mode only)"),
                        ]),
                    ],
                ),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="sewn_models", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, tolerance, execution_mode="single_core"):
        """Sew faces - dispatches to sequential or parallel based on execution_mode."""
        # Extract scalar values from lists (INPUT_IS_LIST behavior)
        tolerance_val = tolerance[0] if isinstance(tolerance, list) else tolerance

        # Extract execution_mode and optional parallel params from DynamicCombo
        if isinstance(execution_mode, list):
            execution_mode = execution_mode[0]
        if isinstance(execution_mode, dict):
            execution_mode_val = execution_mode.get("execution_mode", "single_core")
            num_workers_val = execution_mode.get("num_workers", 4)
            timeout_val = execution_mode.get("timeout", 120)
        else:
            execution_mode_val = execution_mode if isinstance(execution_mode, str) else "single_core"
            num_workers_val = 4
            timeout_val = 120

        # Ensure cad_model is a list
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        if execution_mode_val == "parallel":
            return cls._sew_parallel(cad_models, num_workers_val, timeout_val, tolerance_val)
        else:
            return cls._sew_sequential(cad_models, tolerance_val)

    @classmethod
    def _sew_sequential(cls, cad_models, tolerance):
        """Process models sequentially in main process."""
        import time
        from datetime import datetime
        import folder_paths

        results = []
        all_report_lines = []

        for model_idx, cad_model in enumerate(cad_models):
            file_path = cad_model.get("file_path", f"model_{model_idx}")
            filename = os.path.basename(file_path)
            all_report_lines.append(f"\n{'='*60}\nModel {model_idx + 1}/{len(cad_models)}: {filename}\n{'='*60}")

            result, report = cls._sew_single(cad_model, tolerance)
            results.append(result)
            all_report_lines.append(report)

        combined_report = "\n".join(all_report_lines)
        return io.NodeOutput(results, combined_report)

    @classmethod
    def _sew_single(cls, cad_model, tolerance=1.0):
        """Sew faces using OCC BRepBuilderAPI_Sewing."""
        import time
        from datetime import datetime
        import folder_paths

        # Create persistent log file
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"sew_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # Collect log messages for report output
        report_lines = []

        def _log_msg(msg):
            """Log, write to log file, and collect for report."""
            log.info(msg)
            report_lines.append(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
                f.flush()

        _log_msg(f"[CADabra] Log file: {log_file}")

        shape = _get_occ_shape(cad_model)

        # Count faces before sewing
        _log_msg(f"[CADabra] Counting faces before sewing...")
        t0 = time.time()
        face_count_before = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_count_before += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_faces_before: {time.time() - t0:.3f}s ({face_count_before} faces)")

        # Count connected components before sewing
        _log_msg(f"[CADabra] Counting connected components before sewing...")
        t0 = time.time()
        components_before = cls._count_connected_components(shape)
        _log_msg(f"[CADabra] [TIMING] count_components_before: {time.time() - t0:.3f}s ({components_before} components)")

        # Count free edges before sewing
        _log_msg(f"[CADabra] Counting free edges before sewing...")
        t0 = time.time()
        free_edges_before = cls._count_free_edges(shape)
        _log_msg(f"[CADabra] [TIMING] count_free_edges_before: {time.time() - t0:.3f}s ({free_edges_before} free edges)")

        # Count edges before sewing
        _log_msg(f"[CADabra] Counting edges before sewing...")
        t0 = time.time()
        edge_count_before = 0
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge_count_before += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_edges_before: {time.time() - t0:.3f}s ({edge_count_before} edges)")

        # Count vertices before sewing
        _log_msg(f"[CADabra] Counting vertices before sewing...")
        t0 = time.time()
        vertex_count_before = 0
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex_count_before += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_vertices_before: {time.time() - t0:.3f}s ({vertex_count_before} vertices)")

        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] INPUT SUMMARY:")
        _log_msg(f"[CADabra]   Faces: {face_count_before}")
        _log_msg(f"[CADabra]   Edges: {edge_count_before}")
        _log_msg(f"[CADabra]   Vertices: {vertex_count_before}")
        _log_msg(f"[CADabra]   Connected components: {components_before}")
        _log_msg(f"[CADabra]   Free edges (open): {free_edges_before}")
        _log_msg(f"[CADabra]   Tolerance: {tolerance}")
        _log_msg(f"[CADabra] ========================================")

        # Complexity analysis
        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] COMPLEXITY ANALYSIS:")

        # Check if all faces are disconnected (typical IGES import)
        if components_before == face_count_before and face_count_before > 1:
            _log_msg(f"[CADabra]   WARNING: Every face is disconnected (typical IGES import)")
            _log_msg(f"[CADabra]   No topology - sewing must find matching edges by geometry")

        # Calculate sewing complexity - O(n²) edge comparisons
        if free_edges_before > 0:
            edge_pairs = free_edges_before * (free_edges_before - 1) // 2
            _log_msg(f"[CADabra]   Edge pairs to compare: {edge_pairs:,}")

            if free_edges_before > 50000:
                _log_msg(f"[CADabra]   DANGER: >50k free edges - sewing will likely hang!")
                _log_msg(f"[CADabra]   SUGGESTION: Split model into parts or use CADSewFacesParallel")
            elif free_edges_before > 10000:
                estimated_minutes = edge_pairs / 50_000_000  # rough: 50M comparisons/minute
                _log_msg(f"[CADabra]   WARNING: Very slow sewing expected (~{estimated_minutes:.0f}+ minutes)")
                _log_msg(f"[CADabra]   SUGGESTION: Consider using CADSewFacesParallel with timeout")
            elif free_edges_before > 1000:
                _log_msg(f"[CADabra]   NOTE: Moderate complexity - may take a few minutes")

        # Analyze face areas to detect tiny faces that can hang sewing
        _log_msg(f"[CADabra] Analyzing face areas...")
        t0 = time.time()
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop

            face_areas = []
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                face_areas.append(props.Mass())
                explorer.Next()

            if face_areas:
                min_area = min(face_areas)
                max_area = max(face_areas)
                avg_area = sum(face_areas) / len(face_areas)
                _log_msg(f"[CADabra]   Face area range: {min_area:.6g} to {max_area:.6g} (avg: {avg_area:.6g})")

                # Check for tiny faces relative to tolerance
                tol_squared = tolerance * tolerance
                tiny_faces = sum(1 for a in face_areas if a < tol_squared)
                if tiny_faces > 0:
                    _log_msg(f"[CADabra]   WARNING: {tiny_faces} faces have area < tolerance² ({tol_squared:.6g})")
                    _log_msg(f"[CADabra]   Tiny faces can cause sewing to hang or fail!")

                # Check for huge area variation (sign of mixed-scale geometry)
                if max_area > 0 and min_area > 0:
                    scale_ratio = max_area / min_area
                    if scale_ratio > 1e6:
                        _log_msg(f"[CADabra]   WARNING: Extreme scale variation ({scale_ratio:.0e}x)")
                        _log_msg(f"[CADabra]   This can cause numerical issues in sewing!")
        except Exception as e:
            _log_msg(f"[CADabra]   Could not analyze face areas: {e}")
        _log_msg(f"[CADabra] [TIMING] analyze_areas: {time.time() - t0:.3f}s")

        # Per-face edge analysis - understand why there are so many edges
        _log_msg(f"[CADabra] Analyzing edges per face...")
        t0 = time.time()
        try:
            edges_per_face = []
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                edge_count = 0
                edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
                while edge_explorer.More():
                    edge_count += 1
                    edge_explorer.Next()
                edges_per_face.append(edge_count)
                explorer.Next()

            if edges_per_face:
                min_edges = min(edges_per_face)
                max_edges = max(edges_per_face)
                avg_edges = sum(edges_per_face) / len(edges_per_face)
                _log_msg(f"[CADabra]   Edges per face: min={min_edges}, max={max_edges}, avg={avg_edges:.1f}")

                # Identify faces with abnormally high edge counts
                high_edge_faces = [(i, e) for i, e in enumerate(edges_per_face) if e > 100]
                if high_edge_faces:
                    _log_msg(f"[CADabra]   WARNING: {len(high_edge_faces)} faces have >100 edges!")
                    _log_msg(f"[CADabra]   This suggests fragmented trim curves from IGES import")
                    for face_idx, edge_count in sorted(high_edge_faces, key=lambda x: -x[1])[:5]:
                        _log_msg(f"[CADabra]     Face {face_idx}: {edge_count} edges")
                    if len(high_edge_faces) > 5:
                        _log_msg(f"[CADabra]     ... and {len(high_edge_faces) - 5} more")

                # Distribution summary
                edge_ranges = [0, 0, 0, 0, 0]  # 1-10, 11-50, 51-100, 101-500, 500+
                for e in edges_per_face:
                    if e <= 10: edge_ranges[0] += 1
                    elif e <= 50: edge_ranges[1] += 1
                    elif e <= 100: edge_ranges[2] += 1
                    elif e <= 500: edge_ranges[3] += 1
                    else: edge_ranges[4] += 1
                _log_msg(f"[CADabra]   Edge distribution: 1-10: {edge_ranges[0]}, 11-50: {edge_ranges[1]}, 51-100: {edge_ranges[2]}, 101-500: {edge_ranges[3]}, 500+: {edge_ranges[4]}")
        except Exception as e:
            _log_msg(f"[CADabra]   Could not analyze edges per face: {e}")
        _log_msg(f"[CADabra] [TIMING] analyze_edges_per_face: {time.time() - t0:.3f}s")

        _log_msg(f"[CADabra] ========================================")

        # Initialize sewing tool and add faces
        _log_msg(f"[CADabra] Creating BRepBuilderAPI_Sewing with tolerance={tolerance}...")
        t0 = time.time()
        sewer = BRepBuilderAPI_Sewing(tolerance)
        _log_msg(f"[CADabra] [TIMING] create_sewer: {time.time() - t0:.3f}s")

        _log_msg(f"[CADabra] Adding {face_count_before} faces to sewer...")
        t0 = time.time()
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        while explorer.More():
            face = topods.Face(explorer.Current())
            sewer.Add(face)
            face_idx += 1
            if face_idx % 100 == 0:
                _log_msg(f"[CADabra]   Added {face_idx}/{face_count_before} faces...")
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] add_faces: {time.time() - t0:.3f}s (added {face_idx} faces)")

        # Perform sewing with progress indicator if available
        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] STARTING sewer.Perform() - this may take a while or hang...")
        _log_msg(f"[CADabra] If this is the last message, sewing is stuck!")
        _log_msg(f"[CADabra] ========================================")

        # Try to use OCC progress indicator for real-time progress
        progress_available = False
        try:
            from OCC.Core.Message import Message_ProgressIndicator

            class SewingProgress(Message_ProgressIndicator):
                """Progress indicator that logs to file."""
                def __init__(self, log_func):
                    super().__init__()
                    self._log = log_func
                    self._last_pct = -1
                    self._start_time = time.time()

                def Show(self, theScope, isForce):
                    pct = int(self.GetPosition() * 100)
                    if pct != self._last_pct and pct % 5 == 0:  # Log every 5%
                        elapsed = time.time() - self._start_time
                        self._log(f"[CADabra] Sewing progress: {pct}% (elapsed: {elapsed:.1f}s)")
                        self._last_pct = pct

                def UserBreak(self):
                    return False

            progress = SewingProgress(log)
            progress_range = progress.Start()
            _log_msg(f"[CADabra] Progress indicator active - will report every 5%")
            progress_available = True
        except Exception as e:
            _log_msg(f"[CADabra] Progress indicator not available: {e}")
            _log_msg(f"[CADabra] Sewing will run without progress updates")

        t0 = time.time()
        with log_operation("Sewing", faces=face_count_before, tolerance=tolerance):
            if progress_available:
                try:
                    sewer.Perform(progress_range)
                except TypeError:
                    # Fallback if Perform doesn't accept progress
                    _log_msg(f"[CADabra] Perform(progress) not supported, using Perform()")
                    sewer.Perform()
            else:
                sewer.Perform()
        sew_time = time.time() - t0
        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] sewer.Perform() COMPLETED!")
        _log_msg(f"[CADabra] [TIMING] perform_sew: {sew_time:.3f}s")
        _log_msg(f"[CADabra] ========================================")

        # Get sewing diagnostics
        _log_msg(f"[CADabra] Getting sewing diagnostics...")
        n_free_edges = sewer.NbFreeEdges()
        n_multiple_edges = sewer.NbMultipleEdges()
        n_degenerated = sewer.NbDegeneratedShapes()
        n_deleted = sewer.NbDeletedFaces()
        n_contig_edges = sewer.NbContigousEdges()

        _log_msg(f"[CADabra] SEWING DIAGNOSTICS:")
        _log_msg(f"[CADabra]   Free edges (not sewn): {n_free_edges}")
        _log_msg(f"[CADabra]   Contiguous edges (sewn): {n_contig_edges}")
        _log_msg(f"[CADabra]   Multiple edges (>2 faces share edge): {n_multiple_edges}")
        _log_msg(f"[CADabra]   Degenerated shapes: {n_degenerated}")
        _log_msg(f"[CADabra]   Deleted faces: {n_deleted}")

        _log_msg(f"[CADabra] Getting sewn shape...")
        t0 = time.time()
        sewn_shape = sewer.SewedShape()
        _log_msg(f"[CADabra] [TIMING] get_sewn_shape: {time.time() - t0:.3f}s")

        # Check if shape is null
        if sewn_shape.IsNull():
            _log_msg(f"[CADabra] ERROR: SewedShape() returned a NULL shape!")

        # Get shape type
        shape_type = sewn_shape.ShapeType()
        shape_type_names = {0: "COMPOUND", 1: "COMPSOLID", 2: "SOLID", 3: "SHELL", 4: "FACE", 5: "WIRE", 6: "EDGE", 7: "VERTEX", 8: "SHAPE"}
        _log_msg(f"[CADabra] Sewn shape type: {shape_type_names.get(shape_type, f'UNKNOWN({shape_type})')}")

        # Count faces after sewing
        _log_msg(f"[CADabra] Counting faces after sewing...")
        t0 = time.time()
        face_count_after = 0
        explorer = TopExp_Explorer(sewn_shape, TopAbs_FACE)
        while explorer.More():
            face_count_after += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_faces_after: {time.time() - t0:.3f}s ({face_count_after} faces)")

        # Count connected components after sewing
        _log_msg(f"[CADabra] Counting connected components after sewing...")
        t0 = time.time()
        components_after = cls._count_connected_components(sewn_shape)
        _log_msg(f"[CADabra] [TIMING] count_components_after: {time.time() - t0:.3f}s ({components_after} components)")

        # Count edges and shells for more insight
        _log_msg(f"[CADabra] Counting edges after sewing...")
        t0 = time.time()
        edge_count = 0
        explorer = TopExp_Explorer(sewn_shape, TopAbs_EDGE)
        while explorer.More():
            edge_count += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_edges_after: {time.time() - t0:.3f}s ({edge_count} edges)")

        _log_msg(f"[CADabra] Counting shells after sewing...")
        t0 = time.time()
        shell_count = 0
        explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
        while explorer.More():
            shell_count += 1
            explorer.Next()
        _log_msg(f"[CADabra] [TIMING] count_shells_after: {time.time() - t0:.3f}s ({shell_count} shells)")

        # Count free edges after sewing
        _log_msg(f"[CADabra] Counting free edges after sewing...")
        t0 = time.time()
        free_edges_after = cls._count_free_edges(sewn_shape)
        _log_msg(f"[CADabra] [TIMING] count_free_edges_after: {time.time() - t0:.3f}s ({free_edges_after} free edges)")

        # Print summary
        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] SEWING COMPLETE - SUMMARY:")
        _log_msg(f"[CADabra]   Faces: {face_count_before} -> {face_count_after}")
        _log_msg(f"[CADabra]   Edges: {edge_count_before} -> {edge_count}")
        _log_msg(f"[CADabra]   Components: {components_before} -> {components_after}")
        _log_msg(f"[CADabra]   Free edges: {free_edges_before} -> {free_edges_after}")
        _log_msg(f"[CADabra]   Shells created: {shell_count}")
        _log_msg(f"[CADabra]   Sewing time: {sew_time:.3f}s")
        _log_msg(f"[CADabra]   Tolerance used: {tolerance}")
        if components_after == 1 and free_edges_after == 0:
            _log_msg(f"[CADabra]   STATUS: SUCCESS - watertight shell created!")
        elif components_after == 1:
            _log_msg(f"[CADabra]   STATUS: PARTIAL - single component but {free_edges_after} open edges")
        else:
            _log_msg(f"[CADabra]   STATUS: INCOMPLETE - {components_after} disconnected components")
        _log_msg(f"[CADabra] ========================================")
        _log_msg(f"[CADabra] Log saved to: {log_file}")

        # Return new CAD_MODEL with OCC shape (no STEP round-trip!)
        result = _make_cad_model(sewn_shape, cad_model)

        # Build report string
        report = "\n".join(report_lines)

        return (result, report)

    @staticmethod
    def _count_connected_components(shape):
        """Count disconnected components using face adjacency."""
        all_faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            all_faces.append(topods.Face(explorer.Current()))
            explorer.Next()

        if len(all_faces) == 0:
            return 0

        # Build edge map and adjacency
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        edge_to_faces = {}
        for face_idx, face in enumerate(all_faces):
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_index = edge_map.FindIndex(edge)
                if edge_index > 0:
                    if edge_index not in edge_to_faces:
                        edge_to_faces[edge_index] = set()
                    edge_to_faces[edge_index].add(face_idx)
                edge_explorer.Next()

        adjacency = {i: set() for i in range(len(all_faces))}
        for edge_index, face_indices in edge_to_faces.items():
            face_list = list(face_indices)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adjacency[face_list[i]].add(face_list[j])
                    adjacency[face_list[j]].add(face_list[i])

        # BFS to count components
        visited = set()
        num_components = 0
        for start_face in range(len(all_faces)):
            if start_face in visited:
                continue
            queue = [start_face]
            visited.add(start_face)
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            num_components += 1

        return num_components

    @staticmethod
    def _count_free_edges(shape):
        """Count free (open) edges - edges with only one adjacent face."""
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        edge_face_count = {}
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_index = edge_map.FindIndex(edge)
                if edge_index > 0:
                    edge_face_count[edge_index] = edge_face_count.get(edge_index, 0) + 1
                edge_explorer.Next()
            explorer.Next()

        return sum(1 for count in edge_face_count.values() if count == 1)

    @classmethod
    def _sew_parallel(cls, cad_models, num_workers, timeout, tolerance):
        """Sew faces using subprocess per model with OS-level timeout."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder
        import subprocess
        import tempfile
        import time
        import sys
        import json

        log.info(f"Sew Parallel: Processing {len(cad_models)} models with {num_workers} workers, "
              f"tolerance={tolerance}, timeout={timeout}s")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cadabra_sew_sub_")

        # Create persistent log directory in ComfyUI output folder
        from datetime import datetime
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs", f"sew_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        log.info(f"Sew Parallel: Logs: {log_dir}")

        # Path to the subprocess script
        script_path = os.path.join(os.path.dirname(__file__), "sew_subprocess.py")

        # Prepare work items
        work_items = []
        skipped = 0

        for idx, cm in enumerate(cad_models):
            brep_path = cm.get("brep_path")
            file_path = cm.get("file_path", f"model_{idx}")
            if brep_path is None:
                skipped += 1
                continue

            # Use the existing brep_path as input
            input_brep = brep_path
            output_brep = os.path.join(temp_dir, f"output_{idx}.brep")
            result_file = os.path.join(temp_dir, f"result_{idx}.json")
            # Get filename stem for log file
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            log_file = os.path.join(log_dir, f"{filename_stem}.log")

            work_items.append({
                "idx": idx,
                "input_brep": input_brep,
                "output_brep": output_brep,
                "result_file": result_file,
                "log_file": log_file,
                "file_path": file_path,
            })

        if skipped > 0:
            log.warning(f"Sew Parallel: Skipped {skipped} models without OCC shape")

        if len(work_items) == 0:
            raise ValueError("No valid CAD models to process")

        def run_sew_subprocess(item):
            """Run sewing in subprocess with timeout."""
            cmd = [
                sys.executable,
                script_path,
                item["input_brep"],
                item["output_brep"],
                str(tolerance),
                f"--result-file={item['result_file']}",
                f"--log-file={item['log_file']}"
            ]
            item_start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5  # Small buffer for startup
                )
                elapsed = time.time() - item_start
                # Read result file if it exists
                if os.path.exists(item["result_file"]):
                    with open(item["result_file"], 'r') as f:
                        return {**json.load(f), "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                elif result.returncode == 0:
                    return {"success": True, "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                else:
                    return {"success": False, "error": result.stderr or "Unknown error", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except subprocess.TimeoutExpired:
                elapsed = time.time() - item_start
                return {"success": False, "error": f"Timed out after {timeout}s", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - item_start
                return {"success": False, "error": str(e), "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}

        # Process in parallel using ThreadPoolExecutor (subprocess handles CPU work)
        results = [None] * len(work_items)
        start_time = time.time()
        report_lines = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(run_sew_subprocess, item): i
                    for i, item in enumerate(work_items)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {"success": False, "error": str(e), "file_path": work_items[idx]["file_path"], "log_file": work_items[idx]["log_file"]}
                    completed += 1
                    elapsed = time.time() - start_time
                    _progress_bar(completed, len(work_items), elapsed, prefix="Sewing: ")

            # Build CAD_MODEL dicts from results and print per-model timing
            sewn_models = []
            failed = 0

            # Threshold for showing log file path (seconds)
            SLOW_THRESHOLD = 10.0

            for i, result in enumerate(results):
                elapsed_str = f"{result.get('elapsed', 0):.1f}s" if result else "?"
                elapsed_val = result.get('elapsed', 0) if result else 0
                filename = os.path.basename(result.get("file_path", work_items[i]["file_path"])) if result else os.path.basename(work_items[i]["file_path"])
                log_path = result.get("log_file", work_items[i].get("log_file", "")) if result else work_items[i].get("log_file", "")

                if result is None or not result.get("success", False):
                    failed += 1
                    err = result.get("error", "Unknown") if result else "No result"
                    msg = f"{filename}: FAILED ({elapsed_str}) - {err}"
                    log.info(f"Sew Parallel: {msg}")
                    report_lines.append(msg)
                    if log_path:
                        log.info(f"Sew Parallel:   Log: {log_path}")
                    continue

                output_brep = result.get("output_brep", work_items[i]["output_brep"])
                if os.path.exists(output_brep):
                    # Use the output BREP file directly as brep_path
                    original_cm = cad_models[work_items[i]["idx"]] if work_items[i]["idx"] < len(cad_models) else {}
                    cad_result = {
                        "brep_path": output_brep,
                        "format": "brep",
                        "file_path": result["file_path"],
                    }
                    if "metadata" in original_cm:
                        cad_result["metadata"] = original_cm["metadata"]

                    sewn_models.append(cad_result)

                    # Log detailed diagnostics with timing
                    diag = result.get("diagnostics", {})
                    faces_before = result.get("faces_before", "?")
                    faces_after = result.get("faces_after", "?")
                    comp_before = result.get("components_before", "?")
                    comp_after = result.get("components_after", "?")
                    free_before = result.get("free_edges_before", "?")
                    free_after = diag.get("free_edges", "?")

                    msg = (f"{filename}: faces={faces_before}->{faces_after}, "
                           f"components={comp_before}->{comp_after}, "
                           f"free_edges={free_before}->{free_after} ({elapsed_str})")
                    log.info(f"Sew Parallel: {msg}")
                    report_lines.append(msg)

                    # Print detailed timings if available
                    timings = result.get("timings", {})
                    if timings:
                        perform_time = timings.get("perform_sew", 0)
                        total_time = timings.get("total", 0)
                        log.info(f"Sew Parallel:   sew={perform_time:.1f}s, total={total_time:.1f}s")

                    # Print log path for slow operations
                    if elapsed_val > SLOW_THRESHOLD and log_path:
                        log.info(f"Sew Parallel:   Log: {log_path}")
                else:
                    failed += 1
                    msg = f"{filename}: FAILED ({elapsed_str}) - output file missing"
                    log.info(f"Sew Parallel: {msg}")
                    report_lines.append(msg)
                    if log_path:
                        log.info(f"Sew Parallel:   Log: {log_path}")

            summary = f"Completed: {len(sewn_models)} sewn, {failed} failed"
            log.info(f"Sew Parallel: {summary}")
            report_lines.append(summary)

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Sew Parallel: Cleaned up temp dir")
            except Exception as e:
                log.warning(f"Sew Parallel: Could not clean temp dir: {e}")

        report = "\n".join(report_lines)
        return (sewn_models, report)


class CADGetFilename(io.ComfyNode):
    """
    Extract filename (without extension) from a CAD model.

    Returns the original filename that was used when loading the CAD file.
    Useful for batch processing to preserve original names in output.

    Supports batch processing: input a list of CAD models, get a list of filenames.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADGetFilename",
            display_name="CAD Get Filename",
            category="CADabra/Utility",
            is_input_list=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model"),
            ],
            outputs=[
                io.String.Output(display_name="filename", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, cad_model):
        import os

        # Handle both single and batch inputs
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        filenames = []
        for cm in cad_models:
            file_path = cm.get("file_path", "") if cm else ""
            if file_path:
                basename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(basename)[0]
                filenames.append(name_without_ext)
            else:
                filenames.append("unknown")

        log.info(f" Extracted {len(filenames)} filename(s)")
        return io.NodeOutput(filenames)


class CADCheckOverlappingFaces(io.ComfyNode):
    """
    Detect overlapping or duplicate faces in CAD models.

    Three detection methods are available:
    - bbox_centroid: Fast comparison of bounding boxes, centroids, and areas
    - self_intersect: Medium speed using OCCT's BOPAlgo_CheckerSI
    - mesh_distance: Slow but thorough using BRepExtrema distance calculation

    Returns the number of overlapping face pairs found and a detailed report.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADCheckOverlappingFaces",
            display_name="Check Overlapping Faces",
            category="CADabra/Utility",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model to check for overlapping faces"),
                io.Combo.Input("method", options=["bbox_centroid", "self_intersect", "mesh_distance"],
                               default="bbox_centroid",
                               tooltip="Detection method: bbox_centroid (fast), self_intersect (medium), mesh_distance (slow)"),
                io.Float.Input("tolerance", default=0.01, min=1e-6, max=10.0, step=0.001,
                               tooltip="Distance tolerance for considering faces as overlapping (in model units)"),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
                io.Int.Output(display_name="overlapping_count"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, method, tolerance):
        """Check for overlapping faces using the selected method."""
        import time
        from math import sqrt

        shape = _get_occ_shape(cad_model)

        log.info(f" Checking for overlapping faces using method: {method}, tolerance: {tolerance}")
        t0 = time.time()

        if method == "bbox_centroid":
            overlaps = cls._check_bbox_centroid(shape, tolerance)
        elif method == "self_intersect":
            overlaps = cls._check_self_intersect(shape, tolerance)
        else:  # mesh_distance
            overlaps = cls._check_mesh_distance(shape, tolerance)

        elapsed = time.time() - t0
        log.info(f" Overlap detection completed in {elapsed:.3f}s, found {len(overlaps)} overlapping pairs")

        # Build detailed report
        report = cls._build_report(method, tolerance, overlaps, elapsed)

        return io.NodeOutput(cad_model, len(overlaps), report)

    @staticmethod
    def _check_bbox_centroid(shape, tolerance):
        """Fast duplicate detection using bounding box, centroid, and area comparison."""
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from math import sqrt

        # Collect all faces with their properties
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        while explorer.More():
            face = topods.Face(explorer.Current())

            # Get bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            # Get centroid and area
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            centroid = props.CentreOfMass()

            faces.append({
                'idx': face_idx,
                'face': face,
                'bbox': (xmin, ymin, zmin, xmax, ymax, zmax),
                'centroid': (centroid.X(), centroid.Y(), centroid.Z()),
                'area': props.Mass()
            })
            face_idx += 1
            explorer.Next()

        log.info(f"   Collected {len(faces)} faces for comparison")

        # Compare all pairs
        overlaps = []
        for i, f1 in enumerate(faces):
            for j, f2 in enumerate(faces[i+1:], i+1):
                # Check bbox similarity (all 6 corners within tolerance)
                bbox_match = all(
                    abs(f1['bbox'][k] - f2['bbox'][k]) < tolerance
                    for k in range(6)
                )

                if not bbox_match:
                    continue

                # Check centroid similarity
                centroid_dist = sqrt(sum(
                    (f1['centroid'][k] - f2['centroid'][k])**2
                    for k in range(3)
                ))
                centroid_match = centroid_dist < tolerance

                if not centroid_match:
                    continue

                # Check area similarity (relative tolerance)
                area_diff = abs(f1['area'] - f2['area'])
                max_area = max(f1['area'], f2['area'])
                area_match = area_diff < tolerance * max_area if max_area > 0 else True

                if area_match:
                    overlaps.append({
                        'face1_idx': f1['idx'],
                        'face2_idx': f2['idx'],
                        'centroid_dist': centroid_dist,
                        'area1': f1['area'],
                        'area2': f2['area'],
                        'method': 'bbox_centroid'
                    })

        return overlaps

    @staticmethod
    def _check_self_intersect(shape, tolerance):
        """Use OCCT's built-in self-intersection checker."""
        from OCC.Core.BOPAlgo import BOPAlgo_CheckerSI
        from OCC.Core.TopTools import TopTools_ListOfShape

        # First collect faces for indexing
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(topods.Face(explorer.Current()))
            explorer.Next()

        log.info(f"   Running BOPAlgo_CheckerSI on {len(faces)} faces...")

        checker = BOPAlgo_CheckerSI()
        args = TopTools_ListOfShape()
        args.Append(shape)
        checker.SetArguments(args)
        checker.SetNonDestructive(True)
        checker.SetFuzzyValue(tolerance)

        with log_operation("BOPAlgo_CheckerSI", faces=len(faces), tolerance=tolerance):
            checker.Perform()

        overlaps = []

        # Try to get interferences from the data structure
        try:
            ds = checker.DS()
            if ds is not None:
                # Get number of interferences
                n_interf = ds.NbInterfs()
                log.info(f"   Found {n_interf} interferences")

                # Iterate through shape indices to find face-face intersections
                for i in range(ds.NbSourceShapes()):
                    s1 = ds.Shape(i)
                    if s1.ShapeType() == TopAbs_FACE:
                        # Check for interferences involving this shape
                        for j in range(i + 1, ds.NbSourceShapes()):
                            s2 = ds.Shape(j)
                            if s2.ShapeType() == TopAbs_FACE:
                                # Check if there's an interference between these shapes
                                if ds.HasInterf(i, j):
                                    overlaps.append({
                                        'face1_idx': i,
                                        'face2_idx': j,
                                        'method': 'self_intersect'
                                    })
        except Exception as e:
            log.warning(f" Could not extract interference details: {e}")
            # Fall back to simpler approach - just report that self-intersection was detected
            if checker.HasErrors():
                log.info(f"   Checker reported errors (self-intersection likely)")
                overlaps.append({
                    'face1_idx': -1,
                    'face2_idx': -1,
                    'method': 'self_intersect',
                    'note': 'Self-intersection detected but specific faces not identified'
                })

        return overlaps

    @staticmethod
    def _check_mesh_distance(shape, tolerance):
        """Check mesh vertex distances between faces using BRepExtrema."""
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box

        # First triangulate the shape
        log.info(f"   Triangulating shape...")
        mesh = BRepMesh_IncrementalMesh(shape, tolerance * 10)  # coarse mesh for speed
        mesh.Perform()

        # Get all faces
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            # Get bounding box for quick filtering
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)
            faces.append({
                'face': face,
                'bbox': bbox
            })
            explorer.Next()

        log.info(f"   Checking distances between {len(faces)} faces...")

        overlaps = []
        comparisons = 0

        for i, f1_data in enumerate(faces):
            f1 = f1_data['face']
            bbox1 = f1_data['bbox']

            for j, f2_data in enumerate(faces[i+1:], i+1):
                f2 = f2_data['face']
                bbox2 = f2_data['bbox']

                # Quick bounding box filter - skip if bboxes are far apart
                if not bbox1.IsVoid() and not bbox2.IsVoid():
                    dist_bbox = bbox1.Distance(bbox2)
                    if dist_bbox > tolerance:
                        continue

                comparisons += 1

                # Use BRepExtrema for precise distance
                try:
                    dist_calc = BRepExtrema_DistShapeShape(f1, f2)
                    if dist_calc.IsDone() and dist_calc.Value() < tolerance:
                        overlaps.append({
                            'face1_idx': i,
                            'face2_idx': j,
                            'distance': dist_calc.Value(),
                            'method': 'mesh_distance'
                        })
                except Exception as e:
                    # Skip problematic face pairs
                    pass

            # Progress indicator for large models
            if (i + 1) % 100 == 0:
                log.info(f"   Processed {i + 1}/{len(faces)} faces ({comparisons} precise comparisons)...")

        log.info(f"   Performed {comparisons} precise distance calculations")
        return overlaps

    @staticmethod
    def _build_report(method, tolerance, overlaps, elapsed):
        """Build a detailed text report of findings."""
        report_lines = [
            "Overlapping Faces Report",
            "=" * 50,
            f"Method: {method}",
            f"Tolerance: {tolerance}",
            f"Analysis time: {elapsed:.3f}s",
            f"",
            f"Found {len(overlaps)} overlapping face pair(s)",
            ""
        ]

        if overlaps:
            report_lines.append("Overlapping pairs:")
            report_lines.append("-" * 30)

            for i, overlap in enumerate(overlaps[:50]):  # Limit to first 50
                face1 = overlap.get('face1_idx', '?')
                face2 = overlap.get('face2_idx', '?')

                detail = f"  {i+1}. Face {face1} <-> Face {face2}"

                if 'centroid_dist' in overlap:
                    detail += f" (centroid dist: {overlap['centroid_dist']:.6f})"
                if 'distance' in overlap:
                    detail += f" (distance: {overlap['distance']:.6f})"
                if 'area1' in overlap:
                    detail += f"\n      Areas: {overlap['area1']:.4f}, {overlap['area2']:.4f}"
                if 'note' in overlap:
                    detail += f"\n      Note: {overlap['note']}"

                report_lines.append(detail)

            if len(overlaps) > 50:
                report_lines.append(f"  ... and {len(overlaps) - 50} more")

            report_lines.append("")
            report_lines.append("Recommendation:")
            report_lines.append("  Remove duplicate/overlapping faces before sewing")
            report_lines.append("  to achieve better connectivity results.")
        else:
            report_lines.append("No overlapping faces detected.")

        return "\n".join(report_lines)


class CADSave(io.ComfyNode):
    """
    Save a CAD model to file in various formats.

    Supports STEP, IGES, BREP, and STL formats. Files are saved to the ComfyUI
    output directory with the specified filename.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADSave",
            display_name="Save CAD",
            category="CADabra/IO",
            is_output_node=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model to save"),
                io.String.Input("filename", default="output", tooltip="Output filename (without extension)"),
                io.Combo.Input("format", options=["step", "iges", "brep", "stl"], default="step",
                               tooltip="Output format: STEP (.stp), IGES (.igs), BREP (.brep), or STL (.stl)"),
                io.Float.Input("stl_linear_deflection", default=0.1, min=0.001, max=1.0, step=0.001, optional=True,
                               tooltip="Linear deflection for STL tessellation (lower = finer mesh)"),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, filename, format, stl_linear_deflection=0.1):
        """Save CAD model to file."""
        from OCC.Core.BRepTools import breptools
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IGESControl import IGESControl_Writer
        from OCC.Core.Interface import Interface_Static
        import folder_paths

        shape = _get_occ_shape(cad_model)

        # Get output directory and create cad subfolder
        output_dir = folder_paths.get_output_directory()
        cad_dir = os.path.join(output_dir, "cad")
        os.makedirs(cad_dir, exist_ok=True)

        # Determine extension and write
        if format == "step":
            ext = ".stp"
            output_path = os.path.join(cad_dir, f"{filename}{ext}")

            writer = STEPControl_Writer()
            Interface_Static.SetCVal("write.step.schema", "AP214")
            writer.Transfer(shape, STEPControl_AsIs)
            status = writer.Write(output_path)

            if status != 1:  # IFSelect_RetDone = 1
                raise RuntimeError(f"Failed to write STEP file: {output_path}")

        elif format == "iges":
            ext = ".igs"
            output_path = os.path.join(cad_dir, f"{filename}{ext}")

            writer = IGESControl_Writer()
            writer.AddShape(shape)
            writer.ComputeModel()
            success = writer.Write(output_path)

            if not success:
                raise RuntimeError(f"Failed to write IGES file: {output_path}")

        elif format == "brep":
            ext = ".brep"
            output_path = os.path.join(cad_dir, f"{filename}{ext}")

            success = breptools.Write(shape, output_path)

            if not success:
                raise RuntimeError(f"Failed to write BREP file: {output_path}")

        else:  # stl
            ext = ".stl"
            output_path = os.path.join(cad_dir, f"{filename}{ext}")

            # STL requires tessellation first
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.StlAPI import StlAPI_Writer

            mesh = BRepMesh_IncrementalMesh(shape, stl_linear_deflection, False, 0.5)
            mesh.Perform()

            writer = StlAPI_Writer()
            writer.SetASCIIMode(False)  # Binary STL
            if not writer.Write(shape, output_path):
                raise RuntimeError(f"Failed to write STL file: {output_path}")

        log.info(f" Saved CAD model to: {output_path}")

        return io.NodeOutput(output_path)


class CADSplitComponents(io.ComfyNode):
    """Split a CAD model into separate models based on connected components."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADSplitComponents",
            display_name="Split Components",
            category="CADabra/Utility",
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model to split into components"),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="components", is_output_list=True),
                io.Int.Output(display_name="num_components"),
                io.String.Output(display_name="face_counts"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model):
        """Split CAD model into separate models per connected component."""
        shape = _get_occ_shape(cad_model)

        # Get all faces
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(topods.Face(explorer.Current()))
            explorer.Next()

        if len(faces) == 0:
            log.info(" No faces found in CAD model")
            return ([], 0, "")

        # Build adjacency using shared edges
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        edge_to_faces = {}
        for face_idx, face in enumerate(faces):
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_index = edge_map.FindIndex(edge)
                if edge_index > 0:
                    if edge_index not in edge_to_faces:
                        edge_to_faces[edge_index] = set()
                    edge_to_faces[edge_index].add(face_idx)
                edge_explorer.Next()

        # Build adjacency graph
        adjacency = {i: set() for i in range(len(faces))}
        for edge_index, face_indices in edge_to_faces.items():
            face_list = list(face_indices)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adjacency[face_list[i]].add(face_list[j])
                    adjacency[face_list[j]].add(face_list[i])

        # Find connected components via BFS
        visited = set()
        components_face_indices = []

        for start_face in range(len(faces)):
            if start_face in visited:
                continue
            component = []
            queue = [start_face]
            visited.add(start_face)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components_face_indices.append(component)

        # Build separate CAD_MODEL for each component
        component_models = []
        face_counts = []

        for comp_idx, face_indices in enumerate(components_face_indices):
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)

            for face_idx in face_indices:
                builder.Add(compound, faces[face_idx])

            comp_model = _make_cad_model(compound, cad_model)
            # Add component index to file_path for identification
            orig_path = cad_model.get("file_path", "model")
            comp_model["file_path"] = f"{orig_path}_component_{comp_idx}"

            component_models.append(comp_model)
            face_counts.append(len(face_indices))

        face_counts_str = ", ".join(str(c) for c in face_counts)
        log.info(f" Split into {len(components_face_indices)} components: [{face_counts_str}] faces")

        # Build detailed report
        report_lines = [
            "Connected Components Report",
            "=" * 50,
            f"Total faces: {len(faces)}",
            f"Components found: {len(components_face_indices)}",
            ""
        ]

        for comp_idx, face_indices in enumerate(components_face_indices):
            report_lines.append(f"Component {comp_idx}: {len(face_indices)} faces")
            # For small components, list all face indices
            if len(face_indices) <= 20:
                report_lines.append(f"  Face indices: {face_indices}")
            else:
                # For large components, show first/last few
                first_five = face_indices[:5]
                last_five = face_indices[-5:]
                report_lines.append(f"  Face indices: {first_five} ... {last_five}")
            report_lines.append("")

        report = "\n".join(report_lines)

        return io.NodeOutput(component_models, len(components_face_indices), face_counts_str, report)


class CADHealShape(io.ComfyNode):
    """General-purpose CAD healing using OCCT ShapeFix."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADHealShape",
            display_name="Heal CAD Shape",
            category="CADabra/Utility",
            is_input_list=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model(s) to heal"),
                io.Float.Input("precision", default=0.01, min=0.0001, max=10.0, step=0.001),
                io.Float.Input("max_tolerance", default=1.0, min=0.01, max=100.0, step=0.1),
                io.DynamicCombo.Input("execution_mode",
                    tooltip="single_core: sequential in main process. parallel: subprocesses with timeout and crash isolation",
                    options=[
                        io.DynamicCombo.Option("single_core", []),
                        io.DynamicCombo.Option("parallel", [
                            io.Int.Input("num_workers", default=4, min=1, max=32,
                                tooltip="Number of parallel subprocesses (parallel mode only)"),
                            io.Int.Input("timeout", default=120, min=10, max=600,
                                tooltip="Timeout per model in seconds (parallel mode only)"),
                        ]),
                    ],
                ),
                io.Boolean.Input("fix_small_faces", default=True, optional=True),
                io.Float.Input("small_face_precision", default=0.1, min=0.001, max=10.0, step=0.01, optional=True),
                io.Boolean.Input("fix_small_edges", default=True, optional=True),
                io.Boolean.Input("fix_wire_gaps", default=True, optional=True),
                io.Boolean.Input("merge_colinear_edges", default=False, optional=True,
                    tooltip="Merge adjacent edges on same curve (fixes fragmented IGES trim curves)"),
                io.Boolean.Input("unify_faces", default=False, optional=True,
                    tooltip="Also unify adjacent faces on same surface (more aggressive simplification)"),
                io.Float.Input("angular_tolerance", default=0.01, min=0.001, max=1.0, step=0.001, optional=True, advanced=True,
                    tooltip="Angular tolerance for curve matching in radians (higher = more aggressive)"),
                io.Float.Input("linear_tolerance", default=0.001, min=0.0001, max=1.0, step=0.0001, optional=True, advanced=True,
                    tooltip="Linear tolerance for curve matching (higher = more aggressive)"),
                io.Boolean.Input("merge_g2_edges", default=False, optional=True, advanced=True,
                    tooltip="Merge adjacent edges with G2 curvature continuity"),
                io.Float.Input("g2_tolerance", default=0.01, min=0.0001, max=1.0, step=0.001, optional=True, advanced=True,
                    tooltip="Curvature tolerance for G2 continuity detection"),
                io.Boolean.Input("preserve_quad_faces", default=True, optional=True,
                    tooltip="Don't merge edges on faces with exactly 4 edges"),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="healed_models", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, precision, max_tolerance,
                execution_mode="single_core", num_workers=4, timeout=120,
                fix_small_faces=True, small_face_precision=0.1,
                fix_small_edges=True, fix_wire_gaps=True,
                merge_colinear_edges=False, unify_faces=False,
                angular_tolerance=0.01, linear_tolerance=0.001,
                merge_g2_edges=False, g2_tolerance=0.01, preserve_quad_faces=True):
        """Heal shapes - dispatches to sequential or parallel based on execution_mode."""
        # Extract scalar values from lists (INPUT_IS_LIST behavior)
        # Handle DynamicCombo for execution_mode
        if isinstance(execution_mode, list):
            execution_mode = execution_mode[0]
        if isinstance(execution_mode, dict):
            execution_mode_val = execution_mode.get("execution_mode", "single_core")
            num_workers_val = execution_mode.get("num_workers", 4)
            timeout_val = execution_mode.get("timeout", 120)
        else:
            execution_mode_val = execution_mode if isinstance(execution_mode, str) else "single_core"
            num_workers_val = num_workers[0] if isinstance(num_workers, list) else num_workers
            timeout_val = timeout[0] if isinstance(timeout, list) else timeout
        precision_val = precision[0] if isinstance(precision, list) else precision
        max_tolerance_val = max_tolerance[0] if isinstance(max_tolerance, list) else max_tolerance
        fix_small_faces_val = fix_small_faces[0] if isinstance(fix_small_faces, list) else fix_small_faces
        small_face_precision_val = small_face_precision[0] if isinstance(small_face_precision, list) else small_face_precision
        fix_small_edges_val = fix_small_edges[0] if isinstance(fix_small_edges, list) else fix_small_edges
        fix_wire_gaps_val = fix_wire_gaps[0] if isinstance(fix_wire_gaps, list) else fix_wire_gaps
        merge_colinear_val = merge_colinear_edges[0] if isinstance(merge_colinear_edges, list) else merge_colinear_edges
        unify_faces_val = unify_faces[0] if isinstance(unify_faces, list) else unify_faces
        angular_tolerance_val = angular_tolerance[0] if isinstance(angular_tolerance, list) else angular_tolerance
        linear_tolerance_val = linear_tolerance[0] if isinstance(linear_tolerance, list) else linear_tolerance
        merge_g2_val = merge_g2_edges[0] if isinstance(merge_g2_edges, list) else merge_g2_edges
        g2_tolerance_val = g2_tolerance[0] if isinstance(g2_tolerance, list) else g2_tolerance
        preserve_quad_faces_val = preserve_quad_faces[0] if isinstance(preserve_quad_faces, list) else preserve_quad_faces

        # Ensure cad_model is a list
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        if execution_mode_val == "parallel":
            return cls._heal_parallel(
                cad_models, num_workers_val, timeout_val, precision_val, max_tolerance_val,
                fix_small_faces_val, small_face_precision_val, fix_small_edges_val, fix_wire_gaps_val,
                merge_colinear_val, unify_faces_val, angular_tolerance_val, linear_tolerance_val,
                merge_g2_val, g2_tolerance_val, preserve_quad_faces_val
            )
        else:
            return cls._heal_sequential(
                cad_models, precision_val, max_tolerance_val,
                fix_small_faces_val, small_face_precision_val, fix_small_edges_val, fix_wire_gaps_val,
                merge_colinear_val, unify_faces_val, angular_tolerance_val, linear_tolerance_val,
                merge_g2_val, g2_tolerance_val, preserve_quad_faces_val
            )

    @classmethod
    def _heal_sequential(cls, cad_models, precision, max_tolerance,
                         fix_small_faces, small_face_precision, fix_small_edges, fix_wire_gaps,
                         merge_colinear_edges, unify_faces, angular_tolerance, linear_tolerance,
                         merge_g2_edges=False, g2_tolerance=0.01, preserve_quad_faces=True):
        """Process models sequentially in main process."""
        results = []
        all_report_lines = []

        for model_idx, cad_model in enumerate(cad_models):
            file_path = cad_model.get("file_path", f"model_{model_idx}")
            filename = os.path.basename(file_path)
            all_report_lines.append(f"\n{'='*60}\nModel {model_idx + 1}/{len(cad_models)}: {filename}\n{'='*60}")

            result, report = cls._heal_single(
                cad_model, precision, max_tolerance,
                fix_small_faces, small_face_precision, fix_small_edges, fix_wire_gaps,
                merge_colinear_edges, unify_faces, angular_tolerance, linear_tolerance,
                merge_g2_edges, g2_tolerance, preserve_quad_faces
            )
            results.append(result)
            all_report_lines.append(report)

        combined_report = "\n".join(all_report_lines)
        return io.NodeOutput(results, combined_report)

    @staticmethod
    def _heal_single(cad_model, precision, max_tolerance,
                     fix_small_faces=True, small_face_precision=0.1,
                     fix_small_edges=True, fix_wire_gaps=True,
                     merge_colinear_edges=False, unify_faces=False,
                     angular_tolerance=0.01, linear_tolerance=0.001,
                     merge_g2_edges=False, g2_tolerance=0.01, preserve_quad_faces=True):
        """
        Apply OCCT ShapeFix healing to the CAD model.
        """
        shape = _get_occ_shape(cad_model)

        report_lines = ["CAD Healing Report", "=" * 50]

        # Import OCC classes needed for counting and G2 merge
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE

        # Count faces before
        def count_faces(shp):
            count = 0
            explorer = TopExp_Explorer(shp, TopAbs_FACE)
            while explorer.More():
                count += 1
                explorer.Next()
            return count

        faces_before = count_faces(shape)
        report_lines.append(f"Faces before: {faces_before}")

        result_shape = shape

        # 1. Fix small faces first (if enabled)
        # ShapeFix_FixSmallFace removes spot faces and strip faces
        if fix_small_faces:
            try:
                fix_small = ShapeFix_FixSmallFace()
                fix_small.Init(result_shape)
                fix_small.SetPrecision(small_face_precision)
                fix_small.Perform()
                result_shape = fix_small.FixShape()
                faces_after_small = count_faces(result_shape)
                removed = faces_before - faces_after_small
                report_lines.append(f"Small faces fix: removed {removed} faces (precision: {small_face_precision})")
            except Exception as e:
                report_lines.append(f"Small faces fix error: {e}")

        # 2. Fix wireframe issues (small edges, gaps)
        if fix_small_edges or fix_wire_gaps:
            try:
                wireframe_fix = ShapeFix_Wireframe(result_shape)
                wireframe_fix.SetPrecision(precision)

                if fix_small_edges:
                    wireframe_fix.SetModeDropSmallEdges(True)
                    wireframe_fix.FixSmallEdges()
                    report_lines.append("Small edges fix applied")

                if fix_wire_gaps:
                    wireframe_fix.FixWireGaps()
                    report_lines.append("Wire gaps fix applied")

                result_shape = wireframe_fix.Shape()
            except Exception as e:
                report_lines.append(f"Wireframe fix error: {e}")

        # 3. General shape fix (fixes curves, pcurves, same parameter, etc.)
        try:
            shape_fix = ShapeFix_Shape(result_shape)
            shape_fix.SetPrecision(precision)
            shape_fix.SetMaxTolerance(max_tolerance)
            shape_fix.Perform()
            result_shape = shape_fix.Shape()
            report_lines.append("General shape fix applied")
        except Exception as e:
            report_lines.append(f"General shape fix error: {e}")

        # 4. Merge colinear edges (fixes fragmented IGES trim curves)
        if merge_colinear_edges:
            from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

            try:
                # Count edges before
                def count_edges(shp):
                    count = 0
                    explorer = TopExp_Explorer(shp, TopAbs_EDGE)
                    while explorer.More():
                        count += 1
                        explorer.Next()
                    return count

                edges_before = count_edges(result_shape)
                faces_mid = count_faces(result_shape)

                # UnifySameDomain merges adjacent edges that lie on the same curve
                # Args: shape, UnifyFaces, UnifyEdges=True, ConcatBSplines=True
                # UnifyFaces=True can be more aggressive in simplifying geometry
                unifier = ShapeUpgrade_UnifySameDomain(result_shape, unify_faces, True, True)

                # Set tolerances for curve matching - higher values = more aggressive merging
                unifier.SetAngularTolerance(angular_tolerance)
                unifier.SetLinearTolerance(linear_tolerance)

                unifier.Build()
                result_shape = unifier.Shape()

                edges_after = count_edges(result_shape)
                faces_after_merge = count_faces(result_shape)
                edge_reduction_pct = 100.0 * (1 - edges_after / edges_before) if edges_before > 0 else 0
                face_reduction = faces_mid - faces_after_merge

                report_lines.append(f"Merge colinear edges: {edges_before} -> {edges_after} edges ({edge_reduction_pct:.1f}% reduction)")
                if unify_faces:
                    report_lines.append(f"  Unified faces: {faces_mid} -> {faces_after_merge} (merged {face_reduction})")
                report_lines.append(f"  Tolerances: angular={angular_tolerance} rad, linear={linear_tolerance}")
            except Exception as e:
                report_lines.append(f"Merge colinear edges error: {e}")

        # 5. Merge G2-continuous edges
        log.info(f" merge_g2_edges={merge_g2_edges}, g2_tolerance={g2_tolerance}, preserve_quad_faces={preserve_quad_faces}")
        if merge_g2_edges:
            log.info(" Starting G2 edge merge...")
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                from OCC.Core.BRepLProp import BRepLProp_CLProps
                from OCC.Core.BRepTools import BRepTools_WireExplorer, BRepTools_ReShape
                from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve, geomconvert
                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                from OCC.Core.GeomAbs import GeomAbs_C2
                from OCC.Core.gp import gp_Vec, gp_Dir
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.TopExp import topexp
                from OCC.Core.TopAbs import TopAbs_WIRE
                from OCC.Core.TopoDS import topods
                import math

                def count_edges_shape(shp):
                    count = 0
                    explorer = TopExp_Explorer(shp, TopAbs_EDGE)
                    while explorer.More():
                        count += 1
                        explorer.Next()
                    return count

                def get_edge_vertices(edge):
                    v_first = topexp.FirstVertex(edge)
                    v_last = topexp.LastVertex(edge)
                    return v_first, v_last

                def vertices_same(v1, v2, tol=1e-6):
                    p1 = BRep_Tool.Pnt(v1)
                    p2 = BRep_Tool.Pnt(v2)
                    return p1.Distance(p2) < tol

                def check_g2(edge1, edge2, curv_tol=0.01, ang_tol=0.01):
                    v1_start, v1_end = get_edge_vertices(edge1)
                    v2_start, v2_end = get_edge_vertices(edge2)

                    junction = None
                    if vertices_same(v1_end, v2_start):
                        junction = ('end', 'start')
                    elif vertices_same(v1_start, v2_end):
                        junction = ('start', 'end')
                    elif vertices_same(v1_end, v2_end):
                        junction = ('end', 'end')
                    elif vertices_same(v1_start, v2_start):
                        junction = ('start', 'start')

                    if junction is None:
                        return False

                    adaptor1 = BRepAdaptor_Curve(edge1)
                    adaptor2 = BRepAdaptor_Curve(edge2)

                    u1 = adaptor1.LastParameter() if junction[0] == 'end' else adaptor1.FirstParameter()
                    u2 = adaptor2.FirstParameter() if junction[1] == 'start' else adaptor2.LastParameter()

                    props1 = BRepLProp_CLProps(adaptor1, 2, 1e-6)
                    props2 = BRepLProp_CLProps(adaptor2, 2, 1e-6)
                    props1.SetParameter(u1)
                    props2.SetParameter(u2)

                    if not props1.IsTangentDefined() or not props2.IsTangentDefined():
                        return False

                    tangent1 = gp_Dir()
                    tangent2 = gp_Dir()
                    props1.Tangent(tangent1)
                    props2.Tangent(tangent2)

                    # gp_Dir.Angle() returns angle between directions
                    angle = tangent1.Angle(tangent2)
                    if angle > ang_tol and abs(angle - math.pi) > ang_tol:
                        return False

                    curv1 = props1.Curvature()
                    curv2 = props2.Curvature()
                    max_curv = max(abs(curv1), abs(curv2), 1e-10)
                    rel_diff = abs(curv1 - curv2) / max_curv

                    if rel_diff > curv_tol and abs(curv1 - curv2) > curv_tol:
                        return False

                    return True

                def get_wire_edges(wire):
                    edges = []
                    explorer = BRepTools_WireExplorer(wire)
                    while explorer.More():
                        edges.append(explorer.Current())
                        explorer.Next()
                    return edges

                def find_g2_groups(edges, tol):
                    if not edges:
                        return []
                    groups = []
                    current = [edges[0]]
                    for i in range(1, len(edges)):
                        if check_g2(edges[i-1], edges[i], tol):
                            current.append(edges[i])
                        else:
                            groups.append(current)
                            current = [edges[i]]
                    groups.append(current)
                    if len(groups) > 1 and len(edges) > 2:
                        if check_g2(edges[-1], edges[0], tol):
                            groups[0] = groups[-1] + groups[0]
                            groups.pop()
                    return groups

                def merge_to_bspline(edges):
                    if len(edges) == 1:
                        return edges[0]
                    try:
                        adaptor = BRepAdaptor_Curve(edges[0])
                        first_bs = geomconvert.CurveToBSplineCurve(adaptor.Curve().Curve(), GeomAbs_C2)
                        converter = GeomConvert_CompCurveToBSplineCurve(first_bs)
                        for edge in edges[1:]:
                            adaptor = BRepAdaptor_Curve(edge)
                            bs = geomconvert.CurveToBSplineCurve(adaptor.Curve().Curve(), GeomAbs_C2)
                            if not converter.Add(bs, 1e-6):
                                return None
                        merged = converter.BSplineCurve()
                        builder = BRepBuilderAPI_MakeEdge(merged)
                        if builder.IsDone():
                            return builder.Edge()
                        return None
                    except Exception as e:
                        log.debug("G2 BSpline merge failed: %s", e)
                        return None

                edges_before_g2 = count_edges_shape(result_shape)
                total_faces = count_faces(result_shape)
                log.info(f" G2 merge: scanning {total_faces} faces, {edges_before_g2} edges...")
                reshaper = BRepTools_ReShape()
                total_merged = 0
                faces_processed = 0
                faces_skipped = 0
                face_idx = 0

                face_explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
                while face_explorer.More():
                    face_idx += 1
                    if face_idx % 100 == 0:
                        log.info(f" G2 merge: face {face_idx}/{total_faces}, merged {total_merged} edges so far...")
                    face = topods.Face(face_explorer.Current())

                    edge_count = 0
                    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
                    while wire_exp.More():
                        wire = topods.Wire(wire_exp.Current())
                        edge_count += len(get_wire_edges(wire))
                        wire_exp.Next()

                    if preserve_quad_faces and edge_count == 4:
                        faces_skipped += 1
                        face_explorer.Next()
                        continue

                    faces_processed += 1

                    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
                    while wire_exp.More():
                        wire = topods.Wire(wire_exp.Current())
                        edges = get_wire_edges(wire)

                        if len(edges) >= 2:
                            groups = find_g2_groups(edges, g2_tolerance)
                            for group in groups:
                                if len(group) > 1:
                                    merged = merge_to_bspline(group)
                                    if merged is not None:
                                        reshaper.Replace(group[0], merged)
                                        for e in group[1:]:
                                            reshaper.Remove(e)
                                        total_merged += len(group) - 1

                        wire_exp.Next()

                    face_explorer.Next()

                log.info(f" G2 merge: applying changes...")
                result_shape = reshaper.Apply(result_shape)
                edges_after_g2 = count_edges_shape(result_shape)
                log.info(f" G2 merge done: {edges_before_g2} -> {edges_after_g2} edges")

                report_lines.append(f"Merge G2 edges: {edges_before_g2} -> {edges_after_g2} edges (merged {total_merged})")
                report_lines.append(f"  Processed {faces_processed} faces, skipped {faces_skipped} quad faces")
                report_lines.append(f"  G2 tolerance: {g2_tolerance}")

            except Exception as e:
                report_lines.append(f"Merge G2 edges error: {e}")
                log.error("Failed to merge G2 edges during shape healing", exc_info=True)

        # Count faces after
        faces_after = count_faces(result_shape)
        report_lines.append("")
        report_lines.append(f"Faces after: {faces_after}")
        report_lines.append(f"Total faces removed: {faces_before - faces_after}")

        # Build output model - save to brep file
        result_model = _make_cad_model(result_shape, cad_model, "healed")

        report = "\n".join(report_lines)
        log.info(f" heal_shape: before={faces_before}, after={faces_after}, removed={faces_before - faces_after}")

        return (result_model, report)

    @classmethod
    def _heal_parallel(cls, cad_models, num_workers, timeout, precision, max_tolerance,
                       fix_small_faces, small_face_precision, fix_small_edges, fix_wire_gaps,
                       merge_colinear_edges, unify_faces, angular_tolerance, linear_tolerance,
                       merge_g2_edges=False, g2_tolerance=0.01, preserve_quad_faces=True):
        """Heal shapes using subprocess per model with OS-level timeout."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder
        import subprocess
        import tempfile
        import time
        import sys
        import json

        log.info(f"Heal Parallel: Processing {len(cad_models)} models with {num_workers} workers, timeout={timeout}s")
        log.info(f"Heal Parallel: Healing settings: precision={precision}, max_tolerance={max_tolerance}, "
              f"fix_small_faces={fix_small_faces} (precision={small_face_precision}), "
              f"fix_small_edges={fix_small_edges}, fix_wire_gaps={fix_wire_gaps}")
        log.info(f"Heal Parallel: Edge merging: merge_colinear={merge_colinear_edges}, unify_faces={unify_faces}, "
              f"angular_tolerance={angular_tolerance}, linear_tolerance={linear_tolerance}")
        log.info(f"Heal Parallel: G2 merging: merge_g2={merge_g2_edges}, g2_tolerance={g2_tolerance}, "
              f"preserve_quads={preserve_quad_faces}")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cadabra_heal_sub_")

        # Create persistent log directory in ComfyUI output folder
        from datetime import datetime
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs", f"heal_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        log.info(f"Heal Parallel: Logs: {log_dir}")

        # Path to the subprocess script
        script_path = os.path.join(os.path.dirname(__file__), "heal_subprocess.py")

        # Prepare work items
        work_items = []
        skipped = 0

        for idx, cm in enumerate(cad_models):
            brep_path = cm.get("brep_path")
            file_path = cm.get("file_path", f"model_{idx}")
            if brep_path is None:
                skipped += 1
                continue

            # Use the existing brep_path as input
            input_brep = brep_path
            output_brep = os.path.join(temp_dir, f"output_{idx}.brep")
            result_file = os.path.join(temp_dir, f"result_{idx}.json")
            # Get filename stem for log file
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            log_file = os.path.join(log_dir, f"{filename_stem}.log")

            work_items.append({
                "idx": idx,
                "input_brep": input_brep,
                "output_brep": output_brep,
                "result_file": result_file,
                "log_file": log_file,
                "file_path": file_path,
            })

        if skipped > 0:
            log.warning(f"Heal Parallel: Skipped {skipped} models without OCC shape")

        if len(work_items) == 0:
            raise ValueError("No valid CAD models to process")

        def run_heal_subprocess(item):
            """Run healing in subprocess with timeout."""
            cmd = [
                sys.executable,
                script_path,
                item["input_brep"],
                item["output_brep"],
                str(precision),
                str(max_tolerance),
                f"--result-file={item['result_file']}",
                f"--log-file={item['log_file']}",
                f"--fix-small-faces={'true' if fix_small_faces else 'false'}",
                f"--small-face-precision={small_face_precision}",
                f"--fix-small-edges={'true' if fix_small_edges else 'false'}",
                f"--fix-wire-gaps={'true' if fix_wire_gaps else 'false'}",
                f"--merge-colinear={'true' if merge_colinear_edges else 'false'}",
                f"--unify-faces={'true' if unify_faces else 'false'}",
                f"--angular-tolerance={angular_tolerance}",
                f"--linear-tolerance={linear_tolerance}",
                f"--merge-g2-edges={'true' if merge_g2_edges else 'false'}",
                f"--g2-tolerance={g2_tolerance}",
                f"--preserve-quad-faces={'true' if preserve_quad_faces else 'false'}",
            ]
            item_start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5  # Small buffer for startup
                )
                elapsed = time.time() - item_start
                # Read result file if it exists
                if os.path.exists(item["result_file"]):
                    with open(item["result_file"], 'r') as f:
                        return {**json.load(f), "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                elif result.returncode == 0:
                    return {"success": True, "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                else:
                    return {"success": False, "error": result.stderr or "Unknown error", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except subprocess.TimeoutExpired:
                elapsed = time.time() - item_start
                return {"success": False, "error": f"Timed out after {timeout}s", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - item_start
                return {"success": False, "error": str(e), "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}

        # Process in parallel using ThreadPoolExecutor (subprocess handles CPU work)
        results = [None] * len(work_items)
        start_time = time.time()
        report_lines = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(run_heal_subprocess, item): i
                    for i, item in enumerate(work_items)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {"success": False, "error": str(e), "file_path": work_items[idx]["file_path"], "log_file": work_items[idx]["log_file"]}
                    completed += 1
                    elapsed = time.time() - start_time
                    _progress_bar(completed, len(work_items), elapsed, prefix="Healing: ")

            # Build CAD_MODEL dicts from results and print per-model timing
            healed_models = []
            failed = 0

            # Threshold for showing log file path (seconds)
            SLOW_THRESHOLD = 10.0

            for i, result in enumerate(results):
                elapsed_str = f"{result.get('elapsed', 0):.1f}s" if result else "?"
                elapsed_val = result.get('elapsed', 0) if result else 0
                filename = os.path.basename(result.get("file_path", work_items[i]["file_path"])) if result else os.path.basename(work_items[i]["file_path"])
                log_path = result.get("log_file", work_items[i].get("log_file", "")) if result else work_items[i].get("log_file", "")

                if result is None or not result.get("success", False):
                    failed += 1
                    err = result.get("error", "Unknown") if result else "No result"
                    msg = f"{filename}: FAILED ({elapsed_str}) - {err}"
                    log.info(f"Heal Parallel: {msg}")
                    report_lines.append(msg)
                    if log_path:
                        log.info(f"Heal Parallel:   Log: {log_path}")
                    continue

                output_brep = result.get("output_brep", work_items[i]["output_brep"])
                if os.path.exists(output_brep):
                    # Use the output BREP file directly as brep_path
                    original_cm = cad_models[work_items[i]["idx"]] if work_items[i]["idx"] < len(cad_models) else {}
                    cad_result = {
                        "brep_path": output_brep,
                        "format": "brep",
                        "file_path": result["file_path"],
                    }
                    if "metadata" in original_cm:
                        cad_result["metadata"] = original_cm["metadata"]

                    healed_models.append(cad_result)

                    # Log detailed diagnostics with timing
                    faces_before = result.get("faces_before", "?")
                    faces_after = result.get("faces_after", "?")
                    edges_before = result.get("edges_before", "?")
                    edges_after = result.get("edges_after", "?")

                    msg = (f"{filename}: faces={faces_before}->{faces_after}, "
                           f"edges={edges_before}->{edges_after} ({elapsed_str})")
                    log.info(f"Heal Parallel: {msg}")
                    report_lines.append(msg)

                    # Print detailed timings if available
                    timings = result.get("timings", {})
                    if timings:
                        total_time = timings.get("total", 0)
                        log.info(f"Heal Parallel:   total={total_time:.1f}s")

                    # Print log path for slow operations
                    if elapsed_val > SLOW_THRESHOLD and log_path:
                        log.info(f"Heal Parallel:   Log: {log_path}")
                else:
                    failed += 1
                    msg = f"{filename}: FAILED ({elapsed_str}) - output file missing"
                    log.info(f"Heal Parallel: {msg}")
                    report_lines.append(msg)
                    if log_path:
                        log.info(f"Heal Parallel:   Log: {log_path}")

            summary = f"Completed: {len(healed_models)} healed, {failed} failed"
            log.info(f"Heal Parallel: {summary}")
            report_lines.append(summary)

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Heal Parallel: Cleaned up temp dir")
            except Exception as e:
                log.warning(f"Heal Parallel: Could not clean temp dir: {e}")

        report = "\n".join(report_lines)
        return io.NodeOutput(healed_models, report)


class CADMergeVertices(io.ComfyNode):
    """Merge duplicate vertices in mesh using PyVista/VTK clean."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADMergeVertices",
            display_name="Merge Vertices",
            category="CADabra/Utility",
            is_input_list=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("tolerance", default=1e-6, min=1e-9, max=10.0, step=0.000001,
                               tooltip="Distance tolerance for merging vertices. Default 1e-6 is safe for OCC meshes."),
                io.Float.Input("min_component_ratio", default=0.0, min=0.0, max=0.1, step=0.001,
                               tooltip="Remove mesh components smaller than this ratio of total faces (0=keep all)"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, trimesh, tolerance, min_component_ratio):
        """Merge duplicate vertices using PyVista's clean() which wraps VTK's vtkCleanPolyData."""
        import pyvista as pv
        import numpy as np
        import trimesh as tm

        meshes = trimesh if isinstance(trimesh, list) else [trimesh]
        tol = tolerance[0] if isinstance(tolerance, list) else tolerance
        min_comp_ratio = min_component_ratio[0] if isinstance(min_component_ratio, list) else min_component_ratio

        log.info(f"MergeVertices: Merging vertices with tolerance={tol}, min_component_ratio={min_comp_ratio}")

        results = []
        for i, mesh in enumerate(meshes):
            # Get mesh name for logging
            mesh_name = mesh.metadata.get('file_path', f'mesh_{i}')
            if isinstance(mesh_name, str):
                mesh_name = os.path.basename(mesh_name)

            # Convert to PyVista
            verts = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
            pv_mesh = pv.PolyData(verts, pv_faces)

            # Preserve face attributes as cell data
            if 'cad_face_id' in mesh.face_attributes:
                pv_mesh.cell_data['cad_face_id'] = mesh.face_attributes['cad_face_id']

            # Clean/merge vertices
            verts_before = pv_mesh.n_points
            cleaned = pv_mesh.clean(tolerance=tol, absolute=True)

            # Extract faces using regular_faces (PyVista >= 0.43)
            if hasattr(cleaned, 'regular_faces'):
                new_faces = cleaned.regular_faces
            else:
                # Fallback for older PyVista
                pv_faces_clean = cleaned.faces
                n_cells = cleaned.n_cells
                new_faces = []
                idx = 0
                for _ in range(n_cells):
                    n = pv_faces_clean[idx]
                    if n == 3:
                        new_faces.append([
                            pv_faces_clean[idx + 1],
                            pv_faces_clean[idx + 2],
                            pv_faces_clean[idx + 3]
                        ])
                    idx += n + 1
                new_faces = np.array(new_faces)

            # Create new trimesh WITHOUT automatic processing (to keep faces in sync with attributes)
            new_mesh = tm.Trimesh(vertices=cleaned.points, faces=new_faces, process=False)

            # Get cad_face_ids before any filtering
            # Note: PyVista cell_data includes ALL cells (verts, lines, faces, strips)
            # but regular_faces only returns face cells, so we need to slice correctly
            # Cell ordering: verts -> lines -> faces -> strips
            cad_face_ids = None
            if 'cad_face_id' in cleaned.cell_data:
                all_cad_ids = np.array(cleaned.cell_data['cad_face_id'])
                # Only get the face portion (skip verts and lines)
                face_start = cleaned.n_verts + cleaned.n_lines
                face_end = face_start + len(new_faces)
                cad_face_ids = all_cad_ids[face_start:face_end]

            # Remove degenerate faces manually (keeping cad_face_ids in sync)
            degenerate_mask = new_mesh.nondegenerate_faces()
            num_degenerate = (~degenerate_mask).sum()
            if num_degenerate > 0:
                new_mesh.update_faces(degenerate_mask)
                if cad_face_ids is not None:
                    cad_face_ids = cad_face_ids[degenerate_mask]
                log.info(f"MergeVertices: {mesh_name}: removed {num_degenerate} degenerate faces")

            # Remove duplicate faces created by vertex merging
            unique_mask = new_mesh.unique_faces()
            num_duplicates = (~unique_mask).sum()
            if num_duplicates > 0:
                new_mesh.update_faces(unique_mask)
                if cad_face_ids is not None:
                    cad_face_ids = cad_face_ids[unique_mask]
                log.info(f"MergeVertices: {mesh_name}: removed {num_duplicates} duplicate faces")

            # Filter small disconnected components (if enabled)
            if min_comp_ratio > 0 and len(new_mesh.faces) > 0:
                import trimesh as trimesh_module
                # Get connected components using face adjacency
                components = trimesh_module.graph.connected_components(
                    new_mesh.face_adjacency,
                    nodes=np.arange(len(new_mesh.faces))
                )

                if len(components) > 1:
                    num_faces = len(new_mesh.faces)
                    min_faces = int(num_faces * min_comp_ratio)
                    if min_faces < 1:
                        min_faces = 1

                    # Find largest component and components to keep
                    comp_sizes = {i: len(comp) for i, comp in enumerate(components)}
                    largest_comp_idx = max(comp_sizes.keys(), key=lambda c: comp_sizes[c])

                    # Build mask for faces to keep
                    keep_mask = np.zeros(num_faces, dtype=bool)
                    kept_comps = 0
                    removed_comps = 0
                    for comp_idx, face_indices in enumerate(components):
                        if comp_idx == largest_comp_idx or len(face_indices) >= min_faces:
                            keep_mask[face_indices] = True
                            kept_comps += 1
                        else:
                            removed_comps += 1

                    if removed_comps > 0:
                        new_mesh.update_faces(keep_mask)
                        if cad_face_ids is not None:
                            cad_face_ids = cad_face_ids[keep_mask]
                        log.info(f"MergeVertices: {mesh_name}: removed {removed_comps} small components "
                              f"(kept {kept_comps} components, {keep_mask.sum()} faces)")

            # Store filtered cad_face_ids
            if cad_face_ids is not None:
                new_mesh.face_attributes['cad_face_id'] = cad_face_ids
                new_mesh.metadata['has_cad_face_ids'] = True

            # Copy other metadata
            if hasattr(mesh, 'metadata'):
                for key, value in mesh.metadata.items():
                    if key not in new_mesh.metadata:
                        new_mesh.metadata[key] = value

            verts_after = cleaned.n_points
            log.info(f"MergeVertices: {mesh_name}: {verts_before} -> {verts_after} vertices "
                  f"(merged {verts_before - verts_after})")

            results.append(new_mesh)

        return io.NodeOutput(results)


def _replace_degenerate_face(face, iterations, max_degree):
    """
    Replace a single degenerate face with a filled surface.

    Uses BRepOffsetAPI_MakeFilling to create a new surface through
    the non-degenerate boundary edges.

    Args:
        face: TopoDS_Face with degenerate edge(s)
        iterations: Number of smoothing iterations (NbIter in SetResolParam)
        max_degree: Maximum BSpline degree for result (MaxDeg in SetApproxParam)

    Returns:
        tuple: (new_face or None, status_message)
    """
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCC.Core.GeomAbs import GeomAbs_C0
    from OCC.Core.BRep import BRep_Tool

    # Extract non-degenerate edges
    non_degen_edges = []
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        if not BRep_Tool.Degenerated(edge):
            non_degen_edges.append(edge)
        explorer.Next()

    if len(non_degen_edges) < 2:
        return None, f"Only {len(non_degen_edges)} non-degenerate edge(s)"

    # Create filling surface through boundary edges
    filling = BRepOffsetAPI_MakeFilling()

    # Set approximation parameters: MaxDeg, MaxSegments
    filling.SetApproxParam(max_degree, 9)

    # Set resolution parameters: Degree, NbPtsOnCur, NbIter, Anisotropie
    filling.SetResolParam(3, 15, iterations, False)

    # Add edges as constraints
    for edge in non_degen_edges:
        filling.Add(edge, GeomAbs_C0)

    try:
        filling.Build()
    except Exception as e:
        return None, f"Build failed: {e}"

    if not filling.IsDone():
        return None, "MakeFilling not done"

    # Get the face from the result shape
    result_shape = filling.Shape()
    face_explorer = TopExp_Explorer(result_shape, TopAbs_FACE)
    if face_explorer.More():
        return topods.Face(face_explorer.Current()), "OK"
    else:
        return None, "No face in result"


class CADFixDegenerateFaces(io.ComfyNode):
    """Replace degenerate CAD faces with valid filled surfaces."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADFixDegenerateFaces",
            display_name="Fix Degenerate Faces",
            category="CADabra/Utility",
            is_input_list=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model", tooltip="CAD model(s) with degenerate faces to fix"),
                io.Int.Input("iterations", default=5, min=1, max=20, step=1,
                             tooltip="Smoothness iterations for filling surface (higher = smoother)"),
                io.Int.Input("max_degree", default=8, min=3, max=14, step=1,
                             tooltip="Maximum BSpline degree for replacement surface"),
                io.Float.Input("sew_tolerance", default=0.001, min=1e-6, max=1.0, step=0.0001,
                               tooltip="Tolerance for sewing faces back together after replacement"),
                io.DynamicCombo.Input("execution_mode",
                    tooltip="single_core: sequential in main process. parallel: subprocesses with timeout and crash isolation",
                    options=[
                        io.DynamicCombo.Option("single_core", []),
                        io.DynamicCombo.Option("parallel", [
                            io.Int.Input("num_workers", default=4, min=1, max=32,
                                tooltip="Number of parallel subprocesses (parallel mode only)"),
                            io.Int.Input("timeout", default=120, min=10, max=600,
                                tooltip="Timeout per model in seconds (parallel mode only)"),
                        ]),
                    ],
                ),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, cad_model, iterations, max_degree, sew_tolerance, execution_mode="single_core"):
        """Fix degenerate faces - dispatches to sequential or parallel based on execution_mode."""
        # Handle DynamicCombo for execution_mode
        if isinstance(execution_mode, list):
            execution_mode = execution_mode[0]
        if isinstance(execution_mode, dict):
            execution_mode_val = execution_mode.get("execution_mode", "single_core")
            num_workers_val = execution_mode.get("num_workers", 4)
            timeout_val = execution_mode.get("timeout", 120)
        else:
            execution_mode_val = execution_mode if isinstance(execution_mode, str) else "single_core"
            num_workers_val = 4
            timeout_val = 120

        iter_val = iterations[0] if isinstance(iterations, list) else iterations
        deg_val = max_degree[0] if isinstance(max_degree, list) else max_degree
        tol_val = sew_tolerance[0] if isinstance(sew_tolerance, list) else sew_tolerance

        # Ensure cad_model is a list
        cad_models = cad_model if isinstance(cad_model, list) else [cad_model]

        if execution_mode_val == "parallel":
            return cls._fix_parallel(cad_models, num_workers_val, timeout_val, iter_val, deg_val, tol_val)
        else:
            return cls._fix_sequential(cad_models, iter_val, deg_val, tol_val)

    @classmethod
    def _fix_sequential(cls, cad_models, iterations, max_degree, sew_tolerance):
        """Process all CAD models, replacing degenerate faces."""
        from OCC.Core.BRep import BRep_Tool

        # Handle list params
        iter_val = iterations[0] if isinstance(iterations, list) else iterations
        deg_val = max_degree[0] if isinstance(max_degree, list) else max_degree
        tol_val = sew_tolerance[0] if isinstance(sew_tolerance, list) else sew_tolerance

        results = []
        # Table data: (name, total_faces, degen_faces, fixed, failed)
        table_data = []

        for cm in cad_models:
            shape = _get_occ_shape(cm)

            # Get filename
            file_path = cm.get("file_path", "model")
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            # Find all faces and categorize
            normal_faces = []
            degen_faces = []

            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = topods.Face(face_explorer.Current())

                # Check for degenerate edges
                has_degen = False
                edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
                while edge_exp.More():
                    edge = topods.Edge(edge_exp.Current())
                    if BRep_Tool.Degenerated(edge):
                        has_degen = True
                        break
                    edge_exp.Next()

                if has_degen:
                    degen_faces.append(face)
                else:
                    normal_faces.append(face)

                face_explorer.Next()

            total_faces = len(normal_faces) + len(degen_faces)

            # If no degenerate faces, pass through unchanged
            if len(degen_faces) == 0:
                log.info(f"FixDegenerateFaces: {file_name}: No degenerate faces found")
                table_data.append((file_name, total_faces, 0, 0, 0))
                results.append(cm)
                continue

            # Replace degenerate faces
            fixed_count = 0
            failed_count = 0
            replacement_faces = []

            for degen_face in degen_faces:
                new_face, status = _replace_degenerate_face(degen_face, iter_val, deg_val)
                if new_face is not None:
                    replacement_faces.append(new_face)
                    fixed_count += 1
                else:
                    # Keep original face if replacement fails
                    replacement_faces.append(degen_face)
                    failed_count += 1
                    log.info(f"FixDegenerateFaces: {file_name}: Failed to fix face: {status}")

            # Rebuild shape with replaced faces
            all_result_faces = normal_faces + replacement_faces

            # Sew faces back together
            sewer = BRepBuilderAPI_Sewing(tol_val)
            for face in all_result_faces:
                sewer.Add(face)
            sewer.Perform()
            result_shape = sewer.SewedShape()

            # Create result CAD model
            result_cm = _make_cad_model(result_shape, cm)
            results.append(result_cm)

            log.info(f"FixDegenerateFaces: {file_name}: {len(degen_faces)} degenerate -> "
                  f"{fixed_count} fixed, {failed_count} failed")
            table_data.append((file_name, total_faces, len(degen_faces), fixed_count, failed_count))

        # Create markdown table summary
        name_width = max(len("Model"), max(len(d[0]) for d in table_data)) if table_data else 5
        report_lines = [
            f"| {'Model':<{name_width}} | Faces | Degen | Fixed | Failed |",
            f"|{'-' * (name_width + 2)}|-------|-------|-------|--------|",
        ]
        for name, faces, degen, fixed, failed in table_data:
            report_lines.append(
                f"| {name:<{name_width}} | {faces:>5} | {degen:>5} | {fixed:>5} | {failed:>6} |"
            )

        report = "\n".join(report_lines)
        log.info(f"FixDegenerateFaces: Processed {len(cad_models)} model(s)")

        return io.NodeOutput(results, report)

    @classmethod
    def _fix_parallel(cls, cad_models, num_workers, timeout, iterations, max_degree, sew_tolerance):
        """Fix degenerate faces using subprocess per model with OS-level timeout."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder
        import subprocess
        import tempfile
        import time
        import json
        import sys

        log.info(f"FixDegenerateFaces Parallel: Processing {len(cad_models)} models with {num_workers} workers, timeout={timeout}s")
        log.info(f"FixDegenerateFaces Parallel: Settings: iterations={iterations}, max_degree={max_degree}, sew_tolerance={sew_tolerance}")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cadabra_fixdegen_sub_")

        # Create persistent log directory in ComfyUI output folder
        from datetime import datetime
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        log_dir = os.path.join(output_dir, "cadabra_logs", f"fixdegen_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        log.info(f"FixDegenerateFaces Parallel: Logs: {log_dir}")

        # Path to the subprocess script
        script_path = os.path.join(os.path.dirname(__file__), "fix_degen_subprocess.py")

        # Prepare work items
        work_items = []
        skipped = 0

        for idx, cm in enumerate(cad_models):
            brep_path = cm.get("brep_path")
            file_path = cm.get("file_path", f"model_{idx}")
            if brep_path is None:
                skipped += 1
                continue

            # Use the existing brep_path as input
            input_brep = brep_path
            output_brep = os.path.join(temp_dir, f"output_{idx}.brep")
            result_file = os.path.join(temp_dir, f"result_{idx}.json")
            filename_stem = os.path.splitext(os.path.basename(file_path))[0]
            log_file = os.path.join(log_dir, f"{filename_stem}.log")

            work_items.append({
                "idx": idx,
                "input_brep": input_brep,
                "output_brep": output_brep,
                "result_file": result_file,
                "log_file": log_file,
                "file_path": file_path,
            })

        if skipped > 0:
            log.warning(f"FixDegenerateFaces Parallel: Skipped {skipped} models without OCC shape")

        if len(work_items) == 0:
            raise ValueError("No valid CAD models to process")

        def run_fix_subprocess(item):
            """Run fixing in subprocess with timeout."""
            cmd = [
                sys.executable,
                script_path,
                item["input_brep"],
                item["output_brep"],
                str(iterations),
                str(max_degree),
                str(sew_tolerance),
                f"--result-file={item['result_file']}",
                f"--log-file={item['log_file']}",
            ]
            item_start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5
                )
                elapsed = time.time() - item_start
                if os.path.exists(item["result_file"]):
                    with open(item["result_file"], 'r') as f:
                        return {**json.load(f), "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                elif result.returncode == 0:
                    return {"success": True, "file_path": item["file_path"], "output_brep": item["output_brep"], "log_file": item["log_file"], "elapsed": elapsed}
                else:
                    return {"success": False, "error": result.stderr or "Unknown error", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except subprocess.TimeoutExpired:
                elapsed = time.time() - item_start
                return {"success": False, "error": f"Timed out after {timeout}s", "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - item_start
                return {"success": False, "error": str(e), "file_path": item["file_path"], "log_file": item["log_file"], "elapsed": elapsed}

        # Process in parallel
        results = [None] * len(work_items)
        start_time = time.time()
        table_data = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(run_fix_subprocess, item): i
                    for i, item in enumerate(work_items)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = {"success": False, "error": str(e), "file_path": work_items[idx]["file_path"], "log_file": work_items[idx]["log_file"]}
                    completed += 1
                    elapsed = time.time() - start_time
                    _progress_bar(completed, len(work_items), elapsed, prefix="Fixing: ")

            # Build results
            fixed_models = []
            failed = 0
            SLOW_THRESHOLD = 10.0

            for i, result in enumerate(results):
                elapsed_str = f"{result.get('elapsed', 0):.1f}s" if result else "?"
                elapsed_val = result.get('elapsed', 0) if result else 0
                filename = os.path.basename(result.get("file_path", work_items[i]["file_path"])) if result else os.path.basename(work_items[i]["file_path"])
                filename_stem = os.path.splitext(filename)[0]
                log_path = result.get("log_file", work_items[i].get("log_file", "")) if result else work_items[i].get("log_file", "")

                if result is None or not result.get("success", False):
                    failed += 1
                    err = result.get("error", "Unknown") if result else "No result"
                    log.info(f"FixDegenerateFaces Parallel: {filename}: FAILED ({elapsed_str}) - {err}")
                    if log_path:
                        log.info(f"FixDegenerateFaces Parallel:   Log: {log_path}")
                    table_data.append((filename_stem, "?", "?", 0, 1))
                    continue

                output_brep = result.get("output_brep", work_items[i]["output_brep"])
                if os.path.exists(output_brep):
                    # Use the output BREP file directly as brep_path
                    original_cm = cad_models[work_items[i]["idx"]] if work_items[i]["idx"] < len(cad_models) else {}
                    cad_result = {
                        "brep_path": output_brep,
                        "format": "brep",
                        "file_path": result["file_path"],
                    }
                    if "metadata" in original_cm:
                        cad_result["metadata"] = original_cm["metadata"]

                    fixed_models.append(cad_result)

                    faces_before = result.get("faces_before", "?")
                    degen_before = result.get("degen_before", "?")
                    fixed_count = result.get("fixed", 0)
                    failed_count = result.get("failed", 0)

                    log.info(f"FixDegenerateFaces Parallel: {filename}: "
                          f"degen={degen_before}, fixed={fixed_count}, failed={failed_count} ({elapsed_str})")

                    if elapsed_val > SLOW_THRESHOLD and log_path:
                        log.info(f"FixDegenerateFaces Parallel:   Log: {log_path}")

                    table_data.append((filename_stem, faces_before, degen_before, fixed_count, failed_count))
                else:
                    failed += 1
                    log.info(f"FixDegenerateFaces Parallel: {filename}: FAILED ({elapsed_str}) - output file missing")
                    table_data.append((filename_stem, "?", "?", 0, 1))

            log.info(f"FixDegenerateFaces Parallel: Completed: {len(fixed_models)} fixed, {failed} failed")

            # Create report
            name_width = max(len("Model"), max(len(str(d[0])) for d in table_data)) if table_data else 5
            report_lines = [
                f"| {'Model':<{name_width}} | Faces | Degen | Fixed | Failed |",
                f"|{'-' * (name_width + 2)}|-------|-------|-------|--------|",
            ]
            for name, faces, degen, fixed_c, failed_c in table_data:
                report_lines.append(
                    f"| {str(name):<{name_width}} | {str(faces):>5} | {str(degen):>5} | {fixed_c:>5} | {failed_c:>6} |"
                )
            report = "\n".join(report_lines)

        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
                log.info(f"FixDegenerateFaces Parallel: Cleaned up temp dir")
            except Exception as e:
                log.warning(f"FixDegenerateFaces Parallel: Could not clean temp dir: {e}")

        return io.NodeOutput(fixed_models, report)


# Node mappings for registration
NODE_CLASS_MAPPINGS = {
    "CADExtractFaces": CADExtractFaces,
    "CADSewFaces": CADSewFaces,
    "CADGetFilename": CADGetFilename,
    "CADCheckOverlappingFaces": CADCheckOverlappingFaces,
    "CADSplitComponents": CADSplitComponents,
    "CADSave": CADSave,
    "CADHealShape": CADHealShape,
    "CADMergeVertices": CADMergeVertices,
    "CADFixDegenerateFaces": CADFixDegenerateFaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CADExtractFaces": "CAD Extract Faces",
    "CADSewFaces": "CAD Sew Faces",
    "CADGetFilename": "Get CAD Filename",
    "CADCheckOverlappingFaces": "CAD Check Overlapping Faces",
    "CADSplitComponents": "CAD Split Components",
    "CADSave": "CAD Save",
    "CADHealShape": "CAD Heal Shape",
    "CADMergeVertices": "CAD Merge Vertices",
    "CADFixDegenerateFaces": "CAD Fix Degenerate Faces",
}
