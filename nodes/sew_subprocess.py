#!/usr/bin/env python
"""
Standalone subprocess script for CAD sewing with OS-level timeout.

Usage:
    python sew_subprocess.py <input_brep> <output_brep> <tolerance> [--result-file=path] [--log-file=path]

This script is invoked by subprocess.run() with a timeout, allowing the OS
to kill it if sewing hangs on degenerate geometry.

Exit codes:
    0 - Success
    1 - Error (details in stderr or result file)
"""

import logging
import sys
import argparse
import json
import os
import time

log = logging.getLogger("cadabra")


def count_connected_components(shape):
    """
    Count disconnected components in a shape using face adjacency.

    Returns the number of connected components (1 = fully connected).
    """
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    # Get all faces
    all_faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        all_faces.append(topods.Face(explorer.Current()))
        explorer.Next()

    if len(all_faces) == 0:
        return 0

    # Build indexed map of all edges
    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

    # Build edge index to face indices mapping
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

    # Build adjacency from shared edges
    adjacency = {i: set() for i in range(len(all_faces))}
    for edge_index, face_indices in edge_to_faces.items():
        face_list = list(face_indices)
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adjacency[face_list[i]].add(face_list[j])
                adjacency[face_list[j]].add(face_list[i])

    # Count connected components using BFS
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


def count_free_edges(shape):
    """
    Count free (open) edges in a shape.

    A free edge is shared by only one face (boundary edge).
    """
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    # Build indexed map of all edges
    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

    # Count how many faces each edge belongs to
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

    # Free edges are those with only 1 adjacent face
    free_count = sum(1 for count in edge_face_count.values() if count == 1)
    return free_count


def main():
    parser = argparse.ArgumentParser(description='Sew CAD faces from BREP file')
    parser.add_argument('input_brep', help='Input BREP file path')
    parser.add_argument('output_brep', help='Output BREP file path')
    parser.add_argument('tolerance', type=float, help='Sewing tolerance')
    parser.add_argument('--result-file', help='Path to write JSON result (optional)')
    parser.add_argument('--log-file', help='Path to write verbose log (optional)')
    args = parser.parse_args()

    # If log file specified, redirect stdout/stderr to capture all output
    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if args.log_file:
        log_handle = open(args.log_file, 'w')
        sys.stdout = log_handle
        sys.stderr = log_handle
        log.info(f"[CADabra Sew Subprocess]")
        log.info(f"Input: {args.input_brep}")
        log.info(f"Output: {args.output_brep}")
        log.info(f"Tolerance: {args.tolerance}")
        log.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("=" * 60)
        sys.stdout.flush()

    try:
        result = sew_brep(args.input_brep, args.output_brep, args.tolerance)

        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f)

        # Write final status to log
        if log_handle:
            log.info("=" * 60)
            if result['success']:
                log.info(f"SUCCESS: {result['faces_before']}->{result['faces_after']} faces, "
                      f"components: {result.get('components_before', '?')}->{result.get('components_after', '?')}, "
                      f"free_edges: {result.get('free_edges_before', '?')}->{result['diagnostics']['free_edges']}")
                if 'timings' in result:
                    log.info("\nTimings:")
                    for step, duration in result['timings'].items():
                        log.info(f"  {step}: {duration:.3f}s")
            else:
                log.info(f"FAILED: {result['error']}")
            log.info(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.flush()

        # Restore stdout/stderr
        if log_handle:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()

        if result['success']:
            # Print summary to stdout
            log.info(f"OK: {result['faces_before']}->{result['faces_after']} faces, "
                  f"components={result.get('components_before', '?')}->{result.get('components_after', '?')}, "
                  f"free_edges={result.get('free_edges_before', '?')}->{result['diagnostics']['free_edges']}")
            sys.exit(0)
        else:
            log.info(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        if log_handle:
            log.info(f"\nEXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()
        raise


def sew_brep(input_path, output_path, tolerance):
    """Perform sewing on BREP file with detailed timing."""
    timings = {}

    log.info(f"\n[Sewing CAD file: {os.path.basename(input_path)}]")
    sys.stdout.flush()

    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopoDS import TopoDS_Shape, topods
        from OCC.Core.BRepTools import breptools
        from OCC.Core.BRep import BRep_Builder

        # --- Read input BREP ---
        log.info(f"Reading BREP file...")
        sys.stdout.flush()
        t0 = time.time()
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        breptools.Read(shape, str(input_path), builder)
        timings['read_brep'] = time.time() - t0
        log.info(f"[TIMING] read_brep: {timings['read_brep']:.3f}s")

        # --- Count faces before ---
        log.info(f"Counting faces before sewing...")
        t0 = time.time()
        face_count_before = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_count_before += 1
            explorer.Next()
        timings['count_faces_before'] = time.time() - t0
        log.info(f"[TIMING] count_faces_before: {timings['count_faces_before']:.3f}s ({face_count_before} faces)")

        # --- Count connected components before ---
        log.info(f"Counting connected components before sewing...")
        sys.stdout.flush()
        t0 = time.time()
        components_before = count_connected_components(shape)
        timings['count_components_before'] = time.time() - t0
        log.info(f"[TIMING] count_components_before: {timings['count_components_before']:.3f}s ({components_before} components)")

        # --- Count free edges before ---
        log.info(f"Counting free edges before sewing...")
        sys.stdout.flush()
        t0 = time.time()
        free_edges_before = count_free_edges(shape)
        timings['count_free_edges_before'] = time.time() - t0
        log.info(f"[TIMING] count_free_edges_before: {timings['count_free_edges_before']:.3f}s ({free_edges_before} free edges)")

        # --- Initialize and add faces to sewer ---
        log.info(f"Adding {face_count_before} faces to sewer...")
        sys.stdout.flush()
        t0 = time.time()
        sewer = BRepBuilderAPI_Sewing(tolerance)
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            sewer.Add(face)
            explorer.Next()
        timings['add_faces'] = time.time() - t0
        log.info(f"[TIMING] add_faces: {timings['add_faces']:.3f}s")

        # --- Perform sewing (main operation) ---
        log.info(f"Performing sewing (tolerance={tolerance})...")
        sys.stdout.flush()
        t0 = time.time()
        sewer.Perform()
        timings['perform_sew'] = time.time() - t0
        log.info(f"[TIMING] perform_sew: {timings['perform_sew']:.3f}s")

        # --- Get sewing diagnostics ---
        t0 = time.time()
        diagnostics = {
            "free_edges": sewer.NbFreeEdges(),
            "multiple_edges": sewer.NbMultipleEdges(),
            "degenerated": sewer.NbDegeneratedShapes(),
            "deleted_faces": sewer.NbDeletedFaces(),
        }
        timings['get_diagnostics'] = time.time() - t0
        log.info(f"[TIMING] get_diagnostics: {timings['get_diagnostics']:.3f}s")
        log.info(f"Sewer diagnostics: free_edges={diagnostics['free_edges']}, "
              f"multiple_edges={diagnostics['multiple_edges']}, "
              f"degenerated={diagnostics['degenerated']}, "
              f"deleted_faces={diagnostics['deleted_faces']}")

        sewn_shape = sewer.SewedShape()

        # --- Count faces after ---
        log.info(f"Counting faces after sewing...")
        t0 = time.time()
        face_count_after = 0
        explorer = TopExp_Explorer(sewn_shape, TopAbs_FACE)
        while explorer.More():
            face_count_after += 1
            explorer.Next()
        timings['count_faces_after'] = time.time() - t0
        log.info(f"[TIMING] count_faces_after: {timings['count_faces_after']:.3f}s ({face_count_after} faces)")

        # --- Count connected components after ---
        log.info(f"Counting connected components after sewing...")
        sys.stdout.flush()
        t0 = time.time()
        components_after = count_connected_components(sewn_shape)
        timings['count_components_after'] = time.time() - t0
        log.info(f"[TIMING] count_components_after: {timings['count_components_after']:.3f}s ({components_after} components)")

        # --- Write output ---
        log.info(f"Writing output BREP...")
        sys.stdout.flush()
        t0 = time.time()
        breptools.Write(sewn_shape, str(output_path))
        timings['write_brep'] = time.time() - t0
        log.info(f"[TIMING] write_brep: {timings['write_brep']:.3f}s")

        # --- Calculate total ---
        total = sum(timings.values())
        timings['total'] = total

        # --- Summary ---
        log.info(f"\n[Summary]")
        log.info(f"  Faces: {face_count_before} -> {face_count_after}")
        log.info(f"  Components: {components_before} -> {components_after}")
        log.info(f"  Free edges: {free_edges_before} -> {diagnostics['free_edges']}")
        log.info(f"  Total time: {total:.3f}s")

        return {
            "success": True,
            "faces_before": face_count_before,
            "faces_after": face_count_after,
            "components_before": components_before,
            "components_after": components_after,
            "free_edges_before": free_edges_before,
            "diagnostics": diagnostics,
            "timings": timings,
        }

    except Exception as e:
        log.info(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timings": timings,
        }


if __name__ == '__main__':
    main()
