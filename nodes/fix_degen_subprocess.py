#!/usr/bin/env python
"""
Standalone subprocess script for fixing degenerate CAD faces with OS-level timeout.

Usage:
    python fix_degen_subprocess.py <input_brep> <output_brep> [options]

Options:
    --iterations=N        Smoothness iterations (default: 5)
    --max-degree=N        Max BSpline degree (default: 8)
    --result-file=path    Path to write JSON result
    --log-file=path       Path to write detailed log

This script is invoked by subprocess.run() with a timeout, allowing the OS
to kill it if processing hangs on problematic geometry.

Output is a compound of faces (not sewn). Use CADSewFaces node afterwards
to sew the faces together.

Exit codes:
    0 - Success
    1 - Error (details in stderr or result file)
"""

import sys
import argparse
import json
import os
from datetime import datetime


def fix_degenerate_faces(input_path, output_path, iterations, max_degree, log_file=None):
    """Fix degenerate faces in BREP file. Output is a compound of faces (not sewn)."""
    timings = {}
    import time

    def log(msg):
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    try:
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, topods
        from OCC.Core.BRep import BRep_Builder, BRep_Tool
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape

        # Read input BREP
        t0 = time.time()
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        breptools.Read(shape, str(input_path), builder)
        timings['read_brep'] = time.time() - t0
        log(f"[TIMING] read_brep: {timings['read_brep']:.3f}s")

        # Find all faces and categorize
        t0 = time.time()
        normal_faces = []
        degen_faces = []

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())

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

        timings['categorize_faces'] = time.time() - t0
        total_faces = len(normal_faces) + len(degen_faces)
        log(f"[TIMING] categorize_faces: {timings['categorize_faces']:.3f}s ({total_faces} faces, {len(degen_faces)} degenerate)")

        # If no degenerate faces, just copy input to output
        if len(degen_faces) == 0:
            t0 = time.time()
            breptools.Write(shape, output_path)
            timings['write_brep'] = time.time() - t0
            log(f"[TIMING] write_brep: {timings['write_brep']:.3f}s")

            return {
                "success": True,
                "faces_before": total_faces,
                "faces_after": total_faces,
                "degen_before": 0,
                "degen_after": 0,
                "fixed": 0,
                "failed": 0,
                "timings": timings,
            }

        # Replace degenerate faces
        t0 = time.time()
        fixed_count = 0
        failed_count = 0
        replacement_faces = []

        for degen_face in degen_faces:
            new_face, status = _replace_degenerate_face(degen_face, iterations, max_degree)
            if new_face is not None:
                replacement_faces.append(new_face)
                fixed_count += 1
            else:
                # Keep original face if replacement fails
                replacement_faces.append(degen_face)
                failed_count += 1
                log(f"[WARNING] Failed to fix face: {status}")

        timings['fix_faces'] = time.time() - t0
        log(f"[TIMING] fix_faces: {timings['fix_faces']:.3f}s ({fixed_count} fixed, {failed_count} failed)")

        # Build compound of all faces (no sewing - let user do that with CADSewFaces)
        t0 = time.time()
        all_result_faces = normal_faces + replacement_faces

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for face in all_result_faces:
            builder.Add(compound, face)

        timings['build_compound'] = time.time() - t0
        log(f"[TIMING] build_compound: {timings['build_compound']:.3f}s")

        # Count degenerate edges in result
        t0 = time.time()
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(compound, TopAbs_EDGE, edge_map)
        degen_after = 0
        for i in range(1, edge_map.Size() + 1):
            edge = topods.Edge(edge_map.FindKey(i))
            if BRep_Tool.Degenerated(edge):
                degen_after += 1

        faces_after = len(all_result_faces)

        timings['count_after'] = time.time() - t0
        log(f"[TIMING] count_after: {timings['count_after']:.3f}s")

        # Write output
        t0 = time.time()
        breptools.Write(compound, output_path)
        timings['write_brep'] = time.time() - t0
        log(f"[TIMING] write_brep: {timings['write_brep']:.3f}s")

        total_time = sum(timings.values())
        timings['total'] = total_time

        log(f"\n[Summary]")
        log(f"  Faces: {total_faces} -> {faces_after}")
        log(f"  Degenerate faces: {len(degen_faces)} -> {degen_after} (edges with degen flag)")
        log(f"  Fixed: {fixed_count}, Failed: {failed_count}")
        log(f"  Total time: {total_time:.3f}s")
        log(f"  Note: Output is unsewn compound. Use CADSewFaces to sew.")

        return {
            "success": True,
            "faces_before": total_faces,
            "faces_after": faces_after,
            "degen_before": len(degen_faces),
            "degen_after": degen_after,
            "fixed": fixed_count,
            "failed": failed_count,
            "timings": timings,
        }

    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        log(f"[ERROR] {error_msg}")
        return {"success": False, "error": str(e)}


def _replace_degenerate_face(face, iterations, max_degree):
    """Replace a single degenerate face with a filled surface."""
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCC.Core.GeomAbs import GeomAbs_C0
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods

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


def main():
    parser = argparse.ArgumentParser(description='Fix degenerate faces in BREP file')
    parser.add_argument('input_brep', help='Input BREP file path')
    parser.add_argument('output_brep', help='Output BREP file path')
    parser.add_argument('--iterations', type=int, default=5, help='Smoothness iterations')
    parser.add_argument('--max-degree', type=int, default=8, help='Max BSpline degree')
    parser.add_argument('--result-file', help='Path to write JSON result (optional)')
    parser.add_argument('--log-file', help='Path to write detailed log (optional)')
    args = parser.parse_args()

    # Initialize log file
    if args.log_file:
        with open(args.log_file, 'w') as f:
            f.write(f"[CADabra Fix Degenerate Subprocess]\n")
            f.write(f"Input: {args.input_brep}\n")
            f.write(f"Output: {args.output_brep}\n")
            f.write(f"Iterations: {args.iterations}, MaxDegree: {args.max_degree}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    result = fix_degenerate_faces(
        args.input_brep,
        args.output_brep,
        args.iterations,
        args.max_degree,
        args.log_file
    )

    if args.result_file:
        with open(args.result_file, 'w') as f:
            json.dump(result, f)

    if args.log_file:
        with open(args.log_file, 'a') as f:
            f.write(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if result['success']:
        print(f"OK: {result.get('fixed', 0)} fixed, {result.get('failed', 0)} failed")
        sys.exit(0)
    else:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
