#!/usr/bin/env python
"""
Standalone subprocess script for CAD shape healing with OS-level timeout.

Usage:
    python heal_subprocess.py <input_brep> <output_brep> <precision> <max_tolerance> [options]

Options:
    --result-file=PATH      Path to write JSON result
    --log-file=PATH         Path to write verbose log
    --fix-small-faces       Enable small face removal (default: true)
    --small-face-precision  Precision for small face detection (default: 0.1)
    --fix-small-edges       Enable small edge removal (default: true)
    --fix-wire-gaps         Enable wire gap fixing (default: true)
    --merge-colinear        Enable colinear edge merging (default: false)

This script is invoked by subprocess.run() with a timeout, allowing the OS
to kill it if healing hangs on degenerate geometry.

Exit codes:
    0 - Success
    1 - Error (details in stderr or result file)
"""

import sys
import argparse
import json
import os
import time


def count_faces(shape):
    """Count faces in a shape."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def count_edges(shape):
    """Count edges in a shape."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE

    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def get_edge_vertices(edge):
    """Get start and end vertices of an edge."""
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool

    v_first = topexp.FirstVertex(edge)
    v_last = topexp.LastVertex(edge)
    return v_first, v_last


def vertices_same(v1, v2, tolerance=1e-6):
    """Check if two vertices are at the same location."""
    from OCC.Core.BRep import BRep_Tool

    p1 = BRep_Tool.Pnt(v1)
    p2 = BRep_Tool.Pnt(v2)
    return p1.Distance(p2) < tolerance


def check_g2_continuity(edge1, edge2, curvature_tolerance=0.01, angular_tolerance=0.01):
    """
    Check if two edges have G2 continuity at their junction.

    Returns (is_g2, junction_info) where junction_info contains:
    - param1: 'start' or 'end' of edge1 at junction
    - param2: 'start' or 'end' of edge2 at junction
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepLProp import BRepLProp_CLProps
    from OCC.Core.gp import gp_Vec
    import math

    # Get vertices
    v1_start, v1_end = get_edge_vertices(edge1)
    v2_start, v2_end = get_edge_vertices(edge2)

    # Find junction point
    junction_info = None
    if vertices_same(v1_end, v2_start):
        junction_info = {'param1': 'end', 'param2': 'start'}
    elif vertices_same(v1_start, v2_end):
        junction_info = {'param1': 'start', 'param2': 'end'}
    elif vertices_same(v1_end, v2_end):
        junction_info = {'param1': 'end', 'param2': 'end'}
    elif vertices_same(v1_start, v2_start):
        junction_info = {'param1': 'start', 'param2': 'start'}

    if junction_info is None:
        return False, None

    # Get curve adaptors
    adaptor1 = BRepAdaptor_Curve(edge1)
    adaptor2 = BRepAdaptor_Curve(edge2)

    # Get parameters at junction
    u1 = adaptor1.LastParameter() if junction_info['param1'] == 'end' else adaptor1.FirstParameter()
    u2 = adaptor2.FirstParameter() if junction_info['param2'] == 'start' else adaptor2.LastParameter()

    # Get local properties (2nd order for curvature)
    props1 = BRepLProp_CLProps(adaptor1, 2, 1e-6)
    props2 = BRepLProp_CLProps(adaptor2, 2, 1e-6)

    props1.SetParameter(u1)
    props2.SetParameter(u2)

    # Check G1 (tangent continuity)
    if not props1.IsTangentDefined() or not props2.IsTangentDefined():
        return False, junction_info

    tangent1 = gp_Vec()
    tangent2 = gp_Vec()
    props1.Tangent(tangent1)
    props2.Tangent(tangent2)

    # Tangents should be parallel (or anti-parallel for same direction)
    angle = tangent1.Angle(tangent2)
    if angle > angular_tolerance and abs(angle - math.pi) > angular_tolerance:
        return False, junction_info

    # Check G2 (curvature continuity)
    curv1 = props1.Curvature()
    curv2 = props2.Curvature()

    # For curvature comparison, use relative tolerance for large curvatures
    max_curv = max(abs(curv1), abs(curv2), 1e-10)
    relative_diff = abs(curv1 - curv2) / max_curv

    if relative_diff > curvature_tolerance and abs(curv1 - curv2) > curvature_tolerance:
        return False, junction_info

    return True, junction_info


def get_ordered_wire_edges(wire):
    """Get edges from wire in order."""
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopoDS import topods

    edges = []
    explorer = BRepTools_WireExplorer(wire)
    while explorer.More():
        edge = explorer.Current()
        edges.append(edge)
        explorer.Next()
    return edges


def find_g2_edge_groups(edges, curvature_tolerance):
    """
    Find groups of adjacent edges that have G2 continuity.

    Returns list of lists, where each inner list is a group of G2-continuous edges.
    """
    if not edges:
        return []

    groups = []
    current_group = [edges[0]]

    for i in range(1, len(edges)):
        prev_edge = edges[i - 1]
        curr_edge = edges[i]

        is_g2, _ = check_g2_continuity(prev_edge, curr_edge, curvature_tolerance)

        if is_g2:
            current_group.append(curr_edge)
        else:
            groups.append(current_group)
            current_group = [curr_edge]

    groups.append(current_group)

    # Check if first and last groups can be merged (for closed wires)
    if len(groups) > 1 and len(edges) > 2:
        is_g2, _ = check_g2_continuity(edges[-1], edges[0], curvature_tolerance)
        if is_g2:
            # Merge last group into first
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    return groups


def merge_edges_to_bspline(edges):
    """
    Merge multiple edges into a single B-spline edge.
    """
    from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve, geomconvert
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.GeomAbs import GeomAbs_C2

    if len(edges) == 1:
        return edges[0]

    try:
        # Convert first edge to B-spline
        adaptor = BRepAdaptor_Curve(edges[0])
        first_bspline = geomconvert.CurveToBSplineCurve(adaptor.Curve().Curve(), GeomAbs_C2)

        converter = GeomConvert_CompCurveToBSplineCurve(first_bspline)

        # Add remaining edges
        for edge in edges[1:]:
            adaptor = BRepAdaptor_Curve(edge)
            bspline = geomconvert.CurveToBSplineCurve(adaptor.Curve().Curve(), GeomAbs_C2)
            if not converter.Add(bspline, 1e-6):
                print(f"[WARNING] Failed to add edge to composite curve")
                return None

        # Get merged B-spline
        merged_curve = converter.BSplineCurve()

        # Create edge from merged curve
        builder = BRepBuilderAPI_MakeEdge(merged_curve)
        if builder.IsDone():
            return builder.Edge()
        else:
            print(f"[WARNING] Failed to create edge from merged B-spline")
            return None

    except Exception as e:
        print(f"[WARNING] merge_edges_to_bspline failed: {e}")
        return None


def merge_g2_continuous_edges(shape, curvature_tolerance=0.01, preserve_quads=True):
    """
    Merge adjacent edges that have G2 continuity.

    Args:
        shape: TopoDS_Shape to process
        curvature_tolerance: max difference in curvature to consider G2
        preserve_quads: don't touch faces with exactly 4 edges

    Returns:
        Modified shape with merged edges
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepTools import BRepTools_ReShape

    reshaper = BRepTools_ReShape()
    total_merged = 0
    faces_processed = 0
    faces_skipped = 0

    # Process each face
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())

        # Count edges on this face
        edge_count = 0
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            edge_count += len(get_ordered_wire_edges(wire))
            wire_explorer.Next()

        # Skip quad faces if requested
        if preserve_quads and edge_count == 4:
            faces_skipped += 1
            face_explorer.Next()
            continue

        faces_processed += 1

        # Process each wire on this face
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            edges = get_ordered_wire_edges(wire)

            if len(edges) < 2:
                wire_explorer.Next()
                continue

            # Find G2-continuous groups
            groups = find_g2_edge_groups(edges, curvature_tolerance)

            # Merge groups with more than one edge
            for group in groups:
                if len(group) > 1:
                    merged_edge = merge_edges_to_bspline(group)
                    if merged_edge is not None:
                        # Replace first edge with merged edge
                        reshaper.Replace(group[0], merged_edge)
                        # Remove remaining edges
                        for edge in group[1:]:
                            reshaper.Remove(edge)
                        total_merged += len(group) - 1
                        print(f"  Merged {len(group)} edges into 1")

            wire_explorer.Next()

        face_explorer.Next()

    print(f"[G2 Merge] Processed {faces_processed} faces, skipped {faces_skipped} quad faces, merged {total_merged} edges")

    # Apply all replacements
    result = reshaper.Apply(shape)
    return result


def main():
    parser = argparse.ArgumentParser(description='Heal CAD shape from BREP file')
    parser.add_argument('input_brep', help='Input BREP file path')
    parser.add_argument('output_brep', help='Output BREP file path')
    parser.add_argument('precision', type=float, help='Healing precision')
    parser.add_argument('max_tolerance', type=float, help='Maximum tolerance')
    parser.add_argument('--result-file', help='Path to write JSON result (optional)')
    parser.add_argument('--log-file', help='Path to write verbose log (optional)')
    parser.add_argument('--fix-small-faces', type=str, default='true', help='Fix small faces (true/false)')
    parser.add_argument('--small-face-precision', type=float, default=0.1, help='Small face precision')
    parser.add_argument('--fix-small-edges', type=str, default='true', help='Fix small edges (true/false)')
    parser.add_argument('--fix-wire-gaps', type=str, default='true', help='Fix wire gaps (true/false)')
    parser.add_argument('--merge-colinear', type=str, default='false', help='Merge colinear edges (true/false)')
    parser.add_argument('--unify-faces', type=str, default='false', help='Also unify faces on same surface (true/false)')
    parser.add_argument('--angular-tolerance', type=float, default=0.01, help='Angular tolerance for curve matching (radians)')
    parser.add_argument('--linear-tolerance', type=float, default=0.001, help='Linear tolerance for curve matching')
    parser.add_argument('--merge-g2-edges', type=str, default='false', help='Merge edges with G2 curvature continuity (true/false)')
    parser.add_argument('--g2-tolerance', type=float, default=0.01, help='Curvature tolerance for G2 detection')
    parser.add_argument('--preserve-quad-faces', type=str, default='true', help='Preserve faces with exactly 4 edges (true/false)')
    args = parser.parse_args()

    # Parse boolean args
    fix_small_faces = args.fix_small_faces.lower() == 'true'
    fix_small_edges = args.fix_small_edges.lower() == 'true'
    fix_wire_gaps = args.fix_wire_gaps.lower() == 'true'
    merge_colinear = args.merge_colinear.lower() == 'true'
    unify_faces = args.unify_faces.lower() == 'true'
    merge_g2_edges = args.merge_g2_edges.lower() == 'true'
    preserve_quad_faces = args.preserve_quad_faces.lower() == 'true'

    # If log file specified, redirect stdout/stderr to capture all output
    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if args.log_file:
        log_handle = open(args.log_file, 'w')
        sys.stdout = log_handle
        sys.stderr = log_handle
        print(f"[CADabra Heal Subprocess]")
        print(f"Input: {args.input_brep}")
        print(f"Output: {args.output_brep}")
        print(f"Precision: {args.precision}")
        print(f"Max Tolerance: {args.max_tolerance}")
        print(f"Fix small faces: {fix_small_faces} (precision: {args.small_face_precision})")
        print(f"Fix small edges: {fix_small_edges}")
        print(f"Fix wire gaps: {fix_wire_gaps}")
        print(f"Merge colinear: {merge_colinear}")
        print(f"Unify faces: {unify_faces}")
        print(f"Angular tolerance: {args.angular_tolerance}")
        print(f"Linear tolerance: {args.linear_tolerance}")
        print(f"Merge G2 edges: {merge_g2_edges}")
        print(f"G2 tolerance: {args.g2_tolerance}")
        print(f"Preserve quad faces: {preserve_quad_faces}")
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        sys.stdout.flush()

    try:
        result = heal_brep(
            args.input_brep,
            args.output_brep,
            args.precision,
            args.max_tolerance,
            fix_small_faces,
            args.small_face_precision,
            fix_small_edges,
            fix_wire_gaps,
            merge_colinear,
            unify_faces,
            args.angular_tolerance,
            args.linear_tolerance,
            merge_g2_edges,
            args.g2_tolerance,
            preserve_quad_faces
        )

        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f)

        # Write final status to log
        if log_handle:
            print("=" * 60)
            if result['success']:
                print(f"SUCCESS: {result['faces_before']}->{result['faces_after']} faces, "
                      f"edges: {result.get('edges_before', '?')}->{result.get('edges_after', '?')}")
                if 'timings' in result:
                    print("\nTimings:")
                    for step, duration in result['timings'].items():
                        print(f"  {step}: {duration:.3f}s")
            else:
                print(f"FAILED: {result['error']}")
            print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.flush()

        # Restore stdout/stderr
        if log_handle:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()

        if result['success']:
            # Print summary to stdout
            print(f"OK: faces={result['faces_before']}->{result['faces_after']}, "
                  f"edges={result.get('edges_before', '?')}->{result.get('edges_after', '?')}")
            sys.exit(0)
        else:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        if log_handle:
            print(f"\nEXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()
        raise


def heal_brep(input_path, output_path, precision, max_tolerance,
              fix_small_faces, small_face_precision, fix_small_edges,
              fix_wire_gaps, merge_colinear, unify_faces=False,
              angular_tolerance=0.01, linear_tolerance=0.001,
              merge_g2_edges=False, g2_tolerance=0.01, preserve_quad_faces=True):
    """Perform healing on BREP file with detailed timing."""
    timings = {}

    print(f"\n[Healing CAD file: {os.path.basename(input_path)}]")
    sys.stdout.flush()

    try:
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRepTools import breptools
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wireframe, ShapeFix_FixSmallFace

        # --- Read input BREP ---
        print(f"Reading BREP file...")
        sys.stdout.flush()
        t0 = time.time()
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        breptools.Read(shape, str(input_path), builder)
        timings['read_brep'] = time.time() - t0
        print(f"[TIMING] read_brep: {timings['read_brep']:.3f}s")

        # --- Count entities before ---
        print(f"Counting entities before healing...")
        t0 = time.time()
        faces_before = count_faces(shape)
        edges_before = count_edges(shape)
        timings['count_before'] = time.time() - t0
        print(f"[TIMING] count_before: {timings['count_before']:.3f}s ({faces_before} faces, {edges_before} edges)")

        result_shape = shape
        operations_applied = []

        # --- 1. Fix small faces ---
        if fix_small_faces:
            print(f"Fixing small faces (precision={small_face_precision})...")
            sys.stdout.flush()
            t0 = time.time()
            try:
                fix_small = ShapeFix_FixSmallFace()
                fix_small.Init(result_shape)
                fix_small.SetPrecision(small_face_precision)
                fix_small.Perform()
                result_shape = fix_small.FixShape()
                timings['fix_small_faces'] = time.time() - t0
                faces_after_small = count_faces(result_shape)
                removed = faces_before - faces_after_small
                print(f"[TIMING] fix_small_faces: {timings['fix_small_faces']:.3f}s (removed {removed} faces)")
                operations_applied.append(f"small_faces: removed {removed}")
            except Exception as e:
                timings['fix_small_faces'] = time.time() - t0
                print(f"[WARNING] fix_small_faces error: {e}")

        # --- 2. Fix wireframe (small edges, gaps) ---
        if fix_small_edges or fix_wire_gaps:
            print(f"Fixing wireframe (small_edges={fix_small_edges}, wire_gaps={fix_wire_gaps})...")
            sys.stdout.flush()
            t0 = time.time()
            try:
                wireframe_fix = ShapeFix_Wireframe(result_shape)
                wireframe_fix.SetPrecision(precision)

                if fix_small_edges:
                    wireframe_fix.SetModeDropSmallEdges(True)
                    wireframe_fix.FixSmallEdges()

                if fix_wire_gaps:
                    wireframe_fix.FixWireGaps()

                result_shape = wireframe_fix.Shape()
                timings['fix_wireframe'] = time.time() - t0
                print(f"[TIMING] fix_wireframe: {timings['fix_wireframe']:.3f}s")
                if fix_small_edges:
                    operations_applied.append("small_edges")
                if fix_wire_gaps:
                    operations_applied.append("wire_gaps")
            except Exception as e:
                timings['fix_wireframe'] = time.time() - t0
                print(f"[WARNING] fix_wireframe error: {e}")

        # --- 3. General shape fix ---
        print(f"Applying general shape fix (precision={precision}, max_tol={max_tolerance})...")
        sys.stdout.flush()
        t0 = time.time()
        try:
            shape_fix = ShapeFix_Shape(result_shape)
            shape_fix.SetPrecision(precision)
            shape_fix.SetMaxTolerance(max_tolerance)
            shape_fix.Perform()
            result_shape = shape_fix.Shape()
            timings['fix_shape'] = time.time() - t0
            print(f"[TIMING] fix_shape: {timings['fix_shape']:.3f}s")
            operations_applied.append("general_fix")
        except Exception as e:
            timings['fix_shape'] = time.time() - t0
            print(f"[WARNING] fix_shape error: {e}")

        # --- 4. Merge colinear edges (optional) ---
        if merge_colinear:
            print(f"Merging colinear edges (unify_faces={unify_faces}, "
                  f"angular_tol={angular_tolerance}, linear_tol={linear_tolerance})...")
            sys.stdout.flush()
            t0 = time.time()
            try:
                from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

                edges_mid = count_edges(result_shape)
                faces_mid = count_faces(result_shape)

                # Args: shape, UnifyFaces, UnifyEdges=True, ConcatBSplines=True
                # UnifyFaces=True can be more aggressive in simplifying geometry
                unifier = ShapeUpgrade_UnifySameDomain(result_shape, unify_faces, True, True)

                # Set tolerances for curve matching - higher values = more aggressive merging
                unifier.SetAngularTolerance(angular_tolerance)
                unifier.SetLinearTolerance(linear_tolerance)

                unifier.Build()
                result_shape = unifier.Shape()
                timings['merge_colinear'] = time.time() - t0

                edges_merged = count_edges(result_shape)
                faces_merged = count_faces(result_shape)
                edge_reduction = edges_mid - edges_merged
                face_reduction = faces_mid - faces_merged

                print(f"[TIMING] merge_colinear: {timings['merge_colinear']:.3f}s "
                      f"(edges: {edges_mid}->{edges_merged}, merged {edge_reduction})")
                if unify_faces and face_reduction > 0:
                    print(f"[TIMING]   faces unified: {faces_mid}->{faces_merged}, merged {face_reduction}")
                operations_applied.append(f"merge_colinear: edges-{edge_reduction}, faces-{face_reduction}")
            except ImportError:
                timings['merge_colinear'] = time.time() - t0
                print(f"[WARNING] ShapeUpgrade_UnifySameDomain not available")
            except Exception as e:
                timings['merge_colinear'] = time.time() - t0
                print(f"[WARNING] merge_colinear error: {e}")

        # --- 5. Merge G2-continuous edges (optional) ---
        if merge_g2_edges:
            print(f"Merging G2-continuous edges (g2_tol={g2_tolerance}, preserve_quads={preserve_quad_faces})...")
            sys.stdout.flush()
            t0 = time.time()
            try:
                edges_before_g2 = count_edges(result_shape)
                result_shape = merge_g2_continuous_edges(
                    result_shape, g2_tolerance, preserve_quad_faces
                )
                edges_after_g2 = count_edges(result_shape)
                timings['merge_g2'] = time.time() - t0
                edge_reduction = edges_before_g2 - edges_after_g2
                print(f"[TIMING] merge_g2: {timings['merge_g2']:.3f}s "
                      f"(edges: {edges_before_g2}->{edges_after_g2}, merged {edge_reduction})")
                operations_applied.append(f"merge_g2: edges-{edge_reduction}")
            except Exception as e:
                timings['merge_g2'] = time.time() - t0
                print(f"[WARNING] merge_g2 error: {e}")
                import traceback
                traceback.print_exc()

        # --- Count entities after ---
        print(f"Counting entities after healing...")
        t0 = time.time()
        faces_after = count_faces(result_shape)
        edges_after = count_edges(result_shape)
        timings['count_after'] = time.time() - t0
        print(f"[TIMING] count_after: {timings['count_after']:.3f}s ({faces_after} faces, {edges_after} edges)")

        # --- Write output ---
        print(f"Writing output BREP...")
        sys.stdout.flush()
        t0 = time.time()
        breptools.Write(result_shape, str(output_path))
        timings['write_brep'] = time.time() - t0
        print(f"[TIMING] write_brep: {timings['write_brep']:.3f}s")

        # --- Calculate total ---
        total = sum(timings.values())
        timings['total'] = total

        # --- Summary ---
        print(f"\n[Summary]")
        print(f"  Faces: {faces_before} -> {faces_after} (removed: {faces_before - faces_after})")
        print(f"  Edges: {edges_before} -> {edges_after} (removed: {edges_before - edges_after})")
        print(f"  Operations: {', '.join(operations_applied)}")
        print(f"  Total time: {total:.3f}s")

        return {
            "success": True,
            "faces_before": faces_before,
            "faces_after": faces_after,
            "edges_before": edges_before,
            "edges_after": edges_after,
            "faces_removed": faces_before - faces_after,
            "edges_removed": edges_before - edges_after,
            "operations": operations_applied,
            "timings": timings,
        }

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timings": timings,
        }


if __name__ == '__main__':
    main()
