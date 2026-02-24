#!/usr/bin/env python
"""
Standalone subprocess script for CAD file loading with OS-level timeout.

Usage:
    python load_subprocess.py <input_file> <output_brep> [--result-file=path]

This script is invoked by subprocess.run() with a timeout, allowing the OS
to kill it if loading hangs on problematic CAD files.

Supports: STEP (.step, .stp), IGES (.iges, .igs), BREP (.brep)

Exit codes:
    0 - Success
    1 - Error (details in stderr or result file)
"""

import logging
import sys
import argparse
import json
import os
import math
import time

log = logging.getLogger("cadabra")


# =============================================================================
# Axis Transform Helper Functions (for AutoForm IGES files with _axis.igs)
# =============================================================================

def parse_axis_igs(filepath):
    """
    Parse an IGES axis file containing 3 lines that define a coordinate system.

    AutoForm exports coordinate systems as IGES files with 3 Line entities.
    Each line shares the same start point (origin) and the end points define X, Y, Z directions.

    Args:
        filepath: Path to the _axis.igs file

    Returns:
        dict with 'origin', 'x_dir', 'y_dir', 'z_dir' as (x, y, z) tuples,
        or None if parsing fails
    """
    try:
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods

        # Read IGES file using OpenCASCADE
        reader = IGESControl_Reader()
        status = reader.ReadFile(str(filepath))

        if status != IFSelect_RetDone:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        # Extract all edges (lines) from the shape
        lines_data = []
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)

        while explorer.More():
            edge = topods.Edge(explorer.Current())

            # Get the curve and its parameter range
            curve, first, last = BRep_Tool.Curve(edge)
            if curve is not None:
                # Get start and end points
                start_pnt = curve.Value(first)
                end_pnt = curve.Value(last)

                lines_data.append({
                    'start': (start_pnt.X(), start_pnt.Y(), start_pnt.Z()),
                    'end': (end_pnt.X(), end_pnt.Y(), end_pnt.Z())
                })

            explorer.Next()

        if len(lines_data) < 3:
            return None

        # All lines should share the same origin (start point)
        origin = lines_data[0]['start']
        x_end = lines_data[0]['end']
        y_end = lines_data[1]['end']
        z_end = lines_data[2]['end']

        # Compute direction vectors
        x_dir = (x_end[0] - origin[0], x_end[1] - origin[1], x_end[2] - origin[2])
        y_dir = (y_end[0] - origin[0], y_end[1] - origin[1], y_end[2] - origin[2])
        z_dir = (z_end[0] - origin[0], z_end[1] - origin[1], z_end[2] - origin[2])

        return {
            'origin': origin,
            'x_dir': x_dir,
            'y_dir': y_dir,
            'z_dir': z_dir
        }

    except Exception:
        return None


def build_axis_transform(axis_data):
    """
    Build an OpenCASCADE transformation matrix from parsed axis data.

    The transformation moves geometry from the coordinate system defined in the axis file
    to the world coordinate system (origin at 0,0,0 with standard X,Y,Z axes).

    Args:
        axis_data: dict from parse_axis_igs() with 'origin', 'x_dir', 'y_dir', 'z_dir'

    Returns:
        gp_Trsf transformation, or None if creation fails
    """
    try:
        from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir, gp_Trsf

        origin = axis_data['origin']
        x_dir = axis_data['x_dir']
        z_dir = axis_data['z_dir']

        # Normalize direction vectors
        def normalize(v):
            length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            if length < 1e-10:
                return None
            return (v[0]/length, v[1]/length, v[2]/length)

        x_norm = normalize(x_dir)
        z_norm = normalize(z_dir)

        if x_norm is None or z_norm is None:
            return None

        # Create source coordinate system (from the axis file)
        source_origin = gp_Pnt(origin[0], origin[1], origin[2])
        source_z = gp_Dir(z_norm[0], z_norm[1], z_norm[2])
        source_x = gp_Dir(x_norm[0], x_norm[1], x_norm[2])
        source_ax3 = gp_Ax3(source_origin, source_z, source_x)

        # Target is world coordinates (origin at 0,0,0 with standard axes)
        target_ax3 = gp_Ax3()  # Default: origin=(0,0,0), Z=(0,0,1), X=(1,0,0)

        # Build transformation: source -> target
        trsf = gp_Trsf()
        trsf.SetDisplacement(source_ax3, target_ax3)

        return trsf

    except Exception:
        return None


def apply_transform_to_shape(shape, trsf):
    """
    Apply a transformation to an OpenCASCADE shape.

    Args:
        shape: TopoDS_Shape to transform
        trsf: gp_Trsf transformation

    Returns:
        Transformed TopoDS_Shape, or original shape if transformation fails
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

        transformer = BRepBuilderAPI_Transform(shape, trsf, True)  # True = copy geometry
        if transformer.IsDone():
            return transformer.Shape()
        else:
            return shape
    except Exception:
        return shape


def main():
    parser = argparse.ArgumentParser(description='Load CAD file and output BREP')
    parser.add_argument('input_file', help='Input CAD file path (STEP, IGES, or BREP)')
    parser.add_argument('output_brep', help='Output BREP file path')
    parser.add_argument('--result-file', help='Path to write JSON result (optional)')
    parser.add_argument('--log-file', help='Path to write verbose log including OCC output (optional)')
    args = parser.parse_args()

    # If log file specified, redirect all stdout/stderr to it to capture OCC messages
    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if args.log_file:
        log_handle = open(args.log_file, 'w')
        sys.stdout = log_handle
        sys.stderr = log_handle
        log.info(f"[CADabra Load Subprocess]")
        log.info(f"Input: {args.input_file}")
        log.info(f"Output: {args.output_brep}")
        log.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("=" * 60)
        sys.stdout.flush()

    try:
        result = load_cad_file(args.input_file, args.output_brep)

        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f)

        # Write final status to log
        if log_handle:
            log.info("=" * 60)
            if result['success']:
                axis_str = " (axis applied)" if result.get('axis_transform_applied') else ""
                log.info(f"SUCCESS: Loaded {result.get('format', 'unknown')} with {result.get('num_faces', '?')} faces{axis_str}")
                if 'timings' in result:
                    log.info("\nTimings:")
                    for step, duration in result['timings'].items():
                        log.info(f"  {step}: {duration:.3f}s")
            else:
                log.info(f"FAILED: {result['error']}")
            log.info(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.flush()

        # Restore stdout/stderr for exit message
        if log_handle:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()

        if result['success']:
            axis_str = " (axis applied)" if result.get('axis_transform_applied') else ""
            log.info(f"OK: Loaded {result.get('format', 'unknown')} with {result.get('num_faces', '?')} faces{axis_str}")
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


def load_cad_file(input_path, output_path):
    """Load CAD file and write as BREP with detailed timing.

    All print statements go to stdout which may be redirected to a log file.
    This captures both our timing info AND any OCC internal messages.
    """
    timings = {}

    log.info(f"Reading CAD file: {os.path.basename(input_path)}")
    sys.stdout.flush()

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.BRepTools import breptools
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

        # Determine format from extension
        ext = os.path.splitext(input_path)[1].lower()
        occ_shape = None
        metadata = {}

        # --- Read CAD file ---
        # OCC may print warnings/info here which will be captured in log
        t0 = time.time()
        if ext in ['.step', '.stp']:
            log.info(f"Reading STEP file...")
            sys.stdout.flush()
            reader = STEPControl_Reader()
            status = reader.ReadFile(input_path)
            if status != IFSelect_RetDone:
                return {"success": False, "error": f"Failed to read STEP file: status={status}"}
            log.info(f"Transferring roots...")
            sys.stdout.flush()
            reader.TransferRoots()
            occ_shape = reader.OneShape()
            metadata["format"] = "step"

        elif ext in ['.iges', '.igs']:
            log.info(f"Reading IGES file...")
            sys.stdout.flush()
            reader = IGESControl_Reader()
            status = reader.ReadFile(input_path)
            if status != IFSelect_RetDone:
                return {"success": False, "error": f"Failed to read IGES file: status={status}"}
            log.info(f"Transferring roots...")
            sys.stdout.flush()
            reader.TransferRoots()
            occ_shape = reader.OneShape()
            metadata["format"] = "iges"

            # Try to extract IGES metadata
            try:
                from OCC.Core.IGESData import IGESData_GlobalSection
                iges_model = reader.IGESModel()
                if iges_model:
                    gs = iges_model.GlobalSection()
                    metadata["resolution"] = gs.Resolution()
                    metadata["unit_flag"] = gs.UnitFlag()
                    # Map unit flag to name
                    unit_map = {1: "inches", 2: "mm", 3: "unknown", 4: "ft",
                               5: "miles", 6: "m", 7: "km", 8: "mils",
                               9: "um", 10: "cm", 11: "microinches"}
                    metadata["units"] = unit_map.get(gs.UnitFlag(), "unknown")
                    metadata["author"] = gs.AuthorName().ToCString() if gs.AuthorName() else ""
                    metadata["company"] = gs.CompanyName().ToCString() if gs.CompanyName() else ""
                    log.info(f"IGES metadata: units={metadata['units']}, resolution={metadata['resolution']}")
            except Exception:
                pass  # Metadata extraction is optional

        elif ext == '.brep':
            log.info(f"Reading BREP file...")
            sys.stdout.flush()
            builder = BRep_Builder()
            shape = TopoDS_Shape()
            breptools.Read(shape, input_path, builder)
            occ_shape = shape
            metadata["format"] = "brep"

        else:
            return {"success": False, "error": f"Unsupported format: {ext}"}

        timings['read_cad'] = time.time() - t0
        log.info(f"[TIMING] read_cad: {timings['read_cad']:.3f}s")
        sys.stdout.flush()

        if occ_shape is None or occ_shape.IsNull():
            return {"success": False, "error": "Failed to load shape (null result)"}

        # --- Auto-apply axis transform if matching _axis.igs file exists ---
        axis_applied = False
        timings['parse_axis'] = 0.0
        timings['build_transform'] = 0.0
        timings['apply_transform'] = 0.0

        if ext in ['.iges', '.igs']:
            # Look for axis file: foo.igs -> foo_axis.igs
            base_path = os.path.splitext(input_path)[0]
            axis_path = base_path + '_axis.igs'
            if os.path.exists(axis_path):
                log.info(f"\n[Parsing axis file: {os.path.basename(axis_path)}]")
                sys.stdout.flush()

                t0 = time.time()
                axis_data = parse_axis_igs(axis_path)
                timings['parse_axis'] = time.time() - t0
                log.info(f"[TIMING] parse_axis: {timings['parse_axis']:.3f}s")

                if axis_data:
                    log.info(f"Axis origin: {axis_data['origin']}")
                    log.info(f"Axis X-dir: {axis_data['x_dir']}")
                    log.info(f"Axis Z-dir: {axis_data['z_dir']}")

                    t0 = time.time()
                    trsf = build_axis_transform(axis_data)
                    timings['build_transform'] = time.time() - t0
                    log.info(f"[TIMING] build_transform: {timings['build_transform']:.3f}s")

                    if trsf:
                        log.info(f"\n[Applying transform to shape...]")
                        sys.stdout.flush()
                        t0 = time.time()
                        occ_shape = apply_transform_to_shape(occ_shape, trsf)
                        timings['apply_transform'] = time.time() - t0
                        log.info(f"[TIMING] apply_transform: {timings['apply_transform']:.3f}s")
                        axis_applied = True
            else:
                log.info(f"No axis file found (looked for: {os.path.basename(axis_path)})")

        # --- Count faces ---
        log.info(f"\n[Counting faces...]")
        t0 = time.time()
        num_faces = 0
        explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
        while explorer.More():
            num_faces += 1
            explorer.Next()
        timings['count_faces'] = time.time() - t0
        log.info(f"[TIMING] count_faces: {timings['count_faces']:.3f}s ({num_faces} faces)")

        # --- Write output BREP ---
        log.info(f"\n[Writing BREP output...]")
        sys.stdout.flush()
        t0 = time.time()
        breptools.Write(occ_shape, str(output_path))
        timings['write_brep'] = time.time() - t0
        log.info(f"[TIMING] write_brep: {timings['write_brep']:.3f}s")

        # --- Calculate total ---
        total = sum(timings.values())
        timings['total'] = total

        return {
            "success": True,
            "format": metadata.get("format", ext),
            "num_faces": num_faces,
            "metadata": metadata,
            "axis_transform_applied": axis_applied,
            "timings": timings,
        }

    except Exception as e:
        log.info(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "timings": timings}


if __name__ == '__main__':
    main()
