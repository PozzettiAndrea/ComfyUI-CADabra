#!/usr/bin/env python
"""
Standalone subprocess script for CAD meshing with OS-level timeout.

Usage:
    python mesh_subprocess.py <input_brep> <output_json> <linear_deflection> <angular_deflection> [options]

Options:
    --merge-vertices      Merge duplicate vertices (default: true)
    --extract-face-ids    Include face IDs in output (default: true)
    --result-file=path    Path to write JSON result

This script is invoked by subprocess.run() with a timeout, allowing the OS
to kill it if meshing hangs on problematic geometry.

Exit codes:
    0 - Success
    1 - Error (details in stderr or result file)
"""

import sys
import argparse
import json
import numpy as np
from collections import defaultdict


def filter_small_components(vertices, faces, cad_face_ids, min_ratio):
    """
    Remove small disconnected components from mesh.

    Uses scipy's connected_components on face adjacency graph (same as trimesh)
    to identify components, then keeps only those above the size threshold.

    Args:
        vertices: List of [x, y, z] vertex coordinates
        faces: List of [v0, v1, v2] face indices
        cad_face_ids: List of CAD face IDs per triangle (or None)
        min_ratio: Minimum ratio of total faces to keep a component

    Returns:
        tuple: (filtered_vertices, filtered_faces, filtered_cad_face_ids, num_removed_components)
    """
    if len(faces) == 0:
        return vertices, faces, cad_face_ids, 0

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces_array = np.array(faces)
    num_faces = len(faces_array)

    # Build edge-to-face mapping (only for manifold edges with exactly 2 faces)
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(faces_array):
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]])),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Build adjacency matrix (only manifold edges connect faces)
    rows = []
    cols = []
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:  # Only manifold edges
            rows.append(face_list[0])
            cols.append(face_list[1])

    if len(rows) == 0:
        # No adjacency - each face is its own component
        # Keep only faces meeting threshold (likely just the first one)
        return vertices, faces[:1], cad_face_ids[:1] if cad_face_ids else None, num_faces - 1

    data = np.ones(len(rows))
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(num_faces, num_faces))
    adj_matrix = adj_matrix + adj_matrix.T  # Make symmetric

    # Find connected components
    n_components, labels = connected_components(adj_matrix, directed=False)

    if n_components <= 1:
        return vertices, faces, cad_face_ids, 0

    # Count faces per component
    from collections import Counter
    comp_sizes = Counter(labels)

    # Determine minimum face count to keep
    min_faces = int(num_faces * min_ratio)
    if min_faces < 1:
        min_faces = 1

    # Find largest component
    largest_comp = max(comp_sizes.keys(), key=lambda c: comp_sizes[c])

    # Collect faces to keep (largest + any above threshold)
    kept_face_indices = []
    num_removed = 0

    for face_idx in range(num_faces):
        comp = labels[face_idx]
        if comp == largest_comp or comp_sizes[comp] >= min_faces:
            kept_face_indices.append(face_idx)

    removed_comps = set()
    for comp, size in comp_sizes.items():
        if comp != largest_comp and size < min_faces:
            removed_comps.add(comp)
    num_removed = len(removed_comps)

    if num_removed == 0:
        return vertices, faces, cad_face_ids, 0

    # Filter faces
    filtered_faces = faces_array[kept_face_indices]

    # Filter cad_face_ids if present
    filtered_cad_face_ids = None
    if cad_face_ids is not None:
        filtered_cad_face_ids = [cad_face_ids[i] for i in kept_face_indices]

    # Remap vertex indices (remove unreferenced vertices)
    used_verts = np.unique(filtered_faces.flatten())
    vert_remap = {old: new for new, old in enumerate(used_verts)}

    filtered_verts = [vertices[i] for i in used_verts]
    filtered_faces = [[vert_remap[v] for v in face] for face in filtered_faces]

    return filtered_verts, filtered_faces, filtered_cad_face_ids, num_removed


def main():
    parser = argparse.ArgumentParser(description='Mesh BREP file using OCC')
    parser.add_argument('input_brep', help='Input BREP file path')
    parser.add_argument('output_json', help='Output JSON file path (mesh data)')
    parser.add_argument('linear_deflection', type=float, help='Linear deflection for meshing')
    parser.add_argument('angular_deflection', type=float, help='Angular deflection for meshing')
    parser.add_argument('--merge-vertices', action='store_true', default=True, help='Merge duplicate vertices')
    parser.add_argument('--no-merge-vertices', dest='merge_vertices', action='store_false')
    parser.add_argument('--merge-tolerance', type=float, default=1e-5,
                        help='Tolerance for merging duplicate vertices (default: 1e-5)')
    parser.add_argument('--extract-face-ids', action='store_true', default=True, help='Extract face IDs')
    parser.add_argument('--no-extract-face-ids', dest='extract_face_ids', action='store_false')
    parser.add_argument('--result-file', help='Path to write JSON result (optional)')
    parser.add_argument('--min-component-ratio', type=float, default=0,
                        help='Keep components with at least this ratio of faces (0 = only largest, default)')
    args = parser.parse_args()

    result = mesh_brep(
        args.input_brep,
        args.output_json,
        args.linear_deflection,
        args.angular_deflection,
        args.merge_vertices,
        args.merge_tolerance,
        args.extract_face_ids,
        args.min_component_ratio
    )

    if args.result_file:
        with open(args.result_file, 'w') as f:
            json.dump(result, f)

    if result['success']:
        print(f"OK: {result.get('num_vertices', '?')} vertices, {result.get('num_faces', '?')} triangles")
        sys.exit(0)
    else:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)


def mesh_brep(input_path, output_path, linear_deflection, angular_deflection, merge_vertices, merge_tolerance, extract_face_ids, min_component_ratio=0):
    """Mesh BREP file and output mesh data as JSON."""
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.BRep import BRep_Tool, BRep_Builder
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.TopoDS import topods, TopoDS_Shape
        from OCC.Core.BRepTools import breptools

        # Read input BREP
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        breptools.Read(shape, str(input_path), builder)

        # Mesh the shape
        mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
        mesher.Perform()

        if not mesher.IsDone():
            return {"success": False, "error": "BRepMesh failed"}

        # Extract triangulation
        all_verts = []
        all_faces = []
        cad_face_ids = []
        vertex_offset = 0
        cad_face_idx = 0

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)

            if tri is not None:
                trsf = loc.Transformation()

                # Extract vertices with transformation
                for i in range(1, tri.NbNodes() + 1):
                    pnt = tri.Node(i)
                    pnt.Transform(trsf)
                    all_verts.append([pnt.X(), pnt.Y(), pnt.Z()])

                # Extract triangles with proper orientation handling
                is_reversed = face.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    triangle = tri.Triangle(i)
                    n1, n2, n3 = triangle.Get()
                    if is_reversed:
                        all_faces.append([vertex_offset + n1 - 1, vertex_offset + n3 - 1, vertex_offset + n2 - 1])
                    else:
                        all_faces.append([vertex_offset + n1 - 1, vertex_offset + n2 - 1, vertex_offset + n3 - 1])
                    cad_face_ids.append(cad_face_idx)

                vertex_offset += tri.NbNodes()

            cad_face_idx += 1
            explorer.Next()

        if len(all_verts) == 0:
            return {"success": False, "error": "No vertices produced"}

        # Note: Vertex merging and degenerate face removal is done in cad_nodes.py
        # after trimesh object is created (using trimesh.nondegenerate_faces())
        # This subprocess just outputs raw OCC mesh data

        # Remove small disconnected components (debris from OCC meshing)
        if min_component_ratio > 0 and len(all_faces) > 0:
            all_verts, all_faces, cad_face_ids, num_removed = filter_small_components(
                all_verts, all_faces, cad_face_ids if extract_face_ids else None, min_component_ratio
            )
            if num_removed > 0:
                print(f"[mesh_subprocess] Removed {num_removed} small disconnected components")

        # Write output JSON
        mesh_data = {
            "vertices": all_verts,
            "faces": all_faces,
        }
        if extract_face_ids:
            mesh_data["cad_face_ids"] = cad_face_ids

        with open(output_path, 'w') as f:
            json.dump(mesh_data, f)

        return {
            "success": True,
            "num_vertices": len(all_verts),
            "num_faces": len(all_faces),
            "num_cad_faces": cad_face_idx,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == '__main__':
    main()
