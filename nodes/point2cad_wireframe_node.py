"""
Point2CAD to WireframeInfo Node

Transforms Point2CAD segmented point cloud into a batch of face objects,
each containing point cloud data and boundary wire edges as B-splines.

Detects edges between adjacent faces and finds corner intersections.
Output can be visualized in the batch VTK viewer with both points and wires.
"""

import numpy as np
import trimesh
from typing import Tuple, List, Dict, Any, Optional

try:
    from scipy.spatial import cKDTree
    from scipy.interpolate import splprep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed. Point2CAD wireframe extraction will be limited.")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] pyvista not installed. Mesh intersection edges will not be available.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] open3d not installed. Ball pivoting mesh reconstruction will not be available.")


def _fit_segment_mesh(points: np.ndarray, method: str = "ball_pivoting", bbox_diag: float = 1.0) -> Optional[trimesh.Trimesh]:
    """
    Fit a mesh to segment points using specified reconstruction method.

    Args:
        points: Point cloud vertices (N, 3)
        method: Reconstruction method - "ball_pivoting", "convex_hull", or "alpha_shape"
        bbox_diag: Bounding box diagonal for computing radii

    Returns:
        trimesh.Trimesh or None if reconstruction fails
    """
    if len(points) < 4:
        return None

    try:
        if method == "convex_hull":
            # Fast convex hull - works well for convex faces
            return trimesh.convex.convex_hull(points)

        elif method == "ball_pivoting" and HAS_OPEN3D:
            # Ball pivoting - good for dense point clouds with normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=bbox_diag * 0.05, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

            # Adaptive radii based on point cloud density
            radii = [bbox_diag * r for r in [0.005, 0.01, 0.02, 0.04, 0.08]]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

            if len(mesh.triangles) == 0:
                print(f"[WARN] Ball pivoting produced no triangles, falling back to convex hull")
                return trimesh.convex.convex_hull(points)

            return trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles)
            )

        elif method == "alpha_shape":
            # Alpha shape - works well for non-convex planar faces
            try:
                from scipy.spatial import Delaunay
                # Compute alpha shape
                alpha = bbox_diag * 0.1  # Adaptive alpha based on model size

                # Use trimesh's alpha shape implementation
                mesh = trimesh.Trimesh(*trimesh.points.point_cloud_to_mesh(points, pitch=alpha))
                if len(mesh.faces) > 0:
                    return mesh
            except Exception as e:
                print(f"[WARN] Alpha shape failed: {e}")

            # Fallback to convex hull
            return trimesh.convex.convex_hull(points)

        else:
            # Default fallback: convex hull
            return trimesh.convex.convex_hull(points)

    except Exception as e:
        print(f"[WARN] Mesh fitting failed ({method}): {e}")
        return None


def _compute_mesh_intersection_edges(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> Optional[np.ndarray]:
    """
    Compute intersection curve between two meshes using PyVista.

    Args:
        mesh_a: First trimesh
        mesh_b: Second trimesh

    Returns:
        Array of edge points (N, 3) or None if no intersection
    """
    if not HAS_PYVISTA:
        return None

    if mesh_a is None or mesh_b is None:
        return None

    try:
        # Convert trimesh to PyVista
        pv_a = pv.wrap(mesh_a)
        pv_b = pv.wrap(mesh_b)

        # Compute intersection
        intersection, _, _ = pv_a.intersection(
            pv_b, split_first=False, split_second=False
        )

        if intersection.n_points > 0:
            edge_points = np.array(intersection.points)
            return edge_points

    except Exception as e:
        print(f"[WARN] Mesh intersection failed: {e}")

    return None


class Point2CADToWireframeInfo:
    """
    Transform segmented point cloud into face objects with B-spline boundary wires.

    Takes the segmented_cloud output from Point2CADSegmentation and creates
    a batch of trimesh objects, one per face/segment. Each trimesh contains:
    - Point cloud vertices for that segment
    - B-spline boundary wire stored in metadata for VTK rendering
    - Edge information for adjacent faces
    - Vertex attributes: segment_id, primitive_type, is_boundary
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_cloud": ("TRIMESH", {
                    "tooltip": "Segmented point cloud from Point2CADSegmentation node."
                }),
            },
            "optional": {
                "k_neighbors": ("INT", {
                    "default": 15,
                    "min": 3,
                    "max": 50,
                    "tooltip": "Number of neighbors to check for boundary detection."
                }),
                "min_points": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 1000,
                    "tooltip": "Minimum points per face to include in output."
                }),
                "spline_samples": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 500,
                    "tooltip": "Number of points to sample on the fitted B-spline boundary. Lower = cleaner wire."
                }),
                "spline_smoothing": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "B-spline smoothing factor. Higher = smoother but less accurate."
                }),
                "detect_edges": ("BOOLEAN", {
                    "default": True,
                    "label_on": "detect edges",
                    "label_off": "skip edges",
                    "tooltip": "Detect shared edges between adjacent faces."
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Distance threshold for detecting adjacent boundary points (relative to bounding box)."
                }),
                "include_unsegmented": ("BOOLEAN", {
                    "default": False,
                    "label_on": "include",
                    "label_off": "exclude",
                    "tooltip": "Include points with label -1 (unsegmented/background)."
                }),
                "edge_method": (["mesh_intersection", "point_boundary"], {
                    "default": "mesh_intersection",
                    "tooltip": "Method for edge detection: mesh_intersection (cleaner but slower) or point_boundary (faster but noisier)."
                }),
                "mesh_method": (["ball_pivoting", "convex_hull", "alpha_shape"], {
                    "default": "ball_pivoting",
                    "tooltip": "Method for fitting mesh to segment points (only used with mesh_intersection edge method)."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("face_batch", "summary")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "extract_wireframe"
    CATEGORY = "CADabra/Point2CAD"

    def _detect_boundary_points(
        self,
        all_points: np.ndarray,
        all_labels: np.ndarray,
        segment_mask: np.ndarray,
        segment_id: int,
        k_neighbors: int = 15
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """
        Detect THIN boundary points using the "interface points" method.

        Instead of marking all points that have different neighbors (thick band),
        we only keep points that are the CLOSEST point to some point in another segment.
        This gives a single-point-thick boundary.

        Returns:
            Tuple of (boundary_mask, boundary_points, neighbor_boundary_dict)
            neighbor_boundary_dict maps neighbor_segment_id -> boundary points near that neighbor
        """
        segment_points = all_points[segment_mask]
        num_segment_points = len(segment_points)
        segment_indices = np.where(segment_mask)[0]  # Global indices of segment points

        if num_segment_points < 4:
            return np.ones(num_segment_points, dtype=bool), segment_points, {}

        if not HAS_SCIPY:
            return np.ones(num_segment_points, dtype=bool), segment_points, {}

        # Build KDTree on THIS segment's points
        segment_tree = cKDTree(segment_points)

        # Find all points NOT in this segment
        other_mask = ~segment_mask
        other_points = all_points[other_mask]
        other_labels = all_labels[other_mask]

        if len(other_points) == 0:
            # No other segments, no boundary
            return np.zeros(num_segment_points, dtype=bool), np.array([]).reshape(0, 3), {}

        # For each point in OTHER segments, find its nearest point in THIS segment
        distances, nearest_in_segment = segment_tree.query(other_points, k=1)

        # The boundary points are the UNIQUE nearest points
        # (points that are "seen" by at least one point in another segment)
        thin_boundary_local_indices = np.unique(nearest_in_segment)

        # Create boundary mask
        boundary_mask = np.zeros(num_segment_points, dtype=bool)
        boundary_mask[thin_boundary_local_indices] = True
        boundary_points = segment_points[thin_boundary_local_indices]

        # Track which neighbor segments each boundary point borders
        # For edge detection between faces
        neighbor_boundary_points = {}  # neighbor_id -> list of local point indices

        for i, (other_pt_idx, nearest_seg_idx) in enumerate(zip(range(len(other_points)), nearest_in_segment)):
            neighbor_id = other_labels[other_pt_idx]
            if neighbor_id >= 0:  # Skip background
                if neighbor_id not in neighbor_boundary_points:
                    neighbor_boundary_points[neighbor_id] = set()
                neighbor_boundary_points[neighbor_id].add(nearest_seg_idx)

        # Convert to coordinate arrays
        neighbor_boundary_coords = {
            nid: segment_points[list(indices)]
            for nid, indices in neighbor_boundary_points.items()
            if len(indices) > 0
        }

        return boundary_mask, boundary_points, neighbor_boundary_coords

    def _cluster_boundary_points(
        self,
        boundary_points: np.ndarray,
        gap_threshold: float = None
    ) -> List[np.ndarray]:
        """
        Cluster boundary points into separate connected loops.

        Uses distance-based clustering to detect when boundary points
        form multiple disconnected loops (e.g., top and bottom of a cylinder).

        Args:
            boundary_points: All boundary points (N, 3)
            gap_threshold: Maximum distance between consecutive points in same loop.
                          If None, computed automatically from point density.

        Returns:
            List of point arrays, one per boundary loop
        """
        if len(boundary_points) < 4:
            return [boundary_points]

        if not HAS_SCIPY:
            return [boundary_points]

        # First, order all points using greedy nearest neighbor
        ordered = self._order_boundary_points_single(boundary_points)

        # Compute distances between consecutive ordered points
        dists = np.linalg.norm(np.diff(ordered, axis=0), axis=1)

        # Auto-compute gap threshold if not provided
        if gap_threshold is None:
            # Use median distance * factor as threshold
            median_dist = np.median(dists)
            gap_threshold = median_dist * 5.0  # Points > 5x median are gaps

        # Find gap indices where distance is much larger than typical
        gap_indices = np.where(dists > gap_threshold)[0]

        if len(gap_indices) == 0:
            # Single continuous loop
            return [ordered]

        # Split into multiple loops at gaps
        loops = []
        start_idx = 0

        for gap_idx in gap_indices:
            end_idx = gap_idx + 1
            if end_idx - start_idx >= 4:  # Minimum 4 points per loop
                loops.append(ordered[start_idx:end_idx])
            start_idx = end_idx

        # Don't forget the last segment
        if len(ordered) - start_idx >= 4:
            loops.append(ordered[start_idx:])

        # If we got no valid loops, return original as single loop
        if len(loops) == 0:
            return [ordered]

        return loops

    def _order_boundary_points_single(self, boundary_points: np.ndarray) -> np.ndarray:
        """Order boundary points into a connected polyline using nearest-neighbor traversal."""
        if len(boundary_points) < 3:
            return boundary_points

        n_points = len(boundary_points)

        if not HAS_SCIPY:
            return boundary_points

        tree = cKDTree(boundary_points)
        ordered_indices = [0]
        remaining = set(range(1, n_points))

        while remaining:
            current_pt = boundary_points[ordered_indices[-1]]
            k = min(len(remaining) + 1, n_points)
            distances, indices = tree.query(current_pt, k=k)

            found = False
            for idx in indices:
                if idx in remaining:
                    ordered_indices.append(idx)
                    remaining.remove(idx)
                    found = True
                    break

            if not found and remaining:
                next_idx = remaining.pop()
                ordered_indices.append(next_idx)

        return boundary_points[ordered_indices]

    def _order_boundary_points(self, boundary_points: np.ndarray) -> np.ndarray:
        """Order boundary points into a connected polyline (legacy wrapper)."""
        return self._order_boundary_points_single(boundary_points)

    def _fit_bspline(
        self,
        points: np.ndarray,
        num_samples: int = 50,
        smoothing: float = 0.1,
        closed: bool = True
    ) -> np.ndarray:
        """
        Fit a B-spline to ordered points and resample.

        Args:
            points: Ordered points (N, 3)
            num_samples: Number of points to sample on the spline
            smoothing: Smoothing factor (0 = interpolate, higher = smoother)
            closed: Whether the curve should be closed

        Returns:
            Resampled points on the B-spline (num_samples, 3)
        """
        if len(points) < 4:
            return points

        if not HAS_SCIPY:
            # Fallback: uniform subsampling
            indices = np.linspace(0, len(points) - 1, num_samples, dtype=int)
            return points[indices]

        try:
            # Close the curve by appending first point
            if closed:
                points_closed = np.vstack([points, points[0]])
            else:
                points_closed = points

            # Fit B-spline
            # s = smoothing factor, k = degree (3 = cubic)
            tck, u = splprep(
                [points_closed[:, 0], points_closed[:, 1], points_closed[:, 2]],
                s=smoothing * len(points),  # Scale smoothing by number of points
                k=min(3, len(points_closed) - 1),  # Cubic or lower if not enough points
                per=closed  # Periodic (closed) curve
            )

            # Sample the spline
            u_new = np.linspace(0, 1, num_samples)
            x_new, y_new, z_new = splev(u_new, tck)

            return np.column_stack([x_new, y_new, z_new])

        except Exception as e:
            print(f"[WARN] B-spline fitting failed: {e}, using original points")
            # Fallback: uniform subsampling
            indices = np.linspace(0, len(points) - 1, min(num_samples, len(points)), dtype=int)
            return points[indices]

    def _detect_shared_edges(
        self,
        face_boundaries: Dict[int, np.ndarray],
        face_neighbor_boundaries: Dict[int, Dict[int, np.ndarray]],
        threshold: float
    ) -> List[Dict]:
        """
        Detect shared edges between adjacent faces.

        Args:
            face_boundaries: Dict mapping face_id -> boundary B-spline points
            face_neighbor_boundaries: Dict mapping face_id -> {neighbor_id -> raw boundary points}
            threshold: Distance threshold for matching points

        Returns:
            List of edge dicts with keys: faces, points, corners
        """
        edges = []
        processed_pairs = set()

        for face_id, neighbor_dict in face_neighbor_boundaries.items():
            for neighbor_id, boundary_pts in neighbor_dict.items():
                # Skip if already processed this pair
                pair_key = tuple(sorted([face_id, neighbor_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Skip if neighbor doesn't exist in our face set
                if neighbor_id not in face_boundaries:
                    continue

                # Get the boundary points from this face that border the neighbor
                face_border_pts = boundary_pts

                # Get the boundary points from neighbor that border this face
                if face_id in face_neighbor_boundaries.get(neighbor_id, {}):
                    neighbor_border_pts = face_neighbor_boundaries[neighbor_id][face_id]
                else:
                    continue

                if len(face_border_pts) < 2 or len(neighbor_border_pts) < 2:
                    continue

                # Combine and fit a shared edge spline
                # Use points from both sides that are close together
                combined_pts = np.vstack([face_border_pts, neighbor_border_pts])

                # Order the combined points
                ordered_edge = self._order_boundary_points(combined_pts)

                # Fit B-spline to the edge
                edge_spline = self._fit_bspline(
                    ordered_edge,
                    num_samples=30,  # Fewer points for edges
                    smoothing=0.05,
                    closed=False  # Edges are open curves
                )

                # Find corner points (endpoints of the edge)
                corners = [edge_spline[0], edge_spline[-1]]

                edges.append({
                    'faces': pair_key,
                    'points': edge_spline,
                    'corners': corners,
                    'num_source_points': len(combined_pts)
                })

        return edges

    def _extract_edges_via_intersection(
        self,
        segment_meshes: Dict[int, trimesh.Trimesh],
        adjacency_pairs: List[Tuple[int, int]],
        spline_samples: int = 30,
        spline_smoothing: float = 0.05
    ) -> List[Dict]:
        """
        Extract edge curves by intersecting adjacent segment meshes.

        Uses PyVista mesh intersection to get clean edge curves between
        fitted segment meshes.

        Args:
            segment_meshes: Dict mapping segment_id -> fitted trimesh
            adjacency_pairs: List of (segment_i, segment_j) pairs that are adjacent
            spline_samples: Number of samples for B-spline fitting
            spline_smoothing: Smoothing factor for B-spline

        Returns:
            List of edge dicts with keys: faces, points, corners
        """
        edges = []

        for seg_i, seg_j in adjacency_pairs:
            mesh_i = segment_meshes.get(seg_i)
            mesh_j = segment_meshes.get(seg_j)

            if mesh_i is None or mesh_j is None:
                continue

            # Compute intersection curve
            edge_points = _compute_mesh_intersection_edges(mesh_i, mesh_j)

            if edge_points is None or len(edge_points) < 3:
                continue

            # Order the edge points
            ordered_edge = self._order_boundary_points(edge_points)

            # Fit B-spline to edge for clean curve
            edge_spline = self._fit_bspline(
                ordered_edge,
                num_samples=spline_samples,
                smoothing=spline_smoothing,
                closed=False  # Edges are open curves
            )

            # Find corners (endpoints)
            corners = [edge_spline[0], edge_spline[-1]]

            edges.append({
                'faces': (seg_i, seg_j),
                'points': edge_spline,
                'corners': corners,
                'num_source_points': len(edge_points)
            })

        return edges

    def _find_adjacent_segments(
        self,
        all_points: np.ndarray,
        all_labels: np.ndarray,
        segment_ids: List[int],
        k_neighbors: int = 15
    ) -> List[Tuple[int, int]]:
        """
        Find pairs of adjacent segments using KNN neighbor analysis.

        Args:
            all_points: All point cloud vertices
            all_labels: Segment labels for each point
            segment_ids: List of valid segment IDs
            k_neighbors: Number of neighbors to check

        Returns:
            List of (segment_i, segment_j) tuples where i < j
        """
        if not HAS_SCIPY:
            return []

        adjacency = set()
        tree = cKDTree(all_points)

        # Sample subset of points for efficiency
        sample_size = min(len(all_points), 5000)
        sample_indices = np.random.choice(len(all_points), sample_size, replace=False)

        for idx in sample_indices:
            point_label = all_labels[idx]
            if point_label < 0:  # Skip background
                continue

            # Find k nearest neighbors
            _, neighbor_indices = tree.query(all_points[idx], k=k_neighbors)

            for n_idx in neighbor_indices:
                neighbor_label = all_labels[n_idx]
                if neighbor_label >= 0 and neighbor_label != point_label:
                    # Found adjacent pair
                    pair = tuple(sorted([int(point_label), int(neighbor_label)]))
                    adjacency.add(pair)

        # Filter to only include valid segment IDs
        valid_pairs = [
            (a, b) for a, b in adjacency
            if a in segment_ids and b in segment_ids
        ]

        return valid_pairs

    def _create_face_trimesh(
        self,
        segment_points: np.ndarray,
        boundary_loops: List[np.ndarray],
        boundary_mask: np.ndarray,
        segment_id: int,
        primitive_type: int,
        edges: List[Dict],
        segment_normals: np.ndarray = None
    ) -> trimesh.PointCloud:
        """
        Create a trimesh point cloud with multiple B-spline boundary wires.

        Args:
            boundary_loops: List of B-spline boundary loops (can be multiple for cylinders, etc.)
        """
        face_mesh = trimesh.PointCloud(segment_points)

        # Set normals if available
        if segment_normals is not None and len(segment_normals) == len(segment_points):
            face_mesh.vertex_normals = segment_normals

        # Initialize vertex_attributes dict (PointCloud doesn't have this by default)
        if not hasattr(face_mesh, 'vertex_attributes') or face_mesh.vertex_attributes is None:
            face_mesh.vertex_attributes = {}

        # Set vertex attributes
        face_mesh.vertex_attributes['segment_id'] = np.full(
            len(segment_points), segment_id, dtype=np.int32
        )
        face_mesh.vertex_attributes['primitive_type'] = np.full(
            len(segment_points), primitive_type, dtype=np.int32
        )
        face_mesh.vertex_attributes['is_boundary'] = boundary_mask.astype(np.int32)

        # Store all boundary loops
        # For VTK rendering, we concatenate all loops into boundary_wire
        # but also store them separately in boundary_loops
        all_boundary_points = []
        boundary_loops_list = []

        for loop in boundary_loops:
            if len(loop) > 2:
                all_boundary_points.extend(loop.tolist())
                boundary_loops_list.append(loop.tolist())

        # Primary wire for VTK (first/largest loop)
        if boundary_loops_list:
            # Use largest loop as primary boundary_wire
            largest_loop = max(boundary_loops_list, key=len)
            face_mesh.metadata['boundary_wire'] = largest_loop
        else:
            face_mesh.metadata['boundary_wire'] = []

        # Store all loops separately
        face_mesh.metadata['boundary_loops'] = boundary_loops_list
        face_mesh.metadata['num_boundary_loops'] = len(boundary_loops_list)

        face_mesh.metadata['face_id'] = int(segment_id)
        face_mesh.metadata['primitive_type'] = int(primitive_type)
        face_mesh.metadata['num_boundary_points'] = sum(len(loop) for loop in boundary_loops_list)
        face_mesh.metadata['has_boundary_wire'] = len(boundary_loops_list) > 0

        # Store edges that involve this face
        face_edges = [e for e in edges if segment_id in e['faces']]
        if face_edges:
            face_mesh.metadata['edges'] = [
                {
                    'neighbor': e['faces'][0] if e['faces'][1] == segment_id else e['faces'][1],
                    'points': e['points'].tolist(),
                    'corners': [c.tolist() for c in e['corners']]
                }
                for e in face_edges
            ]
            face_mesh.metadata['num_edges'] = len(face_edges)

            # Collect all corner points
            all_corners = []
            for e in face_edges:
                all_corners.extend(e['corners'])
            face_mesh.metadata['corners'] = [c.tolist() for c in all_corners]
        else:
            face_mesh.metadata['edges'] = []
            face_mesh.metadata['num_edges'] = 0
            face_mesh.metadata['corners'] = []

        return face_mesh

    def extract_wireframe(
        self,
        segmented_cloud,
        k_neighbors: int = 15,
        min_points: int = 50,
        spline_samples: int = 50,
        spline_smoothing: float = 0.1,
        detect_edges: bool = True,
        edge_threshold: float = 0.05,
        include_unsegmented: bool = False,
        edge_method: str = "mesh_intersection",
        mesh_method: str = "ball_pivoting"
    ) -> Tuple[List[trimesh.PointCloud], str]:
        """
        Extract wireframe faces with B-spline boundaries from segmented point cloud.

        Args:
            segmented_cloud: Input segmented point cloud with 'label' vertex attribute
            k_neighbors: Number of neighbors for boundary detection
            min_points: Minimum points per face
            spline_samples: Points to sample on B-spline
            spline_smoothing: B-spline smoothing factor
            detect_edges: Whether to detect shared edges
            edge_threshold: Distance threshold for adjacency
            include_unsegmented: Include background points
            edge_method: "mesh_intersection" (clean) or "point_boundary" (fast)
            mesh_method: "ball_pivoting", "convex_hull", or "alpha_shape"
        """
        if segmented_cloud is None:
            raise ValueError("No segmented cloud provided. Connect Point2CADSegmentation output.")

        # Extract data from input
        all_points = np.array(segmented_cloud.vertices, dtype=np.float32)

        # Compute bounding box for relative thresholds
        bbox_extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        bbox_diag = np.linalg.norm(bbox_extent)
        abs_edge_threshold = edge_threshold * bbox_diag

        # Get labels from vertex attributes
        if hasattr(segmented_cloud, 'vertex_attributes') and 'label' in segmented_cloud.vertex_attributes:
            all_labels = segmented_cloud.vertex_attributes['label']
        else:
            raise ValueError("Input point cloud has no 'label' vertex attribute.")

        # Get primitive types if available
        if 'primitive_type' in segmented_cloud.vertex_attributes:
            all_primitives = segmented_cloud.vertex_attributes['primitive_type']
        else:
            all_primitives = np.zeros(len(all_points), dtype=np.int32)

        # Get normals if available
        all_normals = None
        if hasattr(segmented_cloud, 'vertex_normals') and segmented_cloud.vertex_normals is not None:
            all_normals = np.array(segmented_cloud.vertex_normals, dtype=np.float32)

        # Get unique segment IDs
        unique_labels = np.unique(all_labels)
        if not include_unsegmented:
            unique_labels = unique_labels[unique_labels >= 0]

        print(f"[Point2CAD WireframeInfo] Processing {len(unique_labels)} segments")
        print(f"   k_neighbors={k_neighbors}, spline_samples={spline_samples}, smoothing={spline_smoothing}")
        print(f"   edge_method={edge_method}, mesh_method={mesh_method}")

        PRIMITIVE_NAMES = {
            0: "Background", 1: "Plane", 2: "BSpline", 3: "Cone",
            4: "Cylinder", 5: "Sphere", 6: "Torus", 7: "Revolution",
            8: "Extrusion", 9: "Other"
        }

        # First pass: collect boundary info for all faces
        face_data = {}
        face_boundaries = {}
        face_neighbor_boundaries = {}

        for segment_id in unique_labels:
            segment_mask = all_labels == segment_id
            segment_points = all_points[segment_mask]
            num_points = len(segment_points)

            if num_points < min_points:
                continue

            # Detect boundary points and neighbor info
            boundary_mask, boundary_points, neighbor_boundaries = self._detect_boundary_points(
                all_points, all_labels, segment_mask, segment_id, k_neighbors
            )

            if len(boundary_points) < 4:
                continue

            # Cluster boundary points into separate loops (e.g., top/bottom of cylinder)
            boundary_loops_raw = self._cluster_boundary_points(boundary_points)

            # Fit B-spline to each boundary loop
            boundary_loops = []
            for loop_pts in boundary_loops_raw:
                if len(loop_pts) >= 4:
                    loop_spline = self._fit_bspline(
                        loop_pts,
                        num_samples=spline_samples,
                        smoothing=spline_smoothing,
                        closed=True
                    )
                    boundary_loops.append(loop_spline)

            if not boundary_loops:
                continue

            # Get majority primitive type
            segment_primitives = all_primitives[segment_mask]
            unique_prims, counts = np.unique(segment_primitives, return_counts=True)
            majority_prim = unique_prims[np.argmax(counts)]

            # Get normals
            segment_normals = all_normals[segment_mask] if all_normals is not None else None

            face_data[segment_id] = {
                'points': segment_points,
                'boundary_mask': boundary_mask,
                'boundary_loops': boundary_loops,  # List of spline loops
                'primitive_type': majority_prim,
                'normals': segment_normals,
                'num_raw_boundary': len(boundary_points),
                'num_loops': len(boundary_loops)
            }
            # Use first loop for edge detection (usually the main boundary)
            face_boundaries[segment_id] = boundary_loops[0] if boundary_loops else np.array([])
            face_neighbor_boundaries[segment_id] = neighbor_boundaries

        # Second pass: detect shared edges between faces
        edges = []
        segment_meshes = {}  # For mesh intersection method

        if detect_edges and len(face_data) > 1:
            print(f"   Detecting shared edges between {len(face_data)} faces...")

            if edge_method == "mesh_intersection" and HAS_PYVISTA:
                # Mesh intersection method: fit meshes to segments and intersect
                print(f"   Using mesh intersection method ({mesh_method} fitting)...")

                # Fit mesh to each segment
                for segment_id, data in face_data.items():
                    fitted_mesh = _fit_segment_mesh(
                        data['points'],
                        method=mesh_method,
                        bbox_diag=bbox_diag
                    )
                    if fitted_mesh is not None:
                        segment_meshes[segment_id] = fitted_mesh
                        print(f"      Segment {segment_id}: fitted mesh with {len(fitted_mesh.faces)} faces")
                    else:
                        print(f"      Segment {segment_id}: mesh fitting failed")

                # Find adjacent segment pairs
                valid_segment_ids = list(face_data.keys())
                adjacency_pairs = self._find_adjacent_segments(
                    all_points, all_labels, valid_segment_ids, k_neighbors
                )
                print(f"   Found {len(adjacency_pairs)} adjacent segment pairs")

                # Extract edges via intersection
                edges = self._extract_edges_via_intersection(
                    segment_meshes,
                    adjacency_pairs,
                    spline_samples=spline_samples // 2,  # Fewer samples for edges
                    spline_smoothing=spline_smoothing * 0.5
                )
                print(f"   Extracted {len(edges)} edges via mesh intersection")

            else:
                # Point boundary method: use original neighbor-based detection
                if edge_method == "mesh_intersection" and not HAS_PYVISTA:
                    print(f"   [WARN] PyVista not available, falling back to point_boundary method")

                edges = self._detect_shared_edges(
                    face_boundaries,
                    face_neighbor_boundaries,
                    abs_edge_threshold
                )

            print(f"   Found {len(edges)} shared edges")

        # Third pass: create face trimeshes
        face_batch = []
        for segment_id, data in face_data.items():
            face_mesh = self._create_face_trimesh(
                data['points'],
                data['boundary_loops'],  # Pass list of loops
                data['boundary_mask'],
                segment_id,
                data['primitive_type'],
                edges,
                data['normals']
            )
            face_batch.append(face_mesh)

            prim_name = PRIMITIVE_NAMES.get(int(data['primitive_type']), f"Type-{data['primitive_type']}")
            num_face_edges = face_mesh.metadata.get('num_edges', 0)
            num_loops = data.get('num_loops', 1)
            print(f"   Face {segment_id}: {len(data['points'])} pts, {num_loops} loops, {num_face_edges} edges, {prim_name}")

        # Generate summary
        summary_lines = [
            f"Extracted {len(face_batch)} faces with B-spline boundaries",
            f"Spline: {spline_samples} samples, smoothing={spline_smoothing}",
            f"Detected {len(edges)} shared edges",
            "",
            "Face details:"
        ]

        for face in face_batch:
            face_id = face.metadata.get('face_id', '?')
            prim_type = face.metadata.get('primitive_type', 0)
            prim_name = PRIMITIVE_NAMES.get(prim_type, f"Type-{prim_type}")
            n_pts = len(face.vertices)
            n_loops = face.metadata.get('num_boundary_loops', 1)
            n_edges = face.metadata.get('num_edges', 0)
            n_corners = len(face.metadata.get('corners', []))
            summary_lines.append(f"  Face {face_id}: {n_pts} pts, {n_loops} loops, {n_edges} edges, {n_corners} corners, {prim_name}")

        if edges:
            summary_lines.append("")
            summary_lines.append("Edges:")
            for e in edges:
                summary_lines.append(f"  Face {e['faces'][0]} <-> Face {e['faces'][1]}: {len(e['points'])} pts")

        summary = "\n".join(summary_lines)

        print(f"[Point2CAD WireframeInfo] Complete: {len(face_batch)} faces, {len(edges)} edges")

        return (face_batch, summary)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Point2CADToWireframeInfo": Point2CADToWireframeInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Point2CADToWireframeInfo": "Point2CAD to WireframeInfo",
}
