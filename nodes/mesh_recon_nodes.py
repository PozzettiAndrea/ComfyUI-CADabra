"""
Mesh Reconstruction Nodes for CADabra
Implements the Mesh-to-CAD pipeline with 4 key nodes:
1. QuadRemeshNode - Improve mesh quality (Phase 1: trimesh, Future: QuadWild)
2. PointCloudSegmentationNode - Segment point clouds (Phase 1: KMeans, Future: PointNet++)
3. PrimitiveFittingNode - Fit geometric primitives (Phase 1: pyRANSAC3D, Future: SPFN)
4. BrepGenerationNode - Generate B-rep CAD models (Phase 1: pythonocc-core, Future: BrepGen)
"""

import numpy as np
import trimesh
import os
from typing import Dict, List, Tuple, Any

# Optional imports with error handling
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not installed. Point cloud features will be limited.")
    print("   Install with: pip install open3d>=0.17.0")

try:
    import pyransac3d as pyrsc
    HAS_PYRANSAC = True
except ImportError:
    HAS_PYRANSAC = False
    print("⚠️  pyRANSAC3D not installed. Primitive fitting will not work.")
    print("   Install with: pip install pyransac3d>=0.5.0")

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not installed. Segmentation will not work.")
    print("   Install with: pip install scikit-learn>=1.3.0")

# OCC support for CAD_MODEL compatibility and B-rep generation
try:
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace
    )
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    print("⚠️  pythonocc-core not installed. BrepGenerationNode and CAD_MODEL will have limited compatibility.")
    print("   Install with: conda install -c conda-forge pythonocc-core")


# ============================================================================
# Node 1: QuadRemeshNode
# ============================================================================

class QuadRemeshNode:
    """
    Improves mesh quality through remeshing.
    Phase 1: Uses trimesh subdivision and smoothing
    Future: Will integrate QuadWild for true quad meshing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target_edge_length": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "metadata": ("MESH_METADATA",),
                "smooth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "MESH_METADATA")
    FUNCTION = "remesh"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def remesh(self, mesh, target_edge_length: float, iterations: int, metadata=None, smooth: bool = True) -> Tuple:
        """
        Remesh the input triangle mesh to improve quality

        Args:
            mesh: Input TRIMESH object
            target_edge_length: Desired edge length for remeshing
            iterations: Number of subdivision iterations
            metadata: Optional MESH_METADATA dict
            smooth: Whether to apply smoothing

        Returns:
            Tuple containing refined TRIMESH object and updated metadata
        """
        print(f"🔄 QuadRemesh: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Get metadata from input or from trimesh.metadata
        if metadata is None:
            metadata = mesh.metadata.copy() if hasattr(mesh, 'metadata') and mesh.metadata else {}
        else:
            metadata = metadata.copy()

        # mesh is already a trimesh object
        tm = mesh

        # Phase 1 approach: Subdivision + smoothing
        # Calculate subdivision level based on target edge length
        current_edge_length = np.mean([np.linalg.norm(tm.vertices[e[0]] - tm.vertices[e[1]])
                                       for e in tm.edges[:100]])  # Sample 100 edges

        if target_edge_length < current_edge_length:
            # Subdivide to reduce edge length
            for i in range(iterations):
                tm = tm.subdivide()
                print(f"  Subdivision {i+1}/{iterations}: {len(tm.vertices)} vertices")

        # Apply smoothing if requested
        if smooth:
            # Laplacian smoothing
            trimesh.smoothing.filter_laplacian(tm, iterations=iterations)
            print(f"  Applied Laplacian smoothing ({iterations} iterations)")

        # Ensure manifold mesh
        if not tm.is_watertight:
            print("  ⚠️  Warning: Mesh is not watertight. Some operations may fail.")

        # Update metadata
        tm.metadata['cadabra_type'] = metadata.get('type', 'surface')
        tm.metadata['element_size'] = target_edge_length
        tm.metadata['remesh_applied'] = {
            'target_edge_length': target_edge_length,
            'iterations': iterations,
            'smooth': smooth
        }

        # Update metadata dict
        metadata['type'] = metadata.get('type', 'surface')
        metadata['element_size'] = target_edge_length
        metadata['remesh_applied'] = tm.metadata['remesh_applied']
        metadata['vertex_count'] = len(tm.vertices)
        metadata['face_count'] = len(tm.faces)

        print(f"✅ QuadRemesh: Output {len(tm.vertices)} vertices, {len(tm.faces)} faces")

        return (tm, metadata)


# ============================================================================
# Node 2: PointCloudSegmentationNode
# ============================================================================

class PointCloudSegmentationNode:
    """
    Segments point clouds into regions using clustering.
    Phase 1: KMeans clustering (simple, no training required)
    Future: Will integrate PointNet++ for semantic segmentation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TRIMESH,POINT_CLOUD",),
                "num_segments": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 50,
                    "step": 1
                }),
                "use_normals": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sample_points": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 100000,
                    "step": 100
                }),
            }
        }

    RETURN_TYPES = ("SEGMENTED_CLOUD",)
    FUNCTION = "segment"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def segment(self, input_data, num_segments: int, use_normals: bool = True,
                sample_points: int = 10000) -> Tuple[Dict]:
        """
        Segment point cloud using KMeans clustering

        Args:
            input_data: TRIMESH object or POINT_CLOUD dict
            num_segments: Number of segments to create
            use_normals: Include normals in clustering features
            sample_points: Number of points to sample (if input is mesh)

        Returns:
            Tuple containing SEGMENTED_CLOUD dict
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for segmentation. Install with: pip install scikit-learn>=1.3.0")

        # Convert input to point cloud
        if isinstance(input_data, trimesh.Trimesh):
            # Input is TRIMESH object
            print(f"🔍 Segmentation: Converting mesh to point cloud ({sample_points} points)")
            tm = input_data

            # Sample points from mesh surface
            points, face_indices = trimesh.sample.sample_surface(tm, sample_points)
            normals = tm.face_normals[face_indices]

        elif isinstance(input_data, dict) and "faces" in input_data:
            # Input is old MESH dict format (backward compatibility)
            print(f"🔍 Segmentation: Converting mesh dict to point cloud ({sample_points} points)")
            tm = trimesh.Trimesh(vertices=input_data['vertices'], faces=input_data['faces'])

            # Sample points from mesh surface
            points, face_indices = trimesh.sample.sample_surface(tm, sample_points)
            normals = tm.face_normals[face_indices]

        else:
            # Input is POINT_CLOUD dict
            print(f"🔍 Segmentation: Processing point cloud ({len(input_data['points'])} points)")
            points = input_data['points']
            normals = input_data.get('normals', np.zeros_like(points))

        # Prepare features for clustering
        features = points.copy()

        if use_normals and normals is not None and len(normals) > 0:
            # Concatenate position and normal features
            # Weight normals to have similar scale as positions
            normal_weight = 0.5
            features = np.hstack([points, normals * normal_weight])
            print(f"  Using position + normal features ({features.shape[1]} dimensions)")
        else:
            print(f"  Using position features only ({features.shape[1]} dimensions)")

        # Normalize features
        features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # KMeans clustering
        print(f"  Running KMeans with {num_segments} clusters...")
        kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)

        # Calculate confidence scores (distance to cluster center)
        distances = np.min(kmeans.transform(features_normalized), axis=1)
        max_distance = np.max(distances)
        confidences = 1.0 - (distances / (max_distance + 1e-8))

        # Create output
        output = {
            "points": points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "labels": labels.astype(np.int32),
            "num_segments": num_segments,
            "confidences": confidences.astype(np.float32)
        }

        # Print statistics
        for i in range(num_segments):
            count = np.sum(labels == i)
            print(f"  Segment {i}: {count} points ({100*count/len(points):.1f}%)")

        print(f"✅ Segmentation: Created {num_segments} segments")

        return (output,)


# ============================================================================
# Node 2.5: MeshFaceSegmentationNode
# ============================================================================

class MeshFaceSegmentationNode:
    """
    Segment mesh faces using various methods optimized for CAD reconstruction.

    Unlike point cloud segmentation, this preserves mesh topology by working
    directly on faces, making it better for identifying CAD features.

    Methods:
    - facets_coplanar: Group coplanar adjacent faces (fast, perfect for planar CAD)
    - cluster_normals: K-means clustering on face normals (fast, general purpose)
    - region_grow_normals: Region growing by normal similarity (best quality)
    - cluster_curvature: Curvature + normal clustering (identifies cylinders/spheres)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "method": ([
                    "facets_coplanar",
                    "cluster_normals",
                    "region_grow_normals",
                    "cluster_curvature"
                ],),
            },
            "optional": {
                # For facets_coplanar
                "facets_angle_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.57,  # π/2
                    "step": 0.01,
                    "tooltip": "Max angle (radians) between normals for coplanar faces. 0.1 rad ≈ 5.7°"
                }),
                # For cluster methods
                "cluster_num_segments": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of segments for clustering methods"
                }),
                # For region_grow_normals
                "region_grow_normal_threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Cosine similarity threshold (0.95 ≈ 18° max difference)"
                }),
                "region_grow_min_faces": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Minimum faces per segment (smaller segments marked as noise)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    OUTPUT_TOOLTIPS = ("Segmented mesh with face labels in face_attributes",)
    FUNCTION = "segment_mesh"
    CATEGORY = "CADabra/Reconstruction"
    DESCRIPTION = "Segment mesh faces for CAD reconstruction using various methods"

    def segment_mesh(self, mesh, method,
                    facets_angle_threshold=0.1,
                    cluster_num_segments=10,
                    region_grow_normal_threshold=0.95,
                    region_grow_min_faces=10):
        """
        Segment mesh faces using selected method.

        Args:
            mesh: Input trimesh object
            method: Segmentation method to use

        Returns:
            Trimesh with face_labels stored in face_attributes['segment_id']
        """
        import trimesh
        import numpy as np

        print(f"\n[MeshFaceSegmentation] Starting segmentation with method: {method}")
        print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Select and run segmentation method
        if method == "facets_coplanar":
            face_labels, num_segments = self._segment_facets(
                mesh, facets_angle_threshold
            )
        elif method == "cluster_normals":
            face_labels, num_segments = self._segment_cluster_normals(
                mesh, cluster_num_segments
            )
        elif method == "region_grow_normals":
            face_labels, num_segments = self._segment_region_grow(
                mesh, region_grow_normal_threshold, region_grow_min_faces
            )
        elif method == "cluster_curvature":
            face_labels, num_segments = self._segment_cluster_curvature(
                mesh, cluster_num_segments
            )
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        # Calculate segment statistics
        segment_info = self._calculate_segment_info(mesh, face_labels, num_segments)

        # Embed segmentation data as face attributes on the mesh
        if not hasattr(mesh, 'face_attributes'):
            mesh.face_attributes = {}

        mesh.face_attributes['segment_id'] = face_labels.astype(np.float32)

        # Store metadata in mesh metadata dict for downstream nodes
        if not hasattr(mesh, 'metadata'):
            mesh.metadata = {}
        mesh.metadata['segmentation'] = {
            'num_segments': num_segments,
            'segment_info': segment_info,
            'method': method
        }

        # Print summary
        print(f"\n✅ Segmentation complete:")
        print(f"  Method: {method}")
        print(f"  Segments: {num_segments}")
        print(f"  Unlabeled faces: {np.sum(face_labels == -1)}")
        for i in range(min(5, num_segments)):
            count = np.sum(face_labels == i)
            area = segment_info['segment_areas'][i]
            print(f"  Segment {i}: {count} faces, area={area:.2f}")
        if num_segments > 5:
            print(f"  ... and {num_segments - 5} more segments")
        print(f"  Segmentation stored in mesh.face_attributes['segment_id']")

        return (mesh,)

    def _segment_facets(self, mesh, angle_threshold):
        """
        Segment using trimesh facets - groups coplanar adjacent faces.

        This is the simplest and fastest method, perfect for planar CAD faces.
        """
        import trimesh
        import numpy as np

        print(f"[Facets] Angle threshold: {angle_threshold:.3f} rad ({np.degrees(angle_threshold):.1f}°)")

        # Get coplanar face groups
        facets = trimesh.graph.facets(mesh, facet_threshold=angle_threshold)

        # Create label array (-1 = unlabeled)
        face_labels = np.full(len(mesh.faces), -1, dtype=np.int32)

        for segment_id, face_indices in enumerate(facets):
            face_labels[face_indices] = segment_id

        num_segments = len(facets)

        print(f"[Facets] Found {num_segments} coplanar regions")

        return face_labels, num_segments

    def _segment_cluster_normals(self, mesh, num_segments):
        """
        Segment using K-means clustering on face normals.

        Then splits disconnected components within each cluster to enforce
        topological connectivity.
        """
        from sklearn.cluster import KMeans
        import numpy as np
        from scipy.sparse.csgraph import connected_components

        print(f"[ClusterNormals] Target segments: {num_segments}")

        # Get face normals
        normals = mesh.face_normals

        # Cluster normals
        kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normals)

        # Split disconnected components within clusters
        final_labels = np.full(len(mesh.faces), -1, dtype=np.int32)
        current_id = 0

        # Build face adjacency sparse matrix
        from scipy.sparse import csr_matrix
        num_faces = len(mesh.faces)

        # Create adjacency matrix from face_adjacency edges
        edges = mesh.face_adjacency
        data = np.ones(len(edges) * 2)  # Bidirectional
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(num_faces, num_faces))

        for cluster_id in range(num_segments):
            cluster_faces = np.where(labels == cluster_id)[0]

            if len(cluster_faces) == 0:
                continue

            # Extract subgraph for this cluster
            cluster_adj = adjacency_matrix[cluster_faces][:, cluster_faces]

            # Find connected components
            n_components, component_labels = connected_components(
                cluster_adj, directed=False, return_labels=True
            )

            # Assign unique IDs to each component
            for comp_id in range(n_components):
                comp_mask = component_labels == comp_id
                global_indices = cluster_faces[comp_mask]
                final_labels[global_indices] = current_id
                current_id += 1

        print(f"[ClusterNormals] Initial clusters: {num_segments}, Final segments: {current_id}")

        return final_labels, current_id

    def _segment_region_grow(self, mesh, normal_threshold, min_faces):
        """
        Segment using region growing based on normal similarity.

        Grows regions from seed faces, adding neighbors with similar normals.
        Best quality for CAD features but slower.
        """
        import numpy as np
        from collections import deque

        print(f"[RegionGrow] Normal threshold: {normal_threshold:.3f} (cos similarity)")
        print(f"[RegionGrow] Min faces per segment: {min_faces}")

        num_faces = len(mesh.faces)
        face_labels = np.full(num_faces, -1, dtype=np.int32)
        segment_id = 0

        # Build face adjacency map
        adjacency_list = {i: [] for i in range(num_faces)}
        for edge in mesh.face_adjacency:
            adjacency_list[edge[0]].append(edge[1])
            adjacency_list[edge[1]].append(edge[0])

        # Get face normals
        normals = mesh.face_normals

        # Process each face as potential seed
        for seed_face in range(num_faces):
            if face_labels[seed_face] >= 0:
                continue  # Already labeled

            # Start new region
            queue = deque([seed_face])
            region_faces = []
            seed_normal = normals[seed_face]

            while queue:
                face = queue.popleft()

                if face_labels[face] >= 0:
                    continue  # Already labeled

                # Check normal similarity
                similarity = np.dot(normals[face], seed_normal)
                if similarity < normal_threshold:
                    continue

                # Add to region
                face_labels[face] = segment_id
                region_faces.append(face)

                # Add neighbors to queue
                for neighbor in adjacency_list[face]:
                    if face_labels[neighbor] < 0:
                        queue.append(neighbor)

            # Only keep regions with enough faces
            if len(region_faces) >= min_faces:
                segment_id += 1
            else:
                # Mark small regions as noise
                for face in region_faces:
                    face_labels[face] = -1

        print(f"[RegionGrow] Created {segment_id} segments")
        print(f"[RegionGrow] Unlabeled faces (noise): {np.sum(face_labels == -1)}")

        return face_labels, segment_id

    def _segment_cluster_curvature(self, mesh, num_segments):
        """
        Segment using Gaussian curvature + face normals.

        Useful for identifying cylindrical/spherical features vs planar regions.
        """
        from sklearn.cluster import KMeans
        import numpy as np

        print(f"[ClusterCurvature] Target segments: {num_segments}")

        # Compute discrete Gaussian curvature at vertices
        try:
            from trimesh.curvature import discrete_gaussian_curvature_measure
            vertex_curvature = discrete_gaussian_curvature_measure(
                mesh.vertices, mesh.faces, mesh.area_faces
            )
        except Exception as e:
            print(f"[ClusterCurvature] Warning: Curvature computation failed: {e}")
            print(f"[ClusterCurvature] Falling back to normal clustering only")
            return self._segment_cluster_normals(mesh, num_segments)

        # Convert vertex curvature to face curvature (average of face vertices)
        face_curvature = np.mean(vertex_curvature[mesh.faces], axis=1)

        # Combine normals and curvature as features
        features = np.column_stack([
            mesh.face_normals,                     # 3 dims
            face_curvature.reshape(-1, 1),        # 1 dim
            mesh.face_adjacency_angles.mean() * np.ones((len(mesh.faces), 1))  # relative position
        ])

        # Normalize features
        features_mean = features.mean(axis=0)
        features_std = features.std(axis=0) + 1e-8
        features_normalized = (features - features_mean) / features_std

        # Cluster
        kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
        face_labels = kmeans.fit_predict(features_normalized)

        print(f"[ClusterCurvature] Created {num_segments} segments")
        print(f"[ClusterCurvature] Curvature range: [{vertex_curvature.min():.4f}, {vertex_curvature.max():.4f}]")

        return face_labels, num_segments

    def _calculate_segment_info(self, mesh, face_labels, num_segments):
        """Calculate statistics for each segment."""
        import numpy as np

        segment_areas = np.zeros(num_segments, dtype=np.float32)
        segment_face_counts = np.zeros(num_segments, dtype=np.int32)
        segment_avg_normals = np.zeros((num_segments, 3), dtype=np.float32)

        for seg_id in range(num_segments):
            seg_mask = face_labels == seg_id
            segment_face_counts[seg_id] = np.sum(seg_mask)

            if segment_face_counts[seg_id] > 0:
                # Sum areas of faces in this segment
                segment_areas[seg_id] = np.sum(mesh.area_faces[seg_mask])

                # Average normal (normalized)
                avg_normal = np.mean(mesh.face_normals[seg_mask], axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 0:
                    segment_avg_normals[seg_id] = avg_normal / norm

        return {
            "segment_areas": segment_areas,
            "segment_face_counts": segment_face_counts,
            "segment_avg_normals": segment_avg_normals
        }


# ============================================================================
# Node 2b: MeshSegmentToPointCloudNode
# ============================================================================

class MeshSegmentToPointCloudNode:
    """
    Convert face-segmented mesh to point cloud for primitive fitting.

    Takes a TRIMESH with face_attributes['segment_id'] from MeshFaceSegmentation
    and converts it to a point cloud (TRIMESH with vertices only) where each
    face centroid becomes a point with a segment label.

    This preserves face-level segmentation while creating a format compatible
    with primitive fitting operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    OUTPUT_TOOLTIPS = ("Point cloud as TRIMESH with vertex_attributes['segment_id']",)
    FUNCTION = "convert"
    CATEGORY = "CADabra/Reconstruction"
    DESCRIPTION = "Convert face-segmented mesh to point cloud using face centroids"

    def convert(self, mesh):
        """
        Convert face-segmented mesh to point cloud.

        Args:
            mesh: Input trimesh with face_attributes['segment_id']

        Returns:
            TRIMESH point cloud with vertex_attributes['segment_id']
        """
        import numpy as np

        # Validate input has segmentation data
        if not hasattr(mesh, 'face_attributes') or 'segment_id' not in mesh.face_attributes:
            raise ValueError(
                "Input mesh must have face_attributes['segment_id']. "
                "Connect this node to MeshFaceSegmentation output."
            )

        print(f"\n[MeshSegmentToPointCloud] Converting mesh to point cloud...")
        print(f"  Input: {len(mesh.faces)} faces")

        # Get face labels
        face_labels = mesh.face_attributes['segment_id']

        # Calculate face centroids (average of 3 vertices)
        face_centroids = (mesh.vertices[mesh.faces[:, 0]] +
                         mesh.vertices[mesh.faces[:, 1]] +
                         mesh.vertices[mesh.faces[:, 2]]) / 3.0

        # Get face normals
        face_normals = mesh.face_normals

        # Filter out noise faces (label == -1)
        valid_mask = face_labels >= 0
        valid_centroids = face_centroids[valid_mask]
        valid_normals = face_normals[valid_mask]
        valid_labels = face_labels[valid_mask]

        # Create point cloud as TRIMESH
        # Vertices = points, faces = empty array
        point_cloud = trimesh.Trimesh(
            vertices=valid_centroids,
            faces=np.empty((0, 3), dtype=np.int32),  # No faces
            process=False  # Don't process (no faces to process)
        )

        # Store segmentation labels as vertex attributes
        point_cloud.vertex_attributes['segment_id'] = valid_labels.astype(np.float32)

        # Store normals as vertex attributes (useful for visualization)
        point_cloud.vertex_attributes['normals'] = valid_normals.astype(np.float32)

        # Copy metadata from input mesh
        if hasattr(mesh, 'metadata') and 'segmentation' in mesh.metadata:
            point_cloud.metadata = {'segmentation': mesh.metadata['segmentation']}

        # Print summary
        num_segments = len(np.unique(valid_labels))
        print(f"\n✅ Conversion complete:")
        print(f"  Output: {len(valid_centroids)} points ({num_segments} segments)")
        print(f"  Filtered: {np.sum(~valid_mask)} noise faces")
        print(f"  Point cloud stored as TRIMESH with vertex_attributes")

        return (point_cloud,)


# ============================================================================
# Node 3: PrimitiveFittingNode
# ============================================================================

class PrimitiveFittingNode:
    """
    Fits geometric primitives to segmented point cloud.

    Accepts both TRIMESH (point cloud from MeshSegmentToPointCloud) and
    SEGMENTED_CLOUD (from PointCloudSegmentation) for maximum compatibility.

    Phase 1: Uses pyRANSAC3D for robust fitting
    Future: Will integrate SPFN for learned primitive fitting
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_cloud": ("TRIMESH,SEGMENTED_CLOUD",),
                "primitive_type": (["auto", "plane", "cylinder", "sphere", "cone"],),
                "ransac_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "display": "slider"
                }),
            },
            "optional": {
                "min_points": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10
                }),
            }
        }

    RETURN_TYPES = ("PRIMITIVES", "STRING")
    RETURN_NAMES = ("primitives", "summary")
    FUNCTION = "fit_primitives"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def fit_primitives(self, segmented_cloud, primitive_type: str,
                      ransac_threshold: float, min_points: int = 100) -> Tuple[Dict, str]:
        """
        Fit geometric primitives to each segment using RANSAC

        Args:
            segmented_cloud: SEGMENTED_CLOUD dict or TRIMESH point cloud with vertex_attributes['segment_id']
            primitive_type: Type of primitive to fit ("auto" or specific type)
            ransac_threshold: RANSAC inlier threshold
            min_points: Minimum points required for fitting

        Returns:
            Tuple containing PRIMITIVES dict and summary string
        """
        if not HAS_PYRANSAC:
            raise ImportError("pyRANSAC3D is required for primitive fitting. Install with: pip install pyransac3d>=0.5.0")

        # Convert input to standard format (points, labels, num_segments)
        if isinstance(segmented_cloud, trimesh.Trimesh):
            # Input is TRIMESH (point cloud from MeshSegmentToPointCloud)
            print(f"[PrimitiveFitting] Input: TRIMESH point cloud")

            if not hasattr(segmented_cloud, 'vertex_attributes') or 'segment_id' not in segmented_cloud.vertex_attributes:
                raise ValueError(
                    "TRIMESH input must have vertex_attributes['segment_id']. "
                    "Connect this node to MeshSegmentToPointCloud output."
                )

            points = segmented_cloud.vertices.astype(np.float32)
            labels = segmented_cloud.vertex_attributes['segment_id'].astype(np.int32)

            # Get num_segments from metadata or count unique labels
            if hasattr(segmented_cloud, 'metadata') and 'segmentation' in segmented_cloud.metadata:
                num_segments = segmented_cloud.metadata['segmentation']['num_segments']
            else:
                num_segments = len(np.unique(labels[labels >= 0]))

        elif isinstance(segmented_cloud, dict):
            # Input is SEGMENTED_CLOUD (from PointCloudSegmentation)
            print(f"[PrimitiveFitting] Input: SEGMENTED_CLOUD dict")
            points = segmented_cloud['points']
            labels = segmented_cloud['labels']
            num_segments = segmented_cloud['num_segments']

        else:
            raise TypeError(
                f"Input must be TRIMESH or SEGMENTED_CLOUD, got {type(segmented_cloud)}"
            )

        print(f"🔧 Primitive Fitting: Processing {num_segments} segments")

        primitives = []
        assignments = np.full(len(points), -1, dtype=np.int32)

        # Track statistics for summary
        stats = {
            'skipped': 0,
            'failed': 0,
            'fitted': 0,
            'planes': 0,
            'cylinders': 0,
            'spheres': 0,
            'cones': 0,
            'total_inlier_ratio': 0.0
        }

        for segment_id in range(num_segments):
            # Get points for this segment
            segment_mask = labels == segment_id
            segment_points = points[segment_mask]

            if len(segment_points) < min_points:
                print(f"  Segment {segment_id}: Skipped (only {len(segment_points)} points, need {min_points})")
                stats['skipped'] += 1
                continue

            print(f"  Segment {segment_id}: Fitting {primitive_type} to {len(segment_points)} points")

            # Fit primitive based on type
            if primitive_type == "auto":
                # Try different primitives and pick best fit
                best_primitive = self._fit_best_primitive(segment_points, ransac_threshold)
            else:
                # Fit specific primitive type
                best_primitive = self._fit_primitive(segment_points, primitive_type, ransac_threshold)

            if best_primitive is not None:
                best_primitive['segment_id'] = segment_id
                primitives.append(best_primitive)
                stats['fitted'] += 1

                # Count by primitive type
                prim_type = best_primitive['type']
                plural_key = prim_type + 's'
                if plural_key in stats:
                    stats[plural_key] += 1

                # Update assignments for inlier points
                if 'inliers' in best_primitive:
                    global_indices = np.where(segment_mask)[0]
                    inlier_global_indices = global_indices[best_primitive['inliers']]
                    assignments[inlier_global_indices] = len(primitives) - 1

                    inlier_ratio = np.sum(best_primitive['inliers']) / len(segment_points)
                    stats['total_inlier_ratio'] += inlier_ratio
                    print(f"    ✓ Fitted {best_primitive['type']}: {inlier_ratio*100:.1f}% inliers")
            else:
                print(f"    ✗ Failed to fit primitive")
                stats['failed'] += 1

        output = {
            "primitives": primitives,
            "assignments": assignments,
            "num_primitives": len(primitives)
        }

        # Create summary string
        avg_inlier = (stats['total_inlier_ratio'] / stats['fitted'] * 100) if stats['fitted'] > 0 else 0
        summary = f"""Primitive Fitting Results:
─────────────────────────────
Total Segments: {num_segments}
  ✓ Fitted: {stats['fitted']}
  ⊘ Skipped: {stats['skipped']} (< {min_points} points)
  ✗ Failed: {stats['failed']}

Primitives by Type:
  Planes: {stats['planes']}
  Cylinders: {stats['cylinders']}
  Spheres: {stats['spheres']}
  Cones: {stats['cones']}

Average Inlier Ratio: {avg_inlier:.1f}%
Total Points: {len(points)}
"""

        print(f"✅ Primitive Fitting: Fitted {len(primitives)} primitives")

        return (output, summary)

    def _fit_primitive(self, points: np.ndarray, prim_type: str, threshold: float) -> Dict:
        """Fit a specific primitive type to points"""
        try:
            if prim_type == "plane":
                plane = pyrsc.Plane()
                equation, inliers = plane.fit(points, thresh=threshold, maxIteration=1000)
                if equation is not None:
                    return {
                        "type": "plane",
                        "params": {
                            "equation": equation,  # [a, b, c, d] for ax+by+cz+d=0
                            "normal": equation[:3] / np.linalg.norm(equation[:3])
                        },
                        "inliers": inliers,
                        "points": points[inliers],
                        "confidence": len(inliers) / len(points)
                    }

            elif prim_type == "cylinder":
                cyl = pyrsc.Cylinder()
                center, axis, radius, inliers = cyl.fit(points, thresh=threshold, maxIteration=1000)
                if center is not None:
                    return {
                        "type": "cylinder",
                        "params": {
                            "center": center,
                            "axis": axis,
                            "radius": radius
                        },
                        "inliers": inliers,
                        "points": points[inliers],
                        "confidence": len(inliers) / len(points)
                    }

            elif prim_type == "sphere":
                sphere = pyrsc.Sphere()
                center, radius, inliers = sphere.fit(points, thresh=threshold, maxIteration=1000)
                if center is not None:
                    return {
                        "type": "sphere",
                        "params": {
                            "center": center,
                            "radius": radius
                        },
                        "inliers": inliers,
                        "points": points[inliers],
                        "confidence": len(inliers) / len(points)
                    }

            # Note: pyRANSAC3D doesn't have cone fitting built-in
            # Would need to implement custom cone fitting or use SPFN in Phase 2

        except Exception as e:
            print(f"    Error fitting {prim_type}: {e}")
            return None

        return None

    def _fit_best_primitive(self, points: np.ndarray, threshold: float) -> Dict:
        """Try fitting different primitives and return the best one"""
        best_primitive = None
        best_score = 0.0

        for prim_type in ["plane", "cylinder", "sphere"]:
            primitive = self._fit_primitive(points, prim_type, threshold)
            if primitive is not None and primitive['confidence'] > best_score:
                best_score = primitive['confidence']
                best_primitive = primitive

        return best_primitive


# ============================================================================
# Node 4: BrepGenerationNode
# ============================================================================

class BrepGenerationNode:
    """
    Generates B-rep CAD model from fitted primitives.
    Uses pythonocc-core (OpenCASCADE) directly for B-rep construction.
    Future: Will integrate BrepGen diffusion model for complex topologies
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primitives": ("PRIMITIVES",),
                "output_path": ("STRING", {
                    "default": "output/reconstructed.step",
                    "multiline": False
                }),
            },
            "optional": {
                "combine_shapes": ("BOOLEAN", {"default": True}),
                "tolerance": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0001,
                    "max": 0.1,
                    "step": 0.0001
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING")
    RETURN_NAMES = ("cad_model", "file_path")
    FUNCTION = "generate_brep"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def generate_brep(self, primitives: Dict, output_path: str,
                     combine_shapes: bool = True, tolerance: float = 0.001) -> Tuple[Dict, str]:
        """
        Generate B-rep CAD model from primitives using pythonocc-core

        Args:
            primitives: PRIMITIVES dict with fitted shapes
            output_path: Path to save STEP file
            combine_shapes: Whether to combine into single shape
            tolerance: Geometric tolerance for operations

        Returns:
            Tuple containing CAD_MODEL dict and file path
        """
        if not HAS_OCC:
            raise RuntimeError("pythonocc-core is required for B-rep generation. "
                             "Install with: conda install -c conda-forge pythonocc-core")

        primitives_list = primitives['primitives']

        print(f"  B-rep Generation: Converting {len(primitives_list)} primitives to CAD (using OCC)")

        shapes = []  # Store created OCC shapes

        for i, prim in enumerate(primitives_list):
            prim_type = prim['type']
            params = prim['params']

            try:
                if prim_type == "plane":
                    # Create a bounded planar face
                    shape = self._create_plane_face(params, prim['points'])
                    if shape is not None:
                        shapes.append(shape)
                        print(f"  Primitive {i}: Plane -> Face")

                elif prim_type == "cylinder":
                    # Create cylindrical solid
                    shape = self._create_cylinder(params, prim['points'])
                    if shape is not None:
                        shapes.append(shape)
                        print(f"  Primitive {i}: Cylinder -> Solid")

                elif prim_type == "sphere":
                    # Create spherical solid
                    shape = self._create_sphere(params)
                    if shape is not None:
                        shapes.append(shape)
                        print(f"  Primitive {i}: Sphere -> Solid")

                else:
                    print(f"  Primitive {i}: {prim_type} not yet supported")

            except Exception as e:
                print(f"  Primitive {i}: Error creating {prim_type}: {e}")

        if not shapes:
            raise ValueError("No shapes were successfully created from primitives")

        # Combine shapes if requested
        if combine_shapes and len(shapes) > 1:
            print(f"  Combining {len(shapes)} shapes into single B-rep...")
            try:
                result_shape = shapes[0]
                for i in range(1, len(shapes)):
                    fuse = BRepAlgoAPI_Fuse(result_shape, shapes[i])
                    if fuse.IsDone():
                        result_shape = fuse.Shape()
                    else:
                        print(f"  Failed to fuse shape {i}")
                occ_shape = result_shape
                print(f"  Successfully combined shapes")
            except Exception as e:
                print(f"  Failed to combine shapes: {e}")
                print(f"  Keeping shapes in compound")
                # Create compound of all shapes
                occ_shape = self._make_compound(shapes)
        elif len(shapes) == 1:
            occ_shape = shapes[0]
        else:
            occ_shape = self._make_compound(shapes)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Save to STEP file
        print(f"  Saving to: {output_path}")
        try:
            writer = STEPControl_Writer()
            writer.Transfer(occ_shape, STEPControl_AsIs)
            status = writer.Write(output_path)
            if status != IFSelect_RetDone:
                raise RuntimeError(f"STEP write returned status {status}")
        except Exception as e:
            raise RuntimeError(f"Failed to write STEP file to {output_path}: {e}")

        # Get topology statistics using TopExp_Explorer
        topology = self._count_topology(occ_shape)

        # Create CAD_MODEL output (compatible with all CAD nodes)
        output = {
            "occ_shape": occ_shape,
            "file_path": output_path,
            "format": ".step",
            "topology": topology,
            "num_primitives": len(primitives_list)
        }

        print(f"  B-rep Generation: Created B-rep with {topology['volumes']} volumes, "
              f"{topology['faces']} faces, {topology['edges']} edges, {topology['vertices']} vertices")
        print(f"   Saved to: {output_path}")

        return (output, output_path)

    def _make_compound(self, shapes: List) -> TopoDS_Compound:
        """Create a compound from multiple shapes"""
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for shape in shapes:
            builder.Add(compound, shape)
        return compound

    def _count_topology(self, shape) -> Dict[str, int]:
        """Count topological entities in a shape"""
        counts = {"faces": 0, "edges": 0, "vertices": 0, "volumes": 0}

        for topo_type, key in [(TopAbs_SOLID, "volumes"), (TopAbs_FACE, "faces"),
                                (TopAbs_EDGE, "edges"), (TopAbs_VERTEX, "vertices")]:
            explorer = TopExp_Explorer(shape, topo_type)
            while explorer.More():
                counts[key] += 1
                explorer.Next()

        return counts

    def _create_plane_face(self, params: Dict, points: np.ndarray):
        """Create a bounded planar face from plane parameters and points using OCC"""
        # Get plane normal and point
        normal = params['normal']

        # Find a point on the plane (use mean of inlier points)
        center = np.mean(points, axis=0)

        # Compute 2D bounding box of projected points
        u_axis = np.array([normal[1], -normal[0], 0])
        if np.linalg.norm(u_axis) < 1e-8:
            u_axis = np.array([1, 0, 0])
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
        v_axis = np.cross(normal, u_axis)

        u_coords = np.dot(points - center, u_axis)
        v_coords = np.dot(points - center, v_axis)

        u_min, u_max = np.min(u_coords), np.max(u_coords)
        v_min, v_max = np.min(v_coords), np.max(v_coords)

        # Create corners in 3D space
        corners = [
            center + u_min * u_axis + v_min * v_axis,
            center + u_max * u_axis + v_min * v_axis,
            center + u_max * u_axis + v_max * v_axis,
            center + u_min * u_axis + v_max * v_axis,
        ]

        # Create OCC points
        occ_points = [gp_Pnt(float(c[0]), float(c[1]), float(c[2])) for c in corners]

        # Create edges connecting the points
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(4):
            next_i = (i + 1) % 4
            edge = BRepBuilderAPI_MakeEdge(occ_points[i], occ_points[next_i]).Edge()
            wire_builder.Add(edge)

        if not wire_builder.IsDone():
            return None

        wire = wire_builder.Wire()

        # Create face from wire
        face_builder = BRepBuilderAPI_MakeFace(wire)
        if not face_builder.IsDone():
            return None

        return face_builder.Face()

    def _create_cylinder(self, params: Dict, points: np.ndarray):
        """Create a cylinder from parameters using OCC"""
        center = params['center']
        axis = params['axis']
        radius = params['radius']

        # Compute cylinder height from point cloud extent
        axis_normalized = axis / (np.linalg.norm(axis) + 1e-8)
        projections = np.dot(points - center, axis_normalized)
        height = np.max(projections) - np.min(projections)

        # Adjust center to start at minimum projection
        min_proj = np.min(projections)
        start_center = center + min_proj * axis_normalized

        # Create cylinder using OCC
        # gp_Ax2 defines the coordinate system: origin + Z direction
        origin = gp_Pnt(float(start_center[0]), float(start_center[1]), float(start_center[2]))
        direction = gp_Dir(float(axis_normalized[0]), float(axis_normalized[1]), float(axis_normalized[2]))
        ax2 = gp_Ax2(origin, direction)

        cylinder = BRepPrimAPI_MakeCylinder(ax2, float(radius), float(height))
        if not cylinder.IsDone():
            return None

        return cylinder.Shape()

    def _create_sphere(self, params: Dict):
        """Create a sphere from parameters using OCC"""
        center = params['center']
        radius = params['radius']

        # Create sphere using OCC
        origin = gp_Pnt(float(center[0]), float(center[1]), float(center[2]))
        sphere = BRepPrimAPI_MakeSphere(origin, float(radius))
        if not sphere.IsDone():
            return None

        return sphere.Shape()


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QuadRemesh": QuadRemeshNode,
    "PointCloudSegmentation": PointCloudSegmentationNode,
    "MeshFaceSegmentation": MeshFaceSegmentationNode,
    "MeshSegmentToPointCloud": MeshSegmentToPointCloudNode,
    "PrimitiveFitting": PrimitiveFittingNode,
    "BrepGeneration": BrepGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuadRemesh": "Quad Remesh (Mesh Reconstruction)",
    "PointCloudSegmentation": "Point Cloud Segmentation",
    "MeshFaceSegmentation": "Mesh Face Segmentation",
    "MeshSegmentToPointCloud": "Mesh Segment to Point Cloud",
    "PrimitiveFitting": "Primitive Fitting (RANSAC)",
    "BrepGeneration": "B-rep Generation (CAD)",
}
