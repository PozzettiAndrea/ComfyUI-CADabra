"""
Mesh Reconstruction Nodes for CADabra
1. MeshFaceSegmentationNode - Segment mesh faces optimized for CAD reconstruction
"""

import logging
import numpy as np
import trimesh
import os
from typing import Dict, List, Tuple, Any

from comfy_api.latest import io

log = logging.getLogger("cadabra")

from sklearn.cluster import KMeans


# ============================================================================
# Node 1: MeshFaceSegmentationNode
# ============================================================================

class MeshFaceSegmentationNode(io.ComfyNode):
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
    def define_schema(cls):
        return io.Schema(
            node_id="MeshFaceSegmentation",
            display_name="Mesh Face Segmentation",
            category="CADabra/Reconstruction",
            description="Segment mesh faces for CAD reconstruction using various methods",
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Combo.Input("method", options=[
                    "facets_coplanar",
                    "cluster_normals",
                    "region_grow_normals",
                    "cluster_curvature"
                ]),
                io.Float.Input("facets_angle_threshold", default=0.1, min=0.001, max=1.57, step=0.01,
                               tooltip="Max angle (radians) between normals for coplanar faces. 0.1 rad ~ 5.7°",
                               optional=True),
                io.Int.Input("cluster_num_segments", default=10, min=2, max=100, step=1,
                             tooltip="Number of segments for clustering methods",
                             optional=True),
                io.Float.Input("region_grow_normal_threshold", default=0.95, min=0.5, max=1.0, step=0.01,
                               tooltip="Cosine similarity threshold (0.95 ~ 18° max difference)",
                               optional=True),
                io.Int.Input("region_grow_min_faces", default=10, min=1, max=1000, step=1,
                             tooltip="Minimum faces per segment (smaller segments marked as noise)",
                             optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="TRIMESH",
                                            tooltip="Segmented mesh with face labels in face_attributes"),
            ],
        )

    @classmethod
    def execute(cls, mesh, method,
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
            NodeOutput with Trimesh with face_labels stored in face_attributes['segment_id']
        """
        import trimesh
        import numpy as np

        log.info(f"MeshFaceSegmentation] Starting segmentation with method: {method}")
        log.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Select and run segmentation method
        if method == "facets_coplanar":
            face_labels, num_segments = cls._segment_facets(
                mesh, facets_angle_threshold
            )
        elif method == "cluster_normals":
            face_labels, num_segments = cls._segment_cluster_normals(
                mesh, cluster_num_segments
            )
        elif method == "region_grow_normals":
            face_labels, num_segments = cls._segment_region_grow(
                mesh, region_grow_normal_threshold, region_grow_min_faces
            )
        elif method == "cluster_curvature":
            face_labels, num_segments = cls._segment_cluster_curvature(
                mesh, cluster_num_segments
            )
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        # Calculate segment statistics
        segment_info = cls._calculate_segment_info(mesh, face_labels, num_segments)

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
        log.info(f"OK] Segmentation complete:")
        log.info(f"Method: {method}")
        log.info(f"Segments: {num_segments}")
        log.info(f"Unlabeled faces: {np.sum(face_labels == -1)}")
        for i in range(min(5, num_segments)):
            count = np.sum(face_labels == i)
            area = segment_info['segment_areas'][i]
            log.info(f"Segment {i}: {count} faces, area={area:.2f}")
        if num_segments > 5:
            log.info(f"... and {num_segments - 5} more segments")
        log.info(f"Segmentation stored in mesh.face_attributes['segment_id']")

        return io.NodeOutput(mesh)

    @staticmethod
    def _segment_facets(mesh, angle_threshold):
        """
        Segment using trimesh facets - groups coplanar adjacent faces.

        This is the simplest and fastest method, perfect for planar CAD faces.
        """
        import trimesh
        import numpy as np

        log.info(f"[Facets] Angle threshold: {angle_threshold:.3f} rad ({np.degrees(angle_threshold):.1f}°)")

        # Get coplanar face groups
        facets = trimesh.graph.facets(mesh, facet_threshold=angle_threshold)

        # Create label array (-1 = unlabeled)
        face_labels = np.full(len(mesh.faces), -1, dtype=np.int32)

        for segment_id, face_indices in enumerate(facets):
            face_labels[face_indices] = segment_id

        num_segments = len(facets)

        log.info(f"[Facets] Found {num_segments} coplanar regions")

        return face_labels, num_segments

    @staticmethod
    def _segment_cluster_normals(mesh, num_segments):
        """
        Segment using K-means clustering on face normals.

        Then splits disconnected components within each cluster to enforce
        topological connectivity.
        """
        from sklearn.cluster import KMeans
        import numpy as np
        from scipy.sparse.csgraph import connected_components

        log.info(f"[ClusterNormals] Target segments: {num_segments}")

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

        log.info(f"[ClusterNormals] Initial clusters: {num_segments}, Final segments: {current_id}")

        return final_labels, current_id

    @staticmethod
    def _segment_region_grow(mesh, normal_threshold, min_faces):
        """
        Segment using region growing based on normal similarity.

        Grows regions from seed faces, adding neighbors with similar normals.
        Best quality for CAD features but slower.
        """
        import numpy as np
        from collections import deque

        log.info(f"[RegionGrow] Normal threshold: {normal_threshold:.3f} (cos similarity)")
        log.info(f"[RegionGrow] Min faces per segment: {min_faces}")

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

        log.info(f"[RegionGrow] Created {segment_id} segments")
        log.info(f"[RegionGrow] Unlabeled faces (noise): {np.sum(face_labels == -1)}")

        return face_labels, segment_id

    @classmethod
    def _segment_cluster_curvature(cls, mesh, num_segments):
        """
        Segment using Gaussian curvature + face normals.

        Useful for identifying cylindrical/spherical features vs planar regions.
        """
        from sklearn.cluster import KMeans
        import numpy as np

        log.info(f"[ClusterCurvature] Target segments: {num_segments}")

        # Compute discrete Gaussian curvature at vertices
        try:
            from trimesh.curvature import discrete_gaussian_curvature_measure
            vertex_curvature = discrete_gaussian_curvature_measure(
                mesh.vertices, mesh.faces, mesh.area_faces
            )
        except Exception as e:
            log.info(f"[ClusterCurvature] Warning: Curvature computation failed: {e}")
            log.info(f"[ClusterCurvature] Falling back to normal clustering only")
            return cls._segment_cluster_normals(mesh, num_segments)

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

        log.info(f"[ClusterCurvature] Created {num_segments} segments")
        log.info(f"[ClusterCurvature] Curvature range: [{vertex_curvature.min():.4f}, {vertex_curvature.max():.4f}]")

        return face_labels, num_segments

    @staticmethod
    def _calculate_segment_info(mesh, face_labels, num_segments):
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
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "MeshFaceSegmentation": MeshFaceSegmentationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshFaceSegmentation": "Mesh Face Segmentation",
}
