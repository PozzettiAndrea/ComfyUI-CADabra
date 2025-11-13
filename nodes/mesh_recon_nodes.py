"""
Mesh Reconstruction Nodes for CADabra
Implements the Mesh-to-CAD pipeline with 4 key nodes:
1. QuadRemeshNode - Improve mesh quality (Phase 1: trimesh, Future: QuadWild)
2. PointCloudSegmentationNode - Segment point clouds (Phase 1: KMeans, Future: PointNet++)
3. PrimitiveFittingNode - Fit geometric primitives (Phase 1: pyRANSAC3D, Future: SPFN)
4. BrepGenerationNode - Generate B-rep CAD models (Phase 1: gmsh, Future: BrepGen)
"""

import numpy as np
import trimesh
import gmsh
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

# Note: B-rep generation now uses gmsh.model.occ (OpenCASCADE kernel)
# This eliminates the need for pythonocc-core, simplifying installation
# gmsh is already a required dependency for CAD loading and meshing


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
                "mesh": ("MESH",),
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
                "smooth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "remesh"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def remesh(self, mesh: Dict, target_edge_length: float, iterations: int, smooth: bool = True) -> Tuple[Dict]:
        """
        Remesh the input triangle mesh to improve quality

        Args:
            mesh: Input MESH dict with vertices and faces
            target_edge_length: Desired edge length for remeshing
            iterations: Number of subdivision iterations
            smooth: Whether to apply smoothing

        Returns:
            Tuple containing refined MESH dict
        """
        print(f"🔄 QuadRemesh: Processing mesh with {len(mesh['vertices'])} vertices, {len(mesh['faces'])} faces")

        # Convert to trimesh object
        tm = trimesh.Trimesh(
            vertices=mesh['vertices'],
            faces=mesh['faces']
        )

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

        # Convert back to MESH format
        output_mesh = {
            "vertices": np.array(tm.vertices, dtype=np.float32),
            "faces": np.array(tm.faces, dtype=np.int32),
            "normals": np.array(tm.vertex_normals, dtype=np.float32),
            "type": mesh.get("type", "surface"),
            "element_size": target_edge_length
        }

        print(f"✅ QuadRemesh: Output {len(output_mesh['vertices'])} vertices, {len(output_mesh['faces'])} faces")

        return (output_mesh,)


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
                "input_data": ("MESH,POINT_CLOUD",),
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

    def segment(self, input_data: Dict, num_segments: int, use_normals: bool = True,
                sample_points: int = 10000) -> Tuple[Dict]:
        """
        Segment point cloud using KMeans clustering

        Args:
            input_data: MESH or POINT_CLOUD dict
            num_segments: Number of segments to create
            use_normals: Include normals in clustering features
            sample_points: Number of points to sample (if input is mesh)

        Returns:
            Tuple containing SEGMENTED_CLOUD dict
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for segmentation. Install with: pip install scikit-learn>=1.3.0")

        # Convert input to point cloud
        if "faces" in input_data:
            # Input is MESH - convert to point cloud
            print(f"🔍 Segmentation: Converting mesh to point cloud ({sample_points} points)")
            tm = trimesh.Trimesh(vertices=input_data['vertices'], faces=input_data['faces'])

            # Sample points from mesh surface
            points, face_indices = trimesh.sample.sample_surface(tm, sample_points)
            normals = tm.face_normals[face_indices]

        else:
            # Input is already POINT_CLOUD
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
# Node 3: PrimitiveFittingNode
# ============================================================================

class PrimitiveFittingNode:
    """
    Fits geometric primitives to segmented point cloud.
    Phase 1: Uses pyRANSAC3D for robust fitting
    Future: Will integrate SPFN for learned primitive fitting
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_cloud": ("SEGMENTED_CLOUD",),
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

    RETURN_TYPES = ("PRIMITIVES",)
    FUNCTION = "fit_primitives"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def fit_primitives(self, segmented_cloud: Dict, primitive_type: str,
                      ransac_threshold: float, min_points: int = 100) -> Tuple[Dict]:
        """
        Fit geometric primitives to each segment using RANSAC

        Args:
            segmented_cloud: SEGMENTED_CLOUD dict with labeled points
            primitive_type: Type of primitive to fit ("auto" or specific type)
            ransac_threshold: RANSAC inlier threshold
            min_points: Minimum points required for fitting

        Returns:
            Tuple containing PRIMITIVES dict
        """
        if not HAS_PYRANSAC:
            raise ImportError("pyRANSAC3D is required for primitive fitting. Install with: pip install pyransac3d>=0.5.0")

        points = segmented_cloud['points']
        labels = segmented_cloud['labels']
        num_segments = segmented_cloud['num_segments']

        print(f"🔧 Primitive Fitting: Processing {num_segments} segments")

        primitives = []
        assignments = np.full(len(points), -1, dtype=np.int32)

        for segment_id in range(num_segments):
            # Get points for this segment
            segment_mask = labels == segment_id
            segment_points = points[segment_mask]

            if len(segment_points) < min_points:
                print(f"  Segment {segment_id}: Skipped (only {len(segment_points)} points, need {min_points})")
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

                # Update assignments for inlier points
                if 'inliers' in best_primitive:
                    global_indices = np.where(segment_mask)[0]
                    inlier_global_indices = global_indices[best_primitive['inliers']]
                    assignments[inlier_global_indices] = len(primitives) - 1

                    inlier_ratio = np.sum(best_primitive['inliers']) / len(segment_points)
                    print(f"    ✓ Fitted {best_primitive['type']}: {inlier_ratio*100:.1f}% inliers")
            else:
                print(f"    ✗ Failed to fit primitive")

        output = {
            "primitives": primitives,
            "assignments": assignments,
            "num_primitives": len(primitives)
        }

        print(f"✅ Primitive Fitting: Fitted {len(primitives)} primitives")

        return (output,)

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
    Phase 1: Manual construction using gmsh.model.occ (OpenCASCADE kernel)
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

    RETURN_TYPES = ("BREP_MODEL", "STRING")
    RETURN_NAMES = ("brep_model", "file_path")
    FUNCTION = "generate_brep"
    CATEGORY = "CADabra/Mesh Reconstruction"

    def generate_brep(self, primitives: Dict, output_path: str,
                     combine_shapes: bool = True, tolerance: float = 0.001) -> Tuple[Dict, str]:
        """
        Generate B-rep CAD model from primitives using gmsh

        Args:
            primitives: PRIMITIVES dict with fitted shapes
            output_path: Path to save STEP file
            combine_shapes: Whether to combine into single shape
            tolerance: Geometric tolerance for operations

        Returns:
            Tuple containing BREP_MODEL dict and file path
        """
        primitives_list = primitives['primitives']

        print(f"🏗️  B-rep Generation: Converting {len(primitives_list)} primitives to CAD (using gmsh)")

        # Ensure gmsh is initialized (should already be done in __init__.py)
        if not gmsh.is_initialized():
            gmsh.initialize()

        # Create a new model for this B-rep generation
        model_name = f"brep_model_{id(primitives)}"
        gmsh.model.add(model_name)

        shape_tags = []  # Store (dim, tag) tuples for created shapes

        for i, prim in enumerate(primitives_list):
            prim_type = prim['type']
            params = prim['params']

            try:
                if prim_type == "plane":
                    # Create a bounded planar face
                    tag = self._create_plane_face(params, prim['points'])
                    if tag is not None:
                        shape_tags.append((2, tag))  # 2 = surface dimension
                        print(f"  ✓ Primitive {i}: Plane → Face (tag {tag})")

                elif prim_type == "cylinder":
                    # Create cylindrical surface
                    tag = self._create_cylinder(params, prim['points'])
                    if tag is not None:
                        shape_tags.append((3, tag))  # 3 = volume dimension
                        print(f"  ✓ Primitive {i}: Cylinder → Solid (tag {tag})")

                elif prim_type == "sphere":
                    # Create spherical surface
                    tag = self._create_sphere(params)
                    if tag is not None:
                        shape_tags.append((3, tag))  # 3 = volume dimension
                        print(f"  ✓ Primitive {i}: Sphere → Solid (tag {tag})")

                else:
                    print(f"  ⚠️  Primitive {i}: {prim_type} not yet supported")

            except Exception as e:
                print(f"  ✗ Primitive {i}: Error creating {prim_type}: {e}")

        if not shape_tags:
            gmsh.model.remove()
            raise ValueError("No shapes were successfully created from primitives")

        # Synchronize before boolean operations
        gmsh.model.occ.synchronize()

        # Combine shapes if requested
        if combine_shapes and len(shape_tags) > 1:
            print(f"  Combining {len(shape_tags)} shapes into single B-rep...")
            try:
                # Use gmsh fuse to combine all shapes
                # Start with first shape, fuse with rest
                result_tag = shape_tags[0]
                for i in range(1, len(shape_tags)):
                    # Fuse returns (outDimTags, outDimTagsMap)
                    out_dim_tags, _ = gmsh.model.occ.fuse([result_tag], [shape_tags[i]])
                    if out_dim_tags:
                        result_tag = out_dim_tags[0]
                    else:
                        print(f"  ⚠️  Failed to fuse shape {i}")

                gmsh.model.occ.synchronize()
                print(f"  ✓ Successfully combined shapes")
            except Exception as e:
                print(f"  ⚠️  Failed to combine shapes: {e}")
                print(f"  Keeping shapes separate")

        # Synchronize final geometry
        gmsh.model.occ.synchronize()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Save to STEP file
        print(f"  Saving to: {output_path}")
        try:
            gmsh.write(output_path)
        except Exception as e:
            gmsh.model.remove()
            raise RuntimeError(f"Failed to write STEP file to {output_path}: {e}")

        # Get topology statistics
        entities = gmsh.model.getEntities()
        topology = {
            "faces": len([e for e in entities if e[0] == 2]),  # 2D entities (surfaces)
            "edges": len([e for e in entities if e[0] == 1]),  # 1D entities (curves)
            "vertices": len([e for e in entities if e[0] == 0]),  # 0D entities (points)
            "volumes": len([e for e in entities if e[0] == 3])  # 3D entities (volumes)
        }

        output = {
            "model_name": model_name,
            "shape_tags": shape_tags,
            "file_path": output_path,
            "topology": topology,
            "num_primitives": len(primitives_list)
        }

        print(f"✅ B-rep Generation: Created B-rep with {topology['volumes']} volumes, "
              f"{topology['faces']} faces, {topology['edges']} edges, {topology['vertices']} vertices")
        print(f"   Saved to: {output_path}")

        # Clean up: remove model but keep gmsh initialized for other operations
        # Note: Don't call gmsh.finalize() as it would affect other nodes
        gmsh.model.remove()

        return (output, output_path)

    def _create_plane_face(self, params: Dict, points: np.ndarray):
        """Create a bounded planar face from plane parameters and points using gmsh"""
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

        # Create points in gmsh
        point_tags = []
        for corner in corners:
            tag = gmsh.model.occ.addPoint(float(corner[0]), float(corner[1]), float(corner[2]))
            point_tags.append(tag)

        # Create lines connecting the points
        line_tags = []
        for i in range(4):
            next_i = (i + 1) % 4
            line_tag = gmsh.model.occ.addLine(point_tags[i], point_tags[next_i])
            line_tags.append(line_tag)

        # Create curve loop from lines
        loop_tag = gmsh.model.occ.addCurveLoop(line_tags)

        # Create plane surface from loop
        surface_tag = gmsh.model.occ.addPlaneSurface([loop_tag])

        return surface_tag

    def _create_cylinder(self, params: Dict, points: np.ndarray):
        """Create a cylinder from parameters using gmsh"""
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

        # Create cylinder using gmsh
        # addCylinder(x, y, z, dx, dy, dz, r, tag=-1, angle=2*pi)
        # (x,y,z) = start point, (dx,dy,dz) = axis direction * height
        cyl_tag = gmsh.model.occ.addCylinder(
            float(start_center[0]),
            float(start_center[1]),
            float(start_center[2]),
            float(axis_normalized[0] * height),
            float(axis_normalized[1] * height),
            float(axis_normalized[2] * height),
            float(radius)
        )

        return cyl_tag

    def _create_sphere(self, params: Dict):
        """Create a sphere from parameters using gmsh"""
        center = params['center']
        radius = params['radius']

        # Create sphere using gmsh
        # addSphere(xc, yc, zc, radius, tag=-1, angle1=-pi/2, angle2=pi/2, angle3=2*pi)
        sphere_tag = gmsh.model.occ.addSphere(
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(radius)
        )

        return sphere_tag


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QuadRemesh": QuadRemeshNode,
    "PointCloudSegmentation": PointCloudSegmentationNode,
    "PrimitiveFitting": PrimitiveFittingNode,
    "BrepGeneration": BrepGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuadRemesh": "Quad Remesh (Mesh Reconstruction)",
    "PointCloudSegmentation": "Point Cloud Segmentation",
    "PrimitiveFitting": "Primitive Fitting (RANSAC)",
    "BrepGeneration": "B-rep Generation (CAD)",
}
