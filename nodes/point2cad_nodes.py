"""
Point2CAD Nodes for CADabra (CVPR 2024)
Neural network based point cloud segmentation from Point2CAD paper.

Paper: "Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds"
Project: https://www.obukhov.ai/point2cad.html
GitHub: https://github.com/prs-eth/point2cad

Pipeline:
1. LoadPoint2CADModel - Download/load segmentation network (ParseNet or HPNet)
2. Point2CADSegmentation - Segment point cloud into surface clusters
3. Point2CADSurfaceFitting - Fit primitives to segments (plane, sphere, cylinder, cone)
4. Point2CADTopologyExtraction - Extract edges and corners via intersection
5. Point2CADExportBrep - Export to STEP format

NOTE: SplineNet models for freeform surfaces are not publicly released by authors.
"""

import numpy as np
import torch
import trimesh
import os
from tqdm import tqdm

# Optional imports with error handling
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None
    print("[CADabra] Warning: open3d not installed. Normal estimation will use fallback method.")
    print("   Install with: pip install open3d>=0.17.0")
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Use relative import from parent package
from ..utils.model_loader import download_point2cad_model, get_model_path, list_available_models
from ..utils.parsenet.model import PrimitivesEmbeddingDGCNGn
from ..utils.parsenet.mean_shift import MeanShift

# Optional imports with error handling
try:
    import geomdl
    from geomdl import BSpline
    HAS_GEOMDL = True
except ImportError:
    HAS_GEOMDL = False
    print("[WARN] geomdl not installed. B-spline operations will be limited.")
    print("   Install with: pip install geomdl>=5.3.0")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] pyvista not installed. Advanced visualization will be limited.")
    print("   Install with: pip install pyvista>=0.43.0")

try:
    from rtree import index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
    print("[WARN] rtree not installed. Spatial indexing will be slower.")
    print("   Install with: pip install rtree>=1.0.0")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed.")
    print("   Install with: pip install scipy>=1.11.0")

try:
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_Kernel import Point_3
    from CGAL import CGAL_Polygon_mesh_processing as CGAL_PMP
    HAS_CGAL = True
except ImportError:
    HAS_CGAL = False
    print("[WARN] CGAL not installed. Mesh clipping will not work.")
    print("   Install with: pip install cgal")

try:
    import pymesh
    HAS_PYMESH = True
except ImportError:
    HAS_PYMESH = False
    print("[WARN] PyMesh not installed. Point2CAD mesh clipping will use fallback.")
    print("   Install via CADabra installer or from: https://github.com/PozzettiAndrea/PyMesh/releases")


# ============================================================================
# Node 1: LoadPoint2CADModel
# ============================================================================

class LoadPoint2CADModel:
    """
    Downloads and loads Point2CAD segmentation models from GitHub.
    Models are cached in ComfyUI/models/cadrecon/point2cad/

    Supports:
    - ParseNet (with normals): For point clouds with normal information
    - ParseNet (no normals): For raw point clouds only
    - HPNet: Highest performance, pretrained on ABC dataset
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["ParseNet (with normals)", "ParseNet (no normals)", "HPNet (ABC dataset)"], {
                    "default": "ParseNet (with normals)",
                    "tooltip": "Segmentation network. ParseNet for general use, HPNet for ABC dataset models."
                }),
            },
            "optional": {
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Automatically download models if not found locally."
                }),
            }
        }

    RETURN_TYPES = ("POINT2CAD_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "CADabra/Point2CAD"

    def load_model(self, model: str, auto_download: bool = True) -> Tuple:
        """
        Load or download a Point2CAD segmentation model.
        """
        # Map display names to internal model keys
        model_map = {
            "ParseNet (with normals)": "parsenet_with_normals",
            "ParseNet (no normals)": "parsenet_no_normals",
            "HPNet (ABC dataset)": "hpnet",
        }
        model_type = model_map.get(model, "parsenet_with_normals")

        print(f"[Point2CAD] Loading model: {model_type}")

        # Check if model exists locally
        model_path = get_model_path(model_type)

        if model_path is None:
            if auto_download:
                print(f"[Point2CAD] Model not found locally, downloading...")
                model_path = download_point2cad_model(model_type)

                if model_path is None:
                    raise RuntimeError(f"Failed to download {model_type}. Check network connection.")
            else:
                raise FileNotFoundError(f"Model {model_type} not found and auto_download is disabled.")

        # Load the segmentation model
        try:
            print(f"[Point2CAD] Loading model from: {model_path}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize model architecture
            if model_type == "parsenet_with_normals" or model_type == "hpnet":
                # Mode 0 with 6 channels for points + normals (matching original Point2CAD)
                # Note: Original Point2CAD uses mode=0 even with normals
                model = PrimitivesEmbeddingDGCNGn(
                    embedding=True,
                    emb_size=128,
                    primitives=True,
                    num_primitives=10,
                    loss_function=None,
                    mode=0,
                    num_channels=6,
                )
            elif model_type == "parsenet_no_normals":
                # Mode 0 is for points only (3 channels)
                model = PrimitivesEmbeddingDGCNGn(
                    embedding=True,
                    emb_size=128,
                    primitives=True,
                    num_primitives=10,
                    loss_function=None,
                    mode=0,
                    num_channels=3,
                )
            else:
                print(f"[WARN] Model type {model_type} unknown, using generic ParseNet")
                model = PrimitivesEmbeddingDGCNGn(embedding=True, emb_size=128, primitives=True, num_primitives=10, mode=5, num_channels=6)

            model_data = {
                "model_type": model_type,
                "model_path": str(model_path),
                "model": model,
                "device": device,
                "fitting_models": {}
            }

            # Load weights
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    # Handle potential state dict keys mismatch
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                        
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k
                        # Remove prefixes
                        if name.startswith('module.'):
                            name = name[7:]
                        if name.startswith('affinitynet.'):
                            name = name[12:]
                        new_state_dict[name] = v
                        
                    model.load_state_dict(new_state_dict)
                    model.to(device)
                    model.eval()
                    print(f"[OK] Segmentation model loaded successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model weights: {e}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # NOTE: SplineNet fitting models are NOT publicly released by Point2CAD authors
            # Surface fitting will use analytical primitive fitting only

            info_string = f"Model: {model_type}\nPath: {model_path}\nDevice: {model_data['device']}"
            print(f"[OK] Model ready: {model_type} on {model_data['device']}")

            return (model_data, info_string)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


# ============================================================================
# Node 2: Point2CADSegmentation
# ============================================================================

class Point2CADSegmentation:
    """
    Segment point cloud into surface clusters using ParseNet or HPNet.
    Each cluster represents a single CAD surface (primitive or freeform).

    Methods:
    - ParseNet (no normals): General purpose, for point clouds without normals
    - ParseNet (with normals): General purpose, requires normals
    - HPNet (ABC dataset): Best for CAD models, trained on ABC dataset, requires normals
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (["ParseNet (no normals)", "ParseNet (with normals)", "HPNet (ABC dataset)"], {
                    "default": "HPNet (ABC dataset)",
                    "tooltip": "Segmentation method. HPNet is best for CAD models. ParseNet variants for general use."
                }),
                "point_cloud": ("TRIMESH,POINT_CLOUD", {
                    "tooltip": "Input point cloud to segment."
                }),
                "model": ("POINT2CAD_MODEL", {
                    "tooltip": "Loaded Point2CAD segmentation model."
                }),
                "batch_size": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 16384,
                    "step": 512,
                    "tooltip": "Number of points to process in each batch."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum confidence score required to assign a point to a segment."
                }),
            },
            "optional": {
                "meanshift_quantile": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "Mean Shift bandwidth quantile. Controls segmentation granularity. Lower (0.01-0.02) = more/smaller segments. Higher = fewer/larger merged segments. Original Point2CAD uses 0.015."
                }),
                "min_cluster_points": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Minimum points to form a valid cluster (used in fallback DBSCAN). Higher filters noise, lower captures small details."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("segmented_cloud", "summary")
    FUNCTION = "segment"
    CATEGORY = "CADabra/Point2CAD"

    def _normalize_points_pca(self, points: np.ndarray, return_transform: bool = False):
        """
        Normalize points using PCA alignment (matching original Point2CAD).

        Steps:
        1. Center at mean
        2. PCA to find principal directions
        3. Rotate so smallest eigenvector aligns with [1,0,0]
        4. Scale by max bounding box extent

        Args:
            points: (N, 3) point cloud
            return_transform: If True, also return the transform parameters

        Returns:
            If return_transform=False: Normalized points as float32 array
            If return_transform=True: (normalized_points, transform_dict)
                transform_dict contains: mean, rotation, scale for denormalization
                Denormalize: world = (normalized * scale) @ rotation.T + mean
        """
        EPS = np.finfo(np.float32).eps

        # Store original mean for transform
        mean = np.mean(points, 0)

        # Center
        points_centered = points - mean

        # PCA to find principal directions
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Find smallest eigenvector (flattest direction)
        smallest_ev = eigenvectors[:, np.argmin(eigenvalues)]

        # Compute rotation to align smallest eigenvector with [1,0,0]
        target = np.array([1.0, 0.0, 0.0])
        v = np.cross(smallest_ev, target)
        c = np.dot(smallest_ev, target)
        s = np.linalg.norm(v)

        if s < 1e-10:  # Already aligned or anti-aligned
            if c < 0:  # Anti-aligned, flip
                R = np.diag([-1.0, 1.0, 1.0])
            else:
                R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        # Apply rotation
        points_rotated = (R @ points_centered.T).T

        # Scale by max bounding box extent
        bbox_extent = np.max(points_rotated, 0) - np.min(points_rotated, 0)
        scale = np.max(bbox_extent) + EPS
        points_normalized = points_rotated / scale

        if return_transform:
            transform = {
                'mean': mean.astype(np.float32),
                'rotation': R.astype(np.float32),
                'scale': float(scale),
            }
            return points_normalized.astype(np.float32), transform
        else:
            return points_normalized.astype(np.float32)

    def segment(self, method: str, point_cloud, model, batch_size: int, confidence_threshold: float, meanshift_quantile: float = 0.015, min_cluster_points: int = 10) -> Tuple:
        """
        Segment point cloud into surface clusters.
        """
        # HPNet and ParseNet (with normals) both require normals
        requires_normals = "with normals" in method or "HPNet" in method

        print(f"[Point2CAD] Segmentation")
        print(f"   Method: {method}")
        print(f"   quantile={meanshift_quantile}, requires_normals={requires_normals}")

        # Extract points and normals from input
        if isinstance(point_cloud, (trimesh.Trimesh, trimesh.PointCloud)):
            points = np.array(point_cloud.vertices, dtype=np.float32)
            # Check if normals exist
            has_normals = (hasattr(point_cloud, 'vertex_normals')
                          and point_cloud.vertex_normals is not None
                          and len(point_cloud.vertex_normals) == len(point_cloud.vertices))
            if has_normals:
                normals = np.array(point_cloud.vertex_normals, dtype=np.float32)
            else:
                normals = None
        elif isinstance(point_cloud, dict):
            points = point_cloud.get('points', point_cloud.get('vertices'))
            normals = point_cloud.get('normals', None)
            if points is None:
                raise ValueError("Input dictionary must contain 'points' or 'vertices' key.")
            has_normals = normals is not None
        else:
            raise TypeError(f"Unsupported point_cloud type: {type(point_cloud)}")

        # Strict validation: if method requires normals, input MUST have normals
        if requires_normals and normals is None:
            raise ValueError(
                f"Method '{method}' requires normals but input point cloud has none. "
                "Either use 'ParseNet (no normals)' method, or provide a point cloud with normals "
                "(use 'Estimate Point Cloud Normals' node from GeometryPack)."
            )

        num_points = len(points)
        print(f"   Points: {num_points}")
        print(f"   Has normals: {normals is not None}")

        if model is None or model.get("model") is None:
            raise ValueError("No model provided. Please connect a LoadPoint2CADModel node.")

        # Run Inference
        device = model["device"]
        net = model["model"]
        model_type = model.get("model_type", "unknown")
        print(f"   Model: {model_type} on {device}")

        # Validate model matches selected method
        # HPNet and ParseNet (with normals) both use 6 channels (points + normals)
        # ParseNet (no normals) uses 3 channels (points only)
        model_uses_normals = "no_normals" not in model_type  # hpnet and with_normals both use normals

        if requires_normals and not model_uses_normals:
            raise ValueError(
                f"Method '{method}' requires normals but loaded model is '{model_type}'. "
                "Load 'ParseNet (with normals)' or 'HPNet (ABC dataset)' model instead."
            )
        if not requires_normals and model_uses_normals:
            raise ValueError(
                f"Method '{method}' does not use normals but loaded model is '{model_type}'. "
                "Load 'ParseNet (no normals)' model instead."
            )

        # Warn if method doesn't match model type exactly (but they're compatible)
        if "HPNet" in method and "hpnet" not in model_type:
            print(f"   [WARN] Method is HPNet but model is {model_type}. Using loaded model.")
        elif "ParseNet" in method and "hpnet" in model_type:
            print(f"   [WARN] Method is ParseNet but model is HPNet. Using loaded model.")

        # Normalize points using PCA alignment (matching original Point2CAD)
        # Keep transform for metadata (debugging only - output is world coords)
        normalized_points, norm_transform = self._normalize_points_pca(points, return_transform=True)
        print(f"   Normalized points using PCA alignment (scale={norm_transform['scale']:.4f})")

        # Prepare input tensor
        points_tensor = torch.from_numpy(normalized_points).float().to(device)
        if requires_normals:
            normals_tensor = torch.from_numpy(normals).float().to(device)
            # Input shape: (1, N, 6) -> (1, 6, N) for model
            input_tensor = torch.cat([points_tensor, normals_tensor], dim=1).unsqueeze(0).permute(0, 2, 1)
        else:
            # Input shape: (1, N, 3) -> (1, 3, N) for model
            input_tensor = points_tensor.unsqueeze(0).permute(0, 2, 1)

        print(f"   Input shape: {input_tensor.shape}")

        with torch.no_grad():
            try:
                embedding, primitives_log_prob, _ = net(input_tensor, compute_loss=False)
                
                # Embedding shape: (1, EmbSize, N) -> (N, EmbSize)
                # CRITICAL: Normalize embeddings for MeanShift (it expects unit hypersphere)
                embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)
                
                # Primitives prediction
                # shape: (1, NumPrims, N) -> (N,)
                pred_primitives = torch.max(primitives_log_prob[0], 0)[1].data.cpu().numpy()
                
                print(f"   Embedding shape: {embedding.shape}")
                
            except Exception as e:
                raise RuntimeError(f"Inference failed: {e}")

        # Clustering
        print(f"   Running Mean Shift clustering...")
        ms = MeanShift()

        # Quantile determines bandwidth. Lower quantile = smaller bandwidth = more clusters.
        iterations = 50
        print(f"   Clustering parameters: quantile={meanshift_quantile:.4f}")

        _, _, cluster_ids = ms.guard_mean_shift(
            embedding, meanshift_quantile, iterations, kernel_type="gaussian"
        )
        labels = cluster_ids.cpu().numpy()
        print(f"   MeanShift completed successfully")

        num_clusters = len(np.unique(labels))
        print(f"[OK] Segmentation complete: {num_clusters} segments found")

        # Create TRIMESH output with WORLD coordinates (original input points)
        # Surface fitting will normalize internally
        segmented_cloud = trimesh.PointCloud(points)

        # Set normals (these don't need normalization - just direction vectors)
        if normals is not None:
            segmented_cloud.vertex_normals = normals

        # Set vertex attributes for segmentation data
        if not hasattr(segmented_cloud, 'vertex_attributes'):
            segmented_cloud.vertex_attributes = {}

        segmented_cloud.vertex_attributes['label'] = labels.astype(np.int32)
        segmented_cloud.vertex_attributes['primitive_type'] = pred_primitives.astype(np.int32)
        segmented_cloud.vertex_attributes['confidence'] = np.ones(len(points), dtype=np.float32)

        # Set metadata
        segmented_cloud.metadata['num_segments'] = num_clusters
        segmented_cloud.metadata['is_point_cloud'] = True
        segmented_cloud.metadata['segmentation_method'] = 'point2cad_meanshift'
        segmented_cloud.metadata['model_type'] = model.get('model_type', 'unknown')
        # Store normalization transform for reference/debugging
        segmented_cloud.metadata['normalization'] = {
            'mean': norm_transform['mean'].tolist(),
            'rotation': norm_transform['rotation'].tolist(),
            'scale': norm_transform['scale'],
        }

        # Primitive type names (from ABC Dataset)
        PRIMITIVE_NAMES = {
            0: "Background",
            1: "Plane",
            2: "BSpline",
            3: "Cone",
            4: "Cylinder",
            5: "Sphere",
            6: "Torus",
            7: "Revolution",
            8: "Extrusion",
            9: "Other",
        }

        # Compute majority type per cluster
        cluster_majorities = []
        for cluster_id in np.unique(labels):
            if cluster_id < 0:
                continue
            cluster_mask = labels == cluster_id
            cluster_types = pred_primitives[cluster_mask]
            if len(cluster_types) > 0:
                unique, counts = np.unique(cluster_types, return_counts=True)
                majority_type = unique[np.argmax(counts)]
                cluster_majorities.append(PRIMITIVE_NAMES.get(majority_type, f"Type-{majority_type}"))

        # Per-point type counts
        unique_types = np.unique(pred_primitives)
        type_counts = {t: np.sum(pred_primitives == t) for t in unique_types}
        type_counts_str = ", ".join([
            f"{PRIMITIVE_NAMES.get(t, f'Type-{t}')}({type_counts[t]})"
            for t in unique_types
        ])

        print(f"[OK] Segmentation complete: {num_clusters} clusters")
        print(f"   Cluster majorities: {', '.join(cluster_majorities)}")
        print(f"   Point types: {type_counts_str}")

        summary = f"Segmented {num_points} points into {num_clusters} clusters\n\n"
        summary += f"Model: {model.get('model_type', 'unknown')}\n\n"
        summary += f"Surface types per cluster (majority): {', '.join(cluster_majorities)}\n\n"
        summary += f"Surface types per point: {type_counts_str}"

        return (segmented_cloud, summary)


# ============================================================================
# Node 3: Point2CADSurfaceFitting
# ============================================================================

class Point2CADSurfaceFitting:
    """
    Fit surfaces to segmented point cloud using Point2CAD method.
    - Primitives (plane, sphere, cylinder, cone): analytical least-squares fitting
    - Freeform surfaces: SplineINR neural network trained on-the-fly
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_cloud": ("TRIMESH", {
                    "tooltip": "Segmented point cloud from Point2CADSegmentation."
                }),
            },
            "optional": {
                "use_inr": ("BOOLEAN", {
                    "default": True,
                    "label_on": "primitives + INR",
                    "label_off": "primitives only",
                    "tooltip": "Enable SplineINR for freeform surfaces (slower but better for complex shapes)."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for INR training. CUDA is much faster."
                }),
            }
        }

    RETURN_TYPES = ("SURFACE_PARAMS", "TRIMESH", "STRING")
    RETURN_NAMES = ("surface_params", "surface_meshes", "summary")
    FUNCTION = "fit_surfaces"
    CATEGORY = "CADabra/Point2CAD"

    def _normalize_points(self, points: np.ndarray):
        """
        Normalize points to unit box centered at origin.
        Returns (normalized_points, transform_dict).
        Denormalize: world = normalized * scale + mean
        """
        EPS = np.finfo(np.float32).eps
        mean = np.mean(points, 0)
        points_centered = points - mean
        bbox_extent = np.max(points_centered, 0) - np.min(points_centered, 0)
        scale = np.max(bbox_extent) + EPS
        points_normalized = points_centered / scale
        transform = {
            'mean': mean.astype(np.float32),
            'scale': float(scale),
        }
        return points_normalized.astype(np.float32), transform

    def _denormalize_mesh(self, mesh, transform):
        """
        Transform mesh from normalized space back to world coordinates.
        world = normalized * scale + mean
        """
        if mesh is None:
            return None
        mesh.vertices = mesh.vertices * transform['scale'] + transform['mean']
        return mesh

    def fit_surfaces(self, segmented_cloud, use_inr: bool = True, device: str = "cuda") -> Tuple:
        """
        Fit surfaces to each segment using Point2CAD method.
        """
        if segmented_cloud is None:
            raise ValueError("No segmented cloud provided. Please connect a Point2CADSegmentation node.")

        # Import Point2CAD fitting
        from ..utils.point2cad_fitting import process_one_surface

        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, using CPU")
            device = "cpu"

        # Extract data from TRIMESH (world coordinates)
        points_world = np.array(segmented_cloud.vertices, dtype=np.float32)
        normals = np.array(segmented_cloud.vertex_normals, dtype=np.float32) if hasattr(segmented_cloud, 'vertex_normals') and segmented_cloud.vertex_normals is not None else None
        labels = segmented_cloud.vertex_attributes.get('label', np.zeros(len(points_world), dtype=np.int32))
        num_segments = segmented_cloud.metadata.get('num_segments', len(np.unique(labels)))

        print(f"[Point2CAD] Surface Fitting: {num_segments} segments")
        print(f"   Device: {device}, INR: {use_inr}")

        # Normalize entire point cloud for fitting (isotropic scale to unit box)
        points_normalized, norm_transform = self._normalize_points(points_world)
        print(f"   Normalized points for fitting (scale={norm_transform['scale']:.4f})")

        surface_params_list = []  # Just the analytical params (type, params, normalization)
        num_primitives = 0
        num_freeform = 0

        # Type mapping
        prim_map = {"plane": 1, "cylinder": 4, "sphere": 5, "cone": 3, "open_spline": 2}

        # Process each segment with progress bar
        unique_labels = [l for l in np.unique(labels) if l >= 0]
        pbar = tqdm(unique_labels, desc="Surface Fitting", unit="seg")
        for segment_id in pbar:
            segment_mask = labels == segment_id
            segment_points_world = points_world[segment_mask]
            segment_points_normalized = points_normalized[segment_mask]
            segment_normals = normals[segment_mask] if normals is not None else None

            if len(segment_points_normalized) < 20:
                pbar.set_postfix(seg=segment_id, pts=len(segment_points_normalized), status="SKIP")
                continue

            pbar.set_postfix(seg=segment_id, pts=len(segment_points_normalized), status="fitting...")

            # Use Point2CAD fitting on NORMALIZED points
            try:
                result = process_one_surface(
                    segment_points_normalized,
                    segment_id=segment_id,
                    device=device if use_inr else "cpu",
                    progress_bar=False
                )

                if result is None:
                    pbar.set_postfix(seg=segment_id, pts=len(segment_points_normalized), status="FAIL")
                    continue

                fit_type = result["type"]
                fit_err = result["err"]
                all_errors = result.get("all_errors", {})

                # Update progress bar with result
                pbar.set_postfix(seg=segment_id, result=fit_type.upper(), err=f"{fit_err:.4f}")

                # Print detailed errors (optional verbose output)
                err_strs = []
                for prim, err in all_errors.items():
                    if err is not None:
                        marker = "*" if prim == fit_type else " "
                        err_strs.append(f"{prim}={err:.4f}{marker}")
                    else:
                        err_strs.append(f"{prim}=FAIL ")
                tqdm.write(f"   Seg {segment_id}: {', '.join(err_strs)} -> {fit_type.upper()}")

                # Denormalize mesh back to world coordinates
                result_mesh = result.get("mesh")
                if result_mesh is not None:
                    result_mesh = self._denormalize_mesh(result_mesh, norm_transform)

                # Build surface_params (analytical definition + mesh)
                surface_param = {
                    "segment_id": result.get("segment_id"),
                    "type": result["type"],
                    "params": result.get("params"),  # None for INR
                    "normalization": norm_transform,  # Store the transform used
                    "err": result["err"],
                    "all_errors": result.get("all_errors", {}),
                    "mesh": result_mesh,  # Mesh in WORLD coordinates
                    "inpoints": segment_points_world.copy(),  # World coords for clipping decisions
                }
                surface_params_list.append(surface_param)

                if fit_type == "open_spline":
                    num_freeform += 1
                else:
                    num_primitives += 1

            except Exception as e:
                pbar.set_postfix(seg=segment_id, status=f"ERROR: {e}")
                continue

        num_skipped = num_segments - num_primitives - num_freeform

        # Build surface_params output
        surface_params = {
            "surfaces": surface_params_list,
            "num_primitives": num_primitives,
            "num_freeform": num_freeform,
            "num_skipped": num_skipped,
            "total_surfaces": len(surface_params_list)
        }

        # Build surface_meshes - combine unclipped meshes with connectivity
        surface_meshes = None
        meshes_to_combine = []
        surface_id_list = []
        primitive_type_list = []

        for surf_idx, surf_param in enumerate(surface_params_list):
            mesh = surf_param.get("mesh")
            if mesh is not None and len(mesh.vertices) > 0:
                meshes_to_combine.append(mesh)
                # Track surface_id and primitive_type per face
                num_faces = len(mesh.faces)
                surface_id_list.append(np.full(num_faces, surf_idx, dtype=np.int32))
                primitive_type_list.append(np.full(num_faces, prim_map.get(surf_param["type"], 0), dtype=np.int32))

        if meshes_to_combine:
            surface_meshes = trimesh.util.concatenate(meshes_to_combine)
            if not hasattr(surface_meshes, 'face_attributes'):
                surface_meshes.face_attributes = {}
            surface_meshes.face_attributes['surface_id'] = np.concatenate(surface_id_list)
            surface_meshes.face_attributes['primitive_type'] = np.concatenate(primitive_type_list)
            surface_meshes.metadata['num_surfaces'] = len(surface_params_list)
            surface_meshes.metadata['surface_face_counts'] = [len(m.faces) for m in meshes_to_combine]

        # Build detailed summary
        summary = f"Fitted {len(surface_params_list)} surfaces:\n\n"
        summary += f"Primitives: {num_primitives}, Freeform: {num_freeform}, Skipped: {num_skipped}\n\n"

        for i, surf in enumerate(surface_params_list):
            fit_type = surf["type"]
            fit_err = surf["err"]
            all_errs = surf.get("all_errors", {})
            seg_id = surf.get("segment_id", i)
            mesh = surf.get("mesh")
            num_pts = len(mesh.vertices) if mesh is not None else 0

            # Format all errors
            err_parts = []
            for prim in ["plane", "sphere", "cylinder", "cone", "open_spline"]:
                err = all_errs.get(prim)
                if err is not None:
                    marker = " <-BEST" if prim == fit_type else ""
                    err_parts.append(f"{prim}:{err:.4f}{marker}")

            summary += f"Segment {seg_id}: {fit_type.upper()} (err={fit_err:.4f}, {num_pts} pts)\n\n"
            summary += f"  {', '.join(err_parts)}\n\n"

        print(f"[OK] Surface fitting complete: {num_primitives} primitives, {num_freeform} freeform, {num_skipped} skipped")

        return (surface_params, surface_meshes, summary)


# ============================================================================
# Node 4: Point2CADTopologyExtraction
# ============================================================================

class Point2CADTopologyExtraction:
    """
    Extract topological elements (edges, corners) via surface intersections.
    Builds adjacency matrix for B-rep construction.

    Two modes:
    - PyVista: Mesh-based intersection (like original Point2CAD)
    - OpenCASCADE: Proper B-rep surface-surface intersection
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "surface_params": ("SURFACE_PARAMS", {
                    "tooltip": "Surface parameters from Point2CADSurfaceFitting (type, analytical params, normalization)."
                }),
                "mode": (["PyVista (mesh-based)", "OpenCASCADE (B-rep)"], {
                    "default": "OpenCASCADE (B-rep)",
                    "tooltip": "PyVista: mesh intersection (like original Point2CAD). OpenCASCADE: proper B-rep geometry."
                }),
                "intersection_tolerance": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.0001,
                    "max": 0.1,
                    "step": 0.0001,
                    "display": "slider",
                    "tooltip": "Distance tolerance for detecting intersections between surfaces."
                }),
            },
            "optional": {
                "edge_pruning_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Threshold for removing short or spurious edges."
                }),
            }
        }

    RETURN_TYPES = ("BREP_TOPOLOGY", "TRIMESH", "CAD_MODEL", "STRING")
    RETURN_NAMES = ("topology", "mesh_result", "brep_result", "summary")
    FUNCTION = "extract_topology"
    CATEGORY = "CADabra/Point2CAD"

    def extract_topology(self, surface_params, mode: str, intersection_tolerance: float, edge_pruning_threshold: float = 0.01) -> Tuple:
        """
        Extract topological structure from fitted surfaces.

        Args:
            surface_params: Surface parameters (type, analytical params, normalization)
            mode: "PyVista (mesh-based)" or "OpenCASCADE (B-rep)"
            intersection_tolerance: Tolerance for surface-surface intersection
            edge_pruning_threshold: Threshold for pruning spurious edges

        Returns:
            Tuple containing (topology, mesh_result, brep_result, summary)
        """
        if surface_params is None:
            raise ValueError("No surface_params provided. Please connect a Point2CADSurfaceFitting node.")

        print(f"[Point2CAD] TopologyExtraction: {surface_params['total_surfaces']} surfaces, mode={mode}")

        surface_list = surface_params["surfaces"]
        num_surfaces = len(surface_list)

        # Build pointcloud_list and segment_points from surface_params (no external pointcloud needed)
        pointcloud_list = []
        segment_points = {}
        for i, surface in enumerate(surface_list):
            seg_id = surface.get("segment_id")

            pointcloud_list.append({
                "segment_id": seg_id,
                "type": surface.get("type"),
                "points": None,  # Will sample from surface params directly
            })

            # segment_points not needed - we sample meshes directly from surface params
            segment_points[seg_id] = None

        mesh_result = None
        brep_result = None

        if "PyVista" in mode:
            # PyVista mesh-based intersection (like original Point2CAD)
            topology, mesh_result = self._extract_topology_pyvista(
                surface_list, pointcloud_list, segment_points, intersection_tolerance, edge_pruning_threshold
            )
        else:
            # OpenCASCADE B-rep intersection
            topology, brep_result, mesh_result = self._extract_topology_occ(
                surface_list, pointcloud_list, segment_points, intersection_tolerance, edge_pruning_threshold
            )

        summary = f"Topology extracted ({mode}):\n"
        summary += f"  - Surfaces: {topology['num_surfaces']}\n"
        summary += f"  - Edges: {topology['num_edges']}\n"
        summary += f"  - Corners: {topology['num_corners']}"

        print(f"[OK] Topology extraction complete")

        return (topology, mesh_result, brep_result, summary)

    def _trimesh_to_cgal(self, mesh):
        """Convert trimesh to CGAL Polyhedron_3."""
        # Import CGAL types locally to ensure fresh state
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3 as Poly3
        from CGAL.CGAL_Kernel import Point_3 as Pt3
        from CGAL import CGAL_Polygon_mesh_processing as PMP

        points_vec = PMP.Point_3_Vector()
        polygons_vec = PMP.Polygon_Vector()

        # Get vertices and faces as numpy arrays
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        print(f"      [CGAL] Converting mesh: {len(vertices)} verts, {len(faces)} faces")
        print(f"      [CGAL] Face dtype: {faces.dtype}, shape: {faces.shape}")

        # Add vertices
        for v in vertices:
            points_vec.push_back(Pt3(float(v[0]), float(v[1]), float(v[2])))

        # Add faces - use Python lists directly (CGAL accepts them and converts to tuples)
        for f in faces:
            f_list = f.tolist()  # Convert numpy array to Python list of native ints
            polygons_vec.push_back(f_list)  # Push Python list directly

        if polygons_vec.size() == 0:
            print(f"      [CGAL] No faces to convert")
            return None

        print(f"      [CGAL] Added {points_vec.size()} points, {polygons_vec.size()} polygons")

        poly = Poly3()
        try:
            # Orient the polygon soup first (ensures consistent face orientation)
            PMP.orient_polygon_soup(points_vec, polygons_vec)
            print(f"      [CGAL] Oriented polygon soup")
            # Convert to mesh
            PMP.polygon_soup_to_polygon_mesh(points_vec, polygons_vec, poly)
            print(f"      [CGAL] Created polyhedron: {poly.size_of_vertices()} verts, {poly.size_of_facets()} faces")
        except Exception as e:
            print(f"      [CGAL] Failed to convert mesh: {e}")
            import traceback
            traceback.print_exc()
            return None

        return poly

    def _cgal_to_trimesh(self, poly):
        """Convert CGAL Polyhedron_3 to trimesh."""
        if poly is None or poly.size_of_vertices() == 0:
            return None

        # Get vertices
        vertices = []
        vertex_to_idx = {}
        for i, v in enumerate(poly.vertices()):
            pt = v.point()
            vertices.append([pt.x(), pt.y(), pt.z()])
            # Store mapping for later face lookup
            vertex_to_idx[id(v)] = i

        # Get faces by iterating halfedges
        faces = []
        for f in poly.facets():
            face_verts = []
            he = f.halfedge()
            start_he = he
            while True:
                v = he.vertex()
                pt = v.point()
                # Find matching vertex by position
                for i, vert in enumerate(vertices):
                    if (abs(vert[0] - pt.x()) < 1e-10 and
                        abs(vert[1] - pt.y()) < 1e-10 and
                        abs(vert[2] - pt.z()) < 1e-10):
                        face_verts.append(i)
                        break
                he = he.next()
                if he == start_he:
                    break
            if len(face_verts) >= 3:
                faces.append(face_verts)

        if len(vertices) == 0 or len(faces) == 0:
            return None

        return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    def _clip_meshes_with_libigl(self, trimesh_list, segment_points_list):
        """
        Clip/trim meshes at their intersections using libigl's CGAL backend.
        Similar to PyMesh approach from original Point2CAD.

        Args:
            trimesh_list: List of trimesh.Trimesh objects
            segment_points_list: List of numpy arrays with original segment points

        Returns:
            List of clipped trimesh.Trimesh objects
        """
        from collections import Counter

        try:
            import igl.copyleft.cgal as cgal
        except ImportError:
            print("   [libigl] igl.copyleft.cgal not available, skipping clipping")
            return trimesh_list

        # Filter out None meshes but track indices
        valid_indices = [i for i, m in enumerate(trimesh_list) if m is not None]
        valid_meshes = [trimesh_list[i] for i in valid_indices]

        if len(valid_meshes) < 2:
            print("   [libigl] Not enough meshes to clip, returning originals")
            return trimesh_list

        print(f"   [libigl] Processing {len(valid_meshes)} meshes...")

        try:
            # Merge all meshes and track face sources
            face_sources = []
            vertex_offsets = [0]
            for i, mesh in enumerate(valid_meshes):
                face_sources.extend([i] * len(mesh.faces))
                vertex_offsets.append(vertex_offsets[-1] + len(mesh.vertices))
            face_sources = np.array(face_sources)

            merged = trimesh.util.concatenate(valid_meshes)
            print(f"   [libigl] Merged mesh: {len(merged.vertices)} verts, {len(merged.faces)} faces")

            # Use libigl to resolve self-intersections via union with itself
            VA = np.asarray(merged.vertices, dtype=np.float64)
            FA = np.asarray(merged.faces, dtype=np.int64)

            try:
                # Self-union resolves self-intersections
                VC, FC, J = cgal.mesh_boolean(VA, FA, VA, FA, "union")
                resolved = trimesh.Trimesh(vertices=VC, faces=FC, process=False)
                print(f"   [libigl] Resolved: {len(resolved.vertices)} verts, {len(resolved.faces)} faces")
            except Exception as e:
                print(f"   [libigl] Boolean union failed: {e}, using original meshes")
                return trimesh_list

            # Separate into connected components
            components = resolved.split(only_watertight=False)
            print(f"   [libigl] Split into {len(components)} components")

            if len(components) == 0:
                return trimesh_list

            # For each original surface, find the component(s) closest to its points
            clipped_meshes = [None] * len(trimesh_list)

            for valid_idx, orig_mesh in enumerate(valid_meshes):
                orig_idx = valid_indices[valid_idx]
                seg_pts = segment_points_list[valid_idx] if valid_idx < len(segment_points_list) else None

                if seg_pts is None or len(seg_pts) == 0:
                    # No segment points, use vertices of original mesh
                    seg_pts = orig_mesh.vertices

                # Find which component(s) are closest to segment points
                best_components = []
                for comp in components:
                    closest, distances, _ = trimesh.proximity.closest_point(comp, seg_pts)
                    avg_dist = np.mean(distances)
                    best_components.append((avg_dist, comp))

                # Sort by average distance and take the closest
                best_components.sort(key=lambda x: x[0])

                if best_components:
                    # Take components within 2x the best distance
                    threshold = best_components[0][0] * 2.0 + 0.01
                    selected = [comp for dist, comp in best_components if dist < threshold]

                    if selected:
                        clipped_meshes[orig_idx] = trimesh.util.concatenate(selected) if len(selected) > 1 else selected[0]
                    else:
                        clipped_meshes[orig_idx] = trimesh_list[orig_idx]
                else:
                    clipped_meshes[orig_idx] = trimesh_list[orig_idx]

            # Fill in any remaining None entries with originals
            for i, mesh in enumerate(clipped_meshes):
                if mesh is None and i < len(trimesh_list) and trimesh_list[i] is not None:
                    clipped_meshes[i] = trimesh_list[i]

            return clipped_meshes

        except Exception as e:
            print(f"   [libigl] Error: {e}")
            import traceback
            traceback.print_exc()
            return trimesh_list

    def _extract_topology_pymesh(self, surface_list, pointcloud_list, segment_points, tolerance, pruning_threshold):
        """
        PyMesh-based topology extraction - exact replica of original Point2CAD.
        Uses pymesh.resolve_self_intersection() for proper mesh clipping.

        Reference: https://github.com/prs-eth/point2cad/blob/main/point2cad/io_utils.py
        """
        from ..utils.point2cad_fitting import sample_surface_mesh
        from collections import Counter
        import itertools
        import scipy.spatial.distance

        print(f"   [PyMesh mode] Sampling surfaces as meshes...")

        num_surfaces = len(surface_list)
        edges = []
        corners = []
        adjacency = np.zeros((num_surfaces, num_surfaces), dtype=bool)

        # Sample all surfaces as meshes
        surface_meshes = []
        out_meshes = []  # For PyMesh clipping (matches original Point2CAD structure)
        for i, surface in enumerate(surface_list):
            seg_id = surface.get("segment_id")
            seg_pts = segment_points.get(seg_id, None)

            # Use pre-computed mesh from surface fitting if available
            mesh = surface.get("mesh")
            if mesh is None:
                mesh = sample_surface_mesh(surface, seg_pts, resolution=32)

            surface_meshes.append(mesh)
            out_meshes.append({
                "mesh": mesh,
                "inpoints": surface.get("inpoints"),  # Original segment points for filtering
            })

            if mesh is not None:
                print(f"      Surface {i} ({surface.get('type')}): {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            else:
                print(f"      Surface {i} ({surface.get('type')}): no mesh")

        # =========================================================================
        # PyMesh mesh clipping - exact replica of original Point2CAD save_clipped_meshes()
        # =========================================================================
        if HAS_PYMESH:
            print(f"   [PyMesh] Clipping meshes with resolve_self_intersection...")

            # Convert trimesh to PyMesh meshes (like original save_unclipped_meshes)
            pm_meshes = []
            valid_indices = []
            for i, mesh in enumerate(surface_meshes):
                if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    pm_mesh = pymesh.form_mesh(
                        np.array(mesh.vertices, dtype=np.float64),
                        np.array(mesh.faces, dtype=np.int32)
                    )
                    pm_meshes.append(pm_mesh)
                    valid_indices.append(i)

            if len(pm_meshes) >= 2:
                # Merge all meshes (like original)
                pm_merged = pymesh.merge_meshes(pm_meshes)
                face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)

                # Resolve self-intersections (the key step!)
                pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)

                # Remove duplicated vertices
                pm_resolved, info_dict = pymesh.remove_duplicated_vertices(
                    pm_resolved_ori, tol=1e-6, importance=None
                )

                # Track face sources through resolution
                face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
                face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]

                # Convert to trimesh for connected component analysis
                tri_resolved = trimesh.Trimesh(
                    vertices=pm_resolved.vertices, faces=pm_resolved.faces
                )
                face_adjacency = tri_resolved.face_adjacency

                # Find connected components
                connected_node_labels = trimesh.graph.connected_component_labels(
                    edges=face_adjacency, node_count=len(tri_resolved.faces)
                )

                # Get unique component IDs sorted by frequency
                most_common_groupids = [
                    item[0] for item in Counter(connected_node_labels).most_common()
                ]

                # Create submeshes for each connected component
                submeshes = [
                    trimesh.Trimesh(
                        vertices=np.array(tri_resolved.vertices),
                        faces=np.array(tri_resolved.faces)[np.where(connected_node_labels == item)],
                    )
                    for item in most_common_groupids
                ]

                # Track which original mesh (pm_meshes index) each submesh came from
                indices_sources = [
                    face_sources_from_fit[connected_node_labels == item][0]
                    for item in np.array(most_common_groupids)
                ]

                print(f"   [PyMesh] Resolved into {len(submeshes)} submeshes")

                # For each original surface, select submeshes closest to its inpoints
                # (exact replica of original Point2CAD save_clipped_meshes loop)
                clipped_meshes = [None] * len(surface_meshes)

                for pm_idx, orig_idx in enumerate(valid_indices):
                    one_cluster_points = out_meshes[orig_idx]["inpoints"]

                    if one_cluster_points is None or len(one_cluster_points) == 0:
                        # Fall back to using mesh vertices
                        one_cluster_points = surface_meshes[orig_idx].vertices

                    # Get submeshes that came from this original mesh
                    submeshes_cur = [
                        x
                        for x, y in zip(submeshes, np.array(indices_sources) == pm_idx)
                        if y and len(x.faces) > 2
                    ]

                    if len(submeshes_cur) == 0:
                        clipped_meshes[orig_idx] = surface_meshes[orig_idx]
                        continue

                    # Find which submesh each inpoint is closest to
                    nearest_submesh = np.argmin(
                        np.array(
                            [
                                trimesh.proximity.closest_point(item, one_cluster_points)[1]
                                for item in submeshes_cur
                            ]
                        ).transpose(),
                        -1,
                    )

                    # Count how many inpoints are closest to each submesh
                    counter_nearest = Counter(nearest_submesh).most_common()

                    # Calculate area per point ratio for filtering
                    area_per_point = np.array(
                        [submeshes_cur[item[0]].area / item[1] for item in counter_nearest]
                    )

                    # Keep submeshes with reasonable area/point ratio (original uses multiplier_area=2)
                    multiplier_area = 2
                    nonzero_indices = np.nonzero(area_per_point)[0]
                    if len(nonzero_indices) == 0:
                        clipped_meshes[orig_idx] = surface_meshes[orig_idx]
                        continue

                    result_indices = np.array(counter_nearest)[:, 0][
                        np.logical_and(
                            area_per_point < area_per_point[nonzero_indices[0]] * multiplier_area,
                            area_per_point != 0,
                        )
                    ]

                    result_submesh_list = [submeshes_cur[item] for item in result_indices]

                    if len(result_submesh_list) > 0:
                        clipped_meshes[orig_idx] = trimesh.util.concatenate(result_submesh_list)
                    else:
                        clipped_meshes[orig_idx] = surface_meshes[orig_idx]

                clipped_count = sum(1 for m in clipped_meshes if m is not None)
                print(f"   [PyMesh] Clipped {clipped_count} meshes successfully")

            else:
                print(f"   [PyMesh] Not enough valid meshes to clip, using originals")
                clipped_meshes = surface_meshes
        else:
            print(f"   [PyMesh] Not available, using unclipped meshes")
            print(f"   Install PyMesh from: https://github.com/PozzettiAndrea/PyMesh/releases")
            clipped_meshes = surface_meshes

        # =========================================================================
        # Topology extraction - exact replica of original Point2CAD save_topology()
        # =========================================================================
        if HAS_PYVISTA:
            print(f"   Computing mesh-mesh intersections with PyVista...")

            # Wrap clipped meshes with PyVista (like original)
            valid_clipped = [(i, m) for i, m in enumerate(clipped_meshes) if m is not None]
            filtered_submeshes_pv = [pv.wrap(m) for _, m in valid_clipped]
            clipped_indices = [i for i, _ in valid_clipped]

            # Compute pairwise intersections (exact replica of save_topology)
            filtered_submeshes_pv_combinations = list(
                itertools.combinations(range(len(filtered_submeshes_pv)), 2)
            )

            intersection_curves = []
            for k, (idx_i, idx_j) in enumerate(filtered_submeshes_pv_combinations):
                pv_i = filtered_submeshes_pv[idx_i]
                pv_j = filtered_submeshes_pv[idx_j]
                orig_i = clipped_indices[idx_i]
                orig_j = clipped_indices[idx_j]

                try:
                    intersection, _, _ = pv_i.intersection(
                        pv_j, split_first=False, split_second=False, progress_bar=False
                    )

                    if intersection.n_points > 0:
                        edge_points = np.array(intersection.points)
                        edge_lines = None
                        if intersection.lines is not None and len(intersection.lines) > 0:
                            edge_lines = intersection.lines.reshape(-1, 3)[:, 1:].tolist()

                        edges.append({
                            "surfaces": (orig_i, orig_j),
                            "segment_ids": (surface_list[orig_i].get("segment_id"),
                                           surface_list[orig_j].get("segment_id")),
                            "points": edge_points,
                            "lines": edge_lines,
                            "type": "intersection_curve"
                        })
                        adjacency[orig_i, orig_j] = True
                        adjacency[orig_j, orig_i] = True

                        # Also store for corner detection (like original)
                        intersection_curves.append({
                            "pv_points": edge_points,
                            "pv_lines": edge_lines
                        })

                except Exception as e:
                    pass

            print(f"   Found {len(edges)} edges")

            # Compute corners from edge-edge intersections (exact replica of save_topology)
            print(f"   Computing corners from edge intersections...")

            intersection_curves_combinations_indices = list(
                itertools.combinations(range(len(intersection_curves)), 2)
            )

            for combination_indices in intersection_curves_combinations_indices:
                sample0 = np.array(intersection_curves[combination_indices[0]]["pv_points"])
                sample1 = np.array(intersection_curves[combination_indices[1]]["pv_points"])

                if len(sample0) == 0 or len(sample1) == 0:
                    continue

                dists = scipy.spatial.distance.cdist(sample0, sample1)
                # Original uses dists == 0, we use tolerance for robustness
                row_indices, col_indices = np.where(dists < tolerance)

                if len(row_indices) > 0 and len(col_indices) > 0:
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        corner_point = (sample0[row_idx] + sample1[col_idx]) / 2
                        corners.append({
                            "point": corner_point,
                            "edges": combination_indices,
                            "surfaces": list(set(edges[combination_indices[0]]["surfaces"]) |
                                           set(edges[combination_indices[1]]["surfaces"]))
                        })

            print(f"   Found {len(corners)} corners")

        topology = {
            "surfaces": surface_list,
            "edges": edges,
            "corners": corners,
            "adjacency": adjacency,
            "num_surfaces": num_surfaces,
            "num_edges": len(edges),
            "num_corners": len(corners)
        }

        # Combine clipped meshes for visualization
        mesh_result = None
        valid_meshes = [m for m in clipped_meshes if m is not None]
        if valid_meshes:
            mesh_result = trimesh.util.concatenate(valid_meshes)
            print(f"   Combined mesh: {len(mesh_result.vertices)} verts, {len(mesh_result.faces)} faces")

        return topology, mesh_result

    # Keep old name as alias for backwards compatibility
    _extract_topology_pyvista = _extract_topology_pymesh

    def _extract_topology_occ(self, surface_list, pointcloud_list, segment_points, tolerance, pruning_threshold):
        """OpenCASCADE B-rep topology extraction."""
        print(f"   [OpenCASCADE mode] Creating B-rep surfaces...")

        num_surfaces = len(surface_list)
        edges = []
        corners = []
        adjacency = np.zeros((num_surfaces, num_surfaces), dtype=bool)

        try:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
            from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Ax3, gp_Ax2
            from OCC.Core.Geom import Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface, Geom_ConicalSurface
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve

            # Create OCC faces for each surface
            occ_faces = []
            for i, surface in enumerate(surface_list):
                # Get corresponding pointcloud
                pointcloud = pointcloud_list[i] if i < len(pointcloud_list) else None
                face = self._create_occ_face(surface, pointcloud, segment_points)
                occ_faces.append(face)
                if face is not None:
                    print(f"      Surface {i} ({surface.get('type')}): OCC face created")
                else:
                    print(f"      Surface {i} ({surface.get('type')}): failed to create OCC face")

            # Compute surface-surface intersections
            print(f"   Computing surface-surface intersections...")
            for i in range(num_surfaces):
                for j in range(i + 1, num_surfaces):
                    face_i = occ_faces[i]
                    face_j = occ_faces[j]

                    if face_i is None or face_j is None:
                        continue

                    try:
                        # Compute intersection using BRepAlgoAPI_Section
                        section = BRepAlgoAPI_Section(face_i, face_j)
                        section.Build()

                        if section.IsDone():
                            section_shape = section.Shape()

                            # Extract edge points
                            edge_points = []
                            explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
                            while explorer.More():
                                edge = explorer.Current()
                                # Sample points along edge
                                curve, first, last = BRep_Tool.Curve(edge)
                                if curve is not None:
                                    for t in np.linspace(first, last, 20):
                                        pt = curve.Value(t)
                                        edge_points.append([pt.X(), pt.Y(), pt.Z()])
                                explorer.Next()

                            if len(edge_points) > 0:
                                edge_points = np.array(edge_points)
                                edges.append({
                                    "surfaces": (i, j),
                                    "segment_ids": (surface_list[i].get("segment_id"),
                                                   surface_list[j].get("segment_id")),
                                    "points": edge_points,
                                    "type": "intersection_curve"
                                })
                                adjacency[i, j] = True
                                adjacency[j, i] = True

                    except Exception as e:
                        print(f"      Intersection {i}-{j} failed: {e}")

            print(f"   Found {len(edges)} edges")

            # Create compound of all faces
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for face in occ_faces:
                if face is not None:
                    builder.Add(compound, face)

            brep_result = {
                "occ_shape": compound,
                "format": "occ",
            }

        except ImportError as e:
            print(f"   [WARN] OpenCASCADE not available: {e}")
            brep_result = None
        except Exception as e:
            print(f"   [ERROR] OpenCASCADE topology extraction failed: {e}")
            import traceback
            traceback.print_exc()
            brep_result = None

        topology = {
            "surfaces": surface_list,
            "edges": edges,
            "corners": corners,
            "adjacency": adjacency,
            "num_surfaces": num_surfaces,
            "num_edges": len(edges),
            "num_corners": len(corners)
        }

        # Also generate a mesh for visualization
        from ..utils.point2cad_fitting import sample_surface_mesh
        all_meshes = []
        for i, surface in enumerate(surface_list):
            mesh = sample_surface_mesh(surface, None, resolution=32)
            if mesh is not None:
                # Add surface_id attribute
                if not hasattr(mesh, 'vertex_attributes'):
                    mesh.vertex_attributes = {}
                mesh.vertex_attributes['surface_id'] = np.full(len(mesh.vertices), i, dtype=np.int32)
                all_meshes.append(mesh)

        mesh_result = None
        if all_meshes:
            mesh_result = trimesh.util.concatenate(all_meshes)
            print(f"   Combined mesh: {len(mesh_result.vertices)} verts, {len(mesh_result.faces)} faces")

        return topology, brep_result, mesh_result

    def _create_occ_face(self, surface, pointcloud, segment_points):
        """Create an OCC face from surface parameters."""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Ax3, gp_Ax2
            from OCC.Core.Geom import Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface

            surf_type = surface.get("type")
            params = surface.get("params")
            normalization = surface.get("normalization", {})
            offset = np.array(normalization.get("offset", [0, 0, 0]))
            scale = float(normalization.get("scale", 1.0))

            # For open_spline (INR), use pre-sampled mesh from surface params
            if surf_type == "open_spline":
                mesh = surface.get("mesh")
                if mesh is not None and hasattr(mesh, 'vertices'):
                    return self._create_bspline_from_mesh(mesh)
                else:
                    # Fallback to pointcloud if available
                    return self._create_bspline_from_pointcloud(pointcloud)

            if params is None:
                return None

            # Determine extent from segment points
            seg_id = surface.get("segment_id")
            seg_pts = segment_points.get(seg_id)
            if seg_pts is not None and len(seg_pts) > 0:
                extent = np.max(np.ptp(seg_pts, axis=0)) * 1.5
            else:
                extent = scale * 2

            if surf_type == "plane":
                normal, distance = params
                normal = np.array(normal).flatten()
                distance = float(distance)

                # Transform to world coordinates
                point_norm = normal * distance
                point_world = point_norm * scale + offset

                pln = gp_Pln(
                    gp_Pnt(float(point_world[0]), float(point_world[1]), float(point_world[2])),
                    gp_Dir(float(normal[0]), float(normal[1]), float(normal[2]))
                )
                face = BRepBuilderAPI_MakeFace(pln, float(-extent), float(extent), float(-extent), float(extent)).Face()
                return face

            elif surf_type == "sphere":
                center, radius = params
                center = np.array(center).flatten()
                radius = float(radius)

                # Transform to world coordinates
                center_world = center * scale + offset
                radius_world = radius * scale

                sph = Geom_SphericalSurface(
                    gp_Ax3(gp_Ax2(
                        gp_Pnt(float(center_world[0]), float(center_world[1]), float(center_world[2])),
                        gp_Dir(0, 0, 1)
                    )),
                    float(radius_world)
                )
                face = BRepBuilderAPI_MakeFace(sph, 1e-6).Face()
                return face

            elif surf_type == "cylinder":
                axis, center, radius = params
                axis = np.array(axis).flatten()
                center = np.array(center).flatten()
                radius = float(radius)

                # Transform to world coordinates
                center_world = center * scale + offset
                radius_world = radius * scale

                cyl = Geom_CylindricalSurface(
                    gp_Ax3(gp_Ax2(
                        gp_Pnt(float(center_world[0]), float(center_world[1]), float(center_world[2])),
                        gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
                    )),
                    float(radius_world)
                )
                face = BRepBuilderAPI_MakeFace(cyl, 0.0, 6.283185307179586, float(-extent), float(extent), 1e-6).Face()
                return face

            elif surf_type == "cone":
                # Cone is more complex, skip for now
                return None

        except Exception as e:
            print(f"      Failed to create OCC face for {surface.get('type')}: {e}")
            return None

        return None

    def _create_bspline_from_mesh(self, mesh):
        """Convert INR mesh to OCC B-spline face. Mesh vertices must be a regular NxN grid."""
        try:
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCC.Core.GeomAbs import GeomAbs_C2
            from OCC.Core.TColgp import TColgp_Array2OfPnt
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

            points = np.array(mesh.vertices)
            num_pts = len(points)

            # Determine grid size (should be square from sample_inr_mesh - 50x50)
            mesh_dim = int(np.sqrt(num_pts))
            if mesh_dim * mesh_dim != num_pts:
                print(f"        Mesh ({num_pts} verts) not a square grid, cannot fit B-spline")
                return None

            # Reshape points to grid
            pts_grid = points.reshape(mesh_dim, mesh_dim, 3).transpose(1, 0, 2)

            # Build OCC point array (1-indexed!)
            point_array = TColgp_Array2OfPnt(1, mesh_dim, 1, mesh_dim)
            for i in range(mesh_dim):
                for j in range(mesh_dim):
                    pt = pts_grid[i, j]
                    point_array.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

            # Fit B-spline (degree 3, C2 continuity)
            fitter = GeomAPI_PointsToBSplineSurface(point_array, 3, 3, GeomAbs_C2, 1e-3)
            if not fitter.IsDone():
                print(f"        B-spline fitting failed")
                return None

            # Create face from B-spline surface
            face_maker = BRepBuilderAPI_MakeFace(fitter.Surface(), 1e-6)
            if not face_maker.IsDone():
                print(f"        Face creation failed for B-spline")
                return None

            return face_maker.Face()

        except Exception as e:
            print(f"        B-spline from mesh failed: {e}")
            return None

    def _create_bspline_from_pointcloud(self, pointcloud):
        """Convert pointcloud to OCC B-spline face. Points must be a regular NxN grid."""
        try:
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCC.Core.GeomAbs import GeomAbs_C2
            from OCC.Core.TColgp import TColgp_Array2OfPnt
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

            if pointcloud is None:
                print(f"        No pointcloud for B-spline fitting")
                return None

            # Get points from pointcloud dict
            points = pointcloud.get("points")
            if points is None or len(points) == 0:
                print(f"        Empty pointcloud for B-spline fitting")
                return None

            num_pts = len(points)

            # Determine grid size (should be square from sample_inr_mesh - 50x50)
            mesh_dim = int(np.sqrt(num_pts))
            if mesh_dim * mesh_dim != num_pts:
                print(f"        Pointcloud ({num_pts} pts) not a square grid, cannot fit B-spline")
                return None

            # Reshape points to grid
            # NOTE: sample_inr_mesh uses indexing="xy" in meshgrid, which means:
            #   - rows (i) correspond to v-direction
            #   - cols (j) correspond to u-direction
            # OCC B-spline expects first index = U, second = V
            # So we transpose to get correct UV ordering
            pts_grid = points.reshape(mesh_dim, mesh_dim, 3).transpose(1, 0, 2)

            # Build OCC point array (1-indexed!)
            point_array = TColgp_Array2OfPnt(1, mesh_dim, 1, mesh_dim)
            for i in range(mesh_dim):
                for j in range(mesh_dim):
                    pt = pts_grid[i, j]
                    point_array.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

            # Fit B-spline (degree 3, C2 continuity)
            fitter = GeomAPI_PointsToBSplineSurface(point_array, 3, 3, GeomAbs_C2, 1e-3)
            if not fitter.IsDone():
                print(f"        B-spline fitting failed")
                return None

            # Create face from B-spline surface
            face_maker = BRepBuilderAPI_MakeFace(fitter.Surface(), 1e-6)
            if not face_maker.IsDone():
                print(f"        Face creation failed for B-spline")
                return None

            print(f"        Created B-spline surface ({mesh_dim}x{mesh_dim} pts)")
            return face_maker.Face()

        except Exception as e:
            print(f"        Failed to create B-spline from INR: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Node 5: Point2CADExportBrep
# ============================================================================

class Point2CADExportBrep:
    """
    Export B-rep topology to STEP file format.
    Generates preview mesh for visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "topology": ("BREP_TOPOLOGY", {
                    "tooltip": "B-rep topology structure from Point2CADTopologyExtraction."
                }),
                "output_filename": ("STRING", {
                    "default": "reconstructed_cad.step",
                    "multiline": False,
                    "tooltip": "Name of the output STEP file (e.g., model.step)."
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "output",
                    "multiline": False,
                    "tooltip": "Directory where the STEP file will be saved."
                }),
                "generate_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate a preview mesh of the exported CAD model."
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "TRIMESH", "STRING")
    RETURN_NAMES = ("cad_model", "preview_mesh", "summary")
    FUNCTION = "export_brep"
    CATEGORY = "CADabra/Point2CAD"
    OUTPUT_NODE = True

    def export_brep(self, topology, output_filename: str, output_dir: str = "output", generate_preview: bool = True) -> Tuple:
        """
        Export B-rep to STEP format.
        """
        if topology is None:
            raise ValueError("No topology provided. Please connect a Point2CADTopologyExtraction node.")

        print(f"[Point2CAD] Point2CADExportBrep: Exporting {topology['num_surfaces']} surfaces to STEP")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        step_file = output_path / output_filename
        step_file_str = str(step_file)
        print(f"   Output: {step_file}")

        # Write STEP file using pythonocc
        try:
            from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Ax3, gp_Ax2, gp_Circ
            from OCC.Core.Geom import Geom_BSplineSurface, Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface
            from OCC.Core.TColgp import TColgp_Array2OfPnt
            from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCC.Core.IFSelect import IFSelect_RetDone

            print(f"   [DEBUG] Initializing STEP writer...")
            writer = STEPControl_Writer()

            # Create a compound to hold all vertices and geometry
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)

            # Add a tiny dummy box to ensure the model is never empty
            try:
                dummy_box = BRepPrimAPI_MakeBox(0.001, 0.001, 0.001).Shape()
                builder.Add(compound, dummy_box)
            except Exception as e:
                print(f"   [DEBUG] Failed to create dummy box: {e}")

            total_points = 0
            total_faces = 0
            
            print(f"   Creating STEP geometry from {len(topology['surfaces'])} surfaces...")
            
            for surface in topology["surfaces"]:
                # Export Fitted Surface (Face) using new structure
                surf_type = surface.get("type", "unknown")
                params = surface.get("params")
                normalization = surface.get("normalization", {})
                offset = normalization.get("offset", np.zeros(3))
                scale = normalization.get("scale", 1.0)

                face = None

                try:
                    if surf_type == "plane" and params is not None:
                        # Plane: params = (normal, distance) in normalized space
                        normal, distance = params
                        normal = np.array(normal)
                        distance = float(distance)

                        # Transform to world coords
                        # Point on plane in normalized space: normal * distance
                        point_norm = normal * distance
                        point_world = point_norm * scale + offset

                        pln = gp_Pln(
                            gp_Pnt(float(point_world[0]), float(point_world[1]), float(point_world[2])),
                            gp_Dir(float(normal[0]), float(normal[1]), float(normal[2]))
                        )
                        # Create bounded plane (2x scale extent)
                        extent = float(scale * 2)
                        face = BRepBuilderAPI_MakeFace(pln, -extent, extent, -extent, extent).Face()

                    elif surf_type == "sphere" and params is not None:
                        # Sphere: params = (center, radius) in normalized space
                        center, radius = params
                        center = np.array(center)
                        radius = float(radius)

                        # Transform to world coords
                        center_world = center * scale + offset
                        radius_world = radius * scale

                        sph = Geom_SphericalSurface(
                            gp_Ax3(gp_Ax2(
                                gp_Pnt(float(center_world[0]), float(center_world[1]), float(center_world[2])),
                                gp_Dir(0, 0, 1)
                            )),
                            float(radius_world)
                        )
                        face = BRepBuilderAPI_MakeFace(sph, 1e-6).Face()

                    elif surf_type == "cylinder" and params is not None:
                        # Cylinder: params = (axis, center, radius) in normalized space
                        axis, center, radius = params
                        axis = np.array(axis)
                        center = np.array(center)
                        radius = float(radius)

                        # Transform to world coords
                        center_world = center * scale + offset
                        radius_world = radius * scale

                        from OCC.Core.Geom import Geom_CylindricalSurface
                        cyl = Geom_CylindricalSurface(
                            gp_Ax3(gp_Ax2(
                                gp_Pnt(float(center_world[0]), float(center_world[1]), float(center_world[2])),
                                gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
                            )),
                            float(radius_world)
                        )
                        # Create bounded cylinder
                        extent = float(scale * 2)
                        face = BRepBuilderAPI_MakeFace(cyl, 0.0, 6.283185307179586, -extent, extent, 1e-6).Face()

                    elif surf_type == "cone" and params is not None:
                        # Cone: params = (apex, axis, theta) in normalized space
                        apex, axis, theta = params
                        apex = np.array(apex)
                        axis = np.array(axis)
                        theta = float(theta)

                        # Transform to world coords
                        apex_world = apex * scale + offset

                        from OCC.Core.Geom import Geom_ConicalSurface
                        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone
                        # Create cone using BRepPrimAPI for bounded cone
                        # TODO: Proper cone creation with apex and angle
                        print(f"   [WARN] Cone export not fully implemented yet")

                    elif surf_type == "open_spline":
                        # INR spline: use pre-sampled mesh if available
                        mesh = surface.get("mesh")
                        if mesh is not None:
                            # TODO: Convert trimesh to OCC BSpline surface
                            print(f"   [WARN] INR spline export not implemented yet")

                    if face and not face.IsNull():
                        builder.Add(compound, face)
                        total_faces += 1

                except Exception as e:
                    print(f"   [WARN] Failed to create B-Rep for surface {surf_type}: {e}")

            print(f"   [DEBUG] Added {total_points} vertices and {total_faces} faces to compound.")

            # Transfer
            print(f"   [DEBUG] Transferring compound to STEP writer...")
            transfer_status = writer.Transfer(compound, STEPControl_AsIs)
            print(f"   [DEBUG] Transfer status: {transfer_status} (1=Done)")
            
            # Write
            print(f"   [DEBUG] Writing STEP file to {step_file_str}...")
            status = writer.Write(step_file_str)
            print(f"   [DEBUG] Write status: {status} (1=Done)")

            if status == IFSelect_RetDone:
                print(f"[OK] STEP file written: {total_faces} surfaces, {total_points} vertices exported")
            else:
                raise RuntimeError(f"STEP writer failed with status {status}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to write STEP file: {e}")

        # Create CAD_MODEL for PreviewCADOCC
        cad_model = None
        try:
            import gmsh
            
            # Initialize Gmsh if needed
            if not gmsh.is_initialized():
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                
            # Load the generated STEP file into Gmsh
            gmsh.model.remove()
            gmsh.model.add("point2cad_export")
            gmsh.model.occ.importShapes(step_file_str)
            gmsh.model.occ.synchronize()
            
            # Load into OCC for shape access
            occ_shape = None
            try:
                from OCC.Core.STEPControl import STEPControl_Reader
                from OCC.Core.IFSelect import IFSelect_RetDone
                
                reader = STEPControl_Reader()
                status = reader.ReadFile(step_file_str)
                if status == IFSelect_RetDone:
                    reader.TransferRoots()
                    occ_shape = reader.OneShape()
            except Exception as e:
                print(f"[WARN] Could not load into OCC: {e}")
                
            cad_model = {
                "file_path": step_file_str,
                "occ_shape": occ_shape,
                "format": ".step",
                "model_name": "point2cad_export"
            }
            
        except Exception as e:
            print(f"[WARN] Failed to create CAD_MODEL: {e}")

        # Generate preview mesh
        preview_mesh = None
        if generate_preview:
            print(f"   Generating preview mesh...")
            preview_mesh = self._generate_preview_mesh(topology)
            print(f"[OK] Preview mesh generated")

        summary = f"B-rep exported successfully:\n"
        summary += f"  File: {step_file}\n"
        summary += f"  Surfaces: {topology['num_surfaces']}\n"
        summary += f"  Edges: {topology['num_edges']}\n"
        summary += f"  Corners: {topology['num_corners']}"

        print(f"[OK] Export complete")

        return (cad_model, preview_mesh, summary)

    def _generate_preview_mesh(self, topology):
        """Generate a triangle mesh for visualization from INR meshes."""
        print(f"      Generating preview from {len(topology['surfaces'])} surfaces...")

        # Combine INR meshes for preview (primitives need sampling functions)
        all_meshes = []

        for surface in topology["surfaces"]:
            surf_type = surface.get("type", "unknown")

            if surf_type == "open_spline":
                # INR spline: use pre-sampled mesh if available
                mesh = surface.get("mesh")
                if mesh is not None:
                    all_meshes.append(mesh)
            else:
                # Primitives: TODO - use primitive sampling functions
                # For now, skip primitives in preview
                pass

        if all_meshes:
            # Combine all meshes
            combined = trimesh.util.concatenate(all_meshes)
            print(f"      Preview mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
            return combined
        else:
            print(f"      [WARN] No meshes available for preview (primitives need sampling)")
            return None


# ============================================================================
# Save/Load Surface Params for Offline Debugging
# ============================================================================

class SaveSurfaceParams:
    """Save SURFACE_PARAMS to disk for offline debugging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "surface_params": ("SURFACE_PARAMS",),
                "filename": ("STRING", {"default": "surface_params"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "CADabra/Point2CAD"
    OUTPUT_NODE = True

    def save(self, surface_params, filename):
        import pickle
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        base_name = filename.replace(".pkl", "").replace(".pickle", "")
        pkl_path = os.path.join(output_dir, f"{base_name}.pkl")

        with open(pkl_path, 'wb') as f:
            pickle.dump(surface_params, f)

        num_surfaces = len(surface_params.get("surfaces", []))
        print(f"[SaveSurfaceParams] Saved {num_surfaces} surfaces to {pkl_path}")

        return (pkl_path,)


class LoadSurfaceParams:
    """Load SURFACE_PARAMS from disk for offline debugging."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        # List all .pkl files in output directory
        pkl_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.pkl'):
                    pkl_files.append(f)
        if not pkl_files:
            pkl_files = ["surface_params.pkl"]
        return {
            "required": {
                "filename": (sorted(pkl_files), {"default": pkl_files[0] if pkl_files else "surface_params.pkl"}),
            },
        }

    RETURN_TYPES = ("SURFACE_PARAMS", "STRING")
    RETURN_NAMES = ("surface_params", "summary")
    FUNCTION = "load"
    CATEGORY = "CADabra/Point2CAD"

    @classmethod
    def IS_CHANGED(cls, filename):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        pkl_path = os.path.join(output_dir, filename)
        if os.path.exists(pkl_path):
            return os.path.getmtime(pkl_path)
        return float('nan')

    def load(self, filename):
        import pickle
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        pkl_path = os.path.join(output_dir, filename)

        with open(pkl_path, 'rb') as f:
            surface_params = pickle.load(f)

        num_surfaces = len(surface_params.get("surfaces", []))
        num_primitives = surface_params.get("num_primitives", 0)
        num_freeform = surface_params.get("num_freeform", 0)

        summary = f"Loaded {num_surfaces} surfaces from {filename}\n"
        summary += f"  Primitives: {num_primitives}\n"
        summary += f"  Freeform: {num_freeform}\n"
        for i, surf in enumerate(surface_params.get("surfaces", [])):
            summary += f"  Surface {i}: {surf.get('type')} (err={surf.get('err', 0):.4f})\n"

        print(f"[LoadSurfaceParams] {summary}")

        return (surface_params, summary)


# ============================================================================
# Node 6: Point2CADSurfaceFittingOCC
# ============================================================================

class Point2CADSurfaceFittingOCC:
    """
    Surface fitting with OCC/NURBS output for CAD export.

    Uses geomdl or nurbsdiff for B-spline fitting:
    - geomdl: Lightweight, uses INR encoder for UV parameterization
    - nurbsdiff: Direct Chamfer distance optimization (requires pytorch3d)

    Outputs proper OCC geometry that can be exported to STEP/IGES.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_cloud": ("TRIMESH", {
                    "tooltip": "Segmented point cloud from Point2CADSegmentation."
                }),
            },
            "optional": {
                "ctrl_pts_u": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of B-spline control points in U direction."
                }),
                "ctrl_pts_v": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of B-spline control points in V direction."
                }),
                "nurbs_degree": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 5,
                    "step": 1,
                    "tooltip": "B-spline degree (2=quadratic, 3=cubic, etc.)."
                }),
                "spline_backend": (["auto", "geomdl", "nurbsdiff"], {
                    "default": "auto",
                    "tooltip": "Backend for B-spline fitting. 'auto' uses nurbsdiff if available, else geomdl."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for fitting. CUDA is much faster."
                }),
            }
        }

    RETURN_TYPES = ("OCC_COMPOUND", "CAD_MODEL", "SURFACE_PARAMS", "STRING")
    RETURN_NAMES = ("occ_compound", "cad_model", "surface_params", "summary")
    FUNCTION = "fit_surfaces_occ"
    CATEGORY = "CADabra/Point2CAD"

    def _normalize_points(self, points: np.ndarray):
        """Normalize points to unit box centered at origin."""
        EPS = np.finfo(np.float32).eps
        mean = np.mean(points, 0)
        points_centered = points - mean
        bbox_extent = np.max(points_centered, 0) - np.min(points_centered, 0)
        scale = np.max(bbox_extent) + EPS
        points_normalized = points_centered / scale
        transform = {
            'mean': mean.astype(np.float32),
            'scale': float(scale),
        }
        return points_normalized.astype(np.float32), transform

    def _denormalize_mesh(self, mesh, transform):
        """Transform mesh from normalized space back to world coordinates."""
        if mesh is None:
            return None
        mesh.vertices = mesh.vertices * transform['scale'] + transform['mean']
        return mesh

    def fit_surfaces_occ(self, segmented_cloud, ctrl_pts_u: int = 8, ctrl_pts_v: int = 8,
                         nurbs_degree: int = 3, spline_backend: str = "auto",
                         device: str = "cuda") -> Tuple:
        """
        Fit surfaces and output OCC geometry.
        """
        if segmented_cloud is None:
            raise ValueError("No segmented cloud provided.")

        # Import fitting modules
        from ..utils.point2cad_fitting import process_one_surface
        from ..utils.point2cad_fitting.primitives import (
            fit_plane_numpy, fit_sphere_numpy, fitcylinder, fitcone
        )
        from ..utils.point2cad_fitting.bspline_fitting import (
            fit_bspline_surface, sample_bspline_mesh, sample_nurbsdiff_mesh,
            HAS_GEOMDL, HAS_NURBSDIFF
        )
        from ..utils.occ_conversion import (
            plane_to_occ, sphere_to_occ, cylinder_to_occ, cone_to_occ,
            geomdl_to_occ_bspline, nurbsdiff_to_occ_bspline,
            build_compound, compute_surface_bounds_from_points,
            mesh_boundary_to_wire, HAS_OCC
        )

        if not HAS_OCC:
            raise ImportError("OpenCASCADE not available. Install pythonocc-core.")

        if not HAS_GEOMDL and not HAS_NURBSDIFF:
            raise ImportError("Neither geomdl nor nurbsdiff available for B-spline fitting.")

        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, using CPU")
            device = "cpu"

        # Extract data
        points_world = np.array(segmented_cloud.vertices, dtype=np.float32)
        labels = segmented_cloud.vertex_attributes.get('label', np.zeros(len(points_world), dtype=np.int32))
        num_segments = segmented_cloud.metadata.get('num_segments', len(np.unique(labels)))

        print(f"[Point2CAD OCC] Surface Fitting: {num_segments} segments")
        print(f"   Backend: {spline_backend}, Control pts: {ctrl_pts_u}x{ctrl_pts_v}, Degree: {nurbs_degree}")

        # Normalize points (isotropic scale to unit box)
        points_normalized, norm_transform = self._normalize_points(points_world)

        surface_params_list = []
        occ_surfaces = []
        occ_bounds = []  # UV bounds for each surface (None for B-splines)
        occ_boundary_wires = []  # Boundary wires for trimmed faces

        unique_labels = [l for l in np.unique(labels) if l >= 0]
        pbar = tqdm(unique_labels, desc="OCC Surface Fitting", unit="seg")

        for segment_id in pbar:
            segment_mask = labels == segment_id
            segment_points_world = points_world[segment_mask]
            segment_points_normalized = points_normalized[segment_mask]

            if len(segment_points_normalized) < 20:
                pbar.set_postfix(seg=segment_id, status="SKIP")
                continue

            pbar.set_postfix(seg=segment_id, status="fitting...")

            try:
                # Use process_one_surface for consistent fitting logic with non-OCC version
                result = process_one_surface(
                    segment_points_normalized,
                    segment_id=segment_id,
                    device=device,
                    progress_bar=False
                )

                if result is None:
                    pbar.set_postfix(seg=segment_id, status="FAIL")
                    continue

                fit_type = result["type"]
                fit_err = result["err"]
                all_errors = result.get("all_errors", {})

                # Print detailed errors
                err_strs = []
                for prim, err in all_errors.items():
                    if err is not None:
                        marker = "*" if prim == fit_type else " "
                        err_strs.append(f"{prim}={err:.4f}{marker}")
                tqdm.write(f"   Seg {segment_id}: {', '.join(err_strs)} -> {fit_type.upper()}")

                # Fit on world coordinates using numpy/scipy (no torch after type selection)
                occ_surf = None
                surf_bounds = None
                params = None
                boundary_wire = None  # For trimmed faces (only used for B-splines)

                if fit_type == "plane":
                    normal, distance = fit_plane_numpy(segment_points_world)
                    center_world = np.mean(segment_points_world, axis=0)
                    occ_surf = plane_to_occ(normal, distance, center_world)
                    params = (normal.tolist(), float(distance))
                    surf_bounds = compute_surface_bounds_from_points(occ_surf, segment_points_world)

                elif fit_type == "sphere":
                    center, radius = fit_sphere_numpy(segment_points_world)
                    occ_surf = sphere_to_occ(center, radius)
                    params = (center.tolist(), float(radius))
                    surf_bounds = compute_surface_bounds_from_points(occ_surf, segment_points_world)

                elif fit_type == "cylinder":
                    axis, center, radius, _ = fitcylinder(segment_points_world)
                    occ_surf = cylinder_to_occ(axis, center, radius)
                    params = (axis.tolist(), center.tolist(), float(radius))
                    surf_bounds = compute_surface_bounds_from_points(occ_surf, segment_points_world)

                elif fit_type == "cone":
                    apex, axis, theta, _, failure = fitcone(segment_points_world)
                    if not failure and apex is not None:
                        occ_surf = cone_to_occ(apex, axis, theta)
                        params = (apex.tolist(), axis.tolist(), float(theta))
                        surf_bounds = compute_surface_bounds_from_points(occ_surf, segment_points_world)

                elif fit_type == "open_spline":
                    # Fit B-spline directly with OCC on world coordinates
                    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
                    from OCC.Core.TColgp import TColgp_Array2OfPnt
                    from OCC.Core.gp import gp_Pnt
                    from OCC.Core.GeomAbs import GeomAbs_C2

                    # Grid the points for B-spline fitting
                    # Simple approach: create a grid from the point cloud
                    pts = segment_points_world
                    n_pts = len(pts)
                    grid_size = min(int(np.sqrt(n_pts)), 20)  # Max 20x20 grid

                    # Project points to a 2D parameterization using PCA
                    centroid = pts.mean(axis=0)
                    pts_centered = pts - centroid
                    cov = np.cov(pts_centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Sort by eigenvalue descending
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvectors = eigenvectors[:, idx]

                    # Project to 2D (use two largest principal components)
                    uv = pts_centered @ eigenvectors[:, :2]
                    u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
                    v_min, v_max = uv[:, 1].min(), uv[:, 1].max()

                    # Create grid
                    u_grid = np.linspace(u_min, u_max, grid_size)
                    v_grid = np.linspace(v_min, v_max, grid_size)

                    # For each grid cell, find nearest point
                    from scipy.interpolate import griddata
                    grid_u, grid_v = np.meshgrid(u_grid, v_grid)
                    grid_points = griddata(uv, pts, (grid_u, grid_v), method='linear')

                    # Save valid mask BEFORE NaN filling (for boundary extraction)
                    valid_mask = ~np.isnan(grid_points[:, :, 0])

                    # Fill NaN with nearest neighbor (OCC needs complete grid)
                    nan_mask = ~valid_mask
                    if nan_mask.any():
                        grid_points_nearest = griddata(uv, pts, (grid_u, grid_v), method='nearest')
                        grid_points[nan_mask] = grid_points_nearest[nan_mask]

                    # Create OCC point array
                    occ_points = TColgp_Array2OfPnt(1, grid_size, 1, grid_size)
                    for i in range(grid_size):
                        for j in range(grid_size):
                            pt = grid_points[j, i]  # Note: griddata returns [v, u] order
                            occ_points.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

                    # Fit B-spline surface
                    try:
                        approx = GeomAPI_PointsToBSplineSurface(occ_points, 3, 8, GeomAbs_C2, 0.1)
                        if approx.IsDone():
                            occ_surf = approx.Surface()
                        else:
                            occ_surf = None
                            print(f"    B-spline fitting failed")
                    except Exception as e:
                        occ_surf = None
                        print(f"    B-spline fitting error: {e}")

                    # Try to extract boundary from grid mesh
                    boundary_wire = None
                    try:
                        import trimesh
                        # Create mesh from grid - ONLY where all 4 corners have valid data
                        # This avoids degenerate triangles from NaN-filled regions
                        vertices = grid_points.reshape(-1, 3)
                        faces = []
                        for i in range(grid_size - 1):
                            for j in range(grid_size - 1):
                                # Check if all 4 corners are valid (not NaN-filled)
                                if valid_mask[j, i] and valid_mask[j, i+1] and valid_mask[j+1, i] and valid_mask[j+1, i+1]:
                                    v0 = j * grid_size + i
                                    v1 = v0 + 1
                                    v2 = v0 + grid_size
                                    v3 = v2 + 1
                                    faces.append([v0, v1, v2])
                                    faces.append([v1, v3, v2])

                        if len(faces) > 0:
                            grid_mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
                            grid_mesh.remove_unreferenced_vertices()
                            boundary_wire = mesh_boundary_to_wire(grid_mesh)
                            if boundary_wire:
                                tqdm.write(f"   Seg {segment_id}: extracted boundary wire ({len(grid_mesh.vertices)} verts)")
                    except Exception as e:
                        tqdm.write(f"   Seg {segment_id}: boundary extraction failed: {e}")

                    # Fallback: if no wire, use UV bounds from B-spline surface
                    if boundary_wire is None and occ_surf is not None:
                        u1, u2, v1, v2 = occ_surf.Bounds()
                        surf_bounds = (u1, u2, v1, v2)
                        tqdm.write(f"   Seg {segment_id}: using UV bounds [{u1:.2f},{u2:.2f}]x[{v1:.2f},{v2:.2f}]")
                    else:
                        surf_bounds = None  # Wire will be used instead

                if occ_surf is not None:
                    occ_surfaces.append(occ_surf)
                    occ_bounds.append(surf_bounds)
                    occ_boundary_wires.append(boundary_wire)  # None for primitives, wire for B-splines

                surface_params_list.append({
                    "segment_id": segment_id,
                    "type": fit_type,
                    "params": params,
                    "err": float(fit_err),
                    "inpoints": segment_points_world.copy(),
                })
                pbar.set_postfix(seg=segment_id, result=fit_type.upper())

            except Exception as e:
                pbar.set_postfix(seg=segment_id, status=f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Build OCC compound with UV bounds and boundary wires for proper tessellation
        occ_compound = build_compound(occ_surfaces, occ_bounds, occ_boundary_wires) if occ_surfaces else None

        # Build surface_params output
        num_primitives = sum(1 for s in surface_params_list if s["type"] in ["plane", "sphere", "cylinder", "cone"])
        num_bspline = sum(1 for s in surface_params_list if s["type"] == "open_spline")

        surface_params = {
            "surfaces": surface_params_list,
            "num_primitives": num_primitives,
            "num_freeform": num_bspline,
            "num_skipped": num_segments - len(surface_params_list),
            "total_surfaces": len(surface_params_list),
        }

        # Summary
        summary = f"Fitted {len(surface_params_list)} surfaces (OCC):\n\n"
        summary += f"Primitives: {num_primitives}, B-spline: {num_bspline}\n\n"
        summary += f"Backend: {spline_backend}, Control pts: {ctrl_pts_u}x{ctrl_pts_v}\n"

        print(f"[OK] OCC Surface fitting complete: {num_primitives} primitives, {num_bspline} B-splines")

        # Build CAD_MODEL dict for downstream nodes (CADFaceAnalysis, etc.)
        cad_model = None
        if occ_compound is not None:
            cad_model = {
                "occ_shape": occ_compound,
                "format": "occ",
                "source": "point2cad_surface_fitting",
            }

        return (occ_compound, cad_model, surface_params, summary)


# ============================================================================
# Register Nodes
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadPoint2CADModel": LoadPoint2CADModel,
    "Point2CADSegmentation": Point2CADSegmentation,
    "Point2CADSurfaceFitting": Point2CADSurfaceFitting,
    "Point2CADSurfaceFittingOCC": Point2CADSurfaceFittingOCC,
    "Point2CADTopologyExtraction": Point2CADTopologyExtraction,
    "Point2CADExportBrep": Point2CADExportBrep,
    "SaveSurfaceParams": SaveSurfaceParams,
    "LoadSurfaceParams": LoadSurfaceParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPoint2CADModel": "Load Point2CAD Model",
    "Point2CADSegmentation": "Point2CAD Segmentation",
    "Point2CADSurfaceFitting": "Point2CAD Surface Fitting",
    "Point2CADSurfaceFittingOCC": "Point2CAD Surface Fitting (OCC)",
    "Point2CADTopologyExtraction": "Point2CAD Topology Extraction",
    "Point2CADExportBrep": "Point2CAD Export B-rep",
    "SaveSurfaceParams": "Save Surface Params",
    "LoadSurfaceParams": "Load Surface Params",
}
