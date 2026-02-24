"""
BGPSeg Nodes for ComfyUI-CADabra

Boundary-Guided Primitive Instance Segmentation of Point Clouds.

Paper: "BGPSeg: Boundary-Guided Primitive Instance Segmentation of Point Clouds"
IEEE Transactions on Image Processing (2025) - Fang et al.

Pipeline:
1. LoadBGPSegModels - Download/load the BoundaryNet + BGFE models
2. BGPSegSegmentation - Segment point cloud into primitive instances

Output is compatible with Point2CADSurfaceFitting for CAD reconstruction.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

# Optional imports with error handling
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[CADabra] Warning: trimesh not installed.")


# ============================================================================
# Node 1: LoadBGPSegModels
# ============================================================================

class LoadBGPSegModels:
    """
    Downloads and loads BGPSeg pretrained models from Google Drive.

    Downloads:
    - BGPSeg_model.pth (BGFE main model - 10 primitive types)
    - Boundary_model.pth (BoundaryNet boundary predictor)

    Source: https://drive.google.com/drive/folders/1qev6yadvatGxGm-9HQBNiNIIpDNe9D1A

    BGPSeg uses a two-stage architecture:
    1. BoundaryNet predicts which points lie on primitive boundaries
    2. BGFE uses boundary guidance for better instance segmentation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["BGPSeg (ABC Dataset)"], {
                    "default": "BGPSeg (ABC Dataset)",
                    "tooltip": "BGPSeg pretrained on ABC Primitive dataset (10 primitive types: plane, sphere, cylinder, cone, torus, etc.)"
                }),
            },
            "optional": {
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Automatically download models from Google Drive if not found locally."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for model inference. CUDA strongly recommended for performance. Note: CUDA required for custom point cloud operations."
                }),
            }
        }

    RETURN_TYPES = ("BGPSEG_MODELS", "STRING")
    RETURN_NAMES = ("models", "model_info")
    FUNCTION = "load_models"
    CATEGORY = "CADabra/BGPSeg"

    def load_models(
        self,
        model_variant: str,
        auto_download: bool = True,
        device: str = "cuda",
    ) -> Tuple:
        """Load BGPSeg models from local cache or download from Google Drive."""

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("[BGPSeg] CUDA not available, falling back to CPU")
            print("[BGPSeg] Warning: CPU inference will be very slow")
            device = "cpu"

        # Get models directory
        from ..utils.model_loader import get_bgpseg_models_dir, download_bgpseg_models

        models_dir = get_bgpseg_models_dir()

        # Check if models exist
        bgpseg_path = models_dir / "BGPSeg_model.pth"
        boundary_path = models_dir / "Boundary_model.pth"

        if not (bgpseg_path.exists() and boundary_path.exists()):
            if auto_download:
                print("[BGPSeg] Models not found, downloading...")
                result = download_bgpseg_models()
                if result is None:
                    raise RuntimeError(
                        "Failed to download BGPSeg models. "
                        "Please download manually from Google Drive: "
                        "https://drive.google.com/drive/folders/1qev6yadvatGxGm-9HQBNiNIIpDNe9D1A"
                    )
            else:
                raise FileNotFoundError(
                    f"BGPSeg models not found in {models_dir}. "
                    "Enable auto_download or download manually."
                )

        # Build CUDA extensions if needed
        print("[BGPSeg] Checking CUDA extensions...")
        try:
            from ..utils.bgpseg.cuda_ops import build_ops
            if device == "cuda":
                build_ops()
        except Exception as e:
            print(f"[BGPSeg] Warning: CUDA ops build check failed: {e}")
            if device == "cuda":
                print("[BGPSeg] Inference may fail without CUDA extensions")

        # Load models
        print(f"[BGPSeg] Loading models on {device}...")
        from ..utils.bgpseg.inference import load_bgpseg_models

        boundary_model, bgpseg_model = load_bgpseg_models(
            models_dir=models_dir,
            device=device,
            in_channels=6,  # xyz + normals
            num_classes=10,  # 10 primitive types
        )

        # Package model data
        model_data = {
            "boundary_model": boundary_model,
            "bgpseg_model": bgpseg_model,
            "device": device,
            "models_dir": str(models_dir),
        }

        info_string = (
            f"Model: BGPSeg (Boundary-Guided Primitive Segmentation)\n"
            f"Variant: {model_variant}\n"
            f"Device: {device}\n"
            f"Primitive types: 10 (Plane, Sphere, Cylinder, Cone, Torus, etc.)\n"
            f"Models directory: {models_dir}"
        )

        print("[OK] BGPSeg models loaded successfully")
        return (model_data, info_string)


# ============================================================================
# Node 2: BGPSegSegmentation
# ============================================================================

class BGPSegSegmentation:
    """
    Segment point cloud using BGPSeg two-stage architecture.

    Pipeline:
    1. BoundaryNet predicts boundary points
    2. BGFE extracts embeddings using boundary-guided attention
    3. Mean-shift clustering groups embeddings into instances
    4. Per-point primitive type classification

    Output format is compatible with Point2CADSurfaceFitting for CAD reconstruction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("TRIMESH", {
                    "tooltip": "Input point cloud or mesh. Must have vertex normals. Use a mesh-to-pointcloud node if needed."
                }),
                "models": ("BGPSEG_MODELS", {
                    "tooltip": "Loaded BGPSeg models from LoadBGPSegModels node."
                }),
            },
            "optional": {
                "bandwidth": ("FLOAT", {
                    "default": 1.31,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Mean-shift clustering bandwidth. Higher values = fewer, larger clusters. Default 1.31 from BGPSeg paper."
                }),
                "boundary_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for boundary point classification. Points with boundary probability above this are marked as boundaries."
                }),
                "cluster_batch_size": ("INT", {
                    "default": 700,
                    "min": 100,
                    "max": 2000,
                    "step": 100,
                    "tooltip": "Batch size for GPU mean-shift clustering. Reduce if running out of memory."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRIMESH", "STRING")
    RETURN_NAMES = ("segmented_cloud", "boundary_points", "summary")
    FUNCTION = "segment"
    CATEGORY = "CADabra/BGPSeg"

    def segment(
        self,
        point_cloud,
        models: Dict[str, Any],
        bandwidth: float = 1.31,
        boundary_threshold: float = 0.5,
        cluster_batch_size: int = 700,
    ) -> Tuple:
        """Segment point cloud using BGPSeg."""

        if not HAS_TRIMESH:
            raise ImportError("trimesh is required for BGPSeg segmentation")

        # Extract points and normals from input
        if isinstance(point_cloud, trimesh.Trimesh):
            if len(point_cloud.faces) > 0:
                # It's a mesh - sample points from surface
                print("[BGPSeg] Sampling points from mesh surface...")
                points, face_indices = trimesh.sample.sample_surface(point_cloud, 8192)
                # Get normals at sampled points
                normals = point_cloud.face_normals[face_indices]
            else:
                # It's a point cloud stored as Trimesh
                points = np.asarray(point_cloud.vertices)
                if hasattr(point_cloud, 'vertex_normals') and point_cloud.vertex_normals is not None:
                    normals = np.asarray(point_cloud.vertex_normals)
                else:
                    raise ValueError("Point cloud must have vertex normals for BGPSeg segmentation")
        elif isinstance(point_cloud, trimesh.PointCloud):
            points = np.asarray(point_cloud.vertices)
            if hasattr(point_cloud, 'vertex_normals') and point_cloud.vertex_normals is not None:
                normals = np.asarray(point_cloud.vertex_normals)
            else:
                raise ValueError("Point cloud must have vertex normals for BGPSeg segmentation")
        elif isinstance(point_cloud, dict):
            points = point_cloud.get('points', point_cloud.get('vertices'))
            normals = point_cloud.get('normals', point_cloud.get('vertex_normals'))
            if points is None or normals is None:
                raise ValueError("Point cloud dict must contain 'points' and 'normals'")
            points = np.asarray(points)
            normals = np.asarray(normals)
        else:
            raise ValueError(f"Unsupported input type: {type(point_cloud)}")

        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)

        print(f"[BGPSeg] Input: {len(points)} points")

        # Get models
        boundary_model = models["boundary_model"]
        bgpseg_model = models["bgpseg_model"]
        device = models["device"]

        # Run inference
        from ..utils.bgpseg.inference import run_bgpseg_inference, summarize_segmentation

        results = run_bgpseg_inference(
            points=points,
            normals=normals,
            boundary_model=boundary_model,
            bgpseg_model=bgpseg_model,
            device=device,
            bandwidth=bandwidth,
            boundary_threshold=boundary_threshold,
            cluster_batch_size=cluster_batch_size,
        )

        labels = results['labels']
        primitive_types = results['primitive_types']
        boundary_mask = results['boundary_mask']
        boundary_probs = results['boundary_probs']
        num_clusters = results['num_clusters']

        # Create segmented point cloud output
        # Format compatible with Point2CADSurfaceFitting
        segmented_cloud = trimesh.PointCloud(points)

        # Store segmentation data in vertex attributes
        # This format matches Point2CAD output for pipeline compatibility
        segmented_cloud.metadata['vertex_attributes'] = {
            'label': labels.astype(np.int32),
            'primitive_type': primitive_types.astype(np.int32),
            'boundary': boundary_mask.astype(np.int32),
            'boundary_prob': boundary_probs.astype(np.float32),
            'confidence': np.ones(len(points), dtype=np.float32),
        }
        segmented_cloud.metadata['num_segments'] = num_clusters
        segmented_cloud.metadata['is_point_cloud'] = True
        segmented_cloud.metadata['segmentation_method'] = 'bgpseg'
        segmented_cloud.metadata['model_type'] = 'BGPSeg'

        # Also store as direct attributes for compatibility
        segmented_cloud.vertex_normals = normals

        # Create boundary points visualization
        boundary_indices = np.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            boundary_points_cloud = trimesh.PointCloud(
                points[boundary_indices],
                colors=np.tile([255, 0, 0, 255], (len(boundary_indices), 1))  # Red for boundaries
            )
        else:
            # Empty point cloud if no boundaries
            boundary_points_cloud = trimesh.PointCloud(np.zeros((0, 3)))

        # Generate summary
        summary = summarize_segmentation(labels, primitive_types, boundary_mask)

        print(f"[OK] BGPSeg segmentation complete: {num_clusters} instances")

        return (segmented_cloud, boundary_points_cloud, summary)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadBGPSegModels": LoadBGPSegModels,
    "BGPSegSegmentation": BGPSegSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBGPSegModels": "Load BGPSeg Models",
    "BGPSegSegmentation": "BGPSeg Segmentation",
}
