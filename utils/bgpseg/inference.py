"""
BGPSeg Inference Pipeline

Simplified two-stage inference for point cloud segmentation:
1. BoundaryNet predicts boundary points
2. BGFE extracts embeddings using boundary guidance
3. Mean-shift clustering produces instance segments
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .model import BoundaryPredictor, BGPSeg, PRIMITIVE_TYPES
from .mean_shift import cluster_embeddings


def load_bgpseg_models(
    models_dir: Path,
    device: str = "cuda",
    in_channels: int = 6,
    num_classes: int = 10
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load pretrained BGPSeg models.

    Args:
        models_dir: Directory containing model checkpoints
        device: Device to load models on
        in_channels: Input feature dimension (6 = xyz + normals)
        num_classes: Number of primitive types (10)

    Returns:
        boundary_model: BoundaryNet model
        bgpseg_model: BGFE main model
    """
    # Load BoundaryNet
    boundary_model = BoundaryPredictor(in_channels=in_channels)
    boundary_path = models_dir / "Boundary_model.pth"

    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary model not found: {boundary_path}")

    print(f"[BGPSeg] Loading boundary model from {boundary_path}")
    checkpoint = torch.load(boundary_path, map_location=device)

    # Handle DataParallel state dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    boundary_model.load_state_dict(new_state_dict)
    boundary_model.to(device).eval()

    # Load BGFE (main model)
    # Note: in_channels is 7 for BGFE because it adds boundary probability
    bgpseg_model = BGPSeg(in_channels=7, num_classes=num_classes)
    bgpseg_path = models_dir / "BGPSeg_model.pth"

    if not bgpseg_path.exists():
        raise FileNotFoundError(f"BGPSeg model not found: {bgpseg_path}")

    print(f"[BGPSeg] Loading main model from {bgpseg_path}")
    checkpoint = torch.load(bgpseg_path, map_location=device)

    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    bgpseg_model.load_state_dict(new_state_dict)
    bgpseg_model.to(device).eval()

    print(f"[BGPSeg] Models loaded successfully on {device}")
    return boundary_model, bgpseg_model


def run_bgpseg_inference(
    points: np.ndarray,
    normals: np.ndarray,
    boundary_model: torch.nn.Module,
    bgpseg_model: torch.nn.Module,
    device: str = "cuda",
    bandwidth: float = 1.31,
    boundary_threshold: float = 0.5,
    cluster_batch_size: int = 700
) -> Dict[str, np.ndarray]:
    """
    Run BGPSeg two-stage inference on a point cloud.

    Args:
        points: (N, 3) Point coordinates
        normals: (N, 3) Point normals
        boundary_model: Loaded BoundaryNet model
        bgpseg_model: Loaded BGFE model
        device: Device for inference
        bandwidth: Mean-shift clustering bandwidth
        boundary_threshold: Threshold for boundary classification
        cluster_batch_size: GPU batch size for clustering

    Returns:
        Dictionary with:
            - labels: (N,) Cluster/instance labels per point
            - primitive_types: (N,) Predicted primitive type per point (0-9)
            - boundary_mask: (N,) Boolean mask of boundary points
            - boundary_probs: (N,) Boundary probability per point
            - num_clusters: Number of clusters found
            - embeddings: (N, D) Per-point embeddings (optional, for debugging)
    """
    # Ensure inputs are numpy arrays
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    if points.shape[0] != normals.shape[0]:
        raise ValueError(f"Points and normals must have same length: {points.shape[0]} vs {normals.shape[0]}")

    n_points = points.shape[0]
    print(f"[BGPSeg] Running inference on {n_points} points")

    # Prepare input tensors
    coord = torch.from_numpy(points).float().to(device)
    normals_t = torch.from_numpy(normals).float().to(device)
    offset = torch.tensor([n_points], dtype=torch.int32).to(device)

    # Create dummy edges tensor for inference (not used without mesh connectivity)
    # The boundaryops require edges but in inference mode without mesh, we use a dummy
    edges = torch.zeros((n_points, 16), dtype=torch.int64).to(device)

    with torch.no_grad():
        # Stage 1: Boundary prediction
        print("[BGPSeg] Stage 1: Predicting boundaries...")
        boundary_pred = boundary_model([coord, normals_t, offset])
        # boundary_pred: [N, 2] logits for non-boundary/boundary

        # Get boundary probabilities and mask
        boundary_probs = torch.softmax(boundary_pred, dim=1)[:, 1]
        boundary_mask = (boundary_probs > boundary_threshold).int()

        # Stage 2: BGFE with boundary guidance
        print("[BGPSeg] Stage 2: Extracting primitive embeddings...")
        primitive_embedding, type_per_point = bgpseg_model(
            [coord, normals_t, offset],
            edges=edges,
            boundary_gt=boundary_mask,
            boundary_pred=boundary_pred,
            is_train=False
        )
        # primitive_embedding: [N, 32]
        # type_per_point: [N, 10]

    # Get primitive type predictions
    primitive_types = type_per_point.argmax(dim=1).cpu().numpy()

    # Mean-shift clustering on embeddings
    print(f"[BGPSeg] Stage 3: Clustering embeddings (bandwidth={bandwidth})...")
    labels, num_clusters = cluster_embeddings(
        primitive_embedding,
        bandwidth=bandwidth,
        batch_size=cluster_batch_size
    )

    print(f"[BGPSeg] Found {num_clusters} primitive instances")

    # Convert tensors to numpy
    boundary_mask_np = boundary_mask.cpu().numpy().astype(bool)
    boundary_probs_np = boundary_probs.cpu().numpy()
    embeddings_np = primitive_embedding.cpu().numpy()

    return {
        'labels': labels,
        'primitive_types': primitive_types,
        'boundary_mask': boundary_mask_np,
        'boundary_probs': boundary_probs_np,
        'num_clusters': num_clusters,
        'embeddings': embeddings_np,
    }


def get_primitive_type_name(type_id: int) -> str:
    """Get the human-readable name for a primitive type ID."""
    return PRIMITIVE_TYPES.get(type_id, f"Unknown ({type_id})")


def summarize_segmentation(
    labels: np.ndarray,
    primitive_types: np.ndarray,
    boundary_mask: np.ndarray
) -> str:
    """
    Generate a summary string for segmentation results.

    Args:
        labels: (N,) Cluster labels
        primitive_types: (N,) Primitive type predictions
        boundary_mask: (N,) Boundary point mask

    Returns:
        Summary string
    """
    n_points = len(labels)
    n_clusters = len(np.unique(labels))
    n_boundary = boundary_mask.sum()

    # Count primitive types
    type_counts = {}
    for i in range(10):
        count = (primitive_types == i).sum()
        if count > 0:
            type_counts[PRIMITIVE_TYPES[i]] = count

    lines = [
        f"BGPSeg Segmentation Summary:",
        f"  Total points: {n_points}",
        f"  Clusters found: {n_clusters}",
        f"  Boundary points: {n_boundary} ({100*n_boundary/n_points:.1f}%)",
        f"  Primitive type distribution:",
    ]

    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {type_name}: {count} ({100*count/n_points:.1f}%)")

    return "\n".join(lines)
