"""
CAD-Recode Inference Pipeline
Handles point cloud preprocessing and code generation.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from transformers import AutoTokenizer


def normalize_point_cloud(
    points: np.ndarray,
    scale_range: float = 2.0,
    center: bool = True
) -> np.ndarray:
    """
    Normalize point cloud to fit within a centered cube.

    Args:
        points: (N, 3) point coordinates
        scale_range: Size of the bounding cube (default 2.0 for [-1, 1])
        center: Whether to center the point cloud at origin

    Returns:
        (N, 3) normalized point coordinates
    """
    points = points.copy()

    if center:
        # Center at origin using bounding box center
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        center_point = (bounds_min + bounds_max) / 2.0
        points = points - center_point

    # Scale to fit in cube of size scale_range
    extents = points.max(axis=0) - points.min(axis=0)
    max_extent = max(extents)
    if max_extent > 0:
        points = points * (scale_range / max_extent)

    return points


def sample_points_fps(
    points: np.ndarray,
    n_points: int = 256,
    n_pre_points: int = 8192
) -> np.ndarray:
    """
    Sample points using Farthest Point Sampling (v1.5 improvement).
    Pure numpy implementation - no pytorch3d dependency.

    This produces more uniform coverage than random sampling.

    Args:
        points: (N, 3) point coordinates
        n_points: Number of points to sample (default 256 for CAD-Recode)
        n_pre_points: Maximum points to consider before FPS

    Returns:
        (n_points, 3) sampled point coordinates
    """
    n = len(points)

    if n <= n_points:
        # Pad if not enough points
        if n < n_points:
            indices = np.random.choice(n, n_points, replace=True)
            return points[indices]
        return points

    # Limit input size for efficiency
    if n > n_pre_points:
        indices = np.random.choice(n, n_pre_points, replace=False)
        points = points[indices]
        n = n_pre_points

    # Pure numpy FPS implementation
    selected = np.zeros(n_points, dtype=np.int64)
    distances = np.full(n, np.inf)

    # Start with random point
    selected[0] = np.random.randint(n)

    for i in range(1, n_points):
        # Distance from last selected point to all points
        last_point = points[selected[i - 1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)

        # Update minimum distances
        distances = np.minimum(distances, dist_to_last)

        # Select point with maximum distance to any selected point
        selected[i] = np.argmax(distances)

    return points[selected]


def sample_points_random(
    points: np.ndarray,
    n_points: int = 256
) -> np.ndarray:
    """
    Sample points using random sampling.

    Args:
        points: (N, 3) point coordinates
        n_points: Number of points to sample

    Returns:
        (n_points, 3) sampled point coordinates
    """
    if len(points) <= n_points:
        if len(points) < n_points:
            indices = np.random.choice(len(points), n_points, replace=True)
            return points[indices]
        return points

    indices = np.random.choice(len(points), n_points, replace=False)
    return points[indices]


def mesh_to_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int = 256,
    use_fps: bool = True
) -> np.ndarray:
    """
    Convert mesh to point cloud by sampling surface points.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh faces
        n_points: Number of points to sample
        use_fps: Use farthest point sampling (slower but better coverage)

    Returns:
        (n_points, 3) sampled point coordinates
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # Sample more points than needed, then use FPS/random to select
        surface_points, _ = trimesh.sample.sample_surface(mesh, n_points * 32)

        if use_fps:
            return sample_points_fps(surface_points, n_points)
        else:
            return sample_points_random(surface_points, n_points)
    except ImportError:
        # Fallback: just sample from vertices
        return sample_points_fps(vertices, n_points) if use_fps else sample_points_random(vertices, n_points)


def prepare_input(
    point_cloud: np.ndarray,
    tokenizer,
    n_points: int = 256,
    normalize: bool = True,
    use_fps: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare point cloud and tokenizer inputs for CAD-Recode inference.

    Args:
        point_cloud: (N, 3) point coordinates
        tokenizer: Qwen2 tokenizer
        n_points: Number of points (default 256)
        normalize: Whether to normalize point cloud
        use_fps: Use farthest point sampling

    Returns:
        Tuple of (input_ids, attention_mask, point_cloud_tensor)
    """
    # Preprocess point cloud
    if normalize:
        point_cloud = normalize_point_cloud(point_cloud)

    if use_fps:
        point_cloud = sample_points_fps(point_cloud, n_points)
    else:
        point_cloud = sample_points_random(point_cloud, n_points)

    # Create input sequence: [point_tokens...] [<|im_start|>]
    # Point tokens use pad_token_id as placeholder
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + \
                [tokenizer('<|im_start|>')['input_ids'][0]]

    # Attention mask: -1 for point positions (special marker), 1 for text
    attention_mask = [-1] * len(point_cloud) + [1]

    return (
        torch.tensor(input_ids).unsqueeze(0),
        torch.tensor(attention_mask).unsqueeze(0),
        torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0)
    )


def generate_code(
    model,
    tokenizer,
    point_cloud: np.ndarray,
    n_points: int = 256,
    max_tokens: int = 768,
    normalize: bool = True,
    use_fps: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Generate CadQuery code from point cloud using CAD-Recode model.

    Args:
        model: CADRecode model instance
        tokenizer: Qwen2 tokenizer
        point_cloud: (N, 3) point coordinates
        n_points: Number of points to sample (default 256)
        max_tokens: Maximum tokens to generate (default 768)
        normalize: Whether to normalize point cloud
        use_fps: Use farthest point sampling
        device: Device to run on (None = auto-detect from model)

    Returns:
        Generated CadQuery Python code string
    """
    if device is None:
        device = next(model.parameters()).device

    # Prepare inputs
    input_ids, attention_mask, pc_tensor = prepare_input(
        point_cloud, tokenizer, n_points, normalize, use_fps
    )

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pc_tensor = pc_tensor.to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_cloud=pc_tensor,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and extract code
    text = tokenizer.batch_decode(output)[0]

    # Extract code between markers
    start_marker = '<|im_start|>'
    end_marker = '<|endoftext|>'

    start = text.find(start_marker)
    if start != -1:
        start += len(start_marker)
    else:
        start = 0

    end = text.find(end_marker)
    if end == -1:
        end = len(text)

    code = text[start:end].strip()
    return code


def load_tokenizer(model_id: str = 'Qwen/Qwen2-1.5B'):
    """
    Load the Qwen2 tokenizer configured for CAD-Recode.

    Args:
        model_id: HuggingFace model ID for tokenizer

    Returns:
        Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        pad_token='<|im_end|>',
        padding_side='left'
    )
    return tokenizer
