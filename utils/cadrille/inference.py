"""
Cadrille Inference Pipeline

Handles input preprocessing and code generation for multi-modal CAD reconstruction.
Supports point cloud, image, and text inputs.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Union, Dict, Any
from PIL import Image

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def normalize_point_cloud(
    points: np.ndarray,
    scale_range: float = 2.0,
    center: bool = True
) -> np.ndarray:
    """
    Normalize point cloud to fit within a centered cube.

    Same normalization as CAD-Recode: center at origin and scale to [-1, 1].

    Args:
        points: (N, 3) point coordinates
        scale_range: Size of the bounding cube (default 2.0 for [-1, 1])
        center: Whether to center the point cloud at origin

    Returns:
        (N, 3) normalized point coordinates
    """
    points = points.copy().astype(np.float32)

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
    Sample points using Farthest Point Sampling (FPS).

    Pure numpy implementation - works without pytorch3d.
    Produces more uniform coverage than random sampling.

    Args:
        points: (N, 3) point coordinates
        n_points: Number of points to sample (default 256 for Cadrille)
        n_pre_points: Maximum points to consider before FPS

    Returns:
        (n_points, 3) sampled point coordinates
    """
    n = len(points)

    if n <= n_points:
        if n < n_points:
            # Pad with repeated points
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
        last_point = points[selected[i - 1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)

    return points[selected]


def sample_points_fps_torch(
    points: np.ndarray,
    n_points: int = 256,
    n_pre_points: int = 8192
) -> np.ndarray:
    """
    Sample points using pytorch3d FPS if available, else fall back to numpy.

    Args:
        points: (N, 3) point coordinates
        n_points: Number of points to sample
        n_pre_points: Max points before sampling

    Returns:
        (n_points, 3) sampled point coordinates
    """
    try:
        from pytorch3d.ops import sample_farthest_points

        n = len(points)
        if n > n_pre_points:
            indices = np.random.choice(n, n_pre_points, replace=False)
            points = points[indices]

        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        _, ids = sample_farthest_points(points_tensor, K=n_points)
        ids = ids[0].numpy()
        return points[ids]

    except ImportError:
        return sample_points_fps(points, n_points, n_pre_points)


def render_mesh_views(
    mesh,
    img_size: int = 128,
    normalize_scale: float = 200.0
) -> List[Image.Image]:
    """
    Render 4 orthogonal views of a mesh for image-based inference.

    Args:
        mesh: trimesh.Trimesh object
        img_size: Output image size (default 128x128 per view)
        normalize_scale: Scale factor for mesh normalization

    Returns:
        List of 4 PIL Images from different viewpoints
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required for mesh rendering")

    # Normalize mesh to [0, 1] cube centered at 0.5
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(trimesh.transformations.scale_matrix(1 / normalize_scale))
    mesh_copy.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    # Set mesh color (yellow-ish like Cadrille training data)
    mesh_copy.visual.face_colors = [255, 255, 136, 255]

    # Camera directions: 4 orthogonal views matching Cadrille
    camera_directions = [
        [1, 1, 1],      # Front-right-top
        [-1, -1, -1],   # Back-left-bottom
        [-1, 1, -1],    # Back-right-bottom
        [1, -1, 1],     # Front-left-top
    ]

    center = np.array([0.5, 0.5, 0.5])
    camera_distance = 1.8
    images = []

    for direction in camera_directions:
        scene = trimesh.Scene(mesh_copy)

        # Camera position
        direction = np.array(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        camera_pos = center + direction * camera_distance

        # Build camera transform (look at center)
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_transform = np.eye(4)
        camera_transform[:3, 0] = right
        camera_transform[:3, 1] = up
        camera_transform[:3, 2] = -forward
        camera_transform[:3, 3] = camera_pos

        scene.camera_transform = camera_transform

        # Render
        try:
            png_bytes = scene.save_image(resolution=(img_size, img_size))
            img = Image.open(trimesh.util.wrap_as_stream(png_bytes))
            images.append(img.convert('RGB'))
        except Exception as e:
            print(f"[Cadrille] Warning: rendering failed ({e})")
            images.append(Image.new('RGB', (img_size, img_size), color=(200, 200, 200)))

    return images


def prepare_point_cloud_input(
    point_cloud: np.ndarray,
    processor,
    n_points: int = 256,
    normalize: bool = True,
    use_fps: bool = True,
) -> Dict[str, Any]:
    """
    Prepare point cloud input for Cadrille inference.

    Args:
        point_cloud: (N, 3) point coordinates
        processor: Qwen2-VL processor
        n_points: Number of points to sample
        normalize: Whether to normalize point cloud
        use_fps: Use FPS (True) or random sampling (False)

    Returns:
        Dictionary with model inputs
    """
    # Sample points
    if use_fps:
        points = sample_points_fps_torch(point_cloud, n_points)
    else:
        if len(point_cloud) > n_points:
            indices = np.random.choice(len(point_cloud), n_points, replace=False)
            points = point_cloud[indices]
        else:
            points = point_cloud

    # Normalize
    if normalize:
        points = normalize_point_cloud(points)

    # Build message
    message = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Generate cadquery code'}
        ]
    }]

    # Apply chat template with generation prompt
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # Prepend point tokens (using pad token as placeholder)
    points_inputs = ''.join(n_points * [processor.tokenizer.pad_token])
    text = points_inputs + text

    # Process text
    inputs = processor(text=[text], padding=True, return_tensors='pt')

    # Add point cloud
    inputs['point_clouds'] = torch.tensor(points.astype(np.float32)).unsqueeze(0)
    inputs['is_pc'] = torch.tensor([True], dtype=torch.bool)
    inputs['is_img'] = torch.tensor([False], dtype=torch.bool)

    return inputs


def prepare_image_input(
    images: List[Image.Image],
    processor,
) -> Dict[str, Any]:
    """
    Prepare image input for Cadrille inference.

    Args:
        images: List of 4 PIL Images (orthogonal views)
        processor: Qwen2-VL processor

    Returns:
        Dictionary with model inputs
    """
    if not HAS_QWEN_VL_UTILS:
        raise ImportError("qwen_vl_utils required for image mode")

    # Add border to images
    from PIL import ImageOps
    images_bordered = [ImageOps.expand(img, border=3, fill='black') for img in images]

    # Combine into 2x2 grid
    combined = Image.fromarray(np.vstack((
        np.hstack((np.array(images_bordered[0]), np.array(images_bordered[1]))),
        np.hstack((np.array(images_bordered[2]), np.array(images_bordered[3])))
    )))

    # Build message with video (Cadrille uses video token for multi-view)
    message = [{
        'role': 'user',
        'content': [
            {'type': 'video', 'video': [combined], 'fps': 1.0},
            {'type': 'text', 'text': 'Generate cadquery code'}
        ]
    }]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # Process vision info
    image_inputs, video_inputs = process_vision_info([message])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    )

    # Add modality flags
    inputs['point_clouds'] = torch.zeros(1, 256, 3)
    inputs['is_pc'] = torch.tensor([False], dtype=torch.bool)
    inputs['is_img'] = torch.tensor([True], dtype=torch.bool)

    return inputs


def prepare_text_input(
    description: str,
    processor,
) -> Dict[str, Any]:
    """
    Prepare text input for Cadrille inference.

    Args:
        description: Text description of desired CAD model
        processor: Qwen2-VL processor

    Returns:
        Dictionary with model inputs
    """
    message = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': description}
        ]
    }]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], padding=True, return_tensors='pt')

    # Add empty modality flags
    inputs['point_clouds'] = torch.zeros(1, 256, 3)
    inputs['is_pc'] = torch.tensor([False], dtype=torch.bool)
    inputs['is_img'] = torch.tensor([False], dtype=torch.bool)

    return inputs


def generate_code(
    model,
    processor,
    inputs: Dict[str, Any],
    max_tokens: int = 768,
    device: Optional[str] = None,
) -> str:
    """
    Generate CadQuery code from prepared inputs.

    Args:
        model: Cadrille model instance
        processor: Qwen2-VL processor
        inputs: Prepared inputs from prepare_*_input functions
        max_tokens: Maximum tokens to generate
        device: Device to run on (None = auto-detect)

    Returns:
        Generated CadQuery Python code string
    """
    if device is None:
        device = next(model.parameters()).device

    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    point_clouds = inputs['point_clouds'].to(device)
    is_pc = inputs['is_pc'].to(device)
    is_img = inputs['is_img'].to(device)

    # Handle optional video inputs
    pixel_values_videos = inputs.get('pixel_values_videos')
    video_grid_thw = inputs.get('video_grid_thw')
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            is_pc=is_pc,
            is_img=is_img,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            max_new_tokens=max_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Trim input tokens from output
    generated_ids_trimmed = generated_ids[0, input_ids.shape[1]:]

    # Decode
    code = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return code.strip()
