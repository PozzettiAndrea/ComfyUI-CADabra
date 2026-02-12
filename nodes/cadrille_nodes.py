"""
Cadrille Nodes for ComfyUI-CADabra

Multi-modal CAD reconstruction from point clouds, images, or text.

Paper: "Cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning"
Authors: Kolodiazhnyi et al.
Project: https://github.com/col14m/cadrille

Pipeline:
1. LoadCadrilleModel - Download/load the Cadrille model from HuggingFace
2. CadrilleInference - Generate CadQuery code from point cloud, image, or text
3. CADRecodeExecute - Safely execute CadQuery code (reuses existing node)
"""

import os
import tempfile
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List

from .cad_common import make_cad_model

# Optional imports with error handling
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[Cadrille] Warning: trimesh not installed.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[Cadrille] Warning: PIL not installed.")


# ============================================================================
# Node 1: LoadCadrilleModel
# ============================================================================

class LoadCadrilleModel:
    """
    Downloads and loads Cadrille models from HuggingFace.
    Models are cached by HuggingFace transformers library.

    Supports:
    - SFT model: Supervised fine-tuned (default)
    - RL model: Reinforcement learning fine-tuned
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["SFT (recommended)", "RL"], {
                    "default": "SFT (recommended)",
                    "tooltip": "Model version. SFT is supervised fine-tuned, RL adds reinforcement learning."
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on. 'auto' uses CUDA if available."
                }),
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended for best speed/quality balance."
                }),
            }
        }

    RETURN_TYPES = ("CADRILLE_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "CADabra/Cadrille"

    def load_model(
        self,
        model_version: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> Tuple:
        """Load Cadrille model from HuggingFace."""
        from transformers import AutoProcessor

        # Import model class
        from ..utils.cadrille import Cadrille

        # Map version to HuggingFace model ID
        version_map = {
            "SFT (recommended)": "maksimko123/cadrille",
            "RL": "maksimko123/cadrille-rl",
        }
        model_id = version_map.get(model_version, "maksimko123/cadrille")
        version = "RL" if "RL" in model_version else "SFT"

        print(f"[Cadrille] Loading model: {model_id}")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        try:
            # Load processor (tokenizer + image processor)
            print("[Cadrille] Loading processor...")
            processor = AutoProcessor.from_pretrained(
                'Qwen/Qwen2-VL-2B-Instruct',
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                padding_side='left'
            )

            # Load model without flash attention
            print(f"[Cadrille] Downloading/loading model (this may take a moment on first run)...")
            model = Cadrille.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device if device != "cpu" else None,
            )

            if device == "cpu":
                model = model.to(device)

            model = model.eval()

            print(f"[OK] Cadrille model loaded on {device}")

            # Package model data
            model_data = {
                "model": model,
                "processor": processor,
                "device": device,
                "version": version,
                "model_id": model_id,
            }

            info_string = (
                f"Model: Cadrille {version}\n"
                f"HuggingFace ID: {model_id}\n"
                f"Device: {device}\n"
                f"Dtype: {torch_dtype}"
            )

            return (model_data, info_string)

        except Exception as e:
            raise RuntimeError(f"Failed to load Cadrille model: {str(e)}")


# ============================================================================
# Node 2: CadrilleInference (Point Cloud)
# ============================================================================

class CadrilleInference:
    """
    Generate CadQuery Python code from a point cloud using Cadrille.

    Takes a point cloud (N, 3) and produces executable CadQuery code
    that reconstructs the CAD model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CADRILLE_MODEL",),
                "point_cloud": ("TRIMESH",),
            },
            "optional": {
                "num_points": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Number of points sampled from the input. Cadrille was trained with 256 points - using this value gives best results. Lower values may lose geometric detail."
                }),
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "tooltip": "Maximum tokens (code length) to generate. Simple shapes need ~256-512, complex models need ~768-2048. Increase if output appears truncated."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Normalize point cloud to [-1, 1] range using Cadrille's normalization."
                }),
                "use_fps": ("BOOLEAN", {
                    "default": True,
                    "label_on": "FPS",
                    "label_off": "random",
                    "tooltip": "Use Farthest Point Sampling for uniform coverage, or random sampling."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cadquery_code", "code_preview")
    FUNCTION = "generate"
    CATEGORY = "CADabra/Cadrille"
    OUTPUT_NODE = True

    def generate(
        self,
        model: Dict[str, Any],
        point_cloud,
        num_points: int = 256,
        max_tokens: int = 768,
        normalize: bool = True,
        use_fps: bool = True,
    ) -> dict:
        """Generate CadQuery code from mesh or point cloud."""
        from ..utils.cadrille import generate_code, prepare_point_cloud_input
        import trimesh as tm

        # Extract points from input
        n_surface_samples = 8192

        if isinstance(point_cloud, tm.Trimesh) and len(point_cloud.faces) > 0:
            print(f"[Cadrille] Sampling {n_surface_samples} points from mesh surface...")
            points, _ = tm.sample.sample_surface(point_cloud, n_surface_samples)
            points = np.asarray(points, dtype=np.float32)
        elif isinstance(point_cloud, tm.PointCloud):
            points = np.asarray(point_cloud.vertices, dtype=np.float32)
        elif isinstance(point_cloud, tm.Trimesh):
            points = np.asarray(point_cloud.vertices, dtype=np.float32)
        elif isinstance(point_cloud, dict):
            points = point_cloud.get("points", point_cloud.get("vertices"))
            if points is not None:
                points = np.asarray(points, dtype=np.float32)
        elif isinstance(point_cloud, np.ndarray):
            points = np.asarray(point_cloud, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(point_cloud)}")

        if points is None:
            raise ValueError("Input does not contain valid point data")

        print(f"[Cadrille] Input: {len(points)} points")

        # Get model components
        nn_model = model["model"]
        processor = model["processor"]
        device = model["device"]

        # Prepare inputs
        print(f"[Cadrille] Preparing point cloud (num_points={num_points})...")
        from ..utils.cadrille.inference import sample_points_fps_torch, normalize_point_cloud

        # Sample points
        if use_fps:
            sampled_points = sample_points_fps_torch(points, num_points)
        else:
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                sampled_points = points[indices]
            else:
                sampled_points = points

        # Normalize
        if normalize:
            sampled_points = normalize_point_cloud(sampled_points)

        # Prepare model inputs
        inputs = prepare_point_cloud_input(
            sampled_points,
            processor,
            n_points=num_points,
            normalize=False,  # Already normalized above
            use_fps=False,  # Already sampled above
        )

        # Generate code
        print(f"[Cadrille] Generating CadQuery code (max_tokens={max_tokens})...")
        code = generate_code(
            model=nn_model,
            processor=processor,
            inputs=inputs,
            max_tokens=max_tokens,
            device=device,
        )

        print(f"[OK] Generated {len(code)} characters of CadQuery code")

        # Create preview (truncate for display)
        preview = code[:500] + ("..." if len(code) > 500 else "")

        return {"ui": {"text": (preview,)}, "result": (code, preview)}


# ============================================================================
# Node 3: CadrilleImageInference
# ============================================================================

class CadrilleImageInference:
    """
    Generate CadQuery Python code from images using Cadrille.

    Takes 4 orthogonal view images and produces executable CadQuery code.
    Can also render views automatically from a mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CADRILLE_MODEL",),
            },
            "optional": {
                "mesh": ("TRIMESH", {
                    "tooltip": "Mesh to render views from. If provided, images are rendered automatically."
                }),
                "image_front": ("IMAGE", {
                    "tooltip": "Front view image (if not using mesh)."
                }),
                "image_back": ("IMAGE", {
                    "tooltip": "Back view image (if not using mesh)."
                }),
                "image_left": ("IMAGE", {
                    "tooltip": "Left view image (if not using mesh)."
                }),
                "image_right": ("IMAGE", {
                    "tooltip": "Right view image (if not using mesh)."
                }),
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "tooltip": "Maximum tokens to generate. Increase for complex models."
                }),
                "img_size": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 256,
                    "step": 32,
                    "tooltip": "Size of rendered view images (if using mesh input)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cadquery_code", "code_preview")
    FUNCTION = "generate"
    CATEGORY = "CADabra/Cadrille"

    def generate(
        self,
        model: Dict[str, Any],
        mesh=None,
        image_front=None,
        image_back=None,
        image_left=None,
        image_right=None,
        max_tokens: int = 768,
        img_size: int = 128,
    ) -> Tuple[str, str]:
        """Generate CadQuery code from images."""
        from ..utils.cadrille import generate_code, render_mesh_views
        from ..utils.cadrille.inference import prepare_image_input

        nn_model = model["model"]
        processor = model["processor"]
        device = model["device"]

        # Get images either from mesh or from inputs
        if mesh is not None:
            print(f"[Cadrille] Rendering {img_size}x{img_size} views from mesh...")
            images = render_mesh_views(mesh, img_size=img_size)
        elif all(img is not None for img in [image_front, image_back, image_left, image_right]):
            # Convert from tensor to PIL
            def tensor_to_pil(tensor):
                if tensor.dim() == 4:
                    tensor = tensor[0]
                arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(arr)

            images = [
                tensor_to_pil(image_front),
                tensor_to_pil(image_back),
                tensor_to_pil(image_left),
                tensor_to_pil(image_right),
            ]
        else:
            raise ValueError("Provide either mesh or all 4 images")

        print(f"[Cadrille] Preparing image input...")
        inputs = prepare_image_input(images, processor)

        print(f"[Cadrille] Generating CadQuery code (max_tokens={max_tokens})...")
        code = generate_code(
            model=nn_model,
            processor=processor,
            inputs=inputs,
            max_tokens=max_tokens,
            device=device,
        )

        print(f"[OK] Generated {len(code)} characters of CadQuery code")
        preview = code[:500] + ("..." if len(code) > 500 else "")

        return (code, preview)


# ============================================================================
# Node 4: CadrilleTextInference
# ============================================================================

class CadrilleTextInference:
    """
    Generate CadQuery Python code from a text description using Cadrille.

    Takes a natural language description and produces executable CadQuery code.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CADRILLE_MODEL",),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "A cube with rounded edges",
                    "tooltip": "Natural language description of the CAD model to generate."
                }),
            },
            "optional": {
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "tooltip": "Maximum tokens to generate. Increase for complex models."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cadquery_code", "code_preview")
    FUNCTION = "generate"
    CATEGORY = "CADabra/Cadrille"

    def generate(
        self,
        model: Dict[str, Any],
        description: str,
        max_tokens: int = 768,
    ) -> Tuple[str, str]:
        """Generate CadQuery code from text description."""
        from ..utils.cadrille import generate_code
        from ..utils.cadrille.inference import prepare_text_input

        nn_model = model["model"]
        processor = model["processor"]
        device = model["device"]

        print(f"[Cadrille] Generating from description: '{description[:50]}...'")

        inputs = prepare_text_input(description, processor)

        code = generate_code(
            model=nn_model,
            processor=processor,
            inputs=inputs,
            max_tokens=max_tokens,
            device=device,
        )

        print(f"[OK] Generated {len(code)} characters of CadQuery code")
        preview = code[:500] + ("..." if len(code) > 500 else "")

        return (code, preview)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadCadrilleModel": LoadCadrilleModel,
    "CadrilleInference": CadrilleInference,
    "CadrilleImageInference": CadrilleImageInference,
    "CadrilleTextInference": CadrilleTextInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCadrilleModel": "Load Cadrille Model",
    "CadrilleInference": "Cadrille Inference (Point Cloud)",
    "CadrilleImageInference": "Cadrille Inference (Image)",
    "CadrilleTextInference": "Cadrille Inference (Text)",
}
