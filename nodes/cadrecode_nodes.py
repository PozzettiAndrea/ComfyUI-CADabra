"""
CAD-Recode Nodes for ComfyUI-CADabra
Generates CadQuery Python code from point clouds using neural networks.

Paper: CAD-Recode: Reverse Engineering CAD Code from Point Clouds (ICCV 2025)
Authors: Rukhovich et al., University of Luxembourg
Project: https://cad-recode.github.io/

Pipeline:
1. LoadCADRecodeModel - Download/load the CAD-Recode model from HuggingFace
2. CADRecodeInference - Generate CadQuery code from point cloud
3. CadQueryExecute - Execute CadQuery code to produce CAD model
"""

import os
import tempfile
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any

from .cad_common import make_cad_model

# Optional imports with error handling
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[CADabra] Warning: trimesh not installed.")


# ============================================================================
# Node 1: LoadCADRecodeModel
# ============================================================================

class LoadCADRecodeModel:
    """
    Downloads and loads CAD-Recode models from HuggingFace.
    Models are cached by HuggingFace transformers library.

    Supports:
    - v1.5 (recommended): Latest model with improved training
    - v1: Original release model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["v1.5 (recommended)", "v1 (original)"], {
                    "default": "v1.5 (recommended)",
                    "tooltip": "Model version. v1.5 has better performance with FPS and larger coordinate range."
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on. 'auto' uses CUDA if available."
                }),
                "use_flash_attention": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Use Flash Attention 2 for faster inference (requires CUDA)."
                }),
            }
        }

    RETURN_TYPES = ("CADRECODE_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "CADabra/CAD-Recode"

    def load_model(
        self,
        model_version: str,
        device: str = "auto",
        use_flash_attention: bool = True,
    ) -> Tuple:
        """Load CAD-Recode model from HuggingFace."""
        from transformers import AutoTokenizer

        # Import here to avoid loading at startup
        from ..utils.cadrecode import CADRecode, load_tokenizer

        # Map version to HuggingFace model ID
        version_map = {
            "v1.5 (recommended)": "filapro/cad-recode-v1.5",
            "v1 (original)": "filapro/cad-recode",
        }
        model_id = version_map.get(model_version, "filapro/cad-recode-v1.5")
        version = "v1.5" if "v1.5" in model_version else "v1"

        print(f"[CAD-Recode] Loading model: {model_id}")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine attention implementation
        attn_impl = None
        if use_flash_attention and device == "cuda":
            try:
                attn_impl = "flash_attention_2"
                print("[CAD-Recode] Using Flash Attention 2")
            except:
                attn_impl = None
                print("[CAD-Recode] Flash Attention not available, using default")

        try:
            # Load tokenizer
            print("[CAD-Recode] Loading tokenizer...")
            tokenizer = load_tokenizer()

            # Load model
            print(f"[CAD-Recode] Downloading/loading model (this may take a moment on first run)...")
            model = CADRecode.from_pretrained(
                model_id,
                torch_dtype="auto",
                attn_implementation=attn_impl,
            ).eval().to(device)

            print(f"[OK] CAD-Recode model loaded on {device}")

            # Package model data
            model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "version": version,
                "model_id": model_id,
            }

            info_string = (
                f"Model: CAD-Recode {version}\n"
                f"HuggingFace ID: {model_id}\n"
                f"Device: {device}\n"
                f"Flash Attention: {attn_impl is not None}"
            )

            return (model_data, info_string)

        except Exception as e:
            raise RuntimeError(f"Failed to load CAD-Recode model: {str(e)}")


# ============================================================================
# Node 2: CADRecodeInference
# ============================================================================

class CADRecodeInference:
    """
    Generate CadQuery Python code from a point cloud using CAD-Recode.

    Takes a point cloud (N, 3) and produces executable CadQuery code
    that reconstructs the CAD model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CADRECODE_MODEL",),
                "point_cloud": ("TRIMESH",),  # Accepts TRIMESH from GeometryPack's MeshToPointCloud
            },
            "optional": {
                "num_points": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Number of points sampled from the input point cloud to feed into the model. CAD-Recode was trained with 256 points - using this value gives best results. Lower values (64-128) may lose geometric detail. Higher values (512-1024) can capture more detail but weren't used in training."
                }),
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "tooltip": "Maximum number of tokens (code characters/words) the model can generate. Simple shapes need fewer tokens (~256-512), complex models with many operations need more (~768-2048). If output code appears truncated, increase this value."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Normalize point cloud to [-1, 1] cube."
                }),
                "use_fps": ("BOOLEAN", {
                    "default": True,
                    "label_on": "FPS",
                    "label_off": "random",
                    "tooltip": "Use Farthest Point Sampling (v1.5 improvement) or random sampling."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cadquery_code", "code_preview")
    FUNCTION = "generate"
    CATEGORY = "CADabra/CAD-Recode"

    def generate(
        self,
        model: Dict[str, Any],
        point_cloud,  # TRIMESH (mesh or point cloud)
        num_points: int = 256,
        max_tokens: int = 768,
        normalize: bool = True,
        use_fps: bool = True,
    ) -> Tuple[str, str]:
        """Generate CadQuery code from mesh or point cloud."""
        from ..utils.cadrecode import generate_code
        import trimesh

        # Sample points from mesh surface or extract from point cloud
        n_surface_samples = 8192  # Sample this many points from surface first

        if isinstance(point_cloud, trimesh.Trimesh) and len(point_cloud.faces) > 0:
            # It's a mesh with faces - sample from surface
            print(f"[CAD-Recode] Sampling {n_surface_samples} points from mesh surface...")
            points, _ = trimesh.sample.sample_surface(point_cloud, n_surface_samples)
            points = np.asarray(points, dtype=np.float32)
        elif isinstance(point_cloud, trimesh.PointCloud):
            # Already a point cloud
            points = np.asarray(point_cloud.vertices, dtype=np.float32)
        elif isinstance(point_cloud, trimesh.Trimesh):
            # Trimesh with no faces - use vertices
            points = np.asarray(point_cloud.vertices, dtype=np.float32)
        elif isinstance(point_cloud, dict):
            # Legacy dict format
            points = point_cloud.get("points", point_cloud.get("vertices"))
            if points is not None:
                points = np.asarray(points, dtype=np.float32)
        elif isinstance(point_cloud, np.ndarray):
            points = np.asarray(point_cloud, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(point_cloud)}")

        if points is None:
            raise ValueError("Input does not contain valid point data")

        print(f"[CAD-Recode] Input: {len(points)} points")

        # Get model components
        nn_model = model["model"]
        tokenizer = model["tokenizer"]
        device = model["device"]

        # Generate code
        print(f"[CAD-Recode] Generating CadQuery code (num_points={num_points}, max_tokens={max_tokens})...")

        code = generate_code(
            model=nn_model,
            tokenizer=tokenizer,
            point_cloud=points,
            n_points=num_points,
            max_tokens=max_tokens,
            normalize=normalize,
            use_fps=use_fps,
            device=device,
        )

        print(f"[OK] Generated {len(code)} characters of CadQuery code")

        # Create preview (first 500 chars)
        preview = code[:500] + ("..." if len(code) > 500 else "")

        return (code, preview)


# ============================================================================
# Node 3: CadQueryExecute
# ============================================================================

class CadQueryExecute:
    """
    Execute CadQuery code to produce a CAD model.

    Runs the code in an isolated subprocess with timeout to handle
    potential crashes, memory leaks, or infinite loops.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cadquery_code": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "CadQuery Python code to execute. Should define 'r' as the result."
                }),
            },
            "optional": {
                "timeout": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "Maximum execution time in seconds."
                }),
                "validate_first": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Validate code syntax before execution."
                }),
            }
        }

    RETURN_TYPES = ("CAD_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("cad_model", "step_file", "status")
    FUNCTION = "execute"
    CATEGORY = "CADabra/CadQuery"

    def execute(
        self,
        cadquery_code: str,
        timeout: float = 10.0,
        validate_first: bool = True,
    ) -> Tuple:
        """Execute CadQuery code and return CAD model."""
        from ..utils.cadrecode import execute_cadquery_code, validate_cadquery_code

        if not cadquery_code or not cadquery_code.strip():
            raise ValueError("No CadQuery code provided")

        # Validate if requested
        if validate_first:
            is_valid, error = validate_cadquery_code(cadquery_code)
            if not is_valid:
                raise ValueError(f"Code validation failed: {error}")
            print("[CAD-Recode] Code validation passed")

        # Execute in subprocess
        print(f"[CAD-Recode] Executing CadQuery code (timeout={timeout}s)...")

        # Create persistent output directory
        output_dir = tempfile.mkdtemp(prefix='cadrecode_output_')

        success, result, stl_path = execute_cadquery_code(
            code=cadquery_code,
            timeout=timeout,
            output_dir=output_dir,
        )

        if not success:
            status = f"Execution failed: {result}"
            print(f"[ERROR] {status}")
            raise RuntimeError(status)

        step_path = result
        print(f"[OK] CAD model generated: {step_path}")

        # Load the STEP file into OCC
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone

            reader = STEPControl_Reader()
            status_code = reader.ReadFile(step_path)

            if status_code != IFSelect_RetDone:
                raise RuntimeError(f"Failed to read generated STEP file")

            reader.TransferRoots()
            occ_shape = reader.OneShape()

            cad_model = make_cad_model(occ_shape)
            cad_model["file_path"] = step_path
            cad_model["source"] = "cadrecode"

            status = f"Success: Generated CAD model saved to {step_path}"

            return (cad_model, step_path, status)

        except ImportError:
            # If OCC is not available, still return the file path
            print("[WARN] OCC not available, returning file path only")
            status = f"Success (no OCC): STEP file at {step_path}"
            return (None, step_path, status)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadCADRecodeModel": LoadCADRecodeModel,
    "CADRecodeInference": CADRecodeInference,
    "CadQueryExecute": CadQueryExecute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCADRecodeModel": "Load CAD-Recode Model",
    "CADRecodeInference": "CAD-Recode Inference",
    "CadQueryExecute": "CadQuery Execute",
}
