"""
NeurCADRecon Nodes for ComfyUI-CADabra

Neural implicit CAD surface reconstruction from point clouds.
Enforces zero Gaussian curvature for developable (CAD-like) surfaces.

Paper: NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces (SIGGRAPH 2024)
Authors: Qiujie Dong et al.

Pipeline:
1. LoadNeurCADReconModel - Initialize fresh SIREN network
2. NeurCADReconTrain - Per-object training from point cloud (5-15 min)
3. NeurCADReconInference - Extract mesh via marching cubes
4. NeurCADReconLoadCheckpoint - Load previously saved checkpoint
"""

import os
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

# Optional imports with error handling
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[CADabra] Warning: trimesh not installed.")

try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False
    print("[CADabra] Warning: PyMCubes not installed. NeurCADRecon mesh extraction may use slower fallback.")


# ============================================================================
# Node 1: LoadNeurCADReconModel
# ============================================================================

class LoadNeurCADReconModel:
    """
    Initialize a fresh NeurCADRecon network for per-object training.

    Creates a SIREN-based neural network that learns a signed distance function (SDF).
    The network uses sine activations with geometric initialization optimized
    for implicit surface representation.

    This network needs to be trained on your specific point cloud - it doesn't
    use pretrained weights. Use NeurCADReconTrain to train on your data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run training on. 'auto' uses CUDA if available."
                }),
                "init_type": (["siren", "geometric_sine", "mfgi"], {
                    "default": "siren",
                    "tooltip": "Network initialization type. 'siren' is recommended for most cases."
                }),
            }
        }

    RETURN_TYPES = ("NEURCADRECON_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "CADabra/NeurCADRecon"

    def load_model(
        self,
        device: str = "auto",
        init_type: str = "siren",
    ) -> Tuple:
        """Initialize NeurCADRecon configuration (network is created during training)."""
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[NeurCADRecon] Configured for {device}")
        print(f"[NeurCADRecon] Initialization type: {init_type}")

        # Store configuration only - network will be created during training
        # to avoid ComfyUI's inference_mode issues with network weights
        model_data = {
            "network": None,  # Created during training
            "device": device,
            "trained": False,
            "init_type": init_type,
            "cp": None,
            "scale": None,
            "bbox": None,
            # Network config for lazy initialization
            "network_config": {
                "in_dim": 3,
                "decoder_hidden_dim": 256,
                "nl": "sine",
                "decoder_n_hidden_layers": 4,
                "init_type": init_type,
                "sphere_init_params": [1.6, 1.0],
                "udf": False,
            }
        }

        info_string = (
            f"Model: NeurCADRecon\n"
            f"Device: {device}\n"
            f"Init type: {init_type}\n"
            f"Status: Ready for training (network created on first train)"
        )

        return (model_data, info_string)


# ============================================================================
# Node 2: NeurCADReconTrain
# ============================================================================

class NeurCADReconTrain:
    """
    Train NeurCADRecon on a single point cloud.

    This is a per-object overfitting approach - the network learns to represent
    a single shape at high quality. Training typically takes 5-15 minutes on GPU.

    The key innovation is the Morse loss which enforces zero Gaussian curvature,
    producing sharp edges characteristic of CAD surfaces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NEURCADRECON_MODEL",),
                "point_cloud": ("TRIMESH",),
            },
            "optional": {
                "num_iterations": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 50000,
                    "step": 1000,
                    "tooltip": "Number of training iterations. 10000 is typical for good quality."
                }),
                "batch_size": ("INT", {
                    "default": 20000,
                    "min": 5000,
                    "max": 50000,
                    "step": 5000,
                    "tooltip": "Number of points sampled per iteration. Higher uses more GPU memory."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 5e-5,
                    "min": 1e-6,
                    "max": 1e-3,
                    "step": 1e-6,
                    "tooltip": "Learning rate for Adam optimizer."
                }),
                "loss_preset": (["balanced", "sharp_edges", "smooth_surfaces"], {
                    "default": "balanced",
                    "tooltip": "Loss weight preset. 'sharp_edges' increases morse weight for CAD-like shapes."
                }),
                "save_checkpoint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save checkpoint after training for later use."
                }),
                "checkpoint_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save checkpoint. Empty uses default location."
                }),
                "log_interval": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 2000,
                    "tooltip": "Print loss every N iterations."
                }),
            }
        }

    RETURN_TYPES = ("NEURCADRECON_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("trained_model", "checkpoint_path", "training_log")
    FUNCTION = "train"
    CATEGORY = "CADabra/NeurCADRecon"
    DESCRIPTION = """Train NeurCADRecon to learn an SDF from a point cloud.

Loss Terms (shown during training):
• SDF: Distance error at surface points. Good: < 0.01
• Eikonal: Gradient magnitude should be 1 (|∇f|=1). Good: < 0.1
• Morse: Gaussian curvature regularization for sharp CAD edges. Good: < 1.0
• Total: Weighted sum. Typically starts 100-500, ends 10-50.

Training takes 5-15 min on GPU for 10k iterations."""

    def train(
        self,
        model: Dict[str, Any],
        point_cloud,
        num_iterations: int = 10000,
        batch_size: int = 20000,
        learning_rate: float = 5e-5,
        loss_preset: str = "balanced",
        save_checkpoint: bool = True,
        checkpoint_dir: str = "",
        log_interval: int = 500,
    ) -> Tuple:
        """Train NeurCADRecon on a point cloud."""
        from ..utils.neurcadrecon import (
            TrainingConfig,
            train_neurcadrecon,
        )

        # Extract points from input
        points = self._extract_points(point_cloud)
        print(f"[NeurCADRecon] Training on point cloud with {len(points)} points")

        # Create training config
        config = TrainingConfig(
            num_iterations=num_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_preset=loss_preset,
            save_checkpoint=save_checkpoint,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
        )

        # Get network config and device from model
        # Network is created inside training to avoid ComfyUI inference_mode issues
        network = model.get("network")  # May be None
        network_config = model.get("network_config")
        device = model["device"]

        print(f"[NeurCADRecon] Starting training for {num_iterations} iterations...")
        print(f"[NeurCADRecon] Loss preset: {loss_preset}")
        print(f"[NeurCADRecon] This may take 5-15 minutes on GPU")

        # Train
        result = train_neurcadrecon(
            network=network,
            points=points,
            config=config,
            device=device,
            network_config=network_config,
        )

        # Format training log
        log_lines = [
            f"Training completed",
            f"Final loss: {result['final_loss']:.6f}",
            f"Iterations: {num_iterations}",
        ]
        if result.get("checkpoint_path"):
            log_lines.append(f"Checkpoint: {result['checkpoint_path']}")

        for entry in result.get("training_log", [])[-5:]:
            log_lines.append(
                f"Iter {entry['iteration']}: loss={entry['loss']:.5f}, "
                f"sdf={entry['sdf_term']:.5f}, eik={entry['eikonal_term']:.5f}, "
                f"morse={entry['morse_term']:.5f}"
            )

        training_log = "\n".join(log_lines)
        checkpoint_path = result.get("checkpoint_path", "")

        print(f"[NeurCADRecon] Training complete! Final loss: {result['final_loss']:.6f}")

        return (result, checkpoint_path, training_log)

    def _extract_points(self, point_cloud) -> np.ndarray:
        """Extract numpy points from various input types."""
        if HAS_TRIMESH:
            if isinstance(point_cloud, trimesh.Trimesh) and len(point_cloud.faces) > 0:
                # Sample points from mesh surface
                points, _ = trimesh.sample.sample_surface(point_cloud, 50000)
                return points.astype(np.float32)
            elif isinstance(point_cloud, trimesh.PointCloud):
                return np.asarray(point_cloud.vertices, dtype=np.float32)

        if isinstance(point_cloud, np.ndarray):
            return point_cloud.astype(np.float32)

        if isinstance(point_cloud, dict):
            # Handle CADabra mesh dict format
            if 'mesh' in point_cloud and HAS_TRIMESH:
                mesh = point_cloud['mesh']
                if isinstance(mesh, trimesh.Trimesh):
                    points, _ = trimesh.sample.sample_surface(mesh, 50000)
                    return points.astype(np.float32)
            if 'vertices' in point_cloud:
                return np.asarray(point_cloud['vertices'], dtype=np.float32)
            if 'points' in point_cloud:
                return np.asarray(point_cloud['points'], dtype=np.float32)

        raise ValueError(f"Unsupported point cloud input type: {type(point_cloud)}")


# ============================================================================
# Node 3: NeurCADReconInference
# ============================================================================

class NeurCADReconInference:
    """
    Extract a mesh from a trained NeurCADRecon model.

    Uses marching cubes on the implicit SDF to generate a triangle mesh.
    Higher grid resolution produces more detailed meshes but takes longer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("NEURCADRECON_MODEL",),
            },
            "optional": {
                "grid_resolution": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Resolution of the marching cubes grid. Higher = more detail but slower."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "inference"
    CATEGORY = "CADabra/NeurCADRecon"

    def inference(
        self,
        model: Dict[str, Any],
        grid_resolution: int = 256,
    ) -> Tuple:
        """Extract mesh from trained model."""
        from ..utils.neurcadrecon import extract_mesh

        if not model.get("trained", False):
            raise RuntimeError(
                "Model has not been trained yet. "
                "Please use NeurCADReconTrain first, or load a checkpoint with NeurCADReconLoadCheckpoint."
            )

        print(f"[NeurCADRecon] Extracting mesh at resolution {grid_resolution}...")

        mesh = extract_mesh(
            model_data=model,
            grid_resolution=grid_resolution,
            verbose=True,
        )

        status = f"Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        print(f"[NeurCADRecon] {status}")

        return (mesh, status)


# ============================================================================
# Node 4: NeurCADReconLoadCheckpoint
# ============================================================================

class NeurCADReconLoadCheckpoint:
    """
    Load a trained NeurCADRecon model from a checkpoint file.

    Allows reusing previously trained models without re-training.
    Checkpoints are saved by NeurCADReconTrain when save_checkpoint is enabled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .pth checkpoint file."
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on."
                }),
            }
        }

    RETURN_TYPES = ("NEURCADRECON_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_checkpoint"
    CATEGORY = "CADabra/NeurCADRecon"

    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: str = "auto",
    ) -> Tuple:
        """Load model from checkpoint."""
        from ..utils.neurcadrecon import load_checkpoint as load_ckpt

        if not checkpoint_path or checkpoint_path.strip() == "":
            raise ValueError(
                "checkpoint_path is required.\n\n"
                "This node loads a PREVIOUSLY TRAINED checkpoint.\n"
                "If you want to train a new model, use:\n"
                "  1. 'Load NeurCADRecon Model' (initialize fresh network)\n"
                "  2. 'NeurCADRecon Train' (train on your point cloud)\n"
                "  3. 'NeurCADRecon Inference' (extract mesh)\n\n"
                "The 'Load Checkpoint' node is only for resuming from a saved .pth file."
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[NeurCADRecon] Loading checkpoint from: {checkpoint_path}")
        print(f"[NeurCADRecon] Device: {device}")

        model_data = load_ckpt(checkpoint_path, device=device)

        info_string = (
            f"Model: NeurCADRecon\n"
            f"Device: {device}\n"
            f"Init type: {model_data.get('init_type', 'siren')}\n"
            f"Status: Loaded from checkpoint\n"
            f"Checkpoint: {checkpoint_path}"
        )

        print(f"[NeurCADRecon] Checkpoint loaded successfully")

        return (model_data, info_string)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadNeurCADReconModel": LoadNeurCADReconModel,
    "NeurCADReconTrain": NeurCADReconTrain,
    "NeurCADReconInference": NeurCADReconInference,
    "NeurCADReconLoadCheckpoint": NeurCADReconLoadCheckpoint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNeurCADReconModel": "Load NeurCADRecon Model",
    "NeurCADReconTrain": "NeurCADRecon Train",
    "NeurCADReconInference": "NeurCADRecon Inference",
    "NeurCADReconLoadCheckpoint": "NeurCADRecon Load Checkpoint",
}
