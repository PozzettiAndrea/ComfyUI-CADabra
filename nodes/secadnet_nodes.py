"""
SECAD-Net Nodes for ComfyUI-CADabra

Reconstructs 3D shapes from voxel grids using sketch-extrude CAD primitives.

Paper: SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations
GitHub: https://github.com/BunnySoCrazy/SECAD-Net

Pipeline:
1. MeshToVoxel - Convert mesh to 64³ voxel grid
2. LoadSECADNetModel - Load pretrained SECAD-Net model
3. SECADNetInference - Generate mesh from voxels via latent code
4. SECADNetFinetune - Optional per-shape optimization for better quality
"""

import os
import tempfile
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any

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
    print("[CADabra] Warning: PyMCubes not installed. SECAD-Net mesh generation will not work.")
    print("   Install with: pip install PyMCubes")


# ============================================================================
# Node 1: LoadSECADNetModel
# ============================================================================

class LoadSECADNetModel:
    """
    Load pretrained SECAD-Net model for sketch-extrude CAD reconstruction.

    Automatically downloads pretrained weights from Google Drive if not found locally.
    Models are cached in ComfyUI/models/cadrecon/secadnet/

    The model consists of:
    - Encoder: Converts 64³ voxels to 256D latent code
    - Decoder: Converts latent code to primitive parameters
    - Generator: Computes occupancy from primitives
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["ABC Dataset (recommended)"], {
                    "default": "ABC Dataset (recommended)",
                    "tooltip": "Pretrained model version. ABC Dataset model is recommended."
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on. 'auto' uses CUDA if available."
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Automatically download model if not found locally."
                }),
                "custom_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to custom checkpoint (.pth file). Overrides model_version."
                }),
            }
        }

    RETURN_TYPES = ("SECADNET_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "CADabra/SECAD-Net"

    def load_model(
        self,
        model_version: str,
        device: str = "auto",
        auto_download: bool = True,
        custom_checkpoint: str = "",
    ) -> Tuple:
        """Load SECAD-Net model components."""
        from ..utils.secadnet import Encoder, Decoder, Generator
        from ..utils.model_loader import get_secadnet_model_path, download_secadnet_model

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SECAD-Net] Loading model on {device}")

        # Initialize model components
        # Note: num_primitives=4 is fixed to match the pretrained ABC model
        ef_dim = 32  # Default encoder feature dimension
        num_primitives = 4  # Fixed - must match pretrained model
        encoder = Encoder(ef_dim=ef_dim).to(device)
        decoder = Decoder(ef_dim=ef_dim, num_primitives=num_primitives).to(device)
        generator = Generator(num_primitives=num_primitives, sharpness=150).to(device)

        # Determine checkpoint path
        checkpoint_path = None

        if custom_checkpoint and os.path.exists(custom_checkpoint):
            checkpoint_path = custom_checkpoint
            print(f"[SECAD-Net] Using custom checkpoint: {checkpoint_path}")
        else:
            # Map version to model name
            model_name = "secadnet_abc"  # Default/only option for now

            # Check if model exists locally
            checkpoint_path = get_secadnet_model_path(model_name)

            if checkpoint_path is None and auto_download:
                print(f"[SECAD-Net] Model not found locally, downloading...")
                checkpoint_path = download_secadnet_model(model_name)

        # Load checkpoint if available
        if checkpoint_path and os.path.exists(str(checkpoint_path)):
            print(f"[SECAD-Net] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(str(checkpoint_path), map_location=device)

            # Handle different checkpoint formats
            # Format 1: Original SECAD-Net format with *_state_dict keys
            if 'encoder_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print("[SECAD-Net] Loaded encoder weights")
            if 'decoder_state_dict' in checkpoint:
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                print("[SECAD-Net] Loaded decoder weights")
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                print("[SECAD-Net] Loaded generator weights")

            # Format 2: Simple key format
            if 'encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder'])
            if 'decoder' in checkpoint:
                decoder.load_state_dict(checkpoint['decoder'])
            if 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])

            # Format 3: Combined model_state_dict
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                encoder_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
                decoder_state = {k.replace('decoder.', ''): v for k, v in state.items() if k.startswith('decoder.')}
                generator_state = {k.replace('generator.', ''): v for k, v in state.items() if k.startswith('generator.')}
                if encoder_state:
                    encoder.load_state_dict(encoder_state)
                if decoder_state:
                    decoder.load_state_dict(decoder_state)
                if generator_state:
                    generator.load_state_dict(generator_state)

            print("[SECAD-Net] Checkpoint loaded successfully")
        else:
            print("[SECAD-Net] No checkpoint found, using randomly initialized weights")
            print("[SECAD-Net] For best results, enable auto_download or provide a checkpoint")

        # Set to eval mode
        encoder.eval()
        decoder.eval()
        generator.eval()

        model_data = {
            "encoder": encoder,
            "decoder": decoder,
            "generator": generator,
            "device": device,
            "num_primitives": num_primitives,
            "ef_dim": ef_dim,
        }

        info_string = (
            f"Model: SECAD-Net\n"
            f"Device: {device}\n"
            f"Primitives: {num_primitives}\n"
            f"Encoder dim: {ef_dim * 8} (256D latent)"
        )

        return (model_data, info_string)


# ============================================================================
# Node 2: MeshToVoxel
# ============================================================================

class MeshToVoxel:
    """
    Convert a mesh to a 64³ voxel grid for SECAD-Net input.

    Uses trimesh's voxelization with configurable resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "resolution": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 128,
                    "step": 16,
                    "tooltip": "Voxel grid resolution. SECAD-Net expects 64³."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Normalize mesh to fit in unit cube before voxelization."
                }),
            }
        }

    RETURN_TYPES = ("VOXEL_GRID", "TRIMESH", "STRING")
    RETURN_NAMES = ("voxels", "voxel_mesh", "info")
    FUNCTION = "voxelize"
    CATEGORY = "CADabra/SECAD-Net"

    def voxelize(
        self,
        mesh,
        resolution: int = 64,
        normalize: bool = True,
    ) -> Tuple:
        """Convert mesh to voxel grid."""
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh is required for voxelization")

        # Get mesh from input
        if isinstance(mesh, trimesh.Trimesh):
            input_mesh = mesh
        elif isinstance(mesh, dict) and 'mesh' in mesh:
            input_mesh = mesh['mesh']
        else:
            raise ValueError(f"Unsupported mesh input type: {type(mesh)}")

        print(f"[SECAD-Net] Voxelizing mesh: {len(input_mesh.vertices)} vertices, {len(input_mesh.faces)} faces")

        # Normalize to unit cube if requested
        if normalize:
            # Center at origin
            input_mesh = input_mesh.copy()
            input_mesh.vertices -= input_mesh.bounding_box.centroid
            # Scale to fit in [-0.5, 0.5]
            scale = max(input_mesh.bounding_box.extents)
            if scale > 0:
                input_mesh.vertices /= scale

        # Compute voxel pitch
        bounds = input_mesh.bounds
        extent = bounds[1] - bounds[0]
        pitch = max(extent) / resolution

        # Voxelize and fill interior (SECAD-Net expects solid voxels, not just surface)
        voxel_grid = input_mesh.voxelized(pitch=pitch)
        try:
            # Fill interior voxels for solid shapes
            voxel_grid = voxel_grid.fill()
            print(f"[SECAD-Net] Filled interior voxels")
        except Exception as e:
            print(f"[SECAD-Net] Warning: Could not fill voxels (mesh may not be watertight): {e}")

        # Convert to dense array and ensure correct size
        voxel_array = voxel_grid.matrix.astype(np.float32)

        # Pad or crop to target resolution
        current_shape = voxel_array.shape
        target_shape = (resolution, resolution, resolution)

        if current_shape != target_shape:
            # Create target array
            result = np.zeros(target_shape, dtype=np.float32)

            # Compute offsets to center the voxels
            offsets = [(target_shape[i] - current_shape[i]) // 2 for i in range(3)]

            # Copy voxels with proper bounds
            src_slices = []
            dst_slices = []
            for i in range(3):
                src_start = max(0, -offsets[i])
                src_end = min(current_shape[i], target_shape[i] - offsets[i])
                dst_start = max(0, offsets[i])
                dst_end = dst_start + (src_end - src_start)
                src_slices.append(slice(src_start, src_end))
                dst_slices.append(slice(dst_start, dst_end))

            result[tuple(dst_slices)] = voxel_array[tuple(src_slices)]
            voxel_array = result

        # Create a new VoxelGrid with the properly sized array
        encoding = trimesh.voxel.encoding.DenseEncoding(voxel_array > 0.5)
        output_voxel_grid = trimesh.voxel.VoxelGrid(encoding)

        # Create box mesh for visualization
        voxel_mesh = output_voxel_grid.as_boxes()

        info = (
            f"Voxel grid: {resolution}³\n"
            f"Filled voxels: {int(voxel_array.sum())}\n"
            f"Fill ratio: {voxel_array.mean() * 100:.1f}%"
        )

        print(f"[SECAD-Net] {info.replace(chr(10), ', ')}")

        return (output_voxel_grid, voxel_mesh, info)


# ============================================================================
# Node 3: SECADNetInference
# ============================================================================

class SECADNetInference:
    """
    Run SECAD-Net inference to generate a mesh from voxel input.

    Pipeline:
    1. Encode voxels to 256D latent code
    2. Decode to primitive parameters (quaternions, translations, heights)
    3. Generate occupancy field using neural sketch heads
    4. Extract mesh using marching cubes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SECADNET_MODEL",),
                "voxels": ("VOXEL_GRID",),
            },
            "optional": {
                "grid_resolution": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 256,
                    "step": 32,
                    "tooltip": "Output mesh sampling resolution. Higher = more detailed mesh but slower. This is independent of input voxel resolution (always 64³). Queries the occupancy field at N³ points before marching cubes."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Marching cubes threshold for surface extraction."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SECADNET_LATENT", "STRING")
    RETURN_NAMES = ("mesh", "latent_code", "status")
    FUNCTION = "inference"
    CATEGORY = "CADabra/SECAD-Net"

    def inference(
        self,
        model: Dict[str, Any],
        voxels,  # Can be VoxelGrid or tensor
        grid_resolution: int = 128,
        threshold: float = 0.5,
    ) -> Tuple:
        """Run SECAD-Net inference."""
        if not HAS_MCUBES:
            raise RuntimeError("PyMCubes is required for mesh extraction. Install with: pip install PyMCubes")

        encoder = model["encoder"]
        decoder = model["decoder"]
        generator = model["generator"]
        device = model["device"]

        # Convert VoxelGrid to tensor if needed
        if hasattr(voxels, 'matrix'):  # trimesh.VoxelGrid
            voxel_array = voxels.matrix.astype(np.float32)
            voxels = torch.from_numpy(voxel_array).unsqueeze(0).unsqueeze(0)

        # Move voxels to device
        voxels = voxels.to(device)

        print(f"[SECAD-Net] Running inference...")

        with torch.no_grad():
            # Encode voxels to latent code
            latent_code = encoder(voxels)
            print(f"[SECAD-Net] Latent code: {latent_code.shape}")

            # Decode to primitive parameters
            primitive_params = decoder(latent_code)
            print(f"[SECAD-Net] Primitive params: {primitive_params.shape}")


            # Create query grid matching SECAD-Net's (-0.5, 0.5) range
            # Original: samples[:, :3] = (samples[:, :3]+0.5)/N-0.5
            N = grid_resolution
            x = torch.arange(N).float()
            y = torch.arange(N).float()
            z = torch.arange(N).float()
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            # Normalize to (-0.5, 0.5) like original SECAD-Net
            xx = (xx + 0.5) / N - 0.5
            yy = (yy + 0.5) / N - 0.5
            zz = (zz + 0.5) / N - 0.5
            query_points = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3).to(device)

            print(f"[SECAD-Net] Query points: {query_points.shape}")

            # Generate occupancy field (process in chunks to save memory)
            chunk_size = 100000
            num_points = query_points.shape[1]
            occupancy_list = []

            for i in range(0, num_points, chunk_size):
                chunk = query_points[:, i:i + chunk_size, :]
                occ, _, _ = generator(chunk, primitive_params, latent_code)
                occupancy_list.append(occ.cpu())

            occupancy = torch.cat(occupancy_list, dim=1)
            occupancy_grid = occupancy.reshape(grid_resolution, grid_resolution, grid_resolution).numpy()

        print(f"[SECAD-Net] Occupancy range: [{occupancy_grid.min():.3f}, {occupancy_grid.max():.3f}]")

        # Extract mesh using marching cubes
        vertices, faces = mcubes.marching_cubes(occupancy_grid, threshold)

        # Scale vertices to (-0.5, 0.5) matching original SECAD-Net
        # Original: mesh_points = (mesh_points + 0.5) / N - 0.5
        vertices = (vertices + 0.5) / grid_resolution - 0.5

        print(f"[SECAD-Net] Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")

        # Create trimesh
        output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Store latent code for potential fine-tuning
        latent_data = {
            "code": latent_code.cpu(),
            "primitive_params": primitive_params.cpu(),
        }

        status = f"Success: Generated mesh with {len(vertices)} vertices, {len(faces)} faces"

        return (output_mesh, latent_data, status)


# ============================================================================
# Node 4: SECADNetFinetune
# ============================================================================

class SECADNetFinetune:
    """
    Fine-tune SECAD-Net latent code for improved reconstruction quality.

    Per-shape optimization that adjusts the latent code to better match
    the target shape. Takes ~50 iterations for noticeable improvement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SECADNET_MODEL",),
                "voxels": ("VOXEL_GRID",),
                "latent_code": ("SECADNET_LATENT",),
            },
            "optional": {
                "num_epochs": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 2000,
                    "step": 50,
                    "tooltip": "Number of optimization epochs. 200-500 recommended for good results."
                }),
                "num_samples": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024,
                    "tooltip": "Number of point samples per iteration. Higher = better but slower."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0005,
                    "min": 0.00001,
                    "max": 0.01,
                    "step": 0.0001,
                    "tooltip": "Learning rate for optimization. Lower values (0.0001-0.001) are more stable."
                }),
            }
        }

    RETURN_TYPES = ("SECADNET_LATENT", "STRING")
    RETURN_NAMES = ("optimized_latent", "status")
    FUNCTION = "finetune"
    CATEGORY = "CADabra/SECAD-Net"

    def finetune(
        self,
        model: Dict[str, Any],
        voxels,  # Can be VoxelGrid or tensor
        latent_code: Dict[str, Any],
        num_epochs: int = 50,
        num_samples: int = 4096,
        learning_rate: float = 0.001,
    ) -> Tuple:
        """Fine-tune latent code for better reconstruction."""
        device = model["device"]

        # IMPORTANT: ComfyUI runs with inference_mode(True) globally
        # Model weights loaded in inference mode are "inference tensors"
        # We need to re-create the models inside inference_mode(False) to get normal tensors
        from ..utils.secadnet import Decoder, Generator

        with torch.inference_mode(False):
            # Re-create decoder and generator with normal tensors
            decoder = Decoder(ef_dim=model["ef_dim"], num_primitives=model["num_primitives"]).to(device)
            generator = Generator(num_primitives=model["num_primitives"], sharpness=150).to(device)

            # Copy weights from the original models (clone to get normal tensors)
            decoder.load_state_dict({k: v.clone() for k, v in model["decoder"].state_dict().items()})
            generator.load_state_dict({k: v.clone() for k, v in model["generator"].state_dict().items()})

            # Set to train mode
            decoder.train()
            generator.train()

            # Get initial latent code and make it a learnable parameter
            code = latent_code["code"].to(device).clone().detach().requires_grad_(True)

            # Create optimizer - optimize latent code, decoder, and generator like original
            optimizer = torch.optim.Adam(
                [
                    {"params": [code], "lr": learning_rate},
                    {"params": decoder.parameters(), "lr": learning_rate},
                    {"params": generator.parameters(), "lr": learning_rate},
                ],
                betas=(0.5, 0.999)
            )

            # Get ground truth occupancy from voxels
            if hasattr(voxels, 'matrix'):  # trimesh.VoxelGrid
                voxels_np = voxels.matrix.astype(np.float32)
            else:
                voxels_np = voxels.squeeze().cpu().numpy() if torch.is_tensor(voxels) else voxels.squeeze()

            # Ensure voxels are the right size
            if voxels_np.shape != (64, 64, 64):
                print(f"[SECAD-Net] Warning: Voxel shape is {voxels_np.shape}, expected (64, 64, 64)")
                # Try to handle different shapes
                if len(voxels_np.shape) > 3:
                    voxels_np = voxels_np.squeeze()

            print(f"[SECAD-Net] Starting fine-tuning for {num_epochs} epochs...")

            best_loss = float('inf')
            best_code = code.clone().detach()

            for epoch in range(num_epochs):
                optimizer.zero_grad()

                # Sample random points from voxel grid
                grid_size = voxels_np.shape[0]
                indices = np.random.randint(0, grid_size, size=(num_samples, 3))
                points = (indices + 0.5) / grid_size - 0.5  # Map to (-0.5, 0.5) like SECAD-Net
                gt_occ = voxels_np[indices[:, 0], indices[:, 1], indices[:, 2]]

                points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
                gt_tensor = torch.from_numpy(gt_occ).float().to(device)

                # Forward pass
                primitive_params = decoder(code)

                pred_occ, total_2d_occ, _ = generator(points_tensor, primitive_params, code)

                # Compute loss
                loss = torch.nn.functional.mse_loss(pred_occ.squeeze(), gt_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_code = code.clone().detach()

                if (epoch + 1) % 100 == 0:
                    print(f"[SECAD-Net] Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

            print(f"[SECAD-Net] Fine-tuning complete. Best loss: {best_loss:.6f}")

            # Return optimized latent (still inside inference_mode(False))
            optimized_latent = {
                "code": best_code.cpu(),
                "primitive_params": decoder(best_code).cpu().detach(),
            }

        status = f"Fine-tuning complete. Best loss: {best_loss:.6f}"

        return (optimized_latent, status)


# ============================================================================
# Node 5: SECADNetLatentToMesh
# ============================================================================

class SECADNetLatentToMesh:
    """
    Generate mesh from a SECAD-Net latent code (typically fine-tuned).

    This node bypasses the encoder and directly uses a pre-computed or
    fine-tuned latent code to generate a mesh. Use this after SECADNetFinetune.

    IMPORTANT: SECAD-Net requires fine-tuning per shape for good results.
    The encoder only provides an initialization - fine-tuning is essential!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SECADNET_MODEL",),
                "latent_code": ("SECADNET_LATENT",),
            },
            "optional": {
                "grid_resolution": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 256,
                    "step": 32,
                    "tooltip": "Output mesh sampling resolution."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Marching cubes threshold for surface extraction."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "generate_mesh"
    CATEGORY = "CADabra/SECAD-Net"

    def generate_mesh(
        self,
        model: Dict[str, Any],
        latent_code: Dict[str, Any],
        grid_resolution: int = 128,
        threshold: float = 0.5,
    ) -> Tuple:
        """Generate mesh from latent code."""
        if not HAS_MCUBES:
            raise RuntimeError("PyMCubes is required. Install with: pip install PyMCubes")

        decoder = model["decoder"]
        generator = model["generator"]
        device = model["device"]

        # Get latent code and primitive params
        code = latent_code["code"].to(device)

        print(f"[SECAD-Net] Generating mesh from latent code...")

        with torch.no_grad():
            # Get primitive parameters from decoder
            primitive_params = decoder(code)

            # Create query grid
            N = grid_resolution
            x = torch.arange(N).float()
            y = torch.arange(N).float()
            z = torch.arange(N).float()
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            xx = (xx + 0.5) / N - 0.5
            yy = (yy + 0.5) / N - 0.5
            zz = (zz + 0.5) / N - 0.5
            query_points = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3).to(device)

            # Generate occupancy field
            chunk_size = 100000
            num_points = query_points.shape[1]
            occupancy_list = []

            for i in range(0, num_points, chunk_size):
                chunk = query_points[:, i:i + chunk_size, :]
                occ, _, _ = generator(chunk, primitive_params, code)
                occupancy_list.append(occ.cpu())

            occupancy = torch.cat(occupancy_list, dim=1)
            occupancy_grid = occupancy.reshape(grid_resolution, grid_resolution, grid_resolution).numpy()

        print(f"[SECAD-Net] Occupancy range: [{occupancy_grid.min():.3f}, {occupancy_grid.max():.3f}]")

        # Extract mesh
        vertices, faces = mcubes.marching_cubes(occupancy_grid, threshold)
        vertices = (vertices + 0.5) / grid_resolution - 0.5

        print(f"[SECAD-Net] Generated mesh: {len(vertices)} vertices, {len(faces)} faces")

        output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        status = f"Generated mesh with {len(vertices)} vertices, {len(faces)} faces"

        return (output_mesh, status)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadSECADNetModel": LoadSECADNetModel,
    "MeshToVoxel": MeshToVoxel,
    "SECADNetInference": SECADNetInference,
    "SECADNetFinetune": SECADNetFinetune,
    "SECADNetLatentToMesh": SECADNetLatentToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSECADNetModel": "Load SECAD-Net Model",
    "MeshToVoxel": "Mesh to Voxel",
    "SECADNetInference": "SECAD-Net Inference (Initial)",
    "SECADNetFinetune": "SECAD-Net Fine-tune (Required!)",
    "SECADNetLatentToMesh": "SECAD-Net Latent to Mesh",
}
