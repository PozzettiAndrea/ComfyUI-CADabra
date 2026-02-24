"""
NeurCADRecon Training Module

Training loop wrapper for ComfyUI integration.
Adapted from NeurCADRecon (SIGGRAPH 2024) paper by Dong et al.
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from .network import Network
from .losses import MorseLoss, get_loss_weights
from .dataset import ReconDataset
from .utils import implicit2mesh, count_parameters


class TrainingConfig:
    """Configuration for NeurCADRecon training."""

    def __init__(
        self,
        num_iterations: int = 10000,
        batch_size: int = 20000,
        learning_rate: float = 5e-5,
        loss_preset: str = 'balanced',
        grid_resolution: int = 256,
        grid_range: float = 1.1,
        grad_clip_norm: float = 10.0,
        log_interval: int = 500,
        mesh_interval: int = 2000,
        save_checkpoint: bool = True,
        checkpoint_dir: str = "",
        morse_near: bool = True,
        bidirectional_morse: bool = True,
    ):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_preset = loss_preset
        self.grid_resolution = grid_resolution
        self.grid_range = grid_range
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.mesh_interval = mesh_interval
        self.save_checkpoint = save_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.morse_near = morse_near
        self.bidirectional_morse = bidirectional_morse


def get_default_checkpoint_dir() -> Path:
    """Get default checkpoint directory in ComfyUI output folder."""
    current_dir = Path(__file__).parent.parent.parent  # utils/neurcadrecon -> ComfyUI-CADabra
    comfyui_dir = current_dir.parent.parent  # custom_nodes -> ComfyUI
    checkpoint_dir = comfyui_dir / "output" / "neurcadrecon_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(model_data: Dict[str, Any], checkpoint_dir: Path,
                    prefix: str = "neurcadrecon") -> str:
    """Save training checkpoint.

    Args:
        model_data: Model data dictionary with network, cp, scale, etc.
        checkpoint_dir: Directory to save checkpoint
        prefix: Filename prefix

    Returns:
        Path to saved checkpoint
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pth"
    path = checkpoint_dir / filename

    checkpoint = {
        "state_dict": model_data["network"].state_dict(),
        "cp": model_data.get("cp"),
        "scale": model_data.get("scale"),
        "bbox": model_data.get("bbox"),
        "config": {
            "init_type": model_data.get("init_type", "siren"),
            "decoder_hidden_dim": 256,
            "decoder_n_hidden_layers": 4,
        }
    }

    torch.save(checkpoint, path)
    return str(path)


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Target device

    Returns:
        Model data dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct network
    config = checkpoint.get("config", {})
    net = Network(
        in_dim=3,
        decoder_hidden_dim=config.get("decoder_hidden_dim", 256),
        nl='sine',
        decoder_n_hidden_layers=config.get("decoder_n_hidden_layers", 4),
        init_type=config.get("init_type", "siren"),
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()

    return {
        "network": net,
        "device": device,
        "trained": True,
        "init_type": config.get("init_type", "siren"),
        "cp": checkpoint.get("cp"),
        "scale": checkpoint.get("scale"),
        "bbox": checkpoint.get("bbox"),
    }


def train_neurcadrecon(
    network,  # Can be Network instance or None
    points: np.ndarray,
    config: TrainingConfig,
    device: str = "cuda",
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    check_interrupted: Optional[Callable[[], bool]] = None,
    network_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Train NeurCADRecon network on a point cloud.

    Args:
        network: Initialized Network instance, or None if network_config is provided
        points: Point cloud as numpy array (N, 3)
        config: Training configuration
        device: Training device ('cuda' or 'cpu')
        progress_callback: Optional callback(iteration, total, loss_dict)
        check_interrupted: Optional callback to check for interruption
        network_config: Network configuration dict (used if network is None)

    Returns:
        Dictionary with training results:
            network: Trained network
            cp: Center point for denormalization
            scale: Scale factor for denormalization
            bbox: Bounding box
            checkpoint_path: Path to saved checkpoint (if enabled)
            final_loss: Final loss value
            training_log: List of logged loss values
    """
    # ComfyUI runs with inference_mode(True) globally - we need to escape it for training
    # Wrap the entire training in inference_mode(False)
    return _train_neurcadrecon_impl(
        network, points, config, device, progress_callback, check_interrupted, network_config
    )


def _train_neurcadrecon_impl(
    network,
    points: np.ndarray,
    config: TrainingConfig,
    device: str = "cuda",
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    check_interrupted: Optional[Callable[[], bool]] = None,
    network_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Internal training implementation wrapped in inference_mode(False)."""
    # ComfyUI runs with inference_mode(True) globally - escape it for training
    with torch.inference_mode(False):
        return _do_training(
            network, points, config, device, progress_callback, check_interrupted, network_config
        )


def _do_training(
    network,  # Can be Network or None (with network_config)
    points: np.ndarray,
    config: TrainingConfig,
    device: str = "cuda",
    progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    check_interrupted: Optional[Callable[[], bool]] = None,
    network_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Actual training logic."""
    # Create network inside inference_mode(False) if not provided
    if network is None and network_config is not None:
        print("[NeurCADRecon] Creating network inside training context...")
        network = Network(**network_config)
        network.to(device)
        print(f"[NeurCADRecon] Network created with {count_parameters(network):,} parameters")

    # Create dataset
    dataset = ReconDataset(
        points=points,
        n_points=config.batch_size,
        n_samples=config.num_iterations,
        grid_range=config.grid_range,
    )

    # Get normalization parameters
    cp, scale, bbox = dataset.get_normalization_params()

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # ComfyUI compatibility
        pin_memory=True if device == "cuda" else False,
    )

    # Move network to device
    network.to(device)
    network.train()

    # Setup optimizer
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=0.0)

    # Setup AMP for mixed precision training (significant speedup on GPU)
    use_amp = device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Setup loss
    loss_weights = get_loss_weights(config.loss_preset)
    criterion = MorseLoss(
        weights=loss_weights,
        loss_type='siren_wo_n_w_morse',
        div_decay='none',
        bidirectional_morse=config.bidirectional_morse,
    )

    # Training loop
    training_log = []
    final_loss = float('inf')
    start_time = time.time()

    for batch_idx, data in enumerate(train_dataloader):
        # Check for interruption
        if check_interrupted is not None and check_interrupted():
            print("[NeurCADRecon] Training interrupted by user")
            break

        # Move data to device, clone to escape any inference mode issues, and enable gradients
        # Clone is important to create fresh tensors that can participate in autograd
        mnfld_points = data['points'].to(device).clone().requires_grad_(True)
        mnfld_n_gt = data['mnfld_n'].to(device).clone()
        nonmnfld_points = data['nonmnfld_points'].to(device).clone().requires_grad_(True)
        near_points = data['near_points'].to(device).clone().requires_grad_(True)

        # Forward pass with AMP
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output_pred = network(
                nonmnfld_points, mnfld_points,
                near_points=near_points if config.morse_near else None
            )

            # Compute loss
            loss_dict, _ = criterion(
                output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt,
                near_points=near_points if config.morse_near else None
            )

        # Backward pass with scaled gradients
        scaler.scale(loss_dict["loss"]).backward()

        # Gradient clipping (must unscale first)
        if config.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # Record loss
        final_loss = loss_dict["loss"].item()

        # Progress callback
        if progress_callback is not None:
            progress_callback(batch_idx, config.num_iterations, {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            })

        # Logging with timing
        if batch_idx % config.log_interval == 0:
            log_entry = {
                'iteration': batch_idx,
                'loss': final_loss,
                'sdf_term': loss_dict['sdf_term'].item(),
                'eikonal_term': loss_dict['eikonal_term'].item(),
                'morse_term': loss_dict['morse_term'].item() if torch.is_tensor(loss_dict['morse_term']) else loss_dict['morse_term'],
            }
            training_log.append(log_entry)

            # Calculate timing and ETA
            elapsed = time.time() - start_time
            if batch_idx > 0:
                its = batch_idx / elapsed
                remaining_iters = config.num_iterations - batch_idx
                eta_seconds = remaining_iters / its
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"{eta_min}:{eta_sec:02d}"
            else:
                its = 0.0
                eta_str = "calculating..."

            pct = 100.0 * batch_idx / config.num_iterations

            print(f"[NeurCADRecon] {batch_idx}/{config.num_iterations} ({pct:.1f}%) | "
                  f"{its:.1f} it/s | ETA: {eta_str} | "
                  f"Loss: {final_loss:.4f} (sdf={log_entry['sdf_term']:.4f}, eik={log_entry['eikonal_term']:.3f}, morse={log_entry['morse_term']:.3f})")

    # Prepare result
    network.eval()

    result = {
        "network": network,
        "device": device,
        "trained": True,
        "init_type": network.init_type,
        "cp": cp,
        "scale": scale,
        "bbox": bbox,
        "final_loss": final_loss,
        "training_log": training_log,
        "checkpoint_path": None,
    }

    # Save checkpoint if requested
    if config.save_checkpoint:
        if config.checkpoint_dir:
            checkpoint_dir = Path(config.checkpoint_dir)
        else:
            checkpoint_dir = get_default_checkpoint_dir()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result["checkpoint_path"] = save_checkpoint(result, checkpoint_dir)
        print(f"[NeurCADRecon] Checkpoint saved to: {result['checkpoint_path']}")

    return result


def extract_mesh(
    model_data: Dict[str, Any],
    grid_resolution: int = 256,
    verbose: bool = True,
) -> "trimesh.Trimesh":
    """Extract mesh from trained model.

    Args:
        model_data: Model data dictionary from training
        grid_resolution: Marching cubes grid resolution
        verbose: Whether to print progress

    Returns:
        trimesh.Trimesh object
    """
    network = model_data["network"]
    device = model_data.get("device", "cuda")
    cp = model_data.get("cp", np.zeros(3))
    scale = model_data.get("scale", 1.0)
    bbox = model_data.get("bbox", np.array([[-1, 1], [-1, 1], [-1, 1]]))

    network.eval()

    mesh = implicit2mesh(
        network.decoder,
        mods=None,
        grid_res=grid_resolution,
        translate=-cp,
        scale=1 / scale,
        get_mesh=True,
        device=device,
        bbox=bbox,
        verbose=verbose,
    )

    return mesh
