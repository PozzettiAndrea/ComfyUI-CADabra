"""
Model loader utilities for Point2CAD and SECAD-Net
Downloads and caches pretrained models from GitHub/Google Drive
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Dict

# Point2CAD model URLs and metadata
# Models are in point2cad/logs/pretrained_models/ subdirectory
# NOTE: Only segmentation models (ParseNet) are publicly available.
#       SplineNet fitting models are NOT released by the authors.
POINT2CAD_MODELS = {
    "parsenet_with_normals": {
        "url": "https://github.com/prs-eth/point2cad/raw/main/point2cad/logs/pretrained_models/parsenet.pth",
        "filename": "parsenet.pth",
        "description": "ParseNet model trained with normal information"
    },
    "parsenet_no_normals": {
        "url": "https://github.com/prs-eth/point2cad/raw/main/point2cad/logs/pretrained_models/parsenet_no_normals.pth",
        "filename": "parsenet_no_normals.pth",
        "description": "ParseNet model for raw point clouds without normals"
    },
    "hpnet": {
        "url": "https://drive.google.com/uc?export=download&id=1fj84kyD9CGT8j61IW-xSWZ5q4q5IpoYx",
        "filename": "hpnet_abc.pth",
        "description": "HPNet model (highest performance, pretrained on ABC dataset)"
    },
    # SplineNet models - NOT publicly available from Point2CAD authors
    # If you have trained your own, place them in the models directory:
    # "closed_spline": {"filename": "closed_spline.pth", ...}
    # "open_spline": {"filename": "open_spline.pth", ...}
}

# SECAD-Net model URLs and metadata
# Pretrained on ABC dataset for sketch-extrude CAD reconstruction
# Original source: https://drive.google.com/drive/folders/1hLJ6JYI1eXoFG-Qc13Bx3o1s-nt9yHkH
SECADNET_MODELS = {
    "secadnet_abc": {
        # Google Drive direct download link (file ID extracted from folder)
        "url": "https://drive.google.com/uc?export=download&id=1hLJ6JYI1eXoFG-Qc13Bx3o1s-nt9yHkH",
        "filename": "ModelParameters/best.pth",  # Downloaded as folder structure
        "description": "SECAD-Net pretrained on ABC dataset (4 primitives)",
        "folder_id": "1hLJ6JYI1eXoFG-Qc13Bx3o1s-nt9yHkH",
    },
}


def get_models_dir() -> Path:
    """
    Get the models directory for Point2CAD models.
    Creates ComfyUI/models/cadrecon/point2cad/ if it doesn't exist.

    Returns:
        Path to models directory
    """
    # Navigate up from ComfyUI-CADabra to ComfyUI/models/cadrecon/point2cad/
    current_dir = Path(__file__).parent.parent  # ComfyUI-CADabra/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "point2cad"

    # Create directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir


def get_model_path(model_name: str) -> Optional[Path]:
    """
    Get the path to a specific Point2CAD model.

    Args:
        model_name: Name of the model (parsenet_with_normals, parsenet_no_normals, hpnet)

    Returns:
        Path to the model file, or None if not found
    """
    if model_name not in POINT2CAD_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(POINT2CAD_MODELS.keys())}")

    models_dir = get_models_dir()
    filename = POINT2CAD_MODELS[model_name]["filename"]
    model_path = models_dir / filename

    if model_path.exists():
        return model_path
    else:
        return None


def download_file(url: str, destination: Path, description: str = "Downloading") -> bool:
    """
    Download a file with progress reporting.

    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"[Download] {description}...")
        print(f"   URL: {url}")
        print(f"   Saving to: {destination}")

        # Create a progress hook
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        # Download the file
        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # New line after progress
        print(f"[OK] Download complete!")
        return True

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        return False


def download_point2cad_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """
    Download a Point2CAD model from GitHub if not already cached.

    Args:
        model_name: Name of the model to download
        force_download: If True, re-download even if file exists

    Returns:
        Path to the downloaded model file, or None if download failed
    """
    if model_name not in POINT2CAD_MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"   Available models: {list(POINT2CAD_MODELS.keys())}")
        return None

    model_info = POINT2CAD_MODELS[model_name]
    models_dir = get_models_dir()
    model_path = models_dir / model_info["filename"]

    # Check if model already exists
    if model_path.exists() and not force_download:
        print(f"[OK] Model already downloaded: {model_path}")
        return model_path

    # Download the model
    print(f"🔍 Model not found locally, downloading {model_name}...")
    print(f"   Description: {model_info['description']}")

    success = download_file(
        url=model_info["url"],
        destination=model_path,
        description=f"Downloading {model_name}"
    )

    if success:
        return model_path
    else:
        return None


def list_available_models() -> Dict[str, bool]:
    """
    List all available Point2CAD models and their download status.

    Returns:
        Dictionary mapping model names to whether they are downloaded
    """
    models_status = {}

    for model_name in POINT2CAD_MODELS.keys():
        model_path = get_model_path(model_name)
        models_status[model_name] = model_path is not None

    return models_status


# ============================================================================
# SECAD-Net Model Loader Functions
# ============================================================================

def get_secadnet_models_dir() -> Path:
    """
    Get the models directory for SECAD-Net models.
    Creates ComfyUI/models/cadrecon/secadnet/ if it doesn't exist.
    """
    current_dir = Path(__file__).parent.parent  # ComfyUI-CADabra/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "secadnet"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_secadnet_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a specific SECAD-Net model."""
    if model_name not in SECADNET_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(SECADNET_MODELS.keys())}")

    models_dir = get_secadnet_models_dir()
    filename = SECADNET_MODELS[model_name]["filename"]
    model_path = models_dir / filename

    if model_path.exists():
        return model_path
    return None


def download_secadnet_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """
    Download a SECAD-Net model from Google Drive if not already cached.

    Note: Google Drive downloads may require gdown for large files.
    Install with: pip install gdown
    """
    if model_name not in SECADNET_MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"   Available models: {list(SECADNET_MODELS.keys())}")
        return None

    model_info = SECADNET_MODELS[model_name]
    models_dir = get_secadnet_models_dir()
    model_path = models_dir / model_info["filename"]

    # Check if model already exists
    if model_path.exists() and not force_download:
        print(f"[OK] Model already downloaded: {model_path}")
        return model_path

    print(f"[SECAD-Net] Model not found locally, downloading {model_name}...")
    print(f"   Description: {model_info['description']}")

    # Try using gdown for Google Drive (handles large files better)
    try:
        import gdown
        folder_id = model_info.get("folder_id")
        if folder_id:
            print(f"[SECAD-Net] Downloading from Google Drive folder...")
            # Download entire folder
            output_dir = str(models_dir)
            gdown.download_folder(
                id=folder_id,
                output=output_dir,
                quiet=False,
            )
            # Check for downloaded .pth files - gdown puts them in ModelParameters/
            pth_files = list(models_dir.glob("**/*.pth"))
            if pth_files:
                # Find best.pth specifically
                best_pth = None
                for pth in pth_files:
                    if pth.name == "best.pth":
                        best_pth = pth
                        break
                if best_pth is None:
                    best_pth = pth_files[0]

                # Ensure model_path parent exists and move file if needed
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if best_pth != model_path:
                    import shutil
                    shutil.copy2(best_pth, model_path)
                print(f"[OK] Model downloaded: {model_path}")
                return model_path
            else:
                print("[WARN] gdown completed but no .pth files found")
    except ImportError:
        print("[WARN] gdown not installed. Install with: pip install gdown")
        print("[WARN] Falling back to direct download (may fail for large files)...")

    # If download failed, provide manual instructions
    print(f"\n[WARN] Automatic download failed. Please download manually:")
    print(f"   1. Visit: https://drive.google.com/drive/folders/{model_info.get('folder_id', '')}")
    print(f"   2. Download the .pth file")
    print(f"   3. Save to: {model_path}")
    return None


# ============================================================================
# BGPSeg Model Loader Functions
# ============================================================================

# BGPSeg model URLs and metadata
# Pretrained on ABC Primitive dataset for boundary-guided primitive segmentation
# Original source: https://drive.google.com/drive/folders/1qev6yadvatGxGm-9HQBNiNIIpDNe9D1A
BGPSEG_MODELS = {
    "bgpseg_abc": {
        "folder_id": "1qev6yadvatGxGm-9HQBNiNIIpDNe9D1A",
        "files": {
            "BGPSeg_model.pth": "BGFE main model (10 primitive types)",
            "Boundary_model.pth": "BoundaryNet boundary predictor",
        },
        "description": "BGPSeg pretrained on ABC Primitive dataset (boundary-guided segmentation)",
    },
}


def get_bgpseg_models_dir() -> Path:
    """
    Get the models directory for BGPSeg models.
    Creates ComfyUI/models/cadrecon/bgpseg/ if it doesn't exist.
    """
    current_dir = Path(__file__).parent.parent  # ComfyUI-CADabra/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "bgpseg"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_bgpseg_model_path(filename: str) -> Optional[Path]:
    """Get the path to a specific BGPSeg model file."""
    models_dir = get_bgpseg_models_dir()
    model_path = models_dir / filename
    if model_path.exists():
        return model_path
    return None


def download_bgpseg_models(force_download: bool = False) -> Optional[Path]:
    """
    Download BGPSeg models from Google Drive if not already cached.

    Downloads both:
    - BGPSeg_model.pth (BGFE main model)
    - Boundary_model.pth (BoundaryNet predictor)

    Note: Google Drive downloads may require gdown for large files.
    Install with: pip install gdown
    """
    model_info = BGPSEG_MODELS["bgpseg_abc"]
    models_dir = get_bgpseg_models_dir()

    # Check if already downloaded
    bgpseg_path = models_dir / "BGPSeg_model.pth"
    boundary_path = models_dir / "Boundary_model.pth"

    if bgpseg_path.exists() and boundary_path.exists() and not force_download:
        print(f"[OK] BGPSeg models already downloaded: {models_dir}")
        return models_dir

    print(f"[BGPSeg] Models not found locally, downloading...")
    print(f"   Description: {model_info['description']}")

    # Try using gdown for Google Drive
    try:
        import gdown
        folder_id = model_info["folder_id"]
        print(f"[BGPSeg] Downloading from Google Drive folder...")

        gdown.download_folder(
            id=folder_id,
            output=str(models_dir),
            quiet=False,
        )

        # Verify downloads
        if bgpseg_path.exists() and boundary_path.exists():
            print(f"[OK] BGPSeg models downloaded: {models_dir}")
            return models_dir
        else:
            print(f"[WARN] Some model files missing after download")

    except ImportError:
        print("[WARN] gdown not installed. Install with: pip install gdown")

    # Provide manual instructions if download failed
    print(f"\n[WARN] Automatic download failed. Please download manually:")
    print(f"   1. Visit: https://drive.google.com/drive/folders/{model_info['folder_id']}")
    print(f"   2. Download both .pth files:")
    for filename, desc in model_info["files"].items():
        print(f"      - {filename} ({desc})")
    print(f"   3. Save to: {models_dir}")
    return None


def list_bgpseg_models() -> Dict[str, bool]:
    """
    List BGPSeg model files and their download status.

    Returns:
        Dictionary mapping filenames to whether they are downloaded
    """
    models_status = {}
    for filename in BGPSEG_MODELS["bgpseg_abc"]["files"].keys():
        model_path = get_bgpseg_model_path(filename)
        models_status[filename] = model_path is not None
    return models_status


if __name__ == "__main__":
    # Test the model loader
    print("Model Loader Test")
    print("=" * 50)

    print(f"\nPoint2CAD models directory: {get_models_dir()}")
    print("Point2CAD models:")
    status = list_available_models()
    for model_name, is_downloaded in status.items():
        status_icon = "+" if is_downloaded else "-"
        description = POINT2CAD_MODELS[model_name]["description"]
        print(f"  {status_icon} {model_name}: {description}")

    print(f"\nSECAD-Net models directory: {get_secadnet_models_dir()}")
    print("SECAD-Net models:")
    for model_name, info in SECADNET_MODELS.items():
        model_path = get_secadnet_model_path(model_name)
        status_icon = "+" if model_path else "-"
        print(f"  {status_icon} {model_name}: {info['description']}")

    print(f"\nBGPSeg models directory: {get_bgpseg_models_dir()}")
    print("BGPSeg models:")
    bgpseg_status = list_bgpseg_models()
    for filename, is_downloaded in bgpseg_status.items():
        status_icon = "+" if is_downloaded else "-"
        desc = BGPSEG_MODELS["bgpseg_abc"]["files"][filename]
        print(f"  {status_icon} {filename}: {desc}")
