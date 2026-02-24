"""
CAD-Recode Integration for ComfyUI-CADabra

Paper: CAD-Recode: Reverse Engineering CAD Code from Point Clouds (ICCV 2025)
Authors: Rukhovich et al., University of Luxembourg
Project: https://cad-recode.github.io/
GitHub: https://github.com/filaPro/cad-recode

CAD-Recode transforms 3D point clouds into executable CadQuery Python code,
achieving state-of-the-art performance on DeepCAD, Fusion360, and CC3D benchmarks.

Usage:
    from utils.cadrecode import CADRecode, generate_code, execute_cadquery_code

    # Load model
    model = CADRecode.from_pretrained('filapro/cad-recode-v1.5')
    tokenizer = load_tokenizer()

    # Generate code from point cloud
    code = generate_code(model, tokenizer, point_cloud)

    # Execute safely
    success, result, stl_path = execute_cadquery_code(code)
"""

from .model import CADRecode, FourierPointEncoder
from .inference import (
    generate_code,
    normalize_point_cloud,
    sample_points_fps,
    sample_points_random,
    mesh_to_point_cloud,
    prepare_input,
    load_tokenizer,
)
from .execution import (
    execute_cadquery_code,
    execute_cadquery_code_persistent,
    validate_cadquery_code,
    cleanup_temp_files,
)

__all__ = [
    # Model
    'CADRecode',
    'FourierPointEncoder',
    # Inference
    'generate_code',
    'normalize_point_cloud',
    'sample_points_fps',
    'sample_points_random',
    'mesh_to_point_cloud',
    'prepare_input',
    'load_tokenizer',
    # Execution
    'execute_cadquery_code',
    'execute_cadquery_code_persistent',
    'validate_cadquery_code',
    'cleanup_temp_files',
]

__version__ = '1.0.0'
