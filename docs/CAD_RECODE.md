# CAD-Recode Integration Guide

## Overview

CAD-Recode is a neural network that generates executable CadQuery Python code directly from 3D point clouds. Unlike traditional CAD reconstruction methods that fit geometric primitives, CAD-Recode produces human-readable parametric code that can be edited and modified.

**Paper**: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds" (ICCV 2025)
**Authors**: Rukhovich et al., University of Luxembourg
**Project Page**: https://cad-recode.github.io/

---

## How It Works

CAD-Recode uses a vision-language model architecture:

1. **Point Cloud Encoding**: Input points are encoded using Fourier positional encoding
2. **Code Generation**: A fine-tuned Qwen2-1.5B language model generates CadQuery code autoregressively
3. **Execution**: The generated code is executed to produce a STEP file

The model was trained on the DeepCAD dataset with ~180,000 CAD models converted to CadQuery format.

---

## Node Pipeline

CAD-Recode uses a 3-node pipeline:

### 1. Load CAD-Recode Model

Downloads and loads the CAD-Recode model from HuggingFace.

**Parameters**:
- `model_version`: Choose between v1.5 (recommended) or v1 (original)
- `device`: "auto", "cuda", or "cpu"
- `use_flash_attention`: Enable Flash Attention 2 for faster inference

**Outputs**:
- `model`: Loaded model ready for inference
- `model_info`: Model metadata string

### 2. CAD-Recode Inference

Generates CadQuery code from a point cloud.

**Inputs**:
- `model`: Model from LoadCADRecodeModel
- `point_cloud`: TRIMESH (mesh or point cloud)

**Parameters**:
- `num_points`: Number of points to sample (see detailed explanation below)
- `max_tokens`: Maximum tokens to generate (see detailed explanation below)
- `normalize`: Normalize point cloud to [-1, 1] cube
- `use_fps`: Use Farthest Point Sampling (recommended for v1.5)

**Outputs**:
- `cadquery_code`: Generated Python code
- `code_preview`: First 500 characters preview

### 3. CAD-Recode Execute

Safely executes the generated CadQuery code.

**Parameters**:
- `cadquery_code`: Code to execute
- `timeout`: Maximum execution time (1-60 seconds)
- `validate_first`: Check syntax before execution

**Outputs**:
- `cad_model`: CAD_MODEL for downstream nodes
- `step_file`: Path to generated STEP file
- `status`: Execution status message

---

## Understanding num_points and max_tokens

### num_points

**What it does**: Controls how many points are sampled from your input point cloud before being fed to the neural network.

**How it works**:
1. Your input mesh/point cloud may have thousands or millions of points
2. The model samples exactly `num_points` from this input
3. If `use_fps=True`, Farthest Point Sampling ensures uniform coverage
4. If `use_fps=False`, random sampling is used

**Training configuration**: CAD-Recode was trained with **256 points**

**Recommended values**:
| Value | Use Case |
|-------|----------|
| 256 (default) | Best overall accuracy - matches training |
| 128 | Faster inference, may lose fine details |
| 512-1024 | Experimental - more detail but not trained this way |
| 64 | Minimum useful - significant detail loss |

**Why 256 works best**: The model learned to interpret geometric features from exactly 256 points during training. Using the same count gives the most reliable results.

---

### max_tokens

**What it does**: Limits the maximum length of generated CadQuery code (measured in tokens, roughly words/symbols).

**How it works**:
1. The language model generates code one token at a time
2. Generation stops when either:
   - An end-of-sequence token is produced
   - The `max_tokens` limit is reached
3. If code is cut off mid-statement, the output will be invalid

**Impact on output**:
- **Too low**: Code may be truncated, causing syntax errors
- **Too high**: No harm, but wastes compute if unused
- **Just right**: Complete code with efficient generation

**Recommended values**:
| Value | Use Case |
|-------|----------|
| 256-384 | Simple shapes (box, cylinder, basic extrusions) |
| 512-768 | Moderate complexity (multiple operations, chamfers) |
| 768-1024 | Complex models (many features, boolean operations) |
| 1024-2048 | Very complex models (intricate details, many steps) |

**How to tell if you need more tokens**:
- Generated code ends abruptly (e.g., `result = cq.Workplane("XY").box(`)
- Syntax errors when executing
- Missing closing parentheses or statements

**Example token counts for reference**:
```python
# Simple box (~50 tokens)
import cadquery as cq
r = cq.Workplane("XY").box(10, 10, 5)

# Moderate complexity (~200 tokens)
import cadquery as cq
r = cq.Workplane("XY").box(10, 10, 5).faces(">Z").workplane().hole(3)

# Complex model (~500+ tokens)
import cadquery as cq
r = (cq.Workplane("XY")
     .box(20, 20, 10)
     .faces(">Z").workplane()
     .pushPoints([(5, 5), (-5, -5), (5, -5), (-5, 5)])
     .hole(2)
     .faces("<Z").workplane()
     .rect(15, 15).extrude(-5)
     .edges("|Z").fillet(1))
```

---

## Model Versions

### v1.5 (Recommended)
- Improved training with Farthest Point Sampling
- Larger coordinate range support
- Better generalization to unseen shapes
- HuggingFace: `filapro/cad-recode-v1.5`

### v1 (Original)
- Original paper release
- Uses random sampling
- HuggingFace: `filapro/cad-recode`

---

## Example Workflow

```
[Load 3D Mesh]
    ↓
[MeshToPointCloud] (optional, if starting from mesh)
    ↓
[Load CAD-Recode Model] (v1.5, cuda)
    ↓
[CAD-Recode Inference] (num_points=256, max_tokens=768)
    ↓
[CAD-Recode Execute] (timeout=10s)
    ↓
[CAD Model Output] → STEP file + preview mesh
```

---

## Performance

### Typical Timings (RTX 3090):
- Model loading: 5-10 seconds (first time, cached after)
- Inference: 2-5 seconds per model
- Code execution: 1-10 seconds depending on complexity

### Memory Requirements:
- GPU: ~4-6 GB VRAM
- CPU: Works but slower (10-30x)

---

## Troubleshooting

### Code execution fails with syntax error
- **Cause**: `max_tokens` too low, code truncated
- **Solution**: Increase `max_tokens` to 1024 or higher

### Output doesn't match input shape
- **Cause**: Point sampling lost important details
- **Solution**: Ensure input has enough points, keep `num_points=256`

### Model produces nonsense code
- **Cause**: Input point cloud not normalized or too sparse
- **Solution**: Enable `normalize=True`, ensure adequate input point count

### CUDA out of memory
- **Cause**: Flash Attention not available
- **Solution**: Set `device="cpu"` or install flash-attn package

### Execution timeout
- **Cause**: Generated code has infinite loop or complex geometry
- **Solution**: Increase `timeout` or simplify input

---

## Comparison with Point2CAD

| Feature | CAD-Recode | Point2CAD |
|---------|------------|-----------|
| **Output** | Editable Python code | B-rep geometry |
| **Primitives** | Via code (box, cylinder, etc.) | Direct fitting |
| **Freeform** | Limited (code-based) | Neural INR surfaces |
| **Editability** | Excellent (source code) | Limited |
| **Speed** | Fast (~5 seconds) | Slower (~1-2 minutes) |
| **Best for** | Parametric models, quick prototyping | Complex freeform surfaces |

**Recommendation**: Use CAD-Recode for quick parametric reconstruction when you want editable code. Use Point2CAD for complex organic shapes requiring precise surface fitting.

---

## References

1. **CAD-Recode Paper**: https://cad-recode.github.io/
2. **HuggingFace Models**: https://huggingface.co/filapro
3. **CadQuery Documentation**: https://cadquery.readthedocs.io/
4. **DeepCAD Dataset**: https://github.com/ChrisWu1997/DeepCAD
