# SplineINR - Surface Fitting with Neural Implicit Representation

SplineINR is a neural implicit representation that learns UV parameterization for freeform surfaces. It trains on-the-fly for each surface segment, making it adaptable to arbitrary geometries.

**Reference:** [Point2CAD (CVPR 2024)](https://github.com/prs-eth/point2cad)

## Architecture

- **Type:** Auto-encoder with UV parameterization
- **Encoder:** 3D point coordinates -> 2D UV coordinates
- **Decoder:** 2D UV coordinates -> 3D point coordinates
- **Layers:** Combined SIREN + ResBlock architecture

```
Input (N, 3) -> Encoder -> UV (N, 2) -> Decoder -> Output (N, 3)
```

## Input Format

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `points` | `(N, 3)` | Point cloud segment (XYZ) |

**Requirements:**
- Minimum 20 points per segment
- Points should be from a single surface (after segmentation)

## Training

SplineINR trains **on-the-fly** for each surface segment:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_fit_steps` | 1000 | Training iterations |
| `lr` | 0.1 | Initial learning rate |
| `device` | `cuda` | Training device |

**Training time:** ~1-2 seconds per surface on GPU

**Process:**
1. Normalize points (center, scale by std)
2. Split 90/10 train/validation
3. Train with L1 loss and learning rate decay
4. Evaluate on validation set

## Output Format

| Output | Type | Description |
|--------|------|-------------|
| `model` | `SplineINR` | Trained neural network |
| `err` | `float` | Mean reconstruction error |
| `val_err_l2` | `float` | Validation L2 error |
| `is_good_fit` | `bool` | True if `val_err < 1e-4` |
| `mesh_uv` | `trimesh.Trimesh` | Sampled surface mesh |

## Closure Modes

For surfaces with periodic boundaries:

| Mode | Description | Example |
|------|-------------|---------|
| `is_u_closed=False, is_v_closed=False` | Open surface | Flat panel |
| `is_u_closed=True, is_v_closed=False` | Closed in U | Cylinder (around axis) |
| `is_v_closed=True, is_u_closed=False` | Closed in V | Cylinder (along axis) |
| `is_u_closed=True, is_v_closed=True` | Doubly closed | Torus |

The fitting pipeline tries all 4 combinations and selects the best fit.

## Surface Fitting Pipeline

`process_one_surface()` fits both primitives and SplineINR, then selects the best:

### Primitives (Analytical Fitting)

| Surface | Method | Parameters |
|---------|--------|------------|
| **Plane** | Weighted SVD | normal, distance |
| **Sphere** | Least squares | center, radius |
| **Cylinder** | Eberly's method | axis, center, radius |
| **Cone** | Scipy `least_squares` | apex, axis, half-angle |

### Freeform (Neural Fitting)

| Surface | Method | Parameters |
|---------|--------|------------|
| **Open Spline** | SplineINR | trained model, UV bounds |

### Selection Logic

```python
# Collect all errors
errors = [plane_err, sphere_err, cylinder_err, cone_err, inr_err]
best_idx = argmin(errors)

# Prefer primitives if error is similar
if best_idx == 4:  # INR was best
    prim_min = min(plane_err, sphere_err, cylinder_err, cone_err)
    if prim_min < 0.008 or prim_min < inr_err + 0.001:
        best_idx = argmin([plane_err, sphere_err, cylinder_err, cone_err])
```

## Mesh Sampling

After fitting, sample a mesh from the trained INR:

```python
from utils.point2cad_fitting.spline_inr import sample_inr_mesh

mesh = sample_inr_mesh(
    fit_out,           # Output from fit_one_inr_spline
    mesh_dim=50,       # Grid resolution (50x50)
    uv_margin=0.2      # Margin around UV bounds
)
```

Returns a `trimesh.Trimesh` with:
- Vertices: Sampled 3D points from decoder
- Faces: Regular grid triangulation

## Usage in ComfyUI

1. **Segmentation:** Use `Point2CAD Segmentation` to get surface clusters
2. **Fitting:** Connect to `Point2CAD Surface Fitting` node
   - `use_inr=True`: Enable SplineINR for freeform surfaces
   - `device="cuda"`: Use GPU for faster training

## Example Output

```
Segment 0: 1234 points -> plane (err: 0.0012)
Segment 1: 567 points -> cylinder (err: 0.0034)
Segment 2: 890 points -> open_spline (err: 0.0089)
```

## Known Limitations

- INR fitting requires GPU for reasonable speed (CPU is ~10x slower)
- Very small segments (<20 points) are skipped
- Complex multi-lobed surfaces may not fit well
- `is_good_fit` threshold (`val_err < 1e-4`) may be too strict for some geometries
