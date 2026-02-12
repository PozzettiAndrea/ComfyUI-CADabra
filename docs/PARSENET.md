# ParseNet - Point Cloud Segmentation Model

ParseNet is a DGCNN-based neural network for segmenting point clouds into CAD surface primitives. It predicts per-point primitive types and generates embeddings for clustering.

**Reference:** [Point2CAD (CVPR 2024)](https://github.com/prs-eth/point2cad)

## Architecture

- **Encoder:** Dynamic Graph CNN (DGCNN) with EdgeConv layers
- **Class:** `PrimitivesEmbeddingDGCNGn` in `utils/parsenet/model.py`
- **Features:**
  - 4 EdgeConv layers with k-NN graph construction
  - GroupNorm for batch-independent normalization
  - Per-point feature extraction with global context

## Input Format

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `points` | `(batch, C, N)` | Point cloud tensor |

Where:
- `C = 3`: XYZ coordinates only (for `parsenet_no_normals`)
- `C = 6`: XYZ + normals (for `parsenet_with_normals`, `hpnet`)
- `N`: Number of points (variable, typically 4096-16384)

**Preprocessing:**
1. Center at mean
2. PCA alignment (flattest direction aligned with X-axis)
3. Scale by max bounding box extent

## Output Format

| Output | Shape | Description |
|--------|-------|-------------|
| `embeddings` | `(batch, 128, N)` | Per-point feature embeddings |
| `primitive_log_prob` | `(batch, 10, N)` | Log probabilities for primitive types |

## Primitive Types

The network predicts 10 primitive types per point:

| Type ID | Surface Type | Description |
|---------|-------------|-------------|
| 0 | Other/Background | Unclassified points |
| 1 | Plane | Flat surfaces |
| 2 | B-spline | Freeform/open spline surfaces |
| 3 | Cone | Conical surfaces |
| 4 | Cylinder | Cylindrical surfaces |
| 5 | Sphere | Spherical surfaces |
| 6-9 | Reserved | Other surface types (torus, etc.) |

## Clustering (Mean Shift)

After inference, points are clustered using Mean Shift on normalized embeddings:

```python
from utils.parsenet.mean_shift import MeanShift

ms = MeanShift()
embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
_, _, cluster_ids = ms.guard_mean_shift(
    embeddings_normalized,
    quantile=0.015,      # Bandwidth parameter
    iterations=50,
    kernel_type="gaussian"
)
```

**Parameters:**
- `quantile`: Controls cluster granularity (lower = more clusters). Default: 0.015
- `iterations`: Max Mean Shift iterations. Default: 50

## Pre-trained Weights

| Model | Input Channels | Description |
|-------|---------------|-------------|
| `parsenet_with_normals` | 6 | Points + normals, general purpose |
| `parsenet_no_normals` | 3 | Points only, when normals unavailable |
| `hpnet` | 6 | Highest performance, trained on ABC dataset |

Weights are automatically downloaded to `ComfyUI/models/cadrecon/point2cad/`

## Usage in ComfyUI

1. **Load Model:** Use `Load Point2CAD Model` node
2. **Segment:** Connect to `Point2CAD Segmentation` node
3. **Outputs:**
   - `segmented_cloud`: TRIMESH with `vertex_attributes['label']` and `vertex_attributes['primitive_type']`
   - `summary`: Text summary of segmentation results

## Example Pipeline

```
LoadPoint2CADModel(model="ParseNet (with normals)")
    |
    v
Point2CADSegmentation(
    point_cloud=<input>,
    batch_size=4096,
    confidence_threshold=0.5,
    meanshift_quantile=0.015
)
    |
    v
Point2CADSurfaceFitting(...)
```
