# Point2CAD Integration Guide

## Overview

Point2CAD is a state-of-the-art method from CVPR 2024 for reconstructing CAD models from 3D point clouds using a hybrid analytic-neural approach. Unlike RANSAC-based methods which struggle with freeform surfaces, Point2CAD handles both primitives (planes, cylinders, spheres, cones) AND complex freeform surfaces like the Utah teapot's spout and handle.

**Paper**: "Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds"
**Project Page**: https://www.obukhov.ai/point2cad.html
**GitHub**: https://github.com/prs-eth/point2cad

---

## Why Point2CAD is Better Than RANSAC

| Feature | RANSAC | Point2CAD |
|---------|--------|-----------|
| **Primitives** | ✅ Planes, cylinders, spheres | ✅ Planes, cylinders, spheres, cones |
| **Freeform Surfaces** | ❌ Not supported | ✅ Neural implicit fields |
| **Parameter Tuning** | ❌ Manual per-input | ✅ Automatic via neural network |
| **Accuracy** | ⚠️ Sensitive to noise | ✅ 18%+ better than RANSAC-based methods |
| **Complex Shapes** | ❌ Fails on Utah teapot | ✅ Handles teapot, automotive parts, etc. |

---

## Installation

### 1. Install Dependencies

The Point2CAD nodes require additional Python packages. Install them using:

```bash
cd ComfyUI/custom_nodes/ComfyUI-CADabra
pip install -r requirements.txt
```

New dependencies added:
- `geomdl>=5.3.0` - NURBS/B-spline operations
- `pyvista>=0.43.0` - Advanced mesh processing and visualization
- `rtree>=1.0.0` - Spatial indexing for surface intersection

### 2. Model Downloads

Models are automatically downloaded when first used. They will be cached in:
```
ComfyUI/models/cadrecon/point2cad/
```

Available models:
- **parsenet_with_normals.pth** (~50-200 MB) - For point clouds with normals
- **parsenet_no_normals.pth** (~50-200 MB) - For raw point clouds
- **hpnet_abc.pth** (~50-200 MB) - Highest performance, pretrained on ABC dataset

You can also manually download models using:
```bash
cd ComfyUI/custom_nodes/ComfyUI-CADabra
python -m utils.model_loader
```

---

## Node Pipeline

Point2CAD reconstruction uses a 5-node pipeline:

### 1. **LoadPoint2CADModel**
Downloads and loads the segmentation network (ParseNet or HPNet).

**Parameters**:
- `model`: Choose model architecture:
  - `ParseNet (with normals)`: For point clouds with normal vectors (6-channel input)
  - `ParseNet (no normals)`: For raw point clouds only (3-channel input)
  - `HPNet (ABC dataset)`: Highest performance model, pretrained on ABC dataset
- `auto_download`: Automatically download if model is missing (default: True)
- `force_redownload`: Force re-download even if model exists

**Outputs**:
- `model`: Loaded Point2CAD segmentation model
- `model_info`: Model metadata string

#### Understanding Model Mode

The model uses a DGCNN (Dynamic Graph CNN) encoder with different "modes" for handling normals:

| Mode | First Layer KNN | Description |
|------|-----------------|-------------|
| **0** | Euclidean distance | Standard KNN treating all channels equally |
| **5** | Weighted distance | `pos_dist * (1 + normal_dist)` - emphasizes position over normals |

**Current Configuration**: Both `parsenet_with_normals` and `hpnet` use `mode=0` with 6 channels (matching the original Point2CAD paper). This means normals are included in the input but the KNN graph is computed using standard Euclidean distance in 6D space.

---

### 2. **Point2CADSegmentation**
Segments point cloud into surface clusters using ParseNet neural network with Mean Shift clustering.

**Inputs**:
- `point_cloud`: TRIMESH or POINT_CLOUD
- `model`: Point2CAD model from LoadPoint2CADModel node

**Parameters**:
- `batch_size`: Points per inference batch (default: 4096)
- `confidence_threshold`: Minimum confidence for segment assignment (default: 0.5)
- `use_normals`: Use normal information for segmentation (default: True)
- `meanshift_quantile`: **Critical parameter** - Controls segmentation granularity (default: 0.015)
  - Lower values (0.01-0.02) → More segments, finer detail
  - Higher values (0.05-0.1) → Fewer segments, merged surfaces
  - Original Point2CAD paper uses 0.015
- `min_cluster_points`: Minimum points per cluster for DBSCAN fallback (default: 10)

**Outputs**:
- `segmented_cloud`: Point cloud with segment labels and primitive type predictions
- `preview_cloud`: Colored point cloud for visualization
- `summary`: Segmentation statistics

#### Preprocessing Pipeline

The segmentation node normalizes input points using PCA alignment (matching original Point2CAD):

1. **Center**: Subtract mean from all points
2. **PCA Rotation**: Align the "flattest" direction with the X-axis
3. **Scale**: Normalize by max bounding box extent

This canonical orientation helps the network generalize better across different input orientations.

#### Mean Shift Clustering

After the neural network extracts 128-dimensional embeddings for each point, Mean Shift clustering groups points into surface segments:

1. Embeddings are L2-normalized to lie on a unit hypersphere
2. Bandwidth is computed from K-nearest neighbor distances using the quantile parameter
3. Points converge to local density peaks (cluster centers)
4. Non-maximum suppression assigns final labels

**Guard Parameters** (internal):
- `num_samples`: 10000 (points sampled for bandwidth estimation)
- `iterations`: 50 (mean shift iterations)
- `max_clusters`: 49 (if exceeded, bandwidth is increased by 1.2x)

---

### 3. **Point2CADSurfaceFitting**
Fits geometric primitives and freeform surfaces to each segment.

**Inputs**:
- `segmented_cloud`: Segmented point cloud from previous node

**Parameters**:
- `use_inr`: Enable SplineINR for freeform surfaces (default: True)
- `device`: Compute device - "cuda" or "cpu" (default: cuda)

**Outputs**:
- `surfaces`: Collection of fitted surfaces (primitives + freeform)
- `fitting_preview`: Point cloud with fitted surface IDs
- `summary`: Surface fitting statistics

#### Fitting Algorithm

The fitting pipeline tries all geometric types for each segment and picks the best fit:

**1. Primitive Fitting** (fast, analytical):
- **Plane**: Least-squares via SVD - fits normal and distance
- **Sphere**: Least-squares center + radius optimization
- **Cylinder**: Eberly's method - optimizes axis + center + radius
- **Cone**: Nonlinear least-squares for apex + axis + angle

**2. SplineINR** (neural, for freeform):
- Trains a small neural network ON-THE-FLY (1000 steps, ~1-2 sec per surface)
- Architecture: SIREN + ResBlock hybrid (encoder-decoder)
- Encoder: 3D -> UV parameterization
- Decoder: UV -> 3D surface reconstruction
- Tries 4 closure configurations (open/closed in U and V)
- No pretrained weights needed - learns per-surface

**3. Selection**:
- Compare errors across all fits
- Prefer primitives if error < 0.008 or primitive error < INR error + 0.001
- This prevents overfitting with neural surfaces when primitives suffice

---

### 4. **Point2CADTopologyExtraction**
Extracts topological elements (edges, corners) via surface intersections.

**Inputs**:
- `surfaces`: Fitted surfaces from previous node

**Parameters**:
- `intersection_tolerance`: Tolerance for surface-surface intersection (default: 0.001)
- `edge_pruning_threshold`: Threshold for pruning spurious edges (default: 0.01)

**Outputs**:
- `topology`: B-rep topology (surfaces, edges, corners, adjacency)
- `summary`: Topology statistics

**How it works**:
1. Compute pairwise surface-surface intersections → edges
2. Compute pairwise edge-edge intersections → corners
3. Build adjacency matrix for B-rep construction

---

### 5. **Point2CADExportBrep**
Exports B-rep topology to STEP file format with optional preview mesh.

**Inputs**:
- `topology`: B-rep topology from previous node

**Parameters**:
- `output_filename`: Output STEP file name (default: "reconstructed_cad.step")
- `output_dir`: Output directory (default: "output")
- `generate_preview`: Generate preview mesh for visualization (default: True)

**Outputs**:
- `step_file_path`: Path to exported STEP file
- `preview_mesh`: Triangle mesh for visualization
- `summary`: Export statistics

---

## Example Workflow: Utah Teapot Reconstruction

The example workflow demonstrates the complete Point2CAD pipeline:

**File**: `workflows/point2cad_teapot_reconstruction.json`

**Pipeline**:
```
[Load Utah Teapot STL]
    ↓
[LoadPoint2CADModel] (parsenet_with_normals)
    ↓
[Point2CADSegmentation] (segment into surfaces)
    ↓
[Point2CADSurfaceFitting] (fit primitives + freeform INR)
    ↓
[Point2CADTopologyExtraction] (extract edges & corners)
    ↓
[Point2CADExportBrep] → utah_teapot_reconstructed.step
    ↓
[Preview Mesh] (visualize result)
```

### How to Use:

1. Load the workflow in ComfyUI:
   ```
   Load → workflows/point2cad_teapot_reconstruction.json
   ```

2. Ensure you have the Utah teapot mesh:
   ```
   ComfyUI/input/3d/utah_teapot.stl
   ```

3. Run the workflow:
   - Models will auto-download on first run
   - Progress is shown in console
   - Output STEP file saved to `output/utah_teapot_reconstructed.step`

4. View results:
   - Preview mesh shown in the workflow
   - STEP file can be opened in FreeCAD, SolidWorks, etc.

---

## Performance Expectations

### Utah Teapot Example:
- **Input**: ~5,000 vertices (STL mesh)
- **Segmentation**: ~5-10 seconds (GPU) / ~20-30 seconds (CPU)
- **Surface Fitting**: ~30-60 seconds (depends on number of freeform surfaces)
- **Topology Extraction**: ~5-10 seconds
- **Total Time**: ~1-2 minutes with GPU

### Memory Requirements:
- **GPU**: 4-8 GB VRAM recommended
- **CPU**: 8-16 GB RAM
- **Disk**: ~1 GB for models + temporary files

---

## Current Status & Future Work

### Implemented:
- [x] Model downloader with auto-caching
- [x] All 5 nodes with complete interface
- [x] **Full ParseNet/HPNet neural network integration** (matching original Point2CAD)
- [x] **PCA-based point normalization** (matching original)
- [x] **Mean Shift clustering** with correct parameters (quantile=0.015, num_samples=10000)
- [x] **Complete primitive fitting** (plane, sphere, cylinder, cone)
- [x] **SplineINR neural surface fitting** (trains on-the-fly per surface)
- [x] Topology extraction framework
- [x] STEP export using pythonocc
- [x] Example workflows (ABC dataset, Utah teapot)

### In Progress:
- [ ] Parallel surface fitting on GPU
- [ ] Advanced topology validation

### Future:
- [ ] Fine-tuning segmentation network on custom datasets
- [ ] Interactive editing of fitted surfaces
- [ ] Multi-resolution INR for large models

---

## Troubleshooting

### Model Download Fails
**Issue**: Network error when downloading models
**Solution**:
- Check internet connection
- Manually download from GitHub:
  ```bash
  mkdir -p ComfyUI/models/cadrecon/point2cad
  cd ComfyUI/models/cadrecon/point2cad
  wget https://github.com/prs-eth/point2cad/raw/main/point2cad/logs/parsenet_with_normals.pth
  ```

### Segmentation Produces Too Many/Few Segments
**Issue**: Wrong number of surface segments
**Solution**:
- Adjust `meanshift_quantile` parameter:
  - **Too few segments**: Decrease quantile (try 0.010-0.015)
  - **Too many segments**: Increase quantile (try 0.02-0.05)
- Original Point2CAD uses 0.015 which works well for most CAD models
- For very complex models with many small surfaces, try lower values (0.005-0.01)
- For simple models with large surfaces, try higher values (0.02-0.03)

### Surface Fitting is Slow
**Issue**: Freeform INR optimization takes too long
**Solution**:
- Reduce `max_inr_iterations` (try 500 instead of 1000)
- Use GPU by enabling `use_gpu` parameter
- Simplify input mesh before reconstruction

### STEP Export is Empty/Invalid
**Issue**: Generated STEP file is a placeholder
**Solution**:
- This is expected in Phase 1
- Full STEP B-rep export coming in Phase 2
- Preview mesh shows reconstructed geometry

---

## Comparison with Existing Nodes

### vs. RANSAC PrimitiveFitting Node:
| Feature | RANSAC Node | Point2CAD |
|---------|-------------|-----------|
| Freeform surfaces | ❌ | ✅ |
| Auto-tuning | ❌ | ✅ |
| Neural segmentation | ❌ | ✅ |
| Complex shapes | ❌ | ✅ |
| Speed | ⚡ Fast | ⚠️ Slower but accurate |

**Recommendation**: Use RANSAC for simple mechanical parts with only primitives. Use Point2CAD for complex shapes like the Utah teapot, automotive parts, or anything with curved surfaces.

---

## References

1. **Point2CAD Paper** (CVPR 2024):
   https://arxiv.org/abs/2312.04962

2. **Project Page**:
   https://www.obukhov.ai/point2cad.html

3. **Official GitHub**:
   https://github.com/prs-eth/point2cad

4. **ABC Dataset** (Training data):
   https://deep-geometry.github.io/abc-dataset/

5. **Related Work**:
   - Point2Primitive (2025): https://arxiv.org/abs/2505.02043
   - SPFN (CVPR 2019): https://arxiv.org/abs/1811.08988
   - HPNet: https://github.com/SimingYan/HPNet

---

## Questions?

- **GitHub Issues**: https://github.com/prs-eth/point2cad/issues
- **ComfyUI-CADabra Issues**: [Your repository]
- **Documentation**: This file!
