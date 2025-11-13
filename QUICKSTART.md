# 🚀 Quick Start - Mesh-to-CAD Pipeline

Get up and running with the mesh reconstruction pipeline in 5 minutes!

---

## 📦 Installation

### 1. Install Python Dependencies

**Simple! All dependencies install via pip/uv (no conda required):**

```bash
cd /workspace/cad_node/ComfyUI/custom_nodes/comfyui-cadabra

# Install all Phase 1 dependencies
pip install open3d>=0.17.0 scikit-learn>=1.3.0 scipy>=1.11.0 pyransac3d>=0.5.0
```

**Or with uv (faster):**
```bash
uv pip install open3d scikit-learn scipy pyransac3d
```

**That's it!** B-rep generation now uses gmsh.model.occ (already installed with gmsh), so no pythonocc-core or conda needed!

### 2. Restart ComfyUI

After installing dependencies, restart ComfyUI to load the new nodes.

---

## 🎮 Using the Nodes

### Available Nodes (in ComfyUI Node Menu)

Under **CADabra/Mesh Reconstruction**:
1. **Quad Remesh (Mesh Reconstruction)** - Improves mesh quality
2. **Point Cloud Segmentation** - Segments into regions
3. **Primitive Fitting (RANSAC)** - Detects geometric shapes
4. **B-rep Generation (CAD)** - Creates CAD model

### Basic Workflow

```
┌─────────────────────┐
│  CAD_Load_Gmsh      │ Load existing CAD or mesh file
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  CAD_Mesh_Gmsh      │ Generate triangle mesh
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Quad Remesh        │ Improve mesh quality (optional)
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Point Cloud        │ Segment into parts
│  Segmentation       │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  Primitive Fitting  │ Fit geometric primitives
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  B-rep Generation   │ Create CAD model
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│  PreviewCADOCC      │ View result
└─────────────────────┘
```

---

## 🔧 Node Parameters

### QuadRemesh
- **mesh**: Input mesh from CAD_Mesh_Gmsh
- **target_edge_length**: Desired edge size (default: 0.1)
- **iterations**: Subdivision iterations (default: 3)
- **smooth**: Apply smoothing (default: true)

### PointCloudSegmentation
- **input_data**: MESH or POINT_CLOUD
- **num_segments**: Number of regions (default: 5)
  - Start with 3-5 for simple parts
  - Increase for complex parts with many features
- **use_normals**: Include surface orientation in clustering (default: true)
- **sample_points**: Points to sample from mesh (default: 10000)

### PrimitiveFitting
- **segmented_cloud**: From PointCloudSegmentation
- **primitive_type**: auto, plane, cylinder, sphere, cone
  - "auto": Tries all types, picks best fit
  - Specific type: Forces that primitive
- **ransac_threshold**: Inlier distance threshold (default: 0.01)
  - Smaller = tighter fit, fewer inliers
  - Larger = looser fit, more inliers
- **min_points**: Minimum points to attempt fitting (default: 100)

### BrepGeneration
- **primitives**: From PrimitiveFitting
- **output_path**: Where to save STEP file (default: "output/reconstructed.step")
- **combine_shapes**: Merge into single shape (default: true)
- **tolerance**: Geometric tolerance (default: 0.001)

---

## 🧪 Test Cases

### Test 1: Simple Box

**Goal**: Verify basic pipeline works

1. Create a simple box mesh (or load from assets)
2. Set `num_segments = 6` (one per face)
3. Set `primitive_type = "plane"`
4. Check output: should have 6 planar faces

**Expected Result**: 6 rectangular faces forming a box

### Test 2: Cylinder

**Goal**: Test cylindrical primitive fitting

1. Load or create cylindrical mesh
2. Set `num_segments = 3` (top, bottom, side)
3. Set `primitive_type = "auto"`
4. Check output: should detect cylinder + 2 planes

**Expected Result**: Cylinder with capped ends

### Test 3: Mechanical Part

**Goal**: Full pipeline with realistic part

1. Load: `assets/spiral_wind_turbine.stp`
2. Generate mesh with element_size = 0.1
3. Remesh with target_edge_length = 0.05
4. Segment with num_segments = 10-15
5. Fit primitives with auto detection
6. Generate B-rep

**Expected Result**: Reconstructed CAD model approximating original

---

## 🐛 Troubleshooting

### Missing Dependencies

**"open3d not installed"**
```bash
pip install open3d>=0.17.0
```

**"pyRANSAC3D not installed"** (optional but recommended)
```bash
pip install pyransac3d>=0.5.0
```

**"scikit-learn not installed"**
```bash
pip install scikit-learn>=1.3.0
```

### Segmentation Issues

**Segmentation produces poor results**
- Try increasing `num_segments`
- Enable `use_normals` for better separation
- Check mesh quality - remesh first if needed

### Primitive Fitting Issues

**Primitive fitting fails**
- Lower `ransac_threshold` for tighter fits
- Increase `ransac_threshold` for noisy data
- Ensure segments have enough points (check `min_points`)
- Try specific primitive type instead of "auto"

### B-rep Generation Issues

**B-rep generation fails**
- Verify primitives were successfully fitted (check console output)
- Try `combine_shapes = false` for complex shapes
- Check output path is writable (create output/ directory if needed)
- Check gmsh is initialized (should be automatic)

**"gmsh not initialized" error**
- This should never happen as gmsh is initialized in __init__.py
- If it does, report as a bug

### Mesh Issues

**Mesh has no normals**
- Run through CAD_Mesh_Gmsh first (auto-computes normals)
- Or use remesh node to compute normals

---

## 💡 Tips & Best Practices

### For Best Results

1. **Start Simple**: Test with basic shapes (box, cylinder) first
2. **Mesh Quality Matters**: Use remesh node if input mesh is poor quality
3. **Segment Count**:
   - Too few: Multiple features merged together
   - Too many: Single features split apart
   - Rule of thumb: Count visible faces/features and add 20%
4. **RANSAC Threshold**:
   - Start with default (0.01)
   - Adjust based on mesh units (if model in mm, use smaller value)
5. **Check Intermediate Results**: Look at segmentation before fitting
6. **Iterative Refinement**: Adjust parameters and re-run
7. **No conda needed**: B-rep generation uses gmsh (already installed!)

### Performance Optimization

- **sample_points**: More points = better accuracy but slower
  - 5,000: Fast, acceptable for simple shapes
  - 10,000: Good balance (default)
  - 50,000+: High accuracy for complex parts
- **Remesh**: Optional but improves downstream quality
- **GPU**: Not required for Phase 1 (CPU only)

### Common Workflows

**Workflow 1: Scan-to-CAD**
```
External Mesh (OBJ/STL) → Remesh → Segment → Fit → B-rep
```

**Workflow 2: CAD-to-CAD (Simplification)**
```
CAD Load → Mesh → Fit Primitives → B-rep (simplified version)
```

**Workflow 3: Repair/Reconstruction**
```
Damaged CAD → Mesh → Remesh → Segment → Fit → B-rep (repaired)
```

---

## 📊 Example Parameters by Use Case

### High-Detail Reconstruction
```
QuadRemesh:
  - target_edge_length: 0.02
  - iterations: 5
PointCloudSegmentation:
  - num_segments: 20
  - sample_points: 50000
PrimitiveFitting:
  - ransac_threshold: 0.005
```

### Fast Approximation
```
QuadRemesh: (skip)
PointCloudSegmentation:
  - num_segments: 5
  - sample_points: 5000
PrimitiveFitting:
  - ransac_threshold: 0.02
```

### Noisy Data
```
QuadRemesh:
  - smooth: true
  - iterations: 10
PrimitiveFitting:
  - ransac_threshold: 0.05 (higher tolerance)
```

---

## 📖 Next Steps

1. **Test the basic workflow** with a simple shape
2. **Experiment with parameters** on your specific data
3. **Check ROADMAP.md** for future enhancements
4. **Check IMPLEMENTATION_STATUS.md** for known limitations
5. **Report issues** on GitHub

---

## 🤝 Need Help?

- **Documentation**: Check ROADMAP.md and IMPLEMENTATION_STATUS.md
- **Examples**: See `assets/` folder for test CAD files
- **Issues**: Report bugs or request features on GitHub
- **Questions**: Check prompt.txt for original specification

Happy meshing! 🎨
