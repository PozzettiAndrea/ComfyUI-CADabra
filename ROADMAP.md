# 🎯 Mesh-to-CAD Pipeline Roadmap

## Overview
Building a complete mesh-to-CAD reconstruction pipeline for ComfyUI with 4 key nodes that transform triangle meshes into parametric B-rep CAD models.

---

## 📋 Phase 1: MVP (Minimum Viable Product)

**Goal**: Get a basic working pipeline for simple mechanical parts

### 1.1 Project Structure ✅
- [x] Create `nodes/` folder for modular organization
- [x] Create `nodes/__init__.py` for node registration
- [x] Create `nodes/mesh_recon_nodes.py` for the 4 new nodes
- [x] Update main `__init__.py` to import from nodes module
- [x] Move existing nodes to `nodes/cad_nodes.py`

### 1.2 Dependencies ✅
**Added to requirements.txt**:
- `open3d>=0.17.0` ✅ - Point cloud processing
- `pyransac3d>=0.5.0` ✅ - Primitive fitting (RANSAC-based, optional)
- `scikit-learn>=1.3.0` ✅ - KMeans clustering for segmentation
- `scipy>=1.11.0` ✅ - Spatial operations

**Already available**:
- `trimesh>=3.20.0` ✓
- `numpy>=1.24.0` ✓
- `torch>=2.0.0` ✓
- `gmsh>=4.11.0` ✓ - **Now used for B-rep generation via gmsh.model.occ!**

**No longer required**:
- ~~`pythonocc-core`~~ - Replaced by gmsh.model.occ for Phase 1

### 1.3 QuadRemeshNode 🎯 **START HERE**
**Status**: Not Started
**Priority**: HIGH - Simplest node, good starting point

**Input**: `MESH` (from CAD_Mesh_Gmsh)
- vertices: np.array (Nx3)
- faces: np.array (Mx3)

**Processing**:
- Phase 1: Use trimesh subdivision/remeshing
- Improve triangle quality
- Optionally increase/decrease mesh resolution

**Output**: Refined `MESH` dict

**Future**: Integrate QuadWild for true quad meshing (Phase 2)

### 1.4 PointCloudSegmentationNode
**Status**: Not Started
**Priority**: HIGH

**Input**: `MESH` or `POINT_CLOUD`

**Processing**:
- Convert mesh to point cloud (sample vertices + compute normals)
- Phase 1 approach: KMeans clustering
  - Simple, no training required
  - Segment based on spatial proximity + normals
- Number of clusters: user-defined parameter

**Output**: `SEGMENTED_CLOUD`
```python
{
    "points": np.array (Nx3),
    "normals": np.array (Nx3),
    "labels": np.array (N,),  # segment IDs
    "num_segments": int,
    "confidences": np.array (N,)  # optional
}
```

**Future**: Replace with PointNet++ (Phase 2)

### 1.5 PrimitiveFittingNode
**Status**: Not Started
**Priority**: MEDIUM

**Input**: `SEGMENTED_CLOUD`

**Processing**:
- For each segment, fit primitives using pyRANSAC3D:
  - Plane (most common in mechanical parts)
  - Cylinder (shafts, holes)
  - Sphere (fillets, ball joints)
  - Cone (tapers)
- Auto-detect primitive type or user-specified
- Refine with RANSAC for robustness

**Output**: `PRIMITIVES`
```python
{
    "primitives": [
        {
            "type": "plane",
            "params": {...},  # type-specific
            "points": np.array,  # points belonging to this primitive
            "inliers": np.array,  # boolean mask
            "confidence": float
        },
        ...
    ],
    "assignments": np.array (N,)  # point to primitive mapping
}
```

**Future**: Integrate SPFN for better fitting (Phase 2)

### 1.6 BrepGenerationNode ✅
**Status**: **Implemented using gmsh.model.occ**
**Priority**: MEDIUM-LOW (Most complex)

**Input**: `PRIMITIVES`

**Processing**:
- Convert each primitive to gmsh OCC geometry:
  - Plane → `gmsh.model.occ.addPlaneSurface()` (bounded rectangular face)
  - Cylinder → `gmsh.model.occ.addCylinder()`
  - Sphere → `gmsh.model.occ.addSphere()`
  - Cone → `gmsh.model.occ.addCone()` (future)
- Phase 1: Boolean union using `gmsh.model.occ.fuse()`
- Export to STEP using `gmsh.write()`

**Output**: `BREP_MODEL`
```python
{
    "model_name": str,  # gmsh model name
    "shape_tags": [(dim, tag), ...],  # gmsh dimension tags
    "file_path": str,  # Saved STEP file
    "topology": {
        "volumes": int,
        "faces": int,
        "edges": int,
        "vertices": int
    }
}
```

**Key Advantage**: Uses gmsh (already a dependency), eliminating need for pythonocc-core!

**Challenges**:
- Topology construction (intersections, trimming)
- Surface-surface intersections
- Edge blending/fillets

**Future**: Integrate BrepGen diffusion model (Phase 3)

### 1.7 Testing & Validation
- [ ] Unit tests for each node
- [ ] End-to-end pipeline test with simple shapes
- [ ] Test with existing assets (spiral_wind_turbine.stp, etc.)
- [ ] Validate output with PreviewCADOCC node

---

## 📋 Phase 2: Improvements

**Goal**: Better accuracy and robustness

### 2.1 QuadWild Integration
- [ ] Compile QuadWild C++ library
- [ ] Create Python bindings
- [ ] Replace trimesh remeshing with QuadWild
- [ ] True quad mesh output (better for NURBS fitting)

**Fallback**: Use InstantMesh if compilation fails

### 2.2 PointNet++ Segmentation
- [ ] Clone repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- [ ] Load pretrained weights
- [ ] Create inference wrapper
- [ ] Replace KMeans clustering
- [ ] Fine-tune on ABC dataset (CAD models)

**Benefits**: Semantic understanding (recognize features like holes, bosses, ribs)

### 2.3 SPFN Primitive Fitting
- [ ] Clone repo: https://github.com/lingxiaoli94/SPFN
- [ ] Integrate supervised learning approach
- [ ] Better primitive detection (vs. pure RANSAC)
- [ ] Handle more complex shapes

### 2.4 Advanced B-rep Construction
- [ ] Implement surface-surface intersection
- [ ] Add edge trimming
- [ ] Boolean operations (union, subtract)
- [ ] Fillet/chamfer detection and generation

---

## 📋 Phase 3: Advanced Features

**Goal**: Production-ready, end-to-end learning

### 3.1 BrepGen Diffusion Model
- [ ] Clone repo: https://github.com/samxuxiang/BrepGen
- [ ] Download pretrained weights
- [ ] Integrate inference pipeline
- [ ] Replace manual B-rep construction
- [ ] Support complex topologies

**Paper**: https://arxiv.org/pdf/2401.15563

### 3.2 End-to-End Training
- [ ] Create training dataset (ABC, DeepCAD)
- [ ] Joint training: segmentation + fitting + reconstruction
- [ ] Differentiable RANSAC
- [ ] Geometric loss functions

### 3.3 Advanced Features
- [ ] Support for NURBS surfaces (not just primitives)
- [ ] Freeform surface reconstruction
- [ ] Multi-patch surface fitting
- [ ] Feature curve detection
- [ ] Constraint solving (parallel, perpendicular, etc.)

---

## 📦 Data Flow Pipeline

```
Input Triangle Mesh (from CAD_Mesh_Gmsh or external)
    ↓
QuadRemeshNode (improve mesh quality)
    ↓
PointCloudSegmentationNode (segment into parts)
    ↓
PrimitiveFittingNode (fit geometric primitives)
    ↓
BrepGenerationNode (construct B-rep CAD model)
    ↓
Output STEP file (viewable in PreviewCADOCC)
```

---

## 🔧 Technical Architecture

### Custom Data Types

```python
MESH = {
    "vertices": np.array (Nx3),
    "faces": np.array (MxK),
    "normals": np.array (Nx3),  # optional
    "type": str,  # "surface" or "volume"
}

POINT_CLOUD = {
    "points": np.array (Nx3),
    "normals": np.array (Nx3),
    "colors": np.array (Nx3),  # optional
}

SEGMENTED_CLOUD = {
    "points": np.array (Nx3),
    "normals": np.array (Nx3),
    "labels": np.array (N,),
    "num_segments": int,
}

PRIMITIVES = {
    "primitives": list[dict],
    "assignments": np.array (N,),
}

BREP_MODEL = {
    "shape": TopoDS_Shape,
    "file_path": str,
    "topology": dict,
}
```

### Integration with Existing Nodes

**Connect to existing pipeline**:
- `CAD_Load_Gmsh` → Load reference CAD
- `CAD_Mesh_Gmsh` → Generate input mesh
- `QuadRemeshNode` → **[NEW]** Start of our pipeline
- `PreviewCADOCC` → Visualize output B-rep

---

## 📚 Key Papers & Repositories

### Phase 1
- **pyRANSAC3D**: https://github.com/leomariga/pyRANSAC-3D
- **Open3D**: https://github.com/isl-org/Open3D

### Phase 2
- **QuadWild** (SIGGRAPH 2021): https://dl.acm.org/doi/10.1145/3450626.3459941
  - Repo: https://github.com/nicopietroni/quadwild
- **PointNet++** (NeurIPS 2017): https://arxiv.org/pdf/1706.02413
  - Repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **SPFN** (CVPR 2019): https://arxiv.org/pdf/1811.08988
  - Repo: https://github.com/lingxiaoli94/SPFN

### Phase 3
- **BrepGen** (SIGGRAPH 2024): https://arxiv.org/pdf/2401.15563
  - Repo: https://github.com/samxuxiang/BrepGen
- **multiBaySAC** (PLOS ONE 2015): https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0117341&type=printable

---

## 🎯 Current Status

**Active Phase**: Phase 1 MVP
**Current Sprint**: Setting up project structure + QuadRemeshNode
**Next Up**: PointCloudSegmentationNode

---

## 📝 Notes

- **Start simple**: Focus on making something that works for basic shapes first
- **Iterate quickly**: MVP → Test → Improve → Repeat
- **GPU required**: For PointNet++, SPFN, BrepGen (Phase 2+)
- **Data formats**: STEP for B-rep output, OBJ/PLY for meshes
- **ComfyUI patterns**: Each node handles batching, includes preview capability
- **Error handling**: Validate topology at each stage, fail gracefully

---

## 🤝 Contributing

This is an iterative development process. Each phase builds on the previous:
1. Get Phase 1 working with simple shapes
2. Add Phase 2 improvements for better accuracy
3. Phase 3 for production-ready pipeline

Focus on practical results over perfect solutions!
