
# ComfyUI-CADabra

CAD file processing and ML-based surface reconstruction nodes for ComfyUI.

![CAD Viewer](docs/viewer.png)
![CAD Viewer](docs/meshing.png)



https://github.com/user-attachments/assets/fe90654a-9751-4643-b447-b4dfa2e06e99



https://github.com/user-attachments/assets/8e36f28d-4752-41e6-acfc-75d115329aa8



## Nodes

**CAD_Load** - Load STEP, IGES, and BREP CAD files using OpenCASCADE (with BREP caching for fast reloads)

**CAD_Mesh_Gmsh** - Generate 2D/3D meshes from CAD models with configurable element size and algorithms

**CAD_Mesh_Gmsh_Advanced** - Advanced mesh generation with:
- Size field controls (min/max element size)
- Curvature-based adaptive sizing
- Extended algorithms (MeshAdapt, BAMG, DelQuad, MMG3D, HXT)
- Higher-order elements (1st through 5th order)
- Subdivision and refinement options
- JSON input for custom gmsh options

**Mesh_Optimize_Gmsh** - Post-process meshes with:
- Mesh optimization (Netgen, high-order)
- Smoothing iterations
- Triangle-to-quad recombination
- Multiple recombination algorithms (Simple, Blossom, etc.)
- JSON input for custom gmsh options

**CAD_Convert_Format** - Convert between CAD formats (STEP ↔ IGES ↔ BREP ↔ STL) using OpenCASCADE

**ML_SurfaceRecon** - Surface reconstruction using ML models (stub for hustvl/surface-recon, point2surf, patchnetsurface, point2shape)

**ML_FeatureDetection** - CAD feature detection using ML models (stub for autodesk/feature-detection-cad, hustvl/cad-features)

**PreviewCADOCC** - Interactive Three.js-based 3D viewer for CAD models and meshes

## Standalone Model Packages

The following ML-based reconstruction models are also available as standalone ComfyUI extensions:

| Package | Description |
|---------|-------------|
| [ComfyUI-Point2CAD](https://github.com/PozzettiAndrea/ComfyUI-Point2CAD) | Point cloud to CAD reconstruction (CVPR 2024) |
| [ComfyUI-SECADNET](https://github.com/PozzettiAndrea/ComfyUI-SECADNET) | Sketch-extrude CAD reconstruction from voxels |
| [ComfyUI-Cadrille](https://github.com/PozzettiAndrea/ComfyUI-Cadrille) | Multi-modal CAD from point clouds, images, or text |
| [ComfyUI-BGPSeg](https://github.com/PozzettiAndrea/ComfyUI-BGPSeg) | Boundary-guided primitive segmentation (IEEE TIP 2025) |
| [ComfyUI-NeurCADRecon](https://github.com/PozzettiAndrea/ComfyUI-NeurCADRecon) | Neural implicit CAD reconstruction (SIGGRAPH 2024) |

## Notes

- ML nodes are currently stubs - integrate HuggingFace models as needed
- For publishing to Comfy Registry, see [PUBLISHING.md](PUBLISHING.md)

## Future Work

- Point2CAD: Convert INR freeform surfaces to NURBS instead of B-spline for better CAD software compatibility

## Resources

- [Awesome-3D-Generation](https://github.com/BunnySoCrazy/Awesome-3D-Generation) - Curated list of 3D generative AI papers with visual previews
- [Awesome-Neural-CAD](https://github.com/BunnySoCrazy/Awesome-Neural-CAD) - Curated list of neural CAD papers (generation, reconstruction, analysis)

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-CADabra/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.
