# ComfyUI-CADabra

CAD file processing and ML-based surface reconstruction nodes for ComfyUI.

## Nodes

**CAD_Load_Gmsh** - Load STEP, IGES, and BREP CAD files using Gmsh

**CAD_Mesh_Gmsh** - Generate 2D/3D meshes from CAD models with configurable element size and algorithms

**ML_SurfaceRecon** - Surface reconstruction using ML models (stub for hustvl/surface-recon, point2surf, patchnetsurface, point2shape)

**ML_FeatureDetection** - CAD feature detection using ML models (stub for autodesk/feature-detection-cad, hustvl/cad-features)

**CAD_Viewer** - Interactive Three.js-based 3D viewer for CAD models and meshes

## Notes

ML nodes are currently stubs - integrate HuggingFace models as needed.
