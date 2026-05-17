> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.

# ComfyUI-CADabra

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `CADabra` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-CADabra.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-CADabra.git
   cd ComfyUI-CADabra
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-CADabra/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-CADabra/issues). Very grateful for your help! 🙏

---


CAD file processing and ML-based surface reconstruction nodes for ComfyUI.

<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-CADabra/">
<img src="https://pozzettiandrea.github.io/ComfyUI-CADabra/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-CADabra/">View Live Test Gallery →</a></b>
</div>



## Resources

- [Awesome-3D-Generation](https://github.com/BunnySoCrazy/Awesome-3D-Generation) - Curated list of 3D generative AI papers with visual previews
- [Awesome-Neural-CAD](https://github.com/BunnySoCrazy/Awesome-Neural-CAD) - Curated list of neural CAD papers (generation, reconstruction, analysis)

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-CADabra/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.
