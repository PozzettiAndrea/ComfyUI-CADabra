"""Ultimate CAD Inspection — report validity / watertightness / manifoldness / topology of a B-rep.

The CAD analog of GeometryPack's "Ultimate Mesh Inspection". Works on the B-rep directly
(no meshing), so the answers are exact:

  - Valid B-rep        (BRepCheck_Analyzer)
  - Watertight/closed  (ShapeAnalysis_Shell free-edge count == 0; seam-aware, unlike a naive
                        edge->face ancestor count which miscounts periodic-surface seams)
  - Manifold           (no edge shared by >2 faces / bad-oriented edges)
  - Topology counts    (solids / shells / faces / edges / vertices), free / non-manifold /
                        degenerate edge counts
  - Geometry           (volume if solid, surface area, bounding box / extents / diagonal)
"""
import logging
from comfy_api.latest import io

log = logging.getLogger("CADabra")

CHECK = "✓"   # ✓
CROSS = "✗"   # ✗


def _inspect(shape):
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import (TopAbs_EDGE, TopAbs_SHELL, TopAbs_SOLID,
                                 TopAbs_FACE, TopAbs_VERTEX)
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    def count(typ):
        m = TopTools_IndexedMapOfShape(); topexp.MapShapes(shape, typ, m); return m.Size()

    def edges_in(comp):
        c = 0; ex = TopExp_Explorer(comp, TopAbs_EDGE)
        while ex.More():
            c += 1; ex.Next()
        return c

    d = {}
    d["valid"] = bool(BRepCheck_Analyzer(shape).IsValid())
    d["solids"] = count(TopAbs_SOLID)
    d["shells"] = count(TopAbs_SHELL)
    d["faces"] = count(TopAbs_FACE)
    d["edges"] = count(TopAbs_EDGE)
    d["verts"] = count(TopAbs_VERTEX)

    # Free / bad (non-manifold) edges — seam-aware via ShapeAnalysis_Shell.
    sas = ShapeAnalysis_Shell()
    sx = TopExp_Explorer(shape, TopAbs_SHELL)
    while sx.More():
        sas.LoadShells(sx.Current()); sx.Next()
    sas.CheckOrientedShells(shape, True, False)
    d["free_edges"] = edges_in(sas.FreeEdges()) if sas.HasFreeEdges() else 0
    d["bad_edges"] = edges_in(sas.BadEdges()) if sas.HasBadEdges() else 0

    deg = 0
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        if BRep_Tool.Degenerated(topods.Edge(ex.Current())):
            deg += 1
        ex.Next()
    d["degenerate_edges"] = deg

    g = GProp_GProps()
    try:
        brepgprop.VolumeProperties(shape, g); d["volume"] = g.Mass()
    except Exception:
        d["volume"] = None
    g2 = GProp_GProps(); brepgprop.SurfaceProperties(shape, g2); d["area"] = g2.Mass()

    bb = Bnd_Box(); brepbndlib.Add(shape, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    d["bbox_min"] = (xmin, ymin, zmin); d["bbox_max"] = (xmax, ymax, zmax)
    d["extents"] = (xmax - xmin, ymax - ymin, zmax - zmin)
    d["diagonal"] = (sum((b - a) ** 2 for a, b in zip(d["bbox_min"], d["bbox_max"]))) ** 0.5

    d["watertight"] = d["free_edges"] == 0 and d["shells"] >= 1
    d["manifold"] = d["bad_edges"] == 0
    return d


def _report(d, name=""):
    yn = lambda b: f"{CHECK} Yes" if b else f"{CROSS} No"
    L = []
    if name:
        L.append(f"=== Ultimate CAD Inspection: {name} ===")
    else:
        L.append("=== Ultimate CAD Inspection ===")
    L.append(f"Valid B-rep:         {yn(d['valid'])}   (BRepCheck_Analyzer)")
    L.append(f"Watertight (closed): {yn(d['watertight'])}   ({d['free_edges']} free edges)")
    L.append(f"Manifold:            {yn(d['manifold'])}   ({d['bad_edges']} non-manifold edges)")
    L.append(f"Solid:               {yn(d['solids'] >= 1)}   ({d['solids']} solid{'s' if d['solids'] != 1 else ''})")
    L.append("")
    L.append("Topology:")
    L.append(f"  Solids:   {d['solids']}")
    L.append(f"  Shells:   {d['shells']}")
    L.append(f"  Faces:    {d['faces']}")
    L.append(f"  Edges:    {d['edges']}   (free: {d['free_edges']}, non-manifold: {d['bad_edges']}, degenerate: {d['degenerate_edges']})")
    L.append(f"  Vertices: {d['verts']}")
    L.append("")
    L.append("Geometry:")
    if d["volume"] is not None and d["solids"] >= 1:
        L.append(f"  Volume:        {d['volume']:,.3f}")
    L.append(f"  Surface area:  {d['area']:,.3f}")
    bmin = ", ".join(f"{v:.3f}" for v in d["bbox_min"])
    bmax = ", ".join(f"{v:.3f}" for v in d["bbox_max"])
    ext = " x ".join(f"{v:.3f}" for v in d["extents"])
    L.append(f"  Bounding box:  [{bmin}] to [{bmax}]")
    L.append(f"  Extents:       {ext}")
    L.append(f"  Diagonal:      {d['diagonal']:.3f}")
    return "\n".join(L)


class CADUltimateInspection(io.ComfyNode):
    """Inspect a CAD B-rep: validity, watertightness, manifoldness, topology, free edges, geometry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CADUltimateInspection",
            display_name="Ultimate CAD Inspection",
            category="CADabra/Analysis",
            is_output_node=True,
            inputs=[
                io.Custom("CAD_MODEL").Input("cad_model",
                    tooltip="CAD model (STEP/IGES/BREP) to inspect. Analysed directly on the B-rep (no meshing), so results are exact."),
            ],
            outputs=[
                io.Custom("CAD_MODEL").Output(display_name="cad_model"),
                io.String.Output(display_name="info"),
                io.Boolean.Output(display_name="is_watertight"),
                io.Boolean.Output(display_name="is_valid"),
                io.Int.Output(display_name="free_edges"),
            ],
        )

    @classmethod
    def execute(cls, cad_model):
        from .utils.brep_cache import get_occ_shape

        models = cad_model if isinstance(cad_model, (list, tuple)) else [cad_model]
        reports = []
        first = None
        for i, m in enumerate(models):
            try:
                shape = get_occ_shape(m)
                d = _inspect(shape)
            except Exception as e:
                log.exception("[UltimateInspection] failed")
                reports.append(f"=== CAD {i} === FAILED: {type(e).__name__}: {e}")
                continue
            name = (m.get("file_path") or "").split("/")[-1] if isinstance(m, dict) else ""
            reports.append(_report(d, name if len(models) > 1 else ""))
            if first is None:
                first = d

        info = "\n\n".join(reports) if reports else "No CAD model to inspect."
        log.info("[UltimateInspection]\n%s", info)
        if first is None:
            return io.NodeOutput(cad_model, info, False, False, 0, ui={"text": (info,)})
        return io.NodeOutput(cad_model, info, bool(first["watertight"]),
                             bool(first["valid"]), int(first["free_edges"]),
                             ui={"text": (info,)})


NODE_CLASS_MAPPINGS = {"CADUltimateInspection": CADUltimateInspection}
NODE_DISPLAY_NAME_MAPPINGS = {"CADUltimateInspection": "Ultimate CAD Inspection"}
