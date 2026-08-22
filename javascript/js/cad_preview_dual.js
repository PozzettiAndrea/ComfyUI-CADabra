/**
 * ComfyUI-CADabra - Dual CAD Preview Widget
 *
 * Side-by-side / overlay viewer for two tessellated CAD shapes, modeled on GeometryPack's
 * "Preview Mesh Dual" (shares the identical vtk-gltf.mjs + js/utils viewer stack). In OVERLAY
 * mode the two shapes are coloured distinctly: the node bakes a per-vertex `mesh_id` field
 * (0 = cad_1, 1 = cad_2) and this widget auto-selects it so the shapes render in two colours.
 */

import { app } from "../../../scripts/app.js";
import { getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel } from "./utils/uiComponents.js";
import { buildDualMeshInfoHTML, formatTypeCounts } from "./utils/formatting.js";
import { createViewerManager, createErrorHandler, buildViewUrl, createLoadDualMeshMessage } from "./utils/postMessage.js";

// Colour the two shapes by mesh_id in overlay (0 vs 1 -> two ends of the colormap).
const OVERLAY_FIELD = "mesh_id";
const OVERLAY_COLORMAP = "Cool to Warm";   // diverging: cad_1 -> blue, cad_2 -> red

app.registerExtension({
    name: "cadabra.cadpreview.dual",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PreviewCADDual") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const viewerState = { show_edges: false, camera_state: "", selected_field: "",
                                  selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

            const container = createContainer();
            const iframe = createIframe(getViewerUrl("viewer_cad_dual"), { minHeight: "550px" });
            const infoPanel = createInfoPanel("CAD info will appear here after execution");
            container.appendChild(iframe);
            container.appendChild(infoPanel);

            const widget = this.addDOMWidget("preview_cad_dual", "CAD_PREVIEW_DUAL", container, {
                getValue() { return JSON.stringify(viewerState); },
                setValue(v) { try { Object.assign(viewerState, JSON.parse(v)); } catch (e) {} },
            });
            widget.computeSize = () => [768, 680];

            this.cadViewerIframeDual = iframe;
            this.cadInfoPanelDual = infoPanel;
            this.setSize([768, 680]);

            const node = this;
            window.addEventListener('message', (event) => {
                // Without this check, every open viewer instance's listener
                // fires for every iframe's messages, not just its own.
                if (event.source !== iframe.contentWindow) return;
                if (event.data.type === 'WIDGET_UPDATE') {
                    const { widget: name, value } = event.data;
                    if (name in viewerState) viewerState[name] = value;
                    const w = node.widgets?.find(w => w.name === name);
                    if (w) w.value = value;
                }
            });

            const viewerManager = createViewerManager(iframe, "[CADabra Dual]");
            window.addEventListener('message', createErrorHandler(infoPanel, "[CADabra Dual]", iframe));

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (!message?.layout) return;

                const layout = message.layout[0];
                let postMessageData;

                if (layout === 'side_by_side') {
                    if (!message?.mesh_1_file || !message?.mesh_2_file) return;
                    infoPanel.innerHTML = buildDualMeshInfoHTML({
                        mode: "fields", layout: layout,
                        mesh1: { vertices: message.vertex_count_1?.[0] || 'N/A', faces: message.face_count_1?.[0] || 'N/A',
                                 extents: message.extents_1?.[0] || [], isWatertight: message.is_watertight_1?.[0],
                                 faceTypes: formatTypeCounts(message.face_type_counts_1?.[0]),
                                 edgeTypes: formatTypeCounts(message.edge_type_counts_1?.[0]) },
                        mesh2: { vertices: message.vertex_count_2?.[0] || 'N/A', faces: message.face_count_2?.[0] || 'N/A',
                                 extents: message.extents_2?.[0] || [], isWatertight: message.is_watertight_2?.[0],
                                 faceTypes: formatTypeCounts(message.face_type_counts_2?.[0]),
                                 edgeTypes: formatTypeCounts(message.edge_type_counts_2?.[0]) },
                        commonFields: message.common_fields?.[0] || [],
                    });
                    postMessageData = createLoadDualMeshMessage({
                        layout: layout,
                        mesh1Filepath: buildViewUrl(message.mesh_1_file[0]),
                        mesh2Filepath: buildViewUrl(message.mesh_2_file[0]),
                        opacity1: message.opacity_1?.[0] ?? 1.0,
                        opacity2: message.opacity_2?.[0] ?? 1.0,
                        showEdges: viewerState.show_edges,
                        cameraState: viewerState.camera_state,
                        selectedField: viewerState.selected_field,
                        selectedChannel: viewerState.selected_channel,
                        selectedColormap: viewerState.selected_colormap,
                    });
                } else {
                    // Overlay: colour the two shapes distinctly by mesh_id unless the user picked a field.
                    if (!message?.mesh_file) return;
                    infoPanel.innerHTML = buildDualMeshInfoHTML({
                        mode: "fields", layout: "overlay",
                        mesh1: { vertices: message.vertex_count_1?.[0] || 'N/A', faces: message.face_count_1?.[0] || 'N/A',
                                 faceTypes: formatTypeCounts(message.face_type_counts_1?.[0]),
                                 edgeTypes: formatTypeCounts(message.edge_type_counts_1?.[0]) },
                        mesh2: { vertices: message.vertex_count_2?.[0] || 'N/A', faces: message.face_count_2?.[0] || 'N/A',
                                 faceTypes: formatTypeCounts(message.face_type_counts_2?.[0]),
                                 edgeTypes: formatTypeCounts(message.edge_type_counts_2?.[0]) },
                        commonFields: message.common_fields?.[0] || [],
                    });
                    const overlayField = viewerState.selected_field || OVERLAY_FIELD;
                    const overlayColormap = viewerState.selected_field ? viewerState.selected_colormap : OVERLAY_COLORMAP;
                    postMessageData = createLoadDualMeshMessage({
                        layout: layout,
                        meshFilepath: buildViewUrl(message.mesh_file[0]),
                        opacity1: message.opacity_1?.[0] ?? 1.0,
                        opacity2: message.opacity_2?.[0] ?? 1.0,
                        showEdges: viewerState.show_edges,
                        cameraState: viewerState.camera_state,
                        selectedField: overlayField,
                        selectedChannel: viewerState.selected_channel,
                        selectedColormap: overlayColormap,
                    });
                }

                viewerManager.switchViewer("fields", getViewerUrl("viewer_cad_dual"), postMessageData);
            };

            return r;
        };
    }
});
