/**
 * CAD Edge Detail Analyzer Widget for ComfyUI-CADabra
 * 2D visualization of edge curves with detailed parameter display
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the edge detail viewer widget for CADEdgeDetailAnalyzer node
 */
function createEdgeDetailWidget(node, nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // Create container
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.backgroundColor = "#1a1a1a";
        container.style.borderRadius = "4px";
        container.style.overflow = "hidden";

        // Create iframe for the 2D viewer
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.flex = "1 1 0";
        iframe.style.minHeight = "0";
        iframe.style.border = "none";
        iframe.style.backgroundColor = "#1a1a1a";
        iframe.src = new URL(`../viewer_cad_edge_detail.html?v=${Date.now()}`, import.meta.url).href;

        container.appendChild(iframe);

        // Add DOM widget
        const widget = this.addDOMWidget(
            "edge_detail_viewer",
            "CAD_EDGE_DETAIL_VIEWER",
            container,
            {
                getValue() {
                    return "";
                },
                setValue(v) {
                    // Not used for output nodes
                }
            }
        );

        widget.computeSize = function(width) {
            return [width, 500];
        };

        // Store reference
        this.detailIframe = iframe;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            // Check for error
            if (message.error) {
                console.error("[CADabra EdgeDetail] Error:", message.error[0]);
                return;
            }

            const edgeId = message.edge_id?.[0];
            const edgeType = message.edge_type?.[0];
            const length = message.length?.[0];
            const isFree = message.is_free?.[0];
            const projectionPlane = message.projection_plane?.[0];
            const points2d = message.points_2d?.[0] || [];
            const vertices = message.vertices?.[0];
            const tangents = message.tangents?.[0];
            const curvature = message.curvature?.[0];
            const params = message.params?.[0] || {};
            const adjacentFaces = message.adjacent_faces?.[0] || [];
            const connectedEdges = message.connected_edges?.[0] || {};

            // Send data to iframe
            const sendData = () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_EDGE_DETAIL",
                        edge_id: edgeId,
                        edge_type: edgeType,
                        length: length,
                        is_free: isFree,
                        projection_plane: projectionPlane,
                        points_2d: points2d,
                        vertices: vertices,
                        tangents: tangents,
                        curvature: curvature,
                        params: params,
                        adjacent_faces: adjacentFaces,
                        connected_edges: connectedEdges,
                        nodeId: this.id
                    }, "*");
                }
            };

            if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") {
                sendData();
            } else {
                iframe.onload = sendData;
            }
        };

        return result;
    };
}

app.registerExtension({
    name: "cadabra.cadedge.detail",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADEdgeDetailAnalyzer") {
            createEdgeDetailWidget(this, nodeType);
        }
    }
});
