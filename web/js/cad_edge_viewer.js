/**
 * CAD Edge Viewer Extension for ComfyUI-CADabra
 * Interactive wireframe/edge inspection widget
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the edge viewer widget for CADEdgeViewer node
 */
function createEdgeViewerWidget(node, nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // Create container for viewer
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.backgroundColor = "#1a1a1a";
        container.style.borderRadius = "4px";
        container.style.overflow = "hidden";

        // Create iframe for interactive viewer
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.flex = "1 1 0";
        iframe.style.minHeight = "0";
        iframe.style.border = "none";
        iframe.style.backgroundColor = "#1a1a1a";
        iframe.src = "/extensions/ComfyUI-CADabra/viewer_edge.html?v=" + Date.now();

        container.appendChild(iframe);

        // Create navigation bar
        const navBar = document.createElement("div");
        navBar.style.display = "flex";
        navBar.style.alignItems = "center";
        navBar.style.gap = "4px";
        navBar.style.padding = "6px 8px";
        navBar.style.backgroundColor = "#2a2a2a";
        navBar.style.borderTop = "1px solid #333";
        navBar.style.flexShrink = "0";

        // Helper to create styled buttons
        const createButton = (label) => {
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.style.padding = "4px 8px";
            btn.style.fontSize = "11px";
            btn.style.backgroundColor = "#333";
            btn.style.border = "1px solid #555";
            btn.style.color = "#ccc";
            btn.style.cursor = "pointer";
            btn.style.borderRadius = "3px";
            btn.onmouseenter = () => btn.style.backgroundColor = "#444";
            btn.onmouseleave = () => btn.style.backgroundColor = "#333";
            return btn;
        };

        // Fit to View button
        const fitButton = createButton("Fit");
        fitButton.title = "Fit to View";

        // Settings button
        const settingsButton = createButton("\u2699");
        settingsButton.title = "Settings";
        settingsButton.style.fontSize = "14px";
        settingsButton.style.padding = "2px 6px";

        // Find Edge button
        const findButton = createButton("Find");
        findButton.title = "Find Edge by ID";

        // View direction buttons
        const viewButtons = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'];
        const viewButtonElements = viewButtons.map(view => {
            const btn = createButton(view);
            btn.title = `View from ${view}`;
            btn.style.padding = "4px 6px";
            btn.addEventListener("click", () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({ type: "SET_VIEW", direction: view }, "*");
                }
            });
            return btn;
        });

        // Separator
        const separator = document.createElement("span");
        separator.style.width = "1px";
        separator.style.height = "16px";
        separator.style.backgroundColor = "#555";
        separator.style.margin = "0 4px";

        // Assemble navigation bar
        viewButtonElements.forEach(btn => navBar.appendChild(btn));
        navBar.appendChild(separator);
        navBar.appendChild(fitButton);
        navBar.appendChild(findButton);
        navBar.appendChild(settingsButton);

        container.appendChild(navBar);

        // Fit button handler
        fitButton.addEventListener("click", () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: "FIT_VIEW" }, "*");
            }
        });

        // Settings button handler
        settingsButton.addEventListener("click", () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: "OPEN_SETTINGS" }, "*");
            }
        });

        // Find Edge button handler
        findButton.addEventListener("click", () => {
            const input = prompt("Enter Edge ID to find:");
            if (input !== null && input.trim() !== "") {
                const edgeId = parseInt(input.trim(), 10);
                if (!isNaN(edgeId) && edgeId >= 0) {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: "SELECT_EDGE_BY_ID",
                            edgeId: edgeId
                        }, "*");
                    }
                } else {
                    alert("Please enter a valid edge ID (non-negative integer)");
                }
            }
        });

        // Create info panel below viewer
        const infoPanel = document.createElement("div");
        infoPanel.style.backgroundColor = "#2a2a2a";
        infoPanel.style.borderTop = "1px solid #444";
        infoPanel.style.padding = "12px";
        infoPanel.style.fontSize = "11px";
        infoPanel.style.fontFamily = "monospace";
        infoPanel.style.color = "#e0e0e0";
        infoPanel.style.lineHeight = "1.4";
        infoPanel.style.flexShrink = "0";
        infoPanel.style.height = "220px";
        infoPanel.style.overflowY = "auto";
        infoPanel.style.overflowX = "hidden";
        infoPanel.innerHTML = "<div style='color: #888;'>Waiting for edge analysis data...</div>";

        container.appendChild(infoPanel);

        // Add DOM widget to node
        const widget = this.addDOMWidget(
            "edge_viewer",
            "CAD_EDGE_VIEWER",
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
            return [width, 600];
        };

        // Store references
        this.edgeIframe = iframe;
        this.edgeInfoPanel = infoPanel;
        this.lastLoadedEdgeFile = null;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const edgeFile = message.edge_file?.[0];
            const visualizationMode = message.visualization_mode?.[0] || "normal";
            const numEdges = message.num_edges?.[0] || 0;
            const numVertices = message.num_vertices?.[0] || 0;
            const numFaces = message.num_faces?.[0] || 0;
            const boundsMin = message.bounds_min?.[0] || [0, 0, 0];
            const boundsMax = message.bounds_max?.[0] || [0, 0, 0];
            const edgeTypeCounts = message.edge_type_counts?.[0] || {};
            const edgeStats = message.edge_stats?.[0] || {};
            const freeEdgeCount = message.free_edge_count?.[0] || 0;

            if (!edgeFile) {
                console.error("[CADabra Edge] Missing edge file data");
                return;
            }

            // Format edge type distribution
            let edgeTypeStr = Object.entries(edgeTypeCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4)
                .map(([type, count]) => `${type}: ${count}`)
                .join(', ');
            if (Object.keys(edgeTypeCounts).length > 4) {
                edgeTypeStr += ', ...';
            }

            // Update info panel with global statistics
            infoPanel.innerHTML = `
                <div style="margin-bottom: 8px;">
                    <strong style="color: #4a9eff;">Edge Analysis</strong>
                </div>
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: 11px;">
                    <span style="color: #888;">File:</span>
                    <span>${edgeFile}</span>

                    <span style="color: #888;">Topology:</span>
                    <span>${numEdges} edges, ${numVertices} vertices, ${numFaces} faces</span>

                    <span style="color: #888;">Edges:</span>
                    <span>${freeEdgeCount} free (boundary), ${numEdges - freeEdgeCount} shared (internal)</span>

                    <span style="color: #888;">Types:</span>
                    <span>${edgeTypeStr || '-'}</span>

                    ${edgeStats.avg ? `
                    <span style="color: #888;">Length:</span>
                    <span>avg ${edgeStats.avg.toFixed(2)}, total ${edgeStats.total?.toFixed(2) || '-'}</span>
                    ` : ''}

                    <span style="color: #888;">Bounds:</span>
                    <span>[${boundsMin.map(v => v.toFixed(1)).join(', ')}] to [${boundsMax.map(v => v.toFixed(1)).join(', ')}]</span>

                    <span style="color: #888;">Mode:</span>
                    <span>${visualizationMode}</span>
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444; color: #888; font-size: 11px;">
                    Click on edges in the viewer to inspect them
                </div>
            `;

            // Skip reload if same file already loaded
            if (this.lastLoadedEdgeFile === edgeFile) {
                console.log("[CADabra Edge] Same file already loaded, skipping reload");
                return;
            }

            // Construct file URL for viewer
            const edgeUrl = `/view?filename=${encodeURIComponent(edgeFile)}&type=output&subfolder=`;

            // Track loaded file
            this.lastLoadedEdgeFile = edgeFile;

            // Wait for iframe to load, then send data
            const sendDataWhenReady = () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_EDGE_ANALYSIS",
                        edgeUrl: edgeUrl,
                        edgeFile: edgeFile,
                        visualizationMode: visualizationMode,
                        nodeId: this.id,
                        timestamp: Date.now()
                    }, "*");
                }
            };

            if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") {
                sendDataWhenReady();
            } else {
                iframe.onload = sendDataWhenReady;
            }
        };

        // Listen for messages from iframe (edge selection updates)
        window.addEventListener("message", (event) => {
            if (event.data.type === "EDGE_SELECTED" && event.data.nodeId === this.id) {
                const edgeData = event.data.edgeData;

                if (!edgeData) {
                    // Deselection - show global stats
                    this.onExecuted(this.lastExecutionData || {});
                    return;
                }

                // Format connected edges
                let connectedStr = '-';
                if (edgeData.connected_edges) {
                    const startEdges = edgeData.connected_edges.start || [];
                    const endEdges = edgeData.connected_edges.end || [];
                    const allConnected = [...new Set([...startEdges, ...endEdges])];
                    if (allConnected.length > 0) {
                        connectedStr = allConnected.slice(0, 8).join(', ');
                        if (allConnected.length > 8) connectedStr += `, ... (${allConnected.length} total)`;
                    }
                }

                // Update info panel with selected edge data
                infoPanel.innerHTML = `
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #ffa500;">Selected Edge #${edgeData.edge_id}</strong>
                        <button onclick="window.postMessage({type: 'DESELECT_EDGE_WIDGET', nodeId: ${this.id}}, '*')"
                                style="float: right; padding: 2px 8px; background: #444; color: #fff; border: none; border-radius: 3px; cursor: pointer; font-size: 10px;">
                            Clear
                        </button>
                    </div>
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 11px;">
                        <span style="color: #888;">Type:</span>
                        <span style="color: #4aff4a;">${edgeData.edge_type || 'Unknown'}</span>

                        <span style="color: #888;">Length:</span>
                        <span>${edgeData.length !== null ? edgeData.length.toFixed(4) : '-'}</span>

                        <span style="color: #888;">Boundary:</span>
                        <span>${edgeData.is_free ? 'Yes (free edge)' : 'No (internal)'}</span>

                        ${edgeData.curvature ? `
                        <span style="color: #888;">Curvature:</span>
                        <span>${edgeData.curvature.min === edgeData.curvature.max ?
                            edgeData.curvature.avg.toFixed(4) :
                            edgeData.curvature.min.toFixed(4) + ' - ' + edgeData.curvature.max.toFixed(4)}</span>
                        ` : ''}

                        ${edgeData.vertices ? `
                        <span style="color: #888;">Start:</span>
                        <span>[${edgeData.vertices.start.map(v => v.toFixed(2)).join(', ')}]</span>

                        <span style="color: #888;">End:</span>
                        <span>[${edgeData.vertices.end.map(v => v.toFixed(2)).join(', ')}]</span>
                        ` : ''}

                        ${edgeData.adjacent_faces && edgeData.adjacent_faces.length > 0 ? `
                        <span style="color: #888;">Adj. Faces:</span>
                        <span>${edgeData.adjacent_faces.join(', ')}</span>
                        ` : ''}

                        <span style="color: #888;">Connected:</span>
                        <span>${connectedStr}</span>

                        ${edgeData.radius !== undefined ? `
                        <span style="color: #888;">Radius:</span>
                        <span>${edgeData.radius.toFixed(4)}</span>
                        ` : ''}

                        ${edgeData.center !== undefined ? `
                        <span style="color: #888;">Center:</span>
                        <span>[${edgeData.center.map(v => v.toFixed(2)).join(', ')}]</span>
                        ` : ''}
                    </div>
                `;
            } else if (event.data.type === "DESELECT_EDGE_WIDGET" && event.data.nodeId === this.id) {
                // Clear selection in viewer
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "DESELECT_EDGE"
                    }, "*");
                }
            }
        });

        // Store last execution data for re-rendering info panel
        const originalOnExecuted = this.onExecuted;
        this.onExecuted = function(message) {
            this.lastExecutionData = message;
            originalOnExecuted.call(this, message);
        };

        return result;
    };
}

app.registerExtension({
    name: "cadabra.cadedge.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADEdgeViewer") {
            createEdgeViewerWidget(this, nodeType);
        }
    }
});
