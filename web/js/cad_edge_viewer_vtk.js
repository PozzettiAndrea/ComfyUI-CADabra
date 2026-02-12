/**
 * CAD Edge Viewer (VTK) Extension for ComfyUI-CADabra
 * Uses VTK.js with Trackball interactor for free 3D rotation
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the VTK edge viewer widget for CADEdgeViewerVTK node
 */
function createVTKEdgeViewerWidget(node, nodeType) {
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

        // Create iframe for VTK.js viewer
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.flex = "1 1 0";
        iframe.style.minHeight = "0";
        iframe.style.border = "none";
        iframe.style.backgroundColor = "#1a1a1a";
        iframe.src = "/extensions/ComfyUI-CADabra/viewer_edge_vtk.html?v=" + Date.now();

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
        fitButton.title = "Fit to View (or press R)";

        // VTK label
        const vtkLabel = document.createElement("span");
        vtkLabel.textContent = "VTK";
        vtkLabel.style.color = "#4CAF50";
        vtkLabel.style.fontSize = "10px";
        vtkLabel.style.fontWeight = "bold";
        vtkLabel.style.padding = "2px 6px";
        vtkLabel.style.backgroundColor = "rgba(76, 175, 80, 0.2)";
        vtkLabel.style.borderRadius = "3px";
        vtkLabel.title = "VTK.js with Trackball controls - Free 3D rotation!";

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
        navBar.appendChild(vtkLabel);
        navBar.appendChild(separator.cloneNode());
        viewButtonElements.forEach(btn => navBar.appendChild(btn));
        navBar.appendChild(separator);
        navBar.appendChild(fitButton);

        container.appendChild(navBar);

        // Fit button handler
        fitButton.addEventListener("click", () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: "FIT_VIEW" }, "*");
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
        infoPanel.style.height = "180px";
        infoPanel.style.overflowY = "auto";
        infoPanel.style.overflowX = "hidden";
        infoPanel.innerHTML = `
            <div style='color: #888;'>
                <div>Waiting for edge data...</div>
                <div style='margin-top: 8px; color: #4CAF50;'>
                    <strong>VTK Trackball Controls:</strong>
                </div>
                <div style='margin-top: 4px;'>
                    Left Mouse: <strong>Free 3D Rotation</strong><br>
                    Middle/Shift+Left: Pan<br>
                    Right/Scroll: Zoom<br>
                    R Key: Reset Camera
                </div>
            </div>
        `;

        container.appendChild(infoPanel);

        // Add DOM widget to node
        const widget = this.addDOMWidget(
            "vtk_edge_viewer",
            "CAD_EDGE_VIEWER_VTK",
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
            return [width, 550];
        };

        // Store references
        this.vtkIframe = iframe;
        this.vtkInfoPanel = infoPanel;
        this.lastLoadedVtpFile = null;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const vtpFile = message.vtp_file?.[0];
            const splineDataFile = message.spline_data_file?.[0];  // May be null
            const visualizationMode = message.visualization_mode?.[0] || "normal";
            const numEdges = message.num_edges?.[0] || 0;
            const numFaces = message.num_faces?.[0] || 0;
            const boundsMin = message.bounds_min?.[0] || [0, 0, 0];
            const boundsMax = message.bounds_max?.[0] || [0, 0, 0];
            const edgeTypeCounts = message.edge_type_counts?.[0] || {};
            const freeEdgeCount = message.free_edge_count?.[0] || 0;

            if (!vtpFile) {
                console.error("[CADabra VTK Edge] Missing VTP file data");
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
                    <strong style="color: #4CAF50;">Edge Viewer (VTK)</strong>
                    <span style="float: right; color: #888; font-size: 10px;">${visualizationMode} mode</span>
                </div>
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 11px;">
                    <span style="color: #888;">File:</span>
                    <span>${vtpFile}</span>

                    <span style="color: #888;">Edges:</span>
                    <span>${numEdges} total (${freeEdgeCount} free, ${numEdges - freeEdgeCount} shared)</span>

                    <span style="color: #888;">Types:</span>
                    <span>${edgeTypeStr || '-'}</span>

                    <span style="color: #888;">Bounds:</span>
                    <span>[${boundsMin.map(v => v.toFixed(1)).join(', ')}] to [${boundsMax.map(v => v.toFixed(1)).join(', ')}]</span>
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444; color: #888; font-size: 10px;">
                    Trackball: Free 3D rotation | Click edges to inspect | Press R to reset
                </div>
            `;

            // Skip reload if same file already loaded
            if (this.lastLoadedVtpFile === vtpFile) {
                console.log("[CADabra VTK Edge] Same file already loaded, skipping reload");
                return;
            }

            // Construct file URLs for viewer
            const vtpUrl = `/view?filename=${encodeURIComponent(vtpFile)}&type=output&subfolder=`;
            const splineDataUrl = splineDataFile
                ? `/view?filename=${encodeURIComponent(splineDataFile)}&type=output&subfolder=`
                : null;

            // Track loaded file
            this.lastLoadedVtpFile = vtpFile;

            // Wait for iframe to load, then send data
            const sendDataWhenReady = () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_EDGE_VTK",
                        vtpUrl: vtpUrl,
                        splineDataUrl: splineDataUrl,  // May be null if no spline edges
                        vtpFile: vtpFile,
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
                const { edgeId, edgeType, length, isFree, splineData } = event.data;

                // Update selected_edge_id input widget
                const edgeIdWidget = this.widgets?.find(w => w.name === "selected_edge_id");
                if (edgeIdWidget) {
                    edgeIdWidget.value = edgeId;
                    // Mark graph as needing update
                    if (app.graph) {
                        app.graph.setDirtyCanvas(true);
                    }
                }

                // Build spline parameters section if available
                let splineHTML = '';
                if (splineData && (edgeType === 'BSpline' || edgeType === 'Bezier')) {
                    const rationalStr = splineData.is_rational ? 'Yes (NURBS)' : 'No';
                    const knotsStr = splineData.knots_flat.map(k => k.toFixed(2)).join(', ');
                    const ctrlPtsCount = splineData.control_points.length;

                    // Format planarity info with deviation value and percentage
                    let planarStr = '-';
                    let planarColor = '#888';
                    if (splineData.planarity && splineData.planarity.max_deviation !== null) {
                        const dev = splineData.planarity.max_deviation;
                        const pct = splineData.planarity.deviation_percent;
                        const plane = splineData.planarity.plane;

                        // Format percentage string
                        const pctStr = pct !== null ? (pct < 0.01 ? pct.toExponential(1) : pct.toFixed(2)) + '%' : '';

                        if (plane === 'point') {
                            planarStr = 'point (degenerate)';
                            planarColor = '#888';
                        } else if (plane === 'line') {
                            planarStr = 'line (collinear)';
                            planarColor = '#44ff88';
                        } else if (dev < 1e-6) {
                            planarStr = `${plane} (exact)`;
                            planarColor = '#44ff88';
                        } else if (dev < 0.001) {
                            planarStr = `${plane} (±${dev.toExponential(1)}, ${pctStr})`;
                            planarColor = '#44ff88';
                        } else if (dev < 0.1) {
                            planarStr = `${plane} (±${dev.toFixed(3)}, ${pctStr})`;
                            planarColor = '#ffcc44';
                        } else {
                            planarStr = `3D (dev: ${dev.toFixed(2)}, ${pctStr})`;
                            planarColor = '#ff8844';
                        }
                    }

                    splineHTML = `
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444;">
                            <strong style="color: #ff44ff;">Spline Parameters</strong>
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 10px; margin-top: 4px;">
                                <span style="color: #888;">Degree:</span>
                                <span>${splineData.degree}</span>
                                <span style="color: #888;">Rational:</span>
                                <span>${rationalStr}</span>
                                <span style="color: #888;">Poles:</span>
                                <span>${ctrlPtsCount}</span>
                                <span style="color: #888;">Planar:</span>
                                <span style="color: ${planarColor};">${planarStr}</span>
                            </div>
                            <div style="margin-top: 6px; font-size: 10px;">
                                <span style="color: #888;">Knots:</span>
                                <span style="color: #44ff88; font-family: monospace; word-break: break-all;">[${knotsStr}]</span>
                            </div>
                            <div style="margin-top: 4px; color: #888; font-size: 9px;">
                                See viewer for full control point coordinates
                            </div>
                        </div>
                    `;
                }

                // Update info panel with selected edge data
                infoPanel.innerHTML = `
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #ffa500;">Selected Edge #${edgeId}</strong>
                        <span style="float: right; color: #888; font-size: 10px;">VTK Trackball</span>
                    </div>
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 11px;">
                        <span style="color: #888;">Type:</span>
                        <span style="color: #4aff4a;">${edgeType}</span>

                        <span style="color: #888;">Length:</span>
                        <span>${length !== undefined ? length.toFixed(4) : '-'}</span>

                        <span style="color: #888;">Status:</span>
                        <span>${isFree ? 'Free (boundary edge)' : 'Shared (internal edge)'}</span>
                    </div>
                    ${splineHTML}
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444; color: #888; font-size: 10px;">
                        Click elsewhere to select another edge
                    </div>
                `;
            }
        });

        return result;
    };
}

app.registerExtension({
    name: "cadabra.cadedge.vtk.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADEdgeViewerVTK") {
            createVTKEdgeViewerWidget(this, nodeType);
        }
    }
});
