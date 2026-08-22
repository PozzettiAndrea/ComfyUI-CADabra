/**
 * CAD Analysis Viewer Extension for ComfyUI-CADabra
 * Handles both CADFaceAnalysis (legacy) and CADAnalysisViewer nodes
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the viewer widget for a node
 */
function createViewerWidget(node, nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // Viewer state persisted via DOM widget serialization
        const viewerState = { camera_state: "", selected_face_id: null };

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
        iframe.src = new URL(`../viewer_cad_analysis.html?v=${Date.now()}`, import.meta.url).href;

        container.appendChild(iframe);

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

        // Create batch navigation bar
        const batchNavBar = document.createElement("div");
        batchNavBar.style.backgroundColor = "#1a1a1a";
        batchNavBar.style.borderTop = "1px solid #444";
        batchNavBar.style.padding = "8px 12px";
        batchNavBar.style.display = "flex";
        batchNavBar.style.alignItems = "center";
        batchNavBar.style.justifyContent = "center";
        batchNavBar.style.gap = "12px";
        batchNavBar.style.fontSize = "12px";
        batchNavBar.style.color = "#ccc";
        batchNavBar.style.flexShrink = "0";

        // Previous button
        const prevButton = createButton("<");
        prevButton.title = "Previous";

        // Index display
        const indexLabel = document.createElement("span");
        indexLabel.textContent = "1 / 1";
        indexLabel.style.minWidth = "50px";
        indexLabel.style.textAlign = "center";
        indexLabel.style.fontFamily = "monospace";
        indexLabel.style.fontWeight = "bold";
        indexLabel.style.fontSize = "11px";

        // Next button
        const nextButton = createButton(">");
        nextButton.title = "Next";

        // Assemble batch navigation bar
        batchNavBar.appendChild(prevButton);
        batchNavBar.appendChild(indexLabel);
        batchNavBar.appendChild(nextButton);

        container.appendChild(batchNavBar);

        // Track batch state
        let currentBatchSize = 1;
        let currentIndex = 0;

        // Find the index widget (created by ComfyUI from INPUT_TYPES)
        const findIndexWidget = () => this.widgets?.find(w => w.name === 'index');

        // Update button states
        const updateNavigationButtons = () => {
            prevButton.disabled = currentIndex === 0;
            nextButton.disabled = currentIndex >= currentBatchSize - 1;

            if (prevButton.disabled) {
                prevButton.style.opacity = "0.4";
                prevButton.style.cursor = "not-allowed";
            } else {
                prevButton.style.opacity = "1";
                prevButton.style.cursor = "pointer";
            }

            if (nextButton.disabled) {
                nextButton.style.opacity = "0.4";
                nextButton.style.cursor = "not-allowed";
            } else {
                nextButton.style.opacity = "1";
                nextButton.style.cursor = "pointer";
            }
        };

        // Button click handlers
        prevButton.addEventListener("click", () => {
            const indexWidget = findIndexWidget();
            if (indexWidget && currentIndex > 0) {
                indexWidget.value = currentIndex - 1;
                app.queuePrompt();
            }
        });

        nextButton.addEventListener("click", () => {
            const indexWidget = findIndexWidget();
            if (indexWidget && currentIndex < currentBatchSize - 1) {
                indexWidget.value = currentIndex + 1;
                app.queuePrompt();
            }
        });

        // Create navigation bar for view controls
        const navBar = document.createElement("div");
        navBar.style.display = "flex";
        navBar.style.alignItems = "center";
        navBar.style.gap = "4px";
        navBar.style.padding = "6px 8px";
        navBar.style.backgroundColor = "#2a2a2a";
        navBar.style.borderTop = "1px solid #333";
        navBar.style.flexShrink = "0";

        // Fit to View button
        const fitButton = createButton("Fit");
        fitButton.title = "Fit to View";

        // Settings button
        const settingsButton = createButton("[Settings]");
        settingsButton.title = "Settings";
        settingsButton.style.fontSize = "14px";
        settingsButton.style.padding = "2px 6px";

        // Find Face button
        const findButton = createButton("Find");
        findButton.title = "Find Face by ID";

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

        // Find Face button handler
        findButton.addEventListener("click", () => {
            const input = prompt("Enter Face ID to find:");
            if (input !== null && input.trim() !== "") {
                const faceId = parseInt(input.trim(), 10);
                if (!isNaN(faceId) && faceId >= 0) {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: "SELECT_FACE_BY_ID",
                            faceId: faceId
                        }, "*");
                    }
                } else {
                    alert("Please enter a valid face ID (non-negative integer)");
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
        infoPanel.style.height = "250px";
        infoPanel.style.overflowY = "auto";
        infoPanel.style.overflowX = "hidden";
        infoPanel.innerHTML = "<div style='color: #888;'>Waiting for analysis data...</div>";

        container.appendChild(infoPanel);

        // Add DOM widget to node
        const widget = this.addDOMWidget(
            "analysis_viewer",
            "CAD_ANALYSIS_VIEWER",
            container,
            {
                getValue() {
                    return JSON.stringify(viewerState);
                },
                setValue(v) {
                    try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                }
            }
        );

        widget.computeSize = function(width) {
            return [width, 650];
        };

        // Store references
        this.analysisIframe = iframe;
        this.analysisInfoPanel = infoPanel;
        this.lastLoadedMeshFile = null;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const meshFile = message.mesh_file?.[0] || message.step_file?.[0];
            const analysisFile = message.analysis_file?.[0];
            const format = message.format?.[0] || "occ";
            const originalFormat = message.original_format?.[0] || "unknown";
            const numVolumes = message.num_volumes?.[0] || 0;
            const numFaces = message.num_faces?.[0] || 0;
            const numEdges = message.num_edges?.[0] || 0;
            const boundsMin = message.bounds_min?.[0] || [0, 0, 0];
            const boundsMax = message.bounds_max?.[0] || [0, 0, 0];
            const extents = message.extents?.[0] || [0, 0, 0];
            const linearDeflection = message.linear_deflection?.[0] || 0.1;
            const visualizationMode = message.visualization_mode?.[0] || "normal";

            // Get batch metadata
            currentBatchSize = message.batch_size?.[0] || 1;
            currentIndex = message.current_index?.[0] || 0;

            // Update navigation controls
            indexLabel.textContent = `${currentIndex + 1} / ${currentBatchSize}`;
            updateNavigationButtons();

            // Dynamically update index widget max based on actual batch size
            const indexWidget = findIndexWidget();
            if (indexWidget) {
                indexWidget.options.max = currentBatchSize - 1;
                // Clamp current value if out of range
                if (indexWidget.value >= currentBatchSize) {
                    indexWidget.value = currentBatchSize - 1;
                }
            }

            if (!meshFile || !analysisFile) {
                console.error("[CADabra] Missing file data from analysis node");
                return;
            }

            // Update info panel with global statistics
            infoPanel.innerHTML = `
                <div style="margin-bottom: 8px;">
                    <strong style="color: #4a9eff;">CAD Analysis</strong>
                </div>
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px; font-size: 11px;">
                    <span style="color: #888;">File:</span>
                    <span>${meshFile}</span>

                    <span style="color: #888;">Format:</span>
                    <span>${originalFormat} -> ${format}</span>

                    <span style="color: #888;">Topology:</span>
                    <span>${numVolumes} volumes, ${numFaces} faces, ${numEdges} edges</span>

                    <span style="color: #888;">Extents:</span>
                    <span>[${extents.map(v => v.toFixed(2)).join(', ')}]</span>

                    <span style="color: #888;">Bounds:</span>
                    <span>[${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]</span>

                    <span style="color: #888;">Deflection:</span>
                    <span>${linearDeflection}</span>
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #444; color: #888; font-size: 11px;">
                    Click on faces in the viewer to analyze them
                </div>
            `;

            // Skip reload if same file already loaded
            if (this.lastLoadedMeshFile === meshFile) {
                console.log("[CADabra] Same file already loaded, skipping reload");
                return;
            }

            // Construct file URLs for viewer
            const meshUrl = `/view?filename=${encodeURIComponent(meshFile)}&type=output&subfolder=`;
            const analysisUrl = `/view?filename=${encodeURIComponent(analysisFile)}&type=output&subfolder=`;

            // Track loaded file
            this.lastLoadedMeshFile = meshFile;

            // Wait for iframe to load, then send data
            const sendDataWhenReady = () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_CAD_ANALYSIS",
                        meshUrl: meshUrl,
                        analysisUrl: analysisUrl,
                        meshFile: meshFile,
                        analysisFile: analysisFile,
                        linearDeflection: linearDeflection,
                        visualizationMode: visualizationMode,
                        nodeId: this.id,
                        cameraState: viewerState.camera_state,
                        selectedFaceId: viewerState.selected_face_id,
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

        // Listen for messages from iframe (face selection updates)
        window.addEventListener("message", (event) => {
            // Without this check, every open viewer instance's listener
            // fires for every iframe's messages, not just its own.
            if (event.source !== iframe.contentWindow) return;
            if (event.data.type === "FACE_SELECTED" && event.data.nodeId === this.id) {
                const faceData = event.data.faceData;

                if (!faceData) {
                    // Deselection - show global stats
                    this.onExecuted(this.lastExecutionData || {});
                    return;
                }

                // Update info panel with selected face data
                infoPanel.innerHTML = `
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #ffa500;">Selected Face #${faceData.face_id}</strong>
                        <button onclick="window.postMessage({type: 'DESELECT_FACE', nodeId: ${this.id}}, '*')"
                                style="float: right; padding: 2px 8px; background: #444; color: #fff; border: none; border-radius: 3px; cursor: pointer; font-size: 10px;">
                            Clear Selection
                        </button>
                    </div>
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: 11px;">
                        <span style="color: #888;">Surface Type:</span>
                        <span style="color: #4aff4a;">${faceData.surface_type || 'Unknown'}</span>

                        <span style="color: #888;">B-rep Type:</span>
                        <span>${faceData.curvature_type || 'Unknown'}</span>

                        <span style="color: #888;">Planar:</span>
                        <span>${faceData.is_planar !== null ? (faceData.is_planar ? 'Yes' : 'No') : 'Unknown'}</span>

                        <span style="color: #888;">Area:</span>
                        <span>${faceData.area !== null ? faceData.area.toFixed(4) : 'N/A'}</span>

                        ${faceData.centroid ? `
                        <span style="color: #888;">Centroid:</span>
                        <span>[${faceData.centroid.map(v => v.toFixed(3)).join(', ')}]</span>
                        ` : ''}

                        ${faceData.normal ? `
                        <span style="color: #888;">Normal:</span>
                        <span>[${faceData.normal.map(v => v.toFixed(3)).join(', ')}]</span>
                        ` : ''}

                        ${faceData.edge_count !== undefined ? `
                        <span style="color: #888;">Edge Count:</span>
                        <span>${faceData.edge_count}</span>
                        ` : ''}

                        ${faceData.adjacent_faces && faceData.adjacent_faces.length > 0 ? `
                        <span style="color: #888;">Adjacent Faces:</span>
                        <span>${faceData.adjacent_faces.join(', ')} (${faceData.adjacent_faces.length} total)</span>
                        ` : ''}
                    </div>
                `;
            } else if (event.data.type === "DESELECT_FACE") {
                // Clear selection in viewer
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "DESELECT_FACE"
                    }, "*");
                }
            } else if (event.data.type === "WIDGET_UPDATE" && event.data.nodeId === this.id) {
                // Sync viewer state from iframe (camera position, face selection)
                const { name, value } = event.data;
                if (name in viewerState) viewerState[name] = value;
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
    name: "cadabra.cadanalysis.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Handle both CADFaceAnalysis (now computation-only but may still be used)
        // and CADAnalysisViewer (the dedicated viewer node)
        if (nodeData.name === "CADAnalysisViewer") {
            createViewerWidget(this, nodeType);
        }
    }
});
