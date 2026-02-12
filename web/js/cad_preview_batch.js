import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.cadpreview.batch",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match the PreviewCADBatch node from Python NODE_CLASS_MAPPINGS
        if (nodeData.name === "PreviewCADBatch") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + navigation + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";

                // Create iframe for VTK.js viewer (using analysis viewer which loads VTP)
                const iframe = document.createElement("iframe");
                iframe.src = "/extensions/ComfyUI-CADabra/viewer_analysis.html?v=" + Date.now();
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";

                // Create navigation bar for batch controls
                const navBar = document.createElement("div");
                navBar.style.backgroundColor = "#1a1a1a";
                navBar.style.borderTop = "1px solid #444";
                navBar.style.padding = "8px 12px";
                navBar.style.display = "flex";
                navBar.style.alignItems = "center";
                navBar.style.justifyContent = "center";
                navBar.style.gap = "12px";
                navBar.style.fontSize = "12px";
                navBar.style.color = "#ccc";
                navBar.style.flexShrink = "0";

                // Helper to create styled buttons
                const createButton = (text, width = null) => {
                    const btn = document.createElement("button");
                    btn.textContent = text;
                    btn.style.padding = "4px 8px";
                    btn.style.cursor = "pointer";
                    btn.style.backgroundColor = "#333";
                    btn.style.color = "#ccc";
                    btn.style.border = "1px solid #555";
                    btn.style.borderRadius = "3px";
                    btn.style.fontSize = "11px";
                    if (width) btn.style.width = width;
                    btn.onmouseenter = () => btn.style.backgroundColor = "#444";
                    btn.onmouseleave = () => btn.style.backgroundColor = "#333";
                    return btn;
                };

                // Previous button
                const prevButton = createButton("◀");
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
                const nextButton = createButton("▶");
                nextButton.title = "Next";

                // Separator
                const separator = document.createElement("span");
                separator.style.width = "1px";
                separator.style.height = "16px";
                separator.style.backgroundColor = "#555";
                separator.style.margin = "0 4px";

                // Fit to View button
                const fitButton = createButton("Fit");
                fitButton.title = "Fit to View";

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

                // Separator for view buttons
                const separator2 = document.createElement("span");
                separator2.style.width = "1px";
                separator2.style.height = "16px";
                separator2.style.backgroundColor = "#555";
                separator2.style.margin = "0 4px";

                // Assemble navigation bar
                navBar.appendChild(prevButton);
                navBar.appendChild(indexLabel);
                navBar.appendChild(nextButton);
                navBar.appendChild(separator);
                viewButtonElements.forEach(btn => navBar.appendChild(btn));
                navBar.appendChild(separator2);
                navBar.appendChild(fitButton);

                // Fit button handler
                fitButton.addEventListener("click", () => {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({ type: "FIT_VIEW" }, "*");
                    }
                });

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#2a2a2a";
                infoPanel.style.borderTop = "1px solid #333";
                infoPanel.style.padding = "8px";
                infoPanel.style.fontSize = "11px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.innerHTML = '<span style="color: #888;">CAD info will appear here after execution</span>';

                // Add iframe, navigation, and info panel to container
                container.appendChild(iframe);
                container.appendChild(navBar);
                container.appendChild(infoPanel);

                // Add DOM widget to the node
                const widget = this.addDOMWidget(
                    "preview_cad_batch",
                    "CAD_PREVIEW_BATCH",
                    container,
                    {
                        getValue() { return ""; },
                        setValue(v) { }
                    }
                );

                // Set widget size (increased for navigation + info panel)
                widget.computeSize = () => [512, 700];

                // Store references
                this.cadViewerIframeBatch = iframe;
                this.cadInfoPanelBatch = infoPanel;
                this.cadNavBarBatch = navBar;
                this.lastLoadedMeshFile = null;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Find the index widget (created by ComfyUI from INPUT_TYPES)
                const indexWidget = this.widgets.find(w => w.name === 'index');

                // Track batch state
                let currentBatchSize = 1;
                let currentIndex = 0;

                // Add callback to auto-execute when index changes manually
                if (indexWidget) {
                    const originalCallback = indexWidget.callback;
                    indexWidget.callback = function(value) {
                        const result = originalCallback?.apply(this, arguments);
                        currentIndex = value;
                        app.queuePrompt();
                        return result;
                    };
                }

                // Button click handlers
                prevButton.addEventListener("click", () => {
                    if (indexWidget && currentIndex > 0) {
                        indexWidget.value = currentIndex - 1;
                        app.queuePrompt();
                    }
                });

                nextButton.addEventListener("click", () => {
                    if (indexWidget && currentIndex < currentBatchSize - 1) {
                        indexWidget.value = currentIndex + 1;
                        app.queuePrompt();
                    }
                });

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

                // Set initial node size
                this.setSize([512, 700]);

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    console.log("[CADabra Batch] onExecuted:", message);

                    if (message) {
                        const meshFile = message.mesh_file ? message.mesh_file[0] : null;

                        // Get batch metadata
                        currentBatchSize = message.batch_size?.[0] || 1;
                        currentIndex = message.current_index?.[0] || 0;

                        // Update navigation controls
                        indexLabel.textContent = `${currentIndex + 1} / ${currentBatchSize}`;
                        updateNavigationButtons();

                        // Dynamically update index widget max based on actual batch size
                        if (indexWidget) {
                            indexWidget.options.max = currentBatchSize - 1;

                            // Clamp current value if out of range
                            if (indexWidget.value >= currentBatchSize) {
                                indexWidget.value = currentBatchSize - 1;
                            }
                        }

                        const format = message.format ? message.format[0] : "unknown";
                        const originalFormat = message.original_format ? message.original_format[0] : "unknown";
                        const numVolumes = message.num_volumes ? message.num_volumes[0] : 0;
                        const numFaces = message.num_faces ? message.num_faces[0] : 0;
                        const numEdges = message.num_edges ? message.num_edges[0] : 0;
                        const linearDeflection = message.linear_deflection ? message.linear_deflection[0] : 0.1;
                        const boundsMin = message.bounds_min ? message.bounds_min[0] : [0, 0, 0];
                        const boundsMax = message.bounds_max ? message.bounds_max[0] : [0, 0, 0];
                        const extents = message.extents ? message.extents[0] : [0, 0, 0];

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="margin-bottom: 4px;"><strong>CAD Model Info (${currentIndex + 1}/${currentBatchSize})</strong></div>
                            <div>Format: ${originalFormat} → ${format}</div>
                            <div>Topology: ${numVolumes} volumes, ${numFaces} faces, ${numEdges} edges</div>
                            <div>Deflection: ${linearDeflection}</div>
                            <div>Extents: [${extents.map(v => v.toFixed(2)).join(', ')}]</div>
                            <div>Bounds: [${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]</div>
                        `;

                        if (!meshFile) {
                            console.error("[CADabra Batch] No mesh_file in message");
                            return;
                        }

                        // Skip reload if same file
                        if (this.lastLoadedMeshFile === meshFile) {
                            console.log("[CADabra Batch] Same file already loaded, skipping");
                            return;
                        }
                        this.lastLoadedMeshFile = meshFile;

                        // Build VTP file URL
                        const meshUrl = `/view?filename=${encodeURIComponent(meshFile)}&type=output&subfolder=`;

                        console.log("[CADabra Batch] Loading VTP:", meshUrl);

                        // Send message to iframe to load VTP mesh
                        const sendDataWhenReady = () => {
                            if (iframe.contentWindow) {
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_CAD_ANALYSIS",
                                    meshUrl: meshUrl,
                                    analysisUrl: null,  // No analysis data for batch preview
                                    meshFile: meshFile,
                                    analysisFile: null,
                                    linearDeflection: linearDeflection,
                                    visualizationMode: "normal",
                                    nodeId: this.id,
                                    timestamp: Date.now()
                                }, "*");
                            }
                        };

                        if (iframeLoaded) {
                            sendDataWhenReady();
                        } else {
                            iframe.onload = sendDataWhenReady;
                        }
                    }
                };

                return r;
            };
        }
    }
});
