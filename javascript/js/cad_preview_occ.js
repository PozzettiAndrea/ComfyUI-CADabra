import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.cadpreview.occ",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match the PreviewCADOCC node from Python NODE_CLASS_MAPPINGS
        if (nodeData.name === "PreviewCADOCC") {

            // Hook into node creation lifecycle
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer and info
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";

                // Create iframe for VTK.js viewer (using analysis viewer which loads VTP)
                const iframe = document.createElement("iframe");
                iframe.src = new URL(`../viewer_cad_analysis.html?v=${Date.now()}`, import.meta.url).href;
                iframe.style.width = "100%";
                iframe.style.height = "500px";
                iframe.style.border = "1px solid #333";
                iframe.style.borderRadius = "4px";
                iframe.style.backgroundColor = "#1a1a1a";

                // Create navigation bar
                const navBar = document.createElement("div");
                navBar.style.display = "flex";
                navBar.style.alignItems = "center";
                navBar.style.gap = "4px";
                navBar.style.padding = "6px 8px";
                navBar.style.backgroundColor = "#2a2a2a";
                navBar.style.borderTop = "1px solid #333";

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

                // Fit button handler
                fitButton.addEventListener("click", () => {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({ type: "FIT_VIEW" }, "*");
                    }
                });

                // Create info panel for CAD metadata
                const infoPanel = document.createElement("div");
                infoPanel.style.padding = "8px";
                infoPanel.style.backgroundColor = "#2a2a2a";
                infoPanel.style.color = "#ccc";
                infoPanel.style.fontSize = "11px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.borderTop = "1px solid #333";
                infoPanel.style.maxHeight = "100px";
                infoPanel.style.overflowY = "auto";
                infoPanel.innerHTML = "<div>Waiting for CAD model...</div>";

                container.appendChild(iframe);
                container.appendChild(navBar);
                container.appendChild(infoPanel);

                // Add DOM widget to the node
                const widget = this.addDOMWidget(
                    "preview_occ",          // Widget name
                    "CAD_PREVIEW_OCC",      // Widget type
                    container,              // DOM element
                    {
                        getValue() { return ""; },
                        setValue(v) { }
                    }
                );

                // Set widget size (width, height)
                widget.computeSize = () => [512, 650];

                // Store references for later access
                this.cadViewerIframeOCC = iframe;
                this.cadInfoPanelOCC = infoPanel;
                this.lastLoadedMeshFile = null;

                // Handle execution results - receives UI data from Python
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    console.log("[CADabra Preview] onExecuted:", message);

                    if (message) {
                        const meshFile = message.mesh_file ? message.mesh_file[0] : null;
                        const format = message.format ? message.format[0] : "unknown";
                        const originalFormat = message.original_format ? message.original_format[0] : "unknown";
                        const numVolumes = message.num_volumes ? message.num_volumes[0] : 0;
                        const numFaces = message.num_faces ? message.num_faces[0] : 0;
                        const numEdges = message.num_edges ? message.num_edges[0] : 0;
                        const linearDeflection = message.linear_deflection ? message.linear_deflection[0] : 0.1;
                        const boundsMin = message.bounds_min ? message.bounds_min[0] : [0, 0, 0];
                        const boundsMax = message.bounds_max ? message.bounds_max[0] : [0, 0, 0];
                        const extents = message.extents ? message.extents[0] : [0, 0, 0];
                        const faceTypeCounts = message.face_type_counts ? message.face_type_counts[0] : {};
                        const edgeTypeCounts = message.edge_type_counts ? message.edge_type_counts[0] : {};

                        const formatTypeCounts = (counts) => Object.entries(counts)
                            .sort((a, b) => b[1] - a[1])
                            .map(([name, count]) => `${count} ${name}${count === 1 ? '' : 's'}`)
                            .join(', ');
                        const faceTypesLine = formatTypeCounts(faceTypeCounts);
                        const edgeTypesLine = formatTypeCounts(edgeTypeCounts);

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="margin-bottom: 4px;"><strong>CAD Model Info</strong></div>
                            <div>Format: ${originalFormat} -> ${format}</div>
                            <div>Topology: ${numVolumes} volumes, ${numFaces} faces, ${numEdges} edges</div>
                            <div>Face types: ${faceTypesLine || 'n/a'}</div>
                            <div>Edge types: ${edgeTypesLine || 'n/a'}</div>
                            <div>Deflection: ${linearDeflection}</div>
                            <div>Extents: [${extents.map(v => v.toFixed(2)).join(', ')}]</div>
                            <div>Bounds: [${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]</div>
                        `;

                        if (!meshFile) {
                            console.error("[CADabra Preview] No mesh_file in message");
                            return;
                        }

                        // Skip reload if same file
                        if (this.lastLoadedMeshFile === meshFile) {
                            console.log("[CADabra Preview] Same file already loaded, skipping");
                            return;
                        }
                        this.lastLoadedMeshFile = meshFile;

                        // Build VTP file URL
                        const meshUrl = `/view?filename=${encodeURIComponent(meshFile)}&type=output&subfolder=`;

                        console.log("[CADabra Preview] Loading VTP:", meshUrl);

                        // Send message to iframe to load VTP mesh
                        const sendDataWhenReady = () => {
                            if (iframe.contentWindow) {
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_CAD_ANALYSIS",
                                    meshUrl: meshUrl,
                                    analysisUrl: null,  // No analysis data for simple preview
                                    meshFile: meshFile,
                                    analysisFile: null,
                                    linearDeflection: linearDeflection,
                                    visualizationMode: "normal",
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
                    }
                };

                return r;
            };
        }
    }
});
