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

                // Create iframe for OpenCascade.js viewer
                const iframe = document.createElement("iframe");
                iframe.src = "/extensions/ComfyUI-CADabra/viewer_occ.html?v=" + Date.now();
                iframe.style.width = "100%";
                iframe.style.height = "600px";
                iframe.style.border = "1px solid #333";
                iframe.style.borderRadius = "4px";
                iframe.style.backgroundColor = "#1a1a1a";

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
                widget.computeSize = () => [512, 720];

                // Store references for later access
                this.cadViewerIframeOCC = iframe;
                this.cadInfoPanelOCC = infoPanel;

                // Handle execution results - receives UI data from Python
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    // message IS the UI data from Python (not message.ui!)
                    if (message?.cad_file && message.cad_file[0]) {
                        const filename = message.cad_file[0];
                        const format = message.format ? message.format[0] : "unknown";
                        const originalFormat = message.original_format ? message.original_format[0] : "unknown";
                        const numVolumes = message.num_volumes ? message.num_volumes[0] : 0;
                        const numFaces = message.num_faces ? message.num_faces[0] : 0;
                        const numEdges = message.num_edges ? message.num_edges[0] : 0;
                        const boundsMin = message.bounds_min ? message.bounds_min[0] : [0, 0, 0];
                        const boundsMax = message.bounds_max ? message.bounds_max[0] : [0, 0, 0];
                        const extents = message.extents ? message.extents[0] : [0, 0, 0];

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="margin-bottom: 4px;"><strong>CAD Model Info</strong></div>
                            <div>File: ${filename}</div>
                            <div>Format: ${originalFormat} → ${format}</div>
                            <div>Topology: ${numVolumes} volumes, ${numFaces} faces, ${numEdges} edges</div>
                            <div>Extents: [${extents.map(v => v.toFixed(2)).join(', ')}]</div>
                            <div>Bounds: [${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]</div>
                        `;

                        // Build file path using ComfyUI's /view endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        console.log("[CADabra OCC] Loading CAD file:", filepath);

                        // Send message to iframe to load CAD model
                        iframe.contentWindow.postMessage({
                            type: "LOAD_CAD",
                            filepath: filepath,
                            format: format,
                            timestamp: Date.now()
                        }, "*");
                    }
                };

                return r;
            };
        }
    }
});
