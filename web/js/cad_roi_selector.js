import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.roi_selector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "CADROISelector") return;

        console.log("[CADabra ROI] Registering extension for CADROISelector");

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            console.log("[CADabra ROI] Node created, id:", this.id);

            // Create container
            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.display = "flex";
            container.style.flexDirection = "column";

            // Create iframe for ROI selector viewer
            const iframe = document.createElement("iframe");
            iframe.src = "/extensions/ComfyUI-CADabra/viewer_roi_selector.html?v=" + Date.now();
            iframe.style.width = "100%";
            iframe.style.height = "500px";
            iframe.style.border = "1px solid #333";
            iframe.style.borderRadius = "4px";
            iframe.style.backgroundColor = "#1a1a1a";

            // Create info panel
            const infoPanel = document.createElement("div");
            infoPanel.style.padding = "8px";
            infoPanel.style.backgroundColor = "#2a2a2a";
            infoPanel.style.color = "#ccc";
            infoPanel.style.fontSize = "11px";
            infoPanel.style.fontFamily = "monospace";
            infoPanel.style.borderTop = "1px solid #333";
            infoPanel.innerHTML = "<div>Click and drag on the model to select ROI</div>";

            container.appendChild(iframe);
            container.appendChild(infoPanel);

            // Add DOM widget
            const widget = this.addDOMWidget(
                "roi_viewer",
                "CAD_ROI_VIEWER",
                container,
                {
                    getValue() { return ""; },
                    setValue(v) { }
                }
            );

            widget.computeSize = () => [450, 560];

            // Store references
            this.roiSelectorIframe = iframe;
            this.roiInfoPanel = infoPanel;

            // Get reference to node
            const node = this;

            // Helper to find the roi_value widget
            const findRoiWidget = () => {
                const w = node.widgets?.find(w => w.name === "roi_value");
                console.log("[CADabra ROI] findRoiWidget:", w ? "found" : "not found", "widgets:", node.widgets?.map(w => w.name));
                return w;
            };

            // Listen for ROI updates from iframe
            const messageHandler = (event) => {
                if (event.data.type === "ROI_SELECTED" && event.data.nodeId === node.id) {
                    console.log("[CADabra ROI] Received ROI_SELECTED:", event.data.roi);

                    // Find and update the roi_value widget
                    const roiWidget = findRoiWidget();
                    if (roiWidget) {
                        roiWidget.value = event.data.roi;
                        console.log("[CADabra ROI] Updated roi_value widget to:", roiWidget.value);

                        // Force widget callback if exists
                        if (roiWidget.callback) {
                            roiWidget.callback(roiWidget.value);
                        }
                    } else {
                        console.warn("[CADabra ROI] roi_value widget not found!");
                    }

                    // Update info panel
                    if (event.data.roi) {
                        const parts = event.data.roi.split(",").map(v => parseFloat(v).toFixed(2));
                        infoPanel.innerHTML = `
                            <div><strong>ROI Selected:</strong></div>
                            <div>X: ${parts[0]} to ${parts[2]}</div>
                            <div>Y: ${parts[1]} to ${parts[3]}</div>
                            <div style="margin-top: 4px; color: #88ff88;">Ready to use with raytracer</div>
                        `;
                    }

                    // Mark graph as dirty
                    if (app.graph) {
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            };
            window.addEventListener("message", messageHandler);

            // Clean up listener when node is removed
            const onRemoved = this.onRemoved;
            this.onRemoved = function() {
                window.removeEventListener("message", messageHandler);
                onRemoved?.apply(this, arguments);
            };

            // Handle execution results
            const onExecuted = this.onExecuted;
            this.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                console.log("[CADabra ROI] onExecuted called:", message);

                if (message) {
                    const glbFilename = message.glb_file ? message.glb_file[0] : null;
                    const boundsMin = message.bounds_min ? message.bounds_min[0] : [0, 0, 0];
                    const boundsMax = message.bounds_max ? message.bounds_max[0] : [0, 0, 0];
                    const currentRoi = message.current_roi ? message.current_roi[0] : "";
                    const meshVertexCount = message.mesh_vertex_count ? message.mesh_vertex_count[0] : 0;
                    const meshFaceCount = message.mesh_face_count ? message.mesh_face_count[0] : 0;
                    const faceBboxes = message.face_bboxes ? message.face_bboxes[0] : [];
                    const triangleFaceIds = message.triangle_face_ids ? message.triangle_face_ids[0] : [];

                    console.log("[CADabra ROI] Extracted - glb:", glbFilename, "roi:", currentRoi, "faces:", faceBboxes.length, "triangles:", triangleFaceIds.length);

                    // Update info panel with model info
                    let infoHtml = `
                        <div><strong>Model loaded</strong></div>
                        <div>Mesh: ${meshVertexCount} vertices, ${meshFaceCount} triangles, ${faceBboxes.length} CAD faces</div>
                        <div>Bounds: [${boundsMin.map(v => v.toFixed(2)).join(", ")}] to [${boundsMax.map(v => v.toFixed(2)).join(", ")}]</div>
                    `;

                    if (currentRoi) {
                        const parts = currentRoi.split(",").map(v => parseFloat(v).toFixed(2));
                        infoHtml += `
                            <div style="margin-top: 4px;"><strong>Current ROI:</strong></div>
                            <div>X: ${parts[0]} to ${parts[2]}</div>
                            <div>Y: ${parts[1]} to ${parts[3]}</div>
                        `;
                    } else {
                        infoHtml += `<div style="margin-top: 4px; color: #aaa;">Click face or drag to select ROI</div>`;
                    }

                    infoPanel.innerHTML = infoHtml;

                    // Build GLB path
                    const glbpath = glbFilename ? `/view?filename=${encodeURIComponent(glbFilename)}&type=temp` : null;

                    if (glbpath) {
                        console.log("[CADabra ROI] Sending LOAD_MODEL to iframe, currentRoi:", currentRoi);

                        // Send to iframe with face data
                        iframe.contentWindow.postMessage({
                            type: "LOAD_MODEL",
                            nodeId: node.id,
                            glbUrl: glbpath,
                            boundsMin: boundsMin,
                            boundsMax: boundsMax,
                            currentRoi: currentRoi,
                            faceBboxes: faceBboxes,
                            triangleFaceIds: triangleFaceIds
                        }, "*");
                    }
                }
            };

            return r;
        };
    }
});
