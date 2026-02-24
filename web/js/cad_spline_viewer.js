/**
 * CAD Spline Viewer Extension for ComfyUI-CADabra
 * Interactive visualization of B-spline, Bezier, and NURBS surface parameters.
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the spline viewer widget for a node
 */
function createSplineViewerWidget(node, nodeType) {
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
        iframe.src = "/extensions/ComfyUI-CADabra/viewer_spline.html?v=" + Date.now();

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

        // Face info label
        const faceLabel = document.createElement("span");
        faceLabel.style.color = "#888";
        faceLabel.style.fontSize = "11px";
        faceLabel.style.marginRight = "auto";
        faceLabel.textContent = "Face: -/-";

        // Surface type label
        const typeLabel = document.createElement("span");
        typeLabel.style.padding = "2px 8px";
        typeLabel.style.backgroundColor = "#2d3d4a";
        typeLabel.style.color = "#4a9eff";
        typeLabel.style.borderRadius = "3px";
        typeLabel.style.fontSize = "10px";
        typeLabel.style.fontWeight = "600";
        typeLabel.textContent = "-";

        // Assemble navigation bar
        navBar.appendChild(faceLabel);
        navBar.appendChild(typeLabel);

        container.appendChild(navBar);

        // Add DOM widget to node
        const widget = this.addDOMWidget(
            "spline_viewer",
            "CAD_SPLINE_VIEWER",
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
        this.splineIframe = iframe;
        this.splineFaceLabel = faceLabel;
        this.splineTypeLabel = typeLabel;
        this.lastLoadedFile = null;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const splineFile = message.spline_file?.[0];
            const surfaceType = message.surface_type?.[0] || "-";
            const faceIndex = message.face_index?.[0] ?? 0;
            const totalFaces = message.total_faces?.[0] ?? 1;

            if (!splineFile) {
                console.error("[CADabra] Missing spline file from node");
                return;
            }

            // Update labels
            faceLabel.textContent = `Face: ${faceIndex}/${totalFaces - 1}`;
            typeLabel.textContent = surfaceType;

            // Update type label color based on surface type
            if (surfaceType === "BSpline") {
                typeLabel.style.backgroundColor = "#2d3d4a";
                typeLabel.style.color = "#2196f3";
            } else if (surfaceType === "Bezier") {
                typeLabel.style.backgroundColor = "#4a3d2d";
                typeLabel.style.color = "#ff9800";
            } else if (surfaceType === "Plane") {
                typeLabel.style.backgroundColor = "#3d2d4a";
                typeLabel.style.color = "#9c27b0";
            } else {
                typeLabel.style.backgroundColor = "#2a2a2a";
                typeLabel.style.color = "#888";
            }

            // Skip reload if same file already loaded
            if (this.lastLoadedFile === splineFile) {
                console.log("[CADabra] Same spline file already loaded, skipping reload");
                return;
            }

            // Construct file URL for viewer
            const jsonUrl = `/view?filename=${encodeURIComponent(splineFile)}&type=output&subfolder=`;

            // Track loaded file
            this.lastLoadedFile = splineFile;

            // Wait for iframe to load, then send data
            const sendDataWhenReady = () => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_SPLINE",
                        jsonUrl: jsonUrl,
                        splineFile: splineFile,
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

        // Listen for messages from iframe
        window.addEventListener("message", (event) => {
            if (event.data.type === "SPLINE_VIEWER_READY" && this.lastLoadedFile) {
                // Re-send data if viewer was reloaded
                const jsonUrl = `/view?filename=${encodeURIComponent(this.lastLoadedFile)}&type=output&subfolder=`;
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: "LOAD_SPLINE",
                        jsonUrl: jsonUrl,
                        splineFile: this.lastLoadedFile,
                        nodeId: this.id,
                        timestamp: Date.now()
                    }, "*");
                }
            }
        });

        return result;
    };
}

app.registerExtension({
    name: "cadabra.cadspline.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADSplineViewer") {
            createSplineViewerWidget(this, nodeType);
        }
    }
});
