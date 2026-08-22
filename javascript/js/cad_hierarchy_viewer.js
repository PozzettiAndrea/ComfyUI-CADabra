/**
 * CAD Hierarchy Tree Viewer Extension for ComfyUI-CADabra
 * Displays the topological structure of a CAD model as a collapsible tree.
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the hierarchy viewer widget for a node
 */
function createHierarchyViewerWidget(node, nodeType) {
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
        iframe.src = new URL(`../viewer_cad_hierarchy.html?v=${Date.now()}`, import.meta.url).href;

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

        // Expand All button
        const expandBtn = createButton("Expand All");
        expandBtn.title = "Expand all tree nodes";
        expandBtn.addEventListener("click", () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: "EXPAND_ALL" }, "*");
            }
        });

        // Collapse All button
        const collapseBtn = createButton("Collapse All");
        collapseBtn.title = "Collapse all tree nodes";
        collapseBtn.addEventListener("click", () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: "COLLAPSE_ALL" }, "*");
            }
        });

        // Assemble navigation bar
        navBar.appendChild(expandBtn);
        navBar.appendChild(collapseBtn);

        container.appendChild(navBar);

        // Add DOM widget to node
        const widget = this.addDOMWidget(
            "hierarchy_viewer",
            "CAD_HIERARCHY_VIEWER",
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
            return [width, 560];
        };

        // Store references
        this.hierarchyIframe = iframe;
        this.lastLoadedFile = null;

        // Build a /view URL for a file in the output dir (mesh/edges VTPs live there too)
        const viewUrl = (filename) => `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

        const sendLoadMessage = () => {
            const state = this.lastHierarchyState;
            if (!state || !iframe.contentWindow) return;
            iframe.contentWindow.postMessage({
                type: "LOAD_HIERARCHY",
                jsonUrl: viewUrl(state.hierarchyFile),
                meshUrl: state.meshFile ? viewUrl(state.meshFile) : null,
                edgesUrl: state.edgesFile ? viewUrl(state.edgesFile) : null,
                boundsMin: state.boundsMin,
                boundsMax: state.boundsMax,
                hierarchyFile: state.hierarchyFile,
                nodeId: this.id,
                timestamp: Date.now()
            }, "*");
        };

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const hierarchyFile = message.hierarchy_file?.[0];

            if (!hierarchyFile) {
                console.error("[CADabra] Missing hierarchy file from node");
                return;
            }

            // Skip reload if same file already loaded
            if (this.lastLoadedFile === hierarchyFile) {
                console.log("[CADabra] Same hierarchy file already loaded, skipping reload");
                return;
            }

            this.lastLoadedFile = hierarchyFile;
            this.lastHierarchyState = {
                hierarchyFile,
                meshFile: message.mesh_file?.[0] || null,
                edgesFile: message.edges_file?.[0] || null,
                boundsMin: message.bounds_min?.[0] || [0, 0, 0],
                boundsMax: message.bounds_max?.[0] || [1, 1, 1],
            };

            // Wait for iframe to load, then send data
            if (iframe.contentDocument && iframe.contentDocument.readyState === "complete") {
                sendLoadMessage();
            } else {
                iframe.onload = sendLoadMessage;
            }
        };

        // Listen for messages from iframe
        window.addEventListener("message", (event) => {
            // Without this check, every open viewer instance's listener
            // fires for every iframe's messages, not just its own.
            if (event.source !== iframe.contentWindow) return;
            if (event.data.type === "HIERARCHY_VIEWER_READY" && this.lastHierarchyState) {
                // Re-send data if viewer was reloaded
                sendLoadMessage();
            }
        });

        return result;
    };
}

app.registerExtension({
    name: "cadabra.cadhierarchy.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADHierarchyTree") {
            createHierarchyViewerWidget(this, nodeType);
        }
    }
});
