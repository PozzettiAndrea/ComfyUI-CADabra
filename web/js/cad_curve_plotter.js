/**
 * CAD Curve Plotter Extension for ComfyUI-CADabra
 * 2D visualization of B-spline/Bezier curves with control points
 */

import { app } from "../../../scripts/app.js";

/**
 * Create the curve plotter widget for CADCurvePlotter node
 */
function createCurvePlotterWidget(node, nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // Create container for viewer
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.backgroundColor = "#1a1a2e";
        container.style.borderRadius = "4px";
        container.style.overflow = "hidden";

        // Create iframe for curve plotter viewer
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.flex = "1 1 0";
        iframe.style.minHeight = "0";
        iframe.style.border = "none";
        iframe.style.backgroundColor = "#1a1a2e";
        // Prevent blurriness from CSS transforms
        iframe.style.transform = "translateZ(0)";
        iframe.style.imageRendering = "pixelated";
        iframe.src = "/extensions/ComfyUI-CADabra/viewer_curve_plotter.html?v=" + Date.now();

        container.appendChild(iframe);

        // Create info bar at bottom
        const infoBar = document.createElement("div");
        infoBar.style.display = "flex";
        infoBar.style.alignItems = "center";
        infoBar.style.justifyContent = "space-between";
        infoBar.style.gap = "8px";
        infoBar.style.padding = "6px 10px";
        infoBar.style.backgroundColor = "#252542";
        infoBar.style.borderTop = "1px solid #333";
        infoBar.style.flexShrink = "0";
        infoBar.style.fontSize = "10px";
        infoBar.style.color = "#888";

        // Label
        const label = document.createElement("span");
        label.innerHTML = '<span style="color: #ff44ff; font-weight: bold;">2D</span> Curve Plotter';

        // Status
        const status = document.createElement("span");
        status.id = "plotter-status";
        status.textContent = "Waiting for data...";
        status.style.color = "#666";

        infoBar.appendChild(label);
        infoBar.appendChild(status);
        container.appendChild(infoBar);

        // Add widget to node
        const widget = this.addDOMWidget("curve_plotter", "CADCurvePlotter", container, {
            getValue() {
                return null;
            },
            setValue(v) {
                // Not used
            },
        });

        widget.computeSize = function () {
            return [400, 350];
        };

        // Store references
        this.plotterIframe = iframe;
        this.plotterStatus = status;

        // Handle execution results
        this.onExecuted = function(message) {
            if (!message) return;

            const plotData = message.plot_data?.[0];
            if (!plotData) {
                status.textContent = "No plot data";
                status.style.color = "#ff8844";
                return;
            }

            try {
                const data = JSON.parse(plotData);

                // Update status
                if (data.success) {
                    status.textContent = `Edge #${data.edge_id} | ${data.curve_type} | Deg ${data.degree}`;
                    status.style.color = "#44ff88";
                } else {
                    status.textContent = "Error - see viewer";
                    status.style.color = "#ff4444";
                }

                // Send data to iframe with parent DPR info
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: 'LOAD_PLOT_DATA',
                        data: data,
                        parentDPR: window.devicePixelRatio || 1
                    }, '*');
                }
            } catch (e) {
                console.error('Failed to parse plot data:', e);
                status.textContent = "Parse error";
                status.style.color = "#ff4444";
            }
        };

        return result;
    };

    // Ensure widget resizes properly
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function(size) {
        if (onResize) {
            onResize.apply(this, arguments);
        }
        // Trigger redraw in iframe on resize
        if (this.plotterIframe && this.plotterIframe.contentWindow) {
            this.plotterIframe.contentWindow.dispatchEvent(new Event('resize'));
        }
    };
}

// Register the extension
app.registerExtension({
    name: "CADabra.CurvePlotter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADCurvePlotter") {
            createCurvePlotterWidget(nodeType.prototype, nodeType);
        }
    }
});
