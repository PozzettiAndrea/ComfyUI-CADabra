import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.cadfaceanalysis",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CADFaceAnalysis") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create text display container for report
                const container = document.createElement("div");
                container.style.padding = "8px";
                container.style.backgroundColor = "#2a2a2a";
                container.style.color = "#ccc";
                container.style.fontSize = "11px";
                container.style.fontFamily = "monospace";
                container.style.whiteSpace = "pre-wrap";
                container.style.borderRadius = "4px";
                container.style.marginTop = "4px";
                container.style.maxHeight = "200px";
                container.style.overflowY = "auto";
                container.textContent = "Run to analyze CAD faces...";

                // Add DOM widget
                const widget = this.addDOMWidget("report_display", "TEXT_DISPLAY", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });
                widget.computeSize = () => [300, 150];

                this.reportContainer = container;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (message?.text?.[0]) {
                        this.reportContainer.textContent = message.text[0];
                    }
                };

                return r;
            };
        }
    }
});
