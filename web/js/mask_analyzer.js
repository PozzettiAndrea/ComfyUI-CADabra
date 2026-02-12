import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.maskanalyzer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MaskAnalyzer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create text display container
                const container = document.createElement("div");
                container.style.padding = "8px";
                container.style.backgroundColor = "#2a2a2a";
                container.style.color = "#ccc";
                container.style.fontSize = "12px";
                container.style.fontFamily = "monospace";
                container.style.whiteSpace = "pre-wrap";
                container.style.borderRadius = "4px";
                container.style.marginTop = "4px";
                container.textContent = "Run to analyze mask...";

                // Add DOM widget
                const widget = this.addDOMWidget("stats_display", "TEXT_DISPLAY", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });
                widget.computeSize = () => [220, 100];

                this.statsContainer = container;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (message?.text?.[0]) {
                        this.statsContainer.textContent = message.text[0];
                    }
                };

                return r;
            };
        }
    }
});
