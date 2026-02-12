import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "cadabra.cadrilleinference",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CadrilleInference") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const container = document.createElement("div");
                container.style.padding = "8px";
                container.style.backgroundColor = "#1a2a1a";
                container.style.color = "#9d9";
                container.style.fontSize = "10px";
                container.style.fontFamily = "monospace";
                container.style.whiteSpace = "pre-wrap";
                container.style.borderRadius = "4px";
                container.style.maxHeight = "200px";
                container.style.overflowY = "auto";
                container.style.border = "1px solid #3a5a3a";
                container.textContent = "Run to generate CadQuery code...";

                const widget = this.addDOMWidget("code_display", "CODE_DISPLAY", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });
                widget.computeSize = () => [320, 180];
                this.codeContainer = container;

                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (message?.text?.[0]) {
                        this.codeContainer.textContent = message.text[0];
                    }
                };
                return r;
            };
        }
    }
});
