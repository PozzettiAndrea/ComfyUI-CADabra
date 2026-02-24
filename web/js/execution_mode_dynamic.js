/**
 * ComfyUI-CADabra Execution Mode Widget Management
 *
 * Controls visibility of num_workers and timeout widgets based on execution_mode.
 * When execution_mode = "single_core": hide num_workers, timeout
 * When execution_mode = "parallel": show num_workers, timeout
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false;

function log(...args) {
    if (DEBUG) {
        console.log("[CADabra-ExecMode]", ...args);
    }
}

console.log("[CADabra-ExecMode] Loading execution_mode_dynamic.js extension...");

// List of node names that have execution_mode
const NODES_WITH_EXECUTION_MODE = [
    "CADSewFaces",
    "CADHealShape",
    "CADFixDegenerateFaces",
    "CAD_Load_From_Folder",
    "CAD_Remesh_OCC_Advanced"
];

/**
 * Hide a widget by removing it from the node's widgets array.
 * Stores original index for restoration.
 */
function hideWidget(node, widget) {
    if (!widget) return;

    // If already hidden, skip
    if (widget._hidden) {
        log("Widget already hidden:", widget.name);
        return;
    }

    log("Hiding widget:", widget.name);

    // Find widget's current index in the widgets array
    const index = node.widgets.indexOf(widget);
    if (index === -1) {
        log("  ERROR: Widget not found in node.widgets array!");
        return;
    }

    // Store original properties for restoration
    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    // Store the original index and mark as hidden
    widget._originalIndex = index;
    widget._hidden = true;

    // Remove widget from the array
    node.widgets.splice(index, 1);

    log("  Widget removed from array at index:", index);
}

/**
 * Show a widget by re-inserting it into the node's widgets array
 * at its original position.
 */
function showWidget(node, widget) {
    if (!widget) return;

    // If not hidden, skip
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

    // Restore original properties
    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;

        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    // Re-insert widget into array at original position
    const targetIndex = widget._originalIndex;
    const insertIndex = Math.min(targetIndex, node.widgets.length);

    node.widgets.splice(insertIndex, 0, widget);

    log("  Widget restored to array at index:", insertIndex);

    // Clear hidden flag
    widget._hidden = false;
}

/**
 * Refresh the node's canvas and recalculate its size after widget changes.
 */
function refreshNodeSize(node) {
    // Force canvas redraw immediately
    node.setDirtyCanvas(true, true);
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }

    // Delay size recalculation to ensure widget hiding completes
    requestAnimationFrame(() => {
        // Recalculate node size after widgets are fully hidden
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);

        // Mark as dirty again after resize
        node.setDirtyCanvas(true, true);
        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }

        // Final redraw after everything settles
        requestAnimationFrame(() => {
            if (app.canvas) {
                app.canvas.draw(true, true);
            }
            log("Widget visibility update complete");
        });
    });
}

/**
 * Setup execution mode visibility for a node.
 * Finds execution_mode, num_workers, and timeout widgets and wires up callbacks.
 */
function setupExecutionModeNode(node) {
    const executionModeWidget = node.widgets?.find(w => w.name === "execution_mode");
    const numWorkersWidget = node.widgets?.find(w => w.name === "num_workers");
    const timeoutWidget = node.widgets?.find(w => w.name === "timeout");

    if (!executionModeWidget) {
        log("No execution_mode widget found for node:", node.type);
        return;
    }

    log("Setting up execution_mode for node:", node.type);
    log("Found widgets:", {
        execution_mode: !!executionModeWidget,
        num_workers: !!numWorkersWidget,
        timeout: !!timeoutWidget
    });

    /**
     * Update visibility of num_workers and timeout based on execution_mode.
     */
    const updateForExecutionMode = (mode) => {
        log("Updating for execution_mode:", mode);

        if (mode === "parallel") {
            showWidget(node, numWorkersWidget);
            showWidget(node, timeoutWidget);
        } else {
            // single_core - hide parallel-only widgets
            hideWidget(node, numWorkersWidget);
            hideWidget(node, timeoutWidget);
        }

        refreshNodeSize(node);
    };

    // Wire up callback for execution_mode
    const origCallback = executionModeWidget.callback;
    executionModeWidget.callback = function(value) {
        log("execution_mode changed to:", value);
        const result = origCallback?.apply(this, arguments);
        updateForExecutionMode(value);
        return result;
    };

    // Initialize visibility based on current value
    updateForExecutionMode(executionModeWidget.value);

    log("Execution mode setup complete for:", node.type);
}

// Main extension registration
app.registerExtension({
    name: "cadabra.execution_mode.dynamic",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (NODES_WITH_EXECUTION_MODE.includes(nodeData.name)) {
            log("beforeRegisterNodeDef: Found", nodeData.name);

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                log("onNodeCreated: Setting up", nodeData.name);

                // Use setTimeout to ensure widgets are fully initialized
                const node = this;
                setTimeout(() => {
                    setupExecutionModeNode(node);
                }, 100);

                return r;
            };
        }
    }
});

log("CADabra execution mode dynamic extension loaded");
