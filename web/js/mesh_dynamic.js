/**
 * ComfyUI-CADabra Dynamic Mesh Node Widget Management
 *
 * This extension manages the visibility of mesh-related widgets
 * based on user selections in the CAD_Mesh_Gmsh_Advanced node.
 *
 * Conditional visibility:
 * - algorithm_3d: Only shown when output_dim = "3D Volume"
 * - elements_per_2pi, extend_from_boundary: Only shown when size_mode = "Curvature Adaptive"
 * - optimization_passes: Only shown when optimize = true
 */

import { app } from "../../../scripts/app.js";

const DEBUG = true;  // Enable debugging

function log(...args) {
    if (DEBUG) {
        console.log("[CADabra-Mesh]", ...args);
    }
}

console.log("[CADabra-Mesh] Loading mesh_dynamic.js extension...");

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

// Main extension registration
app.registerExtension({
    name: "cadabra.mesh.dynamic",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match the CAD_Mesh_Gmsh_Advanced node
        if (nodeData.name === "CAD_Mesh_Gmsh_Advanced") {
            log("beforeRegisterNodeDef: Found CAD_Mesh_Gmsh_Advanced");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                log("onNodeCreated: Setting up CAD_Mesh_Gmsh_Advanced node");

                // Use setTimeout to ensure widgets are fully initialized
                const node = this;
                setTimeout(() => {
                    setupMeshNode(node);
                }, 100);

                return r;
            };
        }
    }
});

function setupMeshNode(node) {
    // Find all relevant widgets
    const outputDimWidget = node.widgets?.find(w => w.name === "output_dim");
    const algorithm3dWidget = node.widgets?.find(w => w.name === "algorithm_3d");
    const sizeModeWidget = node.widgets?.find(w => w.name === "size_mode");
    const targetSizeWidget = node.widgets?.find(w => w.name === "target_size");
    const sizeFactorWidget = node.widgets?.find(w => w.name === "size_factor");
    const minSizeFactorWidget = node.widgets?.find(w => w.name === "min_size_factor");
    const maxSizeFactorWidget = node.widgets?.find(w => w.name === "max_size_factor");
    const elementsPer2piWidget = node.widgets?.find(w => w.name === "elements_per_2pi");
    const extendFromBoundaryWidget = node.widgets?.find(w => w.name === "extend_from_boundary");
    const optimizeWidget = node.widgets?.find(w => w.name === "optimize");
    const optimizationPassesWidget = node.widgets?.find(w => w.name === "optimization_passes");

    log("Found widgets:", {
        output_dim: !!outputDimWidget,
        algorithm_3d: !!algorithm3dWidget,
        size_mode: !!sizeModeWidget,
        target_size: !!targetSizeWidget,
        size_factor: !!sizeFactorWidget,
        min_size_factor: !!minSizeFactorWidget,
        max_size_factor: !!maxSizeFactorWidget,
        elements_per_2pi: !!elementsPer2piWidget,
        extend_from_boundary: !!extendFromBoundaryWidget,
        optimize: !!optimizeWidget,
        optimization_passes: !!optimizationPassesWidget
    });

    // Log all widget names for debugging
    log("All widget names:", node.widgets?.map(w => w.name));

    /**
     * Update visibility of algorithm_3d based on output dimension.
     * Only show 3D algorithm when generating 3D volume meshes.
     * Special handling: insert right after algorithm_2d for logical grouping.
     * Also triggers updateExtendFromBoundary since it depends on output_dim.
     */
    const updateForOutputDim = (outputDim) => {
        log("Updating for output_dim:", outputDim);

        if (outputDim === "3D Volume") {
            // Show algorithm_3d right after algorithm_2d
            if (algorithm3dWidget && algorithm3dWidget._hidden) {
                // Find algorithm_2d index and insert right after it
                const algo2dIndex = node.widgets.indexOf(
                    node.widgets.find(w => w.name === "algorithm_2d")
                );
                if (algo2dIndex !== -1) {
                    algorithm3dWidget._originalIndex = algo2dIndex + 1;
                }
            }
            showWidget(node, algorithm3dWidget);
        } else {
            hideWidget(node, algorithm3dWidget);
        }

        // Update extend_from_boundary based on combined conditions
        updateExtendFromBoundary();

        refreshNodeSize(node);
    };

    /**
     * Update visibility of extend_from_boundary based on BOTH output_dim AND size_mode.
     * Only show when: output_dim = "3D Volume" AND size_mode = "Curvature Adaptive"
     */
    const updateExtendFromBoundary = () => {
        const outputDim = outputDimWidget?.value;
        const sizeMode = sizeModeWidget?.value;

        log("Updating extend_from_boundary visibility:", { outputDim, sizeMode });

        // Only show when BOTH conditions are met
        if (outputDim === "3D Volume" && sizeMode === "Curvature Adaptive") {
            showWidget(node, extendFromBoundaryWidget);
        } else {
            hideWidget(node, extendFromBoundaryWidget);
        }
    };

    /**
     * Update visibility of size-related widgets based on size mode.
     * - Curvature Adaptive: show elements_per_2pi; hide target_size, size factors
     * - Relative/Absolute: show target_size, size factors; hide curvature settings
     * Note: extend_from_boundary is handled separately by updateExtendFromBoundary()
     */
    const updateForSizeMode = (sizeMode) => {
        log("Updating for size_mode:", sizeMode);

        if (sizeMode === "Curvature Adaptive") {
            // Curvature mode: size is determined by geometry, not user input
            hideWidget(node, targetSizeWidget);
            hideWidget(node, minSizeFactorWidget);
            hideWidget(node, maxSizeFactorWidget);
            // Show curvature-specific settings
            showWidget(node, elementsPer2piWidget);
            // extend_from_boundary handled by updateExtendFromBoundary()
        } else {
            // Relative or Absolute mode: user specifies sizes
            showWidget(node, targetSizeWidget);
            showWidget(node, minSizeFactorWidget);
            showWidget(node, maxSizeFactorWidget);
            // Hide curvature-specific settings
            hideWidget(node, elementsPer2piWidget);
            // extend_from_boundary handled by updateExtendFromBoundary()
        }

        // Update extend_from_boundary based on combined conditions
        updateExtendFromBoundary();

        refreshNodeSize(node);
    };

    /**
     * Update visibility of optimization passes based on optimize checkbox.
     * Only show optimization_passes when optimize is enabled.
     */
    const updateForOptimize = (optimize) => {
        log("Updating for optimize:", optimize);

        if (optimize) {
            showWidget(node, optimizationPassesWidget);
        } else {
            hideWidget(node, optimizationPassesWidget);
        }

        refreshNodeSize(node);
    };

    // Wire up callback for output_dim
    if (outputDimWidget) {
        const origCallback = outputDimWidget.callback;
        outputDimWidget.callback = function(value) {
            log("output_dim changed to:", value);
            const result = origCallback?.apply(this, arguments);
            updateForOutputDim(value);
            return result;
        };
        // Initialize visibility
        updateForOutputDim(outputDimWidget.value);
    }

    // Wire up callback for size_mode
    if (sizeModeWidget) {
        const origCallback = sizeModeWidget.callback;
        sizeModeWidget.callback = function(value) {
            log("size_mode changed to:", value);
            const result = origCallback?.apply(this, arguments);
            updateForSizeMode(value);
            return result;
        };
        // Initialize visibility
        updateForSizeMode(sizeModeWidget.value);
    }

    // Wire up callback for optimize
    if (optimizeWidget) {
        const origCallback = optimizeWidget.callback;
        optimizeWidget.callback = function(value) {
            log("optimize changed to:", value);
            const result = origCallback?.apply(this, arguments);
            updateForOptimize(value);
            return result;
        };
        // Initialize visibility
        updateForOptimize(optimizeWidget.value);
    }

    log("CAD_Mesh_Gmsh_Advanced setup complete");
}

log("CADabra mesh dynamic extension loaded");
