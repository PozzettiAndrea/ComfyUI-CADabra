"""
CPU-only tests for ComfyUI-CADabra nodes
"""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestMLStubs:
    """Test ML stub nodes (CPU-only)"""

    def test_surface_recon_stub(self):
        """Test ML_SurfaceRecon stub returns expected structure"""
        from nodes import ML_SurfaceRecon

        node = ML_SurfaceRecon()
        mock_mesh = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            "faces": np.array([[0, 1, 2]]),
            "type": "2D"
        }

        result = node.reconstruct_surface(mock_mesh, "hustvl/surface-recon", 0.5)
        surface_data = result[0]

        assert "vertices" in surface_data
        assert "faces" in surface_data
        assert "model_used" in surface_data
        assert surface_data["model_used"] == "hustvl/surface-recon"
        assert surface_data["status"] == "stub_implementation"

    def test_feature_detection_stub(self):
        """Test ML_FeatureDetection stub returns expected structure"""
        from nodes import ML_FeatureDetection

        node = ML_FeatureDetection()
        mock_mesh = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            "faces": np.array([[0, 1, 2]]),
            "type": "2D"
        }

        result = node.detect_features(mock_mesh, "autodesk/feature-detection-cad", 0.7)
        features_data = result[0]

        assert "features" in features_data
        assert "model_used" in features_data
        assert features_data["model_used"] == "autodesk/feature-detection-cad"
        assert features_data["status"] == "stub_implementation"
        assert isinstance(features_data["features"], list)


class TestNodeStructure:
    """Test node structure and interfaces"""

    def test_node_class_mappings(self):
        """Test all required nodes are registered"""
        from nodes import NODE_CLASS_MAPPINGS

        required_nodes = [
            "CAD_Load_Gmsh",
            "CAD_Mesh_Gmsh",
            "ML_SurfaceRecon",
            "ML_FeatureDetection",
            "CAD_Viewer"
        ]

        for node_name in required_nodes:
            assert node_name in NODE_CLASS_MAPPINGS

    def test_input_types_structure(self):
        """Test nodes have proper INPUT_TYPES"""
        from nodes import NODE_CLASS_MAPPINGS

        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            assert hasattr(node_class, "INPUT_TYPES")
            input_types = node_class.INPUT_TYPES()
            assert isinstance(input_types, dict)
            assert "required" in input_types or "optional" in input_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
