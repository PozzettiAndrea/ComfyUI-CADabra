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
        from nodes import NODE_CLASS_MAPPINGS

        ML_SurfaceRecon = NODE_CLASS_MAPPINGS["ML_SurfaceRecon"]
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
        from nodes import NODE_CLASS_MAPPINGS

        ML_FeatureDetection = NODE_CLASS_MAPPINGS["ML_FeatureDetection"]
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
            "CAD_Mesh_Gmsh_Advanced",
            "Mesh_Optimize_Gmsh",
            "ML_SurfaceRecon",
            "ML_FeatureDetection",
            "PreviewCADOCC"
        ]

        for node_name in required_nodes:
            assert node_name in NODE_CLASS_MAPPINGS, f"Missing node: {node_name}"

    def test_input_types_structure(self):
        """Test nodes have proper INPUT_TYPES"""
        from nodes import NODE_CLASS_MAPPINGS

        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            assert hasattr(node_class, "INPUT_TYPES")
            input_types = node_class.INPUT_TYPES()
            assert isinstance(input_types, dict)
            assert "required" in input_types or "optional" in input_types


class TestAdvancedMeshNodes:
    """Test advanced mesh generation and optimization nodes"""

    def test_advanced_mesh_input_types(self):
        """Test CAD_Mesh_Gmsh_Advanced has expected parameters"""
        from nodes import NODE_CLASS_MAPPINGS

        CAD_Mesh_Gmsh_Advanced = NODE_CLASS_MAPPINGS["CAD_Mesh_Gmsh_Advanced"]
        input_types = CAD_Mesh_Gmsh_Advanced.INPUT_TYPES()

        # Check required parameters
        assert "required" in input_types
        assert "cad_model" in input_types["required"]
        assert "mesh_type" in input_types["required"]
        assert "element_size" in input_types["required"]

        # Check optional advanced parameters
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "min_element_size" in optional
        assert "max_element_size" in optional
        assert "use_curvature_sizing" in optional
        assert "curvature_divisions" in optional
        assert "algorithm_2d" in optional
        assert "algorithm_3d" in optional
        assert "element_order" in optional
        assert "subdivision_algorithm" in optional
        assert "gmsh_options" in optional

    def test_advanced_mesh_algorithms(self):
        """Test CAD_Mesh_Gmsh_Advanced has extended algorithms"""
        from nodes import NODE_CLASS_MAPPINGS

        CAD_Mesh_Gmsh_Advanced = NODE_CLASS_MAPPINGS["CAD_Mesh_Gmsh_Advanced"]
        input_types = CAD_Mesh_Gmsh_Advanced.INPUT_TYPES()
        optional = input_types["optional"]

        # Check 2D algorithms include extended options
        algo_2d = optional["algorithm_2d"][0]
        assert "MeshAdapt" in algo_2d
        assert "BAMG" in algo_2d
        assert "DelQuad" in algo_2d

        # Check 3D algorithms include extended options
        algo_3d = optional["algorithm_3d"][0]
        assert "MMG3D" in algo_3d
        assert "HXT" in algo_3d

    def test_advanced_mesh_element_orders(self):
        """Test CAD_Mesh_Gmsh_Advanced supports higher-order elements"""
        from nodes import NODE_CLASS_MAPPINGS

        CAD_Mesh_Gmsh_Advanced = NODE_CLASS_MAPPINGS["CAD_Mesh_Gmsh_Advanced"]
        input_types = CAD_Mesh_Gmsh_Advanced.INPUT_TYPES()
        element_orders = input_types["optional"]["element_order"][0]

        assert "1" in element_orders
        assert "2" in element_orders
        assert "3" in element_orders
        assert "4" in element_orders
        assert "5" in element_orders

    def test_mesh_optimize_input_types(self):
        """Test Mesh_Optimize_Gmsh has expected parameters"""
        from nodes import NODE_CLASS_MAPPINGS

        Mesh_Optimize_Gmsh = NODE_CLASS_MAPPINGS["Mesh_Optimize_Gmsh"]
        input_types = Mesh_Optimize_Gmsh.INPUT_TYPES()

        # Check required parameters
        assert "required" in input_types
        assert "mesh" in input_types["required"]

        # Check optional optimization parameters
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "optimize" in optional
        assert "optimize_netgen" in optional
        assert "optimize_ho" in optional
        assert "smooth_steps" in optional
        assert "recombine" in optional
        assert "recombine_algorithm" in optional
        assert "recombine_angle" in optional
        assert "gmsh_options" in optional

    def test_mesh_optimize_recombine_algorithms(self):
        """Test Mesh_Optimize_Gmsh has recombination algorithms"""
        from nodes import NODE_CLASS_MAPPINGS

        Mesh_Optimize_Gmsh = NODE_CLASS_MAPPINGS["Mesh_Optimize_Gmsh"]
        input_types = Mesh_Optimize_Gmsh.INPUT_TYPES()
        recombine_algos = input_types["optional"]["recombine_algorithm"][0]

        assert "Simple" in recombine_algos
        assert "Blossom" in recombine_algos
        assert "SimpleFull" in recombine_algos
        assert "BlossomFull" in recombine_algos

    def test_mesh_optimize_functional(self):
        """Test Mesh_Optimize_Gmsh processes mesh data"""
        from nodes import NODE_CLASS_MAPPINGS

        Mesh_Optimize_Gmsh = NODE_CLASS_MAPPINGS["Mesh_Optimize_Gmsh"]
        node = Mesh_Optimize_Gmsh()
        mock_mesh = {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            "faces": np.array([[0, 1, 2]]),
            "type": "2D",
            "element_size": 1.0
        }

        # Test with optimization enabled
        result = node.optimize_mesh(
            mesh=mock_mesh,
            optimize=True,
            recombine=True,
            recombine_algorithm="Blossom"
        )
        optimized_mesh = result[0]

        # Check structure is preserved
        assert "vertices" in optimized_mesh
        assert "faces" in optimized_mesh
        assert "type" in optimized_mesh
        assert optimized_mesh["type"] == "2D"

        # Check metadata is added
        assert "optimizations_applied" in optimized_mesh
        optimizations = optimized_mesh["optimizations_applied"]
        assert optimizations["optimize"] is True
        assert optimizations["recombine"] is True
        assert optimizations["recombine_algorithm"] == "Blossom"

    def test_advanced_mesh_return_types(self):
        """Test nodes have correct return types"""
        from nodes import NODE_CLASS_MAPPINGS

        CAD_Mesh_Gmsh_Advanced = NODE_CLASS_MAPPINGS["CAD_Mesh_Gmsh_Advanced"]
        Mesh_Optimize_Gmsh = NODE_CLASS_MAPPINGS["Mesh_Optimize_Gmsh"]

        # Check advanced mesh node
        assert hasattr(CAD_Mesh_Gmsh_Advanced, "RETURN_TYPES")
        assert CAD_Mesh_Gmsh_Advanced.RETURN_TYPES == ("MESH",)

        # Check optimize node
        assert hasattr(Mesh_Optimize_Gmsh, "RETURN_TYPES")
        assert Mesh_Optimize_Gmsh.RETURN_TYPES == ("MESH",)

    def test_advanced_mesh_category(self):
        """Test nodes are in correct category"""
        from nodes import NODE_CLASS_MAPPINGS

        CAD_Mesh_Gmsh_Advanced = NODE_CLASS_MAPPINGS["CAD_Mesh_Gmsh_Advanced"]
        Mesh_Optimize_Gmsh = NODE_CLASS_MAPPINGS["Mesh_Optimize_Gmsh"]

        assert CAD_Mesh_Gmsh_Advanced.CATEGORY == "CADabra/Advanced"
        assert Mesh_Optimize_Gmsh.CATEGORY == "CADabra/Advanced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
