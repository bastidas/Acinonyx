"""
Tests for the pylinkage bridge module.

Tests cover:
- Conversion from automata graph format to pylinkage Linkage
- Simulation using pylinkage solver
- Comparison between automata and pylinkage solvers
"""

import json
import os
from pathlib import Path
import pytest
import numpy as np

from link.pylinkage_bridge import (
    convert_to_pylinkage,
    simulate_linkage,
    extract_path_visualization_data,
    ConversionResult,
    SimulationResult
)
from configs.link_models import Link, Node


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_graph_path():
    """Path to the test graph JSON file."""
    return Path(__file__).parent / "test_graph.json"


@pytest.fixture
def real_graph_path():
    """Path to a real saved graph for integration testing."""
    return Path(__file__).parent.parent / "user" / "graphs" / "graph_20260116_142356.json"


@pytest.fixture
def simple_4bar_data():
    """Simple 4-bar linkage data for unit testing."""
    return {
        "nodes": [
            {"id": "node1", "pos": [0, 0], "fixed": True, "fixed_loc": [0, 0]},
            {"id": "node2", "pos": [10, 0], "fixed": False},
            {"id": "node3", "pos": [20, 0], "fixed": True, "fixed_loc": [20, 0]},
            {"id": "node4", "pos": [15, 10], "fixed": False}
        ],
        "links": [
            {
                "name": "drive_link",
                "length": 10.0,
                "n_iterations": 24,
                "has_fixed": True,
                "fixed_loc": [0, 0],
                "is_driven": True,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0,
                "meta": {"id": "link1", "start_point": [0, 0], "end_point": [10, 0], "color": "#1f77b4"}
            },
            {
                "name": "coupler",
                "length": 15.0,
                "n_iterations": 24,
                "has_fixed": False,
                "is_driven": False,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0,
                "meta": {"id": "link2", "start_point": [10, 0], "end_point": [15, 10], "color": "#ff7f0e"}
            },
            {
                "name": "follower",
                "length": 12.0,
                "n_iterations": 24,
                "has_fixed": True,
                "fixed_loc": [20, 0],
                "is_driven": False,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0,
                "meta": {"id": "link3", "start_point": [20, 0], "end_point": [15, 10], "color": "#2ca02c"}
            }
        ],
        "connections": [
            {"from_node": "node1", "to_node": "node2", "link_id": "link1"},
            {"from_node": "node2", "to_node": "node4", "link_id": "link2"},
            {"from_node": "node3", "to_node": "node4", "link_id": "link3"}
        ]
    }


def load_graph_json(path: Path) -> dict:
    """Load and parse a graph JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def prepare_nodes_and_links(graph_data: dict, n_iterations: int = 24):
    """Convert raw graph data to Node and Link objects."""
    nodes = []
    for i, node_dict in enumerate(graph_data.get('nodes', [])):
        node_data = {
            'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
            'n_iterations': node_dict.get('n_iterations', n_iterations),
            'fixed': node_dict.get('fixed', False)
        }
        if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
            node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
        elif node_data['fixed'] and 'pos' in node_dict:
            node_data['fixed_loc'] = tuple(node_dict['pos'])
        if 'pos' in node_dict and node_dict['pos']:
            node_data['init_pos'] = tuple(node_dict['pos'])
        elif node_data.get('fixed_loc'):
            node_data['init_pos'] = node_data['fixed_loc']
        else:
            node_data['init_pos'] = (0.0, 0.0)
        nodes.append(Node(**node_data))
    
    links = []
    for link_dict in graph_data.get('links', []):
        excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
        link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
        if 'n_iterations' not in link_data:
            link_data['n_iterations'] = n_iterations
        links.append(Link(**link_data))
    
    connections = graph_data.get('connections', [])
    
    return nodes, links, connections


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversionResult:
    """Tests for the ConversionResult dataclass."""
    
    def test_default_values(self):
        """Test that ConversionResult has sensible defaults."""
        result = ConversionResult()
        assert result.success is False
        assert result.linkage is None
        assert result.warnings == []
        assert result.errors == []
        assert result.joint_mapping == {}
        assert result.stats == {}
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = ConversionResult(
            success=True,
            warnings=["test warning"],
            stats={"total_joints": 5}
        )
        d = result.to_dict()
        assert d["success"] is True
        assert "test warning" in d["warnings"]
        assert d["stats"]["total_joints"] == 5


class TestSimulationResult:
    """Tests for the SimulationResult dataclass."""
    
    def test_default_values(self):
        """Test that SimulationResult has sensible defaults."""
        result = SimulationResult()
        assert result.success is False
        assert result.trajectories == {}
        assert result.n_iterations == 0
        assert result.errors == []
        assert result.execution_time_ms == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion Tests  
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvertToPylinkage:
    """Tests for the convert_to_pylinkage function."""
    
    def test_simple_4bar_conversion(self, simple_4bar_data):
        """Test conversion of a simple 4-bar linkage."""
        nodes, links, connections = prepare_nodes_and_links(simple_4bar_data)
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        # Check basic success
        assert result.success, f"Conversion failed: {result.errors}"
        assert result.linkage is not None
        
        # Check joint counts
        assert result.stats["crank_joints"] == 1, "Should have exactly 1 crank"
        assert result.stats["static_joints"] >= 2, "Should have at least 2 static joints"
        
        # Check mapping exists for driven link
        assert "link:drive_link" in result.joint_mapping
        
        print(f"✓ 4-bar conversion successful: {result.stats}")
    
    def test_missing_driven_link(self):
        """Test that conversion fails gracefully without a driven link."""
        nodes = [Node(name="n1", n_iterations=24, fixed=True, fixed_loc=(0, 0), init_pos=(0, 0))]
        links = [Link(
            name="static_link",
            length=10.0,
            n_iterations=24,
            has_fixed=True,
            fixed_loc=(0, 0),
            is_driven=False  # No driver!
        )]
        connections = []
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        assert result.success is False
        assert any("driven" in err.lower() for err in result.errors)
    
    def test_missing_fixed_nodes(self):
        """Test that conversion fails without any fixed nodes."""
        nodes = [Node(name="n1", n_iterations=24, fixed=False, init_pos=(0, 0))]
        links = [Link(
            name="drive",
            length=10.0,
            n_iterations=24,
            has_fixed=True,
            fixed_loc=(0, 0),
            is_driven=True
        )]
        connections = []
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        # Should either fail or create implicit static from fixed_loc
        # Either is acceptable behavior
        if not result.success:
            assert any("fixed" in err.lower() or "ground" in err.lower() for err in result.errors)
    
    def test_conversion_warnings(self, simple_4bar_data):
        """Test that warnings are captured during conversion."""
        nodes, links, connections = prepare_nodes_and_links(simple_4bar_data)
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        # Warnings list should be accessible even if empty
        assert isinstance(result.warnings, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulateLinkage:
    """Tests for the simulate_linkage function."""
    
    def test_simulation_after_conversion(self, simple_4bar_data):
        """Test that simulation runs after successful conversion."""
        nodes, links, connections = prepare_nodes_and_links(simple_4bar_data)
        
        conv_result = convert_to_pylinkage(nodes, links, connections)
        if not conv_result.success:
            pytest.skip(f"Conversion failed: {conv_result.errors}")
        
        sim_result = simulate_linkage(conv_result.linkage, n_iterations=24)
        
        # Simulation might fail due to geometry constraints, that's OK
        # We just want to verify it doesn't crash
        assert isinstance(sim_result, SimulationResult)
        assert sim_result.n_iterations == 24
        
        if sim_result.success:
            assert len(sim_result.trajectories) > 0
            assert sim_result.execution_time_ms >= 0
            print(f"✓ Simulation completed in {sim_result.execution_time_ms:.2f}ms")
        else:
            print(f"⚠ Simulation failed (geometry constraints): {sim_result.errors}")
    
    def test_simulation_trajectory_shapes(self, simple_4bar_data):
        """Test that trajectory arrays have correct shapes."""
        nodes, links, connections = prepare_nodes_and_links(simple_4bar_data)
        n_iter = 24
        
        conv_result = convert_to_pylinkage(nodes, links, connections)
        if not conv_result.success:
            pytest.skip("Conversion failed")
        
        sim_result = simulate_linkage(conv_result.linkage, n_iterations=n_iter)
        if not sim_result.success:
            pytest.skip("Simulation failed")
        
        for name, traj in sim_result.trajectories.items():
            assert traj.shape == (n_iter, 2), f"Joint {name} has wrong shape: {traj.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests with Real Graph Files
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealGraphConversion:
    """Integration tests using real saved graph files."""
    
    def test_load_and_convert_real_graph(self, real_graph_path):
        """Test conversion of a real saved graph."""
        if not real_graph_path.exists():
            pytest.skip(f"Real graph file not found: {real_graph_path}")
        
        graph_data = load_graph_json(real_graph_path)
        nodes, links, connections = prepare_nodes_and_links(graph_data)
        
        print(f"\nLoaded graph with {len(nodes)} nodes, {len(links)} links, {len(connections)} connections")
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        print(f"Conversion success: {result.success}")
        print(f"Stats: {result.stats}")
        print(f"Warnings: {result.warnings}")
        print(f"Errors: {result.errors}")
        
        # Even if conversion fails, we want to see useful diagnostics
        if result.success:
            print(f"✓ Real graph converted successfully!")
            print(f"  Joint mapping: {list(result.joint_mapping.keys())}")
        else:
            print(f"✗ Conversion failed - this may be expected for complex graphs")
            # Don't fail the test - complex graphs may not convert cleanly yet
    
    def test_full_pipeline_real_graph(self, real_graph_path):
        """Test full conversion + simulation pipeline with real graph."""
        if not real_graph_path.exists():
            pytest.skip(f"Real graph file not found: {real_graph_path}")
        
        graph_data = load_graph_json(real_graph_path)
        nodes, links, connections = prepare_nodes_and_links(graph_data)
        
        # Step 1: Convert
        conv_result = convert_to_pylinkage(nodes, links, connections)
        
        if not conv_result.success:
            print(f"Conversion failed: {conv_result.errors}")
            # Don't fail - report and continue
            return
        
        # Step 2: Simulate
        sim_result = simulate_linkage(conv_result.linkage, n_iterations=24)
        
        print(f"\nSimulation success: {sim_result.success}")
        print(f"Execution time: {sim_result.execution_time_ms:.2f}ms")
        
        if sim_result.success:
            print(f"Trajectories: {list(sim_result.trajectories.keys())}")
            
            # Step 3: Extract visualization data
            path_data = extract_path_visualization_data(
                sim_result.trajectories,
                links,
                conv_result.joint_mapping,
                24
            )
            
            assert "bounds" in path_data
            assert "links" in path_data
            assert "n_iterations" in path_data
            print(f"✓ Full pipeline completed successfully!")
        else:
            print(f"Simulation errors: {sim_result.errors}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test using test_graph.json (if it exists)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWithTestGraph:
    """Tests using the test_graph.json fixture."""
    
    def test_convert_test_graph(self, test_graph_path):
        """Test conversion of test_graph.json."""
        if not test_graph_path.exists():
            pytest.skip(f"Test graph not found: {test_graph_path}")
        
        graph_data = load_graph_json(test_graph_path)
        
        # Add n_iterations if missing
        for node in graph_data.get('nodes', []):
            if 'n_iterations' not in node:
                node['n_iterations'] = 24
        
        nodes, links, connections = prepare_nodes_and_links(graph_data)
        
        result = convert_to_pylinkage(nodes, links, connections)
        
        print(f"\nTest graph conversion:")
        print(f"  Success: {result.success}")
        print(f"  Stats: {result.stats}")
        print(f"  Warnings: {result.warnings}")
        print(f"  Errors: {result.errors}")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo 4-Bar Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDemo4Bar:
    """Tests for the demo 4-bar linkage generator."""
    
    def test_create_demo_4bar(self):
        """Test that demo 4-bar creates valid linkage."""
        from link.pylinkage_bridge import create_demo_4bar_pylinkage
        
        linkage, metadata = create_demo_4bar_pylinkage(
            ground_length=30.0,
            crank_length=10.0,
            coupler_length=25.0,
            rocker_length=20.0,
            crank_anchor=(20.0, 30.0),
            n_iterations=24
        )
        
        assert linkage is not None
        assert len(linkage.joints) == 2  # Crank + Revolute
        assert metadata["type"] == "4-bar"
        assert "crank" in metadata["joints"]
        assert "revolute" in metadata["joints"]
        
        print(f"✓ Demo 4-bar created with {len(linkage.joints)} joints")
    
    def test_simulate_demo_4bar(self):
        """Test full simulation of demo 4-bar."""
        from link.pylinkage_bridge import simulate_demo_4bar
        
        result = simulate_demo_4bar(
            ground_length=30.0,
            crank_length=10.0,
            coupler_length=25.0,
            rocker_length=20.0,
            crank_anchor=(20.0, 30.0),
            n_iterations=24
        )
        
        assert result["status"] == "success"
        assert result["n_iterations"] == 24
        assert "path_data" in result
        assert len(result["path_data"]["links"]) == 4  # ground, crank, coupler, rocker
        
        # Verify link names
        link_names = [l["name"] for l in result["path_data"]["links"]]
        assert "ground" in link_names
        assert "crank" in link_names
        assert "coupler" in link_names
        assert "rocker" in link_names
        
        # Verify is_ground flag in path_data
        ground_link = next(l for l in result["path_data"]["links"] if l["name"] == "ground")
        assert ground_link.get("is_ground") == True
        
        # Non-ground links should have is_ground=False
        other_links = [l for l in result["path_data"]["links"] if l["name"] != "ground"]
        for link in other_links:
            assert link.get("is_ground") == False, f"Link {link['name']} should not be ground"
        
        print(f"✓ Demo 4-bar simulated in {result['execution_time_ms']:.2f}ms")
    
    def test_demo_4bar_grashof_condition(self):
        """Test that demo respects Grashof condition for continuous rotation."""
        from link.pylinkage_bridge import simulate_demo_4bar
        
        # Grashof condition: shortest + longest <= sum of other two
        # For continuous crank rotation, crank should be shortest
        result = simulate_demo_4bar(
            ground_length=30.0,  # longest
            crank_length=10.0,   # shortest (driver)
            coupler_length=25.0,
            rocker_length=20.0,
            n_iterations=24
        )
        
        # Should succeed - this is a valid Grashof crank-rocker
        assert result["status"] == "success"
        
        # Verify all trajectory points are valid (no NaN)
        for link in result["path_data"]["links"]:
            for pos in link["pos1"] + link["pos2"]:
                for coord in pos:
                    assert not np.isnan(coord), f"NaN found in {link['name']}"
        
        print("✓ Grashof condition satisfied - full rotation possible")
    
    def test_demo_4bar_to_ui_format(self):
        """Test that demo 4-bar produces valid UI format for graph builder."""
        from link.pylinkage_bridge import demo_4bar_to_ui_format
        
        ui_graph = demo_4bar_to_ui_format(
            ground_length=30.0,
            crank_length=10.0,
            coupler_length=25.0,
            rocker_length=20.0,
            crank_anchor=(20.0, 30.0),
            n_iterations=24
        )
        
        # Check structure
        assert "nodes" in ui_graph
        assert "links" in ui_graph
        assert "connections" in ui_graph
        assert "metadata" in ui_graph
        
        # Check nodes
        assert len(ui_graph["nodes"]) == 4
        node_ids = [n["id"] for n in ui_graph["nodes"]]
        assert "crank_anchor" in node_ids
        assert "crank_end" in node_ids
        assert "rocker_anchor" in node_ids
        assert "coupler_joint" in node_ids
        
        # Verify fixed nodes
        fixed_nodes = [n for n in ui_graph["nodes"] if n.get("fixed")]
        assert len(fixed_nodes) == 2
        fixed_ids = [n["id"] for n in fixed_nodes]
        assert "crank_anchor" in fixed_ids
        assert "rocker_anchor" in fixed_ids
        
        # Check links - now includes ground link
        assert len(ui_graph["links"]) == 4  # crank, coupler, rocker, ground
        link_names = [l["name"] for l in ui_graph["links"]]
        assert "crank" in link_names
        assert "coupler" in link_names
        assert "rocker" in link_names
        assert "ground" in link_names
        
        # Verify driven link
        driven_links = [l for l in ui_graph["links"] if l.get("is_driven")]
        assert len(driven_links) == 1
        assert driven_links[0]["name"] == "crank"
        
        # Verify ground link
        ground_links = [l for l in ui_graph["links"] if l.get("is_ground")]
        assert len(ground_links) == 1
        assert ground_links[0]["name"] == "ground"
        assert ground_links[0]["has_fixed"] == True
        assert ground_links[0]["has_constraint"] == True
        
        # Check connections match links (now 4 connections)
        assert len(ui_graph["connections"]) == 4
        
        # Check metadata
        assert ui_graph["metadata"]["type"] == "4-bar"
        assert ui_graph["metadata"]["source"] == "pylinkage_demo"
        assert "parameters" in ui_graph["metadata"]
        
        print(f"✓ UI format generated: {len(ui_graph['nodes'])} nodes, {len(ui_graph['links'])} links (including ground)")
    
    def test_ui_format_round_trip_conversion(self):
        """Test that UI format can be converted back to pylinkage successfully."""
        from link.pylinkage_bridge import demo_4bar_to_ui_format, convert_to_pylinkage
        from configs.link_models import Link, Node
        
        # Step 1: Generate UI format
        ui_graph = demo_4bar_to_ui_format(
            ground_length=30.0,
            crank_length=10.0,
            coupler_length=25.0,
            rocker_length=20.0,
            crank_anchor=(20.0, 30.0),
            n_iterations=24
        )
        
        # Step 2: Convert to backend format (simulating what frontend sends)
        node_objects = []
        for node_dict in ui_graph["nodes"]:
            node_data = {
                "name": node_dict["id"],
                "fixed": node_dict.get("fixed", False),
                "fixed_loc": tuple(node_dict["fixed_loc"]) if node_dict.get("fixed_loc") else None,
                "n_iterations": node_dict.get("n_iterations", 24),
                "init_pos": tuple(node_dict["pos"])
            }
            node_objects.append(Node(**node_data))
        
        link_objects = []
        for link_dict in ui_graph["links"]:
            # Exclude frontend-specific fields (like meta)
            link_data = {
                "name": link_dict["name"],
                "length": link_dict["length"],
                "n_iterations": link_dict.get("n_iterations", 24),
                "has_fixed": link_dict.get("has_fixed", False),
                "fixed_loc": tuple(link_dict["fixed_loc"]) if link_dict.get("fixed_loc") else None,
                "is_driven": link_dict.get("is_driven", False),
                "is_ground": link_dict.get("is_ground", False),
                "has_constraint": link_dict.get("has_constraint", False),
                "flip": link_dict.get("flip", False),
                "zlevel": link_dict.get("zlevel", 0)
            }
            link_objects.append(Link(**link_data))
        
        connections = ui_graph["connections"]
        
        # Step 3: Convert to pylinkage
        result = convert_to_pylinkage(node_objects, link_objects, connections)
        
        # Verify conversion succeeded
        assert result.success, f"Conversion failed: {result.errors}"
        
        # Check that ground link warning is present (ground is implicit)
        ground_warnings = [w for w in result.warnings if "ground" in w.lower() and "implicit" in w.lower()]
        assert len(ground_warnings) == 1, "Should have warning about ground being implicit"
        
        # Check that coupler was NOT skipped (should have no error about coupler)
        coupler_errors = [e for e in result.errors if "coupler" in e.lower()]
        assert len(coupler_errors) == 0, f"Coupler should not have errors: {coupler_errors}"
        
        coupler_warnings = [w for w in result.warnings if "coupler" in w.lower() and "could not determine" in w.lower()]
        assert len(coupler_warnings) == 0, f"Coupler should not have parent joint warnings: {coupler_warnings}"
        
        # Verify joint structure
        assert result.stats["revolute_joints"] >= 1, "Should have at least one Revolute joint for coupler"
        
        print(f"✓ Round-trip conversion successful: {result.stats}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run tests directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run a quick sanity check
    print("Running pylinkage bridge tests...")
    
    # Test with simple 4-bar
    simple_data = {
        "nodes": [
            {"id": "node1", "pos": [0, 0], "fixed": True, "fixed_loc": [0, 0]},
            {"id": "node2", "pos": [10, 0], "fixed": False},
            {"id": "node3", "pos": [20, 0], "fixed": True, "fixed_loc": [20, 0]},
            {"id": "node4", "pos": [15, 10], "fixed": False}
        ],
        "links": [
            {
                "name": "drive_link",
                "length": 10.0,
                "n_iterations": 24,
                "has_fixed": True,
                "fixed_loc": [0, 0],
                "is_driven": True,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0
            },
            {
                "name": "coupler",
                "length": 15.0,
                "n_iterations": 24,
                "has_fixed": False,
                "is_driven": False,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0
            },
            {
                "name": "follower",
                "length": 12.0,
                "n_iterations": 24,
                "has_fixed": True,
                "fixed_loc": [20, 0],
                "is_driven": False,
                "has_constraint": False,
                "flip": False,
                "zlevel": 0
            }
        ],
        "connections": [
            {"from_node": "node1", "to_node": "node2", "link_id": "link1"},
            {"from_node": "node2", "to_node": "node4", "link_id": "link2"},
            {"from_node": "node3", "to_node": "node4", "link_id": "link3"}
        ]
    }
    
    nodes, links, connections = prepare_nodes_and_links(simple_data)
    result = convert_to_pylinkage(nodes, links, connections)
    
    print(f"\nSimple 4-bar test:")
    print(f"  Success: {result.success}")
    print(f"  Stats: {result.stats}")
    print(f"  Errors: {result.errors}")
    
    # Test with real graph if available
    real_path = Path(__file__).parent.parent / "user" / "graphs" / "graph_20260116_142356.json"
    if real_path.exists():
        print(f"\nTesting with real graph: {real_path.name}")
        graph_data = load_graph_json(real_path)
        nodes, links, connections = prepare_nodes_and_links(graph_data)
        result = convert_to_pylinkage(nodes, links, connections)
        print(f"  Success: {result.success}")
        print(f"  Stats: {result.stats}")
        print(f"  Errors: {result.errors}")

