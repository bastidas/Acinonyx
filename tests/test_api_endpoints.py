"""
Tests for backend/acinonyx_api.py endpoints.

Tests core API functionality: health checks, graph operations, trajectory computation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.acinonyx_api import analyze_trajectory_endpoint
from backend.acinonyx_api import compute_pylink_trajectory
from backend.acinonyx_api import get_status
from backend.acinonyx_api import list_pylink_graphs
from backend.acinonyx_api import load_demo
from backend.acinonyx_api import load_pylink_graph
from backend.acinonyx_api import prepare_trajectory
from backend.acinonyx_api import root
from backend.acinonyx_api import save_pylink_graph
from backend.acinonyx_api import validate_mechanism_endpoint


@pytest.fixture
def fourbar_data():
    """Load 4bar test linkage data."""
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        return json.load(f)


class TestHealthEndpoints:
    def test_root(self):
        result = root()
        assert result['message'] == 'Acinonyx API is running'

    def test_status(self):
        result = get_status()
        assert result['status'] == 'operational'
        assert 'message' in result


class TestGraphOperations:
    def test_save_and_load_graph(self, fourbar_data):
        # Save
        save_result = save_pylink_graph(fourbar_data)
        assert save_result['status'] == 'success'
        assert 'filename' in save_result
        filename = save_result['filename']

        # Load back
        load_result = load_pylink_graph(filename=filename)
        assert load_result['status'] == 'success'
        assert 'data' in load_result

        # Verify hypergraph structure
        loaded = load_result['data']
        assert 'linkage' in loaded
        assert 'nodes' in loaded['linkage']
        assert 'edges' in loaded['linkage']
        assert len(loaded['linkage']['nodes']) == len(fourbar_data['linkage']['nodes'])

    def test_list_graphs(self, fourbar_data):
        # Ensure at least one graph exists
        save_pylink_graph(fourbar_data)

        result = list_pylink_graphs()
        assert result['status'] == 'success'
        assert 'files' in result
        assert isinstance(result['files'], list)
        assert len(result['files']) > 0

    def test_load_nonexistent_graph(self):
        result = load_pylink_graph(filename='nonexistent_graph_xyz.json')
        assert result['status'] == 'error'


class TestDemoLoading:
    @pytest.mark.parametrize('demo_name', ['4bar', 'leg', 'walker', 'complex', 'intermediate'])
    def test_load_demo(self, demo_name):
        result = load_demo(name=demo_name)
        assert result['status'] == 'success', f"Failed to load {demo_name}: {result.get('message')}"
        assert 'data' in result
        assert result['name'] == demo_name

    def test_load_unknown_demo(self):
        result = load_demo(name='unknown_demo')
        assert result['status'] == 'error'
        assert 'Unknown demo' in result['message']


class TestTrajectoryComputation:
    def test_compute_trajectory(self, fourbar_data):
        result = compute_pylink_trajectory(fourbar_data)
        assert result['status'] == 'success', f"Failed: {result.get('message')}"
        assert 'trajectories' in result
        assert 'joint_types' in result
        assert result['n_steps'] > 0
        assert len(result['trajectories']) > 0

    def test_compute_trajectory_invalid_data(self):
        result = compute_pylink_trajectory({'invalid': 'data'})
        assert result['status'] == 'error'

    def test_validate_mechanism(self, fourbar_data):
        result = validate_mechanism_endpoint(fourbar_data)
        assert result['status'] == 'success'
        assert 'valid' in result


class TestTrajectoryProcessing:
    @pytest.fixture
    def sample_trajectory(self):
        """Generate a simple circular trajectory."""
        import math
        return [[math.cos(t), math.sin(t)] for t in [i * 2 * math.pi / 20 for i in range(20)]]

    def test_analyze_trajectory(self, sample_trajectory):
        result = analyze_trajectory_endpoint({'trajectory': sample_trajectory})
        assert result['status'] == 'success'
        assert 'analysis' in result
        analysis = result['analysis']
        assert 'n_points' in analysis
        assert 'centroid' in analysis
        assert 'total_path_length' in analysis

    def test_prepare_trajectory(self, sample_trajectory):
        result = prepare_trajectory({
            'trajectory': sample_trajectory,
            'target_n_steps': 12,
            'smooth': True,
            'resample': True,
        })
        assert result['status'] == 'success'
        assert result['output_points'] == 12
        assert 'trajectory' in result
        assert len(result['trajectory']) == 12

    def test_prepare_trajectory_empty(self):
        result = prepare_trajectory({'trajectory': []})
        assert result['status'] == 'error'

    def test_prepare_trajectory_too_short(self):
        result = prepare_trajectory({'trajectory': [[0, 0], [1, 1]]})
        assert result['status'] == 'error'
        assert 'too short' in result['message']
