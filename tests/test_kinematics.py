"""Tests for pylink_tools/kinematic.py - trajectory computation and validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylink_tools.kinematic import compute_trajectory
from pylink_tools.kinematic import find_connected_link_groups
from pylink_tools.kinematic import is_hypergraph_format
from pylink_tools.kinematic import validate_mechanism


@pytest.fixture
def fourbar_data():
    """Load 4-bar linkage from test file."""
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        return json.load(f)


class TestFormatDetection:
    def test_hypergraph_format(self, fourbar_data):
        assert is_hypergraph_format(fourbar_data)

    def test_empty_data(self):
        assert not is_hypergraph_format({})


class TestTrajectoryComputation:
    def test_hypergraph_4bar(self, fourbar_data):
        fourbar_data['n_steps'] = 12
        result = compute_trajectory(fourbar_data)
        assert result.success, f'Failed: {result.error}'
        assert result.n_steps == 12
        assert len(result.trajectories) == 4  # 4 joints

    def test_invalid_format(self):
        result = compute_trajectory({'invalid': 'data'})
        assert not result.success
        assert 'Unknown data format' in result.error


class TestLinkGroups:
    def test_connected_group(self, fourbar_data):
        linkage = fourbar_data['linkage']
        groups = find_connected_link_groups(linkage['edges'], linkage['nodes'])
        assert len(groups) == 1
        assert groups[0].is_valid
        assert len(groups[0].links) == 4

    def test_disconnected_groups(self):
        nodes = {
            'A': {'id': 'A', 'role': 'fixed'},
            'B': {'id': 'B', 'role': 'crank'},
            'X': {'id': 'X', 'role': 'fixed'},
            'Y': {'id': 'Y', 'role': 'fixed'},
        }
        edges = {
            'link1': {'source': 'A', 'target': 'B'},
            'link2': {'source': 'X', 'target': 'Y'},
        }
        groups = find_connected_link_groups(edges, nodes)
        assert len(groups) == 2
        assert not any(g.is_valid for g in groups)  # Both invalid (< 3 edges)


class TestValidation:
    def test_valid_4bar(self, fourbar_data):
        result = validate_mechanism(fourbar_data)
        assert result['valid'], f"Validation failed: {result.get('errors')}"

    def test_empty_linkage(self):
        result = validate_mechanism({'linkage': {'nodes': {}, 'edges': {}}})
        assert not result['valid']
        assert 'No nodes defined' in result['errors']
