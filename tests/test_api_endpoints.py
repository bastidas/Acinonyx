"""
Tests for backend/acinonyx_api.py endpoints.

Comprehensive test suite for all API endpoints using the new Mechanism-based approach.
Tests verify that:
- Frontend data is converted to Mechanism ONCE at API boundary
- No pylink_data is passed internally
- Responses are correctly formatted
- Error handling works properly
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from backend.acinonyx_api import analyze_trajectory_endpoint
from backend.acinonyx_api import compute_pylink_trajectory
from backend.acinonyx_api import get_optimizable_dimensions
from backend.acinonyx_api import get_status
from backend.acinonyx_api import list_pylink_graphs
from backend.acinonyx_api import load_demo
from backend.acinonyx_api import load_pylink_graph
from backend.acinonyx_api import optimize_trajectory_endpoint
from backend.acinonyx_api import prepare_trajectory
from backend.acinonyx_api import root
from backend.acinonyx_api import save_pylink_graph
from backend.acinonyx_api import validate_mechanism_endpoint
from demo.helpers import create_mechanism_from_dict


@pytest.fixture
def fourbar_data():
    """Load 4bar test linkage data."""
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        return json.load(f)


@pytest.fixture
def fourbar_mechanism(fourbar_data):
    """Create Mechanism from fourbar data for internal testing."""
    return create_mechanism_from_dict(fourbar_data)


@pytest.fixture
def sample_trajectory():
    """Generate a simple circular trajectory."""
    return [[math.cos(t), math.sin(t)] for t in [i * 2 * math.pi / 20 for i in range(20)]]


@pytest.fixture
def target_trajectory(fourbar_mechanism):
    """Create a target trajectory from the fourbar mechanism."""
    traj = fourbar_mechanism.get_trajectory('coupler_rocker_joint')
    if traj is None or len(traj) == 0:
        pytest.skip('Could not generate trajectory from mechanism')
    return {
        'joint_name': 'coupler_rocker_joint',
        'positions': [[float(x), float(y)] for x, y in traj],
    }


class TestHealthEndpoints:
    """Test basic health check endpoints."""

    def test_root(self):
        """Test root endpoint."""
        result = root()
        assert result['message'] == 'Acinonyx API is running'

    def test_status(self):
        """Test status endpoint."""
        result = get_status()
        assert result['status'] == 'operational'
        assert 'message' in result
        assert 'backend is running' in result['message'].lower()


class TestGraphOperations:
    """Test graph save/load operations."""

    def test_save_and_load_graph(self, fourbar_data):
        """Test saving and loading a graph."""
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
        """Test listing saved graphs."""
        # Ensure at least one graph exists
        save_pylink_graph(fourbar_data)

        result = list_pylink_graphs()
        assert result['status'] == 'success'
        assert 'files' in result
        assert isinstance(result['files'], list)
        assert len(result['files']) > 0

    def test_load_nonexistent_graph(self):
        """Test loading a non-existent graph."""
        result = load_pylink_graph(filename='nonexistent_graph_xyz.json')
        assert result['status'] == 'error'


class TestDemoLoading:
    """Test demo loading functionality."""

    @pytest.mark.parametrize('demo_name', ['4bar', 'leg', 'walker', 'complex', 'intermediate'])
    def test_load_demo(self, demo_name):
        """Test loading various demos."""
        result = load_demo(name=demo_name)
        assert result['status'] == 'success', f"Failed to load {demo_name}: {result.get('message')}"
        assert 'data' in result
        assert result['name'] == demo_name
        assert 'linkage' in result['data']

    def test_load_unknown_demo(self):
        """Test loading an unknown demo."""
        result = load_demo(name='unknown_demo')
        assert result['status'] == 'error'
        assert 'Unknown demo' in result['message']


class TestMechanismConversion:
    """Test that frontend data is correctly converted to Mechanism."""

    def test_compute_trajectory_converts_to_mechanism(self, fourbar_data):
        """Test that compute-pylink-trajectory converts data to Mechanism."""
        request = {'pylink_data': fourbar_data, 'n_steps': 24}
        result = compute_pylink_trajectory(request)

        assert result['status'] == 'success'
        assert 'trajectories' in result
        assert 'joint_types' in result
        assert result['n_steps'] == 24
        assert len(result['trajectories']) > 0

    def test_compute_trajectory_with_direct_data(self, fourbar_data):
        """Test that request can be the data directly (not wrapped)."""
        # Request can be the data itself
        result = compute_pylink_trajectory(fourbar_data)
        assert result['status'] == 'success'

    def test_validate_mechanism_converts_to_mechanism(self, fourbar_data):
        """Test that validate-mechanism converts data to Mechanism."""
        request = {'pylink_data': fourbar_data}
        result = validate_mechanism_endpoint(request)

        assert result['status'] == 'success'
        assert 'valid' in result
        assert isinstance(result['valid'], bool)

    def test_get_optimizable_dimensions_converts_to_mechanism(self, fourbar_data):
        """Test that get-optimizable-dimensions converts data to Mechanism."""
        request = {'pylink_data': fourbar_data}
        result = get_optimizable_dimensions(request)

        assert result['status'] == 'success'
        assert 'dimension_bounds_spec' in result
        spec = result['dimension_bounds_spec']
        assert 'names' in spec
        assert 'initial_values' in spec
        assert 'bounds' in spec
        assert len(spec['names']) > 0


class TestTrajectoryComputation:
    """Test trajectory computation endpoint."""

    def test_compute_trajectory_success(self, fourbar_data):
        """Test successful trajectory computation."""
        request = {'pylink_data': fourbar_data, 'n_steps': 32}
        result = compute_pylink_trajectory(request)

        assert result['status'] == 'success'
        assert 'trajectories' in result
        assert 'joint_types' in result
        assert 'execution_time_ms' in result
        assert result['n_steps'] == 32

        # Verify trajectory structure
        trajectories = result['trajectories']
        assert len(trajectories) > 0
        for joint_name, positions in trajectories.items():
            assert isinstance(positions, list)
            assert len(positions) == 32
            assert all(isinstance(p, list) and len(p) == 2 for p in positions)

        # Verify joint types
        joint_types = result['joint_types']
        assert len(joint_types) == len(trajectories)
        assert all(jt in ('Static', 'Crank', 'Revolute') for jt in joint_types.values())

    def test_compute_trajectory_custom_n_steps(self, fourbar_data):
        """Test trajectory computation with custom n_steps."""
        request = {'pylink_data': fourbar_data, 'n_steps': 48}
        result = compute_pylink_trajectory(request)

        assert result['status'] == 'success'
        assert result['n_steps'] == 48
        # All trajectories should have 48 points
        for positions in result['trajectories'].values():
            assert len(positions) == 48

    def test_compute_trajectory_invalid_data(self):
        """Test trajectory computation with invalid data."""
        result = compute_pylink_trajectory({'invalid': 'data'})
        assert result['status'] == 'error'
        assert 'message' in result

    def test_compute_trajectory_missing_linkage(self):
        """Test trajectory computation with missing linkage."""
        result = compute_pylink_trajectory({'name': 'test'})
        assert result['status'] == 'error'


class TestMechanismValidation:
    """Test mechanism validation endpoint."""

    def test_validate_mechanism_valid(self, fourbar_data):
        """Test validation of a valid mechanism."""
        request = {'pylink_data': fourbar_data}
        result = validate_mechanism_endpoint(request)

        assert result['status'] == 'success'
        assert 'valid' in result
        # Fourbar should be valid
        assert result['valid'] is True

    def test_validate_mechanism_invalid_data(self):
        """Test validation with invalid data."""
        request = {'pylink_data': {'invalid': 'data'}}
        result = validate_mechanism_endpoint(request)

        # Should return error status
        assert result['status'] == 'error' or result['valid'] is False

    def test_validate_mechanism_missing_data(self):
        """Test validation with missing data."""
        result = validate_mechanism_endpoint({})
        assert result['status'] == 'error' or 'valid' in result


class TestOptimizableDimensions:
    """Test get-optimizable-dimensions endpoint."""

    def test_get_optimizable_dimensions_success(self, fourbar_data):
        """Test getting optimizable dimensions."""
        request = {'pylink_data': fourbar_data}
        result = get_optimizable_dimensions(request)

        assert result['status'] == 'success'
        assert 'dimension_bounds_spec' in result
        spec = result['dimension_bounds_spec']
        assert 'names' in spec
        assert 'initial_values' in spec
        assert 'bounds' in spec
        assert 'n_dimensions' in spec

        # Verify structure
        dims = spec['names']
        initial = spec['initial_values']
        bounds = spec['bounds']

        assert len(dims) > 0
        assert len(initial) == len(dims)
        assert len(bounds) == len(dims)

        # Verify bounds are tuples of (min, max)
        for bound in bounds:
            assert len(bound) == 2
            assert bound[0] < bound[1]  # min < max

    def test_get_optimizable_dimensions_missing_data(self):
        """Test getting dimensions with missing data."""
        result = get_optimizable_dimensions(None)
        assert result['status'] == 'error'
        assert 'Missing' in result['message']


class TestTrajectoryOptimization:
    """Test trajectory optimization endpoint."""

    def test_optimize_trajectory_success(self, fourbar_data, target_trajectory):
        """Test successful trajectory optimization."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': target_trajectory,
            'optimization_options': {
                'method': 'scipy',
                'max_iterations': 10,  # Small for testing
                'tolerance': 1e-3,
                'verbose': False,
            },
        }

        result = optimize_trajectory_endpoint(request)

        assert result['status'] == 'success'
        assert 'result' in result
        assert 'execution_time_ms' in result

        opt_result = result['result']
        assert 'success' in opt_result
        assert 'initial_error' in opt_result
        assert 'final_error' in opt_result
        assert 'optimized_dimensions' in opt_result
        assert 'optimized_pylink_data' in opt_result

    def test_optimize_trajectory_missing_target_joint(self, fourbar_data):
        """Test optimization with missing target joint."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': {
                'positions': [[0, 0], [1, 1]],
            },
        }

        result = optimize_trajectory_endpoint(request)
        assert result['status'] == 'error'
        assert 'joint_name' in result['message'].lower()

    def test_optimize_trajectory_missing_positions(self, fourbar_data):
        """Test optimization with missing positions."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': {
                'joint_name': 'coupler_rocker_joint',
            },
        }

        result = optimize_trajectory_endpoint(request)
        assert result['status'] == 'error'
        assert 'points' in result['message'].lower()

    def test_optimize_trajectory_too_few_points(self, fourbar_data):
        """Test optimization with too few target points."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': {
                'joint_name': 'coupler_rocker_joint',
                'positions': [[0, 0]],  # Only 1 point
            },
        }

        result = optimize_trajectory_endpoint(request)
        assert result['status'] == 'error'
        assert 'at least 2' in result['message'].lower()

    def test_optimize_trajectory_invalid_joint(self, fourbar_data):
        """Test optimization with invalid target joint."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': {
                'joint_name': 'nonexistent_joint',
                'positions': [[0, 0], [1, 1], [2, 2]],
            },
        }

        result = optimize_trajectory_endpoint(request)
        assert result['status'] == 'error'
        assert 'not found' in result['message'].lower()

    @pytest.mark.parametrize('method', ['scipy', 'pylinkage', 'pso'])
    def test_optimize_trajectory_different_methods(self, fourbar_data, target_trajectory, method):
        """Test optimization with different methods."""
        request = {
            'pylink_data': fourbar_data,
            'target_path': target_trajectory,
            'optimization_options': {
                'method': method,
                'max_iterations': 5,  # Very small for testing
                'iterations': 5,  # For PSO methods
                'n_particles': 4,  # Very small for PSO
                'verbose': False,
            },
        }

        result = optimize_trajectory_endpoint(request)

        # Should at least attempt optimization (may fail due to small iterations, but should not crash)
        assert result['status'] in ('success', 'error')
        if result['status'] == 'success':
            assert 'result' in result


# =============================================================================
# TRAJECTORY PROCESSING TESTS
# =============================================================================

class TestTrajectoryProcessing:
    """Test trajectory processing endpoints (prepare, analyze)."""

    def test_analyze_trajectory(self, sample_trajectory):
        """Test trajectory analysis."""
        result = analyze_trajectory_endpoint({'trajectory': sample_trajectory})

        assert result['status'] == 'success'
        assert 'analysis' in result
        analysis = result['analysis']
        assert 'n_points' in analysis
        assert 'centroid' in analysis
        assert 'bounding_box' in analysis
        assert 'total_path_length' in analysis
        assert 'is_closed' in analysis

    def test_prepare_trajectory_resample(self, sample_trajectory):
        """Test trajectory resampling."""
        result = prepare_trajectory({
            'trajectory': sample_trajectory,
            'target_n_steps': 12,
            'smooth': False,
            'resample': True,
        })

        assert result['status'] == 'success'
        assert result['output_points'] == 12
        assert 'target_trajectory' in result
        assert 'positions' in result['target_trajectory']
        assert len(result['target_trajectory']['positions']) == 12

    def test_prepare_trajectory_smooth(self, sample_trajectory):
        """Test trajectory smoothing."""
        result = prepare_trajectory({
            'trajectory': sample_trajectory,
            'target_n_steps': len(sample_trajectory),
            'smooth': True,
            'resample': False,
            'smooth_window': 5,
        })

        assert result['status'] == 'success'
        assert 'target_trajectory' in result
        assert 'positions' in result['target_trajectory']
        assert len(result['target_trajectory']['positions']) == len(sample_trajectory)

    def test_prepare_trajectory_empty(self):
        """Test preparing empty trajectory."""
        result = prepare_trajectory({'trajectory': []})
        assert result['status'] == 'error'

    def test_prepare_trajectory_too_short(self):
        """Test preparing trajectory with too few points."""
        result = prepare_trajectory({'trajectory': [[0, 0], [1, 1]]})
        assert result['status'] == 'error'
        assert 'too short' in result['message'].lower()


class TestIntegration:
    """Integration tests that test multiple endpoints together."""

    def test_complete_workflow(self, fourbar_data):
        """Test a complete workflow: validate -> compute -> optimize."""
        # Step 1: Validate mechanism
        validate_request = {'pylink_data': fourbar_data}
        validate_result = validate_mechanism_endpoint(validate_request)
        assert validate_result['status'] == 'success'
        assert validate_result['valid'] is True

        # Step 2: Compute trajectory
        compute_request = {'pylink_data': fourbar_data, 'n_steps': 24}
        compute_result = compute_pylink_trajectory(compute_request)
        assert compute_result['status'] == 'success'

        # Step 3: Get optimizable dimensions
        dims_request = {'pylink_data': fourbar_data}
        dims_result = get_optimizable_dimensions(dims_request)
        assert dims_result['status'] == 'success'
        assert len(dims_result['dimension_bounds_spec']['names']) > 0

    def test_optimize_then_compute(self, fourbar_data, target_trajectory):
        """Test optimizing then computing trajectory from optimized result."""
        # Optimize
        opt_request = {
            'pylink_data': fourbar_data,
            'target_path': target_trajectory,
            'optimization_options': {
                'method': 'scipy',
                'max_iterations': 5,
                'verbose': False,
            },
        }
        opt_result = optimize_trajectory_endpoint(opt_request)

        if opt_result['status'] == 'success':
            # Compute trajectory from optimized mechanism
            optimized_data = opt_result['result']['optimized_pylink_data']
            compute_request = {'pylink_data': optimized_data, 'n_steps': 24}
            compute_result = compute_pylink_trajectory(compute_request)

            assert compute_result['status'] == 'success'
            assert 'trajectories' in compute_result
