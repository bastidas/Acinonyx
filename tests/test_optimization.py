"""
test_optimization.py - Unit tests for linkage trajectory optimization.

Tests cover:
  1. Known solution tests (inverse problem) using achievable_target tools
  2. Real mechanism tests - use saved 4-bar from demo/test_graphs/
  3. Convergence history verification
  4. Error metrics computation
  5. Dimension extraction and application
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimization_helpers import (
    apply_dimensions,
    apply_dimensions_from_array,
    dict_to_dimensions,
    dimensions_to_dict,
    extract_dimensions,
    validate_bounds,
)
from pylink_tools.optimization_types import (
    ConvergenceStats,
    DimensionSpec,
    OptimizationResult,
    TargetTrajectory,
)
from pylink_tools.optimize import (
    analyze_convergence,
    create_fitness_function,
    evaluate_linkage_fit,
    format_convergence_report,
    log_optimization_progress,
    run_pso_optimization,
    run_scipy_optimization,
)
from pylink_tools.trajectory_utils import compute_trajectory_error
from target_gen.achievable_target import create_achievable_target
from target_gen.variation_config import AchievableTargetConfig
from target_gen.variation_config import DimensionVariationConfig
# from pylink_tools.optimize import optimize_trajectory


@pytest.fixture
def fourbar_data():
    """Load 4-bar from demo/test_graphs/4bar.json (hypergraph format)."""
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        data = json.load(f)
    data['n_steps'] = 24
    return data


class TestDimensionExtraction:
    """Tests for extract_dimensions and related functions."""

    def test_extract_dimensions_4bar(self, fourbar_data):
        """Extract dimensions from 4-bar hypergraph."""
        spec = extract_dimensions(fourbar_data)

        # Hypergraph: dimensions come from edges (ground, crank_link, coupler, rocker)
        assert len(spec) >= 3
        assert any('distance' in name for name in spec.names)

    def test_extract_dimensions_bounds(self, fourbar_data):
        """Bounds should be computed from initial values."""
        spec = extract_dimensions(fourbar_data, bounds_factor=2.0)

        # Check that bounds exist and are valid
        for i, name in enumerate(spec.names):
            lower, upper = spec.bounds[i]
            assert lower < upper
            assert lower > 0

    def test_extract_dimensions_custom_bounds(self, fourbar_data):
        """Custom bounds override default computation."""
        spec = extract_dimensions(fourbar_data)
        if spec.names:
            first_dim = spec.names[0]
            custom = {first_dim: (0.5, 500.0)}
            spec_custom = extract_dimensions(fourbar_data, custom_bounds=custom)
            idx = spec_custom.names.index(first_dim)
            assert spec_custom.bounds[idx] == (0.5, 500.0)

    def test_bounds_tuple_format(self, fourbar_data):
        """get_bounds_tuple returns pylinkage format."""
        spec = extract_dimensions(fourbar_data)
        lower, upper = spec.get_bounds_tuple()

        assert len(lower) == len(spec)
        assert len(upper) == len(spec)
        assert all(l < u for l, u in zip(lower, upper))


# =============================================================================
# Test: Dimension Application
# =============================================================================

class TestDimensionApplication:
    """Tests for apply_dimensions and related functions."""

    def test_apply_dimensions_updates_edges(self, fourbar_data):
        """apply_dimensions updates edge distances in hypergraph."""
        spec = extract_dimensions(fourbar_data)
        if not spec.names:
            pytest.skip('No dimensions to test')

        first_dim = spec.names[0]
        new_values = {first_dim: 999.0}

        updated = apply_dimensions(fourbar_data, new_values, spec)

        # Original unchanged
        orig_edges = fourbar_data['linkage']['edges']
        # Updated has new value
        new_edges = updated['linkage']['edges']
        assert orig_edges != new_edges or len(spec) == 0

    def test_apply_dimensions_from_array(self, fourbar_data):
        """apply_dimensions_from_array uses spec order."""
        spec = extract_dimensions(fourbar_data)
        if not spec.names:
            pytest.skip('No dimensions to test')

        # Create new values (all slightly different)
        values = tuple(v * 1.1 for v in spec.initial_values)
        updated = apply_dimensions_from_array(fourbar_data, values, spec)

        # Verify update happened
        spec2 = extract_dimensions(updated)
        for i in range(len(spec.names)):
            assert spec2.initial_values[i] == pytest.approx(values[i], rel=0.01)


# =============================================================================
# Test: Error Computation
# =============================================================================

class TestErrorComputation:
    """Tests for trajectory error computation."""

    def test_compute_error_perfect_match(self):
        """Perfect match gives zero error."""
        positions = [(1, 2), (3, 4), (5, 6)]
        target = TargetTrajectory(joint_name='test', positions=positions)

        error = compute_trajectory_error(positions, target, metric='mse')
        assert error == pytest.approx(0.0, abs=1e-10)

    def test_compute_error_shifted(self):
        """Shifted positions give expected error."""
        target_pos = [(0, 0), (1, 0), (2, 0)]
        computed = [(1, 0), (2, 0), (3, 0)]  # Shifted by 1 in x
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        # Each point is distance 1 away, MSE = 1^2 = 1
        error = compute_trajectory_error(computed, target, metric='mse')
        assert error == pytest.approx(1.0, abs=0.01)

    def test_compute_error_diagonal_shift(self):
        """Diagonal shift gives correct error."""
        target_pos = [(0, 0)]
        computed = [(3, 4)]  # Distance = 5
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        # MSE = 5^2 = 25
        error = compute_trajectory_error(computed, target, metric='mse')
        assert error == pytest.approx(25.0, abs=0.01)

        # RMSE = 5
        rmse = compute_trajectory_error(computed, target, metric='rmse')
        assert rmse == pytest.approx(5.0, abs=0.01)

    def test_error_metrics_max(self):
        """compute_trajectory_error returns max error metric."""
        target_pos = [(0, 0), (0, 0), (0, 0)]
        computed = [(1, 0), (2, 0), (3, 0)]  # Distances: 1, 2, 3
        target = TargetTrajectory(joint_name='test', positions=target_pos)

        max_error = compute_trajectory_error(computed, target, metric='max', phase_invariant=False)
        assert max_error == pytest.approx(3.0, abs=0.01)


# =============================================================================
# Test: Known Solution (Inverse Problem) using achievable_target tools
# =============================================================================

class TestKnownSolution:
    """
    Tests using create_achievable_target to generate targets with KNOWN solutions.

    The key insight: we perturb a valid mechanism to create a target, then verify
    the optimizer can recover the perturbed dimensions (or get close).
    """

    def test_perfect_match_stays_optimal(self, fourbar_data):
        """When target matches current trajectory, optimizer keeps dimensions."""
        result = compute_trajectory(fourbar_data, verbose=False)
        if not result.success:
            pytest.skip(f'Could not compute trajectory: {result.error}')

        target_joint = 'coupler_rocker_joint'
        trajectory = result.trajectories[target_joint]
        target = TargetTrajectory(joint_name=target_joint, positions=trajectory)

        # Error should already be ~0
        metrics = evaluate_linkage_fit(fourbar_data, target)
        assert metrics.rmse < 0.01, 'Initial fit should be perfect'

        # Optimization should maintain ~0 error
        opt_result = run_scipy_optimization(
            fourbar_data, target,
            max_iterations=20, verbose=False,
        )

        assert opt_result.success
        assert opt_result.final_error < 0.01

    def test_achievable_target_scipy(self, fourbar_data):
        """Use create_achievable_target to get known solution, verify scipy recovers."""
        dim_spec = extract_dimensions(fourbar_data)
        target_joint = 'coupler_rocker_joint'

        # Create achievable target with ±20% variation (conservative for testing)
        config = AchievableTargetConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=0.20,
                exclude_dimensions=['ground_distance'],  # Don't vary ground
            ),
            random_seed=42,
            max_attempts=64,
        )

        target_result = create_achievable_target(
            fourbar_data, target_joint, dim_spec, config=config,
        )

        # We now have a target with KNOWN achievable dimensions
        target = target_result.target
        known_dims = target_result.target_dimensions

        # Compute initial error (original linkage vs perturbed target)
        initial_metrics = evaluate_linkage_fit(fourbar_data, target)
        assert initial_metrics.mse > 0, 'Target should differ from original'

        # Run scipy optimization
        opt_result = run_scipy_optimization(
            fourbar_data, target,
            max_iterations=150, verbose=False,
        )

        assert opt_result.success
        # Optimizer should reduce error (may not fully recover due to local minima)
        assert opt_result.final_error < initial_metrics.mse

    def test_achievable_target_pso(self, fourbar_data):
        """Use create_achievable_target to get known solution, verify PSO recovers."""
        dim_spec = extract_dimensions(fourbar_data)
        target_joint = 'coupler_rocker_joint'

        # Create achievable target with smaller variation for PSO
        config = AchievableTargetConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=0.15,
                exclude_dimensions=['ground_distance'],
            ),
            random_seed=123,
            max_attempts=64,
        )

        target_result = create_achievable_target(
            fourbar_data, target_joint, dim_spec, config=config,
        )

        target = target_result.target

        # Compute initial error
        initial_metrics = evaluate_linkage_fit(fourbar_data, target)

        # Run PSO optimization
        opt_result = run_pso_optimization(
            fourbar_data, target,
            n_particles=20, iterations=30, verbose=False,
        )

        assert opt_result.success
        # PSO should also reduce error
        assert opt_result.final_error < initial_metrics.mse


# =============================================================================
# Test: Convergence History
# =============================================================================

class TestConvergenceHistory:
    """Tests for convergence history tracking."""

    def test_scipy_tracks_history(self, fourbar_data):
        """scipy optimization tracks convergence history."""
        result = compute_trajectory(fourbar_data, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        opt_result = run_scipy_optimization(
            fourbar_data, target,
            max_iterations=30, verbose=False,
        )

        assert opt_result.convergence_history is not None
        assert len(opt_result.convergence_history) >= 1

    def test_pso_tracks_history(self, fourbar_data):
        """PSO optimization tracks convergence history."""
        result = compute_trajectory(fourbar_data, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        opt_result = run_pso_optimization(
            fourbar_data, target,
            n_particles=10, iterations=15, verbose=False,
        )

        assert opt_result.convergence_history is not None
        # PSO history has iterations + 1 entries (initial + each iteration)
        assert len(opt_result.convergence_history) == 16


# =============================================================================
# Test: Fitness Function
# =============================================================================

class TestFitnessFunction:
    """Tests for fitness function creation."""

    def test_fitness_function_callable(self, fourbar_data):
        """create_fitness_function returns callable."""
        spec = extract_dimensions(fourbar_data)
        result = compute_trajectory(fourbar_data, verbose=False)
        if not result.success:
            pytest.skip('Could not compute trajectory')

        target = TargetTrajectory(
            joint_name='coupler_rocker_joint',
            positions=result.trajectories['coupler_rocker_joint'],
        )

        fitness = create_fitness_function(fourbar_data, target, spec)

        assert callable(fitness)

        # Call with initial values
        initial_values = tuple(spec.initial_values)
        error = fitness(initial_values)

        assert isinstance(error, float)
        assert error >= 0 or error == float('inf')


# =============================================================================
# Test: Utility Functions
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""

    def test_dimensions_to_dict(self):
        """dimensions_to_dict converts array to named dict."""
        spec = DimensionSpec(
            names=['a', 'b', 'c'],
            initial_values=[1.0, 2.0, 3.0],
            bounds=[(0, 10), (0, 10), (0, 10)],
            joint_mapping={},
        )

        result = dimensions_to_dict((5.0, 6.0, 7.0), spec)

        assert result == {'a': 5.0, 'b': 6.0, 'c': 7.0}

    def test_dict_to_dimensions(self):
        """dict_to_dimensions converts named dict to array."""
        spec = DimensionSpec(
            names=['a', 'b', 'c'],
            initial_values=[1.0, 2.0, 3.0],
            bounds=[(0, 10), (0, 10), (0, 10)],
            joint_mapping={},
        )

        result = dict_to_dimensions({'a': 5.0, 'b': 6.0, 'c': 7.0}, spec)

        assert result == (5.0, 6.0, 7.0)

    def test_validate_bounds_valid(self):
        """validate_bounds returns empty for valid values."""
        spec = DimensionSpec(
            names=['a', 'b'],
            initial_values=[1.0, 2.0],
            bounds=[(0, 10), (0, 10)],
            joint_mapping={},
        )

        violations = validate_bounds((5.0, 5.0), spec)

        assert violations == []

    def test_validate_bounds_violations(self):
        """validate_bounds reports violations."""
        spec = DimensionSpec(
            names=['a', 'b'],
            initial_values=[1.0, 2.0],
            bounds=[(0, 10), (0, 10)],
            joint_mapping={},
        )

        violations = validate_bounds((-1.0, 15.0), spec)

        assert len(violations) == 2


# =============================================================================
# Test: OptimizationResult
# =============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_to_dict(self):
        """OptimizationResult.to_dict() serializes correctly."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={'a': 1.0, 'b': 2.0},
            optimized_pylink_data={'test': 'data'},
            initial_error=10.0,
            final_error=1.0,
            iterations=50,
            convergence_history=[10.0, 5.0, 1.0],
        )

        d = result.to_dict()

        assert d['success'] is True
        assert d['optimized_dimensions'] == {'a': 1.0, 'b': 2.0}
        assert d['initial_error'] == 10.0
        assert d['final_error'] == 1.0


# =============================================================================
# Test: TargetTrajectory
# =============================================================================

class TestTargetTrajectory:
    """Tests for TargetTrajectory dataclass."""

    def test_target_trajectory_basic(self):
        """Basic TargetTrajectory creation."""
        target = TargetTrajectory(
            joint_name='test',
            positions=[(1, 2), (3, 4), (5, 6)],
        )

        assert target.joint_name == 'test'
        assert target.n_steps == 3
        assert target.weights == [1.0, 1.0, 1.0]

    def test_target_trajectory_from_dict(self):
        """TargetTrajectory.from_dict() deserializes correctly."""
        data = {
            'joint_name': 'coupler',
            'positions': [[1, 2], [3, 4]],
            'weights': [1.0, 2.0],
        }

        target = TargetTrajectory.from_dict(data)

        assert target.joint_name == 'coupler'
        assert target.n_steps == 2


# =============================================================================
# Test: Convergence Logging Utilities
# =============================================================================

class TestConvergenceLogging:
    """Tests for convergence analysis and logging utilities."""

    def test_analyze_convergence_basic(self):
        """analyze_convergence computes basic stats correctly."""
        history = [10.0, 8.0, 5.0, 3.0, 2.0, 1.0]

        stats = analyze_convergence(history)

        assert stats.initial_error == 10.0
        assert stats.final_error == 1.0
        assert stats.best_error == 1.0
        assert stats.improvement_pct == pytest.approx(90.0, rel=0.01)

    def test_analyze_convergence_empty(self):
        """analyze_convergence handles empty history."""
        stats = analyze_convergence([])

        assert stats.n_iterations == 0
        assert stats.converged is False

    def test_format_convergence_report(self):
        """format_convergence_report produces readable output."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={'a': 1.5, 'b': 2.5},
            initial_error=10.0,
            final_error=1.0,
            iterations=50,
            convergence_history=[10.0, 5.0, 1.0],
        )

        report = format_convergence_report(result)

        assert 'SUCCESS' in report
        assert 'Initial Error: 10' in report

    def test_log_optimization_progress(self):
        """log_optimization_progress formats iteration info."""
        progress = log_optimization_progress(
            iteration=42,
            current_error=2.5,
            best_error=1.0,
            dimensions=(1.5, 2.5, 3.5),
            dimension_names=['a', 'b', 'c'],
        )

        assert '[  42]' in progress
        assert 'error=2.5' in progress

    def test_convergence_stats_to_dict(self):
        """ConvergenceStats.to_dict() serializes correctly."""
        stats = ConvergenceStats(
            initial_error=10.0,
            final_error=1.0,
            best_error=1.0,
            improvement_pct=90.0,
            n_iterations=50,
            n_evaluations=51,
            converged=True,
            history=[10.0, 5.0, 1.0],
            improvement_per_iteration=[5.0, 4.0],
        )

        d = stats.to_dict()

        assert d['initial_error'] == 10.0
        assert d['converged'] is True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
