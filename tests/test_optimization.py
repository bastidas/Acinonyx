"""
test_optimization.py - Unit tests for linkage trajectory optimization.

Tests cover:
  1. Known solution tests (inverse problem) using achievable_target tools
  2. Real mechanism tests - use saved 4-bar from demo/test_graphs/
  3. Convergence history verification
  4. Error metrics computation
  5. Dimension extraction and application

All tests use Mechanism objects (no pylink_data passing).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.helpers import create_mechanism_from_dict
from optimizers.pylinkage_pso import _constraint_order_and_linkage_dims
from optimizers.scipy_optimizer import run_scipy_optimization
from pylink_tools.mechanism import create_mechanism_fitness
from pylink_tools.optimization_types import OptimizationResult
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.optimization_types import TopologyVariationSpec
from pylink_tools.optimize import optimize_trajectory
from pylink_tools.trajectory_scoring import score_trajectory
from target_gen.achievable_target import create_achievable_target
from target_gen.variation_config import DimensionVariationConfig
from target_gen.variation_config import MechVariationConfig


@pytest.fixture
def fourbar_data():
    """Load 4-bar from demo/test_graphs/4bar.json (hypergraph format)."""
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        data = json.load(f)
    data['n_steps'] = 24
    return data


@pytest.fixture
def fourbar_mechanism(fourbar_data):
    """Create Mechanism from fourbar data."""
    return create_mechanism_from_dict(fourbar_data)


class TestDimensionExtraction:
    """Tests for dimension extraction using Mechanism."""

    def test_get_dimension_spec_4bar(self, fourbar_mechanism):
        """Get dimension spec from 4-bar mechanism."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()

        # Hypergraph: dimensions come from edges (ground, crank_link, coupler, rocker)
        assert len(spec) >= 3
        assert any('distance' in name for name in spec.names)

    def test_get_dimension_spec_bounds(self, fourbar_mechanism):
        """Bounds should be computed from initial values."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()

        # Check that bounds exist and are valid
        for i, name in enumerate(spec.names):
            lower, upper = spec.bounds[i]
            assert lower < upper
            assert lower > 0

    def test_bounds_tuple_format(self, fourbar_mechanism):
        """get_bounds_tuple returns pylinkage format."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()
        lower, upper = spec.get_bounds_tuple()

        assert len(lower) == len(spec)
        assert len(upper) == len(spec)
        assert all(lower_val < upper_val for lower_val, upper_val in zip(lower, upper))


# =============================================================================
# Test: Dimension Application
# =============================================================================

class TestDimensionApplication:
    """Tests for applying dimensions using Mechanism."""

    def test_with_dimensions_updates_mechanism(self, fourbar_mechanism):
        """with_dimensions creates new mechanism with updated dimensions."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()
        if not spec.names:
            pytest.skip('No dimensions to test')

        first_dim = spec.names[0]
        # Use a value within bounds when bounds exist, so we don't trigger validation
        if spec.bounds and len(spec.bounds) > 0:
            lo, hi = spec.bounds[0]
            new_val = (lo + hi) / 2.0
            if abs(new_val - spec.initial_values[0]) < 1e-6:
                new_val = lo + (hi - lo) * 0.25  # ensure different from initial
        else:
            new_val = spec.initial_values[0] * 1.1  # no bounds: any value allowed
        new_values = {first_dim: new_val}

        updated_mechanism = fourbar_mechanism.with_dimensions(new_values)

        # Original unchanged
        orig_spec = fourbar_mechanism.get_dimension_bounds_spec()
        # Updated has new value
        updated_spec = updated_mechanism.get_dimension_bounds_spec()
        idx = updated_spec.names.index(first_dim)
        assert updated_spec.initial_values[idx] == pytest.approx(new_val, rel=0.01)
        assert orig_spec.initial_values[idx] != pytest.approx(new_val, rel=0.01)

    def test_set_dimensions_from_array(self, fourbar_mechanism):
        """set_dimensions uses spec order."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()
        if not spec.names:
            pytest.skip('No dimensions to test')

        # Create new values (all slightly different)
        values = tuple(v * 1.1 for v in spec.initial_values)

        # Create a copy and set dimensions
        test_mechanism = fourbar_mechanism.copy()
        test_mechanism.set_dimensions(values)

        # Verify update happened
        updated_spec = test_mechanism.get_dimension_bounds_spec()
        for i in range(len(spec.names)):
            assert updated_spec.initial_values[i] == pytest.approx(values[i], rel=0.01)


# # =============================================================================
# # Test: Error Computation
# # =============================================================================

# class TestErrorComputation:
#     """Tests for trajectory error computation."""

#     def test_compute_error_perfect_match(self):
#         """Perfect match gives zero error."""
#         positions = [(1, 2), (3, 4), (5, 6)]
#         target = TargetTrajectory(joint_name='test', positions=positions)

#         error = score_trajectory(target, positions, metric='mse')
#         assert error == pytest.approx(0.0, abs=1e-10)

#     def test_compute_error_shifted(self):
#         """Shifted positions give expected error."""
#         target_pos = [(0, 0), (1, 0), (2, 0)]
#         computed = [(1, 0), (2, 0), (3, 0)]  # Shifted by 1 in x
#         target = TargetTrajectory(joint_name='test', positions=target_pos)

#         # Each point is distance 1 away, MSE = 1^2 = 1
#         error = score_trajectory(target, computed, metric='mse')
#         assert error == pytest.approx(1.0, abs=0.01)

#     def test_compute_error_diagonal_shift(self):
#         """Diagonal shift gives correct error."""
#         target_pos = [(0, 0)]
#         computed = [(3, 4)]  # Distance = 5
#         target = TargetTrajectory(joint_name='test', positions=target_pos)

#         # MSE = 5^2 = 25
#         error = score_trajectory(target, computed, metric='mse')
#         assert error == pytest.approx(25.0, abs=0.01)


# =============================================================================
# Test: Known Solution (Inverse Problem) using achievable_target tools
# =============================================================================

class TestKnownSolution:
    """
    Tests using create_achievable_target to generate targets with KNOWN solutions.

    The key insight: we perturb a valid mechanism to create a target, then verify
    the optimizer can recover the perturbed dimensions (or get close).
    """

    def test_perfect_match_stays_optimal(self, fourbar_mechanism):
        """When target matches current trajectory, optimizer keeps dimensions."""
        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )

        # Error should already be ~0 (perfect match)
        from pylink_tools.trajectory_scoring import score_trajectory
        initial_error = score_trajectory(
            target,
            [(x, y) for x, y in trajectory],
            metric='mse',
            phase_invariant=False,
        )
        assert initial_error < 0.01, 'Initial fit should be perfect'

        # Optimization should maintain ~0 error
        mechanism_copy = fourbar_mechanism.copy()
        opt_result = run_scipy_optimization(
            mechanism_copy, target,
            max_iterations=20, verbose=False,
        )

        assert opt_result.success
        assert opt_result.final_error < 0.01

    def test_achievable_target_scipy(self, fourbar_mechanism):
        """Use create_achievable_target to get known solution, verify scipy recovers."""
        dim_spec = fourbar_mechanism.get_dimension_bounds_spec()
        target_joint = 'coupler_rocker_joint'

        # Create achievable target with ±20% variation (conservative for testing)
        config = MechVariationConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=0.20,
                exclude_dimensions=['ground_distance'],  # Don't vary ground
            ),
            random_seed=42,
            max_attempts=64,
        )

        target_result = create_achievable_target(
            fourbar_mechanism, target_joint, dim_spec=dim_spec, config=config,
        )

        # We now have a target with KNOWN achievable dimensions
        target = target_result.target
        known_dims = target_result.target_dimensions

        # Compute initial error (original mechanism vs perturbed target)
        initial_traj = fourbar_mechanism.get_trajectory(target_joint)
        from pylink_tools.trajectory_scoring import score_trajectory
        initial_error = score_trajectory(
            target,
            [(x, y) for x, y in initial_traj],
            metric='mse',
        )
        assert initial_error > 0, 'Target should differ from original'

        # Run scipy optimization
        mechanism_copy = fourbar_mechanism.copy()
        opt_result = run_scipy_optimization(
            mechanism_copy, target,
            max_iterations=150, verbose=False,
        )

        assert opt_result.success
        # Optimizer should reduce error (may not fully recover due to local minima)
        assert opt_result.final_error < initial_error

#     def test_achievable_target_pso(self, fourbar_mechanism):
#         """Use create_achievable_target to get known solution, verify PSO recovers."""
#         dim_spec = fourbar_mechanism.get_dimension_bounds_spec()
#         target_joint = 'coupler_rocker_joint'

#         # Create achievable target with smaller variation for PSO
#         config = MechVariationConfig(
#             dimension_variation=DimensionVariationConfig(
#                 default_variation_range=0.15,
#                 exclude_dimensions=['ground_distance'],
#             ),
#             random_seed=123,
#             max_attempts=64,
#         )

#         target_result = create_achievable_target(
#             fourbar_mechanism, target_joint, dim_spec=dim_spec, config=config,
#         )

#         target = target_result.target

#         # Compute initial error
#         initial_traj = fourbar_mechanism.get_trajectory(target_joint)
#         from pylink_tools.trajectory_scoring import score_trajectory
#         initial_error = compute_trajectory_error(
#             [(x, y) for x, y in initial_traj],
#             target,
#             metric='mse',
#         )

#         # Run PSO optimization (using optimize_trajectory with method='pso')
#         mechanism_copy = fourbar_mechanism.copy()
#         opt_result = optimize_trajectory(
#             mechanism_copy, target,
#             method='pso',
#             n_particles=20, iterations=30, verbose=False,
#         )

#         assert opt_result.success
#         # PSO should also reduce error
#         assert opt_result.final_error < initial_error


# # =============================================================================
# # Test: Convergence History
# # =============================================================================

class TestConvergenceHistory:
    """Tests for convergence history tracking."""

    def test_scipy_tracks_history(self, fourbar_mechanism):
        """scipy optimization tracks convergence history."""
        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )

        mechanism_copy = fourbar_mechanism.copy()
        opt_result = run_scipy_optimization(
            mechanism_copy, target,
            max_iterations=30, verbose=False,
        )

        assert opt_result.convergence_history is not None
        assert len(opt_result.convergence_history) >= 1

    def test_pso_tracks_history(self, fourbar_mechanism):
        """PSO optimization tracks convergence history."""
        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )

        mechanism_copy = fourbar_mechanism.copy()
        opt_result = optimize_trajectory(
            mechanism_copy, target,
            method='pylinkage',  # Use pylinkage PSO (standalone PSO was removed)
            n_particles=10, iterations=15, verbose=False,
        )

        assert opt_result.convergence_history is not None
        # History should have initial error + iterations entries
        assert len(opt_result.convergence_history) >= 1
        # Should have at least initial + final (iterations + 1)
        assert len(opt_result.convergence_history) <= 16  # iterations + 1


# =============================================================================
# Test: PSO dimension alignment (spec len != linkage.get_num_constraints())
# =============================================================================

class TestPylinkagePSODimensionAlignment:
    """
    Test pylinkage PSO when DimensionBoundsSpec has more dimensions than
    linkage.get_num_constraints() (e.g. 28 vs 27). Ensures no "Bounds dimensions
    must match" error and result has valid optimized_dimensions.
    """

    @pytest.fixture
    def leg_data(self):
        """Load leg mechanism from demo/test_graphs/leg.json."""
        test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / 'leg.json'
        if not test_file.exists():
            pytest.skip('leg.json not found')
        with open(test_file) as f:
            data = json.load(f)
        data['n_steps'] = 32
        return data

    @pytest.fixture
    def leg_mechanism(self, leg_data):
        """Create Mechanism from leg data."""
        return create_mechanism_from_dict(leg_data)

    def test_constraint_order_helper_matches_linkage(self, leg_mechanism):
        """_constraint_order_and_linkage_dims returns constraint_order length == linkage_n_dims."""
        spec = leg_mechanism.get_dimension_bounds_spec()
        constraint_order, linkage_n_dims = _constraint_order_and_linkage_dims(
            leg_mechanism, spec,
        )
        actual_linkage_dims = len(tuple(leg_mechanism.linkage.get_num_constraints()))
        assert linkage_n_dims == actual_linkage_dims
        # When spec has more dims than linkage, constraint_order length should equal linkage_n_dims
        if len(spec) > linkage_n_dims:
            assert len(constraint_order) == linkage_n_dims
            assert len(constraint_order) <= len(spec)

    def test_pylinkage_pso_no_bounds_dimension_error(self, leg_mechanism):
        """Run pylinkage PSO on leg mechanism; must not raise Bounds dimensions must match."""
        target_joint = 'toe'
        trajectory = leg_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from leg mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )

        mechanism_copy = leg_mechanism.copy()
        spec = mechanism_copy.get_dimension_bounds_spec()
        opt_result = optimize_trajectory(
            mechanism_copy,
            target,
            method='pylinkage',
            n_particles=8,
            iterations=3,
            verbose=False,
        )

        # Must not fail with bounds dimension mismatch
        if not opt_result.success and opt_result.error:
            assert 'Bounds dimensions' not in opt_result.error, (
                f'PSO must not fail with bounds dimension mismatch: {opt_result.error}'
            )

        if opt_result.success:
            assert len(opt_result.optimized_dimensions) == len(spec)
            for name in spec.names:
                assert name in opt_result.optimized_dimensions
                assert isinstance(opt_result.optimized_dimensions[name], (int, float))


# =============================================================================
# Test: Fitness Function
# =============================================================================

class TestFitnessFunction:
    """Tests for fitness function creation."""

    def test_fitness_function_callable(self, fourbar_mechanism):
        """create_mechanism_fitness returns callable."""
        spec = fourbar_mechanism.get_dimension_bounds_spec()
        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )

        mechanism_copy = fourbar_mechanism.copy()
        fitness = create_mechanism_fitness(
            mechanism_copy, target, target_joint=target_joint,
        )

        assert callable(fitness)

        # Call with initial values
        initial_values = tuple(spec.initial_values)
        error = fitness(initial_values)

        assert isinstance(error, float)
        assert error >= 0 or error == float('inf')


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_to_dict(self):
        """OptimizationResult.to_dict() serializes correctly."""
        result = OptimizationResult(
            success=True,
            optimized_dimensions={'a': 1.0, 'b': 2.0},
            optimized_mechanism=None,
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
# Test: TopologyVariationSpec
# =============================================================================

class TestTopologyVariationSpec:
    """Tests for TopologyVariationSpec (topology optimization constraints)."""

    def test_defaults(self):
        """Default TopologyVariationSpec has expected values."""
        spec = TopologyVariationSpec()
        assert spec.min_links == 4
        assert spec.max_links == 32
        assert spec.min_nodes == 4
        assert spec.max_nodes == 32
        assert spec.new_link_length_min == 1.0
        assert spec.new_link_length_max == 500.0
        assert spec.allowed_new_joint_types == ['revolute']
        assert spec.preserve_crank is True
        assert spec.allow_attach_to_edges is None
        assert spec.forbid_attach_to_edges == []

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict and from_dict roundtrip."""
        spec = TopologyVariationSpec(
            min_links=4,
            max_links=16,
            new_link_length_min=5.0,
            new_link_length_max=200.0,
            allow_attach_to_edges=['e1', 'e2'],
            forbid_attach_to_joints=['ground'],
            allowed_new_joint_types=['revolute', 'fixed'],
            preserve_crank=False,
        )
        data = spec.to_dict()
        restored = TopologyVariationSpec.from_dict(data)
        assert restored.min_links == spec.min_links
        assert restored.max_links == spec.max_links
        assert restored.new_link_length_min == spec.new_link_length_min
        assert restored.new_link_length_max == spec.new_link_length_max
        assert restored.allow_attach_to_edges == spec.allow_attach_to_edges
        assert restored.forbid_attach_to_joints == spec.forbid_attach_to_joints
        assert restored.allowed_new_joint_types == spec.allowed_new_joint_types
        assert restored.preserve_crank == spec.preserve_crank


# =============================================================================
# Test: SCIP optimizer (discretized dimension-only)
# =============================================================================

class TestSCIPOptimizer:
    """Tests for SCIP discretized optimization."""

    def test_scip_returns_result_structure(self, fourbar_mechanism):
        """SCIP returns OptimizationResult with expected fields (may be mock if pyscipopt missing)."""
        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )
        mechanism_copy = fourbar_mechanism.copy()

        result = optimize_trajectory(
            mechanism_copy,
            target,
            method='scip',
            discretization_steps=4,
            verbose=False,
        )

        assert hasattr(result, 'success')
        assert hasattr(result, 'optimized_dimensions')
        assert hasattr(result, 'initial_error')
        assert hasattr(result, 'final_error')
        assert isinstance(result.optimized_dimensions, dict)

    def test_scip_with_small_grid_improves_or_holds(self, fourbar_mechanism):
        """When pyscipopt is available, SCIP with small grid runs and improves or holds error."""
        try:
            from pyscipopt import Model  # noqa: F401
        except ImportError:
            pytest.skip('pyscipopt not installed')

        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )
        mechanism_copy = fourbar_mechanism.copy()

        result = optimize_trajectory(
            mechanism_copy,
            target,
            method='scip',
            discretization_steps=4,
            verbose=False,
        )

        assert result.success
        assert result.final_error <= result.initial_error + 1e-9
        assert result.iterations >= 1
        assert result.convergence_history is not None
        assert len(result.optimized_dimensions) >= 1

    def test_scip_accepts_topology_variation_spec(self, fourbar_mechanism):
        """Passing topology_variation_spec does not raise; SCIP runs dimension-only for now."""
        try:
            from pyscipopt import Model  # noqa: F401
        except ImportError:
            pytest.skip('pyscipopt not installed')

        target_joint = 'coupler_rocker_joint'
        trajectory = fourbar_mechanism.get_trajectory(target_joint)
        if trajectory is None or len(trajectory) == 0:
            pytest.skip('Could not compute trajectory from mechanism')

        target = TargetTrajectory(
            joint_name=target_joint,
            positions=[[float(x), float(y)] for x, y in trajectory],
        )
        mechanism_copy = fourbar_mechanism.copy()
        topo_spec = TopologyVariationSpec(min_links=4, max_links=8)

        result = optimize_trajectory(
            mechanism_copy,
            target,
            method='scip',
            topology_variation_spec=topo_spec,
            discretization_steps=4,
            verbose=False,
        )

        assert result.success
        assert result.best_topology_id is not None


# =============================================================================
# Test: Topology validity (topology_id, BadTopologyCache, Grashof)
# =============================================================================

class TestTopologyValidity:
    """Tests for topology_validity module."""

    def test_topology_id_from_edges(self):
        """topology_id_from_edges returns canonical frozenset of node pairs."""
        from pylink_tools.topology_validity import topology_id_from_edges

        edges1 = {
            'e1': {'source': 'A', 'target': 'B', 'distance': 10},
            'e2': {'source': 'B', 'target': 'C', 'distance': 20},
        }
        edges2 = {
            'x': {'source': 'B', 'target': 'A', 'distance': 5},
            'y': {'source': 'C', 'target': 'B', 'distance': 15},
        }
        id1 = topology_id_from_edges(edges1)
        id2 = topology_id_from_edges(edges2)
        assert id1 == id2
        assert len(id1) == 2
        assert ('A', 'B') in id1
        assert ('B', 'C') in id1

    def test_bad_topology_cache(self):
        """BadTopologyCache add_bad, is_bad, get_reason, to_report_list."""
        from pylink_tools.topology_validity import BadTopologyCache, topology_id_from_edges

        cache = BadTopologyCache()
        tid = frozenset({('A', 'B'), ('B', 'C')})
        assert not cache.is_bad(tid)
        assert cache.get_reason(tid) is None

        cache.add_bad(tid, 'non_grashof')
        assert cache.is_bad(tid)
        assert cache.get_reason(tid) == 'non_grashof'
        assert len(cache) == 1
        report = cache.to_report_list()
        assert len(report) == 1
        assert report[0]['reason'] == 'non_grashof'
        assert 'topology_id' in report[0]

    def test_is_grashof_compliant(self):
        """Grashof: s + l <= p + q for four-bar."""
        from pylink_tools.topology_validity import is_grashof_compliant

        assert is_grashof_compliant([10.0, 20.0, 30.0, 40.0]) is True   # 10+40 <= 20+30
        assert is_grashof_compliant([10.0, 20.0, 20.0, 40.0]) is False  # 10+40=50 > 20+20=40
        assert is_grashof_compliant([1, 2, 3, 4]) is True
        assert is_grashof_compliant([1, 2, 3]) is True  # not 4-bar: no reject

    def test_check_mechanism_validity_uses_cache(self, fourbar_mechanism):
        """check_mechanism_validity marks bad topology in cache when invalid."""
        from pylink_tools.topology_validity import (
            BadTopologyCache,
            check_mechanism_validity,
            topology_id_from_edges,
        )

        doc = fourbar_mechanism.to_dict()
        edges = doc.get('linkage', {}).get('edges', {})
        tid = topology_id_from_edges(edges)
        cache = BadTopologyCache()
        mech_copy = fourbar_mechanism.copy()
        valid, reason = check_mechanism_validity(mech_copy, tid, cache, check_grashof=True)
        # Four-bar with default lengths is typically Grashof and simulates
        if valid:
            assert reason is None
            assert not cache.is_bad(tid)
        else:
            assert cache.is_bad(tid)
            assert cache.get_reason(tid) in ('non_grashof', 'simulation_failed')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
