"""
Tests for trajectory scoring functions.

Tests verify that:
- MSE is always non-negative
- Phase alignment works correctly
- Different metrics work as expected
- Edge cases are handled properly
"""
from __future__ import annotations

import numpy as np
import pytest

from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.trajectory_scoring import score_trajectory


@pytest.fixture
def simple_trajectory():
    """Simple square trajectory."""
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def rotated_trajectory():
    """Same square but rotated (phase shifted)."""
    return [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]


@pytest.fixture
def shifted_trajectory():
    """Same square but translated."""
    return [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)]


@pytest.fixture
def different_trajectory():
    """Different shape (rectangle)."""
    return [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]


class TestScoreTrajectoryMse:
    """Tests for score_trajectory with MSE metric."""

    def test_identical_trajectories(self):
        """MSE should be 0 for identical trajectories."""
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        score = score_trajectory(traj, traj, metric='mse', phase_invariant=True)
        assert score == 0.0, f'MSE should be 0 for identical trajectories, got {score}'

    def test_phase_aligned_identical(self, simple_trajectory, rotated_trajectory):
        """Phase-shifted identical trajectories should have MSE ~0."""
        score = score_trajectory(simple_trajectory, rotated_trajectory, metric='mse', phase_invariant=True)
        assert score >= 0, f'MSE should be non-negative, got {score}'
        assert score < 1e-10, f'Phase-aligned identical trajectories should have MSE ~0, got {score}'

    def test_always_non_negative(self):
        """MSE should always be non-negative."""
        # Test with various trajectories
        test_cases = [
            ([(0, 0), (1, 0), (1, 1)], [(0, 0), (2, 0), (2, 2)]),
            ([(0, 0), (1, 0)], [(5, 5), (6, 5)]),
            ([(0, 0), (1, 0), (1, 1), (0, 1)], [(10, 10), (11, 10), (11, 11), (10, 11)]),
        ]

        for traj1_list, traj2_list in test_cases:
            score = score_trajectory(traj1_list, traj2_list, metric='mse', phase_invariant=True)
            assert score >= 0, f'MSE should be non-negative. traj1={traj1_list}, traj2={traj2_list}, score={score}'

    def test_different_trajectories(self, simple_trajectory, different_trajectory):
        """Different trajectories should have positive MSE."""
        score = score_trajectory(simple_trajectory, different_trajectory, metric='mse', phase_invariant=True)
        assert score > 0, f'MSE should be positive for different trajectories, got {score}'
        assert score >= 0, f'MSE should be non-negative, got {score}'

    def test_symmetry(self):
        """MSE should be symmetric: score(traj1, traj2) == score(traj2, traj1)."""
        traj1 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        traj2 = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]

        score_12 = score_trajectory(traj1, traj2, metric='mse', phase_invariant=True)
        score_21 = score_trajectory(traj2, traj1, metric='mse', phase_invariant=True)

        assert abs(score_12 - score_21) < 1e-10, \
            f'MSE should be symmetric: score(traj1, traj2)={score_12}, score(traj2, traj1)={score_21}'

    def test_translation_invariance_not_applied(self):
        """Translation should affect MSE when translation_invariant=False."""
        traj1 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        traj2 = [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)]

        score = score_trajectory(traj1, traj2, metric='mse', phase_invariant=True, translation_invariant=False)
        # Should be large due to translation
        assert score > 10, f'Translated trajectories should have large MSE, got {score}'


class TestScoreTrajectory:
    """Tests for score_trajectory - unified scoring function."""

    def test_mse_metric(self, simple_trajectory):
        """Test MSE metric."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=simple_trajectory,
        )
        computed = simple_trajectory  # Identical

        score = score_trajectory(
            target,
            computed,
            metric='mse',
            phase_invariant=True,
        )

        assert score >= 0, f'MSE should be non-negative, got {score}'
        assert score < 1e-10, f'Identical trajectories should have MSE ~0, got {score}'

    def test_phase_invariance(self, simple_trajectory, rotated_trajectory):
        """Phase-shifted identical trajectories should have low error."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=simple_trajectory,
        )

        score = score_trajectory(
            target,
            rotated_trajectory,
            metric='mse',
            phase_invariant=True,
        )

        assert score >= 0, f'Score should be non-negative, got {score}'
        assert score < 1e-10, \
            f'Phase-aligned identical trajectories should have score ~0, got {score}'

    def test_no_phase_invariance(self, simple_trajectory, rotated_trajectory):
        """Without phase invariance, phase-shifted trajectories should have higher error."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=simple_trajectory,
        )

        score_no_phase = score_trajectory(
            target,
            rotated_trajectory,
            metric='mse',
            phase_invariant=False,
        )

        score_with_phase = score_trajectory(
            target,
            rotated_trajectory,
            metric='mse',
            phase_invariant=True,
        )

        assert score_no_phase >= score_with_phase, \
            f'Score without phase alignment ({score_no_phase}) should be >= score with phase alignment ({score_with_phase})'

    def test_translation_invariance(self, simple_trajectory, shifted_trajectory):
        """With translation invariance, translated identical shapes should have low error."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=simple_trajectory,
        )

        score_with_translation_inv = score_trajectory(
            target,
            shifted_trajectory,
            metric='mse',
            phase_invariant=True,
            translation_invariant=True,
        )

        score_without_translation_inv = score_trajectory(
            target,
            shifted_trajectory,
            metric='mse',
            phase_invariant=True,
            translation_invariant=False,
        )

        assert score_with_translation_inv >= 0, \
            f'Score with translation invariance should be non-negative, got {score_with_translation_inv}'
        assert score_with_translation_inv < score_without_translation_inv, \
            f'Translation-invariant score ({score_with_translation_inv}) should be < non-invariant ({score_without_translation_inv})'

    def test_different_lengths(self, simple_trajectory):
        """Different length trajectories should return inf."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=simple_trajectory,
        )
        computed = simple_trajectory[:2]  # Shorter

        score = score_trajectory(
            target,
            computed,
            metric='mse',
        )

        assert score == float('inf'), \
            f'Different length trajectories should return inf, got {score}'

    def test_empty_trajectory(self):
        """Empty trajectory should return 0."""
        target = TargetTrajectory(
            joint_name='test_joint',
            positions=[],
        )

        score = score_trajectory(
            target,
            [],
            metric='mse',
        )

        assert score == 0.0, f'Empty trajectory should return 0, got {score}'


class TestScoringEdgeCases:
    """Tests for edge cases and potential bugs."""

    def test_negative_mse_bug_check(self):
        """Verify that MSE is never negative (regression test for bug)."""
        # Test various trajectory pairs that might trigger edge cases
        test_cases = [
            # Same trajectories
            ([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)]),
            # Very different trajectories
            ([(0, 0), (1, 0)], [(100, 100), (101, 100)]),
            # One point trajectories
            ([(0, 0)], [(1, 1)]),
            # Large values
            ([(0, 0), (1000, 0), (1000, 1000)], [(0, 0), (2000, 0), (2000, 2000)]),
            # Small values
            ([(0, 0), (0.001, 0), (0.001, 0.001)], [(0, 0), (0.002, 0), (0.002, 0.002)]),
        ]

        for traj1_list, traj2_list in test_cases:
            traj1 = np.array(traj1_list, dtype=np.float64)
            traj2 = np.array(traj2_list, dtype=np.float64)

            # Test score_trajectory
            target = TargetTrajectory(
                joint_name='test',
                positions=traj2_list,
            )
            score = score_trajectory(
                target,
                traj1_list,
                metric='mse',
                phase_invariant=True,
            )
            assert score >= 0, \
                f'score_trajectory returned negative score: {score} for traj1={traj1_list}, traj2={traj2_list}'

    def test_phase_alignment_consistency(self):
        """Test that phase alignment gives consistent results."""
        traj1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        traj2 = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)  # Rotated

        # Should get same result both ways
        traj1_list = [(float(x), float(y)) for x, y in traj1]
        traj2_list = [(float(x), float(y)) for x, y in traj2]
        target1 = TargetTrajectory(joint_name='test', positions=traj1_list)
        target2 = TargetTrajectory(joint_name='test', positions=traj2_list)

        score_12 = score_trajectory(target1, traj2_list, metric='mse', phase_invariant=True)
        score_21 = score_trajectory(target2, traj1_list, metric='mse', phase_invariant=True)

        assert abs(score_12 - score_21) < 1e-10, \
            f'Phase-aligned MSE should be symmetric: {score_12} vs {score_21}'
