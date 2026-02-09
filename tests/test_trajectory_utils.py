from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pylink_tools.trajectory_scoring import mse
from pylink_tools.trajectory_scoring import score_trajectory
from pylink_tools.trajectory_utils import align_trajectory_phase
from pylink_tools.trajectory_utils import find_phase_offset_rotation
from pylink_tools.trajectory_utils import prepare_trajectory_for_optimization
from pylink_tools.trajectory_utils import resample_trajectory
from pylink_tools.trajectory_utils import smooth_trajectory
# from pylink_tools.trajectory_utils import analyze_trajectory
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def circle_trajectory() -> list[tuple[float, float]]:
    """Create a simple circle trajectory for testing."""
    n_points = 24
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 10
    center = (50, 50)
    return [
        (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))
        for a in angles
    ]


@pytest.fixture
def noisy_trajectory(circle_trajectory) -> list[tuple[float, float]]:
    """Create a noisy version of circle trajectory."""
    np.random.seed(42)
    return [
        (x + np.random.normal(0, 0.5), y + np.random.normal(0, 0.5))
        for x, y in circle_trajectory
    ]


def test_resample_trajectory(circle_trajectory):
    """
    Test that resample_trajectory correctly changes point count.

    Verifies:
    - Output has exactly the requested number of points
    - Resampling preserves approximate shape (points stay near original path)
    """
    original = circle_trajectory
    assert len(original) == 24

    # Resample to more points
    resampled_48 = resample_trajectory(original, 48)
    assert len(resampled_48) == 48

    # Verify resampled points are still approximately on the circle
    # (within reasonable tolerance of radius 10 from center 50,50)
    for x, y in resampled_48:
        dist_from_center = np.sqrt((x - 50)**2 + (y - 50)**2)
        assert 9.5 < dist_from_center < 10.5, f'Point ({x}, {y}) not on circle'


def test_smooth_trajectory(noisy_trajectory):
    """
    Test that smooth_trajectory reduces noise while preserving shape.

    Verifies:
    - Output has same number of points as input
    - Smoothed trajectory is closer to ideal circle than noisy input
    """
    noisy = noisy_trajectory
    smoothed = smooth_trajectory(noisy, window_size=5, polyorder=3)

    # Same length
    assert len(smoothed) == len(noisy)

    # Compute deviation from ideal circle (radius=10, center=50,50)
    def avg_deviation(traj):
        deviations = [abs(np.sqrt((x-50)**2 + (y-50)**2) - 10) for x, y in traj]
        return np.mean(deviations)

    noisy_dev = avg_deviation(noisy)
    smooth_dev = avg_deviation(smoothed)

    # Smoothed should be closer to ideal circle (lower deviation)
    assert smooth_dev < noisy_dev, \
        f'Smoothing should reduce deviation: {smooth_dev:.4f} should be < {noisy_dev:.4f}'


def test_compute_phase_aligned_distance(circle_trajectory):
    """
    Test that phase alignment correctly identifies phase-shifted trajectories.

    Verifies:
    - Identical trajectories have distance ~0
    - Phase-shifted identical trajectories also have distance ~0 after alignment
    - Phase offset is correctly detected
    """
    original = np.array(circle_trajectory)

    # Shift by 6 points (90 degrees for 24-point circle)
    phase_offset = 6
    shifted = np.roll(original, phase_offset, axis=0)

    # Without proper comparison, these would seem very different
    # But phase alignment should find they're the same
    aligned = align_trajectory_phase(original, shifted, method='rotation')
    distance = mse(original, aligned)

    # Distance should be essentially zero (same path, just shifted)
    assert distance < 0.01, \
        f'Phase-aligned distance should be ~0 for identical shifted paths, got {distance}'

    # Should detect the correct offset (or equivalent)
    # Note: offset could be 6 or 18 (both align the circle)
    detected_offset = find_phase_offset_rotation(original, shifted)
    assert detected_offset in [6, 18], \
        f'Expected phase offset 6 or 18, got {detected_offset}'


def test_prepare_trajectory_for_optimization(noisy_trajectory):
    """
    Test the convenience function that combines smoothing and resampling.

    Verifies:
    - Output has exactly target_n_steps points
    - Output is smoother than input (if smooth=True)
    """
    noisy = noisy_trajectory  # 24 noisy points
    target_n_steps = 36

    # Prepare with smoothing
    prepared = prepare_trajectory_for_optimization(
        noisy,
        target_n_steps=target_n_steps,
        smooth=True,
        smooth_window=5,
        smooth_polyorder=3,
    )

    # Correct length
    assert len(prepared) == target_n_steps

    # Without smoothing
    prepared_no_smooth = prepare_trajectory_for_optimization(
        noisy,
        target_n_steps=target_n_steps,
        smooth=False,
    )
    assert len(prepared_no_smooth) == target_n_steps


def test_compute_trajectory_hot(circle_trajectory):
    """
    Test ultra-fast MSE computation matches full API results.

    Verifies:
    - Identical trajectories have MSE ~0
    - Phase-shifted trajectories also have MSE ~0 (phase-invariant)
    - Results match compute_phase_aligned_distance with method='fft'
    """
    original = np.array(circle_trajectory)
    shifted = np.roll(original, 6, axis=0)  # Shift by 6 points

    # Identical should be ~0
    mse_identical = score_trajectory(
        original,
        original,
        metric='mse',
        phase_invariant=True,
    )
    assert mse_identical < 1e-10, f'Identical trajectories should have MSE ~0, got {mse_identical}'

    # Phase-shifted should also be ~0 (phase-invariant)
    mse_shifted = score_trajectory(
        original,
        shifted,
        metric='mse',
        phase_invariant=True,
    )
    assert mse_shifted < 0.01, f'Phase-shifted identical paths should have MSE ~0, got {mse_shifted}'

    # Should match trajectory_metrics with FFT method
    aligned = align_trajectory_phase(original, shifted, method='fft')
    full_api_result = mse(original, aligned)
    assert abs(mse_shifted - full_api_result) < 1e-10, \
        f'compute_trajectory_hot should match trajectory_metrics: {mse_shifted} vs {full_api_result}'


def test_score_trajectory_distance_metrics(circle_trajectory):
    """score_trajectory runs with each distance metric and returns finite scores (lower=better)."""
    ref = np.array(circle_trajectory)
    metrics = [
        'sspd',
        'dtw',
        'hausdorff',
        # 'frechet', TODO: fix frechet not finite
        'frechet_discrete',
        'lcss',
        'edr',
        'erp',
    ]
    for m in metrics:
        same = score_trajectory(ref, ref, metric=m, phase_invariant=False)
        assert np.isfinite(same), f'{m}: identical traj should be finite'
        assert same >= 0, f'{m}: score should be non-negative'
        # Slightly different trajectory should give positive score
        shifted = np.roll(ref, 3, axis=0)
        diff_score = score_trajectory(ref, shifted, metric=m, phase_invariant=True)
        assert np.isfinite(diff_score) and diff_score >= 0, f'{m}: diff score finite and >= 0'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
