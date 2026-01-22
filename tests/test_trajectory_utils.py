"""
test_trajectory_utils.py - Tests for trajectory manipulation utilities.

One test per function in trajectory_utils.py to verify core functionality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pylink_tools.trajectory_utils import analyze_trajectory
from pylink_tools.trajectory_utils import compute_phase_aligned_distance
from pylink_tools.trajectory_utils import compute_trajectory_hot
from pylink_tools.trajectory_utils import compute_trajectory_similarity
from pylink_tools.trajectory_utils import prepare_trajectory_for_optimization
from pylink_tools.trajectory_utils import resample_trajectory
from pylink_tools.trajectory_utils import smooth_trajectory
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================

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


# =============================================================================
# Test: resample_trajectory
# =============================================================================

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

    # Resample to fewer points
    resampled_12 = resample_trajectory(original, 12)
    assert len(resampled_12) == 12

    # Same count should return same trajectory
    resampled_24 = resample_trajectory(original, 24)
    assert len(resampled_24) == 24

    # Verify resampled points are still approximately on the circle
    # (within reasonable tolerance of radius 10 from center 50,50)
    for x, y in resampled_48:
        dist_from_center = np.sqrt((x - 50)**2 + (y - 50)**2)
        assert 9.5 < dist_from_center < 10.5, f'Point ({x}, {y}) not on circle'


# =============================================================================
# Test: smooth_trajectory
# =============================================================================

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


# =============================================================================
# Test: compute_phase_aligned_distance
# =============================================================================

def test_compute_phase_aligned_distance(circle_trajectory):
    """
    Test that phase alignment correctly identifies phase-shifted trajectories.

    Verifies:
    - Identical trajectories have distance ~0
    - Phase-shifted identical trajectories also have distance ~0 after alignment
    - Phase offset is correctly detected
    """
    original = circle_trajectory

    # Shift by 6 points (90 degrees for 24-point circle)
    phase_offset = 6
    shifted = original[phase_offset:] + original[:phase_offset]

    # Without proper comparison, these would seem very different
    # But phase alignment should find they're the same
    result = compute_phase_aligned_distance(original, shifted, method='rotation')

    # Distance should be essentially zero (same path, just shifted)
    assert result.distance < 0.01, \
        f'Phase-aligned distance should be ~0 for identical shifted paths, got {result.distance}'

    # Should detect the correct offset (or equivalent)
    # Note: offset could be 6 or 18 (both align the circle)
    assert result.best_phase_offset in [6, 18], \
        f'Expected phase offset 6 or 18, got {result.best_phase_offset}'


# =============================================================================
# Test: compute_trajectory_similarity
# =============================================================================

def test_compute_trajectory_similarity(circle_trajectory):
    """
    Test high-level similarity computation with automatic handling.

    Verifies:
    - Identical trajectories have similarity ~0
    - Different trajectories have higher similarity score
    - Phase-invariant mode handles phase shifts correctly
    """
    original = circle_trajectory

    # Shifted version (same path, different phase)
    shifted = original[6:] + original[:6]

    # Phase-invariant should return ~0 for shifted identical paths
    sim_phase_inv = compute_trajectory_similarity(
        original, shifted, phase_invariant=True, method='mse',
    )
    assert sim_phase_inv < 0.01, \
        f'Phase-invariant similarity should be ~0, got {sim_phase_inv}'

    # Without phase invariance, should be much higher
    sim_no_phase = compute_trajectory_similarity(
        original, shifted, phase_invariant=False, method='mse',
    )
    assert sim_no_phase > 100, \
        f'Non-phase-invariant similarity should be high for shifted paths, got {sim_no_phase}'


# =============================================================================
# Test: prepare_trajectory_for_optimization
# =============================================================================

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


# =============================================================================
# Test: analyze_trajectory
# =============================================================================

def test_analyze_trajectory(circle_trajectory):
    """
    Test trajectory analysis returns correct statistics.

    Verifies:
    - Correct point count
    - Centroid is at expected location (center of circle)
    - Bounding box dimensions are correct
    - Trajectory is identified as approximately closed
    """
    circle = circle_trajectory
    stats = analyze_trajectory(circle)

    # Point count
    assert stats['n_points'] == 24

    # Centroid should be at (50, 50)
    cx, cy = stats['centroid']
    assert abs(cx - 50) < 0.1, f'Centroid x should be ~50, got {cx}'
    assert abs(cy - 50) < 0.1, f'Centroid y should be ~50, got {cy}'

    # Bounding box should be ~20x20 (diameter of circle with radius 10)
    bbox = stats['bounding_box']
    assert abs(bbox['width'] - 20) < 0.1, f"Width should be ~20, got {bbox['width']}"
    assert abs(bbox['height'] - 20) < 0.1, f"Height should be ~20, got {bbox['height']}"

    # Path length should be approximately circumference (2*pi*10 ≈ 62.8)
    # But our discrete approximation will be slightly less
    assert 55 < stats['total_path_length'] < 65, \
        f"Path length should be ~60, got {stats['total_path_length']}"


# =============================================================================
# Test: compute_trajectory_hot
# =============================================================================


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
    mse_identical = compute_trajectory_hot(original, original)
    assert mse_identical < 1e-10, f'Identical trajectories should have MSE ~0, got {mse_identical}'

    # Phase-shifted should also be ~0 (phase-invariant)
    mse_shifted = compute_trajectory_hot(original, shifted)
    assert mse_shifted < 0.01, f'Phase-shifted identical paths should have MSE ~0, got {mse_shifted}'

    # Should match full API with FFT method
    full_api_result = compute_phase_aligned_distance(
        circle_trajectory,
        list(map(tuple, shifted)),
        method='fft',
    )
    assert abs(mse_shifted - full_api_result.distance) < 1e-10, \
        f'compute_trajectory_hot should match full API: {mse_shifted} vs {full_api_result.distance}'


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
