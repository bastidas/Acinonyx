"""
trajectory_utils.py - Trajectory manipulation utilities.

This module provides essential tools for working with mechanism trajectories:
  - Resampling: Match trajectory point counts between different sources
  - Smoothing: Reduce noise in captured/irregular trajectories
  - Analysis: Statistics and information about trajectories

Note: For distance metrics, see trajectory_metrics.py.
For error scoring during optimization, see trajectory_scoring.py.

=============================================================================
CRITICAL PARAMETERS - Understanding Their Impact
=============================================================================

N_STEPS (Simulation Step Count):
    The number of points computed per full revolution of the crank.

    EFFECTS:
    - Higher N_STEPS = smoother trajectories, better optimization accuracy
    - Higher N_STEPS = slower simulation (linear scaling)
    - Higher N_STEPS = more memory for trajectory storage

    RECOMMENDATIONS:
    - Quick testing: 12-24 steps
    - Normal optimization: 24-48 steps
    - High precision: 48 above steps

    IMPORTANT: Target trajectory and simulation MUST have same N_STEPS!
    Use resample_trajectory() to match point counts.

SMOOTHING_WINDOW:
    Window size for trajectory smoothing (Savitzky-Golay filter).
    Must be odd number, typically 3-11.

    EFFECTS:
    - Larger window = more smoothing, may lose sharp features
    - Smaller window = preserves detail, less noise reduction

    RECOMMENDATIONS:
    - Light smoothing: window=3, polyorder=2
    - Medium smoothing: window=5, polyorder=3
    - Heavy smoothing: window=7-11, polyorder=3

SMOOTHING_POLYORDER:
    Polynomial order for Savitzky-Golay filter. Must be < window size.

    EFFECTS:
    - Higher order = preserves peaks/valleys better
    - Lower order = more aggressive smoothing

    RECOMMENDATIONS:
    - Preserve sharp corners: polyorder=2-3
    - Smooth curves: polyorder=3-4

=============================================================================
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
Trajectory = list[tuple[float, float]]
TrajectoryArray = np.ndarray  # Shape: (n_points, 2)


def resample_trajectory(
    trajectory: Trajectory,
    target_n_points: int,
    method: Literal['linear', 'cubic', 'parametric'] = 'parametric',
    closed: bool = True,
) -> Trajectory:
    """
    Resample a trajectory to have a specific number of points.

    This is CRITICAL when:
    - Target trajectory has different point count than simulation N_STEPS
    - Combining trajectories from different sources
    - Changing resolution for optimization

    Args:
        trajectory: Original trajectory as list of (x, y) tuples
        target_n_points: Desired number of output points
        method: Interpolation method
            - "linear": Fast, may create sharp corners
            - "cubic": Smooth curves, can overshoot
            - "parametric": Best for closed curves (recommended)
        closed: If True (default), treat the trajectory as a closed loop
                where the last point connects back to the first point.
                This ensures the closing segment is included in resampling.

    Returns:
        Resampled trajectory with exactly target_n_points points.
        For closed=True, the last point will NOT duplicate the first
        (the closure is implicit).

    Example:
        >>> original = [(0,0), (1,0), (1,1), (0,1)]  # 4 points, closed square
        >>> resampled = resample_trajectory(original, 8, closed=True)
        >>> len(resampled)
        8

    Note:
        For cyclic trajectories (linkage paths), use method="parametric"
        and closed=True to ensure proper interpolation around the full loop.
    """
    if len(trajectory) == target_n_points:
        return trajectory  # No resampling needed

    if len(trajectory) < 2:
        raise ValueError('Trajectory must have at least 2 points')

    if target_n_points < 2:
        raise ValueError('target_n_points must be at least 2')

    traj = np.array(trajectory)

    if method == 'linear':
        return _resample_linear(traj, target_n_points, closed=closed)
    elif method == 'cubic':
        return _resample_cubic(traj, target_n_points, closed=closed)
    elif method == 'parametric':
        return _resample_parametric(traj, target_n_points, closed=closed)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'cubic', or 'parametric'")


def _resample_linear(traj: np.ndarray, n_points: int, closed: bool = True) -> Trajectory:
    """Linear interpolation resampling for closed or open curves."""
    n_orig = len(traj)

    if closed:
        # For closed curves, add the first point at the end to complete the loop
        traj_closed = np.vstack([traj, traj[0:1]])
        n_closed = n_orig + 1

        # Parameter t goes around the full closed loop
        t_orig = np.arange(n_closed)
        # Don't include endpoint since it's the same as start
        t_new = np.linspace(0, n_closed - 1, n_points, endpoint=False)

        x_new = np.interp(t_new, t_orig, traj_closed[:, 0])
        y_new = np.interp(t_new, t_orig, traj_closed[:, 1])
    else:
        # Open curve - standard interpolation
        t_orig = np.arange(n_orig)
        t_new = np.linspace(0, n_orig - 1, n_points)

        x_new = np.interp(t_new, t_orig, traj[:, 0])
        y_new = np.interp(t_new, t_orig, traj[:, 1])

    return [(float(x), float(y)) for x, y in zip(x_new, y_new)]


def _resample_cubic(traj: np.ndarray, n_points: int, closed: bool = True) -> Trajectory:
    """Cubic spline interpolation resampling for closed or open curves."""
    from scipy.interpolate import CubicSpline

    n_orig = len(traj)

    if closed:
        # For closed curves, use periodic boundary conditions
        t_orig = np.arange(n_orig)
        # Don't include endpoint since it wraps around
        t_new = np.linspace(0, n_orig, n_points, endpoint=False)

        # Create periodic cubic splines for x and y
        cs_x = CubicSpline(t_orig, traj[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_orig, traj[:, 1], bc_type='periodic')
    else:
        # Open curve - standard cubic spline
        t_orig = np.arange(n_orig)
        t_new = np.linspace(0, n_orig - 1, n_points)

        cs_x = CubicSpline(t_orig, traj[:, 0])
        cs_y = CubicSpline(t_orig, traj[:, 1])

    x_new = cs_x(t_new)
    y_new = cs_y(t_new)

    return [(float(x), float(y)) for x, y in zip(x_new, y_new)]


def _resample_parametric(traj: np.ndarray, n_points: int, closed: bool = True) -> Trajectory:
    """
    Parametric resampling based on arc length.

    Best for closed curves - distributes points evenly along the entire path,
    including the segment from the last point back to the first.
    """

    if closed:
        # For closed curves, include the closing segment (last point to first)
        traj_closed = np.vstack([traj, traj[0:1]])

        # Compute cumulative arc length including the closing segment
        diffs = np.diff(traj_closed, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_length = np.zeros(len(traj_closed))
        arc_length[1:] = np.cumsum(segment_lengths)

        total_length = arc_length[-1]

        if total_length == 0:
            return [tuple(traj[0])] * n_points

        # Target arc lengths (evenly spaced around the full loop)
        # Don't include endpoint since it's the same as start
        target_arc = np.linspace(0, total_length, n_points, endpoint=False)

        # Interpolate x and y as functions of arc length
        interp_x = interp1d(arc_length, traj_closed[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(arc_length, traj_closed[:, 1], kind='linear', fill_value='extrapolate')
    else:
        # Open curve - don't include closing segment
        diffs = np.diff(traj, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_length = np.zeros(len(traj))
        arc_length[1:] = np.cumsum(segment_lengths)

        total_length = arc_length[-1]

        if total_length == 0:
            return [tuple(traj[0])] * n_points

        target_arc = np.linspace(0, total_length, n_points)

        interp_x = interp1d(arc_length, traj[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(arc_length, traj[:, 1], kind='linear', fill_value='extrapolate')

    x_new = interp_x(target_arc)
    y_new = interp_y(target_arc)

    return [(float(x), float(y)) for x, y in zip(x_new, y_new)]

# Smoothing Functions


def smooth_trajectory(
    trajectory: Trajectory,
    window_size: int = 5,
    polyorder: int = 3,
    method: Literal['savgol', 'moving_avg', 'gaussian'] = 'savgol',
) -> Trajectory:
    """
    Smooth a trajectory to reduce noise while preserving shape.

    Use this when:
    - Target trajectory comes from noisy measurements
    - Hand-drawn or digitized paths need cleanup
    - Reducing high-frequency oscillations

    Args:
        trajectory: Original trajectory as list of (x, y) tuples
        window_size: Size of smoothing window (must be odd for savgol)
        polyorder: Polynomial order for savgol filter (must be < window_size)
        method: Smoothing method
            - "savgol": Savitzky-Golay filter (preserves peaks, recommended)
            - "moving_avg": Simple moving average (aggressive smoothing)
            - "gaussian": Gaussian-weighted average (natural smoothing)

    Returns:
        Smoothed trajectory with same number of points

    Example:
        >>> noisy = [(0, 0.1), (1, -0.05), (2, 0.08), ...]  # Noisy data
        >>> smooth = smooth_trajectory(noisy, window_size=5)
        >>> # smooth now has reduced noise

    Hyperparameter Guide:
        Light smoothing:   window_size=2-4, polyorder=2
        Medium smoothing:  window_size=8-16, polyorder=3
        Heavy smoothing:   window_size=32-64, polyorder=3
    """
    if len(trajectory) < 3:
        return trajectory  # Can't smooth very short trajectories

    traj = np.array(trajectory)

    # Ensure window_size is valid (clamp to trajectory length)
    window_size = min(window_size, len(traj))
    window_size = max(2, window_size)

    if method == 'savgol':
        # Savgol requires odd window >= 3 and polyorder < window
        if window_size < 3:
            window_size = 3
        if window_size % 2 == 0:
            window_size += 1  # Make odd
        polyorder = min(polyorder, window_size - 1)
        return _smooth_savgol(traj, window_size, polyorder)
    elif method == 'moving_avg':
        # Moving average works with any window size >= 2
        return _smooth_moving_avg(traj, window_size)
    elif method == 'gaussian':
        # Gaussian works with any window size >= 2
        return _smooth_gaussian(traj, window_size)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'savgol', 'moving_avg', or 'gaussian'")


def _smooth_savgol(traj: np.ndarray, window: int, polyorder: int) -> Trajectory:
    """Savitzky-Golay filter - preserves peaks and valleys."""
    from scipy.signal import savgol_filter

    # Apply filter with cyclic boundary (wrap mode for closed curves)
    x_smooth = savgol_filter(traj[:, 0], window, polyorder, mode='wrap')
    y_smooth = savgol_filter(traj[:, 1], window, polyorder, mode='wrap')

    return [(float(x), float(y)) for x, y in zip(x_smooth, y_smooth)]


def _smooth_moving_avg(traj: np.ndarray, window: int) -> Trajectory:
    """Simple moving average - aggressive but simple."""
    kernel = np.ones(window) / window

    # Pad for cyclic boundary
    pad = window // 2
    x_padded = np.concatenate([traj[-pad:, 0], traj[:, 0], traj[:pad, 0]])
    y_padded = np.concatenate([traj[-pad:, 1], traj[:, 1], traj[:pad, 1]])

    x_smooth = np.convolve(x_padded, kernel, mode='valid')
    y_smooth = np.convolve(y_padded, kernel, mode='valid')

    return [(float(x), float(y)) for x, y in zip(x_smooth, y_smooth)]


def _smooth_gaussian(traj: np.ndarray, window: int) -> Trajectory:
    """Gaussian-weighted smoothing - natural-looking results."""
    from scipy.ndimage import gaussian_filter1d

    sigma = window / 4  # Approximate window to sigma conversion

    # Use wrap mode for closed curves
    x_smooth = gaussian_filter1d(traj[:, 0], sigma, mode='wrap')
    y_smooth = gaussian_filter1d(traj[:, 1], sigma, mode='wrap')

    return [(float(x), float(y)) for x, y in zip(x_smooth, y_smooth)]


def center_trajectory(traj: np.ndarray) -> np.ndarray:
    """
    Center a trajectory at the origin.

    Used for translation-invariant trajectory comparison.

    Args:
        traj: Trajectory array of shape (n, 2)

    Returns:
        Centered trajectory (same shape as input)
    """
    centroid = np.mean(traj, axis=0)
    return traj - centroid


def find_phase_offset_fft(traj1: np.ndarray, traj2: np.ndarray) -> int:
    """
    Find best phase offset using FFT cross-correlation.

    Args:
        traj1: Reference trajectory, shape (n, 2)
        traj2: Trajectory to align, shape (n, 2)

    Returns:
        Best phase offset (number of positions to shift traj2)
    """
    n = len(traj1)
    if n == 0:
        return 0

    # Treat 2D trajectory as complex signal: z = x + iy
    z1 = traj1[:, 0] + 1j * traj1[:, 1]
    z2 = traj2[:, 0] + 1j * traj2[:, 1]

    # Circular cross-correlation via FFT
    cross_corr = np.fft.ifft(np.fft.fft(z1) * np.conj(np.fft.fft(z2)))

    # Find lag with maximum correlation
    argmax_k = int(np.argmax(np.real(cross_corr)))
    best_offset = (n - argmax_k) % n  # Convert to rotation convention

    return best_offset


def find_phase_offset_rotation(traj1: np.ndarray, traj2: np.ndarray) -> int:
    """
    Find best phase offset by trying all rotations.

    Args:
        traj1: Reference trajectory, shape (n, 2)
        traj2: Trajectory to align, shape (n, 2)

    Returns:
        Best phase offset (number of positions to shift traj2)
    """
    n = len(traj1)
    best_mse = float('inf')
    best_offset = 0

    for offset in range(n):
        t2_rotated = np.roll(traj2, -offset, axis=0)
        diff = traj1 - t2_rotated
        mse = np.mean(np.sum(diff**2, axis=1))

        if mse < best_mse:
            best_mse = mse
            best_offset = offset

    return best_offset


def align_trajectory_phase(
    traj1: np.ndarray,
    traj2: np.ndarray,
    method: Literal['fft', 'rotation'] = 'fft',
) -> np.ndarray:
    """
    Align trajectory2 to trajectory1 by finding best phase offset.

    Args:
        traj1: Reference trajectory, shape (n, 2)
        traj2: Trajectory to align, shape (n, 2)
        method: Phase alignment method ('fft' or 'rotation')

    Returns:
        Aligned trajectory2 (same shape as input)
    """
    if method == 'fft':
        best_offset = find_phase_offset_fft(traj1, traj2)
    elif method == 'rotation':
        best_offset = find_phase_offset_rotation(traj1, traj2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fft' or 'rotation'")

    return np.roll(traj2, -best_offset, axis=0)


def prepare_trajectory_for_optimization(
    trajectory: Trajectory,
    target_n_steps: int,
    smooth: bool = True,
    smooth_window: int = 5,
    smooth_polyorder: int = 3,
    closed: bool = True,
) -> Trajectory:
    """
    Prepare a trajectory for use in optimization.

    This convenience function applies standard preprocessing:
    1. Optional smoothing (if trajectory is noisy)
    2. Resampling to match simulation step count

    Args:
        trajectory: Raw trajectory data
        target_n_steps: Simulation N_STEPS to match
        smooth: Whether to apply smoothing
        smooth_window: Smoothing window size
        smooth_polyorder: Smoothing polynomial order
        closed: If True (default), treat trajectory as closed/cyclic
                (the last point connects back to the first)

    Returns:
        Preprocessed trajectory ready for optimization

    Example:
        >>> raw_target = load_digitized_path("hand_drawn.csv")  # 157 points, noisy
        >>> N_STEPS = 48
        >>> target = prepare_trajectory_for_optimization(raw_target, N_STEPS)
        >>> len(target)  # Now exactly 48 points, smoothed
        48
    """
    result = trajectory

    if smooth and len(trajectory) >= smooth_window:
        result = smooth_trajectory(
            result,
            window_size=smooth_window,
            polyorder=smooth_polyorder,
        )

    if len(result) != target_n_steps:
        result = resample_trajectory(result, target_n_steps, closed=closed)

    return result


def analyze_trajectory(trajectory: Trajectory) -> dict:
    """
    Compute statistics about a trajectory.

    Useful for understanding trajectory properties before optimization.

    Args:
        trajectory: Trajectory to analyze

    Returns:
        Dictionary with trajectory statistics
    """
    traj = np.array(trajectory)
    n = len(traj)

    # Basic stats
    centroid = np.mean(traj, axis=0)

    # Bounding box
    x_min, y_min = np.min(traj, axis=0)
    x_max, y_max = np.max(traj, axis=0)

    # Path length
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(segment_lengths)

    # Closure (how close start is to end)
    closure_gap = np.sqrt(np.sum((traj[0] - traj[-1])**2))

    # Roughness (average change in direction)
    if n >= 3:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        roughness = np.mean(angle_changes)
    else:
        roughness = 0.0

    return {
        'n_points': int(n),
        'centroid': (float(centroid[0]), float(centroid[1])),
        'bounding_box': {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'width': float(x_max - x_min),
            'height': float(y_max - y_min),
        },
        'total_path_length': float(total_length),
        'closure_gap': float(closure_gap),
        'is_closed': bool(closure_gap < total_length * 0.01),  # <1% of path length
        'roughness': float(roughness),
        'avg_segment_length': float(total_length / (n - 1)) if n > 1 else 0.0,
    }


def print_trajectory_info(trajectory: Trajectory, name: str = 'Trajectory') -> None:
    """Print formatted trajectory statistics."""
    stats = analyze_trajectory(trajectory)

    print(f"\n{'='*50}")
    print(f'  {name} Analysis')
    print(f"{'='*50}")
    print(f"  Points:        {stats['n_points']}")
    print(f"  Centroid:      ({stats['centroid'][0]:.2f}, {stats['centroid'][1]:.2f})")
    print(f"  Bounding box:  {stats['bounding_box']['width']:.2f} x {stats['bounding_box']['height']:.2f}")
    print(f"  Path length:   {stats['total_path_length']:.2f}")
    print(f"  Closed curve:  {'Yes' if stats['is_closed'] else 'No'} (gap: {stats['closure_gap']:.4f})")
    print(f"  Roughness:     {stats['roughness']:.4f} rad/segment")
    print(f"{'='*50}\n")
