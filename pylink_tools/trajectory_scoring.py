"""
trajectory_scoring.py - Unified trajectory scoring for optimization.

This module provides a unified function to compute similarity scores between
trajectories. Used by fitness functions during optimization.

Key functions:
  - score_trajectory: Unified scoring function (MSE or distance metrics)

Performance Notes:
  - score_trajectory: Optimized with fast paths for common cases
  - Fast path (FFT + MSE): ~0.03-0.05ms for n=24-48

Design:
  - No mechanism imports (avoids circular dependencies)
  - trajectory_metrics provides error metric functions
  - optimization_types provides TargetTrajectory dataclass
  - All functions use numpy arrays for efficiency (no list conversions)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np

from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.trajectory_utils import align_trajectory_phase
# from pylink_tools.trajectory_metrics import mse
# from pylink_tools.trajectory_metrics import rmse
# from pylink_tools.trajectory_utils import find_phase_offset_rotation
# from functools import lru_cache
# Import distance functions directly from traj_dist.pydist
# We ensure the project root is on sys.path so traj_dist can be found

# Add project root to Python path if not already there
# This ensures traj_dist can be imported regardless of where the script is run from
_project_root = Path(__file__).parent.parent
_project_root_str = str(_project_root.resolve())
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

_pydist_available = False
_import_error = None

try:
    # Import directly from pydist - this will work now that project root is on path
    # and distance.py uses lazy import for linecell (so geohash2 not required)
    from traj_dist.pydist.sspd import e_sspd
    from traj_dist.pydist.dtw import e_dtw
    from traj_dist.pydist.hausdorff import e_hausdorff
    from traj_dist.pydist.frechet import frechet
    from traj_dist.pydist.discret_frechet import discret_frechet
    from traj_dist.pydist.lcss import e_lcss
    from traj_dist.pydist.edr import e_edr
    from traj_dist.pydist.erp import e_erp
    _pydist_available = True
    _import_error = None
except ImportError as e:
    _pydist_available = False
    _import_error = str(e)
except Exception as e:
    _pydist_available = False
    _import_error = f'Unexpected error: {type(e).__name__}: {str(e)}'


def mse(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between trajectories.

    Assumes trajectories are already phase-aligned and pre-processed.

    Args:
        traj1: First trajectory, shape (n, 2)
        traj2: Second trajectory, shape (n, 2)

    Returns:
        MSE value (float) - lower is better
    """
    if len(traj1) != len(traj2):
        raise ValueError(
            f'Trajectories must have same length. Got {len(traj1)} and {len(traj2)}',
        )

    diff = traj1 - traj2
    return float(np.mean(np.sum(diff**2, axis=1)))


def _prepare_trajectories(
    traj1: np.ndarray,
    traj2: np.ndarray,
    phase_invariant: bool,
    phase_align_method: Literal['fft', 'rotation'],
    translation_invariant: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare trajectories for metric computation with optional optimizations.

    Applies phase alignment and/or translation invariance as requested.
    Returns metadata that may contain pre-computed scores for fast paths.

    Args:
        traj1: Reference trajectory, shape (n, 2)
        traj2: Trajectory to compare, shape (n, 2)
        phase_invariant: If True, align traj2 to traj1
        phase_align_method: Phase alignment method ('fft' or 'rotation')
        translation_invariant: If True, center both trajectories

    Returns:
        (prepared_traj1, prepared_traj2, metadata) tuple
        metadata may contain:
            - 'phase_offset': Best phase offset found
            - 'mse_at_best_offset': MSE at best offset (if rotation method used)
    """
    metadata = {}

    # Early return if no preprocessing needed - avoid unnecessary copies
    if not phase_invariant and not translation_invariant:
        return traj1, traj2, metadata

    # Only copy when we need to modify
    t1 = traj1.copy() if translation_invariant else traj1
    t2 = traj2.copy() if (phase_invariant or translation_invariant) else traj2

    # Phase alignment must be done first (before centering)
    if phase_invariant:
        if phase_align_method == 'rotation':
            # Rotation method computes MSE for each offset - capture it
            n = len(t1)
            best_mse = float('inf')
            best_offset = 0

            for offset in range(n):
                t2_rotated = np.roll(t2, -offset, axis=0)
                diff = t1 - t2_rotated
                mse_val = np.mean(np.sum(diff**2, axis=1))

                if mse_val < best_mse:
                    best_mse = mse_val
                    best_offset = offset

            metadata['phase_offset'] = best_offset
            metadata['mse_at_best_offset'] = float(best_mse)
            t2 = np.roll(t2, -best_offset, axis=0)
        else:  # fft
            t2 = align_trajectory_phase(t1, t2, method='fft')

    # Translation invariance (centering)
    # Optimized: compute centroids once and center both trajectories
    if translation_invariant:
        centroid1 = np.mean(t1, axis=0)
        centroid2 = np.mean(t2, axis=0)
        t1 = t1 - centroid1
        t2 = t2 - centroid2

    return t1, t2, metadata


def score_trajectory(
    traj1: np.ndarray | list[tuple[float, float]] | TargetTrajectory,
    traj2: np.ndarray | list[tuple[float, float]],
    metric: str = 'mse',
    *,
    phase_invariant: bool = True,
    phase_align_method: Literal['fft', 'rotation'] = 'fft',
    translation_invariant: bool = False,
    **metric_kwargs,
) -> float:
    """
    Compute similarity score between two trajectories.

    Returns a single score (MSE or distance metric) where lower is better.
    This is the unified function replacing compute_trajectory_error and
    compute_trajectory_distance.

    Args:
        traj1: Reference trajectory. Can be:
            - numpy array (n, 2)
            - list of (x, y) tuples
            - TargetTrajectory (extracts .positions_array)
        traj2: Trajectory to compare. Can be:
            - numpy array (n, 2)
            - list of (x, y) tuples
        metric: Which metric to compute:
            - 'mse': Mean squared error (DEFAULT)
            - Distance metrics: 'sspd', 'dtw', 'hausdorff', 'frechet',
              'frechet_discrete', 'lcss', 'edr', 'erp'
        phase_invariant: If True, find optimal phase alignment first
        phase_align_method: Phase alignment algorithm:
            - 'fft': FFT cross-correlation, O(n log n), fastest (DEFAULT)
            - 'rotation': Brute-force, O(n²), guaranteed optimal
        translation_invariant: If True, center both trajectories before
            comparison, focusing on SHAPE rather than absolute position.
        **metric_kwargs: Additional parameters for specific metrics:
            - eps: Threshold for lcss/edr (default: 20.0)
            - g: Gap parameter for erp (default: zeros array)

    Returns:
        Float score value (lower is better)

    Performance Notes:
        - Fast path optimizations:
          * Rotation + MSE: reuses MSE computed during phase alignment
          * FFT + MSE + no translation: uses optimized FFT path

    Example:
        >>> traj1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        >>> traj2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        >>> score = score_trajectory(traj1, traj2, metric='mse')
        >>> score = score_trajectory(traj1, traj2, metric='sspd', phase_invariant=True)
    """
    # Extract array from TargetTrajectory if needed
    if isinstance(traj1, TargetTrajectory):
        t1 = traj1.positions_array
    elif isinstance(traj1, list):
        t1 = np.array(traj1, dtype=np.float64)
    else:
        t1 = np.asarray(traj1, dtype=np.float64)

    # Convert traj2 to numpy array if needed
    if isinstance(traj2, list):
        t2 = np.array(traj2, dtype=np.float64)
    else:
        t2 = np.asarray(traj2, dtype=np.float64)

    # Early validation
    n = len(t1)
    if n != len(t2):
        return float('inf')

    if n == 0:
        return 0.0

    # Fast path: Rotation method + MSE + no translation invariance
    # Reuse MSE computed during phase alignment
    if (
        metric == 'mse' and
        phase_invariant and
        phase_align_method == 'rotation' and
        not translation_invariant
    ):
        n = len(t1)
        best_mse = float('inf')

        for offset in range(n):
            t2_rotated = np.roll(t2, -offset, axis=0)
            diff = t1 - t2_rotated
            mse_val = np.mean(np.sum(diff**2, axis=1))

            if mse_val < best_mse:
                best_mse = mse_val

        return float(best_mse)

    # Fast path: FFT method + MSE + no translation invariance
    # Use optimized FFT path (similar to compute_trajectory_mse_hot)
    if (
        metric == 'mse' and
        phase_invariant and
        phase_align_method == 'fft' and
        not translation_invariant
    ):
        # FFT cross-correlation to find best phase offset
        z1 = t1[:, 0] + 1j * t1[:, 1]
        z2 = t2[:, 0] + 1j * t2[:, 1]
        cross_corr = np.fft.ifft(np.fft.fft(z1) * np.conj(np.fft.fft(z2)))
        best_offset = (n - int(np.argmax(cross_corr.real))) % n
        t2_aligned = np.roll(t2, -best_offset, axis=0)
        diff = t1 - t2_aligned
        return float(np.einsum('ij,ij->', diff, diff) / n)

    # General path: prepare trajectories and compute metric
    t1_prep, t2_prep, metadata = _prepare_trajectories(
        t1, t2, phase_invariant, phase_align_method, translation_invariant,
    )

    # Check if we can short-circuit using metadata
    if (
        metric == 'mse' and
        'mse_at_best_offset' in metadata and
        not translation_invariant
    ):
        return metadata['mse_at_best_offset']

    # Compute MSE metric
    if metric == 'mse':
        return mse(t1_prep, t2_prep)

    # Compute distance metrics
    if not _pydist_available:
        error_msg = 'traj_dist.pydist package is required'
        if '_import_error' in globals():
            error_msg += f': {_import_error}'
        raise ImportError(error_msg)

    # Ensure arrays are properly formatted (contiguous, float64, 2D)
    if not (t1_prep.flags['C_CONTIGUOUS'] and t1_prep.dtype == np.float64):
        t1_prep = np.ascontiguousarray(t1_prep, dtype=np.float64)
    if not (t2_prep.flags['C_CONTIGUOUS'] and t2_prep.dtype == np.float64):
        t2_prep = np.ascontiguousarray(t2_prep, dtype=np.float64)

    # Validate array shapes
    if len(t1_prep.shape) != 2 or t1_prep.shape[1] != 2:
        raise ValueError(f'traj1 must be shape (n, 2), got {t1_prep.shape}')
    if len(t2_prep.shape) != 2 or t2_prep.shape[1] != 2:
        raise ValueError(f'traj2 must be shape (n, 2), got {t2_prep.shape}')

    try:
        if metric == 'sspd':
            return float(e_sspd(t1_prep, t2_prep))
        elif metric == 'dtw':
            return float(e_dtw(t1_prep, t2_prep))
        elif metric == 'hausdorff':
            return float(e_hausdorff(t1_prep, t2_prep))
        elif metric == 'frechet':
            try:
                return float(frechet(t1_prep, t2_prep))
            except (TypeError, IndexError, ValueError) as e:
                if 'list indices' in str(e) or 'must be integers' in str(e):
                    return float('inf')
                raise
        elif metric == 'frechet_discrete':
            return float(discret_frechet(t1_prep, t2_prep))
        elif metric == 'lcss':
            eps = metric_kwargs.get('eps', 20.0)
            return float(e_lcss(t1_prep, t2_prep, eps))
        elif metric == 'edr':
            eps = metric_kwargs.get('eps', 20.0)
            return float(e_edr(t1_prep, t2_prep, eps))
        elif metric == 'erp':
            g = metric_kwargs.get('g', None)
            if g is None:
                g = np.zeros(2, dtype=np.float64)
            return float(e_erp(t1_prep, t2_prep, g))
        else:
            raise ValueError(
                f'Unknown metric: {metric}. '
                f"Use 'mse' or distance metrics: 'sspd', 'dtw', 'hausdorff', "
                f"'frechet', 'frechet_discrete', 'lcss', 'edr', 'erp'",
            )
    except Exception as e:
        raise RuntimeError(
            f'Error computing {metric} score: {type(e).__name__}: {str(e)}',
        ) from e
