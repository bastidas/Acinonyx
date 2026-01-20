"""
optimize.py - Trajectory optimization for linkage mechanisms.

Core functionality:
  - Compute error between computed trajectory and target trajectory
  - Run PSO optimization to fit linkage to target path
  - Convergence analysis and reporting

Design notes:
  - Static joints are fixed (not optimizable)
  - Crank distance and Revolute distances are optimizable
  - Crank angle is NOT optimized (it's the driver)
  - Compatible with pylinkage's optimization API

CRITICAL PARAMETERS (see trajectory_utils.py for full documentation):

  N_STEPS (Simulation Step Count):
    - Higher = better accuracy, slower optimization
    - Target trajectory MUST have same point count as N_STEPS
    - Use trajectory_utils.resample_trajectory() to match counts
    - Recommended: 24-48 for optimization, 48-96 for final results

  PHASE_ALIGNMENT:
    - Trajectories may trace same path but start at different points
    - Without alignment: identical paths can have HUGE errors
    - Use phase_invariant=True in error computation for robust scoring
    - See trajectory_utils.compute_phase_aligned_distance()
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Literal

import numpy as np

from configs.logging_config import get_logger
from pylink_tools.optimization_helpers import apply_dimensions
from pylink_tools.optimization_helpers import apply_dimensions_from_array
from pylink_tools.optimization_helpers import dict_to_dimensions
from pylink_tools.optimization_helpers import dimensions_to_dict
from pylink_tools.optimization_helpers import extract_dimensions
from pylink_tools.optimization_helpers import extract_dimensions_with_custom_bounds
from pylink_tools.optimization_helpers import presample_valid_positions
from pylink_tools.optimization_helpers import validate_bounds
from pylink_tools.optimization_types import ConvergenceStats
from pylink_tools.optimization_types import DimensionSpec
from pylink_tools.optimization_types import ErrorMetrics
from pylink_tools.optimization_types import OptimizationResult
from pylink_tools.optimization_types import TargetTrajectory

# Module logger
logger = get_logger(__name__)

# Import types and helpers from refactored modules

# Re-export for backwards compatibility
__all__ = [
    # Types
    'DimensionSpec',
    'TargetTrajectory',
    'OptimizationResult',
    'ErrorMetrics',
    'ConvergenceStats',
    # Helper functions
    'extract_dimensions',
    'extract_dimensions_with_custom_bounds',
    'apply_dimensions',
    'apply_dimensions_from_array',
    'dimensions_to_dict',
    'dict_to_dimensions',
    'validate_bounds',
    # Error computation
    'compute_trajectory_error',
    'compute_trajectory_error_detailed',
    'evaluate_linkage_fit',
    # Convergence
    'analyze_convergence',
    'format_convergence_report',
    'log_optimization_progress',
    # Optimization
    'create_fitness_function',
    'create_pylinkage_fitness_function',
    'presample_valid_positions',
    'run_scipy_optimization',
    'run_pso_optimization',
    'run_pylinkage_pso',
    'optimize_trajectory',
]


# =============================================================================
# Error Computation
# =============================================================================

def compute_trajectory_error(
    computed_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    metric: str = 'mse',
) -> float:
    """
    Compute error between computed trajectory and target.

    Args:
        computed_trajectory: List of (x, y) positions from simulation
        target: TargetTrajectory with target positions and optional weights
        metric: Error metric to return:
                - "mse": Mean squared error (default)
                - "rmse": Root mean squared error
                - "total": Total squared error (sum)
                - "max": Maximum error at any timestep

    Returns:
        Error value (lower is better, 0 is perfect match)

    Raises:
        ValueError: If trajectories have different lengths
    """
    if len(computed_trajectory) != len(target.positions):
        raise ValueError(
            f'Trajectory length mismatch: computed={len(computed_trajectory)}, '
            f'target={len(target.positions)}',
        )

    metrics = compute_trajectory_error_detailed(computed_trajectory, target)

    if metric == 'mse':
        return metrics.mse
    elif metric == 'rmse':
        return metrics.rmse
    elif metric == 'total':
        return metrics.total_error
    elif metric == 'max':
        return metrics.max_error
    else:
        raise ValueError(f'Unknown metric: {metric}')


# =============================================================================
# Core Error Computation (DRY - shared implementation)
# =============================================================================

def _compute_errors_core(
    computed_trajectory: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
    weights: list[float] | None = None,
) -> tuple[float, float, float, float, list[float]]:
    """
    Core error computation - shared by fast and detailed functions.

    Args:
        computed_trajectory: List of (x, y) computed positions
        target_positions: List of (x, y) target positions (must be same length)
        weights: Optional weights per point (defaults to uniform)

    Returns:
        Tuple of (total_error, mse, rmse, max_error, per_step_errors)
    """
    n = len(computed_trajectory)
    weights = weights or [1.0] * n

    per_step_errors = []
    weighted_squared_sum = 0.0
    total_weight = 0.0
    max_error = 0.0

    for i, (computed, target_pos) in enumerate(zip(computed_trajectory, target_positions)):
        dx = computed[0] - target_pos[0]
        dy = computed[1] - target_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)

        per_step_errors.append(dist)
        weighted_squared_sum += weights[i] * (dist ** 2)
        total_weight += weights[i]
        if dist > max_error:
            max_error = dist

    mse = weighted_squared_sum / total_weight if total_weight > 0 else 0.0
    rmse = np.sqrt(mse)

    return weighted_squared_sum, mse, rmse, max_error, per_step_errors


def _apply_phase_alignment(
    computed_trajectory: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
    return_aligned: bool = False,
    method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
) -> tuple[float, list[tuple[float, float]] | None]:
    """
    Apply phase alignment and return MSE + optionally aligned trajectory.

    Args:
        computed_trajectory: Computed positions
        target_positions: Target positions
        return_aligned: If True, also return the aligned trajectory
        method: Phase alignment method:
            - 'rotation': Brute-force all rotations, O(n²), guaranteed optimal
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!

    Returns:
        (mse, aligned_trajectory or None)
    """
    from pylink_tools.trajectory_utils import compute_phase_aligned_distance

    result = compute_phase_aligned_distance(
        target_positions,
        computed_trajectory,
        method=method,
        return_aligned=return_aligned,
    )
    return result.distance, result.aligned_trajectory


def compute_trajectory_error_fast(
    computed_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    metric: str = 'mse',
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
) -> float:
    """
    Fast error computation for optimization hot loops.

    This is an optimized version that:
    - Uses phase alignment MSE directly (no recomputation)
    - Skips per-step error computation when possible
    - Only computes the requested metric
    - Avoids list/object creation overhead

    ~15-30% faster than compute_trajectory_error_detailed for MSE scoring.

    Args:
        computed_trajectory: List of (x, y) positions from simulation
        target: TargetTrajectory with target positions
        metric: Which metric to compute:
            - 'mse': Mean squared error (DEFAULT, most common)
            - 'rmse': Root mean squared error
            - 'total': Sum of weighted squared errors
            - 'max': Maximum single-point error
        phase_invariant: If True, find optimal phase alignment
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!

    Returns:
        Float error value for the requested metric

    Performance Notes:
        For optimization, 'mse' with phase_invariant=True is fast because
        phase alignment directly returns MSE - no recomputation needed!

    Note:
        For detailed analysis (per-step errors, all metrics), use
        compute_trajectory_error_detailed instead.
    """
    n = len(computed_trajectory)
    if n != len(target.positions):
        return float('inf')

    if n == 0:
        return 0.0

    # FAST PATH for phase-invariant MSE/RMSE/total:
    # Phase alignment directly computes MSE, just use it!
    if phase_invariant and metric in ('mse', 'rmse', 'total'):
        mse, _ = _apply_phase_alignment(
            computed_trajectory, target.positions,
            return_aligned=False, method=phase_align_method,
        )
        if metric == 'mse':
            return mse
        elif metric == 'rmse':
            return np.sqrt(mse)
        else:  # total
            # Approximate: MSE * n (assumes uniform weights)
            return mse * n

    # For 'max' or non-phase-invariant: need to compute per-step errors
    if phase_invariant:
        _, aligned = _apply_phase_alignment(
            computed_trajectory, target.positions,
            return_aligned=True, method=phase_align_method,
        )
        computed_trajectory = aligned

    total, mse, rmse, max_error, _ = _compute_errors_core(
        computed_trajectory, target.positions, target.weights,
    )

    if metric == 'max':
        return max_error
    elif metric == 'mse':
        return mse
    elif metric == 'rmse':
        return rmse
    elif metric == 'total':
        return total
    else:
        return mse  # Default to MSE


def compute_trajectory_error_detailed(
    computed_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
) -> ErrorMetrics:
    """
    Compute detailed error metrics between computed and target trajectory.

    Args:
        computed_trajectory: List of (x, y) positions from simulation
        target: TargetTrajectory with target positions and weights
        phase_invariant: If True, find optimal phase alignment before scoring.
            Use this when trajectories may start at different points!
            Without this, identical paths can have huge errors if out of phase.
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), slow but handles speed variation

    Returns:
        ErrorMetrics with total, mse, rmse, max, and per-step errors

    Note:
        If trajectories have different lengths, use
        trajectory_utils.resample_trajectory() first.
    """
    n = len(computed_trajectory)
    if n != len(target.positions):
        raise ValueError(
            f'Trajectory length mismatch: computed={n}, target={len(target.positions)}. '
            f'Use trajectory_utils.resample_trajectory() to match lengths.',
        )

    if n == 0:
        return ErrorMetrics(
            total_error=0.0,
            mse=0.0,
            rmse=0.0,
            max_error=0.0,
            per_step_errors=[],
        )

    # Apply phase alignment if requested
    if phase_invariant:
        _, aligned = _apply_phase_alignment(
            computed_trajectory,
            target.positions,
            return_aligned=True,
            method=phase_align_method,
        )
        computed_trajectory = aligned

    # Use shared core computation
    total_error, mse, rmse, max_error, per_step_errors = _compute_errors_core(
        computed_trajectory, target.positions, target.weights,
    )

    return ErrorMetrics(
        total_error=total_error,
        mse=mse,
        rmse=rmse,
        max_error=max_error,
        per_step_errors=per_step_errors,
    )


def evaluate_linkage_fit(
    pylink_data: dict,
    target: TargetTrajectory,
    n_steps: int | None = None,
    phase_invariant: bool = True,
    _skip_copy: bool = False,
    _metric: str | None = None,
    _phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
) -> ErrorMetrics | float:
    """
    Evaluate how well a linkage fits a target trajectory.

    This is a convenience function that:
      1. Computes the trajectory for the linkage
      2. Extracts the target joint's path
      3. Computes error metrics (optionally phase-aligned)

    Args:
        pylink_data: Full pylink document
        target: Target trajectory to match
        n_steps: Number of simulation steps (uses target.n_steps if not provided)
        phase_invariant: If True, find optimal phase alignment before scoring.
            RECOMMENDED for real-world targets where starting point may differ.
        _skip_copy: Internal use only. If True, skip deepcopy for performance.
            Only use this when pylink_data is a dedicated working copy that
            can be safely mutated (e.g., in optimization hot loops).
        _metric: Internal use only. If set, use fast path and return only
            the requested metric as a float (for optimization hot loops).
            Values: 'mse', 'rmse', 'total', 'max'. ~30% faster than full path.
        _phase_align_method: Internal use only. Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!

    Returns:
        ErrorMetrics describing the fit quality, OR
        float if _metric is specified (fast path for optimization)

    Note:
        n_steps is CRITICAL - it determines simulation resolution and MUST
        match target.n_steps for accurate scoring. Use a higher value (48-96)
        for better accuracy at the cost of slower optimization.
    """
    from pylink_tools.kinematic import compute_trajectory

    # Set n_steps if not provided
    if n_steps is None:
        n_steps = target.n_steps

    # Update n_steps in pylink_data
    if _skip_copy:
        # Performance path: assume pylink_data is a reusable working copy
        eval_data = pylink_data
        eval_data['n_steps'] = n_steps
    else:
        # Safe path: deep copy to avoid mutation
        eval_data = copy.deepcopy(pylink_data)
        eval_data['n_steps'] = n_steps

    # CRITICAL: Use skip_sync=True to use the dimension values we set,
    # not overwrite them from stale meta.joints visual positions!
    result = compute_trajectory(
        eval_data,
        verbose=False,
        skip_sync=True,
    )

    if not result.success:
        # Return infinite error if simulation fails
        if _metric is not None:
            return float('inf')
        return ErrorMetrics(
            total_error=float('inf'),
            mse=float('inf'),
            rmse=float('inf'),
            max_error=float('inf'),
            per_step_errors=[float('inf')] * n_steps,
        )

    # Get target joint trajectory
    joint_name = target.joint_name
    if joint_name not in result.trajectories:
        raise ValueError(f"Target joint '{joint_name}' not found in trajectories")

    computed = result.trajectories[joint_name]
    # Convert from [[x,y], ...] to [(x,y), ...]
    computed_tuples = [tuple(pos) for pos in computed]

    # Fast path: only compute requested metric (for optimization hot loops)
    # ~30% faster by avoiding per_step_errors list, rmse, max computations
    if _metric is not None:
        return compute_trajectory_error_fast(
            computed_tuples, target,
            metric=_metric,
            phase_invariant=phase_invariant,
            phase_align_method=_phase_align_method,
        )

    # Full path: compute all metrics
    return compute_trajectory_error_detailed(
        computed_tuples,
        target,
        phase_invariant=phase_invariant,
        phase_align_method=_phase_align_method,
    )


# =============================================================================
# Convergence Logging / Analysis
# =============================================================================

def analyze_convergence(
    history: list[float],
    tolerance: float = 1e-6,
) -> ConvergenceStats:
    """
    Analyze convergence history from an optimization run.

    Args:
        history: List of error values (one per iteration)
        tolerance: Convergence tolerance

    Returns:
        ConvergenceStats with analysis results
    """
    if not history:
        return ConvergenceStats(
            initial_error=0.0,
            final_error=0.0,
            best_error=0.0,
            improvement_pct=0.0,
            n_iterations=0,
            n_evaluations=0,
            converged=False,
            history=[],
            improvement_per_iteration=[],
        )

    initial = history[0]
    final = history[-1]
    best = min(history)

    # Handle inf/nan
    if initial == float('inf') or np.isnan(initial):
        improvement_pct = 0.0
    elif initial == 0:
        improvement_pct = 0.0 if final == 0 else -100.0
    else:
        improvement_pct = (1 - final / initial) * 100

    # Compute per-iteration improvement
    improvements = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if prev == float('inf') or curr == float('inf'):
            improvements.append(0.0)
        elif prev == 0:
            improvements.append(0.0)
        else:
            improvements.append(prev - curr)

    # Check if converged (change < tolerance)
    converged = False
    if len(history) >= 2:
        recent_change = abs(history[-1] - history[-2])
        converged = recent_change < tolerance

    return ConvergenceStats(
        initial_error=initial,
        final_error=final,
        best_error=best,
        improvement_pct=improvement_pct,
        n_iterations=len(history) - 1,  # First entry is initial state
        n_evaluations=len(history),
        converged=converged,
        history=history,
        improvement_per_iteration=improvements,
    )


def format_convergence_report(
    result: OptimizationResult,
    include_history: bool = False,
) -> str:
    """
    Format a human-readable convergence report.

    Args:
        result: OptimizationResult from an optimization run
        include_history: Include full iteration history

    Returns:
        Formatted string report
    """
    lines = [
        '=' * 50,
        'OPTIMIZATION CONVERGENCE REPORT',
        '=' * 50,
        f"Status: {'SUCCESS' if result.success else 'FAILED'}",
        f'Initial Error: {result.initial_error:.6f}',
        f'Final Error:   {result.final_error:.6f}',
    ]

    if result.initial_error > 0 and result.initial_error != float('inf'):
        improvement = (1 - result.final_error / result.initial_error) * 100
        lines.append(f'Improvement:   {improvement:.1f}%')

    lines.append(f'Iterations:    {result.iterations}')

    if result.optimized_dimensions:
        lines.append('')
        lines.append('Optimized Dimensions:')
        for name, value in result.optimized_dimensions.items():
            lines.append(f'  {name}: {value:.4f}')

    if result.convergence_history and include_history:
        lines.append('')
        lines.append('Convergence History:')
        for i, err in enumerate(result.convergence_history):
            lines.append(f'  [{i:3d}] {err:.6f}')

    if result.error:
        lines.append('')
        lines.append(f'Error: {result.error}')

    lines.append('=' * 50)
    return '\n'.join(lines)


def log_optimization_progress(
    iteration: int,
    current_error: float,
    best_error: float,
    dimensions: tuple[float, ...] | None = None,
    dimension_names: list[str] | None = None,
) -> str:
    """
    Format a single iteration's progress for logging.

    Args:
        iteration: Current iteration number
        current_error: Error at this iteration
        best_error: Best error seen so far
        dimensions: Current dimension values (optional)
        dimension_names: Names for dimensions (optional)

    Returns:
        Formatted progress string
    """
    parts = [f'[{iteration:4d}]', f'error={current_error:.6f}', f'best={best_error:.6f}']

    if dimensions is not None and dimension_names is not None:
        dim_str = ', '.join(f'{n}={v:.2f}' for n, v in zip(dimension_names[:3], dimensions[:3]))
        if len(dimensions) > 3:
            dim_str += '...'
        parts.append(f'dims=({dim_str})')

    return ' | '.join(parts)


# =============================================================================
# Optimization Wrappers
# =============================================================================

def create_fitness_function(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    metric: str = 'mse',
    verbose: bool = False,
    phase_invariant: bool = False,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    use_cache: bool = False,
) -> Callable[[tuple[float, ...]], float]:
    """
    Create a fitness function for optimization.

    Returns a callable that takes dimension values and returns error score.
    This is the core objective function used by all optimizers.

    Args:
        pylink_data: Base pylink document (will be modified with new dimensions)
        target: Target trajectory to match
        dimension_spec: Specification of optimizable dimensions
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: If True, print evaluation info
        phase_invariant: If True, find optimal phase alignment before scoring.
            RECOMMENDED when target may have different starting point than
            simulation. Adds O(n) overhead per evaluation but prevents
            false errors from phase mismatch.
        phase_align_method: Phase alignment algorithm (only used if phase_invariant=True):
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        use_cache: If True, cache results with LRU cache (useful when optimizer
            may revisit same points). Cache uses rounded dimension values.

    Returns:
        Callable that takes dimension tuple and returns float error

    Example:
        >>> fitness = create_fitness_function(pylink_data, target, spec)
        >>> error = fitness((20.0, 50.0, 40.0))  # Evaluate with these dimensions

    Note on phase_invariant:
        Without phase alignment, two identical paths starting at different
        points can have HUGE errors (e.g., MSE=200 for a 90° phase shift).
        Use phase_invariant=True when:
        - Target comes from external source (captured data, hand-drawn)
        - Target was generated with different initial conditions
        - You're not sure about phase alignment

    Performance note:
        This function creates a reusable working copy of pylink_data and uses
        in-place mutations to avoid expensive deepcopy operations on every
        evaluation. This can provide 2-3x speedup for optimization.
    """
    eval_count = [0]  # Mutable counter for tracking evaluations
    cache_hits = [0]  # Track cache hit rate

    # Create a single working copy for in-place mutations (OPTIMIZATION)
    # This avoids 2x deepcopy per evaluation that was happening before
    working_copy = copy.deepcopy(pylink_data)

    # Detect format: hypergraph (linkage.edges) vs legacy (pylinkage.joints)
    is_hypergraph = 'linkage' in working_copy and 'edges' in working_copy.get('linkage', {})

    # Cache for memoization (optional)
    result_cache: dict[tuple, float] = {} if use_cache else None

    if is_hypergraph:
        # Hypergraph format: dimensions map to edge distances
        # Use edge_mapping if available, fall back to joint_mapping for backwards compatibility
        mapping = getattr(dimension_spec, 'edge_mapping', None) or dimension_spec.joint_mapping
        edges_data = working_copy['linkage']['edges']

        def _apply_dimensions_fast(dimensions: tuple[float, ...]) -> None:
            """Apply dimensions in-place to working_copy edges (hypergraph path)."""
            for i, dim_name in enumerate(dimension_spec.names):
                if dim_name in mapping:
                    edge_id, prop_name = mapping[dim_name]
                    if edge_id in edges_data:
                        edges_data[edge_id][prop_name] = dimensions[i]
    else:
        # Legacy format: dimensions map to joint properties
        mapping = dimension_spec.joint_mapping
        joints_data = working_copy.get('pylinkage', {}).get('joints', [])
        joint_by_name = {j['name']: j for j in joints_data}

        def _apply_dimensions_fast(dimensions: tuple[float, ...]) -> None:
            """Apply dimensions in-place to working_copy joints (legacy path)."""
            for i, dim_name in enumerate(dimension_spec.names):
                if dim_name in mapping:
                    joint_name, prop_name = mapping[dim_name]
                    if joint_name in joint_by_name:
                        joint_by_name[joint_name][prop_name] = dimensions[i]

    def fitness(dimensions: tuple[float, ...]) -> float:
        """
        Evaluate linkage fitness with given dimensions.

        Args:
            dimensions: Tuple of dimension values in spec order

        Returns:
            Error score (lower is better, inf for invalid configurations)
        """
        eval_count[0] += 1

        # Check cache if enabled
        if result_cache is not None:
            # Round to 6 decimal places for cache key
            cache_key = tuple(round(d, 6) for d in dimensions)
            if cache_key in result_cache:
                cache_hits[0] += 1
                return result_cache[cache_key]

        try:
            # Fast in-place dimension application
            _apply_dimensions_fast(dimensions)

            # Evaluate fit using fast path - only computes requested metric
            # This uses compute_trajectory_error_fast which:
            # - Uses phase alignment MSE directly (no recomputation)
            # - Skips per_step_errors list creation
            # - Avoids rmse/max computation when not needed
            # ~30% faster than full metrics computation
            error = evaluate_linkage_fit(
                working_copy,
                target,
                phase_invariant=phase_invariant,
                _skip_copy=True,  # Performance: use working copy directly
                _metric=metric,   # Performance: only compute requested metric
                _phase_align_method=phase_align_method,
            )

            if verbose and eval_count[0] % 50 == 0:
                cache_info = f', cache_hits={cache_hits[0]}' if result_cache is not None else ''
                logger.debug(f'Eval #{eval_count[0]}: error={error:.4f}, dims={dimensions[:3]}...{cache_info}')

            # Store in cache if enabled
            if result_cache is not None:
                cache_key = tuple(round(d, 6) for d in dimensions)
                result_cache[cache_key] = error

            return error

        except Exception as e:
            if verbose:
                logger.warning(f'Eval #{eval_count[0]} failed: {e}')
            return float('inf')

    # Attach metadata to function
    fitness.eval_count = eval_count
    fitness.cache_hits = cache_hits
    fitness.dimension_spec = dimension_spec
    fitness.target = target
    fitness.phase_invariant = phase_invariant
    fitness.phase_align_method = phase_align_method
    fitness.result_cache = result_cache

    return fitness


def create_pylinkage_fitness_function(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    metric: str = 'mse',
) -> Callable:
    """
    Create a fitness function compatible with pylinkage's optimization API.

    This wraps our fitness function with pylinkage's @kinematic_minimization
    decorator pattern for use with pl.particle_swarm_optimization.

    Note: This requires building a Linkage object and is more complex.
    For simpler use, prefer run_scipy_optimization().

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Specification of optimizable dimensions
        metric: Error metric

    Returns:
        Fitness function compatible with pylinkage PSO
    """
    import pylinkage as pl

    @pl.kinematic_minimization
    def fitness_func(loci, **_kwargs):
        """
        Compute error between loci and target trajectory.

        loci: List of joint trajectories, each is list of (x, y) positions
        """
        # Find target joint in loci
        # loci is ordered by linkage.joints order (non-static joints only)
        # We need to figure out which index corresponds to our target joint

        # Get trajectory for last joint (typically the output)
        # In a 4-bar, this is usually the coupler point
        target_locus = []
        for step_positions in loci:
            # step_positions is ((x1,y1), (x2,y2), ...) for all joints at this step
            # We need the position of our target joint
            # For now, use the last non-static joint
            if step_positions and len(step_positions) > 0:
                target_locus.append(step_positions[-1])

        if len(target_locus) != len(target.positions):
            return float('inf')

        # Compute error
        total_error = 0.0
        weights = target.weights or [1.0] * len(target.positions)

        for i, (computed, target_pos) in enumerate(zip(target_locus, target.positions)):
            if computed is None or computed[0] is None:
                return float('inf')
            dx = computed[0] - target_pos[0]
            dy = computed[1] - target_pos[1]
            total_error += weights[i] * (dx * dx + dy * dy)

        if metric == 'mse':
            return total_error / len(target.positions)
        elif metric == 'total':
            return total_error
        else:
            return total_error / len(target.positions)

    return fitness_func


def run_scipy_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    method: str = 'L-BFGS-B',
    metric: str = 'mse',
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
) -> OptimizationResult:
    """
    Run optimization using scipy.optimize.minimize.

    This is often faster than PSO for well-behaved problems and supports
    gradient-based methods like L-BFGS-B.

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        method: Scipy optimizer method ("L-BFGS-B", "SLSQP", "Powell", "Nelder-Mead")
        metric: Error metric for fitness function
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Print progress
        phase_invariant: If True, use phase-aligned scoring
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!

    Returns:
        OptimizationResult with optimized dimensions

    Example:
        >>> result = run_scipy_optimization(pylink_data, target)
        >>> if result.success:
        ...     print(f"Final error: {result.final_error}")
        ...     print(f"Optimized: {result.optimized_dimensions}")
    """
    from scipy.optimize import minimize

    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Create fitness function
    fitness = create_fitness_function(
        pylink_data, target, dimension_spec, metric=metric, verbose=verbose,
        phase_invariant=phase_invariant, phase_align_method=phase_align_method,
    )

    # Compute initial error
    initial_values = tuple(dimension_spec.initial_values)
    initial_error = fitness(initial_values)

    if verbose:
        logger.info(f'Starting scipy optimization ({method})')
        logger.info(f'  Dimensions: {len(dimension_spec)}')
        logger.info(f'  Initial error: {initial_error:.4f}')

    # Convert bounds to scipy format: [(low, high), ...]
    scipy_bounds = dimension_spec.bounds

    # Track convergence history
    history = [initial_error]

    def callback(xk):
        error = fitness(tuple(xk))
        history.append(error)
        if verbose and len(history) % 10 == 0:
            logger.info(f'  Iteration {len(history)}: error={error:.4f}')

    # Run optimization
    try:
        # Build options dict based on method
        options = {'maxiter': max_iterations}

        # Different methods use different tolerance parameter names
        if method in ('L-BFGS-B', 'SLSQP'):
            options['ftol'] = tolerance
        elif method == 'Powell':
            options['ftol'] = tolerance
        elif method == 'Nelder-Mead':
            options['xatol'] = tolerance
            options['fatol'] = tolerance

        result = minimize(
            fun=fitness,
            x0=initial_values,
            method=method,
            bounds=scipy_bounds if method in ('L-BFGS-B', 'SLSQP') else None,
            options=options,
            callback=callback,
        )

        # Extract results
        optimized_values = tuple(result.x)
        final_error = result.fun

        # Build optimized pylink_data
        optimized_pylink_data = apply_dimensions_from_array(
            pylink_data, optimized_values, dimension_spec,
        )

        # Create dimension dict
        optimized_dims = dimensions_to_dict(optimized_values, dimension_spec)

        if verbose:
            logger.info(f'  Converged: {result.success}')
            logger.info(f'  Final error: {final_error:.4f}')
            logger.info(f'  Iterations: {result.nit}')
            logger.info(f'  Function evals: {result.nfev}')

        return OptimizationResult(
            success=result.success,
            optimized_dimensions=optimized_dims,
            optimized_pylink_data=optimized_pylink_data,
            initial_error=initial_error,
            final_error=final_error,
            iterations=result.nit,
            convergence_history=history,
        )

    except Exception as e:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'Optimization failed: {str(e)}',
        )


def run_pso_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    n_particles: int = 32,
    iterations: int = 512,
    metric: str = 'mse',
    verbose: bool = False,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    init_mode: str = 'random',
    init_samples: int = 128,
) -> OptimizationResult:
    """
    Run Particle Swarm Optimization to fit linkage to target trajectory.

    PSO is a population-based optimizer that's good for:
      - Non-convex problems
      - Escaping local minima
      - Problems where gradients are unavailable

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        n_particles: Number of particles in swarm
        iterations: Number of PSO iterations
        metric: Error metric for fitness function
        verbose: Print progress
        phase_invariant: If True, use phase-aligned scoring.
            RECOMMENDED when target may start at different phase than simulation.
            Prevents false errors from phase mismatch at cost of O(n) per eval.
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        init_mode: Particle initialization strategy:
            - 'random': Random positions in bounds (default, original behavior)
            - 'sobol': Pre-sample using Sobol sequence, filter valid, use best
            - 'behnken': Pre-sample using Box-Behnken design (requires 3+ dims)
        init_samples: Number of samples to generate for presampling modes

    Returns:
        OptimizationResult with optimized dimensions

    Example:
        >>> # Standard random initialization
        >>> result = run_pso_optimization(pylink_data, target, n_particles=50)
        >>> # With Sobol presampling for better convergence
        >>> result = run_pso_optimization(
        ...     pylink_data, target, n_particles=50,
        ...     init_mode='sobol', init_samples=256
        ... )

    Note on phase_invariant:
        Set to True when working with:
        - External/captured target trajectories
        - Hand-drawn or digitized paths
        - Any case where starting point alignment is uncertain

    Note on init_mode:
        Using 'sobol' or 'behnken' presampling can dramatically improve
        convergence for constrained mechanisms where random positions
        often produce invalid (infinite error) configurations.
    """
    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    logger.info(f'Starting PSO optimization with phase_invariant={phase_invariant}')
    logger.info(f'  Dimensions: {len(dimension_spec)}')
    logger.info(f'  Particles: {n_particles}')
    logger.info(f'  Iterations: {iterations}')
    logger.info(f'  Metric: {metric}')
    logger.info(f'  Phase invariant: {phase_invariant}')
    logger.info(f'  Init mode: {init_mode}')

    # Create fitness function
    fitness = create_fitness_function(
        pylink_data, target, dimension_spec,
        metric=metric, verbose=verbose,
        phase_invariant=phase_invariant, phase_align_method=phase_align_method,
    )

    # Compute initial error
    initial_values = tuple(dimension_spec.initial_values)
    initial_error = fitness(initial_values)

    if verbose:
        logger.info('Starting PSO optimization')
        logger.info(f'  Dimensions: {len(dimension_spec)}')
        logger.info(f'  Particles: {n_particles}')
        logger.info(f'  Iterations: {iterations}')
        logger.info(f'  Initial error: {initial_error:.4f}')

    # Get bounds
    bounds = dimension_spec.get_bounds_tuple()

    # Pre-sample positions if using advanced init mode
    init_positions = None
    if init_mode in ('sobol', 'behnken'):
        try:
            init_positions, init_scores = presample_valid_positions(
                pylink_data=pylink_data,
                target=target,
                dimension_spec=dimension_spec,
                n_samples=init_samples,
                n_best=n_particles,
                mode=init_mode,
                metric=metric,
                phase_invariant=phase_invariant,
            )
            if verbose and len(init_positions) > 0:
                logger.info(f'  Pre-sampled {len(init_positions)} valid positions')
                logger.info(f'  Best pre-sample score: {init_scores[0]:.4f}')
        except Exception as e:
            logger.warning(f'Presampling failed: {e}. Falling back to random init.')
            init_positions = None

    # Run PSO
    try:
        best_score, best_dims, history = _run_pso_internal(
            fitness_func=fitness,
            initial_values=initial_values,
            bounds=bounds,
            n_particles=n_particles,
            iterations=iterations,
            verbose=verbose,
            init_positions=init_positions,
        )

        # Build optimized pylink_data
        optimized_pylink_data = apply_dimensions_from_array(
            pylink_data, best_dims, dimension_spec,
        )

        # Create dimension dict
        optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

        if verbose:
            logger.info(f'  Final error: {best_score:.4f}')
            logger.info(f'  Improvement: {(1 - best_score/initial_error)*100:.1f}%')

        return OptimizationResult(
            success=True,
            optimized_dimensions=optimized_dims,
            optimized_pylink_data=optimized_pylink_data,
            initial_error=initial_error,
            final_error=best_score,
            iterations=iterations,
            convergence_history=history,
        )

    except Exception as e:
        import traceback
        if verbose:
            logger.error(f'PSO optimization failed: {e}', exc_info=True)
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'PSO optimization failed: {str(e)}',
        )


def run_pylinkage_pso(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    n_particles: int = 32,
    iterations: int = 512,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    init_mode: str = 'random',
    init_samples: int = 128,
) -> OptimizationResult:
    """
    Run Particle Swarm Optimization using pylinkage's native PSO implementation.

    This uses pylinkage.particle_swarm_optimization with the
    @kinematic_minimization decorator, which handles:
      - Building and validating the linkage at each iteration
      - Rejecting invalid configurations (returns inf penalty)
      - Efficient loci computation

    Note: pylinkage uses pyswarms with ring topology which requires
    n_particles >= 18 (neighborhood size k=17). Smaller values will
    be automatically increased.

    Args:
        pylink_data: Base pylink document
        target: Target trajectory to match
        dimension_spec: Dimensions to optimize (extracted if not provided)
        n_particles: Number of particles in swarm (minimum 20 for pylinkage)
        iterations: Number of PSO iterations
        metric: Error metric for fitness function
        verbose: Print progress
        phase_invariant: If True, use phase-aligned scoring.
            RECOMMENDED when target may start at different phase than simulation.
            Prevents false errors from phase mismatch at cost of O(n) per eval.
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        init_mode: Particle initialization strategy:
            - 'random': Small perturbations around initial config (original behavior)
            - 'sobol': Pre-sample using Sobol sequence, filter valid, use best
            - 'behnken': Pre-sample using Box-Behnken design (requires 3+ dims)
        init_samples: Number of samples to generate for presampling modes

    Returns:
        OptimizationResult with optimized dimensions

    Note on init_mode:
        Using 'sobol' or 'behnken' presampling can dramatically improve
        convergence for pylinkage PSO, which historically struggled with
        random initialization landing in invalid regions.
    """
    verbose = True
    # Pylinkage/pyswarms requires minimum particles for ring topology
    MIN_PARTICLES = 20
    if n_particles < MIN_PARTICLES:
        if verbose:
            logger.info(f'  Note: Increasing n_particles from {n_particles} to {MIN_PARTICLES} (pylinkage minimum)')
        n_particles = MIN_PARTICLES
    import pylinkage as pl
    from pylink_tools.kinematic import (
        build_joint_objects, make_linkage, compute_proper_solve_order,
    )

    # Extract dimensions if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Build the base linkage
    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])
    meta = pylink_data.get('meta', {})
    meta_joints = meta.get('joints', {})
    n_steps = target.n_steps

    solve_order = compute_proper_solve_order(joints_data, verbose=False)
    joint_info = {j['name']: j for j in joints_data}
    angle_per_step = 2 * np.pi / n_steps

    joint_objects = build_joint_objects(
        joints_data, solve_order, joint_info, meta_joints, angle_per_step, verbose=False,
    )

    linkage, error = make_linkage(joint_objects, solve_order, pylinkage_data.get('name', 'opt'))

    if error:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f'Failed to build linkage: {error}',
        )

    # Save initial state
    init_coords = linkage.get_coords()
    init_constraints = linkage.get_num_constraints()

    if verbose:
        logger.info('Starting pylinkage PSO optimization')
        logger.info(f'  Dimensions: {len(dimension_spec)}')
        logger.info(f'  Initial constraints: {init_constraints}')
        logger.info(f'  Particles: {n_particles}')
        logger.info(f'  Iterations: {iterations}')
        logger.info(f'  Phase invariant: {phase_invariant}')
        logger.info(f'  Init mode: {init_mode}')

    # Find target joint index in linkage.joints
    target_joint_name = target.joint_name
    target_joint_idx = None
    for idx, joint in enumerate(linkage.joints):
        if joint.name == target_joint_name:
            target_joint_idx = idx
            break

    if target_joint_idx is None:
        # Default to last joint
        target_joint_idx = len(linkage.joints) - 1
        if verbose:
            logger.warning(f"Target joint '{target_joint_name}' not found, using last joint")

    # Track fitness function calls for debugging
    fitness_call_count = [0]

    # Log n_steps for debugging
    logger.info(f'  n_steps for simulation: {n_steps}')
    logger.info(f'  target trajectory points: {len(target.positions)}')

    # Create fitness function WITHOUT @kinematic_minimization decorator
    # The decorator hardcodes 96 iterations which doesn't match our n_steps
    # Instead, we manually control the simulation with the correct n_steps
    def fitness_func(linkage_obj, params, init_pos_arg=None):
        """
        Fitness function for pylinkage PSO.

        Args:
            linkage_obj: The linkage to evaluate
            params: Dimension values to set
            init_pos_arg: Initial joint positions (optional)
        """
        fitness_call_count[0] += 1

        try:
            # Set initial coordinates if provided
            if init_pos_arg is not None:
                linkage_obj.set_coords(init_pos_arg)

            # Set the constraint parameters
            linkage_obj.set_num_constraints(params)

            # Run simulation with OUR n_steps (not pylinkage's hardcoded 96)
            try:
                linkage_obj.rebuild()
                loci = list(linkage_obj.step(iterations=n_steps))
            except Exception as e:
                if fitness_call_count[0] <= 3:
                    logger.debug(f'    [fitness #{fitness_call_count[0]}] rebuild/step failed: {e}')
                return float('inf')

            if not loci or len(loci) != n_steps:
                if fitness_call_count[0] <= 3:
                    logger.debug(f'    [fitness #{fitness_call_count[0]}] loci len mismatch: {len(loci) if loci else 0} vs {n_steps}')
                return float('inf')
            # Extract target joint trajectory from loci
            # loci[step_idx] = ((x1,y1), (x2,y2), ...) positions of all joints at this step
            computed_trajectory = []
            for step_positions in loci:
                if target_joint_idx < len(step_positions):
                    pos = step_positions[target_joint_idx]
                    computed_trajectory.append(pos)
                else:
                    if fitness_call_count[0] <= 3:
                        logger.debug(f'    [fitness #{fitness_call_count[0]}] target_joint_idx out of range')
                    return float('inf')

            if len(computed_trajectory) != len(target.positions):
                if fitness_call_count[0] <= 3:
                    logger.debug(f'    [fitness #{fitness_call_count[0]}] len mismatch: {len(computed_trajectory)} vs {len(target.positions)}')
                return float('inf')

            # Check for invalid positions
            for idx, pos in enumerate(computed_trajectory):
                if pos is None or pos[0] is None:
                    if fitness_call_count[0] <= 3:
                        logger.debug(f'    [fitness #{fitness_call_count[0]}] invalid position at idx={idx}')
                    return float('inf')

            # Use phase-invariant scoring if requested
            if phase_invariant:
                # Use fast path - directly returns the requested metric
                result = compute_trajectory_error_fast(
                    computed_trajectory,
                    target,
                    metric=metric,
                    phase_invariant=True,
                    phase_align_method=phase_align_method,
                )
            else:
                # Direct point-to-point comparison (faster but phase-sensitive)
                total_error = 0.0
                weights = target.weights or [1.0] * len(target.positions)

                for i, (computed, target_pos) in enumerate(zip(computed_trajectory, target.positions)):
                    dx = computed[0] - target_pos[0]
                    dy = computed[1] - target_pos[1]
                    total_error += weights[i] * (dx * dx + dy * dy)

                if metric == 'mse':
                    result = total_error / len(target.positions)
                else:
                    result = total_error

            return result

        except Exception as e:
            if fitness_call_count[0] <= 3:
                logger.debug(f'    [fitness #{fitness_call_count[0]}] EXCEPTION: {e}')
            return float('inf')

    # Get bounds in pylinkage format
    bounds = dimension_spec.get_bounds_tuple()

    # Compute initial error (use same phase_invariant setting as optimization)
    initial_error = float('inf')
    try:
        linkage.rebuild()
        loci_init = list(linkage.step(iterations=n_steps))
        if loci_init:
            computed = [
                step[target_joint_idx] if target_joint_idx < len(step) else (0, 0)
                for step in loci_init
            ]
            metrics = compute_trajectory_error_detailed(
                computed,
                target,
                phase_invariant=phase_invariant,
            )
            initial_error = metrics.mse if metric == 'mse' else metrics.total_error
    except Exception:
        pass

    if verbose:
        logger.info(f'  Initial error: {initial_error:.4f}')

    # Get bounds arrays
    init_constraints_array = np.array(init_constraints)
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    bound_range = upper_bounds - lower_bounds

    # Log detailed bounds information
    logger.info('  Dimension bounds:')
    for i, name in enumerate(dimension_spec.names):
        logger.info(f'    {name}: [{lower_bounds[i]:.2f}, {upper_bounds[i]:.2f}] (initial: {init_constraints_array[i]:.2f})')

    # Verify initial constraints are within bounds
    for i, name in enumerate(dimension_spec.names):
        if init_constraints_array[i] < lower_bounds[i] or init_constraints_array[i] > upper_bounds[i]:
            logger.warning(f'    WARNING: {name} initial value {init_constraints_array[i]:.2f} is OUTSIDE bounds!')

    # Initialize particle positions based on init_mode
    init_pos = np.zeros((n_particles, len(init_constraints)))

    if init_mode in ('sobol', 'behnken'):
        # Pre-sample using DOE methods to find valid configurations
        try:
            presampled_positions, presampled_scores = presample_valid_positions(
                pylink_data=pylink_data,
                target=target,
                dimension_spec=dimension_spec,
                n_samples=init_samples,
                n_best=n_particles,
                mode=init_mode,
                metric=metric,
                phase_invariant=phase_invariant,
            )

            if len(presampled_positions) > 0:
                n_presampled = min(len(presampled_positions), n_particles)
                init_pos[:n_presampled] = presampled_positions[:n_presampled]

                # Fill remaining with perturbations around best presampled position
                if n_presampled < n_particles:
                    best_pos = presampled_positions[0]
                    init_spread = 0.15
                    for i in range(n_presampled, n_particles):
                        perturbation = (np.random.random(len(init_constraints)) - 0.5) * 2 * init_spread * bound_range
                        init_pos[i] = np.clip(best_pos + perturbation, lower_bounds, upper_bounds)

                if verbose:
                    logger.info(f'  Using {n_presampled} pre-sampled positions (best score: {presampled_scores[0]:.4f})')
            else:
                # No valid presampled positions, fall back to random
                logger.warning('Presampling found no valid positions, falling back to random init')
                init_mode = 'random'
        except Exception as e:
            logger.warning(f'Presampling failed: {e}. Falling back to random init.')
            init_mode = 'random'

    if init_mode == 'random':
        # Original behavior: small perturbations around known valid position
        init_spread = 0.25
        for i in range(n_particles):
            if i == 0:
                init_pos[i] = init_constraints_array
            else:
                perturbation = (np.random.random(len(init_constraints)) - 0.5) * 2 * init_spread * bound_range
                init_pos[i] = np.clip(
                    init_constraints_array + perturbation,
                    lower_bounds,
                    upper_bounds,
                )
        if verbose:
            logger.info(f'  Init spread: {init_spread*100:.0f}% of bounds range')

    # Log init_pos statistics (debug level for details)
    logger.debug(f'  Init positions shape: {init_pos.shape}')
    logger.debug(f'  Init positions range: min={init_pos.min(axis=0)}, max={init_pos.max(axis=0)}')

    # Verify all init positions are within bounds
    out_of_bounds_count = 0
    for i in range(n_particles):
        for d in range(len(init_constraints)):
            if init_pos[i, d] < lower_bounds[d] or init_pos[i, d] > upper_bounds[d]:
                out_of_bounds_count += 1
    if out_of_bounds_count > 0:
        logger.warning(f'  WARNING: {out_of_bounds_count} init positions are outside bounds!')

    # Test fitness of first few init positions (debug level)
    logger.debug('  Testing fitness of first 3 init positions:')
    for i in range(min(3, n_particles)):
        # Build a test evaluation
        test_dims = tuple(init_pos[i])
        linkage.set_num_constraints(test_dims)
        try:
            linkage.rebuild()
            test_loci = list(linkage.step(iterations=n_steps))
            if test_loci:
                test_traj = [step[target_joint_idx] if target_joint_idx < len(step) else None for step in test_loci]
                valid = all(pos is not None and pos[0] is not None for pos in test_traj)
                if valid:
                    test_error = compute_trajectory_error_detailed(test_traj, target, phase_invariant=phase_invariant)
                    logger.debug(f'    Position {i}: dims={[f"{d:.2f}" for d in test_dims]} -> error={test_error.mse:.4f} ✓')
                else:
                    logger.debug(f'    Position {i}: dims={[f"{d:.2f}" for d in test_dims]} -> INVALID')
            else:
                logger.debug(f'    Position {i}: dims={[f"{d:.2f}" for d in test_dims]} -> NO LOCI')
        except Exception as e:
            logger.debug(f'    Position {i}: dims={[f"{d:.2f}" for d in test_dims]} -> EXCEPTION: {type(e).__name__}')

    # Reset linkage to initial state before optimization
    linkage.set_num_constraints(init_constraints)
    linkage.set_coords(init_coords)

    logger.debug('  Calling pylinkage particle_swarm_optimization...')
    logger.debug(f'    bounds: lower={bounds[0]}, upper={bounds[1]}')

    try:
        results = pl.particle_swarm_optimization(
            eval_func=fitness_func,
            linkage=linkage,
            bounds=bounds,
            n_particles=n_particles,
            iters=iterations,
            order_relation=min,
            verbose=verbose,
            init_pos=init_pos,  # Pass initial positions through kwargs to pyswarms
        )

        # Extract best result
        # Results is list of (score, dimensions, coords)
        if results and len(results) > 0:
            best_score, best_dims_tuple, best_coords = results[0]

            # Convert to our format
            best_dims = tuple(best_dims_tuple) if not isinstance(best_dims_tuple, tuple) else best_dims_tuple

            # Build optimized pylink_data
            optimized_pylink_data = apply_dimensions_from_array(
                pylink_data, best_dims, dimension_spec,
            )

            optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

            if verbose:
                logger.info(f'  Final error: {best_score:.4f}')
                if initial_error != float('inf') and initial_error > 0:
                    logger.info(f'  Improvement: {(1 - best_score/initial_error)*100:.1f}%')

            return OptimizationResult(
                success=True,
                optimized_dimensions=optimized_dims,
                optimized_pylink_data=optimized_pylink_data,
                initial_error=initial_error,
                final_error=best_score,
                iterations=iterations,
                convergence_history=None,  # pylinkage doesn't provide this easily
            )
        else:
            return OptimizationResult(
                success=False,
                optimized_dimensions={},
                initial_error=initial_error,
                error='PSO returned no results',
            )

    except Exception as e:
        if verbose:
            logger.error(f'pylinkage PSO failed: {e}', exc_info=True)
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'pylinkage PSO failed: {str(e)}',
        )
    finally:
        # Reset linkage to initial state
        try:
            linkage.set_num_constraints(init_constraints)
            linkage.set_coords(init_coords)
        except Exception:
            pass


def _run_pso_internal(
    fitness_func: Callable[[tuple[float, ...]], float],
    initial_values: tuple[float, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    n_particles: int = 32,
    iterations: int = 512,
    w: float = 0.7,      # Inertia weight
    c1: float = 1.5,     # Cognitive parameter
    c2: float = 1.5,     # Social parameter
    verbose: bool = False,
    init_positions: np.ndarray | None = None,
) -> tuple[float, tuple[float, ...], list[float]]:
    """
    Internal PSO implementation.

    Standard PSO algorithm with velocity clamping and boundary handling.

    Args:
        fitness_func: Objective function to minimize
        initial_values: Starting point (used to seed one particle)
        bounds: ((lower...), (upper...)) bounds for each dimension
        n_particles: Swarm size
        iterations: Number of iterations
        w: Inertia weight (momentum)
        c1: Cognitive parameter (personal best attraction)
        c2: Social parameter (global best attraction)
        verbose: Print progress
        init_positions: Optional pre-computed initial positions from presampling.
            If provided, these positions are used instead of random initialization.
            Shape: (n_init, n_dims) where n_init <= n_particles.

    Returns:
        (best_score, best_position, convergence_history)
    """
    n_dims = len(initial_values)
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])

    # Initialize particles
    positions = np.random.uniform(
        lower_bounds, upper_bounds, (n_particles, n_dims),
    )

    if init_positions is not None and len(init_positions) > 0:
        # Use pre-sampled positions (already validated as producing valid mechanisms)
        n_presampled = min(len(init_positions), n_particles)
        positions[:n_presampled] = init_positions[:n_presampled]
        if verbose:
            logger.info(f'  Using {n_presampled} pre-sampled positions, {n_particles - n_presampled} random')
    else:
        # Original behavior: first particle at initial values, rest random
        positions[0] = np.array(initial_values)

    # Initialize velocities (small random values)
    max_velocity = (upper_bounds - lower_bounds) * 0.2
    velocities = np.random.uniform(-max_velocity, max_velocity, (n_particles, n_dims))

    # Initialize personal bests
    personal_best_positions = positions.copy()
    personal_best_scores = np.full(n_particles, float('inf'))

    # Evaluate initial positions
    for i in range(n_particles):
        score = fitness_func(tuple(positions[i]))
        personal_best_scores[i] = score

    # Initialize global best
    best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[best_idx].copy()
    global_best_score = personal_best_scores[best_idx]

    # Track convergence
    history = [global_best_score]

    # Main PSO loop
    for iteration in range(iterations):
        for i in range(n_particles):
            # Generate random factors
            r1, r2 = np.random.random(n_dims), np.random.random(n_dims)

            # Update velocity
            cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Update position
            positions[i] = positions[i] + velocities[i]

            # Handle boundary violations (reflect)
            for d in range(n_dims):
                if positions[i, d] < lower_bounds[d]:
                    positions[i, d] = lower_bounds[d]
                    velocities[i, d] *= -0.5
                elif positions[i, d] > upper_bounds[d]:
                    positions[i, d] = upper_bounds[d]
                    velocities[i, d] *= -0.5

            # Evaluate new position
            score = fitness_func(tuple(positions[i]))

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

        history.append(global_best_score)

        if verbose and (iteration + 1) % 10 == 0:
            logger.info(f'  Iteration {iteration + 1}/{iterations}: best_error={global_best_score:.4f}')

    return global_best_score, tuple(global_best_position), history


def optimize_trajectory(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    custom_bounds: dict[str, tuple[float, float]] | None = None,
    method: str = 'pylinkage',
    metric: str = 'mse',
    verbose: bool = True,
    **kwargs,
) -> OptimizationResult:
    """
    Main entry point for trajectory optimization.

    Finds link lengths that make the mechanism's trajectory best match
    the target positions.

    Args:
        pylink_data: Full pylink document
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Pre-computed dimension spec (optional)
        custom_bounds: Override bounds for specific dimensions
        method: Optimization method:
                - "pylinkage": Particle Swarm Optimization (default, robust)
                - "pso": Alias for pylinkage PSO
                - "scipy": scipy.optimize.minimize with L-BFGS-B (faster)
                - "powell": scipy Powell method (gradient-free)
                - "nelder-mead": Nelder-Mead simplex (gradient-free)
                - "nlopt" / "nlopt_mlsl" / "mlsl": NLopt MLSL with L-BFGS (global)
                - "scip": SCIP mixed-integer solver (for discrete params)
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: Print progress
        **kwargs: Method-specific arguments:
                  Common: phase_invariant (True), phase_align_method ('rotation'|'fft'|'frechet')
                  PSO: n_particles (32), iterations (512),
                       init_mode ('random'|'sobol'|'behnken'), init_samples (128)
                  scipy: max_iterations (100), tolerance (1e-6)
                  nlopt: max_eval (1000), local_max_eval (100), ftol_rel (1e-6)
                  scip: time_limit (300), gap_limit (0.01)

    TODO (medium-effort enhancements):
        - Add `timeout` parameter for PSO/scipy methods (only SCIP has time_limit)
        - Add `callback` parameter for progress monitoring and early stopping
        - Add `warm_start` parameter to initialize from previous OptimizationResult

    Returns:
        OptimizationResult with optimized dimensions and updated pylink_data

    Example:
        >>> target = TargetTrajectory(
        ...     joint_name="coupler",
        ...     positions=[(100, 120), (105, 125), ...]
        ... )
        >>> result = optimize_trajectory(pylink_data, target, method="pylinkage")
        >>> if result.success:
        ...     print(f"Reduced error from {result.initial_error:.2f} to {result.final_error:.2f}")
        ...     optimized_linkage = result.optimized_pylink_data
    """
    # Extract/prepare dimensions
    if dimension_spec is None:
        if custom_bounds:
            dimension_spec = extract_dimensions_with_custom_bounds(pylink_data, custom_bounds)
        else:
            dimension_spec = extract_dimensions(pylink_data)

    if verbose:
        logger.info('Optimizing trajectory fit')
        logger.info(f'  Target joint: {target.joint_name}')
        logger.info(f'  Target steps: {target.n_steps}')
        logger.info(f'  Dimensions: {dimension_spec.names}')
        logger.info(f'  Method: {method}')

    # Route to appropriate optimizer
    method_lower = method.lower()

    # Extract common parameters
    phase_invariant = kwargs.get('phase_invariant', True)
    phase_align_method = kwargs.get('phase_align_method', 'rotation')

    if method_lower == 'pso':
        return run_pso_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            metric=metric,
            verbose=verbose,
            n_particles=kwargs.get('n_particles', 32),
            iterations=kwargs.get('iterations', 512),
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
            init_mode=kwargs.get('init_mode', 'random'),
            init_samples=kwargs.get('init_samples', 128),
        )

    elif method_lower == 'pylinkage':
        return run_pylinkage_pso(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            metric=metric,
            verbose=verbose,
            n_particles=kwargs.get('n_particles', 32),
            iterations=kwargs.get('iterations', 512),
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
            init_mode=kwargs.get('init_mode', 'random'),
            init_samples=kwargs.get('init_samples', 128),
        )

    elif method_lower in ('scipy', 'l-bfgs-b'):
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='L-BFGS-B',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    elif method_lower == 'powell':
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='Powell',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    elif method_lower == 'nelder-mead':
        return run_scipy_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            method='Nelder-Mead',
            metric=metric,
            verbose=verbose,
            max_iterations=kwargs.get('max_iterations', 100),
            tolerance=kwargs.get('tolerance', 1e-6),
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    elif method_lower in ('nlopt', 'nlopt_mlsl', 'mlsl'):
        # NLopt MLSL with L-BFGS local search (global optimizer)
        try:
            from optimizers.nlopt_mlsl import run_nlopt_mlsl, NLoptMLSLConfig
        except ImportError as e:
            return OptimizationResult(
                success=False,
                optimized_dimensions={},
                error=f'Failed to import nlopt_mlsl: {e}',
            )

        config = kwargs.get('config')
        if config is None:
            config = NLoptMLSLConfig(
                max_eval=kwargs.get('max_eval', 1000),
                local_max_eval=kwargs.get('local_max_eval', 100),
                ftol_rel=kwargs.get('ftol_rel', 1e-6),
            )

        return run_nlopt_mlsl(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            config=config,
            metric=metric,
            verbose=verbose,
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    elif method_lower == 'scip':
        # SCIP mixed-integer nonlinear programming
        try:
            from optimizers.scip_optimizer import run_scip_optimization, SCIPConfig
        except ImportError as e:
            return OptimizationResult(
                success=False,
                optimized_dimensions={},
                error=f'Failed to import scip_optimizer: {e}',
            )

        config = kwargs.get('config')
        if config is None:
            config = SCIPConfig(
                time_limit=kwargs.get('time_limit', 300.0),
                gap_limit=kwargs.get('gap_limit', 0.01),
            )

        return run_scip_optimization(
            pylink_data=pylink_data,
            target=target,
            dimension_spec=dimension_spec,
            config=config,
            metric=metric,
            verbose=verbose,
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    else:
        available_methods = [
            'pso', 'pylinkage', 'scipy', 'l-bfgs-b', 'powell', 'nelder-mead',
            'nlopt', 'nlopt_mlsl', 'mlsl', 'scip',
        ]
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f"Unknown optimization method: {method}. Available: {', '.join(available_methods)}",
        )
