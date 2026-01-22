"""
optimize.py - Trajectory optimization for linkage mechanisms.

Core functionality:
  - Compute error between computed trajectory and target trajectory
  - Run optimization to fit linkage to target path
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
from typing import Any
from typing import Literal

import numpy as np

from configs.logging_config import get_logger
from optimizers.pso_optimizer import run_pso_optimization
from optimizers.pylinkage_pso import run_pylinkage_pso
from optimizers.scipy_optimizer import run_scipy_optimization
from pylink_tools.optimization_helpers import extract_dimensions
from pylink_tools.optimization_types import ConvergenceStats
from pylink_tools.optimization_types import DimensionSpec
from pylink_tools.optimization_types import OptimizationResult
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.trajectory_utils import compute_trajectory_error
from pylink_tools.trajectory_utils import compute_trajectory_hot

# Module logger
logger = get_logger(__name__)


def evaluate_linkage_fit(
    pylink_data: dict,
    target: TargetTrajectory,
    n_steps: int | None = None,
    phase_invariant: bool = True,
    _skip_copy: bool = False,
    metric: str = 'mse',
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    translation_invariant: bool = False,
) -> float:
    """
    Evaluate how well a linkage fits a target trajectory.

    This is a convenience function that:
      1. Computes the trajectory for the linkage
      2. Extracts the target joint's path
      3. Computes error metric (optionally phase-aligned)

    Args:
        pylink_data: Full pylink document
        target: Target trajectory to match
        n_steps: Number of simulation steps (uses target.n_steps if not provided)
        phase_invariant: If True, find optimal phase alignment before scoring.
            RECOMMENDED for real-world targets where starting point may differ.
        _skip_copy: Internal use only. If True, skip deepcopy for performance.
            Only use this when pylink_data is a dedicated working copy that
            can be safely mutated (e.g., in optimization hot loops).
        metric: Error metric to compute: 'mse', 'rmse', 'total', 'max'
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        translation_invariant: If True, center both trajectories before
            comparison, focusing on SHAPE rather than absolute position.

    Returns:
        Float error value (lower is better)

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
        return float('inf')

    # Get target joint trajectory
    joint_name = target.joint_name
    if joint_name not in result.trajectories:
        raise ValueError(f"Target joint '{joint_name}' not found in trajectories")

    computed = result.trajectories[joint_name]

    # Hot path for MSE with phase invariance - use numpy directly
    if metric == 'mse' and phase_invariant and not translation_invariant:
        computed_arr = np.array(computed)
        target_arr = np.array(target.positions)
        return compute_trajectory_hot(target_arr, computed_arr)

    # Standard path
    computed_tuples = [tuple(pos) for pos in computed]
    return compute_trajectory_error(
        computed_tuples,
        target,
        metric=metric,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method,
        translation_invariant=translation_invariant,
    )


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


def create_fitness_function(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    metric: str = 'mse',
    verbose: bool = False,
    phase_invariant: bool = False,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    use_cache: bool = False,
    translation_invariant: bool = False,
) -> Any:  # Returns callable with attached metadata attributes
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
        phase_align_method: Phase alignment algorithm (only if phase_invariant=True):
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        use_cache: If True, cache results with LRU cache (useful when optimizer
            may revisit same points). Cache uses rounded dimension values.
        translation_invariant: If True, center both trajectories before
            comparison, focusing on SHAPE rather than absolute position.
            Use this when trajectory position doesn't matter (e.g., shape
            matching, mechanisms with moving static joints).

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
        # Use edge_mapping if available, fall back to joint_mapping for compatibility
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

    def _get_cache_key(dims: tuple[float, ...]) -> tuple:
        """Round dimensions to 6 decimal places for cache key."""
        return tuple(round(d, 6) for d in dims)

    def fitness(dimensions: tuple[float, ...]) -> float:
        """Evaluate linkage fitness with given dimensions."""
        eval_count[0] += 1

        # Check cache if enabled
        if result_cache is not None:
            cache_key = _get_cache_key(dimensions)
            if cache_key in result_cache:
                cache_hits[0] += 1
                return result_cache[cache_key]

        try:
            _apply_dimensions_fast(dimensions)

            # Evaluate fit - uses compute_trajectory_hot for MSE fast path
            error_val = float(
                evaluate_linkage_fit(
                    working_copy,
                    target,
                    phase_invariant=phase_invariant,
                    _skip_copy=True,
                    metric=metric,
                    phase_align_method=phase_align_method,
                    translation_invariant=translation_invariant,
                ),
            )

            if verbose and eval_count[0] % 50 == 0:
                cache_info = f', cache_hits={cache_hits[0]}' if result_cache is not None else ''
                logger.debug(f'Eval #{eval_count[0]}: error={error_val:.4f}, dims={dimensions[:3]}...{cache_info}')

            if result_cache is not None:
                result_cache[_get_cache_key(dimensions)] = error_val

            return error_val

        except Exception as e:
            if verbose:
                logger.warning(f'Eval #{eval_count[0]} failed: {e}')
            return float('inf')

    # Attach metadata to function for introspection
    # Using setattr to avoid type checker complaints about function attributes
    setattr(fitness, 'eval_count', eval_count)
    setattr(fitness, 'cache_hits', cache_hits)
    setattr(fitness, 'dimension_spec', dimension_spec)
    setattr(fitness, 'target', target)
    setattr(fitness, 'phase_invariant', phase_invariant)
    setattr(fitness, 'phase_align_method', phase_align_method)
    setattr(fitness, 'translation_invariant', translation_invariant)
    setattr(fitness, 'result_cache', result_cache)

    return fitness


# =============================================================================
# Main Entry Point
# =============================================================================


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
                - "pso": Standalone PSO implementation
                - "scipy": scipy.optimize.minimize with L-BFGS-B (faster)
                - "powell": scipy Powell method (gradient-free)
                - "nelder-mead": Nelder-Mead simplex (gradient-free)
                - "nlopt" / "nlopt_mlsl" / "mlsl": NLopt MLSL with L-BFGS (global)
                - "scip": SCIP mixed-integer solver (for discrete params)
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: Print progress
        **kwargs: Method-specific arguments:
                  Common: phase_invariant (True),
                          phase_align_method ('rotation'|'fft'|'frechet')
                  PSO: n_particles (32), iterations (512),
                       init_mode ('random'|'sobol'|'behnken'), init_samples (128)
                  scipy: max_iterations (100), tolerance (1e-6)
                  nlopt: max_eval (1000), local_max_eval (100), ftol_rel (1e-6)
                  scip: time_limit (300), gap_limit (0.01)

    Returns:
        OptimizationResult with optimized dimensions and updated pylink_data

    Example:
        >>> target = TargetTrajectory(
        ...     joint_name="coupler",
        ...     positions=[(100, 120), (105, 125), ...]
        ... )
        >>> result = optimize_trajectory(pylink_data, target, method="pylinkage")
        >>> if result.success:
        ...     print(f"Reduced error from {result.initial_error:.2f} "
        ...           f"to {result.final_error:.2f}")
        ...     optimized_linkage = result.optimized_pylink_data
    """
    # Extract/prepare dimensions
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data, custom_bounds=custom_bounds)

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
            from optimizers.nlopt_mlsl import NLoptMLSLConfig, run_nlopt_mlsl
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
            'pso',
            'pylinkage',
            'scipy',
            'l-bfgs-b',
            'powell',
            'nelder-mead',
            'nlopt',
            'nlopt_mlsl',
            'mlsl',
            'scip',
        ]
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f'Unknown optimization method: {method}. ' f"Available: {', '.join(available_methods)}",
        )
