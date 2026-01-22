"""
Scipy-based optimizer for linkage trajectory optimization.

Uses scipy.optimize.minimize with various methods like L-BFGS-B, Powell, and Nelder-Mead.
Good for well-behaved problems and gradient-based optimization.

Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

License: scipy is BSD licensed (permissive for commercial use)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from pylink_tools.optimization_types import (
        DimensionSpec,
        OptimizationResult,
        TargetTrajectory,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ScipyConfig:
    """
    Configuration for scipy optimizer.

    Attributes:
        method: Scipy optimizer method ('L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead')
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    """

    method: str = 'L-BFGS-B'
    max_iterations: int = 100
    tolerance: float = 1e-6


# =============================================================================
# Main Interface
# =============================================================================


def run_scipy_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: ScipyConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Run optimization using scipy.optimize.minimize.

    This is often faster than PSO for well-behaved problems and supports
    gradient-based methods like L-BFGS-B.

    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: Scipy configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring (recommended for external targets)
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        **kwargs: Additional arguments for interface compatibility:
            - method: Override config.method
            - max_iterations: Override config.max_iterations
            - tolerance: Override config.tolerance

    Returns:
        OptimizationResult with:
            - success: True if optimization converged
            - optimized_dimensions: Best found dimension values
            - optimized_pylink_data: Updated linkage with optimized dimensions
            - initial_error: Error before optimization
            - final_error: Best achieved error
            - iterations: Number of iterations
            - convergence_history: Error progression

    Example:
        >>> from optimizers import run_scipy_optimization, ScipyConfig
        >>> config = ScipyConfig(method='L-BFGS-B', max_iterations=200)
        >>> result = run_scipy_optimization(pylink_data, target, config=config)
        >>> if result.success:
        ...     print(f"Final error: {result.final_error}")
        ...     print(f"Optimized: {result.optimized_dimensions}")
    """
    from pylink_tools.optimization_helpers import (
        apply_dimensions_from_array,
        dimensions_to_dict,
        extract_dimensions,
    )
    from pylink_tools.optimization_types import OptimizationResult
    from pylink_tools.optimize import create_fitness_function

    # Use default config if not provided
    if config is None:
        config = ScipyConfig()

    # Allow kwargs to override config
    method = kwargs.get('method', config.method)
    max_iterations = kwargs.get('max_iterations', config.max_iterations)
    tolerance = kwargs.get('tolerance', config.tolerance)

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
        pylink_data,
        target,
        dimension_spec,
        metric=metric,
        verbose=verbose,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method,
    )

    # Compute initial error
    initial_values = tuple(dimension_spec.initial_values)
    initial_error = fitness(initial_values)

    if verbose:
        logger.info(f'Starting scipy optimization ({method})')
        logger.info(f'  Dimensions: {len(dimension_spec)}')
        logger.info(f'  Initial error: {initial_error:.4f}')
        logger.info(f'  Phase invariant: {phase_invariant}')

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
            pylink_data,
            optimized_values,
            dimension_spec,
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
        logger.error(f'Scipy optimization failed: {e}')
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'Optimization failed: {str(e)}',
        )
