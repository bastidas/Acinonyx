"""
NLopt MLSL (Multi-Level Single-Linkage) optimizer with L-BFGS local search.

MLSL is a global optimization algorithm that:
- Uses low-discrepancy sequences to sample the search space
- Runs local optimizations from promising sample points
- Clusters local minima to avoid redundant searches
- Combines global exploration with efficient local refinement

Reference: https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/

License: NLopt is under LGPL license (permissive for commercial use)
"""
from __future__ import annotations

# import copy
import logging
import time
from dataclasses import dataclass
# from dataclasses import field
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.optimization_types import (
        DimensionSpec,
        OptimizationResult,
        TargetTrajectory,
    )

logger = logging.getLogger(__name__)

@dataclass
class NLoptMLSLConfig:
    """
    Configuration for MLSL optimizer.

    Attributes:
        max_eval: Maximum function evaluations for global search
        local_max_eval: Maximum evaluations per local search
        ftol_rel: Relative function tolerance for local convergence
        xtol_rel: Relative parameter tolerance for convergence
        population: Initial population size (0 = auto based on dimensions)
        use_lds: Use Low-Discrepancy Sequences for sampling (recommended)
        local_algorithm: Local optimizer ('lbfgs', 'bobyqa', 'slsqp', 'sbplx')
        gradient_epsilon: Step size for finite difference gradient computation
        stopval: Stop when objective reaches this value (0 = disabled)
    """
    max_eval: int = 1000
    local_max_eval: int = 100
    ftol_rel: float = 1e-6
    xtol_rel: float = 1e-6
    population: int = 0  # 0 = use heuristic default
    use_lds: bool = True  # Use MLSL_LDS variant (recommended)
    local_algorithm: str = 'lbfgs'  # 'lbfgs', 'bobyqa', 'slsqp', 'sbplx'
    gradient_epsilon: float = 1e-6  # For finite difference gradient
    stopval: float = 0.0  # Stop early if error reaches this (0 = disabled)


def run_nlopt_mlsl(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: NLoptMLSLConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Run MLSL global optimization with L-BFGS local search using NLopt.

    MLSL (Multi-Level Single-Linkage) is ideal for:
    - Problems with multiple local minima
    - When good bounds are known but global minimum location is unknown
    - Cases where a gradient-based local search can refine solutions

    The algorithm works by:
    1. Sampling points using low-discrepancy sequences
    2. Running local L-BFGS optimizations from promising points
    3. Clustering results to identify distinct local minima
    4. Returning the best found solution

    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: MLSL configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring (recommended for external targets)
        phase_align_method: Phase alignment algorithm ('rotation', 'fft', 'frechet')
        **kwargs: Additional arguments (ignored, for interface compatibility)

    Returns:
        OptimizationResult with:
            - success: True if optimization converged
            - optimized_dimensions: Best found dimension values
            - optimized_pylink_data: Updated linkage with optimized dimensions
            - initial_error: Error before optimization
            - final_error: Best achieved error
            - iterations: Number of function evaluations
            - convergence_history: Error progression (if available)

    Raises:
        ImportError: If nlopt package is not installed

    Example:
        >>> from optimizers import run_nlopt_mlsl, NLoptMLSLConfig
        >>> config = NLoptMLSLConfig(max_eval=2000, ftol_rel=1e-8)
        >>> result = run_nlopt_mlsl(
        ...     pylink_data, target,
        ...     config=config,
        ...     verbose=True
        ... )
        >>> if result.success:
        ...     print(f"Best error: {result.final_error:.6f}")

    Notes:
        - Requires: pip install nlopt
        - NLopt is LGPL licensed (permissive for commercial use)
        - MLSL_LDS variant uses Sobol sequences for better coverage
        - Local L-BFGS requires gradient estimation (finite differences)
    """
    from pylink_tools.optimization_types import OptimizationResult
    from pylink_tools.optimization_helpers import extract_dimensions, apply_dimensions
    from pylink_tools.optimize import create_fitness_function

    # Import here to provide clear error if not installed
    try:
        import nlopt
    except ImportError:
        error_msg = (
            'NLopt package not installed. Install with: pip install nlopt\n'
            'NLopt is LGPL licensed (permissive for commercial use).'
        )
        logger.error(error_msg)
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=error_msg,
        )

    # Use default config if not provided
    if config is None:
        config = NLoptMLSLConfig()

    # Extract dimension specification if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)

    dim = len(dimension_spec)
    if dim == 0:
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error='No dimensions to optimize',
        )

    fitness_func = create_fitness_function(
        pylink_data,
        target,
        dimension_spec,
        metric=metric,
        verbose=False,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method
    )

    # Compute initial error
    x0 = np.array(dimension_spec.initial_values, dtype=np.float64)
    initial_error = fitness_func(tuple(x0))

    # Get bounds
    lower_bounds = np.array([b[0] for b in dimension_spec.bounds], dtype=np.float64)
    upper_bounds = np.array([b[1] for b in dimension_spec.bounds], dtype=np.float64)

    if verbose:
        logger.info('Starting NLopt MLSL optimization')
        logger.info(f'  Dimensions: {dim}')
        logger.info(f"  Algorithm: MLSL{'_LDS' if config.use_lds else ''} + {config.local_algorithm.upper()}")
        logger.info(f'  Max evaluations: {config.max_eval}')
        logger.info(f'  Local max evaluations: {config.local_max_eval}')
        logger.info(f'  Initial error: {initial_error:.6f}')
        logger.info(f'  Bounds:')
        for name, lo, hi, init in zip(dimension_spec.names, lower_bounds, upper_bounds, x0):
            logger.info(f'    {name}: [{lo:.2f}, {hi:.2f}] (init: {init:.2f})')

    # Track convergence history
    convergence_history = []
    eval_count = [0]
    best_error = [initial_error]

    # Create NLopt-compatible objective
    def nlopt_objective(x, grad):
        eval_count[0] += 1
        error = fitness_func(tuple(x))

        # Track best error
        if error < best_error[0]:
            best_error[0] = error
            convergence_history.append(error)

        # Compute gradient if requested (finite differences)
        if grad.size > 0:
            h = config.gradient_epsilon
            for i in range(len(x)):
                step = h * max(abs(x[i]), 1.0)
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += step
                x_minus[i] -= step
                f_plus = fitness_func(tuple(x_plus))
                f_minus = fitness_func(tuple(x_minus))
                grad[i] = (f_plus - f_minus) / (2 * step)

        return error

    # -------------------------------------------------------------------------
    # Set up local optimizer
    # -------------------------------------------------------------------------
    local_algorithm_map = {
        'lbfgs': nlopt.LD_LBFGS,
        'slsqp': nlopt.LD_SLSQP,
        'mma': nlopt.LD_MMA,
        'bobyqa': nlopt.LN_BOBYQA,  # Gradient-free
        'sbplx': nlopt.LN_SBPLX,    # Gradient-free (improved Nelder-Mead)
        'cobyla': nlopt.LN_COBYLA,  # Gradient-free
    }

    local_alg = local_algorithm_map.get(config.local_algorithm.lower(), nlopt.LD_LBFGS)
    local_opt = nlopt.opt(local_alg, dim)
    local_opt.set_ftol_rel(config.ftol_rel)
    local_opt.set_xtol_rel(config.xtol_rel)
    local_opt.set_maxeval(config.local_max_eval)
    local_opt.set_lower_bounds(lower_bounds)
    local_opt.set_upper_bounds(upper_bounds)

    # -------------------------------------------------------------------------
    # Set up global optimizer (MLSL)
    # -------------------------------------------------------------------------
    if config.use_lds:
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, dim)
    else:
        global_opt = nlopt.opt(nlopt.G_MLSL, dim)

    global_opt.set_min_objective(nlopt_objective)
    global_opt.set_lower_bounds(lower_bounds)
    global_opt.set_upper_bounds(upper_bounds)
    global_opt.set_maxeval(config.max_eval)
    global_opt.set_local_optimizer(local_opt)

    # Set population if specified
    if config.population > 0:
        global_opt.set_population(config.population)

    if config.stopval > 0:
        global_opt.set_stopval(config.stopval)

    # -------------------------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------------------------
    start_time = time.time()
    result_code = None
    xopt = None
    final_error = initial_error

    try:
        xopt = global_opt.optimize(x0)
        final_error = global_opt.last_optimum_value()
        result_code = global_opt.last_optimize_result()

    except nlopt.RoundoffLimited:
        # Roundoff-limited convergence - often still a good result
        logger.warning('NLopt: Roundoff-limited convergence')
        xopt = x0 if xopt is None else xopt
        final_error = best_error[0]
        result_code = nlopt.ROUNDOFF_LIMITED

    except nlopt.ForcedStop:
        logger.warning('NLopt: Optimization was stopped')
        xopt = x0 if xopt is None else xopt
        final_error = best_error[0]
        result_code = nlopt.FORCED_STOP

    except Exception as e:
        logger.error(f'NLopt optimization failed: {e}')
        return OptimizationResult(
            success=False,
            optimized_dimensions=dict(zip(dimension_spec.names, x0)),
            optimized_pylink_data=pylink_data,
            initial_error=initial_error,
            final_error=initial_error,
            iterations=eval_count[0],
            error=str(e),
        )

    elapsed_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # Process results
    # -------------------------------------------------------------------------
    # Map result codes to success/failure
    success_codes = {
        nlopt.SUCCESS,
        nlopt.FTOL_REACHED,
        nlopt.XTOL_REACHED,
        nlopt.STOPVAL_REACHED,
    }
    partial_success_codes = {
        nlopt.MAXEVAL_REACHED,
        nlopt.ROUNDOFF_LIMITED,
    }

    success = result_code in success_codes
    if result_code in partial_success_codes and final_error < initial_error:
        success = True  # Count as success if we improved

    # Build optimized dimensions dict
    optimized_dims = dict(zip(dimension_spec.names, xopt))

    # Apply to pylink_data
    optimized_pylink_data = apply_dimensions(pylink_data, optimized_dims, dimension_spec)

    if verbose:
        result_name = _get_result_name(result_code)
        improvement = (1 - final_error / initial_error) * 100 if initial_error > 0 else 0
        logger.info(f'NLopt MLSL completed:')
        logger.info(f'  Result: {result_name}')
        logger.info(f'  Evaluations: {eval_count[0]}')
        logger.info(f'  Time: {elapsed_time:.2f}s')
        logger.info(f'  Initial error: {initial_error:.6f}')
        logger.info(f'  Final error: {final_error:.6f}')
        logger.info(f'  Improvement: {improvement:.1f}%')
        logger.info(f'  Optimized dimensions:')
        for name, val in optimized_dims.items():
            init_val = dict(zip(dimension_spec.names, x0))[name]
            logger.info(f'    {name}: {init_val:.2f} -> {val:.2f}')

    return OptimizationResult(
        success=success,
        optimized_dimensions=optimized_dims,
        optimized_pylink_data=optimized_pylink_data,
        initial_error=initial_error,
        final_error=final_error,
        iterations=eval_count[0],
        convergence_history=convergence_history if convergence_history else None,
    )


def _get_result_name(result_code) -> str:
    """Map NLopt result code to human-readable name."""
    try:
        import nlopt
        names = {
            nlopt.SUCCESS: 'SUCCESS',
            nlopt.STOPVAL_REACHED: 'STOPVAL_REACHED',
            nlopt.FTOL_REACHED: 'FTOL_REACHED',
            nlopt.XTOL_REACHED: 'XTOL_REACHED',
            nlopt.MAXEVAL_REACHED: 'MAXEVAL_REACHED',
            nlopt.MAXTIME_REACHED: 'MAXTIME_REACHED',
            nlopt.FAILURE: 'FAILURE',
            nlopt.INVALID_ARGS: 'INVALID_ARGS',
            nlopt.OUT_OF_MEMORY: 'OUT_OF_MEMORY',
            nlopt.ROUNDOFF_LIMITED: 'ROUNDOFF_LIMITED',
            nlopt.FORCED_STOP: 'FORCED_STOP',
        }
        return names.get(result_code, f'UNKNOWN({result_code})')
    except Exception:
        return f'CODE({result_code})'


# =============================================================================
# Alternative: Gradient-Free MLSL
# =============================================================================

def run_nlopt_mlsl_gf(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: NLoptMLSLConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Gradient-free variant of MLSL using BOBYQA local search.

    This is a convenience wrapper that sets local_algorithm='bobyqa'
    which doesn't require gradient computation. May be faster for
    expensive objective functions.
    """
    if config is None:
        config = NLoptMLSLConfig(local_algorithm='bobyqa')
    else:
        # Override local algorithm to gradient-free
        config = NLoptMLSLConfig(
            max_eval=config.max_eval,
            local_max_eval=config.local_max_eval,
            ftol_rel=config.ftol_rel,
            xtol_rel=config.xtol_rel,
            population=config.population,
            use_lds=config.use_lds,
            local_algorithm='bobyqa',
            gradient_epsilon=config.gradient_epsilon,
            stopval=config.stopval,
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
        **kwargs,
    )


# =============================================================================
# Algorithm Constants (for reference)
# =============================================================================

NLOPT_ALGORITHMS = """
NLopt Algorithm Reference (for implementation):

Global Optimizers (gradient-free):
- nlopt.GN_DIRECT: DIviding RECTangles
- nlopt.GN_DIRECT_L: Locally-biased DIRECT
- nlopt.GN_CRS2_LM: Controlled Random Search
- nlopt.GN_ISRES: Improved Stochastic Ranking Evolution Strategy
- nlopt.GN_ESCH: ESCH evolutionary algorithm

Global Optimizers (requires local optimizer):
- nlopt.G_MLSL: Multi-Level Single-Linkage
- nlopt.G_MLSL_LDS: MLSL with Low-Discrepancy Sequences (recommended)
- nlopt.GD_STOGO: StoGO global optimization

Local Optimizers (gradient-based):
- nlopt.LD_LBFGS: Limited-memory BFGS (recommended)
- nlopt.LD_SLSQP: Sequential Least-Squares Quadratic Programming
- nlopt.LD_MMA: Method of Moving Asymptotes
- nlopt.LD_TNEWTON: Truncated Newton

Local Optimizers (gradient-free):
- nlopt.LN_BOBYQA: Bound Optimization BY Quadratic Approximation
- nlopt.LN_COBYLA: Constrained Optimization BY Linear Approximations
- nlopt.LN_NEWUOA: NEW Unconstrained Optimization Algorithm
- nlopt.LN_SBPLX: Subplex (improved Nelder-Mead)
"""
