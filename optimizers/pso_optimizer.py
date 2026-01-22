"""
Particle Swarm Optimization (PSO) for linkage trajectory optimization.

A population-based optimizer that's good for:
- Non-convex problems
- Escaping local minima
- Problems where gradients are unavailable

This is a standalone PSO implementation (not using pylinkage's wrapper).
For pylinkage-native PSO, see pylinkage_pso.py.

License: MIT (this is custom implementation)
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PSOConfig:
    """
    Configuration for PSO optimizer.

    Attributes:
        n_particles: Number of particles in swarm
        iterations: Number of PSO iterations
        w: Inertia weight (momentum)
        c1: Cognitive parameter (personal best attraction)
        c2: Social parameter (global best attraction)
        init_mode: Particle initialization strategy:
            - 'random': Random positions in bounds (default)
            - 'sobol': Pre-sample using Sobol sequence, filter valid, use best
            - 'behnken': Pre-sample using Box-Behnken design (requires 3+ dims)
        init_samples: Number of samples to generate for presampling modes
    """

    n_particles: int = 32
    iterations: int = 512
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive parameter
    c2: float = 1.5  # Social parameter
    init_mode: str = 'random'
    init_samples: int = 128


# =============================================================================
# Main Interface
# =============================================================================


def run_pso_optimization(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: PSOConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Run Particle Swarm Optimization to fit linkage to target trajectory.

    PSO is a population-based optimizer that's good for:
      - Non-convex problems
      - Escaping local minima
      - Problems where gradients are unavailable

    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: PSO configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring.
            RECOMMENDED when target may start at different phase than simulation.
            Prevents false errors from phase mismatch at cost of O(n) per eval.
        phase_align_method: Phase alignment algorithm:
            - 'rotation': Brute-force, O(n²), guaranteed optimal (DEFAULT)
            - 'fft': FFT cross-correlation, O(n log n), fastest for large n
            - 'frechet': Fréchet distance, O(n³), avoid in optimization!
        **kwargs: Additional arguments for interface compatibility:
            - n_particles, iterations, init_mode, init_samples: Override config

    Returns:
        OptimizationResult with:
            - success: True if optimization completed
            - optimized_dimensions: Best found dimension values
            - optimized_pylink_data: Updated linkage with optimized dimensions
            - initial_error: Error before optimization
            - final_error: Best achieved error
            - iterations: Number of iterations
            - convergence_history: Best error per iteration

    Example:
        >>> from optimizers import run_pso_optimization, PSOConfig
        >>> # Standard random initialization
        >>> result = run_pso_optimization(pylink_data, target, n_particles=50)
        >>> # With Sobol presampling for better convergence
        >>> config = PSOConfig(n_particles=50, init_mode='sobol', init_samples=256)
        >>> result = run_pso_optimization(pylink_data, target, config=config)

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
    from pylink_tools.optimization_helpers import (
        apply_dimensions_from_array,
        dimensions_to_dict,
        extract_dimensions,
        presample_valid_positions,
    )
    from pylink_tools.optimization_types import OptimizationResult
    from pylink_tools.optimize import create_fitness_function

    # Use default config if not provided
    if config is None:
        config = PSOConfig()

    # Allow kwargs to override config
    n_particles = kwargs.get('n_particles', config.n_particles)
    iterations = kwargs.get('iterations', config.iterations)
    init_mode = kwargs.get('init_mode', config.init_mode)
    init_samples = kwargs.get('init_samples', config.init_samples)

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
            w=config.w,
            c1=config.c1,
            c2=config.c2,
            verbose=verbose,
            init_positions=init_positions,
        )

        # Build optimized pylink_data
        optimized_pylink_data = apply_dimensions_from_array(
            pylink_data,
            best_dims,
            dimension_spec,
        )

        # Create dimension dict
        optimized_dims = dimensions_to_dict(best_dims, dimension_spec)

        if verbose:
            logger.info(f'  Final error: {best_score:.4f}')
            if initial_error > 0 and initial_error != float('inf'):
                logger.info(f'  Improvement: {(1 - best_score / initial_error) * 100:.1f}%')

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
        if verbose:
            logger.error(f'PSO optimization failed: {e}', exc_info=True)
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            initial_error=initial_error,
            error=f'PSO optimization failed: {str(e)}',
        )


# =============================================================================
# Internal PSO Implementation
# =============================================================================


def _run_pso_internal(
    fitness_func: Callable[[tuple[float, ...]], float],
    initial_values: tuple[float, ...],
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
    n_particles: int = 32,
    iterations: int = 512,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
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
    positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, n_dims))

    if init_positions is not None and len(init_positions) > 0:
        # Use pre-sampled positions (already validated as producing valid mechanisms)
        n_presampled = min(len(init_positions), n_particles)
        positions[:n_presampled] = init_positions[:n_presampled]
        if verbose:
            logger.info(f'  Using {n_presampled} pre-sampled positions, ' f'{n_particles - n_presampled} random')
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
