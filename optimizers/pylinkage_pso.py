"""
Pylinkage-native PSO optimizer for linkage trajectory optimization.

Uses Mechanism class for maximum performance:
  - Compiles linkage once (no recompilation per evaluation)
  - Updates joint attributes + syncs to SolverData (avoids invalidation)
  - Uses step_fast() for 5x faster simulation than step()

Reference: https://github.com/HugoFara/pylinkage

License: pylinkage is GPL-3.0 (note: may have commercial restrictions)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism
    from pylink_tools.optimization_types import DimensionBoundsSpec, OptimizationResult, TargetTrajectory

import pylinkage as pl
from pylink_tools.mechanism import create_mechanism_fitness
from pylink_tools.optimization_types import OptimizationResult


logger = logging.getLogger(__name__)


@dataclass
class PylinkagePSOConfig:
    """Configuration for pylinkage PSO optimizer."""
    n_particles: int = 32
    iterations: int = 512
    init_mode: str = 'random'  # 'random', 'sobol', 'behnken'
    init_samples: int = 128
    init_spread: float = 0.25
    # PSO hyperparameters (match our custom PSO for better convergence)
    w: float = 0.7    # Inertia weight (pylinkage default: 0.6)
    c1: float = 1.5   # Cognitive/leader (pylinkage default: 3.0 - too high!)
    c2: float = 1.5   # Social/follower (pylinkage default: 0.1 - too low!)


def run_pylinkage_pso(
    mechanism: Mechanism,
    target: TargetTrajectory,
    dimension_bounds_spec: DimensionBoundsSpec | None = None,
    config: PylinkagePSOConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Run PSO using pylinkage's numba-optimized solver via Mechanism.

    Optimizations via Mechanism class:
      - Compiles once (no per-evaluation recompilation)
      - Direct joint attribute updates (faster than set_num_constraints)
      - Uses step_fast() with numba JIT (5x faster than step())

    Args:
        mechanism: Mechanism object to optimize (will be modified in place)
        target: Target trajectory to match (joint name + positions)
        dimension_bounds_spec: Dimensions to optimize (extracted from mechanism if not provided)
        config: Pylinkage PSO configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Phase alignment algorithm
        **kwargs: Additional arguments (n_particles, iterations, etc.)

    Returns:
        OptimizationResult with optimized dimensions and mechanism state
    """
    # Validate mechanism type
    from pylink_tools.mechanism import Mechanism as MechanismType
    if not isinstance(mechanism, MechanismType):
        return OptimizationResult(
            success=False, optimized_dimensions={},
            error=f'Expected Mechanism object, got {type(mechanism).__name__}. '
                  f'Convert pylink_data to Mechanism using create_mechanism_from_dict()',
        )

    config = config or PylinkagePSOConfig()
    n_particles = max(kwargs.get('n_particles', config.n_particles), 20)
    iterations = kwargs.get('iterations', config.iterations)
    init_mode = kwargs.get('init_mode', config.init_mode)
    init_samples = kwargs.get('init_samples', config.init_samples)
    init_spread = kwargs.get('init_spread', config.init_spread)
    # PSO hyperparameters
    w = kwargs.get('w', config.w)
    c1 = kwargs.get('c1', config.c1)
    c2 = kwargs.get('c2', config.c2)

    # Extract dimensions if not provided
    if dimension_bounds_spec is None:
        dimension_bounds_spec = mechanism.get_dimension_bounds_spec()

    if len(dimension_bounds_spec) == 0:
        return OptimizationResult(
            success=False, optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    # Ensure mechanism has correct n_steps
    if mechanism.n_steps != target.n_steps:
        mechanism._n_steps = target.n_steps

    if verbose:
        logger.info('Starting pylinkage PSO (Mechanism fast path)')
        logger.info(f'  Dimensions: {len(dimension_bounds_spec)}, Particles: {n_particles}, Iterations: {iterations}')

    # Create fast fitness function
    fitness = create_mechanism_fitness(
        mechanism=mechanism,
        target=target,
        target_joint=target.joint_name,
        metric=metric,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method,
    )

    # Wrapper for pylinkage PSO interface (expects linkage, params, init_pos)
    # We'll track history by wrapping the fitness function
    def fitness_func(linkage_obj, params, init_pos_arg=None):
        """Wrapper for pylinkage PSO - delegates to Mechanism fitness."""
        error = fitness(tuple(params))
        # Track best error (pylinkage PSO doesn't provide per-iteration history directly)
        # We'll update history after each iteration via callback if available
        return error

    # Compute initial error
    init_dims = np.array(dimension_bounds_spec.initial_values)
    initial_error = fitness(tuple(init_dims))
    if verbose:
        logger.info(f'  Initial error: {initial_error:.4f}')

    # Track convergence history
    convergence_history = [initial_error]
    best_error_so_far = initial_error

    # Initialize particles
    bounds = dimension_bounds_spec.get_bounds_tuple()
    lower, upper = np.array(bounds[0]), np.array(bounds[1])
    n_dims = len(dimension_bounds_spec)
    init_pos = np.zeros((n_particles, n_dims))

    if init_mode in ('sobol', 'behnken'):
        try:
            # presample_valid_positions still expects pylink_data, so convert mechanism
            # TODO: Refactor presample_valid_positions to take Mechanism directly
            pylink_data_for_presample = mechanism.to_dict()
            presampled, scores = presample_valid_positions(
                pylink_data_for_presample, target, dimension_bounds_spec, n_samples=init_samples,
                n_best=n_particles, mode=init_mode, metric=metric, phase_invariant=phase_invariant,
            )
            if len(presampled) > 0:
                n_pre = min(len(presampled), n_particles)
                init_pos[:n_pre] = presampled[:n_pre]
                for i in range(n_pre, n_particles):
                    init_pos[i] = np.clip(presampled[0] + (np.random.random(n_dims)-0.5)*0.3*(upper-lower), lower, upper)
                if verbose:
                    logger.info(f'  Pre-sampled {n_pre} positions (best: {scores[0]:.4f})')
            else:
                init_mode = 'random'
        except Exception:
            init_mode = 'random'

    if init_mode == 'random':
        init_pos[0] = init_dims
        for i in range(1, n_particles):
            init_pos[i] = np.clip(init_dims + (np.random.random(n_dims)-0.5)*2*init_spread*(upper-lower), lower, upper)

    # Run PSO via pylinkage wrapper (uses pyswarms LocalBestPSO)
    # Track history by monitoring best error across iterations
    try:
        # Track evaluations to approximate iterations
        evaluation_count = [0]
        last_iteration_recorded = [0]

        def tracked_fitness_func(linkage_obj, params, init_pos_arg=None):
            """Fitness function that tracks convergence history."""
            error = fitness_func(linkage_obj, params, init_pos_arg)
            evaluation_count[0] += 1

            # Track best error seen so far
            nonlocal best_error_so_far
            if error < best_error_so_far:
                best_error_so_far = error

            # Record history once per iteration (approximately)
            # Each iteration evaluates n_particles, so we record when we've evaluated
            # a multiple of n_particles
            current_iteration = evaluation_count[0] // n_particles
            if current_iteration > last_iteration_recorded[0] and current_iteration <= iterations:
                convergence_history.append(best_error_so_far)
                last_iteration_recorded[0] = current_iteration

            return error

        results = pl.particle_swarm_optimization(
            eval_func=tracked_fitness_func, linkage=mechanism.linkage, bounds=bounds,
            n_particles=n_particles, iters=iterations, order_relation=min,
            leader=c1, follower=c2, inertia=w,
            verbose=verbose, init_pos=init_pos,
        )

        if results:
            best_score, best_dims, _ = results[0]
            best_dims = tuple(best_dims)

            # Ensure final error is recorded
            if len(convergence_history) == 1 or convergence_history[-1] != best_score:
                convergence_history.append(best_score)

            # Ensure we have at least initial + iterations entries
            # If we have fewer, pad with the best score
            expected_length = iterations + 1
            while len(convergence_history) < expected_length:
                convergence_history.append(best_score)
            # Trim if we have too many
            convergence_history = convergence_history[:expected_length]

            if verbose:
                logger.info(f'  Final error: {best_score:.4f}')
                if initial_error > 0 and initial_error != float('inf'):
                    logger.info(f'  Improvement: {(1 - best_score / initial_error) * 100:.1f}%')

            # Update mechanism with optimized dimensions and return copy
            mechanism.set_dimensions(best_dims)
            optimized_mechanism = mechanism.copy()

            # Create dimension dict: {name: value} mapping
            optimized_dims = dict(zip(dimension_bounds_spec.names, best_dims))

            return OptimizationResult(
                success=True,
                optimized_dimensions=optimized_dims,
                optimized_mechanism=optimized_mechanism,
                initial_error=initial_error,
                final_error=best_score,
                iterations=iterations,
                convergence_history=convergence_history,
            )
        else:
            return OptimizationResult(
                success=False, optimized_dimensions={},
                initial_error=initial_error,
                convergence_history=convergence_history if len(convergence_history) > 1 else None,
                error='PSO returned no results',
            )

    except Exception as e:
        if verbose:
            logger.error(f'PSO failed: {e}', exc_info=True)
        return OptimizationResult(
            success=False, optimized_dimensions={},
            initial_error=initial_error,
            convergence_history=convergence_history if len(convergence_history) > 1 else None,
            error=f'PSO failed: {e}',
        )
