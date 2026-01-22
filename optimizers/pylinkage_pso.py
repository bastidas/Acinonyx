"""
Pylinkage-native PSO optimizer for linkage trajectory optimization.

Uses pylinkage's numba-optimized solver for maximum performance:
  - Compiles linkage once (no recompilation per evaluation)
  - Updates joint attributes + syncs to SolverData (avoids invalidation)
  - Uses step_fast() for 5x faster simulation than step()

Reference: https://github.com/HugoFara/pylinkage

License: pylinkage is GPL-3.0 (note: may have commercial restrictions)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.optimization_types import DimensionSpec, OptimizationResult, TargetTrajectory

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


def build_joint_attr_mapping(linkage, pylink_data: dict, dimension_spec: DimensionSpec):
    """
    Build mapping from dimension index to (joint, attribute_name).

    This allows direct joint attribute updates without invalidating solver_data.
    """
    from pylinkage.joints import Crank, Revolute, Static

    edges = pylink_data.get('linkage', {}).get('edges', {})
    edge_mapping = getattr(dimension_spec, 'edge_mapping', {}) or {}

    # Build (source, target) -> edge_id
    edge_by_nodes = {}
    for edge_id, edge in edges.items():
        edge_by_nodes[(edge['source'], edge['target'])] = edge_id
        edge_by_nodes[(edge['target'], edge['source'])] = edge_id

    # Build edge_id -> (joint, attr_name)
    edge_to_joint_attr = {}
    for joint in linkage.joints:
        if isinstance(joint, Static) and not isinstance(joint, Crank):
            continue
        elif isinstance(joint, Crank):
            if joint.joint0:
                key = (joint.joint0.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r')
        elif isinstance(joint, Revolute):
            if joint.joint0:
                key = (joint.joint0.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r0')
            if joint.joint1:
                key = (joint.joint1.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r1')

    # Map dimension index to (joint, attr)
    mapping = []
    for dim_name in dimension_spec.names:
        if dim_name in edge_mapping:
            edge_id, _ = edge_mapping[dim_name]
            mapping.append(edge_to_joint_attr.get(edge_id))
        else:
            mapping.append(None)

    return mapping


def run_pylinkage_pso(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: PylinkagePSOConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Run PSO using pylinkage's numba-optimized solver.

    Optimizations:
      - Compiles once (no per-evaluation recompilation)
      - Direct joint attribute updates (faster than set_num_constraints)
      - Uses step_fast() with numba JIT (5x faster than step())
    """
    import pylinkage as pl
    from pylinkage.joints import Crank
    from pylinkage.bridge.solver_conversion import update_solver_constraints, update_solver_positions

    from pylink_tools.hypergraph_adapter import is_our_hypergraph_format, to_simulatable_linkage
    from pylink_tools.optimization_helpers import (
        apply_dimensions_from_array, dimensions_to_dict, extract_dimensions, presample_valid_positions,
    )
    from pylink_tools.optimization_types import OptimizationResult
    from pylink_tools.trajectory_utils import compute_trajectory_error

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

    if not is_our_hypergraph_format(pylink_data):
        return OptimizationResult(
            success=False, optimized_dimensions={},
            error='Legacy format not supported. Use linkage.nodes/edges format.',
        )

    dimension_spec = dimension_spec or extract_dimensions(pylink_data)
    if len(dimension_spec) == 0:
        return OptimizationResult(
            success=False, optimized_dimensions={},
            error='No optimizable dimensions found',
        )

    n_steps = target.n_steps
    angle_per_step = 2 * math.pi / n_steps

    # Build and compile linkage once
    try:
        linkage = to_simulatable_linkage(pylink_data)
        for joint in linkage.joints:
            if isinstance(joint, Crank):
                joint.angle = angle_per_step
        linkage.rebuild()
        linkage.compile()  # Creates _solver_data (numpy arrays)
    except Exception as e:
        return OptimizationResult(
            success=False, optimized_dimensions={},
            error=f'Failed to build linkage: {e}',
        )

    # Build fast mapping: dim_idx -> (joint, attr_name)
    joint_attr_mapping = build_joint_attr_mapping(linkage, pylink_data, dimension_spec)

    # Save initial state for reset
    init_positions = linkage.get_coords()
    init_dims = np.array(dimension_spec.initial_values)

    # Find target joint
    target_idx = next(
        (i for i, j in enumerate(linkage.joints) if j.name == target.joint_name),
        len(linkage.joints) - 1,
    )

    if verbose:
        logger.info('Starting pylinkage PSO (numba-optimized, no recompilation)')
        logger.info(f'  Dimensions: {len(dimension_spec)}, Particles: {n_particles}, Iterations: {iterations}')

    def fitness_func(linkage_obj, params, init_pos_arg=None):
        """Ultra-fast fitness: direct joint updates + step_fast()."""
        try:
            # Update joint attributes directly (doesn't invalidate _solver_data)
            for i, ja in enumerate(joint_attr_mapping):
                if ja:
                    setattr(ja[0], ja[1], params[i])

            # Sync constraints to solver data (fast numpy update)
            update_solver_constraints(linkage_obj._solver_data, linkage_obj)

            # Reset positions
            linkage_obj.set_coords(init_positions)
            update_solver_positions(linkage_obj._solver_data, linkage_obj)

            # Run simulation (no recompilation!)
            trajectory = linkage_obj.step_fast(iterations=n_steps)

            if np.isnan(trajectory).any():
                return float('inf')

            # Extract target joint trajectory
            joint_traj = trajectory[:, target_idx, :]
            computed = [(joint_traj[i, 0], joint_traj[i, 1]) for i in range(n_steps)]

            if phase_invariant:
                return compute_trajectory_error(
                    computed, target, metric=metric,
                    phase_invariant=True, phase_align_method=phase_align_method,
                )
            else:
                total = sum((c[0]-t[0])**2 + (c[1]-t[1])**2 for c, t in zip(computed, target.positions))
                return total / len(target.positions) if metric == 'mse' else total
        except Exception:
            return float('inf')

    # Compute initial error
    initial_error = fitness_func(linkage, tuple(init_dims))
    if verbose:
        logger.info(f'  Initial error: {initial_error:.4f}')

    # Initialize particles
    bounds = dimension_spec.get_bounds_tuple()
    lower, upper = np.array(bounds[0]), np.array(bounds[1])
    n_dims = len(dimension_spec)
    init_pos = np.zeros((n_particles, n_dims))

    if init_mode in ('sobol', 'behnken'):
        try:
            presampled, scores = presample_valid_positions(
                pylink_data, target, dimension_spec, n_samples=init_samples,
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
    try:
        results = pl.particle_swarm_optimization(
            eval_func=fitness_func, linkage=linkage, bounds=bounds,
            n_particles=n_particles, iters=iterations, order_relation=min,
            leader=c1, follower=c2, inertia=w,
            verbose=verbose, init_pos=init_pos,
        )

        if results:
            best_score, best_dims, _ = results[0]
            best_dims = tuple(best_dims)

            if verbose:
                logger.info(f'  Final error: {best_score:.4f}')
                if initial_error > 0 and initial_error != float('inf'):
                    logger.info(f'  Improvement: {(1 - best_score / initial_error) * 100:.1f}%')

            return OptimizationResult(
                success=True,
                optimized_dimensions=dimensions_to_dict(best_dims, dimension_spec),
                optimized_pylink_data=apply_dimensions_from_array(pylink_data, best_dims, dimension_spec),
                initial_error=initial_error,
                final_error=best_score,
                iterations=iterations,
            )
        else:
            return OptimizationResult(
                success=False, optimized_dimensions={},
                initial_error=initial_error, error='PSO returned no results',
            )

    except Exception as e:
        if verbose:
            logger.error(f'PSO failed: {e}', exc_info=True)
        return OptimizationResult(
            success=False, optimized_dimensions={},
            initial_error=initial_error, error=f'PSO failed: {e}',
        )
