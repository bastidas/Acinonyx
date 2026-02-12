#!/usr/bin/env python3
"""
Solver Comparison Demo - Compare different optimization solvers.

WHAT THIS DEMO DOES:
====================
Runs the same optimization problem with different solver configurations
and compares their performance on:
- Final error achieved
- Computation time
- Convergence behavior
- Dimension recovery accuracy

COMPARISON MODES:
=================
Change COMPARISON_MODE at the top to switch what's being compared:

- 'all':           Test all optimizers from AVAILABLE_OPTIMIZERS
- 'init_modes':    Compare random vs Sobol initialization
- 'phase_methods': Compare rotation vs FFT phase alignment
- 'solvers':       Compare main solver types (one of each category)
- 'nlopt_mlsl':    Compare NLopt MLSL variants
- 'quick':         Fast test with just 2 solvers

RUN THIS DEMO:
==============
    python demo/comparison_demo.py

Output saved to: user/demo/solver_comparison/
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism

from demo.helpers import load_mechanism
from demo.helpers import print_section
from optimizers import AVAILABLE_OPTIMIZERS
from optimizers.mlsl_optimizer import NLoptMLSLConfig
from pylink_tools.optimization_types import TargetTrajectory
from target_gen import create_achievable_target
from target_gen import MechVariationConfig
from target_gen import DimensionVariationConfig
from target_gen import verify_mechanism_viable
from viz_tools.demo_viz import plot_convergence_comparison
from viz_tools.demo_viz import plot_dimension_bounds
from viz_tools.demo_viz import variation_plot


# =============================================================================
# CONFIGURATION
# =============================================================================

# Which mechanism to optimize
MECHANISM = 'simple'  # Options: 'simple', 'intermediate', 'complex', 'leg'

# What to compare (see docstring for options)
COMPARISON_MODE = 'all'  # 'all', 'init_modes', 'phase_methods', 'solvers', 'nlopt_mlsl', 'quick'

# Note: Set to 'quick' for faster testing, 'all' to test all optimizers

# Optimization parameters
N_PARTICLES = 64       # PSO particles
N_ITERATIONS = 64      # Optimization iterations
MAX_EVAL = 128         # MLSL function evaluations

# Bounds and target generation
BOUNDS_FACTOR = 1.2    # How much dimensions can vary
MIN_LENGTH = 4.0       # Minimum link length
VARIATION_RANGE = 0.5  # Target randomization (±50%)

# Reproducibility
RANDOM_SEED = 42

# Output
OUTPUT_DIR = Path(__file__).parent.parent / 'user' / 'demo' / 'solver_comparison'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# =============================================================================
# SOLVER CONFIGURATIONS
# =============================================================================

@dataclass
class SolverResult:
    """Container for solver results."""
    name: str
    success: bool
    initial_error: float
    final_error: float
    elapsed_time: float
    iterations: int
    optimized_dimensions: dict
    optimized_mechanism: Mechanism | None
    convergence_history: list | None
    error_message: str | None = None


def get_solver_configs(mode: str) -> list[tuple[str, callable, dict]]:
    """
    Get solver configurations based on comparison mode.

    Returns list of (name, solver_func, kwargs) tuples.
    """
    if mode == 'all':
        # Test all optimizers from AVAILABLE_OPTIMIZERS
        configs = []
        for opt_name, opt_info in AVAILABLE_OPTIMIZERS.items():
            solver_func = opt_info['function']
            config_class = opt_info.get('config_class')

            # Build kwargs based on optimizer type
            kwargs = {}
            if config_class:
                if opt_name in ('nlopt', 'nlopt_gf'):
                    kwargs['config'] = config_class(max_eval=MAX_EVAL)
                elif opt_name == 'scipy':
                    # Default scipy uses L-BFGS-B
                    kwargs['config'] = config_class(method='L-BFGS-B', max_iterations=N_ITERATIONS)
                elif opt_name in ('l-bfgs-b', 'powell', 'nelder-mead'):
                    # Scipy variants with specific methods
                    method_name = opt_name.upper()
                    if opt_name == 'l-bfgs-b':
                        method_name = 'L-BFGS-B'
                    elif opt_name == 'nelder-mead':
                        method_name = 'Nelder-Mead'
                    kwargs['config'] = config_class(method=method_name, max_iterations=N_ITERATIONS)
                elif opt_name == 'scip':
                    kwargs['config'] = config_class()  # Uses defaults
                elif opt_name == 'pylinkage':
                    kwargs['config'] = config_class(n_particles=N_PARTICLES, iterations=N_ITERATIONS)

            configs.append((opt_info['description'], solver_func, kwargs))

        return configs

    elif mode == 'init_modes':
        # Compare initialization modes for PSO-based optimizers
        configs = []
        for opt_name, opt_info in AVAILABLE_OPTIMIZERS.items():
            # Find PSO-based optimizers (pylinkage has init_mode support)
            if opt_name == 'pylinkage':
                solver_func = opt_info['function']
                config_class = opt_info.get('config_class')
                if config_class:
                    for init_mode in ('random', 'sobol'):
                        kwargs = {
                            'config': config_class(n_particles=N_PARTICLES, iterations=N_ITERATIONS, init_mode=init_mode),
                        }
                        configs.append((f'{opt_info["description"]} ({init_mode})', solver_func, kwargs))
        return configs

    elif mode == 'phase_methods':
        # Compare phase alignment methods
        from optimizers.scipy_optimizer import ScipyConfig
        configs = []
        # Test a few key optimizers with different phase methods
        test_optimizers = ['l-bfgs-b', 'pylinkage', 'nlopt']
        for opt_name in test_optimizers:
            if opt_name not in AVAILABLE_OPTIMIZERS:
                continue
            opt_info = AVAILABLE_OPTIMIZERS[opt_name]
            solver_func = opt_info['function']
            config_class = opt_info.get('config_class')
            for phase_method in ('rotation', 'fft'):
                kwargs = {'phase_align_method': phase_method}
                if config_class:
                    if opt_name == 'l-bfgs-b':
                        kwargs['config'] = ScipyConfig(method='L-BFGS-B', max_iterations=N_ITERATIONS)
                    elif opt_name == 'pylinkage':
                        kwargs['config'] = config_class(n_particles=N_PARTICLES, iterations=N_ITERATIONS)
                    elif opt_name == 'nlopt':
                        kwargs['config'] = config_class(max_eval=MAX_EVAL)
                configs.append((f'{opt_info["description"]} ({phase_method})', solver_func, kwargs))
        return configs

    elif mode == 'solvers':
        # Compare main solver types (one of each category)
        from optimizers.scipy_optimizer import ScipyConfig
        return [
            (
                'Scipy L-BFGS-B', AVAILABLE_OPTIMIZERS['l-bfgs-b']['function'], {
                    'config': ScipyConfig(method='L-BFGS-B', max_iterations=N_ITERATIONS),
                },
            ),
            (
                'Pylinkage PSO', AVAILABLE_OPTIMIZERS['pylinkage']['function'], {
                    'config': AVAILABLE_OPTIMIZERS['pylinkage']['config_class'](n_particles=N_PARTICLES, iterations=N_ITERATIONS),
                },
            ),
            (
                'NLopt MLSL', AVAILABLE_OPTIMIZERS['nlopt']['function'], {
                    'config': AVAILABLE_OPTIMIZERS['nlopt']['config_class'](max_eval=MAX_EVAL),
                },
            ),
        ]

    elif mode == 'nlopt_mlsl':
        return [
            (
                'MLSL (lbfgs)', AVAILABLE_OPTIMIZERS['nlopt']['function'], {
                    'config': NLoptMLSLConfig(use_lds=False, local_algorithm='lbfgs', max_eval=MAX_EVAL),
                },
            ),
            (
                'MLSL (bobyqa)', AVAILABLE_OPTIMIZERS['nlopt']['function'], {
                    'config': NLoptMLSLConfig(use_lds=False, local_algorithm='bobyqa', max_eval=MAX_EVAL),
                },
            ),
            (
                'MLSL+LDS (lbfgs)', AVAILABLE_OPTIMIZERS['nlopt']['function'], {
                    'config': NLoptMLSLConfig(use_lds=True, local_algorithm='lbfgs', max_eval=MAX_EVAL),
                },
            ),
            (
                'MLSL+LDS (bobyqa)', AVAILABLE_OPTIMIZERS['nlopt']['function'], {
                    'config': NLoptMLSLConfig(use_lds=True, local_algorithm='bobyqa', max_eval=MAX_EVAL),
                },
            ),
        ]

    elif mode == 'quick':
        from optimizers.scipy_optimizer import ScipyConfig
        return [
            (
                'Scipy L-BFGS-B', AVAILABLE_OPTIMIZERS['l-bfgs-b']['function'], {
                    'config': ScipyConfig(method='L-BFGS-B', max_iterations=N_ITERATIONS),
                },
            ),
            (
                'Pylinkage PSO', AVAILABLE_OPTIMIZERS['pylinkage']['function'], {
                    'config': AVAILABLE_OPTIMIZERS['pylinkage']['config_class'](n_particles=N_PARTICLES, iterations=N_ITERATIONS),
                },
            ),
        ]

    else:
        available = ['all', 'init_modes', 'phase_methods', 'solvers', 'nlopt_mlsl', 'quick']
        raise ValueError(f"Unknown mode '{mode}'. Available: {available}")


def run_solver(
    name: str,
    solver_func: callable,
    mechanism: Mechanism,
    target: TargetTrajectory,
    dim_spec,
    **kwargs,
) -> SolverResult:
    """Run a single solver with timing."""
    print(f'\n  Running {name}...')

    # Create a copy of the mechanism for this solver (optimizers modify in place)
    working_mechanism = mechanism.copy()

    # Build solver arguments
    solver_kwargs = {
        'mechanism': working_mechanism,
        'target': target,
        'dimension_bounds_spec': dim_spec,
        'metric': 'mse',
        'verbose': True,
        'phase_invariant': True,
        'phase_align_method': 'rotation',
    }

    # Add optimizer-specific params (if not already in config)
    # Note: Most optimizers now use config objects, so these are fallbacks
    # Only add if config is not provided (configs should contain all needed params)
    if 'config' not in kwargs:
        # Try to detect optimizer type from function name
        func_name = solver_func.__name__
        if 'pso' in func_name.lower() or 'pylinkage' in func_name.lower():
            solver_kwargs['n_particles'] = N_PARTICLES
            solver_kwargs['iterations'] = N_ITERATIONS
            solver_kwargs['init_mode'] = 'random'
            solver_kwargs['init_samples'] = 128
        elif 'scipy' in func_name.lower():
            solver_kwargs['max_iterations'] = N_ITERATIONS
        elif 'nlopt' in func_name.lower() or 'mlsl' in func_name.lower():
            solver_kwargs['max_eval'] = MAX_EVAL

    # Override with custom kwargs
    solver_kwargs.update(kwargs)

    start_time = time.time()
    try:
        result = solver_func(**solver_kwargs)
        elapsed = time.time() - start_time

        return SolverResult(
            name=name,
            success=result.success,
            initial_error=result.initial_error,
            final_error=result.final_error,
            elapsed_time=elapsed,
            iterations=result.iterations or N_ITERATIONS,
            optimized_dimensions=result.optimized_dimensions,
            optimized_mechanism=working_mechanism if result.success else None,
            convergence_history=result.convergence_history,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        print(f'    ERROR: {e}')
        return SolverResult(
            name=name,
            success=False,
            initial_error=float('inf'),
            final_error=float('inf'),
            elapsed_time=elapsed,
            iterations=0,
            optimized_dimensions={},
            optimized_mechanism=None,
            convergence_history=None,
            error_message=str(e),
        )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_results_table(results: list[SolverResult]):
    """Print comparison table."""
    print('\n' + '-' * 90)
    print(f"{'Solver':<25} {'Success':>8} {'Init Err':>12} {'Final Err':>12} {'Time (s)':>10} {'Improve%':>10}")
    print('-' * 90)

    for r in results:
        improvement = (1 - r.final_error / r.initial_error) * 100 if r.initial_error > 0 else 0
        final_str = f'{r.final_error:>12.4f}' if r.final_error != float('inf') else '         inf'
        print(f'{r.name:<25} {str(r.success):>8} {r.initial_error:>12.4f} {final_str} {r.elapsed_time:>10.2f} {improvement:>9.1f}%')

    print('-' * 90)


def print_dimension_table(results: list[SolverResult], dim_spec, initial_dims, target_dims):
    """Print dimension recovery table."""
    print('\nDimension Recovery:')
    print(f"{'Dimension':<30} {'Target':>10}", end='')
    for r in results:
        short_name = r.name[:12] + '..' if len(r.name) > 14 else r.name
        print(f' {short_name:>14}', end='')
    print()

    print('-' * (45 + 15 * len(results)))

    for name in dim_spec.names:
        target_val = target_dims[name]
        print(f'{name:<30} {target_val:>10.2f}', end='')
        for r in results:
            opt_val = r.optimized_dimensions.get(name, initial_dims[name])
            error = abs(opt_val - target_val)
            print(f' {opt_val:>8.2f}({error:>3.1f})', end='')
        print()


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print_section('SOLVER COMPARISON DEMO')
    print(f'Mechanism: {MECHANISM}')
    print(f'Mode: {COMPARISON_MODE}')
    print(f'Output: {OUTPUT_DIR}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load mechanism
    # -------------------------------------------------------------------------
    print_section('Step 1: Load Mechanism')

    mechanism, target_joint, description = load_mechanism(MECHANISM)

    # Verify mechanism is viable
    if not verify_mechanism_viable(mechanism):
        raise RuntimeError('Mechanism failed viability check!')

    dim_spec = mechanism.get_dimension_bounds_spec()
    initial_dims = dict(zip(dim_spec.names, dim_spec.initial_values))

    print(f'\n{description}')
    print(f'Target joint: {target_joint}')
    print(f'Dimensions: {len(dim_spec)}')

    # -------------------------------------------------------------------------
    # Define optimization bounds configuration
    # -------------------------------------------------------------------------
    # Create MechVariationConfig for optimization bounds
    optimization_config = MechVariationConfig(
        dimension_variation=DimensionVariationConfig(
            default_variation_range=BOUNDS_FACTOR,
        ),
    )

    # Convert to DimensionBoundsSpec for optimizers
    from pylink_tools.optimization_types import DimensionBoundsSpec
    optimization_bounds_spec = DimensionBoundsSpec.from_mechanism(
        mechanism, optimization_config,
    )

    print(f'\nOptimization bounds: ±{BOUNDS_FACTOR*100:.0f}% variation allowed')
    print(f'  Optimizing {len(optimization_bounds_spec)} dimensions')

    # -------------------------------------------------------------------------
    # Step 2: Create target
    # -------------------------------------------------------------------------
    print_section('Step 2: Create Target')

    # Use smaller range for complex mechanisms
    effective_range = min(VARIATION_RANGE, 0.25) if MECHANISM in ('complex', 'intermediate') else VARIATION_RANGE
    print(f'\nRandomizing dimensions by ±{effective_range*100:.0f}%...')

    # Create MechVariationConfig for target generation
    target_config = MechVariationConfig(
        dimension_variation=DimensionVariationConfig(
            default_variation_range=effective_range,
        ),
        random_seed=RANDOM_SEED,
    )

    target_result = create_achievable_target(
        mechanism, target_joint, dim_spec=dim_spec, config=target_config,
    )
    target = target_result.target
    target_dims = target_result.target_dimensions

    print(f'Target trajectory: {target.n_steps} points')

    # Compute initial trajectory
    initial_traj = mechanism.get_trajectory(target_joint)

    # -------------------------------------------------------------------------
    # Step 3: Run solvers
    # -------------------------------------------------------------------------
    print_section('Step 3: Run Solvers')

    solver_configs = get_solver_configs(COMPARISON_MODE)
    print(f'\nSolvers to compare: {len(solver_configs)}')
    for name, _, kwargs in solver_configs:
        kwargs_str = ', '.join(f'{k}' for k in kwargs.keys()) if kwargs else 'defaults'
        print(f'  - {name}: {kwargs_str}')

    results = []
    for name, solver_func, kwargs in solver_configs:
        result = run_solver(name, solver_func, mechanism, target, optimization_bounds_spec, **kwargs)
        results.append(result)

    # -------------------------------------------------------------------------
    # Step 4: Compare results
    # -------------------------------------------------------------------------
    print_section('Step 4: Results')

    print_results_table(results)
    print_dimension_table(results, dim_spec, initial_dims, target_dims)

    # Find best
    successful = [r for r in results if r.success]
    if successful:
        best = min(successful, key=lambda r: r.final_error)
        fastest = min(successful, key=lambda r: r.elapsed_time)
        print(f'\nBest error: {best.name} ({best.final_error:.6f})')
        print(f'Fastest: {fastest.name} ({fastest.elapsed_time:.2f}s)')

    # -------------------------------------------------------------------------
    # Step 5: Visualize
    # -------------------------------------------------------------------------
    print_section('Step 5: Visualize')

    # Collect trajectories
    initial_mechanism = mechanism.copy()  # Preserve initial state
    trajectories = {'Initial': initial_traj}
    variation_mechanisms = []
    for r in results:
        if r.success and r.optimized_mechanism:
            opt_traj = r.optimized_mechanism.get_trajectory(target_joint)
            if opt_traj is not None and len(opt_traj) > 0:
                trajectories[r.name] = opt_traj
                variation_mechanisms.append(r.optimized_mechanism)

    # Plot trajectory comparison
    if len(trajectories) > 1:
        variation_plot(
            target_joint=target_joint,
            out_path=OUTPUT_DIR / f'trajectories_{TIMESTAMP}.png',
            base_mechanism=initial_mechanism,
            target_trajectory=target,
            variation_mechanisms=variation_mechanisms,
            title='Solver Comparison: Trajectories',
        )

    # Plot convergence
    histories = {r.name: r.convergence_history for r in results if r.convergence_history}
    if histories:
        plot_convergence_comparison(
            histories,
            out_path=OUTPUT_DIR / f'convergence_{TIMESTAMP}.png',
            title='Convergence Comparison',
            layout='overlay',
        )

    # Plot dimension bounds (best result)
    if successful:
        best = min(successful, key=lambda r: r.final_error)
        plot_dimension_bounds(
            dim_spec,
            out_path=OUTPUT_DIR / f'bounds_{TIMESTAMP}.png',
            initial_values=initial_dims,
            target_values=target_dims,
            optimized_values=best.optimized_dimensions,
            title=f'Dimension Recovery (Best: {best.name})',
        )

    # Save results JSON
    result_data = {
        'timestamp': TIMESTAMP,
        'mechanism': MECHANISM,
        'mode': COMPARISON_MODE,
        'target_dimensions': target_dims,
        'results': [
            {
                'name': r.name,
                'success': r.success,
                'final_error': r.final_error,
                'elapsed_time': r.elapsed_time,
            }
            for r in results
        ],
    }

    with open(OUTPUT_DIR / f'results_{TIMESTAMP}.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    # Summary
    print_section('COMPLETE')
    print('\nOutput files:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
