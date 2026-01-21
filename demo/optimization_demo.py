"""
optimization_demo.py - Demo script for linkage trajectory optimization.

This script demonstrates the full optimization workflow:
1. Load/create a test 4-bar linkage mechanism
2. Create a target trajectory by randomizing link dimensions (and validating it is solvable)
3. Run optimization to find the correct dimensions
4. Visualize results and save to user/demo/


approach:
  - Randomize the link dimensions
  - Compute the trajectory with those randomized dimensions
  - Use that as the target
  - Start from the original dimensions and try to find the randomized ones
  - This is an "inverse problem" with a KNOWN achievable solution

=============================================================================
HYPERPARAMETERS AND CONSTANTS
=============================================================================

PSO (Particle Swarm Optimization) Parameters:
- N_PARTICLES: Number of particles in the swarm (higher = more exploration, slower)
  Recommended: 30-100 for most problems. More particles help escape local minima.

- N_ITERATIONS: Number of PSO iterations (higher = better convergence, slower)
  Recommended: 50-200. Watch convergence history to tune.

Bounds Parameters:
- BOUNDS_FACTOR: How much dimensions can vary from initial values.
  e.g., 2.0 means bounds are [value/2, value*2]
  Tighter bounds = faster convergence but may miss optimal

- MIN_LENGTH: Minimum allowed link length. Prevents degenerate mechanisms.

Target Generation:
- DIMENSION_RANDOMIZE_RANGE: How much to randomize each dimension
  e.g., 0.3 means ±30% of original value

=============================================================================
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from configs.appconfig import USER_DIR
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import analyze_convergence
from pylink_tools.optimize import apply_dimensions
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimize import format_convergence_report
from pylink_tools.optimize import PSOConfig
from pylink_tools.optimize import run_pso_optimization
from pylink_tools.optimize import TargetTrajectory
from viz_tools.opt_viz import plot_convergence_history
from viz_tools.opt_viz import plot_dimension_bounds
from viz_tools.opt_viz import plot_linkage_state
from viz_tools.opt_viz import plot_optimization_summary
from viz_tools.opt_viz import plot_trajectory_comparison
from viz_tools.opt_viz import plot_trajectory_overlay
from configs.appconfig import USER_DIR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONSTANTS AND HYPERPARAMETERS
# =============================================================================

# --- Random Seed (set for reproducibility, or None for random) ---
RANDOM_SEED = 42  # Set to None for truly random results

# --- PSO Optimization Parameters ---
N_PARTICLES = 64          # Number of particles in swarm (32-128 typical)
N_ITERATIONS = 200        # Number of optimization iterations (50-500 typical)

# --- Bounds Parameters ---
BOUNDS_FACTOR = 2.5       # Bounds = [value/factor, value*factor]
MIN_LENGTH = 5.0          # Minimum link length (prevents degeneracy)

# --- Trajectory Parameters ---
N_STEPS = 24              # Number of simulation timesteps per revolution

# --- Error Metric ---
METRIC = 'mse'            # Error metric: "mse", "rmse", "total", "max"

# --- Target Generation ---
# NOTE: Not all dimension combinations produce valid mechanisms
# Larger ranges increase difficulty; smaller ranges are more likely to be valid.
DIMENSION_RANDOMIZE_RANGE = 0.35  # ±35% of original value (conservative)

# --- Output ---
OUTPUT_DIR = USER_DIR / 'demo'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def load_test_4bar():
    """Load the test 4-bar linkage from tests/4bar_test.json."""
    test_file = project_root / 'tests' / '4bar_test.json'

    if not test_file.exists():
        raise FileNotFoundError(f'Test file not found: {test_file}')

    with open(test_file) as f:
        pylink_data = json.load(f)

    pylink_data['n_steps'] = N_STEPS
    return pylink_data


def create_achievable_target(
    pylink_data: dict,
    target_joint: str,
    dim_spec,
    randomize_range: float = 0.5,
    seed: int = None,
    max_attempts: int = 128,
) -> tuple:
    """
    Create a target trajectory that is ACHIEVABLE by randomizing dimensions.

    The key insight: instead of shifting the trajectory (which is impossible
    because fixed pivots can't move), we:
    1. Randomize each dimension by ±randomize_range (e.g., ±50%)
    2. Validate the mechanism is still solvable
    3. Compute the trajectory with those randomized dimensions
    4. Return that as the target

    This creates an "inverse problem" with a viable and known solution

    IMPORTANT: Not all dimension combinations result in valid mechanisms.
    We retry until we find valid dimensions.

    Args:
        pylink_data: Original mechanism
        target_joint: Which joint's trajectory to target
        dim_spec: DimensionSpec for the mechanism
        randomize_range: How much to randomize (0.5 = ±50%)
        seed: Random seed for reproducibility
        max_attempts: Maximum attempts to find valid dimensions

    Returns:
        (target, target_dimensions, target_pylink_data)
        - target: TargetTrajectory object
        - target_dimensions: Dict of the randomized dimension values
        - target_pylink_data: The mechanism with randomized dimensions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Try to find valid randomized dimensions
    for attempt in range(max_attempts):
        # Randomize each dimension
        target_dims = {}
        for name, initial, bounds in zip(dim_spec.names, dim_spec.initial_values, dim_spec.bounds):
            # Random factor between (1-range) and (1+range)
            factor = 1.0 + random.uniform(-randomize_range, randomize_range)
            new_value = initial * factor

            # Clamp to bounds
            new_value = max(bounds[0], min(bounds[1], new_value))
            target_dims[name] = new_value

        # Apply randomized dimensions to create target mechanism
        target_pylink_data = apply_dimensions(pylink_data, target_dims)

        # Compute trajectory with randomized dimensions
        # IMPORTANT: Use skip_sync=True to use our randomized dimensions!
        result = compute_trajectory(target_pylink_data, verbose=False, skip_sync=True)

        if result.success and target_joint in result.trajectories:
            target_traj = result.trajectories[target_joint]

            target = TargetTrajectory(
                joint_name=target_joint,
                positions=[tuple(pos) for pos in target_traj],
            )

            if attempt > 0:
                print(f'  (Found valid target dimensions after {attempt + 1} attempts)')

            return target, target_dims, target_pylink_data

    # If we couldn't find valid dimensions with full range, try smaller range
    print(f"  Warning: Couldn't find valid dimensions with ±{randomize_range*100:.0f}%")
    print('  Trying with smaller range...')

    for smaller_range in [0.15, 0.10, 0.05]:
        for attempt in range(max_attempts):
            target_dims = {}
            for name, initial, bounds in zip(dim_spec.names, dim_spec.initial_values, dim_spec.bounds):
                factor = 1.0 + random.uniform(-smaller_range, smaller_range)
                new_value = initial * factor
                new_value = max(bounds[0], min(bounds[1], new_value))
                target_dims[name] = new_value

            target_pylink_data = apply_dimensions(pylink_data, target_dims)
            result = compute_trajectory(target_pylink_data, verbose=False, skip_sync=True)

            if result.success and target_joint in result.trajectories:
                target_traj = result.trajectories[target_joint]
                target = TargetTrajectory(
                    joint_name=target_joint,
                    positions=[tuple(pos) for pos in target_traj],
                )
                print(f'  Found valid dimensions with ±{smaller_range*100:.0f}%')
                return target, target_dims, target_pylink_data

    raise ValueError('Could not find valid target dimensions after many attempts')


def print_section(title: str):
    """Print a formatted section header."""
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def print_dimension_comparison(dim_spec, initial_dims, target_dims, optimized_dims):
    """Print a comparison table of dimensions."""
    print('\n' + '-' * 70)
    print(f"{'Dimension':<35} {'Initial':>10} {'Target':>10} {'Optimized':>10} {'Error':>10}")
    print('-' * 70)

    for name in dim_spec.names:
        initial = initial_dims[name]
        target = target_dims[name]
        optimized = optimized_dims.get(name, initial)
        error = abs(optimized - target)

        print(f'{name:<35} {initial:>10.2f} {target:>10.2f} {optimized:>10.2f} {error:>10.4f}')

    print('-' * 70)


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run the optimization demo."""

    print_section('LINKAGE OPTIMIZATION DEMO')
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'Timestamp: {TIMESTAMP}')
    if RANDOM_SEED is not None:
        print(f'Random seed: {RANDOM_SEED}')

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load the test 4-bar linkage
    # -------------------------------------------------------------------------
    print_section('Step 1: Load Test Mechanism')

    pylink_data = load_test_4bar()

    print('Loaded 4-bar linkage:')
    joints = pylink_data['pylinkage']['joints']
    for j in joints:
        if j['type'] == 'Static':
            print(f"  - {j['name']} ({j['type']}) at ({j['x']}, {j['y']})")
        elif j['type'] == 'Crank':
            print(f"  - {j['name']} ({j['type']}) distance={j['distance']}")
        elif j['type'] == 'Revolute':
            print(f"  - {j['name']} ({j['type']}) d0={j['distance0']}, d1={j['distance1']}")

    # Extract dimensions
    dim_spec = extract_dimensions(pylink_data, bounds_factor=BOUNDS_FACTOR, min_length=MIN_LENGTH)

    # Store initial dimensions
    initial_dims = {name: val for name, val in zip(dim_spec.names, dim_spec.initial_values)}

    print(f'\nOptimizable dimensions:')
    for name, initial, bounds in zip(dim_spec.names, dim_spec.initial_values, dim_spec.bounds):
        print(f'  - {name}: {initial:.2f} (bounds: {bounds[0]:.2f} - {bounds[1]:.2f})')

    # -------------------------------------------------------------------------
    # Step 2: Create ACHIEVABLE target trajectory
    # -------------------------------------------------------------------------
    print_section('Step 2: Create Achievable Target')

    print(f'\nTarget generation strategy: randomize dimensions by ±{DIMENSION_RANDOMIZE_RANGE*100:.0f}%')
    print('(This ensures the target trajectory is actually achievable!)')

    target_joint = 'coupler_rocker_joint'  # The "output" joint we want to optimize

    target, target_dims, target_pylink_data = create_achievable_target(
        pylink_data,
        target_joint,
        dim_spec,
        randomize_range=DIMENSION_RANDOMIZE_RANGE,
        seed=RANDOM_SEED
    )

    print(f'\nTarget joint: {target.joint_name}')
    print(f'Target trajectory: {target.n_steps} points')

    print('\nTarget dimensions (what optimizer should find):')
    for name in dim_spec.names:
        initial = initial_dims[name]
        target_val = target_dims[name]
        change = ((target_val - initial) / initial * 100)
        print(f'  - {name}: {target_val:.2f} (was {initial:.2f}, change: {change:+.1f}%)')

    # -------------------------------------------------------------------------
    # Step 3: Visualize initial state
    # -------------------------------------------------------------------------
    print_section('Step 3: Visualize Initial State')

    # Plot combined initial and target linkage state
    from viz_tools.opt_viz import plot_linkage_comparison
    plot_linkage_comparison(
        pylink_data,
        target_pylink_data,
        title='Initial vs Target Linkage',
        out_path=OUTPUT_DIR / f'01_initial_vs_target_linkage_{TIMESTAMP}.png',
    )

    # Compute initial trajectory
    result = compute_trajectory(pylink_data, verbose=False)
    current_traj = result.trajectories[target_joint]

    # Plot initial vs target trajectory
    plot_trajectory_comparison(
        current_traj, target,
        title='Initial vs Target Trajectory',
        out_path=OUTPUT_DIR / f'02_initial_vs_target_{TIMESTAMP}.png',
        show_error_vectors=True,
    )

    # -------------------------------------------------------------------------
    # Step 4: Run optimization
    # -------------------------------------------------------------------------
    print_section('Step 4: Run Optimization')

    print('\nOptimization parameters:')
    print('  - Method: PSO (Particle Swarm Optimization)')
    print(f'  - Particles: {N_PARTICLES}')
    print(f'  - Iterations: {N_ITERATIONS}')
    print(f'  - Metric: {METRIC}')
    print(f'  - Bounds factor: {BOUNDS_FACTOR}')

    print('\nRunning optimization...')
    print('(Trying to recover the target dimensions from trajectory alone)\n')

    # Configure PSO optimizer
    pso_config = PSOConfig(
        n_particles=N_PARTICLES,
        iterations=N_ITERATIONS,
    )

    # Run PSO optimization
    opt_result = run_pso_optimization(
        pylink_data=pylink_data,
        target=target,
        dimension_spec=dim_spec,
        config=pso_config,
        metric=METRIC,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Step 5: Analyze results
    # -------------------------------------------------------------------------
    print_section('Step 5: Analyze Results')

    print('\nOptimization completed:')
    print(f'  - Success: {opt_result.success}')
    print(f'  - Initial error: {opt_result.initial_error:.6f}')
    print(f'  - Final error: {opt_result.final_error:.6f}')
    print(f'  - Iterations: {opt_result.iterations}')

    if opt_result.initial_error > 0:
        improvement = (1 - opt_result.final_error / opt_result.initial_error) * 100
        print(f'  - Error reduction: {improvement:.2f}%')

    # Compare dimensions
    print_dimension_comparison(dim_spec, initial_dims, target_dims, opt_result.optimized_dimensions)

    # Analyze convergence
    if opt_result.convergence_history:
        stats = analyze_convergence(opt_result.convergence_history)
        print('\nConvergence analysis:')
        print(f'  - Best error achieved: {stats.best_error:.6f}')
        print(f'  - Converged: {stats.converged}')
        if hasattr(stats, 'plateau_start') and stats.plateau_start is not None:
            print(f'  - Plateau started at iteration: {stats.plateau_start}')

    # -------------------------------------------------------------------------
    # Step 6: Visualize results
    # -------------------------------------------------------------------------
    print_section('Step 6: Visualize Results')

    # Plot convergence history
    if opt_result.convergence_history:
        plot_convergence_history(
            opt_result.convergence_history,
            title='Optimization Convergence',
            out_path=OUTPUT_DIR / f'03_convergence_{TIMESTAMP}.png',
        )

    # Plot optimized linkage state
    if opt_result.optimized_pylink_data:
        plot_linkage_state(
            opt_result.optimized_pylink_data,
            target=target,
            title='Optimized Linkage vs Target Trajectory',
            out_path=OUTPUT_DIR / f'04_optimized_linkage_{TIMESTAMP}.png',
        )

        # Get optimized trajectory
        opt_traj_result = compute_trajectory(opt_result.optimized_pylink_data, verbose=False, skip_sync=True)
        if opt_traj_result.success:
            optimized_traj = opt_traj_result.trajectories[target_joint]

            # Plot all trajectories overlaid
            plot_trajectory_overlay(
                {
                    'Initial': current_traj,
                    'Optimized': optimized_traj,
                },
                target=target,
                title='Trajectory Comparison: Initial vs Optimized vs Target',
                out_path=OUTPUT_DIR / f'07_trajectory_overlay_{TIMESTAMP}.png',
            )

    # Plot optimization summary
    plot_optimization_summary(
        opt_result, dim_spec,
        title='Optimization Summary',
        out_path=OUTPUT_DIR / f'05_optimization_summary_{TIMESTAMP}.png',
    )

    # Plot final bounds with results
    plot_dimension_bounds(
        dim_spec,
        initial_values=initial_dims,
        target_values=target_dims,
        optimized_values=opt_result.optimized_dimensions,
        title='Dimension Bounds with Final Results',
        out_path=OUTPUT_DIR / f'06_bounds_with_results_{TIMESTAMP}.png',
    )

    # -------------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------------
    print_section('Step 7: Save Results')

    # Save optimization result
    result_file = OUTPUT_DIR / f'optimization_result_{TIMESTAMP}.json'

    result_data = {
        'timestamp': TIMESTAMP,
        'parameters': {
            'n_particles': N_PARTICLES,
            'n_iterations': N_ITERATIONS,
            'bounds_factor': BOUNDS_FACTOR,
            'min_length': MIN_LENGTH,
            'n_steps': N_STEPS,
            'metric': METRIC,
            'dimension_randomize_range': DIMENSION_RANDOMIZE_RANGE,
            'random_seed': RANDOM_SEED,
        },
        'target_dimensions': {k: float(v) for k, v in target_dims.items()},
        'initial_dimensions': {k: float(v) for k, v in initial_dims.items()},
        'result': {
            'success': opt_result.success,
            'initial_error': opt_result.initial_error,
            'final_error': opt_result.final_error,
            'iterations': opt_result.iterations,
            'optimized_dimensions': {k: float(v) for k, v in opt_result.optimized_dimensions.items()},
        },
        'dimension_recovery_errors': {
            name: abs(opt_result.optimized_dimensions.get(name, initial_dims[name]) - target_dims[name])
            for name in dim_spec.names
        },
        'convergence_history': opt_result.convergence_history,
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f'Saved results to: {result_file}')

    # Save optimized linkage
    if opt_result.optimized_pylink_data:
        linkage_file = OUTPUT_DIR / f'optimized_linkage_{TIMESTAMP}.json'
        with open(linkage_file, 'w') as f:
            json.dump(opt_result.optimized_pylink_data, f, indent=2)
        print(f'Saved optimized linkage to: {linkage_file}')

    # Save target linkage
    target_file = OUTPUT_DIR / f'target_linkage_{TIMESTAMP}.json'
    with open(target_file, 'w') as f:
        json.dump(target_pylink_data, f, indent=2)
    print(f'Saved target linkage to: {target_file}')

    # Print convergence report
    print('\n' + format_convergence_report(opt_result, include_history=False))

    print_section('DEMO COMPLETE')
    print(f'\nAll outputs saved to: {OUTPUT_DIR}')
    print('Files created:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')

    # Final assessment
    print_section('OPTIMIZATION QUALITY ASSESSMENT')

    total_dim_error = sum(
        abs(opt_result.optimized_dimensions.get(name, initial_dims[name]) - target_dims[name])
        for name in dim_spec.names
    )
    avg_dim_error = total_dim_error / len(dim_spec.names)

    print('\nDimension recovery:')
    print(f'  - Total dimension error: {total_dim_error:.4f}')
    print(f'  - Average dimension error: {avg_dim_error:.4f}')

    if opt_result.final_error < 0.1:
        print('\n✓ EXCELLENT: Trajectory error < 0.1 - Excellent fit!')
    elif opt_result.final_error < 1.0:
        print('\n✓ GOOD: Trajectory error < 1.0 - Good fit')
    elif opt_result.final_error < 10.0:
        print('\n⚠ MODERATE: Trajectory error < 10.0 - Moderate fit')
    else:
        print('\n✗ POOR: Trajectory error >= 10.0 - Poor fit, may need more iterations or different bounds')



if __name__ == '__main__':
    main()
