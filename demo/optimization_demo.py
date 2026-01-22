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
import sys
from datetime import datetime
from pathlib import Path

from configs.appconfig import USER_DIR
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import analyze_convergence
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimize import format_convergence_report
from pylink_tools.optimize import PSOConfig
from pylink_tools.optimize import run_pso_optimization
from pylink_tools.optimize import TargetTrajectory
from target_gen import AchievableTargetConfig
from target_gen import create_achievable_target
from target_gen import DimensionVariationConfig
from target_gen import StaticJointMovementConfig
from viz_tools.opt_viz import plot_convergence_history
from viz_tools.opt_viz import plot_dimension_bounds
from viz_tools.opt_viz import plot_linkage_state
from viz_tools.opt_viz import plot_optimization_summary
from viz_tools.opt_viz import plot_trajectory_comparison
from viz_tools.opt_viz import plot_trajectory_overlay

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONSTANTS AND HYPERPARAMETERS
# =============================================================================

# --- Random Seed (set for reproducibility, or None for random) ---
RANDOM_SEED = 42  # Set to None for truly random results

# --- Mechanism Type ---
# 'simple'       - Basic 4-bar linkage (4 joints, 4 links, ~3 dimensions)
# 'intermediate' - 6-link mechanism (5 joints, 6 links, ~5 dimensions)
# 'complex'      - Multi-link mechanism (10 joints, 16 links, ~15 dimensions)
# 'leg'          - Leg mechanism (17 joints, 28 links, ~28 dimensions)
MECHANISM_TYPE = 'simple'  # <-- CHANGE THIS TO SWITCH MECHANISMS

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


# =============================================================================
# MECHANISM FILE PATHS
# =============================================================================
# Mechanisms are loaded from JSON files in demo/test_graphs/.

TEST_GRAPHS_DIR = Path(__file__).parent / 'test_graphs'
MECHANISM_FILES = {
    'simple': TEST_GRAPHS_DIR / '4bar.json',
    'intermediate': TEST_GRAPHS_DIR / 'intermediate.json',
    'complex': TEST_GRAPHS_DIR / 'complex.json',
    'leg': TEST_GRAPHS_DIR / 'leg.json',
}

# Mechanism configurations: (target_joint, description)
MECHANISM_CONFIG = {
    'simple': ('coupler_rocker_joint', 'Simple 4-bar linkage (4 joints, 4 links, ~3 dimensions)'),
    'intermediate': ('final', 'Intermediate 6-link mechanism (5 joints, 6 links, ~5 dimensions)'),
    'complex': ('final_joint', 'Complex multi-link mechanism (10 joints, 16 links, ~15 dimensions)'),
    'leg': ('toe', 'Leg mechanism (17 joints, 28 links, ~28 dimensions)'),
}


def load_mechanism(mechanism_type: str = 'simple'):
    """
    Load a mechanism based on type.

    Args:
        mechanism_type: 'simple', 'intermediate', 'complex', or 'leg'

    Returns:
        (pylink_data, target_joint_name, mechanism_description)
    """
    if mechanism_type not in MECHANISM_CONFIG:
        available = list(MECHANISM_CONFIG.keys())
        raise ValueError(f'Unknown mechanism_type: {mechanism_type}. Use one of: {available}')

    json_path = MECHANISM_FILES[mechanism_type]
    if not json_path.exists():
        raise FileNotFoundError(f'Mechanism file not found: {json_path}')

    with open(json_path) as f:
        pylink_data = json.load(f)

    pylink_data['n_steps'] = N_STEPS
    target_joint, description = MECHANISM_CONFIG[mechanism_type]

    return pylink_data, target_joint, description


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
    # Step 1: Load the test mechanism
    # -------------------------------------------------------------------------
    print_section('Step 1: Load Test Mechanism')

    print(f'Mechanism type: {MECHANISM_TYPE}')
    pylink_data, target_joint, mech_description = load_mechanism(MECHANISM_TYPE)

    print(f'Loaded: {mech_description}')

    # Print joint summary based on data format
    if 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
        # Legacy format (simple 4-bar from tests/)
        joints = pylink_data['pylinkage']['joints']
        print(f'Joints ({len(joints)}):')
        for j in joints:
            if j['type'] == 'Static':
                print(f"  - {j['name']} ({j['type']}) at ({j['x']}, {j['y']})")
            elif j['type'] == 'Crank':
                print(f"  - {j['name']} ({j['type']}) distance={j['distance']}")
            elif j['type'] == 'Revolute':
                print(f"  - {j['name']} ({j['type']}) d0={j['distance0']}, d1={j['distance1']}")
    elif 'linkage' in pylink_data and 'nodes' in pylink_data['linkage']:
        # Hypergraph format (complex mechanisms)
        nodes = pylink_data['linkage']['nodes']
        edges = pylink_data['linkage']['edges']
        print(f'Joints ({len(nodes)}):')
        for name, node in list(nodes.items())[:6]:  # Show first 6
            role = node.get('role', 'unknown')
            print(f'  - {name} ({role})')
        if len(nodes) > 6:
            print(f'  - ... and {len(nodes) - 6} more joints')
        print(f'Links ({len(edges)}):')
        non_ground_edges = [e for e in edges.values() if e.get('id') != 'ground']
        for edge in non_ground_edges[:5]:  # Show first 5
            print(f"  - {edge['id']}: {edge['source']} -> {edge['target']} ({edge['distance']:.1f})")
        if len(non_ground_edges) > 5:
            print(f'  - ... and {len(non_ground_edges) - 5} more links')

    # Extract dimensions (auto-detects hypergraph vs legacy format)
    dim_spec = extract_dimensions(pylink_data, bounds_factor=BOUNDS_FACTOR, min_length=MIN_LENGTH)

    # Store initial dimensions
    initial_dims = {name: val for name, val in zip(dim_spec.names, dim_spec.initial_values)}

    print(f'\nOptimizable dimensions: {len(dim_spec)}')
    # Show first 6 dimensions, summarize if more
    dims_to_show = list(zip(dim_spec.names, dim_spec.initial_values, dim_spec.bounds))
    for name, initial, bounds in dims_to_show[:6]:
        print(f'  - {name}: {initial:.2f} (bounds: {bounds[0]:.2f} - {bounds[1]:.2f})')
    if len(dims_to_show) > 6:
        print(f'  - ... and {len(dims_to_show) - 6} more dimensions')

    # -------------------------------------------------------------------------
    # Step 2: Create ACHIEVABLE target trajectory
    # -------------------------------------------------------------------------
    print_section('Step 2: Create Achievable Target')

    print(f'\nTarget generation strategy: randomize dimensions by ±{DIMENSION_RANDOMIZE_RANGE*100:.0f}%')
    print('(This ensures the target trajectory is actually achievable!)')
    print(f'Target joint: {target_joint}')

    # =========================================================================
    # Target Generation Configuration Examples
    # =========================================================================
    #
    # VARIATION 1: Adjust all links uniformly (default behavior)
    # ----------------------------------------------------------
    # All dimensions vary by the same percentage range.
    #
    target_config = AchievableTargetConfig(
        dimension_variation=DimensionVariationConfig(
            default_variation_range=DIMENSION_RANDOMIZE_RANGE,  # ±35% for all
        ),
        random_seed=RANDOM_SEED,
    )
    #
    # VARIATION 2: Selective adjustment with per-link control
    # --------------------------------------------------------
    # Exclude some links, or use custom percentage limits per link.
    # Dimension names come from dim_spec.names (e.g., 'crank_distance',
    # 'coupler_rocker_joint_distance0', 'coupler_rocker_joint_distance1').
    #
    # target_config = AchievableTargetConfig(
    #     dimension_variation=DimensionVariationConfig(
    #         default_variation_range=0.3,  # ±30% default
    #         # Exclude specific dimensions from variation (keep at initial value)
    #         exclude_dimensions=['crank_distance'],
    #         # Override range for specific dimensions: (enabled, min_pct, max_pct)
    #         dimension_overrides={
    #             'coupler_rocker_joint_distance0': (True, -0.5, 0.5),  # ±50%
    #             'coupler_rocker_joint_distance1': (True, -0.1, 0.1),  # ±10% (tight)
    #         },
    #     ),
    #     random_seed=RANDOM_SEED,
    # )
    #
    # VARIATION 3: Adjust all links AND static joint positions
    # ---------------------------------------------------------
    # Move static/ground joints in addition to varying link lengths.
    # Useful for testing optimizer robustness to frame changes.
    #
    # target_config = AchievableTargetConfig(
    #     dimension_variation=DimensionVariationConfig(
    #         default_variation_range=0.25,  # ±25% for links
    #     ),
    #     static_joint_movement=StaticJointMovementConfig(
    #         enabled=True,
    #         max_x_movement=10.0,  # ±10 units in X
    #         max_y_movement=10.0,  # ±10 units in Y
    #         # Optionally control per-joint: (enabled, max_x, max_y)
    #         # joint_overrides={'ground': (False, 0, 0)},  # Keep ground fixed
    #     ),
    #     random_seed=RANDOM_SEED,
    # )
    # =========================================================================

    achievable_result = create_achievable_target(
        pylink_data,
        target_joint,
        dim_spec,
        config=target_config,
    )
    target = achievable_result.target
    target_dims = achievable_result.target_dimensions
    target_pylink_data = achievable_result.target_pylink_data

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
