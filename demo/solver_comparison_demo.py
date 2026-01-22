"""
solver_comparison_demo.py - Compare different optimization solvers side-by-side.

This script runs the same optimization problem with different PSO configurations:
1. Custom PSO (random init)
2. Custom PSO (Sobol presampling)
3. Pylinkage PSO (random init)
4. Pylinkage PSO (Sobol presampling)

Results are compared for:
- Final error achieved
- Dimension recovery accuracy
- Computation time
- Convergence behavior (where available)

The key comparison is random vs Sobol presampling initialization, which
demonstrates how intelligent initialization can improve convergence.

=============================================================================
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optimizers.nlopt_mlsl import NLoptMLSLConfig
from optimizers.nlopt_mlsl import run_nlopt_mlsl
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimization_helpers import extract_dimensions_from_edges
from pylink_tools.optimize import run_pso_optimization
from pylink_tools.optimize import run_pylinkage_pso
from pylink_tools.optimize import run_scipy_optimization
from pylink_tools.optimize import TargetTrajectory
from target_gen import create_achievable_target
from target_gen import verify_mechanism_viable
from viz_tools.opt_viz import plot_convergence_history
from viz_tools.opt_viz import plot_dimension_bounds
from viz_tools.opt_viz import plot_trajectory_overlay
# from viz_tools.opt_viz import plot_linkage_state
# from viz_tools.opt_viz import plot_trajectory_comparison

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
N_PARTICLES = 64         # Number of particles in swarm (32-128 typical)
N_ITERATIONS = 64       # Number of optimization iterations (50-500 typical)

#MLSL Parameters
MAX_EVAL = 128          # Maximum number of function evaluations

# --- Bounds Parameters ---
BOUNDS_FACTOR = 1.2       # Bounds = [value/factor, value*factor]
MIN_LENGTH = 4.0          # Minimum link length (prevents degeneracy)

# --- Trajectory Parameters ---
N_STEPS = 32         # Number of simulation timesteps per revolution

# --- Error Metric ---
METRIC = 'mse'            # Error metric: "mse", "rmse", "total", "max"

# --- Target Generation ---
DIMENSION_RANDOMIZE_RANGE = 0.5  # ±50% randomization for target dimensions
COMPLEX_MECH_MAX_RANGE = 0.25     # Max randomization for complex mechanisms (more sensitive)

# --- Output ---
OUTPUT_DIR = project_root / 'user' / 'demo' / 'solver_comparison'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# =============================================================================
# COMPARISON MODE CONFIGURATION
# =============================================================================
# Change COMPARISON_MODE to switch what's being compared.
# Each mode defines a list of (name, solver_func, kwargs) tuples.

# COMPARISON_MODE = 'init_modes'  # <-- CHANGE THIS TO SWITCH COMPARISONS
# COMPARISON_MODE = 'phase_methods'  # <-- CHANGE THIS TO SWITCH COMPARISONS
# COMPARISON_MODE = 'solvers'  # <-- CHANGE THIS TO SWITCH COMPARISONS
COMPARISON_MODE = 'nlopt_mlsl'  # <-- CHANGE THIS TO SWITCH COMPARISONS
# COMPARISON_MODE = 'quick'
# Available modes:
#   'init_modes'       - Compare random vs sobol initialization
#   'phase_methods'    - Compare rotation vs fft phase alignment
#   'solvers'          - Compare PSO vs scipy vs pylinkage
#   'all_pso'          - All PSO variations (init + phase combinations)
#   'nlopt_mlsl'       - Compare NLopt MLSL variants (LDS, local algorithms)
#   'quick'            - Fast: just 2 solvers for quick testing
#   'custom'           - Define your own in CUSTOM_SOLVERS below

# Custom solver definitions (used when COMPARISON_MODE = 'custom')
# Format: (display_name, solver_func_name, kwargs_dict)
CUSTOM_SOLVERS = [
    # Example custom comparison:
    # ('My PSO fft', 'pso', {'init_mode': 'sobol', 'phase_align_method': 'fft'}),
    # ('My scipy', 'scipy', {'phase_align_method': 'rotation'}),
]


def get_solver_configs(mode: str, initial_mechanism: dict | None = None) -> list[tuple[str, callable, dict]]:
    """
    Get solver configurations based on comparison mode.

    Returns list of (display_name, solver_func, kwargs) tuples.
    """
    # Solver function lookup
    SOLVER_FUNCS = {
        'pso': run_pso_optimization,
        'pylinkage': run_pylinkage_pso,
        'scipy': run_scipy_optimization,
    }

    if mode == 'init_modes':
        # Compare initialization strategies
        return [
            ('PSO (random)', run_pso_optimization, {'init_mode': 'random'}),
            ('PSO (sobol)', run_pso_optimization, {'init_mode': 'sobol'}),
            ('Pylinkage (random)', run_pylinkage_pso, {'init_mode': 'random'}),
            ('Pylinkage (sobol)', run_pylinkage_pso, {'init_mode': 'sobol'}),
        ]

    elif mode == 'phase_methods':
        # Compare phase alignment algorithms
        return [
            ('PSO (rotation)', run_pso_optimization, {'phase_align_method': 'rotation'}),
            ('PSO (fft)', run_pso_optimization, {'phase_align_method': 'fft'}),
            ('Scipy (rotation)', run_scipy_optimization, {'phase_align_method': 'rotation'}),
            ('Scipy (fft)', run_scipy_optimization, {'phase_align_method': 'fft'}),
        ]

    elif mode == 'solvers':
        # Compare different solver types
        return [
            ('PSO', run_pso_optimization, {}),
            ('Pylinkage PSO', run_pylinkage_pso, {}),
            # ('Scipy L-BFGS-B', run_scipy_optimization, {'method': 'L-BFGS-B'}),
            # ('Scipy Powell', run_scipy_optimization, {'method': 'Powell'}),
            # ('Scipy Nelder-Mead', run_scipy_optimization, {'method': 'Nelder-Mead'}),
        ]

    elif mode == 'all_pso':
        # All PSO variations: init_mode × phase_align_method
        return [
            ('PSO rand+rot', run_pso_optimization, {'init_mode': 'random', 'phase_align_method': 'rotation'}),
            ('PSO rand+fft', run_pso_optimization, {'init_mode': 'random', 'phase_align_method': 'fft'}),
            ('PSO sobol+rot', run_pso_optimization, {'init_mode': 'sobol', 'phase_align_method': 'rotation'}),
            ('PSO sobol+fft', run_pso_optimization, {'init_mode': 'sobol', 'phase_align_method': 'fft'}),
        ]

    elif mode == 'nlopt_mlsl':
        # Compare NLopt MLSL variants: LDS sampling and local algorithms
        
        # Diagnostic: Check viable ratio to understand landscape feasibility
        try:
            from target_gen.sampling import generate_valid_samples
            from pylink_tools.optimization_helpers import extract_dimensions
            
            dimension_spec = extract_dimensions(initial_mechanism)
            n_test_samples = 64  # Test with 64 samples for diagnostic
            
            print(f"\n  Checking landscape viability (testing {n_test_samples} Sobol samples)...")
            viable_samples, _, n_attempts = generate_valid_samples(
                pylink_data=initial_mechanism,
                dimension_spec=dimension_spec,
                n_requested=n_test_samples,
                max_attempts=500,
                mode='sobol',
                validation='viability',
                selection='first',
                seed=42,
            )
            viable_ratio = len(viable_samples) / n_attempts if n_attempts > 0 else 0
            print(f"  Viable ratio: {viable_ratio:.1%} ({len(viable_samples)}/{n_attempts} samples)")
            
            if viable_ratio < 0.5:
                print("  ⚠️  WARNING: Low viable ratio indicates 'swiss cheese' landscape.")
                print("      Consider using gradient-free local optimizers (bobyqa, sbplx).")
            elif viable_ratio > 0.9:
                print("  ✓ High viable ratio - landscape is mostly feasible.")
        except Exception as e:
            print(f"  Could not check viable ratio: {e}")
        
        #
        # WHY MLSL+LDS MAY FAIL:
        # 1. Linkage geometry constraints: Many dimension combinations create
        #    geometrically impossible 4-bar linkages (links can't reach/close).
        #    The optimizer sees these as inf errors, creating a "swiss cheese"
        #    fitness landscape with many infeasible holes.
        #
        # 2. Local minima: MLSL uses local search from sampled points. If all
        #    nearby samples land in infeasible regions, it may converge to a
        #    suboptimal local minimum.
        #
        # 3. LDS vs random: Low-Discrepancy Sequences (Sobol) provide more
        #    uniform coverage but may miss narrow feasible regions that random
        #    sampling occasionally hits.
        #
        # 4. Gradient issues: Gradient-based methods (lbfgs, slsqp) use finite
        #    differences. Near infeasible boundaries, gradients can be misleading.
        #    Gradient-free methods (bobyqa, sbplx) may be more robust.
        #
        # Gradient-based: lbfgs, slsqp (require 2*dim extra evals per gradient)
        # Gradient-free: bobyqa, sbplx (no gradient overhead, often faster)
        return [
          
            (
                'MLSL (lbfgs)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=False, local_algorithm='lbfgs', max_eval=MAX_EVAL),
                },
            ),
           
            (
                'MLSL (slsqp, grad)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=False, local_algorithm='slsqp', max_eval=MAX_EVAL),
                },
            ),


            # LDS with gradient-based (may fail on swiss cheese landscapes)
            (
                'MLSL+LDS (lbfgs)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=True, local_algorithm='lbfgs', max_eval=MAX_EVAL),
                },
            ),
            # Gradient-based local algorithms
            (
                'MLSL+LDS (slsqp, grad)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=True, local_algorithm='slsqp', max_eval=MAX_EVAL),
                },
            ),
            # Gradient-free local algorithms (often faster due to no gradient overhead)
            (
                'MLSL+LDS (bobyqa, no-grad)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=True, local_algorithm='bobyqa', max_eval=MAX_EVAL),
                },
            ),
            # Non-LDS with random sampling
            (
                'MLSL (bobyqa)', run_nlopt_mlsl, {
                    'config': NLoptMLSLConfig(use_lds=False, local_algorithm='bobyqa', max_eval=MAX_EVAL),
                },
            ),
            # (
            #     'MLSL+LDS (sbplx, no-grad)', run_nlopt_mlsl, {
            #         'config': NLoptMLSLConfig(use_lds=True, local_algorithm='sbplx', max_eval=MAX_EVAL),
            #     },
            # ),
        ]

    elif mode == 'quick':
        # Quick test with just 2 solvers
        return [
            ('PSO (default)', run_pso_optimization, {}),
            ('Scipy (default)', run_scipy_optimization, {}),
        ]

    elif mode == 'custom':
        # User-defined in CUSTOM_SOLVERS
        if not CUSTOM_SOLVERS:
            raise ValueError(
                "COMPARISON_MODE='custom' but CUSTOM_SOLVERS is empty. "
                'Add solver definitions to CUSTOM_SOLVERS list.',
            )
        result = []
        for name, solver_name, kwargs in CUSTOM_SOLVERS:
            if solver_name not in SOLVER_FUNCS:
                raise ValueError(f"Unknown solver '{solver_name}'. Use: {list(SOLVER_FUNCS.keys())}")
            result.append((name, SOLVER_FUNCS[solver_name], kwargs))
        return result

    else:
        available = ['init_modes', 'phase_methods', 'solvers', 'all_pso', 'nlopt_mlsl', 'quick', 'custom']
        raise ValueError(f"Unknown COMPARISON_MODE='{mode}'. Available: {available}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SolverResult:
    """Container for a single solver's results."""
    name: str
    success: bool
    initial_error: float
    final_error: float
    elapsed_time: float
    iterations: int
    optimized_dimensions: dict
    optimized_pylink_data: dict | None
    convergence_history: list | None
    error_message: str | None = None


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


def load_mechanism_from_json(json_path: Path):
    """
    Load a mechanism from a JSON file.

    Args:
        json_path: Path to the mechanism JSON file

    Returns:
        pylink_data dict with n_steps set
    """
    if not json_path.exists():
        raise FileNotFoundError(f'Mechanism file not found: {json_path}')

    with open(json_path) as f:
        pylink_data = json.load(f)

    pylink_data['n_steps'] = N_STEPS
    return pylink_data


def load_mechanism(mechanism_type: str = 'simple'):
    """
    Load a mechanism based on type.

    Args:
        mechanism_type: 'simple', 'intermediate', 'complex', or 'leg'

    Returns:
        (pylink_data, target_joint_name, mechanism_description)
    """
    # Mechanism configurations: (target_joint, description)
    MECHANISM_CONFIG = {
        'simple': ('coupler_rocker_joint', 'Simple 4-bar linkage (4 joints, 4 links, ~3 dimensions)'),
        'intermediate': ('final', 'Intermediate 6-link mechanism (5 joints, 6 links, ~5 dimensions)'),
        'complex': ('final_joint', 'Complex multi-link mechanism (10 joints, 16 links, ~15 dimensions)'),
        'leg': ('toe', 'Leg mechanism (17 joints, 28 links, ~28 dimensions)'),
    }

    if mechanism_type not in MECHANISM_CONFIG:
        available = list(MECHANISM_CONFIG.keys())
        raise ValueError(f'Unknown mechanism_type: {mechanism_type}. Use one of: {available}')

    pylink_data = load_mechanism_from_json(MECHANISM_FILES[mechanism_type])
    target_joint, description = MECHANISM_CONFIG[mechanism_type]

    # Verify the base mechanism is viable
    if not verify_mechanism_viable(pylink_data, target_joint):
        raise RuntimeError(f'Base {mechanism_type} mechanism failed viability check!')

    return pylink_data, target_joint, description


def print_section(title: str):
    """Print a formatted section header."""
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def print_comparison_table(results: list[SolverResult], dim_spec, target_dims):
    """Print a comparison table of solver results."""
    print('\n' + '-' * 100)
    print(f"{'Solver':<28} {'Success':>8} {'Init Err':>12} {'Final Err':>12} {'Time (s)':>10} {'Improve%':>10}")
    print('-' * 100)

    for r in results:
        improvement = (1 - r.final_error / r.initial_error) * 100 if r.initial_error > 0 else 0
        final_err_str = f'{r.final_error:>12.4f}' if r.final_error != float('inf') else '         inf'
        print(f'{r.name:<28} {str(r.success):>8} {r.initial_error:>12.4f} {final_err_str} {r.elapsed_time:>10.2f} {improvement:>9.1f}%')

    print('-' * 100)


def _abbreviate_solver_name(name: str, max_len: int) -> str:
    """
    Abbreviate solver name to fit in column, avoiding unclosed brackets.

    Examples:
        'MLSL+LDS (lbfgs, grad)' -> 'MLSL+LDS/lbfgs'
        'MLSL+LDS (bobyqa, no-grad)' -> 'MLSL+LDS/bobyq'
        'PSO (random)' -> 'PSO/random'
    """
    if len(name) <= max_len:
        return name

    # Try to extract key parts: base name and algorithm/mode
    if '(' in name and ')' in name:
        base = name[:name.index('(')].strip()
        inside = name[name.index('(')+1:name.index(')')].strip()
        # Take first part of what's inside parentheses
        key_part = inside.split(',')[0].strip()
        abbreviated = f'{base}/{key_part}'
        if len(abbreviated) <= max_len:
            return abbreviated
        # Still too long, truncate further
        return abbreviated[:max_len-2] + '..'

    # No parentheses, just truncate with ..
    return name[:max_len-2] + '..'


def print_dimension_recovery(results: list[SolverResult], dim_spec, initial_dims, target_dims):
    """Print dimension recovery accuracy for each solver."""
    # Dynamic width based on number of results
    n_results = len(results)
    col_width = 16  # Wider columns for better readability
    total_width = 30 + 10 + 10 + (col_width * n_results)  # +10 for Initial column

    print('\n' + '-' * total_width)
    header = f"{'Dimension':<30} {'Initial':>10} {'Target':>10}"
    for r in results:
        # Use smart abbreviation to avoid unclosed brackets
        short_name = _abbreviate_solver_name(r.name, col_width - 2)
        header += f' {short_name:>{col_width}}'
    print(header)
    print('-' * total_width)

    for name in dim_spec.names:
        initial_val = initial_dims[name]
        target_val = target_dims[name]
        row = f'{name:<30} {initial_val:>10.2f} {target_val:>10.2f}'
        for r in results:
            opt_val = r.optimized_dimensions.get(name, initial_dims[name])
            error = abs(opt_val - target_val)
            # Format: "  value(err)" to fit col_width=16
            row += f' {opt_val:>8.2f}({error:>5.2f})'
        print(row)

    print('-' * total_width)

    # Print total errors
    print(f"{'TOTAL DIM ERROR':<30} {'':>10} {'':>10}", end='')
    for r in results:
        total_err = sum(
            abs(r.optimized_dimensions.get(name, initial_dims[name]) - target_dims[name])
            for name in dim_spec.names
        )
        print(f' {total_err:>{col_width}.4f}', end='')
    print()


def run_solver_with_timing(
    solver_name: str,
    solver_func,
    pylink_data: dict,
    target: TargetTrajectory,
    dim_spec,
    n_particles: int,
    iterations: int,
    metric: str,
    **extra_kwargs,
) -> SolverResult:
    """Run a solver and capture timing and results.

    Args:
        solver_name: Display name for this solver configuration
        solver_func: The solver function to call
        pylink_data: Linkage data
        target: Target trajectory
        dim_spec: Dimension specification
        n_particles: Number of PSO particles (ignored by scipy)
        iterations: Number of iterations
        metric: Error metric
        **extra_kwargs: Additional kwargs passed to solver (init_mode, phase_align_method, etc.)
    """
    # Build info string from extra kwargs
    info_parts = [f'{k}={v}' for k, v in extra_kwargs.items() if k not in ('init_samples',)]
    info_str = f" ({', '.join(info_parts)})" if info_parts else ''
    print(f'\n  Running {solver_name}{info_str}...')

    # Merge defaults with extra kwargs
    solver_kwargs = {
        'pylink_data': pylink_data,
        'target': target,
        'dimension_spec': dim_spec,
        'metric': metric,
        'verbose': True,
        'phase_invariant': True,
        # Defaults that can be overridden:
        'init_mode': 'random',
        'init_samples': 128,
        'phase_align_method': 'rotation',
    }

    # Add PSO-specific params (scipy ignores these)
    if solver_func in (run_pso_optimization, run_pylinkage_pso):
        solver_kwargs['n_particles'] = n_particles
        solver_kwargs['iterations'] = iterations
    else:
        # scipy-style solvers use max_iterations
        solver_kwargs['max_iterations'] = iterations

    # Override with any extra kwargs
    solver_kwargs.update(extra_kwargs)

    start_time = time.time()
    try:
        opt_result = solver_func(**solver_kwargs)
        elapsed = time.time() - start_time

        return SolverResult(
            name=solver_name,
            success=opt_result.success,
            initial_error=opt_result.initial_error,
            final_error=opt_result.final_error,
            elapsed_time=elapsed,
            iterations=opt_result.iterations or iterations,
            optimized_dimensions=opt_result.optimized_dimensions,
            optimized_pylink_data=opt_result.optimized_pylink_data,
            convergence_history=opt_result.convergence_history,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f'  ERROR: {solver_name} failed with: {e}')
        import traceback
        traceback.print_exc()
        return SolverResult(
            name=solver_name,
            success=False,
            initial_error=float('inf'),
            final_error=float('inf'),
            elapsed_time=elapsed,
            iterations=0,
            optimized_dimensions={},
            optimized_pylink_data=None,
            convergence_history=None,
            error_message=str(e),
        )


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_comparison(
    results: list[SolverResult],
    target: TargetTrajectory,
    target_joint: str,
    initial_traj,
    output_dir: Path,
    timestamp: str,
):
    """Generate comparison visualizations."""

    # Plot all trajectories overlaid
    trajectories = {'Initial': initial_traj}

    for r in results:
        if r.success and r.optimized_pylink_data:
            traj_result = compute_trajectory(r.optimized_pylink_data, verbose=False, skip_sync=True)
            if traj_result.success and target_joint in traj_result.trajectories:
                trajectories[r.name] = traj_result.trajectories[target_joint]

    if len(trajectories) > 1:
        plot_trajectory_overlay(
            trajectories,
            target=target,
            title='Solver Comparison: Trajectory Results',
            out_path=output_dir / f'trajectory_comparison_{timestamp}.png',
        )

    # Plot convergence histories (for solvers that provide them)
    for r in results:
        if r.convergence_history and len(r.convergence_history) > 0:
            plot_convergence_history(
                r.convergence_history,
                title=f'Convergence: {r.name}',
                out_path=output_dir / f'convergence_{r.name.lower().replace(" ", "_")}_{timestamp}.png',
            )


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run the solver comparison demo."""

    print_section('SOLVER COMPARISON DEMO')
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
        # Simple 4-bar format
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
        # Complex mechanism format
        nodes = pylink_data['linkage']['nodes']
        edges = pylink_data['linkage']['edges']
        print(f'Joints ({len(nodes)}):')
        for name, node in nodes.items():
            role = node.get('role', 'unknown')
            print(f'  - {name} ({role})')
        print(f'Links ({len(edges)}):')
        non_ground_edges = [e for e in edges.values() if e.get('id') != 'ground']
        for edge in non_ground_edges[:5]:  # Show first 5
            print(f"  - {edge['id']}: {edge['source']} -> {edge['target']} ({edge['distance']:.1f})")
        if len(non_ground_edges) > 5:
            print(f'  - ... and {len(non_ground_edges) - 5} more links')

    # Extract dimensions (different methods for different mechanism formats)
    if MECHANISM_TYPE in ('complex', 'intermediate'):
        dim_spec = extract_dimensions_from_edges(pylink_data, bounds_factor=BOUNDS_FACTOR, min_length=MIN_LENGTH)
    else:
        dim_spec = extract_dimensions(pylink_data, bounds_factor=BOUNDS_FACTOR, min_length=MIN_LENGTH)

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

    print(f'\nTarget generation: RANDOMIZE dimensions by ±{DIMENSION_RANDOMIZE_RANGE*100:.0f}%')
    print(f'Target joint: {target_joint}')

    # For complex/intermediate mechanisms, use smaller randomization to improve chance of valid config
    effective_range = DIMENSION_RANDOMIZE_RANGE
    if MECHANISM_TYPE in ('complex', 'intermediate') and DIMENSION_RANDOMIZE_RANGE > COMPLEX_MECH_MAX_RANGE:
        effective_range = COMPLEX_MECH_MAX_RANGE
        print(f'  (Reduced to ±{effective_range*100:.0f}% for mechanism stability)')

    achievable_result = create_achievable_target(
        pylink_data,
        target_joint,
        dim_spec,
        randomize_range=effective_range,
        seed=RANDOM_SEED,
    )
    target = achievable_result.target
    target_dims = achievable_result.target_dimensions
    target_pylink_data = achievable_result.target_pylink_data

    print(f'\nTarget joint: {target.joint_name}')
    print(f'Target trajectory: {target.n_steps} points')

    print('\nTarget dimensions (what optimizers should find):')
    for name in dim_spec.names:
        initial = initial_dims[name]
        target_val = target_dims[name]
        change = ((target_val - initial) / initial * 100)
        print(f'  - {name}: {target_val:.2f} (was {initial:.2f}, change: {change:+.1f}%)')

    # Compute initial trajectory for comparison
    result = compute_trajectory(pylink_data, verbose=False)
    initial_traj = result.trajectories[target_joint]

    # -------------------------------------------------------------------------
    # Step 3: Run solvers based on COMPARISON_MODE
    # -------------------------------------------------------------------------
    print_section('Step 3: Run Optimization Solvers')

    print(f'\nComparison mode: {COMPARISON_MODE}')
    print('\nOptimization parameters:')
    print(f'  - Particles: {N_PARTICLES}')
    print(f'  - Iterations: {N_ITERATIONS}')
    print(f'  - Metric: {METRIC}')
    print(f'  - Bounds factor: {BOUNDS_FACTOR}')

    # Get solver configurations based on mode
    solver_configs = get_solver_configs(COMPARISON_MODE, initial_mechanism=pylink_data)
    print(f'\nSolvers to compare ({len(solver_configs)}):')
    for name, func, kwargs in solver_configs:
        kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items()) if kwargs else 'defaults'
        print(f'  - {name}: {kwargs_str}')

    results: list[SolverResult] = []

    for solver_name, solver_func, solver_kwargs in solver_configs:
        solver_result = run_solver_with_timing(
            solver_name=solver_name,
            solver_func=solver_func,
            pylink_data=pylink_data,
            target=target,
            dim_spec=dim_spec,
            n_particles=N_PARTICLES,
            iterations=N_ITERATIONS,
            metric=METRIC,
            **solver_kwargs,
        )
        results.append(solver_result)

    # -------------------------------------------------------------------------
    # Step 4: Compare results
    # -------------------------------------------------------------------------
    print_section('Step 4: Results Comparison')

    print_comparison_table(results, dim_spec, target_dims)
    print_dimension_recovery(results, dim_spec, initial_dims, target_dims)

    # -------------------------------------------------------------------------
    # Step 5: Visualize
    # -------------------------------------------------------------------------
    print_section('Step 5: Visualize Results')

    visualize_comparison(
        results=results,
        target=target,
        target_joint=target_joint,
        initial_traj=initial_traj,
        output_dir=OUTPUT_DIR,
        timestamp=TIMESTAMP,
    )

    # Plot dimension bounds with all results
    optimized_values_list = {r.name: r.optimized_dimensions for r in results if r.success}
    if optimized_values_list:
        # Use the best result for the bounds plot
        best_result = min(results, key=lambda r: r.final_error if r.success else float('inf'))
        if best_result.success:
            plot_dimension_bounds(
                dim_spec,
                initial_values=initial_dims,
                target_values=target_dims,
                optimized_values=best_result.optimized_dimensions,
                title=f'Dimension Bounds (Best: {best_result.name})',
                out_path=OUTPUT_DIR / f'dimension_bounds_{TIMESTAMP}.png',
            )

    # -------------------------------------------------------------------------
    # Step 6: Save results
    # -------------------------------------------------------------------------
    print_section('Step 6: Save Results')

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
        'solver_results': [
            {
                'name': r.name,
                'success': r.success,
                'initial_error': r.initial_error,
                'final_error': r.final_error,
                'elapsed_time': r.elapsed_time,
                'iterations': r.iterations,
                'optimized_dimensions': {k: float(v) for k, v in r.optimized_dimensions.items()},
                'error_message': r.error_message,
                'has_convergence_history': r.convergence_history is not None,
            }
            for r in results
        ],
    }

    result_file = OUTPUT_DIR / f'comparison_result_{TIMESTAMP}.json'
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Saved results to: {result_file}')

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section('COMPARISON SUMMARY')

    print(f'\n  Mode: {COMPARISON_MODE}')

    # Find winner overall
    successful = [r for r in results if r.success]
    if successful:
        best_error = min(successful, key=lambda r: r.final_error)
        best_time = min(successful, key=lambda r: r.elapsed_time)

        print(f'\n  Overall best error: {best_error.name} ({best_error.final_error:.6f})')
        print(f'  Overall fastest:    {best_time.name} ({best_time.elapsed_time:.2f}s)')

        if best_error.name == best_time.name:
            print(f'\n  ★ OVERALL WINNER: {best_error.name} (best on both metrics)')
        else:
            print(f'\n  Trade-off: {best_error.name} is more accurate, {best_time.name} is faster')
    else:
        print('\n  No successful optimizations!')

    # Mode-specific comparisons
    if COMPARISON_MODE == 'init_modes':
        print('\n  --- Init Mode Comparison ---')
        random_results = [r for r in results if 'random' in r.name.lower()]
        sobol_results = [r for r in results if 'sobol' in r.name.lower()]

        if random_results and sobol_results:
            random_errors = [r.final_error for r in random_results if r.success and r.final_error != float('inf')]
            sobol_errors = [r.final_error for r in sobol_results if r.success and r.final_error != float('inf')]

            if random_errors and sobol_errors:
                avg_random = sum(random_errors) / len(random_errors)
                avg_sobol = sum(sobol_errors) / len(sobol_errors)
                print(f'  Avg error (random): {avg_random:.6f}')
                print(f'  Avg error (sobol):  {avg_sobol:.6f}')
                if avg_sobol < avg_random:
                    print(f'  ✓ Sobol presampling improved avg error by {(1 - avg_sobol / avg_random) * 100:.1f}%')

    elif COMPARISON_MODE == 'phase_methods':
        print('\n  --- Phase Alignment Comparison ---')
        rotation_results = [r for r in results if 'rotation' in r.name.lower()]
        fft_results = [r for r in results if 'fft' in r.name.lower()]

        if rotation_results and fft_results:
            rot_times = [r.elapsed_time for r in rotation_results if r.success]
            fft_times = [r.elapsed_time for r in fft_results if r.success]

            if rot_times and fft_times:
                avg_rot = sum(rot_times) / len(rot_times)
                avg_fft = sum(fft_times) / len(fft_times)
                print(f'  Avg time (rotation): {avg_rot:.2f}s')
                print(f'  Avg time (fft):      {avg_fft:.2f}s')
                if avg_fft < avg_rot:
                    print(f'  ✓ FFT is {avg_rot / avg_fft:.1f}x faster on average')

    elif COMPARISON_MODE == 'solvers':
        print('\n  --- Solver Type Comparison ---')
        for r in sorted(successful, key=lambda x: x.final_error):
            print(f'  {r.name}: error={r.final_error:.6f}, time={r.elapsed_time:.2f}s')

    elif COMPARISON_MODE == 'nlopt_mlsl':
        print('\n  --- NLopt MLSL Comparison ---')

        # Check for failures and explain why
        failed = [r for r in results if not r.success or r.final_error == float('inf')]
        high_error = [r for r in results if r.success and r.final_error > 1.0]

        if failed or high_error:
            print('\n  ⚠️  Some optimizers struggled. Common reasons:')
            print('    • Linkage geometry: Large dimension changes can create')
            print('      impossible 4-bar configurations ("swiss cheese" landscape)')
            print('    • Local minima: MLSL may converge to suboptimal solutions')
            print('    • Gradient noise: Finite-difference gradients unreliable')
            print('      near infeasible boundaries')
            print()

        # Compare LDS vs non-LDS
        lds_results = [r for r in results if 'LDS' in r.name]
        non_lds_results = [r for r in results if 'LDS' not in r.name and 'MLSL' in r.name]

        if lds_results and non_lds_results:
            lds_errors = [r.final_error for r in lds_results if r.success and r.final_error != float('inf')]
            non_lds_errors = [r.final_error for r in non_lds_results if r.success and r.final_error != float('inf')]

            if lds_errors and non_lds_errors:
                avg_lds = sum(lds_errors) / len(lds_errors)
                avg_non_lds = sum(non_lds_errors) / len(non_lds_errors)
                print(f'  Avg error (with LDS):    {avg_lds:.6f}')
                print(f'  Avg error (without LDS): {avg_non_lds:.6f}')

        # Compare gradient vs no-gradient
        grad_results = [r for r in results if 'grad)' in r.name and 'no-grad' not in r.name]
        nograd_results = [r for r in results if 'no-grad' in r.name]

        if grad_results and nograd_results:
            grad_times = [r.elapsed_time for r in grad_results if r.success]
            nograd_times = [r.elapsed_time for r in nograd_results if r.success]

            if grad_times and nograd_times:
                avg_grad = sum(grad_times) / len(grad_times)
                avg_nograd = sum(nograd_times) / len(nograd_times)
                print(f'\n  Avg time (gradient):     {avg_grad:.2f}s')
                print(f'  Avg time (no-gradient):  {avg_nograd:.2f}s')
                if avg_nograd < avg_grad:
                    print(f'  ✓ Gradient-free is {avg_grad / avg_nograd:.1f}x faster')
                else:
                    print(f'  ✓ Gradient-based is {avg_nograd / avg_grad:.1f}x faster')

        # Rank by final error
        print('\n  Ranking by final error:')
        for i, r in enumerate(sorted(successful, key=lambda x: x.final_error), 1):
            print(f'    {i}. {r.name}: {r.final_error:.6f} ({r.elapsed_time:.2f}s)')

    print(f'\nAll outputs saved to: {OUTPUT_DIR}')
    print('Files created:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
