#!/usr/bin/env python3
"""
Multi-Solution Demo - Find multiple distinct solutions using Basin Hopping.

WHAT THIS DEMO DOES:
====================
Demonstrates that linkage optimization problems often have multiple solutions:
different dimension configurations that produce similar trajectories.

Uses Basin Hopping global optimization to:
1. Explore the fitness landscape from multiple starting points
2. Discover distinct local minima (solution clusters)
3. Analyze solution diversity and uniqueness

WHY MULTIPLE SOLUTIONS EXIST:
=============================
Linkage trajectory optimization is often "degenerate" - many different
mechanism geometries can produce similar output paths. This demo shows:
- How many distinct solutions exist near the optimum
- How different the dimension values can be while achieving similar error
- Clustering of solutions into distinct "families"

RUN THIS DEMO:
==============
    python demo/multi_demo.py

Output saved to: user/demo/multi_solution/
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo.helpers import load_mechanism
from demo.helpers import print_section
from optimizers.multi_solution import BasinHoppingConfig
from optimizers.multi_solution import run_basin_hopping_multi
from pylink_tools.hypergraph_adapter import extract_dimensions
from target_gen import create_achievable_target
from target_gen import DimensionVariationConfig
from target_gen import MechVariationConfig
from viz_tools.demo_viz import variation_plot
# Note: create_fitness_function removed - use optimize_trajectory or create_mechanism_fitness


# =============================================================================
# CONFIGURATION
# =============================================================================

# Which mechanism (complex mechanisms show more solution diversity)
MECHANISM = 'complex'

# Basin Hopping parameters
BH_ITERATIONS = 8       # Number of basin hopping jumps
BH_TEMPERATURE = 10.5   # Jump acceptance (higher = more exploratory)
BH_STEPSIZE = 0.3       # Perturbation size (fraction of bounds)
LOCAL_METHOD = 'L-BFGS-B'
LOCAL_MAX_ITER = 512

# Solution clustering
EPSILON_THRESHOLD = 5.0     # Error threshold for "near-optimal"
MIN_DISTANCE = 0.4          # Minimum L2 distance for "distinct" solutions

# Target generation
VARIATION_RANGE = 0.55

# Reproducibility
RANDOM_SEED = 42

# Output
OUTPUT_DIR = Path(__file__).parent.parent / 'user' / 'demo' / 'multi_solution'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def print_solution_analysis(result, dim_spec, target_dims):
    """Print analysis of discovered solutions."""
    print(f'\nDiscovered {len(result.solutions)} distinct local minima')
    print(f'Clustered into {result.n_unique_clusters} solution groups')
    print(f'Search space coverage: {result.search_space_coverage:.1%}')
    print(f'Total evaluations: {result.total_evaluations}')

    # Error statistics
    errors = [s.final_error for s in result.solutions]
    print('\nError statistics:')
    print(f'  Best:   {min(errors):.6f}')
    print(f'  Worst:  {max(errors):.6f}')
    print(f'  Mean:   {np.mean(errors):.6f}')
    print(f'  Median: {np.median(errors):.6f}')

    # Top solutions
    print('\nTop 5 solutions:')
    print('  {:>4} {:>12} {:>8} {:>12}'.format('Rank', 'Error', 'Cluster', 'Uniqueness'))
    print('  {} {} {} {}'.format('-'*4, '-'*12, '-'*8, '-'*12))
    for i, sol in enumerate(result.solutions[:5]):
        print(f'  {i+1:>4} {sol.final_error:>12.6f} {sol.cluster_id:>8} {sol.uniqueness_score:>12.3f}')

    # Dimension comparison
    print('\nDimension values (top 3 solutions vs target):')
    top3 = result.solutions[:3]
    print('  {:<25} {:>10}'.format('Dimension', 'Target'), end='')
    for i in range(len(top3)):
        print(f' {"Sol"+str(i+1):>10}', end='')
    print()

    for name in list(dim_spec.names)[:6]:  # Show first 6
        target_val = target_dims.get(name, 0)
        print(f'  {name:<25} {target_val:>10.2f}', end='')
        for sol in top3:
            val = sol.optimized_dimensions.get(name, 0)
            print(f' {val:>10.2f}', end='')
        print()

    if len(dim_spec.names) > 6:
        print(f'  ... and {len(dim_spec.names) - 6} more dimensions')


def create_analysis_plots(result, output_dir, dim_spec):
    """Create error distribution and PCA cluster plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    errors = [s.final_error for s in result.solutions]
    cluster_ids = [s.cluster_id for s in result.solutions]

    # Error distribution
    finite_errors = [e if np.isfinite(e) else np.nan for e in errors]
    if any(np.isfinite(e) for e in errors):
        scatter = ax1.scatter(range(len(errors)), finite_errors, c=cluster_ids, cmap='tab10', s=50, alpha=0.6)
        ax1.axhline(
            result.best_solution.final_error + result.epsilon_threshold,
            color='r', linestyle='--', label=f'ε threshold ({result.epsilon_threshold})',
        )
        ax1.set_xlabel('Solution Index')
        ax1.set_ylabel('Final Error')
        ax1.set_title('Error Distribution by Cluster')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Cluster')

    # PCA cluster visualization
    try:
        from sklearn.decomposition import PCA
        dim_arrays = np.array([list(s.optimized_dimensions.values()) for s in result.solutions])

        if len(dim_arrays) > 2 and dim_arrays.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(dim_arrays)

            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=cluster_ids, cmap='tab10', s=100, alpha=0.7)
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax2.set_title('Solution Clusters (PCA)')
            plt.colorbar(scatter2, ax=ax2, label='Cluster')
    except ImportError:
        ax2.text(0.5, 0.5, 'sklearn not available for PCA', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / f'analysis_{TIMESTAMP}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: analysis_{TIMESTAMP}.png')


def main():
    print_section('MULTI-SOLUTION DEMO')
    print(f'Mechanism: {MECHANISM}')
    print(f'Output: {OUTPUT_DIR}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load mechanism
    # -------------------------------------------------------------------------
    print_section('Step 1: Load Mechanism')

    pylink_data, target_joint, description = load_mechanism(MECHANISM)
    dim_spec, _ = extract_dimensions(pylink_data)

    print(f'\n{description}')
    print(f'Target joint: {target_joint}')
    print(f'Dimensions: {len(dim_spec)}')

    # -------------------------------------------------------------------------
    # Step 2: Create target
    # -------------------------------------------------------------------------
    print_section('Step 2: Create Target')

    print(f'\nRandomizing dimensions by ±{VARIATION_RANGE*100:.0f}%...')

    config = MechVariationConfig(
        dimension_variation=DimensionVariationConfig(default_variation_range=VARIATION_RANGE),
        max_attempts=32,
        random_seed=RANDOM_SEED,
    )

    # Create mechanism first
    from demo.helpers import create_mechanism_from_dict
    mechanism = create_mechanism_from_dict(pylink_data, dim_spec=dim_spec)

    target_result = create_achievable_target(
        mechanism=mechanism,
        target_joint=target_joint,
        dim_spec=dim_spec,
        config=config,
    )

    target = target_result.target
    target_trajectory = target_result.target.positions
    target_dims = target_result.target_dimensions

    print(f'Target trajectory: {len(target_trajectory)} points')

    # Test fitness function
    print('\nTesting fitness function...')
    # Create mechanism for fitness function
    mechanism_for_fitness = create_mechanism_from_dict(pylink_data, dim_spec=dim_spec)
    from pylink_tools.mechanism import create_mechanism_fitness
    fitness_fn = create_mechanism_fitness(
        mechanism=mechanism_for_fitness,
        target=target,
        target_joint=target.joint_name,
        metric='mse',
        phase_invariant=True,
    )
    target_error = fitness_fn(tuple(target_dims[n] for n in dim_spec.names))
    print(f'  Target dimensions error: {target_error:.6f}')

    # -------------------------------------------------------------------------
    # Step 3: Run Basin Hopping
    # -------------------------------------------------------------------------
    print_section('Step 3: Run Basin Hopping')

    bh_config = BasinHoppingConfig(
        n_iterations=BH_ITERATIONS,
        temperature=BH_TEMPERATURE,
        stepsize=BH_STEPSIZE,
        local_method=LOCAL_METHOD,
        local_max_iter=LOCAL_MAX_ITER,
        epsilon_threshold=EPSILON_THRESHOLD,
        min_distance_threshold=MIN_DISTANCE,
        seed=RANDOM_SEED,
    )

    print('\nBasin Hopping config:')
    print(f'  Iterations: {BH_ITERATIONS}')
    print(f'  Temperature: {BH_TEMPERATURE}')
    print(f'  Step size: {BH_STEPSIZE}')
    print(f'  Local optimizer: {LOCAL_METHOD}')

    print('\nRunning optimization...\n')

    result = run_basin_hopping_multi(
        mechanism=create_mechanism_from_dict(pylink_data, dim_spec=dim_spec),
        target=target,
        dimension_bounds_spec=dim_spec,
        config=bh_config,
        metric='mse',
        verbose=True,
        phase_invariant=True,
    )

    print('\nBasin Hopping completed!')

    # -------------------------------------------------------------------------
    # Step 4: Analyze solutions
    # -------------------------------------------------------------------------
    print_section('Step 4: Analyze Solutions')

    print_solution_analysis(result, dim_spec, target_dims)

    # -------------------------------------------------------------------------
    # Step 5: Visualize
    # -------------------------------------------------------------------------
    print_section('Step 5: Visualize')

    # Plot top N solution trajectories
    n_to_plot = min(12, len(result.solutions))
    solution_variations = [sol.optimized_pylink_data for sol in result.solutions[:n_to_plot]]

    variation_plot(
        target_joint=target_joint,
        out_path=OUTPUT_DIR / f'trajectories_{TIMESTAMP}.png',
        base_pylink_data=pylink_data,
        variation_pylink_data=solution_variations,
        title='Multi-Solution Trajectories',
        subtitle=f'Top {n_to_plot} solutions from Basin Hopping',
        show_linkages=False,
    )

    variation_plot(
        target_joint=target_joint,
        out_path=OUTPUT_DIR / f'linkages_{TIMESTAMP}.png',
        base_pylink_data=pylink_data,
        variation_pylink_data=solution_variations,
        title='Multi-Solution Linkages',
        subtitle=f'Top {n_to_plot} solutions',
        show_linkages=True,
    )

    # Create analysis plots
    create_analysis_plots(result, OUTPUT_DIR, dim_spec)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section('COMPLETE')

    print('\nResults:')
    print(f'  Solutions found: {len(result.solutions)}')
    print(f'  Distinct clusters: {result.n_unique_clusters}')
    print(f'  Best error: {result.best_solution.final_error:.6f}')

    print('\nOutput files:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')

    print(f'\nKey insight: {result.n_unique_clusters} distinct mechanism configurations')
    print('can produce trajectories with similar error!')


if __name__ == '__main__':
    main()
