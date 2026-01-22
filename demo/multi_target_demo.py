"""
Multi-Solution Optimization Demo

Demonstrates finding multiple distinct solutions to a linkage optimization problem
using Basin Hopping global optimization. Shows how degenerate fitness landscapes
can have many local optima with similar performance but different geometries.

This demo:
1. Loads a complex multi-link mechanism
2. Creates an achievable target trajectory
3. Runs Basin Hopping to discover multiple distinct solutions
4. Analyzes solution diversity and clustering
5. Visualizes all discovered solutions on a single plot
"""
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from demo.demo_viz import plot_linkage_variations
from demo.demo_viz import plot_variation_comparison
from optimizers.multi_solution import run_basin_hopping_multi, BasinHoppingConfig
from target_gen.achievable_target import create_achievable_target
from target_gen.variation_config import AchievableTargetConfig, DimensionVariationConfig
from pylink_tools.optimization_helpers import extract_dimensions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print('\n' + '='*80)
    print(f'  {title}')
    print('='*80)


def explore_multiple_starting_points(pylink_data, target_traj_obj, dimension_spec, target_dims, bh_config, n_runs=3):
    """
    Run multiple basin hopping searches from different starting points.
    
    This expands the search diversity by:
    - Starting each run from a different random point
    - Using different random seeds
    - Collecting all unique solutions across runs
    
    Args:
        pylink_data: Mechanism data
        target_traj_obj: Target trajectory
        dimension_spec: Dimension specification
        target_dims: Target dimension values
        bh_config: Basin hopping configuration
        n_runs: Number of independent runs
        
    Returns:
        List of MultiSolutionResult objects from each run
    """
    import random
    from dataclasses import replace
    
    print(f"\nRunning {n_runs} independent basin hopping searches...")
    print(f"Each run starts from a different random point within bounds\n")
    
    all_results = []
    
    for run_idx in range(n_runs):
        print(f"--- Run {run_idx + 1}/{n_runs} ---")
        
        # Generate random starting point within bounds
        starting_values = [
            random.uniform(bounds[0], bounds[1])
            for bounds in dimension_spec.bounds
        ]
        
        # Create modified spec with random start
        modified_spec = replace(
            dimension_spec,
            initial_values=starting_values
        )
        
        # Update config with different seed
        run_config = replace(bh_config, seed=bh_config.seed + run_idx * 100)
        
        # Run optimization
        result = run_basin_hopping_multi(
            pylink_data=pylink_data,
            target=target_traj_obj,
            dimension_spec=modified_spec,
            config=run_config,
            metric='mse',
            verbose=True,
            phase_invariant=True,
        )
        
        print(f"  Found {len(result.solutions)} solutions, {result.n_unique_clusters} clusters")
        print(f"  Best error: {result.best_solution.final_error:.6f}\n")
        
        all_results.append(result)
    
    # Combine statistics
    total_solutions = sum(len(r.solutions) for r in all_results)
    best_overall = min((r.best_solution.final_error for r in all_results), default=float('inf'))
    
    print(f"Combined results:")
    print(f"  Total solutions across all runs: {total_solutions}")
    print(f"  Best error overall: {best_overall:.6f}")
    
    return all_results


def print_section(title: str):
    """Print a formatted section header."""
    print('\n' + '='*80)
    print(f'  {title}')
    print('='*80)


def load_complex_mechanism():
    """Load the complex multi-link mechanism from test_graphs."""
    mechanism_path = project_root / 'demo' / 'test_graphs' / 'complex.json'
    
    with open(mechanism_path, 'r') as f:
        pylink_data = json.load(f)
    
    # Set number of simulation steps
    pylink_data['n_steps'] = 32
    
    return pylink_data


def describe_mechanism(pylink_data: dict):
    """Print detailed description of the mechanism."""
    # Handle both legacy format and hypergraph format
    if 'linkage' in pylink_data:
        # Hypergraph format
        linkage = pylink_data['linkage']
        nodes = linkage.get('nodes', {})
        edges = linkage.get('edges', {})
        n_joints = len(nodes)
        n_links = len(edges)
        
        # Count joint types
        joint_types = {}
        for node_name, node_data in nodes.items():
            jtype = node_data.get('jointType', node_data.get('role', 'unknown'))
            joint_types[jtype] = joint_types.get(jtype, 0) + 1
        
        print(f"\nMechanism: Complex multi-link mechanism")
        print(f"  Joints: {n_joints} ({', '.join(f'{v} {k}' for k, v in joint_types.items())})")
        print(f"  Links: {n_links}")
        
        # Show first few joints
        print(f"\nJoint names:")
        for i, (node_name, node_data) in enumerate(list(nodes.items())[:5]):
            jtype = node_data.get('jointType', node_data.get('role', 'unknown'))
            print(f"    {i+1}. {node_name} ({jtype})")
        if n_joints > 5:
            print(f"    ... and {n_joints - 5} more")
    else:
        # Legacy format
        n_joints = len(pylink_data.get('joints', []))
        n_links = len(pylink_data.get('linkages', []))
        
        # Count joint types
        joint_types = {}
        for joint in pylink_data.get('joints', []):
            jtype = joint.get('type', 'unknown')
            joint_types[jtype] = joint_types.get(jtype, 0) + 1
        
        print(f"\nMechanism: Complex multi-link mechanism")
        print(f"  Joints: {n_joints} ({', '.join(f'{v} {k}' for k, v in joint_types.items())})")
        print(f"  Links: {n_links}")
        
        # Show first few joints and links
        print(f"\nJoint names:")
        for i, joint in enumerate(pylink_data.get('joints', [])[:5]):
            jname = joint.get('name', 'unnamed')
            jtype = joint.get('type', 'unknown')
            print(f"    {i+1}. {jname} ({jtype})")
        if n_joints > 5:
            print(f"    ... and {n_joints - 5} more")


def describe_dimensions(dimension_spec):
    """Print detailed description of optimizable dimensions."""
    n_dims = len(dimension_spec)
    print(f"\nOptimizable dimensions: {n_dims}")
    print(f"  {'Dimension':<30} {'Initial':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, initial, bounds in zip(dimension_spec.names, dimension_spec.initial_values, dimension_spec.bounds):
        dim_range = bounds[1] - bounds[0]
        print(f"  {name:<30} {initial:>10.2f} {bounds[0]:>10.2f} {bounds[1]:>10.2f} {dim_range:>10.2f}")


def analyze_solution_diversity(result):
    """Analyze and print statistics about solution diversity."""
    print(f"\n{'='*80}")
    print(f"  SOLUTION DIVERSITY ANALYSIS")
    print(f"{'='*80}")
    
    n_solutions = len(result.solutions)
    print(f"\nDiscovered {n_solutions} distinct local minima")
    print(f"Clustered into {result.n_unique_clusters} distinct solution groups")
    print(f"Search space coverage: {result.search_space_coverage:.1%}")
    print(f"Total function evaluations: {result.total_evaluations}")
    
    # Error statistics
    errors = [s.final_error for s in result.solutions]
    print(f"\nError statistics:")
    print(f"  Best error:    {result.best_solution.final_error:.6f}")
    print(f"  Worst error:   {max(errors):.6f}")
    print(f"  Mean error:    {np.mean(errors):.6f}")
    print(f"  Median error:  {np.median(errors):.6f}")
    print(f"  Std dev:       {np.std(errors):.6f}")
    
    # Near-optimal solutions
    near_optimal = result.get_near_optimal_solutions(epsilon=result.epsilon_threshold)
    print(f"\nNear-optimal solutions (within {result.epsilon_threshold:.3f} of best): {len(near_optimal)}")
    
    # Cluster representatives
    representatives = result.get_cluster_representatives()
    print(f"\nCluster representatives ({len(representatives)} clusters):")
    print(f"  {'Cluster':>8} {'Error':>12} {'Uniqueness':>12} {'Dist to Best':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for sol in representatives:
        print(f"  {sol.cluster_id:>8} {sol.final_error:>12.6f} {sol.uniqueness_score:>12.3f} {sol.distance_to_best:>12.3f}")
    
    # Show top 10 solutions
    print(f"\nTop 10 solutions:")
    print(f"  {'Rank':>5} {'Error':>12} {'Cluster':>8} {'Uniqueness':>12} {'Iteration':>10}")
    print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
    for i, sol in enumerate(result.solutions[:10]):
        print(f"  {i+1:>5} {sol.final_error:>12.6f} {sol.cluster_id:>8} {sol.uniqueness_score:>12.3f} {sol.iterations:>10}")


def compare_solution_dimensions(result, dimension_spec, target_dims: dict):
    """Compare dimension values across multiple solutions."""
    print(f"\n{'='*80}")
    print(f"  DIMENSION COMPARISON ACROSS SOLUTIONS")
    print(f"{'='*80}")
    
    # Get top 5 solutions for comparison
    top_solutions = result.solutions[:5]
    
    print(f"\nComparing top {len(top_solutions)} solutions:")
    
    # Build comparison table
    dim_names = dimension_spec.names
    print(f"\n{'Dimension':<30} {'Target':>10}", end='')
    for i in range(len(top_solutions)):
        print(f" {'Sol'+str(i+1):>10}", end='')
    print()
    
    print(f"{'-'*30} {'-'*10}", end='')
    for i in range(len(top_solutions)):
        print(f" {'-'*10}", end='')
    print()
    
    for dim_name in dim_names:
        target_val = target_dims.get(dim_name, 0.0)
        print(f"{dim_name:<30} {target_val:>10.2f}", end='')
        
        for sol in top_solutions:
            sol_val = sol.optimized_dimensions[dim_name]
            print(f" {sol_val:>10.2f}", end='')
        print()
    
    # Compute pairwise dimension differences
    print(f"\nPairwise dimension L2 distances:")
    print(f"  {'':>10}", end='')
    for i in range(len(top_solutions)):
        print(f" {'Sol'+str(i+1):>10}", end='')
    print()
    
    for i in range(len(top_solutions)):
        print(f"  {'Sol'+str(i+1):>10}", end='')
        x_i = np.array([top_solutions[i].optimized_dimensions[d] for d in dim_names])
        
        for j in range(len(top_solutions)):
            x_j = np.array([top_solutions[j].optimized_dimensions[d] for d in dim_names])
            dist = np.linalg.norm(x_i - x_j)
            print(f" {dist:>10.3f}", end='')
        print()


def visualize_solutions_simple(result, pylink_data, target_trajectory, output_dir: Path, dimension_spec):
    """Create simple trajectory and linkage comparison plots using shared demo_viz functions."""
    print(f"\n{'='*80}")
    print(f"  SIMPLE VISUALIZATION (using demo_viz)")
    print(f"{'='*80}")
    
    target_joint = 'final_joint'
    
    # Get top 5 solutions
    n_to_plot = min(12, len(result.solutions))
    solutions_to_plot = result.solutions[:n_to_plot]
    
    print(f"\nPlotting top {n_to_plot} solutions...")
    
    # Create list of solution pylink_data variations
    solution_variations = [sol.optimized_pylink_data for sol in solutions_to_plot]
    
    # Plot trajectory comparison
    traj_path = output_dir / 'multi_solution_trajectories.png'
    plot_variation_comparison(
        pylink_data,
        target_joint,
        solution_variations,
        title='Multi-Solution Trajectory Comparison',
        subtitle=f'Top {n_to_plot} solutions from Basin Hopping optimization',
        out_path=traj_path,
    )
    
    # Plot linkage comparison
    linkage_path = output_dir / 'multi_solution_linkages.png'
    plot_linkage_variations(
        pylink_data,
        target_joint,
        solution_variations,
        title='Multi-Solution Linkage Comparison',
        subtitle=f'Top {n_to_plot} solutions with different dimensions',
        out_path=linkage_path,
    )


def visualize_all_solutions(result, pylink_data, target_trajectory, output_dir: Path, dimension_spec):
    """Create a single plot showing all discovered solutions."""
    print(f"\n{'='*80}")
    print(f"  DETAILED VISUALIZATION")
    print(f"{'='*80}")
    
    # Target joint to track
    target_joint = 'final_joint'
    
    # Compute initial trajectory (before optimization)
    from pylink_tools.kinematic import compute_trajectory
    initial_result = compute_trajectory(pylink_data, verbose=False)
    initial_trajectory = None
    if initial_result and initial_result.success and target_joint in initial_result.trajectories:
        initial_trajectory = initial_result.trajectories[target_joint]
    
    # Select top 5 solutions to visualize (to keep plot readable)
    n_to_plot = min(5, len(result.solutions))
    solutions_to_plot = result.solutions[:n_to_plot]
    
    print(f"\nPlotting top {n_to_plot} solutions on a single figure...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Color palette for different solutions
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot target trajectory (first subplot)
    ax = axes[0]
    target_x = [p[0] for p in target_trajectory]
    target_y = [p[1] for p in target_trajectory]
    
    # Plot initial trajectory if available
    if initial_trajectory:
        initial_x = [p[0] for p in initial_trajectory]
        initial_y = [p[1] for p in initial_trajectory]
        ax.plot(initial_x, initial_y, 'cyan', linewidth=2, alpha=0.7, 
                label='Initial', marker='s', markersize=4, linestyle=':')
    
    ax.plot(target_x, target_y, 'magenta', linewidth=2, label='Target', marker='o', markersize=4, linestyle='--')
    ax.set_title('Target vs Initial Trajectory', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # Plot each solution
    for i, sol in enumerate(solutions_to_plot):
        ax = axes[i + 1]
        
        # Simulate trajectory for this solution
        # CRITICAL: Use skip_sync=True to prevent overwriting our optimized dimensions
        # with distances computed from unchanged node positions
        from pylink_tools.kinematic import compute_trajectory
        result_sim = compute_trajectory(sol.optimized_pylink_data, verbose=False, skip_sync=True)
        if target_joint in result_sim.trajectories:
            traj = result_sim.trajectories[target_joint]
            traj_x = [p[0] for p in traj]
            traj_y = [p[1] for p in traj]
            
            # Plot initial trajectory (cyan)
            if initial_trajectory:
                initial_x = [p[0] for p in initial_trajectory]
                initial_y = [p[1] for p in initial_trajectory]
                ax.plot(initial_x, initial_y, 'cyan', linewidth=1.5, alpha=0.6, 
                       label='Initial', linestyle=':', marker='s', markersize=2)
            
            # Plot target (magenta)
            ax.plot(target_x, target_y, 'magenta', linewidth=1.5, alpha=0.6, 
                   label='Target', marker='o', markersize=2, linestyle='--')
            
            # Plot solution trajectory (bold color)
            color = colors[i % len(colors)]
            ax.plot(traj_x, traj_y, color=color, linewidth=2.5, 
                   label=f'Solution {i+1}', marker='o', markersize=4, alpha=0.9)
            
            # Plot linkage mechanism at one position (frame 0)
            # Draw links as lines between joints
            if hasattr(result_sim, 'joint_positions') and result_sim.joint_positions:
                frame_idx = 0
                for joint_name, positions in result_sim.joint_positions.items():
                    if len(positions) > frame_idx:
                        x, y = positions[frame_idx]
                        ax.plot(x, y, 'ko', markersize=4, alpha=0.5)
                
                # Draw edges/links
                if 'linkage' in sol.optimized_pylink_data:
                    edges = sol.optimized_pylink_data['linkage'].get('edges', {})
                    for edge_name, edge_data in edges.items():
                        if 'joint1' in edge_data and 'joint2' in edge_data:
                            j1 = edge_data['joint1']
                            j2 = edge_data['joint2']
                            if j1 in result_sim.joint_positions and j2 in result_sim.joint_positions:
                                j1_pos = result_sim.joint_positions[j1]
                                j2_pos = result_sim.joint_positions[j2]
                                if len(j1_pos) > frame_idx and len(j2_pos) > frame_idx:
                                    ax.plot([j1_pos[frame_idx][0], j2_pos[frame_idx][0]],
                                           [j1_pos[frame_idx][1], j2_pos[frame_idx][1]],
                                           'gray', linewidth=1.5, alpha=0.4)
            
            # Add title with error and cluster info
            error_str = f'{sol.final_error:.4f}' if np.isfinite(sol.final_error) else 'inf'
            title = (f'Solution {i+1} (Cluster {sol.cluster_id})\n'
                    f'Error: {error_str}, Uniqueness: {sol.uniqueness_score:.2f}')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.axis('equal')
    
    # Hide unused subplot if we have fewer than 5 solutions
    if n_to_plot < 5:
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'multi_solution_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Create an error comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error distribution (handle inf values)
    errors = [s.final_error for s in result.solutions]
    cluster_ids = [s.cluster_id for s in result.solutions]
    
    # Filter out inf values for better visualization
    finite_errors = [e if np.isfinite(e) else np.nan for e in errors]
    has_finite = any(np.isfinite(e) for e in errors)
    
    if has_finite:
        scatter = ax1.scatter(range(len(errors)), finite_errors, c=cluster_ids, 
                             cmap='tab10', s=50, alpha=0.6)
        if np.isfinite(result.best_solution.final_error):
            ax1.axhline(result.best_solution.final_error + result.epsilon_threshold, 
                       color='r', linestyle='--', linewidth=1, 
                       label=f'Epsilon threshold ({result.epsilon_threshold})')
        ax1.set_xlabel('Solution Index (sorted by error)')
        ax1.set_ylabel('Final Error (MSE)')
        ax1.set_title('Error Distribution Across All Discovered Solutions')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
    else:
        ax1.text(0.5, 0.5, 'All solutions have inf error\n(fitness function issue)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Error Distribution (No Finite Values)')
    
    # Plot 2: Cluster visualization in dimension space (PCA)
    try:
        from sklearn.decomposition import PCA
        
        # Get dimension arrays for all solutions
        dim_arrays = np.array([list(s.optimized_dimensions.values()) for s in result.solutions])
        
        if len(dim_arrays) > 2 and dim_arrays.shape[1] > 2:
            # Use PCA to project to 2D
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(dim_arrays)
            
            # Plot clusters
            scatter2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                  c=cluster_ids, cmap='tab10', s=100, alpha=0.7)
            
            # Add cluster labels
            for cluster_id in set(cluster_ids):
                mask = np.array(cluster_ids) == cluster_id
                centroid = coords_2d[mask].mean(axis=0)
                ax2.text(centroid[0], centroid[1], str(cluster_id), 
                        fontsize=12, fontweight='bold', ha='center', va='center')
            
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax2.set_title('Solution Clusters in Dimension Space (PCA)')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
        else:
            ax2.text(0.5, 0.5, 'Not enough solutions for PCA', 
                    ha='center', va='center', transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'PCA visualization failed:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
    
    plt.tight_layout()
    
    # Save error analysis figure
    output_path2 = output_dir / 'multi_solution_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved analysis to: {output_path2}")


def main():
    """Main demo execution."""
    print_section('MULTI-SOLUTION OPTIMIZATION DEMO')
    
    # Create output directory
    output_dir = project_root / 'user' / 'demo' / 'multi_solution'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # =========================================================================
    # Step 1: Load Mechanism
    # =========================================================================
    print_section('Step 1: Load Complex Mechanism')
    
    pylink_data = load_complex_mechanism()
    describe_mechanism(pylink_data)
    
    dimension_spec = extract_dimensions(pylink_data)
    describe_dimensions(dimension_spec)
    
    # =========================================================================
    # Step 2: Create Achievable Target
    # =========================================================================
    print_section('Step 2: Create Achievable Target Trajectory')
    
    print("\nGenerating target by randomizing dimensions...")
    print("This ensures the target is achievable (not in infeasible region)")
    
    # Configure dimension variations
    config = AchievableTargetConfig(
        dimension_variation=DimensionVariationConfig(
            default_variation_range=0.55,  # ±15% variation
        ),
        max_attempts=32,
        random_seed=42,
    )
    
    target_result = create_achievable_target(
        pylink_data=pylink_data,
        target_joint='final_joint',
        dim_spec=dimension_spec,
        config=config,
    )
    
    # target_result.target is the TargetTrajectory object itself
    target_traj_obj = target_result.target  # TargetTrajectory object
    target_trajectory = target_result.target.positions  # List of (x,y) positions for plotting
    target_dims = target_result.target_dimensions
    
    print(f"\nTarget created successfully!")
    print(f"  Target joint: final_joint")
    print(f"  Trajectory points: {len(target_trajectory)}")
    print(f"\nTarget dimension changes:")
    for name in dimension_spec.names[:5]:  # Show first 5
        initial = dict(zip(dimension_spec.names, dimension_spec.initial_values))[name]
        target = target_dims[name]
        change_pct = ((target - initial) / initial) * 100
        print(f"  {name}: {initial:.2f} -> {target:.2f} ({change_pct:+.1f}%)")
    if len(dimension_spec.names) > 5:
        print(f"  ... and {len(dimension_spec.names) - 5} more dimensions")
    
    # =========================================================================
    # Step 3: Run Basin Hopping Multi-Solution Optimization
    # =========================================================================
    print_section('Step 3: Run Basin Hopping to Discover Multiple Solutions')
    
    print("\nBasin Hopping Configuration:")
    bh_config = BasinHoppingConfig(
        n_iterations=8,           # Number of basin hopping iterations (each iteration = 1 local search)
        temperature=10.5,          # Jump acceptance: higher = more exploratory (try 5.0-20.0)
        stepsize=0.3,              # Perturbation size: fraction of bounds range (try 0.2-0.5)
        local_method='L-BFGS-B',   # Local optimizer
        local_max_iter=512,        # Iterations per local search
        epsilon_threshold=5.0,     # Error threshold for "near-optimal" (solutions within this of best)
        min_distance_threshold=0.4,# Minimum L2 distance for "distinct" solutions (higher = fewer, more distinct)
        seed=42,                   # For reproducibility (change to get different solution sets)
    )
    
    print(f"  Iterations: {bh_config.n_iterations}")
    print(f"  Temperature: {bh_config.temperature}")
    print(f"  Step size: {bh_config.stepsize}")
    print(f"  Local optimizer: {bh_config.local_method}")
    print(f"  Epsilon threshold: {bh_config.epsilon_threshold}")
    print(f"  Distance threshold: {bh_config.min_distance_threshold}")
    print(f"\nThis will discover all local minima visited during basin hopping...")
    print(f"Each distinct minimum (>= {bh_config.min_distance_threshold} apart) will be recorded.\n")
    
    # =========================================================================
    # Starting Point Strategy
    # =========================================================================
    # What gets passed in:
    # 1. pylink_data: Original mechanism geometry (node positions, edge connections)
    # 2. dimension_spec: Contains bounds and ORIGINAL starting values from complex.json
    # 3. modified_spec: Same as dimension_spec but initial_values can be customized
    #
    # Current settings:
    # - Start: ORIGINAL mechanism dimensions (e.g., crank=20.00, coupler=50.00, ...)
    # - Target: Modified dimensions (e.g., crank=21.64, coupler=49.08, ...) 
    # - Bounds: ±50% of original values (e.g., crank bounds: [10, 40])
    # - Basin Hopping will randomly perturb from starting point
    # - Temperature=10.5 controls how far it jumps (higher = more exploration)
    # - Stepsize=0.3 controls perturbation magnitude (as fraction of bounds)
    #
    # The optimizer should find the target dimensions (and possibly other solutions)
    # starting from the original mechanism configuration.
    #
    # To expand search diversity, you can:
    # A) Start from random point within bounds
    # B) Increase temperature (more aggressive jumps)
    # C) Increase stepsize (larger perturbations)
    # D) Run multiple basin hopping runs with different seeds
    # E) Start from midpoint of bounds for more central exploration
    # =========================================================================
    
    # OPTION A: Start from ORIGINAL mechanism dimensions (realistic, non-cheating)
    print("Using original mechanism dimensions as starting point...")
    starting_values = dimension_spec.initial_values  # Original dimensions from complex.json
    
    # OPTION B: Start from random point within bounds (more exploratory)
    # Uncomment to use:
    # print("Using random starting point within bounds...")
    # import random
    # starting_values = [
    #     random.uniform(bounds[0], bounds[1])
    #     for bounds in dimension_spec.bounds
    # ]
    
    # OPTION C: Start from midpoint of bounds (balanced exploration)
    # Uncomment to use:
    # print("Using midpoint of bounds as starting point...")
    # starting_values = [
    #     (bounds[0] + bounds[1]) / 2
    #     for bounds in dimension_spec.bounds
    # ]
    
    # OPTION D: Start from target dimensions (CHEATING - only for debugging!)
    # This gives initial error ≈ 0 because you're starting at the solution
    # Uncomment ONLY for testing the optimization pipeline itself:
    # print("WARNING: Starting from target dimensions (cheating mode for debugging)...")
    # starting_values = [target_dims[name] for name in dimension_spec.names]
    
    # OPTION E: Start from a VALIDATED viable sample (recommended for best results)
    # Uses optimization_helpers tools to find a mechanism-valid starting point
    # Uncomment to use:
    # print("Generating validated viable starting point...")
    # from pylink_tools.optimization_helpers import generate_valid_samples
    # valid_samples = generate_valid_samples(
    #     pylink_data, dimension_spec, n_samples=5, max_attempts=100
    # )
    # if len(valid_samples) > 0:
    #     print(f"  Found {len(valid_samples)} valid samples, using first one")
    #     starting_values = [valid_samples[0][name] for name in dimension_spec.names]
    # else:
    #     print("  No valid samples found, falling back to original dimensions")
    #     starting_values = dimension_spec.initial_values
    
    # Create modified DimensionSpec with chosen starting point
    from dataclasses import replace
    modified_spec = replace(
        dimension_spec, 
        initial_values=starting_values
    )
    
    # DEBUG: Test the fitness function with target dimensions (should be near 0)
    print("\nDEBUG: Testing fitness function with target dimensions...")
    from pylink_tools.optimize import create_fitness_function
    test_fitness = create_fitness_function(
        pylink_data, target_traj_obj,
        modified_spec, metric='mse', phase_invariant=True
    )
    target_dims_tuple = tuple([target_dims[name] for name in dimension_spec.names])
    test_error = test_fitness(target_dims_tuple)
    print(f"  Target dimensions error: {test_error:.6f}")
    if not np.isfinite(test_error):
        print("  WARNING: Target dimensions returning inf! Issue with fitness function.")
    
    # =========================================================================
    # Run Basin Hopping
    # =========================================================================
    # STANDARD: Single run from chosen starting point
    result = run_basin_hopping_multi(
        pylink_data=pylink_data,
        target=target_traj_obj,
        dimension_spec=modified_spec,
        config=bh_config,
        metric='mse',
        verbose=True,
        phase_invariant=True,
        initial_point=None,  # Uses modified_spec.initial_values (can pass custom np.array here)
    )
    
    # ALTERNATIVE: Multiple runs from different starting points (more thorough exploration)
    # Uncomment to use:
    # all_results = explore_multiple_starting_points(
    #     pylink_data, target_traj_obj, dimension_spec, target_dims, bh_config, n_runs=3
    # )
    # result = all_results[0]  # Use first result for visualization, or merge all results
    
    # ADVANCED: Generate validated viable starting point using optimization tools
    # This can significantly improve exploration by starting from a known-good configuration:
    # from pylink_tools.optimization_helpers import generate_valid_samples
    # valid_samples = generate_valid_samples(
    #     pylink_data, dimension_spec, n_samples=5, max_attempts=100
    # )
    # if len(valid_samples) > 0:
    #     initial_pt = np.array([valid_samples[0][name] for name in dimension_spec.names])
    #     result = run_basin_hopping_multi(
    #         pylink_data=pylink_data,
    #         target=target_traj_obj,
    #         dimension_spec=modified_spec,
    #         config=bh_config,
    #         metric='mse',
    #         verbose=True,
    #         phase_invariant=True,
    #         initial_point=initial_pt,  # Custom starting point
    #     )
    
    print(f"\n[OK] Basin Hopping completed successfully!")
    
    # =========================================================================
    # Step 4: Analyze Solution Diversity
    # =========================================================================
    print_section('Step 4: Analyze Solution Diversity')
    
    analyze_solution_diversity(result)
    compare_solution_dimensions(result, dimension_spec, target_dims)
    
    # =========================================================================
    # Step 5: Visualize Solutions
    # =========================================================================
    print_section('Step 5: Visualize All Solutions')
    
    # Create simple comparison plots using shared demo_viz functions
    visualize_solutions_simple(result, pylink_data, target_trajectory, output_dir, dimension_spec)
    
    # Create detailed subplot grid with error analysis
    visualize_all_solutions(result, pylink_data, target_trajectory, output_dir, dimension_spec)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section('SUMMARY')
    
    print(f"\nMULTI-SOLUTION OPTIMIZATION COMPLETE!")
    print(f"\n  Method: Basin Hopping")
    print(f"  Solutions discovered: {len(result.solutions)}")
    print(f"  Distinct clusters: {result.n_unique_clusters}")
    print(f"  Best error: {result.best_solution.final_error:.6f}")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Search coverage: {result.search_space_coverage:.1%}")
    
    near_optimal = result.get_near_optimal_solutions()
    print(f"\n  Near-optimal solutions (epsilon={bh_config.epsilon_threshold}): {len(near_optimal)}")
    
    print(f"\nOUTPUTS saved to: {output_dir}")
    print(f"  - multi_solution_comparison.png")
    print(f"  - multi_solution_analysis.png")
    
    print(f"\n{'='*80}")
    print("\nKEY INSIGHTS:")
    print("  • Multiple distinct solutions can achieve similar low errors")
    print("  • Solutions may differ significantly in dimension values")
    print("  • Clustering helps identify truly distinct solution families")
    print("  • Uniqueness score measures how different each solution is")
    print("  • Degenerate landscapes are common in mechanism synthesis")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
