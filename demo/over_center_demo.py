#!/usr/bin/env python3
"""
Over-Center and Reflection Demo - Explore how initial starting conditions affect optimizer outcomes.

WHAT THIS DEMO DOES:
====================
Shows how reflecting a mechanism over the ground link (the line between two static joints)
affects the optimization landscape:

1. Load a mechanism (4-bar by default)
2. Create variation config that varies only 2 links (not the crank)
3. Create an achievable target trajectory
4. Sample N gridded points from the dimension space and plot contour scores
5. Reflect the mechanism over the ground link line
6. Sample N gridded points from the reflected mechanism's space and plot contour scores

WHY THIS MATTERS:
=================
- Demonstrates how over-centering or reflections affect optimization landscapes
- Shows symmetry in the design space
- Helps understand how initial conditions affect optimizer convergence
- Validates that reflected mechanisms produce different but related landscapes

RUN THIS DEMO:
==============
    python demo/over_center_demo.py

Output saved to: user/demo/over_center/
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from configs.appconfig import USER_DIR
from demo.helpers import load_mechanism, print_section
from pylink_tools.mechanism import Mechanism
from pylinkage.joints import Static
from pylinkage.bridge.solver_conversion import update_solver_positions
from target_gen import MechVariationConfig, create_achievable_target
from target_gen import DimensionVariationConfig
from target_gen.sampling import generate_samples, SamplingResult
from viz_tools.spaces_viz import render_contours
from viz_tools.viz_styling import _save_or_show, STYLE
from viz_tools.demo_viz import variation_plot

# Configuration
MECHANISM = 'simple'  # Use 4-bar by default
N_GRID_SAMPLES = 20  # Number of gradations per dimension (N^2 total samples for 2D)
N_STEPS = 96  # Number of trajectory steps
N_VARIATION_PLOTS = 8  # Number of sample mechanisms to show in variation plots
OUTPUT_DIR = USER_DIR / 'demo' / 'over_center'
TIMESTAMP = ""


def reflect_point_over_line(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> tuple[float, float]:
    """
    Reflect a point over a line defined by two points.
    
    Args:
        point: (x, y) point to reflect
        line_p1: First point on the line
        line_p2: Second point on the line
    
    Returns:
        Reflected (x, y) point
    """
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    # Vector along the line
    dx = x2 - x1
    dy = y2 - y1
    
    # Length of line segment
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-10:
        # Line is degenerate (points are the same), return original point
        return point
    
    # Vector from line_p1 to point
    vx = px - x1
    vy = py - y1
    
    # Project point onto line
    # t = dot(v, line) / ||line||^2
    t = (vx * dx + vy * dy) / line_len_sq
    
    # Closest point on line
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # Reflect: point' = 2 * projection - point
    reflected_x = 2 * proj_x - px
    reflected_y = 2 * proj_y - py
    
    return (reflected_x, reflected_y)


def reflect_mechanism_over_ground_link(mechanism: Mechanism) -> Mechanism:
    """
    Create a new mechanism by reflecting all joints over the ground link line.
    
    The ground link is defined by the two static joints (ground/fixed joints).
    
    Args:
        mechanism: Original mechanism
    
    Returns:
        New Mechanism with all joints reflected over the ground link
    """
    # Find static joints (ground link endpoints)
    static_joints = []
    static_positions = []
    
    for i, joint in enumerate(mechanism.linkage.joints):
        if isinstance(joint, Static):
            static_joints.append((i, joint.name))
            coords = mechanism.linkage.get_coords()
            static_positions.append((coords[i][0], coords[i][1]))
    
    if len(static_joints) < 2:
        raise ValueError(f"Need at least 2 static joints for reflection, found {len(static_joints)}")
    
    # Use first two static joints to define the ground link line
    line_p1 = static_positions[0]
    line_p2 = static_positions[1]
    
    print(f"  Reflecting over line: {static_joints[0][1]} ({line_p1}) to {static_joints[1][1]} ({line_p2})")
    
    # Get all current coordinates
    coords = mechanism.linkage.get_coords()
    new_coords = []
    
    # Reflect all joint positions
    for i, (x, y) in enumerate(coords):
        reflected = reflect_point_over_line((x, y), line_p1, line_p2)
        new_coords.append(reflected)
    
    # Create a copy of the mechanism
    reflected_mechanism = mechanism.copy()
    
    # Apply reflected coordinates
    coords_list = [(float(x), float(y)) for x, y in new_coords]
    reflected_mechanism.linkage.set_coords(coords_list)
    update_solver_positions(reflected_mechanism.linkage._solver_data, reflected_mechanism.linkage)
    
    # Update initial positions cache
    reflected_mechanism._initial_positions = reflected_mechanism.linkage.get_coords()
    
    return reflected_mechanism


def create_variation_config_for_two_links(mechanism: Mechanism, exclude_crank: bool = True) -> DimensionVariationConfig:
    """
    Create a variation config that varies only 2 links, excluding the crank if requested.
    
    Args:
        mechanism: Mechanism to analyze
        exclude_crank: If True, exclude crank-related dimensions
    
    Returns:
        DimensionVariationConfig with only 2 dimensions enabled
    """
    dim_spec = mechanism.get_dimension_bounds_spec()
    dim_names = dim_spec.names
    
    # Find dimensions to exclude (crank-related if requested)
    exclude_list = []
    if exclude_crank:
        # Exclude dimensions with "crank" in the name
        for name in dim_names:
            if 'crank' in name.lower():
                exclude_list.append(name)
    
    # Find which dimensions are NOT excluded
    available_dims = [name for name in dim_names if name not in exclude_list]
    
    if len(available_dims) < 2:
        raise ValueError(f"Need at least 2 non-excluded dimensions, found {len(available_dims)}. "
                        f"Available: {available_dims}, Excluded: {exclude_list}")
    
    # Select first 2 available dimensions to vary
    # Exclude all others
    dims_to_vary = available_dims[:2]
    dims_to_exclude = [name for name in dim_names if name not in dims_to_vary]
    
    print(f"  Varying dimensions: {dims_to_vary}")
    print(f"  Excluding dimensions: {dims_to_exclude}")
    
    return DimensionVariationConfig(
        default_variation_range=0.35,  # ±35% variation
        exclude_dimensions=dims_to_exclude,
    )


def plot_contour_scores(
    result: SamplingResult,
    dimension_bounds_spec,
    title: str,
    filename: str,
) -> None:
    """
    Plot contour scores from sampling result.
    
    Args:
        result: SamplingResult with samples and scores
        dimension_bounds_spec: Dimension specification
        title: Plot title
        filename: Output filename (without extension)
    """
    # Find which dimensions actually vary
    sample_variances = np.var(result.samples, axis=0)
    varying_dim_indices = np.where(sample_variances > 1e-10)[0]
    
    if len(varying_dim_indices) < 2:
        print(f"  Warning: Only {len(varying_dim_indices)} varying dimensions, cannot plot 2D contour")
        return
    
    # Use first two varying dimensions
    dim1_idx = varying_dim_indices[0]
    dim2_idx = varying_dim_indices[1]
    
    dim1_name = dimension_bounds_spec.names[dim1_idx]
    dim2_name = dimension_bounds_spec.names[dim2_idx]
    
    x_coords_all = result.samples[:, dim1_idx]
    y_coords_all = result.samples[:, dim2_idx]
    
    x_bounds = dimension_bounds_spec.bounds[dim1_idx]
    y_bounds = dimension_bounds_spec.bounds[dim2_idx]
    
    # Prepare values array - use NaN for invalid/infinite scores so render_contours can filter
    if result.scores is not None:
        # Create array with same length as samples, using NaN for invalid/infinite scores
        values_all = np.full(len(result.samples), np.nan)
        valid_mask = result.is_valid & np.isfinite(result.scores)
        values_all[valid_mask] = result.scores[valid_mask]
    else:
        values_all = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours and points - pass all data, render_contours will filter based on values
    if values_all is not None and np.any(np.isfinite(values_all)):
        render_contours(
            x_coords=x_coords_all,  # All points
            y_coords=y_coords_all,  # All points
            values=values_all,  # NaN for invalid, finite values for valid
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            x_label=dim1_name,
            y_label=dim2_name,
            value_label='Fitness (MSE)',
            show_contour=True,
            show_contour_lines=False,
            show_points=True,
            point_labels=['valid' if v else 'invalid' for v in result.is_valid],
            n_contour_levels=20,
            grid_resolution=100,
            interpolation_method='linear',
            fig=fig,
            ax=ax,
            verbose=1,
        )
    
    # Set title
    if result.scores is not None:
        finite_scores = result.scores[np.isfinite(result.scores)]
        if len(finite_scores) > 0:
            score_range_str = f'Score range: [{np.min(finite_scores):.2f}, {np.max(finite_scores):.2f}]'
        else:
            score_range_str = 'No finite scores'
    else:
        score_range_str = 'No scores available'
    
    ax.set_title(
        f'{title}\n'
        f'{dim1_name} vs {dim2_name}\n'
        f'{len(result.samples)} total samples ({result.n_valid} valid, {result.n_invalid} invalid), {score_range_str}',
        fontsize=14,
        fontweight='bold',
    )
    
    plt.tight_layout()
    
    # Save plot
    out_path = OUTPUT_DIR / f'{filename}_{TIMESTAMP}.png'
    _save_or_show(fig, out_path, STYLE.dpi)
    print(f"  Saved: {out_path}")


def main():
    print_section('OVER-CENTER AND REFLECTION DEMO')
    print(f'Mechanism: {MECHANISM}')
    print(f'Grid samples per dimension: {N_GRID_SAMPLES} (total: {N_GRID_SAMPLES**2} for 2D)')
    print(f'Output: {OUTPUT_DIR}')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load mechanism
    print('\nLoading mechanism...')
    mechanism, target_joint, description = load_mechanism(MECHANISM, n_steps=N_STEPS)
    dim_spec = mechanism.get_dimension_bounds_spec()
    
    print(f'  {description}')
    print(f'  Target joint: {target_joint}')
    print(f'  Dimensions: {len(dim_spec)}')
    
    # Create variation config (vary only 2 links, not crank)
    print_section('Creating Variation Config')
    dimension_variation_config = create_variation_config_for_two_links(mechanism, exclude_crank=True)
    
    # Create achievable target trajectory
    print_section('Creating Target Trajectory')
    print("Creating achievable target trajectory...")
    achievable_target_result = create_achievable_target(
        mechanism,
        target_joint,
        dim_spec=dim_spec,
        config=MechVariationConfig(
            dimension_variation=dimension_variation_config,
            random_seed=42,
        ),
        n_steps=N_STEPS,
    )
    target_trajectory = achievable_target_result.target
    print(f"Target trajectory created with {len(target_trajectory.positions)} points")
    
    # =============================================================================
    # STEP 1: Sample original mechanism
    # =============================================================================
    print_section('Sampling Original Mechanism')
    # For meshgrid with 2 dimensions, we need to pass total samples = N^2
    # The function will calculate n_gradations = sqrt(n_requested) per dimension
    n_total_samples = N_GRID_SAMPLES ** 2
    print(f"Generating {N_GRID_SAMPLES}x{N_GRID_SAMPLES} grid samples (total: {n_total_samples})...")
    
    result_original = generate_samples(
        mechanism,
        dimension_bounds_spec=dim_spec,
        n_requested=n_total_samples,  # Total samples for meshgrid (will create N^2 for 2D)
        sampling_mode='meshgrid',  # Use meshgrid for regular grid
        dimension_variation_config=dimension_variation_config,
        target_trajectory=target_trajectory,
        target_joint=target_joint,
        metric='mse',
        phase_invariant=True,
        return_mechanisms=True,  # Need mechanisms for variation plots
        return_trajectories=False,
        seed=42,
    )
    
    print(f"Generated {len(result_original.samples)} total samples "
          f"({result_original.n_valid} valid, {result_original.n_invalid} invalid)")
    if result_original.scores is not None:
        finite_scores = result_original.scores[np.isfinite(result_original.scores)]
        if len(finite_scores) > 0:
            print(f"Score range: [{np.min(finite_scores):.4f}, {np.max(finite_scores):.4f}]")
    
    # Plot original mechanism contour
    plot_contour_scores(
        result_original,
        dim_spec,
        title='Original Mechanism - Contour Scores',
        filename='original_contour',
    )
    
    # Create variation plot for original mechanism
    if result_original.mechanisms is not None:
        # Select sample mechanisms (best scoring valid ones)
        # Handle scores array - use it if available, otherwise create default
        if result_original.scores is not None:
            scores_array = result_original.scores
        else:
            scores_array = np.full(len(result_original.mechanisms), np.inf)
        
        valid_mechanisms = [
            (mech, score) for mech, score, is_valid in 
            zip(result_original.mechanisms, scores_array, result_original.is_valid)
            if mech is not None and is_valid and score is not None and np.isfinite(score)
        ]
        
        if valid_mechanisms:
            # Sort by score and take top N
            valid_mechanisms.sort(key=lambda x: x[1])
            sample_mechanisms = [mech for mech, _ in valid_mechanisms[:N_VARIATION_PLOTS]]
            
            print(f"  Creating variation plot with {len(sample_mechanisms)} sample mechanisms...")
            
            # Plot trajectories
            variation_plot(
                target_joint=target_joint,
                out_path=OUTPUT_DIR / f'original_trajectories_{TIMESTAMP}.png',
                base_mechanism=mechanism,
                variation_mechanisms=sample_mechanisms,
                title='Original Mechanism - Sample Trajectories',
                subtitle=f'Showing {len(sample_mechanisms)} best-scoring variations',
                show_linkages=False,
            )
            
            # Plot linkages
            variation_plot(
                target_joint=target_joint,
                out_path=OUTPUT_DIR / f'original_linkages_{TIMESTAMP}.png',
                base_mechanism=mechanism,
                variation_mechanisms=sample_mechanisms,
                title='Original Mechanism - Sample Linkages',
                subtitle=f'Showing {len(sample_mechanisms)} best-scoring variations',
                show_linkages=True,
            )
    
    # =============================================================================
    # STEP 2: Reflect mechanism over ground link
    # =============================================================================
    print_section('Reflecting Mechanism Over Ground Link')
    reflected_mechanism = reflect_mechanism_over_ground_link(mechanism)
    print("Mechanism reflected successfully")
    
    # Get dimension spec for reflected mechanism (should be same as original)
    reflected_dim_spec = reflected_mechanism.get_dimension_bounds_spec()
    
    # =============================================================================
    # STEP 3: Sample reflected mechanism
    # =============================================================================
    print_section('Sampling Reflected Mechanism')
    print(f"Generating {N_GRID_SAMPLES}x{N_GRID_SAMPLES} grid samples (total: {n_total_samples})...")
    
    result_reflected = generate_samples(
        reflected_mechanism,
        dimension_bounds_spec=reflected_dim_spec,
        n_requested=n_total_samples,  # Total samples for meshgrid (will create N^2 for 2D)
        sampling_mode='meshgrid',
        dimension_variation_config=dimension_variation_config,
        target_trajectory=target_trajectory,  # Use same target trajectory
        target_joint=target_joint,
        metric='mse',
        phase_invariant=True,
        return_mechanisms=True,  # Need mechanisms for variation plots
        return_trajectories=False,
        seed=42,  # Same seed for reproducibility
    )
    
    print(f"Generated {len(result_reflected.samples)} total samples "
          f"({result_reflected.n_valid} valid, {result_reflected.n_invalid} invalid)")
    if result_reflected.scores is not None:
        finite_scores = result_reflected.scores[np.isfinite(result_reflected.scores)]
        if len(finite_scores) > 0:
            print(f"Score range: [{np.min(finite_scores):.4f}, {np.max(finite_scores):.4f}]")
    
    # Plot reflected mechanism contour
    plot_contour_scores(
        result_reflected,
        reflected_dim_spec,
        title='Reflected Mechanism - Contour Scores',
        filename='reflected_contour',
    )
    
    # =============================================================================
    # STEP 4: Create comparison plot (original vs reflected)
    # =============================================================================
    print_section('Creating Comparison Plot')
    print("Creating original vs reflected comparison...")
    
    # Plot comparison showing both base mechanisms
    variation_plot(
        target_joint=target_joint,
        out_path=OUTPUT_DIR / f'comparison_trajectories_{TIMESTAMP}.png',
        base_mechanism=mechanism,
        variation_mechanisms=[reflected_mechanism],  # Show reflected as a variation
        target_trajectory=target_trajectory,
        title='Original vs Reflected Mechanism',
        subtitle='Original (bold) vs Reflected (colored) with target trajectory (dashed)',
        show_linkages=False,
    )
    
    variation_plot(
        target_joint=target_joint,
        out_path=OUTPUT_DIR / f'comparison_linkages_{TIMESTAMP}.png',
        base_mechanism=mechanism,
        variation_mechanisms=[reflected_mechanism],
        target_trajectory=target_trajectory,
        title='Original vs Reflected Mechanism (Linkages)',
        subtitle='Original (bold) vs Reflected (colored) with target trajectory (dashed)',
        show_linkages=True,
    )
    
    # Create variation plot for reflected mechanism
    if result_reflected.mechanisms is not None:
        # Select sample mechanisms (best scoring valid ones)
        # Handle scores array - use it if available, otherwise create default
        if result_reflected.scores is not None:
            scores_array = result_reflected.scores
        else:
            scores_array = np.full(len(result_reflected.mechanisms), np.inf)
        
        valid_mechanisms = [
            (mech, score) for mech, score, is_valid in 
            zip(result_reflected.mechanisms, scores_array, result_reflected.is_valid)
            if mech is not None and is_valid and score is not None and np.isfinite(score)
        ]
        
        if valid_mechanisms:
            # Sort by score and take top N
            valid_mechanisms.sort(key=lambda x: x[1])
            sample_mechanisms = [mech for mech, _ in valid_mechanisms[:N_VARIATION_PLOTS]]
            
            print(f"  Creating variation plot with {len(sample_mechanisms)} sample mechanisms...")
            
            # Plot trajectories
            variation_plot(
                target_joint=target_joint,
                out_path=OUTPUT_DIR / f'reflected_trajectories_{TIMESTAMP}.png',
                base_mechanism=reflected_mechanism,
                variation_mechanisms=sample_mechanisms,
                title='Reflected Mechanism - Sample Trajectories',
                subtitle=f'Showing {len(sample_mechanisms)} best-scoring variations',
                show_linkages=False,
            )
            
            # Plot linkages
            variation_plot(
                target_joint=target_joint,
                out_path=OUTPUT_DIR / f'reflected_linkages_{TIMESTAMP}.png',
                base_mechanism=reflected_mechanism,
                variation_mechanisms=sample_mechanisms,
                title='Reflected Mechanism - Sample Linkages',
                subtitle=f'Showing {len(sample_mechanisms)} best-scoring variations',
                show_linkages=True,
            )
    
    # Summary
    print_section('COMPLETE')
    print('\nOutput files:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
