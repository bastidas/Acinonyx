"""
Visualize sampling results with contour plots.

This demo:
1. Loads a mechanism (simple, intermediate, complex, or leg)
2. Creates an achievable target trajectory
3. Generates samples using Sobol sampling
4. Scores each sample using fitness function
5. Visualizes samples and underlying contour plot
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime

import logging

import numpy as np
import matplotlib.pyplot as plt

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s',
)
logging.getLogger('target_gen').setLevel(logging.DEBUG)
logging.getLogger('target_gen.sampling').setLevel(logging.DEBUG)  # Explicitly set for sampling module
logging.getLogger('pylink_tools').setLevel(logging.WARNING)  # Reduce noise

from configs.appconfig import USER_DIR
from demo.helpers import load_mechanism, print_section, print_mechanism_info
from pylink_tools.mechanism import create_mechanism_fitness
from target_gen.achievable_target import create_achievable_target
from target_gen.sampling import generate_samples, generate_valid_samples, generate_good_samples, SamplingResult
from target_gen.variation_config import DimensionVariationConfig
from pylink_tools.optimization_types import DimensionBoundsSpec
from viz_tools.spaces_viz import render_contours
from viz_tools.viz_styling import _save_or_show, STYLE
from multi.dim_tools import reduce_dimensions

# CONFIGURATION - Edit these to change demo behavior
# =============================================================================

# Which mechanism to visualize
MECHANISM = 'complex'  # Options: 'simple', 'intermediate', 'complex', 'leg'

# Output
OUTPUT_DIR = USER_DIR / 'demo' / 'sampling'
#TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TIMESTAMP = ""


def _prepare_plot_data(
    result: SamplingResult,
    dimension_bounds_spec: DimensionBoundsSpec,
    dim1_idx: int,
    dim2_idx: int,
    contour_samples_mask: np.ndarray | None = None,
    reduction_method: str = 'pca',
    reduction_fit_samples: str = 'all',
    reduction_epsilon: float | None = None,
    normalize_for_reduction: bool = True,
) -> tuple[
    np.ndarray,  # x_coords_all
    np.ndarray,  # y_coords_all
    tuple[float, float],  # x_bounds
    tuple[float, float],  # y_bounds
    str,  # dim1_name
    str,  # dim2_name
    dict | None,  # reduction_info
    np.ndarray | None,  # contour_x
    np.ndarray | None,  # contour_y
    np.ndarray | None,  # contour_values
]:
    """
    Prepare plot data (coordinates, bounds, labels) for visualization.
    
    Returns all the data needed to create the plot, including contour data.
    """
    all_samples = result.samples
    n_dims = all_samples.shape[1]
    
    # Detect which dimensions actually vary (have non-zero variance)
    # This handles the case where dimension_bounds_spec has 3+ dimensions but only 2 are varying
    sample_variances = np.var(all_samples, axis=0)
    varying_dim_indices = np.where(sample_variances > 1e-10)[0]  # Threshold for "constant" dimensions
    n_varying_dims = len(varying_dim_indices)
    
    # Determine if we need dimensionality reduction
    if n_varying_dims == 2:
        # Only 2 dimensions vary - use them directly (no PCA needed)
        # Use the first two varying dimensions, or map dim1_idx/dim2_idx to varying dims if valid
        if dim1_idx in varying_dim_indices and dim2_idx in varying_dim_indices:
            # Use requested dimensions if they're both varying
            actual_dim1_idx = dim1_idx
            actual_dim2_idx = dim2_idx
        else:
            # Use first two varying dimensions
            actual_dim1_idx = varying_dim_indices[0]
            actual_dim2_idx = varying_dim_indices[1]
            if dim1_idx not in varying_dim_indices or dim2_idx not in varying_dim_indices:
                print(f"  Note: Requested dims [{dim1_idx}, {dim2_idx}] not all varying. "
                      f"Using varying dims [{actual_dim1_idx}, {actual_dim2_idx}] instead.")
        
        dim1_name = dimension_bounds_spec.names[actual_dim1_idx]
        dim2_name = dimension_bounds_spec.names[actual_dim2_idx]
        x_bounds = dimension_bounds_spec.bounds[actual_dim1_idx]
        y_bounds = dimension_bounds_spec.bounds[actual_dim2_idx]
        x_coords_all = all_samples[:, actual_dim1_idx]
        y_coords_all = all_samples[:, actual_dim2_idx]
        reduction_info = None
    elif n_dims == 2:
        # Exactly 2 dimensions total (and both vary)
        dim1_name = dimension_bounds_spec.names[dim1_idx]
        dim2_name = dimension_bounds_spec.names[dim2_idx]
        x_bounds = dimension_bounds_spec.bounds[dim1_idx]
        y_bounds = dimension_bounds_spec.bounds[dim2_idx]
        x_coords_all = all_samples[:, dim1_idx]
        y_coords_all = all_samples[:, dim2_idx]
        reduction_info = None
    elif n_varying_dims > 2:
        # More than 2 dimensions vary - use dimensionality reduction
        print(f"  Using {reduction_method.upper()} for {n_varying_dims} varying dims (out of {n_dims} total) -> 2D reduction")
        
        # Determine which samples to use for fitting
        if reduction_fit_samples == 'valid':
            fit_mask = result.is_valid
        elif reduction_fit_samples == 'good' and result.scores is not None:
            if reduction_epsilon is None:
                raise ValueError("reduction_epsilon must be provided when reduction_fit_samples='good'")
            fit_mask = result.is_valid & (result.scores <= reduction_epsilon)
        else:
            fit_mask = None  # Use all samples
        
        # Perform reduction
        reduced_samples, reducer, metadata = reduce_dimensions(
            samples=all_samples,
            method=reduction_method,
            n_components=2,  # For 2D visualization
            fit_samples_mask=fit_mask,
            normalize=normalize_for_reduction,
            dimension_bounds=dimension_bounds_spec.bounds,  # For normalization
            random_state=42,
        )
        
        x_coords_all = reduced_samples[:, 0]
        y_coords_all = reduced_samples[:, 1]
        
        # Create axis labels based on method
        if reduction_method == 'pca' and metadata.get('variance_explained') is not None:
            var_explained = metadata['variance_explained']
            dim1_name = f'PC1 ({var_explained[0]:.1%} var)'
            dim2_name = f'PC2 ({var_explained[1]:.1%} var)'
        else:
            dim1_name = 'Dim 1 (reduced)'
            dim2_name = 'Dim 2 (reduced)'
        
        # Bounds are not meaningful after reduction - use min/max of reduced coords
        x_bounds = (np.min(x_coords_all), np.max(x_coords_all))
        y_bounds = (np.min(y_coords_all), np.max(y_coords_all))
        
        # Store reduction info for title
        reduction_info = {
            'method': reduction_method.upper(),
            'n_dims': n_varying_dims,  # Show number of varying dims, not total
            'variance_explained': metadata.get('variance_explained'),
        }
    elif n_varying_dims == 1:
        raise ValueError(f"Cannot plot with only 1 varying dimension. Need at least 2 varying dimensions.")
    elif n_varying_dims == 0:
        raise ValueError(f"Cannot plot: all dimensions are constant (no variation detected).")
    else:
        raise ValueError(f"Cannot plot with n_dims={n_dims}, n_varying_dims={n_varying_dims}. Need at least 2 varying dimensions.")
    
    # Determine which samples to use for contour interpolation
    if contour_samples_mask is not None:
        # Use provided mask
        contour_mask = contour_samples_mask & np.isfinite(result.scores) if result.scores is not None else None
    elif result.scores is not None:
        # Use all samples with finite scores
        contour_mask = np.isfinite(result.scores)
    else:
        contour_mask = None
    
    # Extract contour data
    if contour_mask is not None and np.any(contour_mask):
        contour_x = x_coords_all[contour_mask]
        contour_y = y_coords_all[contour_mask]
        contour_values = result.scores[contour_mask]
    else:
        contour_x = None
        contour_y = None
        contour_values = None
    
    return (
        x_coords_all, y_coords_all, x_bounds, y_bounds,
        dim1_name, dim2_name, reduction_info,
        contour_x, contour_y, contour_values
    )


def _create_plot_figure(
    x_coords_all: np.ndarray,
    y_coords_all: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    dim1_name: str,
    dim2_name: str,
    title: str,
    result: SamplingResult,
    reduction_info: dict | None,
    contour_x: np.ndarray | None,
    contour_y: np.ndarray | None,
    contour_values: np.ndarray | None,
    point_labels: list[str] | None,
    show_points: bool,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a single plot figure with contours and optionally points.
    
    Args:
        show_points: Whether to show individual sample points
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours (if we have valid data)
    if contour_x is not None:
        render_contours(
            x_coords=contour_x,
            y_coords=contour_y,
            values=contour_values,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            x_label=dim1_name,
            y_label=dim2_name,
            value_label='Fitness (MSE)',
            show_contour=True,
            show_contour_lines=False,
            show_points=False,  # Points plotted separately
            n_contour_levels=20,
            n_contour_line_levels=10,
            grid_resolution=100,
            interpolation_method='linear',
            fig=fig,
            ax=ax,
            verbose=1,
        )
    
    # Plot all points with labels (if requested)
    if show_points:
        if point_labels is None:
            # Default: use validity from result
            point_labels = ['valid' if is_valid else 'invalid' for is_valid in result.is_valid]
        
        render_contours(
            x_coords=x_coords_all,
            y_coords=y_coords_all,
            values=None,  # No values, just plotting points
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            x_label=dim1_name,
            y_label=dim2_name,
            show_contour=False,
            show_contour_lines=False,
            show_points=True,
            point_labels=point_labels,
            fig=fig,
            ax=ax,
            verbose=0,
        )
    
    # Calculate score range for title
    if result.scores is not None:
        finite_scores = result.scores[np.isfinite(result.scores)]
        if len(finite_scores) > 0:
            score_range_str = f'Score range: [{np.min(finite_scores):.2f}, {np.max(finite_scores):.2f}]'
        else:
            score_range_str = 'No finite scores'
    else:
        score_range_str = 'No scores available'
    
    # Set title
    if reduction_info is not None:
        title_suffix = f' ({reduction_info["method"]} projection from {reduction_info["n_dims"]}D)'
    else:
        title_suffix = ''
    
    ax.set_title(
        f'{title}{title_suffix}: {dim1_name} vs {dim2_name}\n'
        f'{len(result.samples)} total samples ({result.n_valid} valid, {result.n_invalid} invalid), {score_range_str}',
        fontsize=14,
        fontweight='bold',
    )
    
    plt.tight_layout()
    
    return fig, ax


def plot_sampling_result(
    result: SamplingResult,
    dimension_bounds_spec: DimensionBoundsSpec,
    dim1_idx: int,
    dim2_idx: int,
    title: str,
    filename: str,
    point_labels: list[str] | None = None,
    contour_samples_mask: np.ndarray | None = None,
    reduction_method: str = 'pca',
    reduction_fit_samples: str = 'all',
    reduction_epsilon: float | None = None,
    normalize_for_reduction: bool = True,
) -> None:
    """
    Helper function to plot a sampling result with contours and points.
    
    Creates two plots: one with points visible, one with only contours (for clarity when there are many points).
    
    Args:
        result: SamplingResult to plot
        dimension_bounds_spec: Dimension specification for bounds
        dim1_idx: Index of first dimension for x-axis (ignored if n_dims > 2)
        dim2_idx: Index of second dimension for y-axis (ignored if n_dims > 2)
        title: Plot title
        filename: Output filename (without extension) - will create two files: {filename}_with_points and {filename}_contour_only
        point_labels: Optional labels for each point (e.g., ['valid', 'invalid'])
        contour_samples_mask: Optional boolean mask for which samples to use for contour interpolation.
                             If None, uses all samples with finite scores.
        reduction_method: Dimensionality reduction method ('pca', 'tsne', 'umap') if n_dims > 2
        reduction_fit_samples: Which samples to use for fitting reducer ('all', 'valid', 'good')
        reduction_epsilon: Epsilon threshold for 'good' samples (required if reduction_fit_samples='good')
        normalize_for_reduction: Whether to normalize samples before reduction
    """
    # Prepare plot data (coordinates, bounds, labels, contour data)
    (
        x_coords_all, y_coords_all, x_bounds, y_bounds,
        dim1_name, dim2_name, reduction_info,
        contour_x, contour_y, contour_values
    ) = _prepare_plot_data(
        result=result,
        dimension_bounds_spec=dimension_bounds_spec,
        dim1_idx=dim1_idx,
        dim2_idx=dim2_idx,
        contour_samples_mask=contour_samples_mask,
        reduction_method=reduction_method,
        reduction_fit_samples=reduction_fit_samples,
        reduction_epsilon=reduction_epsilon,
        normalize_for_reduction=normalize_for_reduction,
    )
    
    # Create plot with points
    fig_with_points, ax_with_points = _create_plot_figure(
        x_coords_all=x_coords_all,
        y_coords_all=y_coords_all,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        dim1_name=dim1_name,
        dim2_name=dim2_name,
        title=title,
        result=result,
        reduction_info=reduction_info,
        contour_x=contour_x,
        contour_y=contour_y,
        contour_values=contour_values,
        point_labels=point_labels,
        show_points=True,
    )
    
    # Save plot with points
    out_path_with_points = OUTPUT_DIR / f'{filename}_with_points_{TIMESTAMP}.png'
    _save_or_show(fig_with_points, out_path_with_points, STYLE.dpi)
    print(f"  Saved: {out_path_with_points}")
    
    # Create plot without points (contour only)
    fig_contour_only, ax_contour_only = _create_plot_figure(
        x_coords_all=x_coords_all,
        y_coords_all=y_coords_all,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        dim1_name=dim1_name,
        dim2_name=dim2_name,
        title=title,
        result=result,
        reduction_info=reduction_info,
        contour_x=contour_x,
        contour_y=contour_y,
        contour_values=contour_values,
        point_labels=point_labels,
        show_points=False,
    )
    
    # Save plot without points
    out_path_contour_only = OUTPUT_DIR / f'{filename}_contour_only_{TIMESTAMP}.png'
    _save_or_show(fig_contour_only, out_path_contour_only, STYLE.dpi)
    print(f"  Saved: {out_path_contour_only}")


def main():
    """Main demo function."""
    print_section('SAMPLING VISUALIZATION DEMO')
    print(f'Output: {OUTPUT_DIR}')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load mechanism
    print_section('Loading Mechanism')
    mechanism, target_joint, description = load_mechanism(MECHANISM)
    dimension_bounds_spec = mechanism.get_dimension_bounds_spec()
    print_mechanism_info(mechanism, target_joint, description)
    
    # Configure dimension variation using DimensionVariationConfig
    # This makes it easy to exclude dimensions or set per-dimension variation ranges
    dimension_variation_config = DimensionVariationConfig(
        default_variation_range=10.0,  # ±100% variation by default (allows wide exploration)
        #exclude_dimensions=['rocker_distance'],  # Exclude this dimension from variation
        # You can also use dimension_overrides for fine-grained control:
        # dimension_overrides={
        #     'crank_link_distance': (True, -0.5, 0.5),  # ±50% for crank
        #     'coupler_distance': (True, -1.0, 1.0),    # ±100% for coupler
        # }
    )
    
    print(f"Mechanism loaded with {len(dimension_bounds_spec.names)} dimensions:")
    print(f"  Variation config: default_range=±{dimension_variation_config.default_variation_range*100:.0f}%")
    if dimension_variation_config.exclude_dimensions:
        print(f"  Excluded dimensions: {dimension_variation_config.exclude_dimensions}")
    for i, name in enumerate(dimension_bounds_spec.names):
        initial = dimension_bounds_spec.initial_values[i]
        enabled, min_pct, max_pct = dimension_variation_config.get_variation_for_dimension(name)
        if enabled:
            print(f"  {name}: initial={initial:.2f}, variation=±{max_pct*100:.0f}%")
        else:
            print(f"  {name}: initial={initial:.2f} (FIXED - no variation)")
    
    # Create an achievable target trajectory
    print_section('Creating Target Trajectory')
    print("Creating achievable target trajectory...")
    achievable_target_result = create_achievable_target(
        mechanism,
        target_joint,
        dim_spec=dimension_bounds_spec,
        config=None,
    )
    target_trajectory = achievable_target_result.target
    print(f"Target trajectory created for joint '{target_joint}' with {len(target_trajectory.positions)} points")
    
    # For visualization, we need exactly 2 dimensions
    if len(dimension_bounds_spec.names) < 2:
        print("ERROR: Need at least 2 dimensions for 2D visualization")
        return
    
    print_section('Visualizing Results')
    dim1_idx = 0
    dim2_idx = 1
    dim1_name = dimension_bounds_spec.names[dim1_idx]
    dim2_name = dimension_bounds_spec.names[dim2_idx]
    print(f"Visualizing dimensions: {dim1_name} (x-axis) vs {dim2_name} (y-axis)")
    
    # =============================================================================
    # STEP 1: Generate samples using generate_samples
    # =============================================================================
    print_section('Generating Samples')
    print("Generating samples with viability validation...")
    
    # Configuration for sampling
    n_samples = 2048
    max_attempts=50000
    sampling_mode = 'sobol'# 'full_combinatoric'  # Options: 'sobol', 'full_combinatoric', 'behnken'
    
    result = generate_samples(
        mechanism,
        dimension_bounds_spec=dimension_bounds_spec,
        n_requested=n_samples,
        sampling_mode=sampling_mode,
        dimension_variation_config=dimension_variation_config,
        target_trajectory=target_trajectory,
        target_joint=target_joint,
        metric='mse',
        phase_invariant=True,
        return_mechanisms=False,
        return_trajectories=False,
        seed=42
    )
    
    # Print statistics
    print(f"Generated {len(result.samples)} total samples ({result.n_valid} valid, {result.n_invalid} invalid)")
    print(f"Valid ratio: {result.n_valid}/{result.n_generated} ({result.n_valid/result.n_generated*100:.1f}%)")
    if result.scores is not None:
        finite_scores = result.scores[np.isfinite(result.scores)]
        if len(finite_scores) > 0:
            print(f"Score range: [{np.min(finite_scores):.4f}, {np.max(finite_scores):.4f}]")
            print(f"Mean score: {np.mean(finite_scores):.4f}")
    
    # Plot 2: All samples with valid/invalid labels
    plot_sampling_result(
        result=result,
        dimension_bounds_spec=dimension_bounds_spec,
        dim1_idx=dim1_idx,
        dim2_idx=dim2_idx,
        title='All Samples (generate_samples)',
        filename='sampling_all_samples',
        point_labels=None,  # Will use result.is_valid
        contour_samples_mask=result.is_valid,  # Use only valid samples for contour
    )
    
    # =============================================================================
    # STEP 2: Generate valid samples using generate_valid_samples
    # =============================================================================
    print_section('Generating Valid Samples')
    print("Generating valid samples (viability filtering)...")
    
    result_valid = generate_valid_samples(
        mechanism,
        dimension_bounds_spec=dimension_bounds_spec,
        n_valid_requested=n_samples,
        max_attempts=max_attempts,
        sampling_mode=sampling_mode,
        dimension_variation_config=dimension_variation_config,
        target_trajectory=target_trajectory,
        target_joint=target_joint,
        metric='mse',
        phase_invariant=True,
        return_all=True,
        return_mechanisms=False,
        return_trajectories=False,
        seed=42,
    )
    
    # Print statistics
    print(f"Generated {len(result_valid.samples)} total samples ({result_valid.n_valid} valid, {result_valid.n_invalid} invalid)")
    print(f"Valid ratio: {result_valid.n_valid}/{result_valid.n_generated} ({result_valid.n_valid/result_valid.n_generated*100:.1f}%)")
    if result_valid.scores is not None:
        finite_scores = result_valid.scores[np.isfinite(result_valid.scores)]
        if len(finite_scores) > 0:
            print(f"Score range: [{np.min(finite_scores):.4f}, {np.max(finite_scores):.4f}]")
            print(f"Mean score: {np.mean(finite_scores):.4f}")
    
    # Plot 3: Valid samples from generate_valid_samples
    plot_sampling_result(
        result=result_valid,
        dimension_bounds_spec=dimension_bounds_spec,
        dim1_idx=dim1_idx,
        dim2_idx=dim2_idx,
        title='Valid Samples (generate_valid_samples)',
        filename='sampling_valid_samples',
        point_labels=None,  # Will use result_valid.is_valid
    )
    
    # =============================================================================
    # STEP 3: Generate good samples using generate_good_samples
    # =============================================================================
    print_section('Generating Good Samples')
    print("Generating good samples (fitness filtering)...")
    
    result_good = generate_good_samples(
        mechanism,
        dimension_bounds_spec=dimension_bounds_spec,
        target_trajectory=target_trajectory,
        n_good_requested=n_samples,
        epsilon=500.0,
        max_attempts=max_attempts,
        sampling_mode=sampling_mode,
        dimension_variation_config=dimension_variation_config,
        target_joint=target_joint,
        metric='mse',
        phase_invariant=True,
        return_all=True,
        return_mechanisms=False,
        return_trajectories=False,
        seed=42,
    )
    
    # Print statistics
    print(f"Generated {len(result_good.samples)} total samples ({result_good.n_valid} good, {result_good.n_invalid} not good)")
    print(f"Good ratio: {result_good.n_valid}/{result_good.n_generated} ({result_good.n_valid/result_good.n_generated*100:.1f}%)")
    if result_good.scores is not None:
        good_scores = result_good.scores[result_good.is_valid]
        finite_scores = good_scores[np.isfinite(good_scores)]
        if len(finite_scores) > 0:
            print(f"Score range: [{np.min(finite_scores):.4f}, {np.max(finite_scores):.4f}]")
            print(f"Mean score: {np.mean(finite_scores):.4f}")
    
    # Plot 4: Good samples from generate_good_samples
    point_labels_good = ['good' if is_good else 'not_good' for is_good in result_good.is_valid]
    plot_sampling_result(
        result=result_good,
        dimension_bounds_spec=dimension_bounds_spec,
        dim1_idx=dim1_idx,
        dim2_idx=dim2_idx,
        title='Good Samples (generate_good_samples, epsilon=500)',
        filename='sampling_good_samples',
        point_labels=point_labels_good,
        contour_samples_mask=result_good.is_valid,  # Use only good samples for contour
    )
    
    # Summary
    print_section('COMPLETE')
    print(f'\nOutput files saved to: {OUTPUT_DIR}')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
