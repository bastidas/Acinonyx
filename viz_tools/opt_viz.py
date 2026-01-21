"""
opt_viz.py - Visualization helpers for linkage optimization.

Provides functions to visualize:
  - Current trajectory vs target trajectory
  - Optimization bounds
  - Linkage state at different iterations
  - Convergence progress

Uses seaborn for consistent styling and colorblind-friendly color palettes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Try to import project modules
try:
    from pylink_tools.optimize import (
        DimensionSpec, TargetTrajectory, OptimizationResult,
    )
    from pylink_tools.kinematic import compute_trajectory
except ImportError:
    pass  # Allow file to be imported even if pylink_tools not available


# =============================================================================
# Style Configuration - SEABORN STYLING
# =============================================================================

# Set consistent seaborn style for all plots
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
sns.set_palette('colorblind')

@dataclass
class OptVizStyle:
    """
    Style configuration for optimization visualizations.

    Uses seaborn's colorblind-friendly palette for all colors.
    """
    # Get seaborn's colorblind palette
    _palette = sns.color_palette('colorblind', 10)
    
    # Assign semantic colors from palette
    target_color: str = sns.color_palette('colorblind')[3]      # Red/coral
    current_color: str = sns.color_palette('colorblind')[0]     # Blue
    initial_color: str = sns.color_palette('colorblind')[2]     # Green
    optimized_color: str = sns.color_palette('colorblind')[4]   # Purple
    bounds_color: str = sns.color_palette('gray', 5)[2]         # Gray

    # Additional colors for multi-trajectory plots
    accent_colors: tuple = tuple(sns.color_palette('colorblind', 10))

    # Line properties
    target_linewidth: float = 3.0
    current_linewidth: float = 2.5
    trajectory_alpha: float = 0.85

    # Marker properties
    marker_size: float = 80
    start_marker: str = 'o'
    end_marker: str = 's'

    # Figure properties
    figsize: tuple[int, int] = (12, 10)
    dpi: int = 150

    # Bounds bar properties
    bar_height: float = 0.6
    bar_alpha: float = 0.7


DEFAULT_STYLE = OptVizStyle()


# =============================================================================
# Trajectory Comparison Visualization
# =============================================================================

def plot_trajectory_comparison(
    current_trajectory: list[tuple[float, float]],
    target: TargetTrajectory,
    title: str = 'Trajectory Comparison',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
    show_points: bool = True,
    show_lines: bool = True,
    show_error_vectors: bool = True,
    iteration: int | None = None,
    label_current: str = 'Current',
    label_target: str = 'Target',
) -> None:
    """
    Plot current trajectory vs target trajectory.

    Shows how the computed mechanism path compares to the desired target path.
    Uses high-contrast colors for easy distinction.

    Args:
        current_trajectory: List of (x, y) positions from current mechanism
        target: TargetTrajectory with target positions
        title: Plot title
        out_path: Path to save figure (None to show)
        style: Visualization style
        show_points: Show trajectory points
        show_lines: Show trajectory lines
        show_error_vectors: Show vectors from current to target
        iteration: Optional iteration number to display
        label_current: Label for current trajectory
        label_target: Label for target trajectory
    """
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    current = np.array(current_trajectory)
    target_pos = np.array(target.positions)

    n_points = len(current)
    n_target = len(target_pos)

    # Plot trajectory lines (use ax.plot to preserve point order)
    if show_lines:
        # Target trajectory - DASHED
        ax.plot(
            target_pos[:, 0], target_pos[:, 1],
            color=style.target_color, linewidth=style.target_linewidth,
            alpha=style.trajectory_alpha, linestyle='--', 
            label=label_target, zorder=2,
        )

        # Current trajectory - SOLID
        ax.plot(
            current[:, 0], current[:, 1],
            color=style.current_color, linewidth=style.current_linewidth,
            alpha=style.trajectory_alpha, linestyle='-',
            label=label_current, zorder=3,
        )

    # Plot error vectors (from current to target)
    if show_error_vectors:
        gray_color = sns.color_palette('gray', 5)[1]
        for i in range(min(n_points, n_target)):
            ax.annotate(
                '', xy=(target_pos[i, 0], target_pos[i, 1]),
                xytext=(current[i, 0], current[i, 1]),
                arrowprops=dict(
                    arrowstyle='->', color=gray_color,
                    alpha=0.5, lw=1.5,
                ),
            )

    # Plot trajectory points
    if show_points:
        # Target positions - CORAL with X markers
        ax.scatter(
            target_pos[:, 0], target_pos[:, 1],
            color=style.target_color,
            s=style.marker_size, marker='x',
            linewidths=2.5, zorder=4,
        )

        # Current positions - TEAL with circle markers
        ax.scatter(
            current[:, 0], current[:, 1],
            color=style.current_color,
            s=style.marker_size, marker='o',
            edgecolors='white', linewidths=1.5, zorder=5,
        )

    # Mark start position with star - GREEN
    ax.scatter(
        [current[0, 0]], [current[0, 1]], s=style.marker_size * 2.5,
        marker='*', color=style.initial_color, edgecolors='black',
        linewidths=1.5, zorder=6, label='Start',
    )

    # Compute and display error
    min_len = min(n_points, n_target)
    errors = np.sqrt(np.sum((current[:min_len] - target_pos[:min_len])**2, axis=1))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    mse = np.mean(errors**2)

    # Build title with stats
    full_title = title
    if iteration is not None:
        full_title += f' (Iteration {iteration})'
    full_title += f'\nMSE: {mse:.4f} | Mean Error: {mean_error:.3f} | Max Error: {max_error:.3f}'

    # Apply seaborn styling to labels
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


def plot_trajectory_overlay(
    trajectories: dict[str, list[tuple[float, float]]],
    target: TargetTrajectory,
    labels: dict[str, str] | None = None,
    title: str = 'Trajectory Overlay',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Plot multiple trajectories overlaid with the target.

    Useful for comparing initial vs optimized, or multiple optimization runs.
    Uses distinct colors for each trajectory.

    Args:
        trajectories: Dict of {name: trajectory} to plot
        target: Target trajectory
        labels: Optional custom labels for trajectories
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
    """
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    # Define distinct colors and styles for trajectories
    # Each trajectory gets a unique combination: color + linestyle + marker
    traj_styles = {
        'Initial': {
            'color': style.current_color,     # Teal blue
            'linestyle': ':',                 # Dotted
            'marker': 's',                    # Square (filled)
            'marker_fill': 'full',
            'linewidth': 2.0,
            'zorder': 3,
        },
    }

    # Fallback colors and styles for solver results
    solver_styles = [
        {'color': style.optimized_color, 'linestyle': '-', 'marker': 'o', 'marker_fill': 'full'},  # Purple solid
        {'color': '#27AE60', 'linestyle': '-', 'marker': '^', 'marker_fill': 'full'},  # Green solid
        {'color': '#2980B9', 'linestyle': '-', 'marker': 'v', 'marker_fill': 'full'},  # Blue solid
        {'color': '#8E44AD', 'linestyle': '-', 'marker': 'D', 'marker_fill': 'full'},  # Purple solid
        {'color': '#F39C12', 'linestyle': '-', 'marker': 'p', 'marker_fill': 'full'},  # Orange solid
    ]
    solver_idx = 0

    # Plot target FIRST (use ax.plot to preserve point order)
    target_pos = np.array(target.positions)
    ax.plot(
        target_pos[:, 0], target_pos[:, 1],
        color=style.target_color, linewidth=style.target_linewidth + 1.0,
        linestyle='--', label='Target', zorder=1, alpha=0.85,
    )
    ax.scatter(
        target_pos[:, 0], target_pos[:, 1],
        color=style.target_color, s=style.marker_size * 0.8,
        marker='x', linewidths=2.5, zorder=2, alpha=0.9,
    )

    # Plot each trajectory with distinct style
    for i, (name, traj) in enumerate(trajectories.items()):
        traj_arr = np.array(traj)
        label = labels.get(name, name) if labels else name

        # Get style for this trajectory
        if name in traj_styles:
            ts = traj_styles[name]
        else:
            # Use solver style (cycling through available styles)
            ts = solver_styles[solver_idx % len(solver_styles)]
            solver_idx += 1

        color = ts.get('color', style.accent_colors[i % len(style.accent_colors)])
        linestyle = ts.get('linestyle', '-')
        marker = ts.get('marker', 'o')
        marker_fill = ts.get('marker_fill', 'full')
        linewidth = ts.get('linewidth', style.current_linewidth)
        zorder_base = ts.get('zorder', 3 + i * 2)

        ax.plot(
            traj_arr[:, 0], traj_arr[:, 1],
            color=color, linewidth=linewidth,
            alpha=0.85, label=label, zorder=zorder_base, linestyle=linestyle,
        )

        # Marker scatter with appropriate fill
        facecolor = color if marker_fill == 'full' else 'none'
        ax.scatter(
            traj_arr[:, 0], traj_arr[:, 1],
            color=color, s=style.marker_size * 0.6,
            marker=marker, alpha=0.75, zorder=zorder_base + 1,
            facecolors=facecolor, edgecolors=color if marker_fill == 'none' else 'white',
            linewidths=1.5 if marker_fill == 'none' else 0.5,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


# =============================================================================
# Bounds Visualization
# =============================================================================

def plot_dimension_bounds(
    dimension_spec: DimensionSpec,
    initial_values: dict[str, float] | None = None,
    current_values: dict[str, float] | None = None,
    target_values: dict[str, float] | None = None,
    optimized_values: dict[str, float] | None = None,
    title: str = 'Optimization Bounds',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Visualize the optimization bounds for each dimension.

    Shows a horizontal bar chart with:
    - Gray bar: full bounds range
    - Green marker: initial value
    - Coral marker: target value (if provided)
    - Purple marker: optimized value (if provided)

    Args:
        dimension_spec: DimensionSpec with bounds and initial values
        initial_values: Optional dict of initial dimension values (overrides spec)
        current_values: Optional dict of current dimension values
        target_values: Optional dict of target values to show
        optimized_values: Optional dict of optimized values to show
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
    """
    n_dims = len(dimension_spec)

    fig, ax = plt.subplots(figsize=(12, max(5, n_dims * 1.0)), dpi=style.dpi)

    y_positions = np.arange(n_dims)

    # Track which legends we've added
    legend_added = {'initial': False, 'target': False, 'optimized': False, 'current': False}

    # Plot bounds bars
    for i, (name, bounds) in enumerate(zip(dimension_spec.names, dimension_spec.bounds)):
        lower, upper = bounds
        width = upper - lower

        # Bounds bar using seaborn palette
        edge_color = sns.color_palette('dark', 1)[0]
        ax.barh(
            i, width, left=lower, height=style.bar_height,
            color=style.bounds_color, alpha=style.bar_alpha,
            edgecolor=edge_color, linewidth=1,
        )

        # Initial value marker - GREEN (filled circle)
        if initial_values and name in initial_values:
            initial = initial_values[name]
        else:
            initial = dimension_spec.initial_values[i]

        ax.scatter(
            [initial], [i], color=style.initial_color,
            s=style.marker_size * 1.8, marker='o',
            zorder=5, label='Initial' if not legend_added['initial'] else '',
            edgecolors='white', linewidths=1.5,
        )
        legend_added['initial'] = True

        # Target value marker - CORAL RED (filled square)
        if target_values and name in target_values:
            target = target_values[name]
            ax.scatter(
                [target], [i], color=style.target_color,
                s=style.marker_size * 2.0, marker='s',
                zorder=6, label='Target' if not legend_added['target'] else '',
                edgecolors='white', linewidths=1.5,
            )
            legend_added['target'] = True

        # Optimized value marker - PURPLE (open diamond, won't cover others)
        if optimized_values and name in optimized_values:
            opt = optimized_values[name]
            ax.scatter(
                [opt], [i], color=style.optimized_color,
                s=style.marker_size * 3.0, marker='D',
                facecolors='none', linewidths=2.5,
                zorder=8, label='Optimized' if not legend_added['optimized'] else '',
            )
            legend_added['optimized'] = True

        # Current value marker - TEAL (open triangle, if different from initial)
        if current_values and name in current_values:
            current = current_values[name]
            if abs(current - initial) > 1e-6:
                ax.scatter(
                    [current], [i], color=style.current_color,
                    s=style.marker_size * 2.5, marker='^',
                    facecolors='none', linewidths=2.5,
                    zorder=9, label='Current' if not legend_added['current'] else '',
                )
                legend_added['current'] = True

        # Add bound labels on the sides
        ax.text(
            lower - width * 0.03, i, f'{lower:.1f}',
            ha='right', va='center', fontsize=9, color='#666',
        )
        ax.text(
            upper + width * 0.03, i, f'{upper:.1f}',
            ha='left', va='center', fontsize=9, color='#666',
        )

    # Format dimension names for display (shorter)
    display_names = [_shorten_dim_name(n) for n in dimension_spec.names]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(), by_label.keys(),
            loc='upper right', fontsize=10, framealpha=0.9,
        )

    # Adjust x limits for readability
    all_bounds = dimension_spec.bounds
    x_min = min(b[0] for b in all_bounds)
    x_max = max(b[1] for b in all_bounds)
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)

    sns.despine(ax=ax, left=True)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


def _shorten_dim_name(name: str) -> str:
    """Shorten dimension name for display."""
    # Remove common prefixes/suffixes
    name = name.replace('_distance', '')
    name = name.replace('distance', 'dist')
    # Keep it readable
    if len(name) > 25:
        parts = name.split('_')
        if len(parts) > 2:
            name = '_'.join([parts[0][:4], parts[-1]])
    return name


def plot_bounds_summary(
    dimension_spec: DimensionSpec,
    title: str = 'Dimension Bounds Summary',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Plot a summary table of dimension bounds.

    Args:
        dimension_spec: DimensionSpec with bounds
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(dimension_spec) * 0.6)), dpi=style.dpi)
    ax.axis('off')

    # Create table data
    table_data = []
    for name, initial, bounds in zip(
        dimension_spec.names,
        dimension_spec.initial_values,
        dimension_spec.bounds,
    ):
        display_name = _shorten_dim_name(name)
        table_data.append([display_name, f'{bounds[0]:.2f}', f'{initial:.2f}', f'{bounds[1]:.2f}'])

    columns = ['Dimension', 'Min', 'Initial', 'Max']

    table = ax.table(
        cellText=table_data, colLabels=columns,
        cellLoc='center', loc='center',
        colColours=['#E8E8E8'] * 4,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.6)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


# =============================================================================
# Linkage State Visualization
# =============================================================================

def plot_linkage_state(
    pylink_data: dict,
    target: TargetTrajectory | None = None,
    title: str = 'Linkage State',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
    show_trajectory: bool = True,
    highlight_target_joint: bool = True,
) -> None:
    """
    Plot the current state of a linkage mechanism.

    Shows:
    - Joint positions
    - Link connections
    - Trajectory path (if show_trajectory=True)
    - Target trajectory (if provided)

    Args:
        pylink_data: Pylink document with mechanism data
        target: Optional target trajectory to overlay
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
        show_trajectory: Show computed trajectory
        highlight_target_joint: Highlight the target joint
    """
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    # Compute trajectory with skip_sync=True to use stored dimensions
    result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)

    if not result.success:
        ax.text(
            0.5, 0.5, f'Failed to compute trajectory:\n{result.error}',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='red',
        )
        plt.tight_layout()
        _save_or_show(fig, out_path, style.dpi)
        return

    trajectories = result.trajectories
    joint_types = result.joint_types

    # Distinct colors for each joint from seaborn palette
    palette = sns.color_palette('colorblind', 10)
    joint_colors = {
        'crank_anchor': sns.color_palette('dark', 1)[0],
        'rocker_anchor': sns.color_palette('dark', 1)[0],
        'crank': palette[1],
        'coupler_rocker_joint': style.current_color,
    }

    for i, (joint_name, positions) in enumerate(trajectories.items()):
        pos_arr = np.array(positions)
        joint_type = joint_types.get(joint_name, 'Unknown')

        is_target = target and joint_name == target.joint_name

        # Get color for this joint
        color = joint_colors.get(joint_name, style.accent_colors[i % len(style.accent_colors)])

        # Choose style based on joint type
        if joint_type == 'Static':
            marker = '^'
            size = style.marker_size * 1.8
            # Static joints don't move - just show position
            ax.scatter(
                [pos_arr[0, 0]], [pos_arr[0, 1]],
                color=color, s=size, marker=marker,
                edgecolors='black', linewidths=1.5,
                label=f'{joint_name} (Fixed)', zorder=6,
            )
        else:
            marker = 'o' if joint_type == 'Crank' else 's'
            size = style.marker_size

            # Highlight target joint
            if is_target and highlight_target_joint:
                color = style.current_color
                size *= 1.5

            # Plot trajectory path
            if show_trajectory:
                ax.plot(
                    pos_arr[:, 0], pos_arr[:, 1],
                    color=color, alpha=0.7, linewidth=2.0,
                    label=f'{joint_name} path', zorder=3,
                )

            # Plot current position (first point)
            ax.scatter(
                [pos_arr[0, 0]], [pos_arr[0, 1]],
                color=color, s=size, marker=marker,
                edgecolors='white', linewidths=1.5,
                label=f'{joint_name} ({joint_type})', zorder=5,
            )

    # Plot target trajectory if provided
    if target:
        target_pos = np.array(target.positions)
        ax.plot(
            target_pos[:, 0], target_pos[:, 1],
            color=style.target_color, linewidth=style.target_linewidth,
            linestyle='--', alpha=0.9, label='Target', zorder=2,
        )
        ax.scatter(
            target_pos[:, 0], target_pos[:, 1],
            color=style.target_color, s=style.marker_size * 0.6,
            marker='x', linewidths=2, alpha=0.7, zorder=4,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


def plot_linkage_comparison(
    pylink_data_initial: dict,
    pylink_data_target: dict,
    title: str = 'Linkage Comparison: Initial vs Target',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Plot two linkages overlaid for comparison.
    
    Uses very different markers and linestyles:
    - Initial: Open circles, dotted lines, transparent
    - Target: Open squares, dashed lines, solid
    
    Args:
        pylink_data_initial: Initial pylink document
        pylink_data_target: Target pylink document  
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
    """
    from pylink_tools.kinematic import compute_trajectory
    
    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    # Compute trajectories for both linkages
    result_initial = compute_trajectory(pylink_data_initial, verbose=False, skip_sync=True)
    result_target = compute_trajectory(pylink_data_target, verbose=False, skip_sync=True)

    if not result_initial.success:
        ax.text(
            0.5, 0.5, f'Failed to compute initial trajectory:\n{result_initial.error}',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='red',
        )
        plt.tight_layout()
        _save_or_show(fig, out_path, style.dpi)
        return
        
    if not result_target.success:
        ax.text(
            0.5, 0.5, f'Failed to compute target trajectory:\n{result_target.error}',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='red',
        )
        plt.tight_layout()
        _save_or_show(fig, out_path, style.dpi)
        return

    trajectories_initial = result_initial.trajectories
    joint_types_initial = result_initial.joint_types
    trajectories_target = result_target.trajectories
    joint_types_target = result_target.joint_types

    # Get seaborn palette colors
    palette = sns.color_palette('colorblind', 10)
    
    # Plot INITIAL linkage with open markers, dotted lines, very transparent
    initial_color = style.current_color  # Blue
    
    for i, (joint_name, positions) in enumerate(trajectories_initial.items()):
        pos_arr = np.array(positions)
        joint_type = joint_types_initial.get(joint_name, 'Unknown')
        
        # Use very transparent dotted lines for trajectory paths
        ax.plot(
            pos_arr[:, 0], pos_arr[:, 1],
            color=initial_color, alpha=0.2, linewidth=1.5,
            linestyle=':', zorder=1,
        )
        
        # Open circle markers for joints
        if joint_type == 'Static':
            marker = '^'
            size = style.marker_size * 1.5
        else:
            marker = 'o'
            size = style.marker_size * 0.8
            
        # Plot with open (unfilled) markers
        label = f'Initial {joint_name}' if i == 0 else None
        ax.scatter(
            [pos_arr[0, 0]], [pos_arr[0, 1]],
            facecolors='none', edgecolors=initial_color,
            s=size, marker=marker, linewidths=2,
            alpha=0.7, zorder=3,
            label='Initial Joints' if i == 0 else None,
        )
    
    # Plot TARGET linkage with open markers, dashed lines, more visible
    target_color = style.target_color  # Red/coral
    
    for i, (joint_name, positions) in enumerate(trajectories_target.items()):
        pos_arr = np.array(positions)
        joint_type = joint_types_target.get(joint_name, 'Unknown')
        
        # Dashed lines for trajectory paths
        ax.plot(
            pos_arr[:, 0], pos_arr[:, 1],
            color=target_color, alpha=0.6, linewidth=2.0,
            linestyle='--', zorder=2,
        )
        
        # Open square markers for joints
        if joint_type == 'Static':
            marker = '^'
            size = style.marker_size * 1.5
        else:
            marker = 's'  # Square for target
            size = style.marker_size * 0.9
            
        # Plot with open (unfilled) markers
        ax.scatter(
            [pos_arr[0, 0]], [pos_arr[0, 1]],
            facecolors='none', edgecolors=target_color,
            s=size, marker=marker, linewidths=2.5,
            alpha=0.9, zorder=4,
            label='Target Joints' if i == 0 else None,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


# =============================================================================
# Convergence Visualization
# =============================================================================

def plot_convergence_history(
    history: list[float],
    title: str = 'Optimization Convergence',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
    log_scale: bool = False,
) -> None:
    """
    Plot the convergence history of an optimization run.

    Args:
        history: List of error values over iterations
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
        log_scale: Use logarithmic y-scale
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=style.dpi)

    iterations = np.arange(len(history))

    # Main line
    ax.plot(
        iterations, history, color=style.current_color,
        linewidth=2.5, alpha=0.9,
    )

    # Add points at intervals
    step = max(1, len(history) // 20)
    ax.scatter(
        iterations[::step], [history[i] for i in range(0, len(history), step)],
        color=style.current_color, s=60, marker='o',
        edgecolors='white', linewidths=1, zorder=4,
    )

    # Mark best point - PURPLE
    best_idx = np.argmin(history)
    best_val = history[best_idx]
    ax.scatter(
        [best_idx], [best_val], color=style.optimized_color,
        s=style.marker_size * 2.5, marker='*', zorder=5,
        edgecolors='black', linewidths=1,
        label=f'Best: {best_val:.6f} @ iter {best_idx}',
    )

    # Mark initial and final
    ax.scatter(
        [0], [history[0]], color=style.initial_color,
        s=style.marker_size * 1.5, marker='s', zorder=5,
        edgecolors='black', linewidths=1,
        label=f'Initial: {history[0]:.6f}',
    )
    ax.scatter(
        [len(history)-1], [history[-1]], color=style.target_color,
        s=style.marker_size * 1.5, marker='D', zorder=5,
        edgecolors='black', linewidths=1,
        label=f'Final: {history[-1]:.6f}',
    )

    # Add improvement annotation
    if len(history) > 1 and history[0] != 0:
        improvement = (history[0] - history[-1]) / history[0] * 100
        bg_color = sns.color_palette('pastel', 1)[0]
        edge_color = sns.color_palette('gray', 3)[1]
        ax.annotate(
            f'Error Reduction: {improvement:.1f}%',
            xy=(0.98, 0.98), xycoords='axes fraction',
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5', facecolor=bg_color,
                edgecolor=edge_color, alpha=0.9,
            ),
        )

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Error (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    if log_scale and min(history) > 0:
        ax.set_yscale('log')
    
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


def plot_optimization_summary(
    result: OptimizationResult,
    dimension_spec: DimensionSpec,
    title: str = 'Optimization Summary',
    out_path: str | Path | None = None,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Create a multi-panel summary of an optimization run.

    Includes:
    - Convergence history
    - Dimension comparison (initial vs optimized)
    - Error metrics

    Args:
        result: OptimizationResult from optimization
        dimension_spec: DimensionSpec used
        title: Plot title
        out_path: Path to save figure
        style: Visualization style
    """
    fig = plt.figure(figsize=(16, 12), dpi=style.dpi)

    # Convergence plot (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    if result.convergence_history:
        ax1.plot(
            result.convergence_history, 
            color=style.current_color,
            linewidth=2.5, alpha=0.9,
        )

        # Mark best
        best_idx = np.argmin(result.convergence_history)
        best_val = result.convergence_history[best_idx]
        ax1.scatter(
            [best_idx], [best_val], color=style.optimized_color,
            s=150, marker='*', zorder=5, edgecolors='black',
        )

        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Error', fontsize=11)
        ax1.set_title('Convergence History', fontsize=12, fontweight='bold')
        sns.despine(ax=ax1)
    else:
        ax1.text(0.5, 0.5, 'No history available', ha='center', va='center', fontsize=12)

    # Dimensions comparison (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    n_dims = len(dimension_spec)
    x = np.arange(n_dims)
    width = 0.35

    initial_vals = dimension_spec.initial_values
    opt_vals = [
        result.optimized_dimensions.get(name, initial_vals[i])
        for i, name in enumerate(dimension_spec.names)
    ]

    ax2.bar(
        x - width/2, initial_vals, width, label='Initial',
        color=style.initial_color, alpha=0.85, edgecolor='black', linewidth=0.5,
    )
    ax2.bar(
        x + width/2, opt_vals, width, label='Optimized',
        color=style.optimized_color, alpha=0.85, edgecolor='black', linewidth=0.5,
    )

    # Shorter dimension names
    short_names = [_shorten_dim_name(n) for n in dimension_spec.names]
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Dimension Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    sns.despine(ax=ax2)

    # Error metrics (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    if result.initial_error > 0:
        improvement = (1 - result.final_error/result.initial_error)*100
        improvement_str = f'{improvement:.1f}%'
    else:
        improvement_str = 'N/A'

    metrics_text = f"""
    ╔══════════════════════════════════════════╗
    ║       OPTIMIZATION RESULTS               ║
    ╠══════════════════════════════════════════╣
    ║                                          ║
    ║   Status:      {'✓ SUCCESS' if result.success else '✗ FAILED':<20}   ║
    ║                                          ║
    ║   Initial Error:  {result.initial_error:<20.6f}   ║
    ║   Final Error:    {result.final_error:<20.6f}   ║
    ║                                          ║
    ║   Error Reduction: {improvement_str:<18}   ║
    ║                                          ║
    ║   Iterations:     {result.iterations:<20}   ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
    """

    bg_color = sns.color_palette('pastel', 1)[0]
    edge_color = sns.color_palette('gray', 3)[1]
    ax3.text(
        0.1, 0.9, metrics_text, transform=ax3.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5', facecolor=bg_color,
            edgecolor=edge_color, alpha=0.9,
        ),
    )

    # Optimized dimensions table (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    table_data = []
    for name in dimension_spec.names:
        initial = dimension_spec.initial_values[dimension_spec.names.index(name)]
        opt = result.optimized_dimensions.get(name, initial)
        change = ((opt - initial) / initial * 100) if initial != 0 else 0
        short_name = _shorten_dim_name(name)
        table_data.append([short_name, f'{initial:.2f}', f'{opt:.2f}', f'{change:+.1f}%'])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Dimension', 'Initial', 'Optimized', 'Change'],
        cellLoc='center', loc='center',
        colColours=['#E8E8E8'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.7)

    # Color the cells based on change
    for i, row in enumerate(table_data):
        change_val = float(row[3].replace('%', '').replace('+', ''))
        if abs(change_val) > 10:
            table[(i+1, 3)].set_facecolor('#FADBD8' if change_val < 0 else '#D5F5E3')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_or_show(fig, out_path, style.dpi)


# =============================================================================
# Utility Functions
# =============================================================================

def _save_or_show(fig, out_path: str | Path | None, dpi: int):
    """Save figure to path or show interactively."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved: {out_path}')
    else:
        plt.show()
        plt.close(fig)


def create_optimization_animation(
    pylink_data_sequence: list[dict],
    target: TargetTrajectory,
    out_path: str | Path,
    fps: int = 5,
    style: OptVizStyle = DEFAULT_STYLE,
) -> None:
    """
    Create an animation showing optimization progress.

    Args:
        pylink_data_sequence: List of pylink_data at each iteration
        target: Target trajectory
        out_path: Path to save animation (gif or mp4)
        fps: Frames per second
        style: Visualization style
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    target_pos = np.array(target.positions)

    def update(frame):
        ax.clear()

        pylink_data = pylink_data_sequence[frame]
        result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)

        if result.success and target.joint_name in result.trajectories:
            current = np.array(result.trajectories[target.joint_name])

            # Plot target
            ax.plot(
                target_pos[:, 0], target_pos[:, 1],
                color=style.target_color, linewidth=style.target_linewidth,
                linestyle='--', label='Target',
            )

            # Plot current
            ax.plot(
                current[:, 0], current[:, 1],
                color=style.current_color, linewidth=style.current_linewidth,
                label='Current',
            )

            ax.scatter(
                current[:, 0], current[:, 1],
                color=style.current_color, s=40, marker='o',
                edgecolors='white', linewidths=0.5,
            )

        ax.set_title(f'Optimization Progress - Iteration {frame}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=10)

    anim = FuncAnimation(
        fig, update, frames=len(pylink_data_sequence),
        interval=1000//fps, blit=False,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == '.gif':
        anim.save(out_path, writer='pillow', fps=fps)
    else:
        anim.save(out_path, fps=fps)

    plt.close(fig)
    print(f'Saved animation: {out_path}')
