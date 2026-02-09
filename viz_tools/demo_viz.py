"""
Visualization utilities for linkage demos.

Provides unified plotting functions for:
  - Trajectory variations and comparisons
  - Linkage structure visualization
  - Convergence analysis (solver comparison)
  - Dimension bounds visualization

All functions use consistent styling defined at module level.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pylinkage.joints import Crank
from pylinkage.joints import Static

from pylink_tools.mechanism import Mechanism
from pylink_tools.optimization_types import TargetTrajectory
from viz_tools.opt_viz import _shorten_dim_name
from viz_tools.viz_styling import _save_or_show
from viz_tools.viz_styling import DemoVizStyle
from viz_tools.viz_styling import STYLE


def _get_linkage_geometry_from_mechanism(mechanism: Mechanism) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """
    Extract node roles and edges from a Mechanism for rendering.

    Returns (node_roles, edges) where:
    - node_roles: dict mapping joint_name -> role ('fixed', 'crank', 'follower')
    - edges: list of (source, target, edge_id) tuples

    Note: Uses mechanism.to_dict() only to extract linkage structure (edges),
    not for simulation or trajectory computation.
    """
    node_roles = {}
    edges = []

    linkage = mechanism.linkage

    # Get node roles from joints
    for joint in linkage.joints:
        name = joint.name
        if isinstance(joint, Static):
            node_roles[name] = 'fixed'
        elif isinstance(joint, Crank):
            node_roles[name] = 'crank'
        else:
            node_roles[name] = 'follower'

    # Get edges from mechanism's dict representation (structure only)
    # This is the only use of to_dict() - just for extracting edge connectivity
    mechanism_dict = mechanism.to_dict()
    linkage_dict = mechanism_dict.get('linkage', {})
    edges_dict = linkage_dict.get('edges', {})

    for edge_id, edge_data in edges_dict.items():
        source = edge_data.get('source')
        target = edge_data.get('target')
        if source and target:
            edges.append((source, target, edge_id))

    return node_roles, edges


def _get_initial_positions_from_mechanism(mechanism: Mechanism) -> dict[str, tuple[float, float]]:
    """
    Get initial positions of all joints from a Mechanism.

    Returns dict mapping joint_name -> (x, y)
    """
    coords = mechanism.linkage.get_coords()
    positions = {}
    for i, name in enumerate(mechanism.joint_names):
        if i < len(coords):
            x, y = coords[i]
            positions[name] = (float(x), float(y))
    return positions


def variation_plot(
    target_joint: str,
    out_path: Path | str,
    base_mechanism: Mechanism,
    # Target trajectory (dashed, distinct)
    target_mechanism: Mechanism | None = None,
    target_trajectory: TargetTrajectory | np.ndarray | list | None = None,
    # Additional variations (faded, background)
    variation_mechanisms: list[Mechanism] | None = None,
    variation_trajectories: list[np.ndarray | list] | None = None,
    # Display options
    title: str = 'Trajectory Variation',
    subtitle: str = '',
    show_linkages: bool = False,
    show_start_marker: bool = True,
    figsize: tuple = None,
    style: DemoVizStyle = STYLE,
):
    """
    Unified plot for trajectory and linkage variations.

    This function can display:
    - A base mechanism/trajectory (bold black, in foreground)
    - A target trajectory (dashed red, distinct)
    - Multiple variation trajectories (colored, semi-transparent)

    Supports two modes:
    - Trajectory-only: Just plot trajectory curves
    - With linkages: Also show mechanism structure at initial position

    Args:
        target_joint: Name of joint to track for trajectories
        out_path: Where to save the plot

        base_mechanism: Base mechanism (plots trajectory, optionally linkage)

        target_mechanism: Target mechanism for comparison
        target_trajectory: Pre-computed target trajectory (TargetTrajectory or array)

        variation_mechanisms: List of variation mechanisms
        variation_trajectories: List of pre-computed variation trajectories

        title: Main plot title
        subtitle: Description below title
        show_linkages: If True, draw linkage structures at initial positions
        show_start_marker: If True, mark trajectory start points
        figsize: Figure size override
        style: Style configuration

    Example usage:
        # Plot base with variations (achievable_demo style)
        variation_plot(
            target_joint='foot',
            base_mechanism=original_mechanism,
            variation_mechanisms=variations,
            title='Link Variations',
            out_path='variations.png',
        )

        # Plot target vs optimized (optimization result)
        variation_plot(
            target_joint='foot',
            base_mechanism=optimized_mechanism,
            target_trajectory=target_positions,
            title='Optimization Result',
            out_path='result.png',
        )
    """
    fig_size = figsize or (style.figsize if not show_linkages else (14, 11))
    fig, ax = plt.subplots(figsize=fig_size, dpi=style.dpi)

    # Track geometry info if showing linkages
    node_roles, edges = None, None
    if show_linkages and base_mechanism:
        node_roles, edges = _get_linkage_geometry_from_mechanism(base_mechanism)

    valid_variation_count = 0

    # -------------------------------------------------------------------------
    # Layer 1: Variations (faded, background)
    # -------------------------------------------------------------------------
    if variation_mechanisms or variation_trajectories:
        # Use provided trajectories or compute from mechanisms
        variations = variation_trajectories or []
        var_linkage_states = []
        # print(f'variation_mechanisms: {variation_mechanisms}')
        if variation_mechanisms:
            for var_mech in variation_mechanisms:
                var_traj = var_mech.get_trajectory(target_joint)
                if var_traj is not None and len(var_traj) > 0:
                    variations.append(var_traj)
                    if show_linkages:
                        var_positions = _get_initial_positions_from_mechanism(var_mech)
                        var_linkage_states.append(var_positions)

        # Plot variation trajectories
        for i, var_traj in enumerate(variations):
            # print(f'var_traj: {var_traj}')
            var_arr = np.array(var_traj)
            if var_arr.size == 0:
                continue

            valid_variation_count += 1
            color = style.variation_colors[i % len(style.variation_colors)]

            # Trajectory line
            ax.plot(
                var_arr[:, 0], var_arr[:, 1],
                color=color,
                linewidth=style.variation_linewidth,
                alpha=style.variation_alpha,
                zorder=1,
            )

            # Trajectory points (subtle)
            ax.scatter(
                var_arr[:, 0], var_arr[:, 1],
                color=color,
                s=style.variation_marker_size * 0.6,
                alpha=style.faded_alpha,
                zorder=2,
            )

        # Plot variation linkages if enabled
        if show_linkages and edges and var_linkage_states:
            for i, var_positions in enumerate(var_linkage_states):
                if not var_positions:
                    continue
                color = style.variation_colors[i % len(style.variation_colors)]

                # Draw edges (links)
                for source, target, _ in edges:
                    if source in var_positions and target in var_positions:
                        x1, y1 = var_positions[source]
                        x2, y2 = var_positions[target]
                        ax.plot(
                            [x1, x2], [y1, y2],
                            color=color,
                            linewidth=2.5,
                            alpha=style.variation_alpha,
                            zorder=3,
                            solid_capstyle='round',
                        )

                # Draw nodes
                for name, (x, y) in var_positions.items():
                    role = node_roles.get(name, 'follower') if node_roles else 'follower'
                    marker = '^' if role == 'fixed' else ('D' if role == 'crank' else 'o')
                    size = 60 if role == 'fixed' else (50 if role == 'crank' else 40)
                    ax.scatter(
                        [x], [y],
                        color=color,
                        s=size,
                        marker=marker,
                        alpha=style.variation_alpha,
                        zorder=4,
                        edgecolors='white',
                        linewidths=0.5,
                    )

    # -------------------------------------------------------------------------
    # Layer 2: Target trajectory (dashed, distinct)
    # -------------------------------------------------------------------------
    target_arr = None
    if target_trajectory is not None:
        # Handle TargetTrajectory type
        if hasattr(target_trajectory, 'positions'):
            target_arr = np.array(target_trajectory.positions)
        else:
            target_arr = np.array(target_trajectory)
    elif target_mechanism is not None:
        target_arr = target_mechanism.get_trajectory(target_joint)

    if target_arr is not None and target_arr.size > 0:
        ax.plot(
            target_arr[:, 0], target_arr[:, 1],
            color=style.target_color,
            linewidth=style.target_linewidth,
            linestyle=':',
            alpha=style.target_alpha,
            label='Target',
            zorder=8,
        )
        ax.scatter(
            target_arr[:, 0], target_arr[:, 1],
            color=style.target_color,
            s=style.marker_size,
            marker='p',
            facecolor=None,
            edgecolor=style.target_color,
            linewidths=.8,
            alpha=style.target_alpha,
            zorder=9,
        )

    # -------------------------------------------------------------------------
    # Layer 3: Base mechanism (bold, foreground)
    # -------------------------------------------------------------------------
    base_arr = None
    base_positions = None

    # Determine what to plot for base based on target type:
    # - If target_trajectory is provided → plot base trajectory (trajectory-only)
    # - If target_mechanism is provided → plot base mechanism (with linkages)
    base_arr = base_mechanism.get_trajectory(target_joint)

    # Set base positions for plotting mechanism linkages
    if show_linkages:
        base_positions = _get_initial_positions_from_mechanism(base_mechanism)

    if base_arr is not None and base_arr.size > 0:
        # Base trajectory (bold)
        ax.plot(
            base_arr[:, 0], base_arr[:, 1],
            color=style.base_color,
            linewidth=style.base_linewidth,
            alpha=style.base_alpha,
            label='Original' if variation_mechanisms else f'{target_joint} trajectory',
            zorder=10,
        )
        ax.scatter(
            base_arr[:, 0], base_arr[:, 1],
            color=style.base_color,
            s=style.base_marker_size * 1,
            alpha=0.8,
            edgecolors=None,
            linewidths=1.0,
            zorder=11,
        )

        # Start marker
        if show_start_marker:
            ax.scatter(
                [base_arr[0, 0]], [base_arr[0, 1]],
                color=style.initial_color,
                s=style.marker_size * 1,
                marker='o',
                # fillcolor=None,
                edgecolors=None,
                alpha=.8,
                linewidths=1.0,
                zorder=15,
                label='Start',
            )

    # Draw base linkage if enabled
    if show_linkages and edges and base_positions:
        # Draw edges
        for source, target, _ in edges:
            if source in base_positions and target in base_positions:
                x1, y1 = base_positions[source]
                x2, y2 = base_positions[target]
                ax.plot(
                    [x1, x2], [y1, y2],
                    color=style.base_color,
                    linewidth=style.link_linewidth,
                    alpha=style.base_alpha,
                    zorder=12,
                    solid_capstyle='round',
                )

        # Draw nodes with labels
        legend_added = {'fixed': False, 'crank': False, 'follower': False}
        for name, (x, y) in base_positions.items():
            role = node_roles.get(name, 'follower') if node_roles else 'follower'

            if role == 'fixed':
                marker, size, color = '^', 180, style.muted_color
                label = 'Fixed joint' if not legend_added['fixed'] else None
                legend_added['fixed'] = True
            elif role == 'crank':
                marker, size, color = 'D', 150, style.target_color
                label = 'Crank' if not legend_added['crank'] else None
                legend_added['crank'] = True
            else:
                marker, size, color = 'o', 120, '#3498DB'
                label = 'Follower' if not legend_added['follower'] else None
                legend_added['follower'] = True

            # Highlight target joint
            if name == target_joint:
                size *= 1.3
                color = style.optimized_color
                label = f'Target ({name})'

            ax.scatter(
                [x], [y],
                color=color,
                s=size,
                marker=marker,
                edgecolors='white',
                linewidths=2.5,
                zorder=13,
                label=label,
            )

            # Node label
            ax.annotate(
                name[:15] + ('...' if len(name) > 15 else ''),
                (x, y),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                alpha=0.85,
                zorder=14,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
            )

    full_title = title
    if subtitle:
        full_title += f'\n{subtitle}'

    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')

    # Info text
    if valid_variation_count > 0:
        info_text = f'{valid_variation_count} variations shown'
        if show_linkages:
            info_text += '\nOriginal in bold'
        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)


def plot_convergence_comparison(
    histories: dict[str, list[float]],
    out_path: Path | str,
    title: str = 'Convergence Comparison',
    layout: str = 'overlay',  # 'overlay', 'grid', 'individual'
    log_scale: bool = False,
    style: DemoVizStyle = STYLE,
):
    """
    Plot convergence histories for multiple optimizers.

    Args:
        histories: Dict of {solver_name: error_history_list}
        out_path: Where to save the plot
        title: Plot title
        layout: How to arrange plots:
            - 'overlay': All on same axes
            - 'grid': Subplots in a grid
            - 'individual': Save separate files (out_path used as base)
        log_scale: Use logarithmic y-scale
        style: Style configuration

    Example:
        histories = {
            'PSO': pso_result.convergence_history,
            'L-BFGS-B': lbfgs_result.convergence_history,
            'MLSL': mlsl_result.convergence_history,
        }
        plot_convergence_comparison(histories, 'convergence.png', layout='overlay')
    """
    if not histories:
        print('  Warning: No convergence histories to plot')
        return

    # Filter out empty histories
    histories = {k: v for k, v in histories.items() if v and len(v) > 0}
    if not histories:
        print('  Warning: All convergence histories are empty')
        return

    n_solvers = len(histories)
    colors = style.variation_colors

    if layout == 'overlay':
        # All histories on one plot
        fig, ax = plt.subplots(figsize=style.figsize_wide, dpi=style.dpi)

        for i, (name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)]
            iterations = np.arange(len(history))

            ax.plot(
                iterations, history,
                color=color,
                linewidth=2.0,
                alpha=0.85,
                label=name,
            )

            # Mark best point
            best_idx = np.argmin(history)
            best_val = history[best_idx]
            ax.scatter(
                [best_idx], [best_val],
                color=color,
                s=style.marker_size,
                marker='*',
                edgecolors='white',
                linewidths=1,
                zorder=5,
            )

        ax.set_xlabel('Iteration / Evaluation', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        _save_or_show(fig, out_path, style.dpi)

    elif layout == 'grid':
        # Subplots in a grid
        n_cols = min(3, n_solvers)
        n_rows = (n_solvers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            dpi=style.dpi,
            squeeze=False,
        )
        axes = axes.flatten()

        for i, (name, history) in enumerate(histories.items()):
            ax = axes[i]
            color = colors[i % len(colors)]
            iterations = np.arange(len(history))

            ax.plot(
                iterations, history,
                color=color,
                linewidth=2.0,
                alpha=0.85,
            )

            # Mark key points
            best_idx = np.argmin(history)
            best_val = history[best_idx]
            ax.scatter(
                [best_idx], [best_val], color=style.optimized_color,
                s=100, marker='*', zorder=5, edgecolors='white',
            )
            ax.scatter(
                [0], [history[0]], color=style.initial_color,
                s=60, marker='s', zorder=5, edgecolors='white',
            )
            ax.scatter(
                [len(history)-1], [history[-1]], color=style.target_color,
                s=60, marker='D', zorder=5, edgecolors='white',
            )

            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Error', fontsize=10)

            if log_scale and min(history) > 0:
                ax.set_yscale('log')

            # Add stats annotation
            if len(history) > 1 and history[0] > 0:
                reduction = (1 - history[-1] / history[0]) * 100
                ax.annotate(
                    f'Final: {history[-1]:.4f}\nReduction: {reduction:.1f}%',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                )

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save_or_show(fig, out_path, style.dpi)

    elif layout == 'individual':
        # Save separate files
        out_path = Path(out_path)
        base_name = out_path.stem
        suffix = out_path.suffix

        for i, (name, history) in enumerate(histories.items()):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=style.dpi)
            color = colors[i % len(colors)]
            iterations = np.arange(len(history))

            ax.plot(
                iterations, history,
                color=color,
                linewidth=2.5,
                alpha=0.9,
            )

            # Mark key points
            best_idx = np.argmin(history)
            best_val = history[best_idx]
            ax.scatter(
                [best_idx], [best_val], color=style.optimized_color,
                s=150, marker='*', zorder=5, edgecolors='black', linewidths=1,
                label=f'Best: {best_val:.6f} @ iter {best_idx}',
            )
            ax.scatter(
                [0], [history[0]], color=style.initial_color,
                s=100, marker='s', zorder=5, edgecolors='black', linewidths=1,
                label=f'Initial: {history[0]:.6f}',
            )
            ax.scatter(
                [len(history)-1], [history[-1]], color=style.target_color,
                s=100, marker='D', zorder=5, edgecolors='black', linewidths=1,
                label=f'Final: {history[-1]:.6f}',
            )

            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Error', fontsize=12)
            ax.set_title(f'{title}: {name}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)

            if log_scale and min(history) > 0:
                ax.set_yscale('log')

            # Improvement annotation
            if len(history) > 1 and history[0] > 0:
                reduction = (1 - history[-1] / history[0]) * 100
                ax.annotate(
                    f'Error Reduction: {reduction:.1f}%',
                    xy=(0.98, 0.85), xycoords='axes fraction',
                    ha='right', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F6F3', alpha=0.9),
                )

            plt.tight_layout()

            # Generate filename for this solver
            safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            individual_path = out_path.parent / f'{base_name}_{safe_name}{suffix}'
            _save_or_show(fig, individual_path, style.dpi)


def plot_dimension_bounds(
    dimension_spec,
    out_path: Path | str,
    initial_values: dict[str, float] | None = None,
    target_values: dict[str, float] | None = None,
    optimized_values: dict[str, float] | None = None,
    title: str = 'Dimension Bounds',
    style: DemoVizStyle = STYLE,
):
    """
    Visualize optimization bounds for each dimension.

    Shows a horizontal bar chart with:
    - Gray bar: full bounds range
    - Green circle: initial value
    - Red square: target value
    - Purple diamond: optimized value

    Args:
        dimension_spec: DimensionBoundsSpec with names, bounds, initial_values
        out_path: Where to save the plot
        initial_values: Override initial values from spec
        target_values: Target dimension values to mark
        optimized_values: Optimized dimension values to mark
        title: Plot title
        style: Style configuration
    """
    n_dims = len(dimension_spec)

    fig, ax = plt.subplots(
        figsize=(12, max(5, n_dims * 0.9)),
        dpi=style.dpi,
    )

    y_positions = np.arange(n_dims)
    bar_height = 0.6

    legend_added = {'initial': False, 'target': False, 'optimized': False}

    for i, (name, bounds) in enumerate(zip(dimension_spec.names, dimension_spec.bounds)):
        lower, upper = bounds
        width = upper - lower

        # Bounds bar
        ax.barh(
            i, width, left=lower, height=bar_height,
            color=style.bounds_color, alpha=0.7,
            edgecolor='#7F8C8D', linewidth=1,
        )

        # Initial value marker
        if initial_values and name in initial_values:
            initial = initial_values[name]
        else:
            initial = dimension_spec.initial_values[i]

        ax.scatter(
            [initial], [i],
            color=style.initial_color,
            s=style.marker_size * 1.5,
            marker='o',
            zorder=5,
            label='Initial' if not legend_added['initial'] else '',
            edgecolors='white',
            linewidths=1.5,
        )
        legend_added['initial'] = True

        # Target value marker
        if target_values and name in target_values:
            target = target_values[name]
            ax.scatter(
                [target], [i],
                color=style.target_color,
                s=style.marker_size * 1.8,
                marker='s',
                zorder=6,
                label='Target' if not legend_added['target'] else '',
                edgecolors='white',
                linewidths=1.5,
            )
            legend_added['target'] = True

        # Optimized value marker (open diamond, won't cover others)
        if optimized_values and name in optimized_values:
            opt = optimized_values[name]
            ax.scatter(
                [opt], [i],
                color=style.optimized_color,
                s=style.marker_size * 2.5,
                marker='D',
                facecolors='none',
                linewidths=2.5,
                zorder=8,
                label='Optimized' if not legend_added['optimized'] else '',
            )
            legend_added['optimized'] = True

        # Bound labels
        ax.text(
            lower - width * 0.03, i, f'{lower:.1f}',
            ha='right', va='center', fontsize=9, color='#666',
        )
        ax.text(
            upper + width * 0.03, i, f'{upper:.1f}',
            ha='left', va='center', fontsize=9, color='#666',
        )

    # Format dimension names
    display_names = [_shorten_dim_name(n) for n in dimension_spec.names]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(), by_label.keys(),
            loc='upper right', fontsize=10, framealpha=0.9,
        )

    # Adjust x limits
    all_bounds = dimension_spec.bounds
    x_min = min(b[0] for b in all_bounds)
    x_max = max(b[1] for b in all_bounds)
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)

    # Remove left spine for cleaner look
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, out_path, style.dpi)
