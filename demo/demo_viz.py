"""
Visualization utilities for achievable target demos.

Provides plotting functions for trajectory and linkage comparisons.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pylink_tools.kinematic import compute_trajectory


# Styling
sns.set_theme(style='whitegrid', context='notebook')
COLORS = sns.color_palette('colorblind', 10)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_target_trajectory(pylink_data: dict, target_joint: str):
    """Compute trajectory for target joint."""
    result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)
    if result.success and target_joint in result.trajectories:
        return np.array(result.trajectories[target_joint])
    return None


def get_linkage_geometry(pylink_data: dict):
    """
    Extract node roles and edge connections from pylink_data.

    Returns:
        (node_roles, edges_list) where:
        - node_roles: {name: role}
        - edges_list: [(source_name, target_name, edge_id), ...]
    """
    node_roles = {}
    edges = []

    if 'linkage' in pylink_data and 'nodes' in pylink_data['linkage']:
        # Hypergraph format
        for name, node in pylink_data['linkage']['nodes'].items():
            role = node.get('role', 'follower')
            node_roles[name] = role

        for edge_id, edge in pylink_data['linkage'].get('edges', {}).items():
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                edges.append((source, target, edge_id))

    return node_roles, edges


def compute_linkage_state(pylink_data: dict, target_joint: str):
    """
    Compute trajectory and get node positions at first timestep.

    Returns:
        (trajectories, first_positions, target_traj) where:
        - trajectories: {joint_name: full trajectory array}
        - first_positions: {joint_name: (x, y) at t=0}
        - target_traj: trajectory of target joint as array
    """
    result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)
    if not result.success:
        return None, None, None

    trajectories = {}
    first_positions = {}

    for joint_name, positions in result.trajectories.items():
        traj_arr = np.array(positions)
        trajectories[joint_name] = traj_arr
        first_positions[joint_name] = (traj_arr[0, 0], traj_arr[0, 1])

    target_traj = trajectories.get(target_joint)

    return trajectories, first_positions, target_traj


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_variation_comparison(
    original_pylink_data: dict,
    target_joint: str,
    variations: list[dict],
    title: str,
    subtitle: str,
    out_path: Path,
):
    """
    Plot original mechanism with multiple variations overlaid.

    Args:
        original_pylink_data: The base mechanism
        target_joint: Joint to track
        variations: List of modified pylink_data dicts
        title: Main plot title
        subtitle: Description of variation type
        out_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Compute original trajectory
    original_traj = compute_target_trajectory(original_pylink_data, target_joint)
    if original_traj is None:
        print(f"  Warning: Could not compute original trajectory")
        return

    # Plot variation trajectories (lighter, in background)
    valid_count = 0
    for i, var_data in enumerate(variations):
        var_traj = compute_target_trajectory(var_data, target_joint)
        if var_traj is not None:
            valid_count += 1
            # Use a consistent light color with low alpha
            ax.plot(
                var_traj[:, 0], var_traj[:, 1],
                color=COLORS[i % len(COLORS)],
                linewidth=1.5,
                alpha=0.4,
                zorder=1,
            )
            ax.scatter(
                var_traj[:, 0], var_traj[:, 1],
                color=COLORS[i % len(COLORS)],
                s=20,
                alpha=0.3,
                zorder=2,
            )

    # Plot original trajectory (bold, in foreground)
    ax.plot(
        original_traj[:, 0], original_traj[:, 1],
        color='black',
        linewidth=3.5,
        alpha=1.0,
        label='Original',
        zorder=10,
    )
    ax.scatter(
        original_traj[:, 0], original_traj[:, 1],
        color='black',
        s=60,
        alpha=0.9,
        edgecolors='white',
        linewidths=1.5,
        zorder=11,
    )

    # Mark start point
    ax.scatter(
        [original_traj[0, 0]], [original_traj[0, 1]],
        color='green',
        s=150,
        marker='o',
        edgecolors='white',
        linewidths=2,
        zorder=12,
        label='Start',
    )

    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')

    # Add info text
    ax.text(
        0.02, 0.98,
        f'{valid_count} valid variations shown\n(lighter colored paths)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    ax.legend(loc='upper right', fontsize=11)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_linkage_variations(
    original_pylink_data: dict,
    target_joint: str,
    variations: list[dict],
    title: str,
    subtitle: str,
    out_path: Path,
):
    """
    Plot linkage structure (nodes and links) for original and variations.

    Shows linkages at their initial positions with trajectories of target joint.

    Args:
        original_pylink_data: The base mechanism
        target_joint: Joint to show trajectory for
        variations: List of modified pylink_data dicts
        title: Main plot title
        subtitle: Description of variation type
        out_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    # Get original geometry info and compute state
    node_roles, edges = get_linkage_geometry(original_pylink_data)
    orig_trajs, orig_positions, orig_target_traj = compute_linkage_state(
        original_pylink_data, target_joint,
    )

    if not orig_positions:
        print(f"  Warning: Could not compute original linkage state")
        return

    # Plot variation linkages and trajectories (lighter, in background)
    valid_count = 0
    for i, var_data in enumerate(variations):
        var_trajs, var_positions, var_target_traj = compute_linkage_state(var_data, target_joint)
        if var_positions and var_target_traj is not None:
            valid_count += 1
            color = COLORS[i % len(COLORS)]

            # Draw trajectory of target joint (light)
            ax.plot(
                var_target_traj[:, 0], var_target_traj[:, 1],
                color=color,
                linewidth=1.8,
                alpha=0.5,
                zorder=2,
            )

            # Draw edges (links) at first position
            for source, target, edge_id in edges:
                if source in var_positions and target in var_positions:
                    x1, y1 = var_positions[source]
                    x2, y2 = var_positions[target]
                    ax.plot(
                        [x1, x2], [y1, y2],
                        color=color,
                        linewidth=2.5,
                        alpha=0.6,
                        zorder=3,
                        solid_capstyle='round',
                    )

            # Draw nodes at first position
            for name, (x, y) in var_positions.items():
                role = node_roles.get(name, 'follower')
                marker = '^' if role == 'fixed' else ('D' if role == 'crank' else 'o')
                size = 60 if role == 'fixed' else (50 if role == 'crank' else 40)
                ax.scatter(
                    [x], [y],
                    color=color,
                    s=size,
                    marker=marker,
                    alpha=0.6,
                    zorder=4,
                    edgecolors='white',
                    linewidths=0.5,
                )

    # Draw original target trajectory (bold)
    if orig_target_traj is not None:
        ax.plot(
            orig_target_traj[:, 0], orig_target_traj[:, 1],
            color='black',
            linewidth=3.5,
            alpha=1.0,
            zorder=10,
            label=f'{target_joint} trajectory',
        )
        ax.scatter(
            orig_target_traj[:, 0], orig_target_traj[:, 1],
            color='black',
            s=25,
            alpha=0.7,
            zorder=10,
        )

    # Draw original linkage (bold, in foreground)
    for source, target, edge_id in edges:
        if source in orig_positions and target in orig_positions:
            x1, y1 = orig_positions[source]
            x2, y2 = orig_positions[target]
            ax.plot(
                [x1, x2], [y1, y2],
                color='#2C3E50',
                linewidth=4.5,
                alpha=1.0,
                zorder=11,
                solid_capstyle='round',
            )

    # Draw original nodes with labels
    legend_added = {'fixed': False, 'crank': False, 'follower': False}
    for name, (x, y) in orig_positions.items():
        role = node_roles.get(name, 'follower')

        if role == 'fixed':
            marker = '^'
            size = 180
            color = '#7F8C8D'
            label = 'Fixed joint' if not legend_added['fixed'] else None
            legend_added['fixed'] = True
        elif role == 'crank':
            marker = 'D'
            size = 150
            color = '#E74C3C'
            label = 'Crank' if not legend_added['crank'] else None
            legend_added['crank'] = True
        else:
            marker = 'o'
            size = 120
            color = '#3498DB'
            label = 'Follower' if not legend_added['follower'] else None
            legend_added['follower'] = True

        # Highlight target joint
        if name == target_joint:
            size *= 1.3
            color = '#9B59B6'
            label = f'Target ({name})'

        ax.scatter(
            [x], [y],
            color=color,
            s=size,
            marker=marker,
            edgecolors='white',
            linewidths=2.5,
            zorder=12,
            label=label,
        )

        # Add node label
        ax.annotate(
            name[:15] + ('...' if len(name) > 15 else ''),
            (x, y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            alpha=0.85,
            zorder=13,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
        )

    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_aspect('equal')

    # Add info text
    ax.text(
        0.02, 0.98,
        f'{valid_count} variations shown (colored)\nOriginal linkage in bold black',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    )

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")
