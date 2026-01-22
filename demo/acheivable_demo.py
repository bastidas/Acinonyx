"""
achievable_demo.py - Visualize achievable target variations.

This demo shows three types of achievable target generation:
1. Uniform link variation - all links vary by the same percentage
2. Selective link variation - custom limits per link, some excluded
3. Link + position variation - links vary AND static joints move

For each variation type, generates 12 random valid targets and plots them
with the original mechanism highlighted in bold.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.appconfig import USER_DIR
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimize import extract_dimensions_from_edges
from target_gen import AchievableTargetConfig
from target_gen import create_achievable_target
from target_gen import DimensionVariationConfig
from target_gen import StaticJointMovementConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mechanism to use
MECHANISM_TYPE = 'intermediate'  # 'simple', 'intermediate', 'complex', 'leg'

# Number of variations to generate per plot
N_VARIATIONS = 12

# Variation range for links
VARIATION_RANGE = 0.35  # ±35%

# Output settings
OUTPUT_DIR = USER_DIR / 'demo' / 'achievable_variations'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Mechanism file paths
TEST_GRAPHS_DIR = Path(__file__).parent / 'test_graphs'
MECHANISM_FILES = {
    'simple': TEST_GRAPHS_DIR / '4bar.json',
    'intermediate': TEST_GRAPHS_DIR / 'intermediate.json',
    'complex': TEST_GRAPHS_DIR / 'complex.json',
    'leg': TEST_GRAPHS_DIR / 'leg.json',
}

MECHANISM_CONFIG = {
    'simple': ('coupler_rocker_joint', 'Simple 4-bar'),
    'intermediate': ('final', 'Intermediate 6-link'),
    'complex': ('final_joint', 'Complex multi-link'),
    'leg': ('toe', 'Leg mechanism'),
}

# Styling
sns.set_theme(style='whitegrid', context='notebook')
COLORS = sns.color_palette('colorblind', 10)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_mechanism(mechanism_type: str):
    """Load a mechanism from test_graphs."""
    json_path = MECHANISM_FILES[mechanism_type]
    with open(json_path) as f:
        pylink_data = json.load(f)
    pylink_data['n_steps'] = 32
    target_joint, description = MECHANISM_CONFIG[mechanism_type]
    return pylink_data, target_joint, description


def get_dim_spec(pylink_data: dict, mechanism_type: str):
    """Get dimension spec for mechanism."""
    if mechanism_type in ('complex', 'intermediate', 'leg'):
        return extract_dimensions_from_edges(pylink_data, bounds_factor=1.5, min_length=3.0)
    else:
        return extract_dimensions(pylink_data, bounds_factor=1.5, min_length=3.0)


def compute_target_trajectory(pylink_data: dict, target_joint: str):
    """Compute trajectory for target joint."""
    result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)
    if result.success and target_joint in result.trajectories:
        return np.array(result.trajectories[target_joint])
    return None


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
        original_pylink_data, target_joint
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


def generate_variations(
    pylink_data: dict,
    target_joint: str,
    dim_spec,
    config: AchievableTargetConfig,
    n_variations: int,
) -> list[dict]:
    """Generate multiple achievable target variations."""
    variations = []
    
    for i in range(n_variations):
        # Create a new config with different seed for each variation
        var_config = AchievableTargetConfig(
            dimension_variation=config.dimension_variation,
            static_joint_movement=config.static_joint_movement,
            max_attempts=config.max_attempts,
            fallback_ranges=config.fallback_ranges,
            random_seed=config.random_seed + i if config.random_seed else i * 1000,
        )
        
        try:
            result = create_achievable_target(
                pylink_data,
                target_joint,
                dim_spec,
                config=var_config,
            )
            variations.append(result.target_pylink_data)
        except ValueError as e:
            print(f"  Warning: Could not generate variation {i+1}: {e}")
    
    return variations


# =============================================================================
# VARIATION CONFIGURATIONS
# =============================================================================

def get_variation_configs(dim_spec, base_seed: int = 42):
    """
    Get the three variation configurations.
    
    Returns list of (name, subtitle, config) tuples.
    """
    # Get dimension names for selective variation
    dim_names = dim_spec.names
    
    # Pick dimensions for selective control (if available)
    exclude_dim = dim_names[0] if len(dim_names) > 0 else None
    tight_dim = dim_names[1] if len(dim_names) > 1 else None
    wide_dim = dim_names[2] if len(dim_names) > 2 else None
    
    configs = []
    
    # VARIATION 1: Uniform link variation
    configs.append((
        'Variation 1: Uniform Link Adjustment',
        f'All links vary by ±{VARIATION_RANGE*100:.0f}%',
        AchievableTargetConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
            ),
            random_seed=base_seed,
        ),
    ))
    
    # VARIATION 2: Selective link variation
    overrides = {}
    exclude_list = []
    subtitle_parts = [f'Default: ±{VARIATION_RANGE*100:.0f}%']
    
    if exclude_dim:
        exclude_list.append(exclude_dim)
        subtitle_parts.append(f'Excluded: {exclude_dim[:20]}')
    if tight_dim:
        overrides[tight_dim] = (True, -0.1, 0.1)
        subtitle_parts.append(f'Tight (±10%): {tight_dim[:20]}')
    if wide_dim:
        overrides[wide_dim] = (True, -0.5, 0.5)
        subtitle_parts.append(f'Wide (±50%): {wide_dim[:20]}')
    
    configs.append((
        'Variation 2: Selective Link Control',
        '\n'.join(subtitle_parts),
        AchievableTargetConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
                exclude_dimensions=exclude_list,
                dimension_overrides=overrides,
            ),
            random_seed=base_seed + 1000,
        ),
    ))
    
    # VARIATION 3: Links + static joint positions
    configs.append((
        'Variation 3: Links + Joint Positions',
        f'Links: ±{VARIATION_RANGE*100:.0f}%, Static joints: ±10 units',
        AchievableTargetConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
            ),
            static_joint_movement=StaticJointMovementConfig(
                enabled=True,
                max_x_movement=20.0,
                max_y_movement=20.0,
            ),
            random_seed=base_seed + 2000,
        ),
    ))
    
    return configs


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the achievable variations demo."""
    print('=' * 60)
    print('  ACHIEVABLE TARGET VARIATIONS DEMO')
    print('=' * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {OUTPUT_DIR}')
    print(f'Mechanism type: {MECHANISM_TYPE}')
    print(f'Variations per plot: {N_VARIATIONS}')
    
    # Load mechanism
    print('\nLoading mechanism...')
    pylink_data, target_joint, description = load_mechanism(MECHANISM_TYPE)
    dim_spec = get_dim_spec(pylink_data, MECHANISM_TYPE)
    
    print(f'  {description}')
    print(f'  Target joint: {target_joint}')
    print(f'  Dimensions: {len(dim_spec)}')
    
    # Get variation configurations
    configs = get_variation_configs(dim_spec)
    
    # Generate and plot each variation type
    for i, (title, subtitle, config) in enumerate(configs, 1):
        print(f'\n--- {title} ---')
        print(f'  {subtitle.replace(chr(10), " | ")}')
        
        # Generate variations
        print(f'  Generating {N_VARIATIONS} variations...')
        variations = generate_variations(
            pylink_data,
            target_joint,
            dim_spec,
            config,
            N_VARIATIONS,
        )
        print(f'  Generated {len(variations)} valid variations')
        
        # Plot trajectory variations
        traj_path = OUTPUT_DIR / f'trajectory_variation_{i}_{TIMESTAMP}.png'
        plot_variation_comparison(
            pylink_data,
            target_joint,
            variations,
            title,
            subtitle,
            traj_path,
        )
        
        # Plot linkage variations
        linkage_path = OUTPUT_DIR / f'linkage_variation_{i}_{TIMESTAMP}.png'
        plot_linkage_variations(
            pylink_data,
            target_joint,
            variations,
            title.replace('Variation', 'Linkage Variation'),
            subtitle,
            linkage_path,
        )
    
    # Print summary
    print('\n' + '=' * 60)
    print('  DEMO COMPLETE')
    print('=' * 60)
    print(f'\nOutput files:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
