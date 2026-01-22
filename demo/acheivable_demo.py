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
from demo.demo_viz import plot_linkage_variations
from demo.demo_viz import plot_variation_comparison
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimization_helpers import extract_dimensions_from_edges
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
# Plotting functions now imported from demo_viz modulex_inches='tight')
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
