#!/usr/bin/env python3
"""
Achievable Target Variations Demo - Visualize how dimension changes affect trajectories.

WHAT THIS DEMO DOES:
====================
Shows three types of achievable target generation:

1. UNIFORM VARIATION
   All link dimensions vary by the same percentage (e.g., ±35%)

2. SELECTIVE VARIATION
   Some dimensions are excluded, others have custom ranges
   (e.g., crank fixed, coupler ±50%, rocker ±10%)

3. POSITION VARIATION
   Link dimensions vary AND static joint positions move
   (shows how ground positions affect the trajectory)

For each type, generates N random valid variations and plots them
overlaid on the original mechanism.

WHY THIS MATTERS:
=================
- Shows the "achievable region" around a mechanism
- Demonstrates how different dimension changes affect trajectory
- Helps understand optimization landscapes
- Validates that randomized targets are physically achievable

RUN THIS DEMO:
==============
    python demo/acheivable_demo.py

Output saved to: user/demo/achievable_variations/
"""
from __future__ import annotations

from datetime import datetime

from configs.appconfig import USER_DIR
from demo.helpers import load_mechanism
from demo.helpers import print_section
from pylink_tools.mechanism import Mechanism
from target_gen import create_achievable_target
from target_gen import DimensionVariationConfig
from target_gen import MechVariationConfig
from target_gen import StaticJointMovementConfig
from viz_tools.demo_viz import variation_plot


# Which mechanism to visualize
MECHANISM = 'complex'  # Options: 'simple', 'intermediate', 'complex', 'leg'

# Number of random variations per plot
N_VARIATIONS = 4

# Default variation range for all dimensions
VARIATION_RANGE = 0.35  # ±35%

# Random seed for reproducibility
BASE_SEED = 42

# Number of trajectory steps for simulation
N_STEPS = 96  # Number of points per full revolution of the crank

# Output
OUTPUT_DIR = USER_DIR / 'demo' / 'achievable_variations'
# TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TIMESTAMP = ''


def get_variation_configs(dim_spec):
    """
    Define the three variation types to demonstrate.

    Returns list of (title, subtitle, config) tuples.
    """
    dim_names = dim_spec.names

    configs = []

    # -------------------------------------------------------------------------
    # Type 1: Uniform variation - all dimensions vary equally
    # -------------------------------------------------------------------------
    configs.append((
        'Type 1: Uniform Variation',
        f'All dimensions vary by ±{VARIATION_RANGE*100:.0f}%',
        MechVariationConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
            ),
            random_seed=BASE_SEED,
        ),
    ))

    # -------------------------------------------------------------------------
    # Type 2: Selective variation - custom control per dimension
    # -------------------------------------------------------------------------
    # Pick dimensions to demonstrate selective control
    exclude_dim = dim_names[0] if len(dim_names) > 0 else None
    tight_dim = dim_names[1] if len(dim_names) > 1 else None
    wide_dim = dim_names[2] if len(dim_names) > 2 else None

    overrides = {}
    exclude_list = []
    subtitle_parts = [f'Default: ±{VARIATION_RANGE*100:.0f}%']

    if exclude_dim:
        exclude_list.append(exclude_dim)
        subtitle_parts.append(f'Excluded: {exclude_dim[:20]}')
    if tight_dim:
        overrides[tight_dim] = (True, -0.1, 0.1)  # ±10%
        subtitle_parts.append(f'Tight (±10%): {tight_dim[:20]}')
    if wide_dim:
        overrides[wide_dim] = (True, -0.5, 0.5)  # ±50%
        subtitle_parts.append(f'Wide (±50%): {wide_dim[:20]}')

    configs.append((
        'Type 2: Selective Variation',
        '\n'.join(subtitle_parts),
        MechVariationConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
                exclude_dimensions=exclude_list,
                dimension_overrides=overrides,
            ),
            random_seed=BASE_SEED + 1000,
        ),
    ))

    # -------------------------------------------------------------------------
    # Type 3: Position variation - links + static joint positions
    # -------------------------------------------------------------------------
    configs.append((
        'Type 3: Link + Position Variation',
        f'Links: ±{VARIATION_RANGE*100:.0f}%, Static joints: ±20 units',
        MechVariationConfig(
            dimension_variation=DimensionVariationConfig(
                default_variation_range=VARIATION_RANGE,
            ),
            static_joint_movement=StaticJointMovementConfig(
                enabled=True,
                max_x_movement=20.0,
                max_y_movement=20.0,
            ),
            random_seed=BASE_SEED + 2000,
        ),
    ))

    return configs


def generate_variations(mechanism: Mechanism, target_joint: str, dim_spec, config, n_variations):
    """Generate N random achievable variations."""
    variations = []

    for i in range(n_variations):
        # Each variation gets a different seed
        var_config = MechVariationConfig(
            dimension_variation=config.dimension_variation,
            static_joint_movement=config.static_joint_movement,
            max_attempts=config.max_attempts,
            fallback_ranges=config.fallback_ranges,
            random_seed=config.random_seed + i if config.random_seed else i * 1000,
        )

        try:
            result = create_achievable_target(
                mechanism,
                target_joint,
                dim_spec=dim_spec,
                config=var_config,
                n_steps=N_STEPS,
            )
            # Keep as Mechanism object (no conversion to pylink_data)
            variations.append(result.target_mechanism)
        except ValueError as e:
            print(f'    Warning: Variation {i+1} failed: {e}')

    return variations


def main():
    print_section('ACHIEVABLE TARGET VARIATIONS DEMO')
    print(f'Mechanism: {MECHANISM}')
    print(f'Variations per type: {N_VARIATIONS}')
    print(f'Output: {OUTPUT_DIR}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mechanism
    print('\nLoading mechanism...')
    mechanism, target_joint, description = load_mechanism(MECHANISM, n_steps=N_STEPS)
    dim_spec = mechanism.get_dimension_bounds_spec()

    print(f'  {description}')
    print(f'  Target joint: {target_joint}')
    print(f'  Dimensions: {len(dim_spec)}')
    print(f'  Mechanism created with {len(mechanism.joint_names)} joints')
    print(f'  Simulation steps: {mechanism.n_steps}')

    # Get base trajectory (like in demo_viz.py line 284)
    # base_trajectory = mechanism.get_trajectory(target_joint)
    # print(f'  Base trajectory shape: {base_trajectory.shape}')  # Should be (N_STEPS, 2)

    # Get variation configs
    configs = get_variation_configs(dim_spec)

    # Generate and plot each variation type
    for i, (title, subtitle, config) in enumerate(configs, 1):
        print_section(f'{title}')
        print(f'  {subtitle.replace(chr(10), " | ")}')

        # Generate variations
        print(f'  Generating {N_VARIATIONS} variations...')
        variations = generate_variations(
            mechanism, target_joint, dim_spec, config, N_VARIATIONS,
        )
        print(f'  Generated {len(variations)} valid variations')

        # Plot trajectory variations
        variation_plot(
            target_joint=target_joint,
            out_path=OUTPUT_DIR / f'trajectory_type{i}_{TIMESTAMP}.png',
            base_mechanism=mechanism,
            variation_mechanisms=variations,
            title=title,
            subtitle=subtitle,
            show_linkages=False,
        )

        # Plot linkage variations
        variation_plot(
            target_joint=target_joint,
            out_path=OUTPUT_DIR / f'linkage_type{i}_{TIMESTAMP}.png',
            base_mechanism=mechanism,
            variation_mechanisms=variations,
            title=f'{title} (Linkages)',
            subtitle=subtitle,
            show_linkages=True,
        )

    # Summary
    print_section('COMPLETE')
    print('\nOutput files:')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')


if __name__ == '__main__':
    main()
