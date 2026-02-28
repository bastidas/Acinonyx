"""
Shared utilities for demo scripts.

This module provides common functions used across multiple demos:
- Mechanism loading from test_graphs/
- Dimension extraction for optimization
- Formatted output helpers
"""
from __future__ import annotations

import json
from pathlib import Path

from pylink_tools.hypergraph_adapter import extract_dimensions
from pylink_tools.mechanism import create_mechanism_from_dict as _create_mechanism_from_dict
from pylink_tools.mechanism import Mechanism
from pylink_tools.optimization_types import DimensionBoundsSpec

# =============================================================================
# MECHANISM REGISTRY
# =============================================================================
# Available test mechanisms with their configurations.

TEST_GRAPHS_DIR = Path(__file__).parent / 'test_graphs'

MECHANISMS = {
    'simple': {
        'file': TEST_GRAPHS_DIR / '4bar.json',
        'target_joint': 'coupler_rocker_joint',
        'description': 'Simple 4-bar linkage (4 joints, ~3 dimensions)',
    },
    'intermediate': {
        'file': TEST_GRAPHS_DIR / 'intermediate.json',
        'target_joint': 'final',
        'description': 'Intermediate 6-link mechanism (5 joints, ~5 dimensions)',
    },
    'complex': {
        'file': TEST_GRAPHS_DIR / 'complex.json',
        'target_joint': 'final_joint',
        'description': 'Complex multi-link mechanism (10 joints, ~15 dimensions)',
    },
    'leg': {
        'file': TEST_GRAPHS_DIR / 'leg.json',
        'target_joint': 'toe',
        'description': 'Leg mechanism (17 joints, ~28 dimensions)',
    },
}


def load_fourbar_data():
    """Load 4-bar from demo/test_graphs/4bar.json (hypergraph format)."""
    test_file = Path(__file__).parent / 'test_graphs' / '4bar.json'
    with open(test_file) as f:
        data = json.load(f)
    data['n_steps'] = 24
    return data


def create_mechanism_from_dict(
    pylink_data: dict,
    dim_spec: DimensionBoundsSpec | None = None,
    n_steps: int = 32,
) -> Mechanism:
    """
    Helper to create Mechanism from pylink_data dict.

    Delegates to pylink_tools.mechanism.create_mechanism_from_dict, then
    applies default variation config when bounds are empty (for demos/optimization).

    Args:
        pylink_data: Mechanism data dictionary
        dim_spec: Optional DimensionBoundsSpec (extracted if not provided)

    Returns:
        Mechanism object
    """
    if dim_spec is None:
        dim_spec, _ = extract_dimensions(pylink_data)
    mechanism = _create_mechanism_from_dict(pylink_data, dim_spec=dim_spec, n_steps=n_steps)
    if not dim_spec.bounds:
        from target_gen.variation_config import DEFAULT_VARIATION_CONFIG
        mechanism.apply_variation_config(DEFAULT_VARIATION_CONFIG)
    return mechanism


def load_mechanism(mechanism_type: str, n_steps: int = 32) -> tuple[Mechanism, str, str]:
    """
    Load a mechanism from test_graphs/ and return as Mechanism object.

    Args:
        mechanism_type: One of 'simple', 'intermediate', 'complex', 'leg'
        n_steps: Number of simulation steps per revolution

    Returns:
        (mechanism, target_joint, description)

    Raises:
        ValueError: If mechanism_type is unknown
        FileNotFoundError: If mechanism file doesn't exist
    """
    if mechanism_type not in MECHANISMS:
        available = list(MECHANISMS.keys())
        raise ValueError(f"Unknown mechanism '{mechanism_type}'. Available: {available}")

    config = MECHANISMS[mechanism_type]
    json_path = config['file']

    if not json_path.exists():
        raise FileNotFoundError(f'Mechanism file not found: {json_path}')

    with open(json_path) as f:
        pylink_data = json.load(f)

    pylink_data['n_steps'] = n_steps

    # Create Mechanism from pylink_data
    # For complex mechanisms, we might need edge-based extraction, but for now
    # we'll use the standard extract_dimensions which should work for all
    mechanism = create_mechanism_from_dict(pylink_data)

    return mechanism, config['target_joint'], config['description']


# def get_dimension_bounds_spec(mechanism: Mechanism) -> DimensionBoundsSpec:
#     """
#     Get DimensionBoundsSpec from a Mechanism object.

#     Args:
#         mechanism: Mechanism object

#     Returns:
#         DimensionBoundsSpec with names, initial_values, bounds
#     """
#     return mechanism.get_dimension_bounds_spec()


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print('\n' + '=' * width)
    print(f'  {title}')
    print('=' * width)


def print_mechanism_info(mechanism: Mechanism, target_joint: str, description: str):
    """Print summary of loaded mechanism."""
    print(f'\nMechanism: {description}')
    print(f'Target joint: {target_joint}')
    print(f'Joints: {len(mechanism.joint_names)}')
    print(f'Dimensions: {mechanism.n_dimensions}')


def print_dimensions(dim_spec, max_show: int = 6):
    """Print summary of optimizable dimensions."""
    print(f'\nOptimizable dimensions: {len(dim_spec)}')

    for i, (name, initial, bounds) in enumerate(
        zip(
            dim_spec.names, dim_spec.initial_values, dim_spec.bounds,
        ),
    ):
        if i >= max_show:
            print(f'  ... and {len(dim_spec) - max_show} more')
            break
        print(f'  - {name}: {initial:.2f} (bounds: {bounds[0]:.2f} to {bounds[1]:.2f})')
