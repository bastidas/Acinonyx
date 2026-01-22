"""
target_gen - Generalized utilities for creating achievable optimization targets.

This module provides tools for generating target trajectories that are
guaranteed to be achievable by modifying mechanism dimensions. This is
essential for testing and validating optimization algorithms.

Main components:
    Config classes:
        - AchievableTargetConfig: Master configuration object
        - DimensionVariationConfig: Control per-dimension randomization
        - StaticJointMovementConfig: Control static joint movement
        - TopologyChangeConfig: Future topology modifications (stub)

    Core functions:
        - create_achievable_target(): Generate achievable target with config
        - verify_mechanism_viable(): Check mechanism viability
        - apply_dimension_variations(): Apply configured dimension changes
        - apply_static_joint_movement(): Move static joints

    Result types:
        - AchievableTargetResult: Result from create_achievable_target()

    Topology (future, NOT YET IMPLEMENTED):
        - TopologyChange, AddNodeChange, RemoveNodeChange, etc.
        - validate_topology_change(), apply_topology_change()

Basic usage:
    >>> from solver_tools import create_achievable_target, AchievableTargetConfig
    >>>
    >>> # Simple usage with defaults (±50% dimension variation)
    >>> result = create_achievable_target(pylink_data, "coupler_joint", dim_spec)
    >>> target = result.target
    >>> target_dims = result.target_dimensions
    >>>
    >>> # With configuration
    >>> config = AchievableTargetConfig(
    ...     dimension_variation=DimensionVariationConfig(
    ...         default_variation_range=0.3,
    ...         exclude_dimensions=["ground_distance"],
    ...     ),
    ...     random_seed=42,
    ... )
    >>> result = create_achievable_target(
    ...     pylink_data, "coupler_joint", dim_spec, config=config
    ... )

Legacy API (backward compatible):
    >>> # Old-style call still works
    >>> result = create_achievable_target(
    ...     pylink_data, "coupler_joint", dim_spec,
    ...     randomize_range=0.35, seed=42
    ... )
    >>> # Access as tuple for compatibility
    >>> target = result.target
    >>> target_dims = result.target_dimensions
    >>> target_data = result.target_pylink_data
"""
from __future__ import annotations

from .achievable_target import AchievableTargetResult
from .achievable_target import apply_dimension_variations
from .achievable_target import apply_static_joint_movement
from .achievable_target import create_achievable_target
from .achievable_target import verify_mechanism_viable
from .topology_changes import AddLinkChange
from .topology_changes import AddNodeChange
from .topology_changes import apply_topology_change
from .topology_changes import RemoveLinkChange
from .topology_changes import RemoveNodeChange
from .topology_changes import suggest_topology_changes
from .topology_changes import TopologyChange
from .topology_changes import validate_topology_change
from .variation_config import AchievableTargetConfig
from .variation_config import DimensionVariationConfig
from .variation_config import StaticJointMovementConfig
from .variation_config import TopologyChangeConfig

# Sampling and DOE functions
from .sampling import generate_valid_samples
from .sampling import generate_viable_sobol_samples
from .sampling import get_combinatoric_gradations
from .sampling import get_mech_variations
from .sampling import get_mech_variations_from_spec
from .sampling import presample_valid_positions

# Configuration classes
# Core functions and result types
# Topology changes (stubs - not yet implemented)

__all__ = [
    # Configuration
    'AchievableTargetConfig',
    'DimensionVariationConfig',
    'StaticJointMovementConfig',
    'TopologyChangeConfig',
    # Core functions
    'create_achievable_target',
    'verify_mechanism_viable',
    'apply_dimension_variations',
    'apply_static_joint_movement',
    # Result types
    'AchievableTargetResult',
    # Sampling and DOE
    'get_combinatoric_gradations',
    'get_mech_variations',
    'get_mech_variations_from_spec',
    'presample_valid_positions',
    'generate_viable_sobol_samples',
    'generate_valid_samples',
    # Topology (stubs)
    'TopologyChange',
    'AddNodeChange',
    'RemoveNodeChange',
    'AddLinkChange',
    'RemoveLinkChange',
    'validate_topology_change',
    'apply_topology_change',
    'suggest_topology_changes',
]
