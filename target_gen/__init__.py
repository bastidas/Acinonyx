"""
target_gen - Generalized utilities for creating achievable optimization targets.

This module provides tools for generating target trajectories that are
guaranteed to be achievable by modifying mechanism dimensions. This is
essential for testing and validating optimization algorithms.

Main components:
    Config classes:
        - MechVariationConfig: Master configuration object (dimension_variation, random_seed, max_attempts, etc.)
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

    Sampling functions:
        - generate_valid_samples(): Generate valid mechanism samples (DOE)
        - get_combinatoric_gradations(): Generate evenly-spaced values per dimension
        - get_mech_variations(): Generate all combinatorial variations
        - get_mech_variations_from_spec(): Generate variations from DimensionSpec

    Topology (future, NOT YET IMPLEMENTED):
        - TopologyChange, AddNodeChange, RemoveNodeChange, etc.
        - validate_topology_change(), apply_topology_change()

Basic usage:
    >>> from target_gen import create_achievable_target, MechVariationConfig
    >>> from pylink_tools import Mechanism
    >>>
    >>> # Create mechanism from your data
    >>> mechanism = Mechanism(...)
    >>>
    >>> # Simple usage with defaults (±50% dimension variation)
    >>> result = create_achievable_target(mechanism, "coupler_joint")
    >>> target = result.target
    >>> target_dims = result.target_dimensions
    >>>
    >>> # With configuration
    >>> config = MechVariationConfig(
    ...     dimension_variation=DimensionVariationConfig(
    ...         default_variation_range=0.3,
    ...         exclude_dimensions=["ground_distance"],
    ...     ),
    ...     random_seed=42,
    ... )
    >>> result = create_achievable_target(mechanism, "coupler_joint", config=config)
"""
from __future__ import annotations

from .achievable_target import AchievableTargetResult
from .achievable_target import apply_dimension_variations
from .achievable_target import apply_static_joint_movement
from .achievable_target import create_achievable_target
from .achievable_target import verify_mechanism_viable
from .sampling import generate_good_samples
from .sampling import generate_samples
from .sampling import generate_valid_samples
from .sampling import get_combinatoric_gradations
from .sampling import get_mech_variations
from .sampling import get_mech_variations_from_spec
from .sampling import SamplingResult
from .topology_changes import AddLinkChange
from .topology_changes import AddNodeChange
from .topology_changes import apply_topology_change
from .topology_changes import RemoveLinkChange
from .topology_changes import RemoveNodeChange
from .topology_changes import suggest_topology_changes
from .topology_changes import TopologyChange
from .topology_changes import validate_topology_change
from .variation_config import DimensionVariationConfig
from .variation_config import MechVariationConfig
from .variation_config import StaticJointMovementConfig
from .variation_config import TopologyChangeConfig

__all__ = [
    # Configuration
    'MechVariationConfig',
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
    'SamplingResult',
    # Sampling and DOE
    'get_combinatoric_gradations',
    'get_mech_variations',
    'get_mech_variations_from_spec',
    'generate_samples',
    'generate_valid_samples',
    'generate_good_samples',
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
