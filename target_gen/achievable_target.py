"""
achievable_target.py - Create achievable optimization targets.

This module provides functions to generate target trajectories that are
guaranteed to be achievable by modifying mechanism dimensions. This is
essential for testing and validating optimization algorithms.

The key insight: instead of creating arbitrary targets (which may be
geometrically impossible), we:
1. Start with a valid mechanism
2. Randomly vary its dimensions within bounds
3. Validate the modified mechanism is still viable
4. Use the resulting trajectory as the target

This creates an "inverse problem" with a KNOWN achievable solution.

Main functions:
    create_achievable_target() - Generate achievable target with config
    verify_mechanism_viable()  - Check if mechanism can complete full rotation
    apply_dimension_variations() - Apply configured variations to dimensions
    apply_static_joint_movement() - Move static joints (optional)
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from pylink_tools.kinematic import compute_trajectory
from pylink_tools.optimization_helpers import apply_dimensions
from pylink_tools.optimization_types import DimensionSpec
from pylink_tools.optimization_types import TargetTrajectory

from .variation_config import AchievableTargetConfig
from .variation_config import DimensionVariationConfig
from .variation_config import StaticJointMovementConfig


@dataclass
class AchievableTargetResult:
    """
    Result of create_achievable_target().
    
    Contains the generated target trajectory along with all information
    needed to understand and reproduce the target.
    
    Attributes:
        target: The generated TargetTrajectory object.
        target_dimensions: Dict of dimension values that produce this target.
        target_pylink_data: Modified mechanism data with target dimensions.
        static_joint_movements: Dict of movements applied to static joints
            (if any). Format: {joint_name: (dx, dy)}
        attempts_needed: Number of attempts to find valid configuration.
        fallback_range_used: The variation range that succeeded (None if primary).
    """
    target: TargetTrajectory
    target_dimensions: dict[str, float]
    target_pylink_data: dict
    static_joint_movements: dict[str, tuple[float, float]] = field(default_factory=dict)
    attempts_needed: int = 1
    fallback_range_used: float | None = None


def verify_mechanism_viable(
    pylink_data: dict,
    target_joint: str | None = None,
) -> bool:
    """
    Verify that a mechanism configuration is geometrically viable.
    
    A viable mechanism:
    1. Can complete a full crank rotation without breaking
    2. Has the target joint in its computed trajectories (if specified)
    
    Args:
        pylink_data: The mechanism data to verify.
        target_joint: Optional specific joint to check for in trajectories.
            If provided, the mechanism is only viable if this joint exists.
    
    Returns:
        True if the mechanism is viable, False otherwise.
    
    Example:
        >>> if verify_mechanism_viable(modified_data, "coupler_joint"):
        ...     print("Mechanism is valid")
    """
    try:
        result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)
        if not result.success:
            return False
        if target_joint and target_joint not in result.trajectories:
            return False
        return True
    except Exception:
        return False


def apply_dimension_variations(
    pylink_data: dict,
    dim_spec: DimensionSpec,
    config: DimensionVariationConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, float]]:
    """
    Apply dimension variations according to configuration.
    
    Varies each dimension based on its configuration settings, respecting
    per-dimension overrides and exclusions.
    
    Args:
        pylink_data: Base mechanism data (will not be modified).
        dim_spec: Specification of optimizable dimensions.
        config: Configuration for dimension variation.
        rng: NumPy random generator for reproducibility.
    
    Returns:
        (modified_pylink_data, applied_dimensions)
        - modified_pylink_data: Copy with new dimension values
        - applied_dimensions: Dict mapping dimension names to new values
    
    Example:
        >>> rng = np.random.default_rng(42)
        >>> config = DimensionVariationConfig(default_variation_range=0.3)
        >>> new_data, dims = apply_dimension_variations(data, spec, config, rng)
    """
    target_dims = {}
    
    for name, initial, bounds in zip(
        dim_spec.names, 
        dim_spec.initial_values, 
        dim_spec.bounds,
    ):
        enabled, min_pct, max_pct = config.get_variation_for_dimension(name)
        
        if not enabled:
            # Keep original value
            target_dims[name] = initial
            continue
        
        # Apply random variation within configured range
        factor = 1.0 + rng.uniform(min_pct, max_pct)
        new_value = initial * factor
        
        # Clamp to bounds
        new_value = max(bounds[0], min(bounds[1], new_value))
        target_dims[name] = new_value
    
    # Apply dimensions using the appropriate method for the data format
    modified_data = _apply_dims_for_format(pylink_data, target_dims)
    
    return modified_data, target_dims


def apply_static_joint_movement(
    pylink_data: dict,
    config: StaticJointMovementConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, tuple[float, float]]]:
    """
    Apply random movements to static joints according to configuration.
    
    Moves static (ground/frame) joints within the configured bounds.
    Linked joints are moved together to maintain relative positions.
    
    Args:
        pylink_data: Base mechanism data (will not be modified).
        config: Configuration for static joint movement.
        rng: NumPy random generator for reproducibility.
    
    Returns:
        (modified_pylink_data, movements_applied)
        - modified_pylink_data: Copy with moved joints
        - movements_applied: Dict mapping joint names to (dx, dy) movements
    
    Example:
        >>> config = StaticJointMovementConfig(enabled=True, max_x_movement=5.0)
        >>> rng = np.random.default_rng(42)
        >>> new_data, movements = apply_static_joint_movement(data, config, rng)
    """
    if not config.enabled:
        return pylink_data, {}
    
    result = copy.deepcopy(pylink_data)
    movements: dict[str, tuple[float, float]] = {}
    
    # Track which joints have already been moved (for linked joints)
    moved_joints: set[str] = set()
    
    # Get static joints from the mechanism
    static_joints = _get_static_joints(result)
    
    for joint_name in static_joints:
        if joint_name in moved_joints:
            continue
        
        enabled, max_x, max_y = config.get_movement_for_joint(joint_name)
        
        if not enabled:
            continue
        
        # Generate random movement
        dx = rng.uniform(-max_x, max_x)
        dy = rng.uniform(-max_y, max_y)
        
        # Apply movement to this joint
        _move_static_joint(result, joint_name, dx, dy)
        movements[joint_name] = (dx, dy)
        moved_joints.add(joint_name)
        
        # Apply same movement to linked joints
        for link_a, link_b in config.linked_joints:
            linked_joint = None
            if link_a == joint_name:
                linked_joint = link_b
            elif link_b == joint_name:
                linked_joint = link_a
            
            if linked_joint and linked_joint not in moved_joints:
                _move_static_joint(result, linked_joint, dx, dy)
                movements[linked_joint] = (dx, dy)
                moved_joints.add(linked_joint)
    
    return result, movements


def create_achievable_target(
    pylink_data: dict,
    target_joint: str,
    dim_spec: DimensionSpec,
    config: AchievableTargetConfig | None = None,
    # Legacy API compatibility
    randomize_range: float | None = None,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> AchievableTargetResult:
    """
    Create a target trajectory that is ACHIEVABLE by modifying the mechanism.
    
    This is the main entry point for generating achievable optimization targets.
    It randomizes mechanism dimensions within configured bounds, validates the
    resulting mechanism is viable, and returns the target trajectory.
    
    The function supports both:
    - New config-based API with full control via AchievableTargetConfig
    - Legacy API with simple parameters for backward compatibility
    
    Args:
        pylink_data: Base mechanism data.
        target_joint: Name of the joint whose trajectory to use as target.
        dim_spec: Specification of optimizable dimensions.
        config: Full configuration (overrides legacy params if provided).
        randomize_range: [LEGACY] ±percentage variation (0.5 = ±50%).
        seed: [LEGACY] Random seed for reproducibility.
        max_attempts: [LEGACY] Maximum attempts per variation range.
    
    Returns:
        AchievableTargetResult containing:
        - target: The generated TargetTrajectory
        - target_dimensions: Dict of dimension values that produce this target
        - target_pylink_data: Mechanism with target dimensions applied
        - static_joint_movements: Any static joint movements applied
        - attempts_needed: Number of attempts to find valid config
        - fallback_range_used: Variation range that succeeded (None if primary)
    
    Raises:
        ValueError: If no valid configuration found after all attempts.
        NotImplementedError: If topology changes are enabled in config.
    
    Example (new API):
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
        >>> print(f"Found target after {result.attempts_needed} attempts")
    
    Example (legacy API):
        >>> target, dims, target_data = create_achievable_target(
        ...     pylink_data, "coupler_joint", dim_spec,
        ...     randomize_range=0.35, seed=42
        ... )
    """
    # Handle legacy API vs new config API
    if config is None:
        config = _build_config_from_legacy(randomize_range, seed, max_attempts)
    
    # Validate config
    if config.topology_changes.enabled:
        raise NotImplementedError(
            "Topology changes are not yet implemented. "
            "Set topology_changes.enabled=False in config."
        )
    
    # Initialize random state
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    rng = np.random.default_rng(config.random_seed)
    
    # Build list of variation ranges to try
    # Start with primary range, then fallbacks
    ranges_to_try = [config.dimension_variation.default_variation_range]
    ranges_to_try.extend(config.fallback_ranges)
    
    total_attempts = 0
    
    for range_idx, variation_range in enumerate(ranges_to_try):
        is_fallback = range_idx > 0
        
        # Create a temporary config for this range
        temp_dim_config = DimensionVariationConfig(
            default_variation_range=variation_range,
            default_enabled=config.dimension_variation.default_enabled,
            dimension_overrides=config.dimension_variation.dimension_overrides,
            exclude_dimensions=config.dimension_variation.exclude_dimensions,
        )
        
        for attempt in range(config.max_attempts):
            total_attempts += 1
            
            # Apply dimension variations
            working_data = copy.deepcopy(pylink_data)
            modified_data, target_dims = apply_dimension_variations(
                working_data, dim_spec, temp_dim_config, rng
            )
            
            # Apply static joint movement (if enabled)
            joint_movements = {}
            if config.static_joint_movement.enabled:
                modified_data, joint_movements = apply_static_joint_movement(
                    modified_data, config.static_joint_movement, rng
                )
            
            # Compute trajectory with modified mechanism
            result = compute_trajectory(modified_data, verbose=False, skip_sync=True)
            
            if result.success and target_joint in result.trajectories:
                target_traj = result.trajectories[target_joint]
                target = TargetTrajectory(
                    joint_name=target_joint,
                    positions=[tuple(pos) for pos in target_traj],
                )
                
                # Log success if it took multiple attempts
                if total_attempts > 1:
                    range_info = f" (±{variation_range*100:.0f}%)" if is_fallback else ""
                    print(
                        f"  Found valid target dimensions after {total_attempts} attempts{range_info}"
                    )
                
                return AchievableTargetResult(
                    target=target,
                    target_dimensions=target_dims,
                    target_pylink_data=modified_data,
                    static_joint_movements=joint_movements,
                    attempts_needed=total_attempts,
                    fallback_range_used=variation_range if is_fallback else None,
                )
        
        # Log fallback
        if is_fallback and range_idx < len(ranges_to_try) - 1:
            print(
                f"  Warning: Couldn't find valid dimensions with ±{variation_range*100:.0f}%, "
                f"trying smaller range..."
            )
    
    raise ValueError(
        f"Could not find valid target dimensions after {total_attempts} attempts "
        f"across {len(ranges_to_try)} variation ranges."
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _build_config_from_legacy(
    randomize_range: float | None,
    seed: int | None,
    max_attempts: int | None,
) -> AchievableTargetConfig:
    """Build AchievableTargetConfig from legacy parameters."""
    dim_config = DimensionVariationConfig(
        default_variation_range=randomize_range if randomize_range is not None else 0.5,
    )
    
    return AchievableTargetConfig(
        dimension_variation=dim_config,
        max_attempts=max_attempts if max_attempts is not None else 128,
        random_seed=seed,
    )


def _apply_dims_for_format(pylink_data: dict, target_dims: dict) -> dict:
    """Apply dimensions using the appropriate method for the data format."""
    # Check if this is edges format (complex/hypergraph) or joints format (simple)
    if 'linkage' in pylink_data and 'edges' in pylink_data['linkage']:
        return _apply_edge_dimensions(pylink_data, target_dims)
    else:
        return apply_dimensions(pylink_data, target_dims)


def _apply_edge_dimensions(
    pylink_data: dict,
    dimensions: dict[str, float],
) -> dict:
    """
    Apply dimension values to a linkage edges format mechanism.
    
    Args:
        pylink_data: Mechanism data with 'linkage.edges' structure
        dimensions: Dict mapping dimension names to values
                   (e.g., {'crank_link_distance': 25.0})
    
    Returns:
        Copy of pylink_data with updated edge distances
    """
    result = copy.deepcopy(pylink_data)
    
    linkage = result.get('linkage', {})
    edges = linkage.get('edges', {})
    
    for dim_name, value in dimensions.items():
        # Parse edge_id from dimension name (e.g., 'crank_link_distance' -> 'crank_link')
        if dim_name.endswith('_distance'):
            edge_id = dim_name[:-len('_distance')]
            if edge_id in edges:
                edges[edge_id]['distance'] = value
    
    return result


def _get_static_joints(pylink_data: dict) -> list[str]:
    """Get list of static joint names from mechanism data."""
    static_joints = []
    
    # Check hypergraph format
    if 'linkage' in pylink_data and 'nodes' in pylink_data['linkage']:
        nodes = pylink_data['linkage']['nodes']
        for name, node in nodes.items():
            role = node.get('role', '')
            if role in ('static', 'ground', 'frame', 'fixed'):
                static_joints.append(name)
    
    # Check legacy format
    elif 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
        for joint in pylink_data['pylinkage']['joints']:
            if joint.get('type') == 'Static':
                static_joints.append(joint['name'])
    
    return static_joints


def _move_static_joint(pylink_data: dict, joint_name: str, dx: float, dy: float) -> None:
    """
    Move a static joint by (dx, dy) in place.
    
    Modifies pylink_data directly.
    """
    # Check hypergraph format
    if 'linkage' in pylink_data and 'nodes' in pylink_data['linkage']:
        nodes = pylink_data['linkage']['nodes']
        if joint_name in nodes:
            node = nodes[joint_name]
            # Handle "position": [x, y] array format
            if 'position' in node:
                pos = node['position']
                node['position'] = [pos[0] + dx, pos[1] + dy]
            else:
                # Fallback to x, y fields
                node['x'] = node.get('x', 0) + dx
                node['y'] = node.get('y', 0) + dy
    
    # Check legacy format
    elif 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
        for joint in pylink_data['pylinkage']['joints']:
            if joint['name'] == joint_name and joint.get('type') == 'Static':
                joint['x'] = joint.get('x', 0) + dx
                joint['y'] = joint.get('y', 0) + dy
                break


