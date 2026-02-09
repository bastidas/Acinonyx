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

import logging
import random

import numpy as np
from pylinkage.bridge.solver_conversion import update_solver_positions
from pylinkage.joints import Static

from .variation_config import AchievableTargetResult
from .variation_config import DimensionVariationConfig
from .variation_config import MechVariationConfig
from .variation_config import StaticJointMovementConfig
from pylink_tools.mechanism import Mechanism
from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import TargetTrajectory

logger = logging.getLogger(__name__)


def verify_mechanism_viable(mechanism: Mechanism) -> bool:
    """
    Verify that a mechanism configuration is geometrically viable.

    A viable mechanism can complete a full crank rotation without breaking.

    Args:
        mechanism: Mechanism to verify.

    Returns:
        True if the mechanism is viable, False otherwise.
    """
    try:
        trajectory = mechanism.simulate()

        # Check for NaN (simulation failure)
        if np.isnan(trajectory).any():
            return False

        return True
    except Exception:
        return False


def apply_dimension_variations(
    mechanism: Mechanism,
    dim_spec: DimensionBoundsSpec,
    config: DimensionVariationConfig,
    rng: np.random.Generator,
) -> tuple[Mechanism, dict[str, float]]:
    """
    Apply dimension variations according to configuration.

    Varies each dimension based on its configuration settings, respecting
    per-dimension overrides and exclusions. Returns a new Mechanism with
    modified dimensions.

    Args:
        mechanism: Base mechanism (will not be modified).
        dim_spec: Specification of optimizable dimensions.
        config: Configuration for dimension variation.
        rng: NumPy random generator for reproducibility.

    Returns:
        (modified_mechanism, applied_dimensions)
        - modified_mechanism: New Mechanism with new dimension values
        - applied_dimensions: Dict mapping dimension names to new values
    """
    target_dims = {}

    for name, initial, bounds in zip(
        dim_spec.names,
        dim_spec.initial_values,
        dim_spec.bounds,
    ):
        enabled, min_pct, max_pct = config.get_variation_for_dimension(name)

        if not enabled:
            target_dims[name] = initial
            continue

        # Apply random variation within configured range
        factor = 1.0 + rng.uniform(min_pct, max_pct)
        new_value = initial * factor

        # Clamp to bounds
        new_value = max(bounds[0], min(bounds[1], new_value))
        target_dims[name] = new_value

    # Create new mechanism with modified dimensions using with_dimensions()
    modified_mechanism = mechanism.with_dimensions(target_dims)

    return modified_mechanism, target_dims


def apply_static_joint_movement(
    mechanism: Mechanism,
    config: StaticJointMovementConfig,
    rng: np.random.Generator,
) -> tuple[Mechanism, dict[str, tuple[float, float]]]:
    """
    Apply random movements to static joints according to configuration.

    Moves static (ground/frame) joints within the configured bounds using
    mechanism.linkage.set_coords(). Linked joints are moved together to
    maintain relative positions.

    Args:
        mechanism: Base mechanism (will not be modified).
        config: Configuration for static joint movement.
        rng: NumPy random generator for reproducibility.

    Returns:
        (modified_mechanism, movements_applied)
        - modified_mechanism: New Mechanism with moved joints
        - movements_applied: Dict mapping joint names to (dx, dy) movements
    """
    if not config.enabled:
        return mechanism, {}

    # Get current coordinates and convert to numpy array
    current_coords = mechanism.linkage.get_coords()
    # Convert to numpy array if it's a list/tuple
    if not isinstance(current_coords, np.ndarray):
        current_coords = np.array(current_coords, dtype=np.float64)

    # Create a copy of coordinates to modify
    new_coords = current_coords.copy()

    movements: dict[str, tuple[float, float]] = {}
    moved_joints: set[str] = set()

    # Find static joints by checking joint types
    static_joint_indices: dict[str, int] = {}
    for i, joint in enumerate(mechanism.linkage.joints):
        if isinstance(joint, Static):
            static_joint_indices[joint.name] = i

    for joint_name, joint_idx in static_joint_indices.items():
        if joint_name in moved_joints:
            continue

        enabled, max_x, max_y = config.get_movement_for_joint(joint_name)

        if not enabled:
            continue

        # Generate random movement
        dx = rng.uniform(-max_x, max_x)
        dy = rng.uniform(-max_y, max_y)

        # Apply movement to this joint's coordinates
        new_coords[joint_idx, 0] += dx
        new_coords[joint_idx, 1] += dy
        movements[joint_name] = (dx, dy)
        moved_joints.add(joint_name)

        # Apply same movement to linked joints
        for link_a, link_b in config.linked_joints:
            linked_joint = None
            if link_a == joint_name:
                linked_joint = link_b
            elif link_b == joint_name:
                linked_joint = link_a

            if linked_joint and linked_joint in static_joint_indices and linked_joint not in moved_joints:
                linked_idx = static_joint_indices[linked_joint]
                new_coords[linked_idx, 0] += dx
                new_coords[linked_idx, 1] += dy
                movements[linked_joint] = (dx, dy)
                moved_joints.add(linked_joint)

    # Create a copy of the mechanism and apply coordinate changes directly
    modified_mechanism = mechanism.copy()

    # Convert numpy array back to list of tuples format for set_coords
    # set_coords expects the same format as get_coords() returns
    coords_list = [(float(new_coords[i, 0]), float(new_coords[i, 1]))
                   for i in range(len(new_coords))]

    # Apply the coordinate changes using set_coords
    modified_mechanism.linkage.set_coords(coords_list)
    # Sync to solver
    update_solver_positions(modified_mechanism.linkage._solver_data, modified_mechanism.linkage)

    return modified_mechanism, movements


def create_achievable_target(
    mechanism: Mechanism,
    target_joint: str,
    dim_spec: DimensionBoundsSpec | None = None,
    config: MechVariationConfig | None = None,
    n_steps: int | None = None,
) -> AchievableTargetResult:
    """
    Create a target trajectory that is ACHIEVABLE by modifying the mechanism.

    It randomizes mechanism dimensions within configured bounds, validates the
    resulting mechanism is viable, and returns the target trajectory.

    Very simlar to generate_valid_samples, but only returns a single target trajectory.

    Args:
        mechanism: Base mechanism to modify.
        target_joint: Name of the joint whose trajectory to use as target.
        dim_spec: Specification of optimizable dimensions (extracted if None).
        config: Full configuration (defaults to ±50% variation if None).
        n_steps: Number of simulation steps (points in trajectory). If None, uses
                 the mechanism's current n_steps value.

    Returns:
        AchievableTargetResult containing:
        - target: The generated TargetTrajectory
        - target_dimensions: Dict of dimension values that produce this target
        - target_mechanism: Mechanism with target dimensions applied
        - static_joint_movements: Any static joint movements applied
        - attempts_needed: Number of attempts to find valid config
        - fallback_range_used: Variation range that succeeded (None if primary)

    Raises:
        ValueError: If no valid configuration found after all attempts.
        NotImplementedError: If topology changes are enabled in config.

    Example:
        >>> from pylink_tools import Mechanism
        >>> mechanism = Mechanism(...)  # Create mechanism from your data
        >>> config = MechVariationConfig(
        ...     dimension_variation=DimensionVariationConfig(
        ...         default_variation_range=0.3,
        ...         exclude_dimensions=["ground_distance"],
        ...     ),
        ...     random_seed=42,
        ... )
        >>> result = create_achievable_target(mechanism, "coupler_joint", config=config, n_steps=96)
        >>> print(f"Found target after {result.attempts_needed} attempts")
        >>> print(f"Target trajectory has {result.target.n_steps} points")
    """
    # Default config
    if config is None:
        config = MechVariationConfig()

    # Create a copy of the mechanism with specified n_steps if provided
    if n_steps is not None and n_steps != mechanism._n_steps:
        # Create a copy and modify its n_steps
        working_mechanism = mechanism.copy()
        working_mechanism._n_steps = n_steps
    else:
        working_mechanism = mechanism

    # Extract dimensions if not provided
    if dim_spec is None:
        dim_spec = working_mechanism.get_dimension_bounds_spec()

    # Validate target joint exists
    if target_joint not in working_mechanism.joint_names:
        raise ValueError(f"Target joint '{target_joint}' not found in mechanism. Available: {working_mechanism.joint_names}")

    # Validate config
    if config.topology_changes.enabled:
        raise NotImplementedError(
            'Topology changes are not yet implemented. '
            'Set topology_changes.enabled=False in config.',
        )

    # Validate static joint movement config against mechanism
    if config.static_joint_movement.enabled:
        try:
            config.static_joint_movement.validate(working_mechanism)
        except ValueError as e:
            logger.error(f'StaticJointMovementConfig validation failed: {e}')
            raise

    # Log start of target generation
    logger.info(
        f'Creating achievable target for joint "{target_joint}" '
        f'with {len(dim_spec)} optimizable dimensions',
    )
    logger.debug(
        f'Configuration: variation_range=±{config.dimension_variation.default_variation_range*100:.0f}%, '
        f'static_joint_movement={config.static_joint_movement.enabled}, '
        f'max_attempts={config.max_attempts}, '
        f'seed={config.random_seed}',
    )

    # Initialize random state
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    rng = np.random.default_rng(config.random_seed)

    # Build list of variation ranges to try
    ranges_to_try = [config.dimension_variation.default_variation_range]
    ranges_to_try.extend(config.fallback_ranges)

    logger.debug(f'Trying {len(ranges_to_try)} variation range(s): {[f"±{r*100:.0f}%" for r in ranges_to_try]}')

    total_attempts = 0

    for range_idx, variation_range in enumerate(ranges_to_try):
        is_fallback = range_idx > 0

        if is_fallback:
            logger.info(
                f'Trying fallback variation range ±{variation_range*100:.0f}% '
                f'(attempt {range_idx + 1}/{len(ranges_to_try)})',
            )

        # Create a temporary config for this range
        temp_dim_config = DimensionVariationConfig(
            default_variation_range=variation_range,
            default_enabled=config.dimension_variation.default_enabled,
            dimension_overrides=config.dimension_variation.dimension_overrides,
            exclude_dimensions=config.dimension_variation.exclude_dimensions,
        )

        for attempt in range(config.max_attempts):
            total_attempts += 1

            # Log progress every 10 attempts
            if total_attempts % 10 == 0:
                logger.debug(f'Attempt {total_attempts} (range ±{variation_range*100:.0f}%)')

            # Apply dimension variations (creates new mechanism)
            modified_mechanism, target_dims = apply_dimension_variations(
                working_mechanism, dim_spec, temp_dim_config, rng,
            )

            # Ensure modified mechanism uses the correct n_steps
            if n_steps is not None and modified_mechanism._n_steps != n_steps:
                modified_mechanism._n_steps = n_steps

            # Apply static joint movement (if enabled)
            joint_movements = {}
            if config.static_joint_movement.enabled:
                modified_mechanism, joint_movements = apply_static_joint_movement(
                    modified_mechanism, config.static_joint_movement, rng,
                )
                # Ensure n_steps is preserved after static joint movement
                if n_steps is not None and modified_mechanism._n_steps != n_steps:
                    modified_mechanism._n_steps = n_steps

            # Try to simulate with modified mechanism
            try:
                if verify_mechanism_viable(modified_mechanism):
                    trajectories = modified_mechanism.simulate_dict()

                    if target_joint in trajectories:
                        target_traj = trajectories[target_joint]
                        target = TargetTrajectory(
                            joint_name=target_joint,
                            positions=[tuple(pos) for pos in target_traj],
                        )

                        # Log success
                        range_info = f' with fallback range ±{variation_range*100:.0f}%' if is_fallback else ''
                        if total_attempts == 1:
                            logger.info(f'Found valid target on first attempt{range_info}')
                        else:
                            logger.info(
                                f'Found valid target after {total_attempts} attempts{range_info}',
                            )

                        if joint_movements:
                            logger.debug(
                                f'Applied static joint movements: {list(joint_movements.keys())}',
                            )

                        logger.debug(
                            f'Target dimensions: {len(target_dims)} dimensions modified, '
                            f'trajectory has {len(target_traj)} points',
                        )

                        return AchievableTargetResult(
                            target=target,
                            target_dimensions=target_dims,
                            target_mechanism=modified_mechanism,
                            static_joint_movements=joint_movements,
                            attempts_needed=total_attempts,
                            fallback_range_used=variation_range if is_fallback else None,
                        )
            except Exception as e:
                # Simulation failed, try again
                logger.debug(f'Attempt {total_attempts} failed: {type(e).__name__}')
                continue

        # Log fallback warning
        if is_fallback and range_idx < len(ranges_to_try) - 1:
            logger.warning(
                f"Couldn't find valid dimensions with ±{variation_range*100:.0f}% "
                f'after {config.max_attempts} attempts, trying smaller range...',
            )
        elif not is_fallback and len(ranges_to_try) > 1:
            logger.warning(
                f"Couldn't find valid dimensions with primary range ±{variation_range*100:.0f}% "
                f'after {config.max_attempts} attempts, trying fallback ranges...',
            )

    # All attempts failed
    logger.error(
        f'Failed to find valid target dimensions after {total_attempts} attempts '
        f'across {len(ranges_to_try)} variation range(s)',
    )
    logger.error(
        f'Mechanism may be over-constrained. Consider: '
        f'1) Increasing max_attempts (current: {config.max_attempts}), '
        f'2) Using smaller variation ranges, '
        f'3) Adjusting dimension bounds, '
        f'4) Checking mechanism geometry',
    )

    raise ValueError(
        f'Could not find valid target dimensions after {total_attempts} attempts '
        f'across {len(ranges_to_try)} variation ranges.',
    )
