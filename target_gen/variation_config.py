"""
config.py - Configuration dataclasses for achievable target generation.

This module contains all configuration dataclasses used by the solver_tools
module for generating achievable optimization targets.

Configuration hierarchy:
    AchievableTargetConfig (main)
    ├── DimensionVariationConfig   - Control per-dimension randomization
    ├── StaticJointMovementConfig  - Control static joint movement (optional)
    └── TopologyChangeConfig       - Future: add/remove links/nodes (stub)
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class DimensionVariationConfig:
    """
    Per-dimension variation settings for target generation.
    
    Controls which dimensions to vary and by how much when creating
    achievable targets.
    
    Attributes:
        default_variation_range: Global ±percentage variation (0.5 = ±50%)
            Applied to all dimensions not explicitly configured.
        default_enabled: Whether dimensions are varied by default.
        dimension_overrides: Per-dimension settings.
            Maps dimension name to (enabled, min_pct, max_pct).
            e.g., {"crank_distance": (True, -0.3, 0.3)} for ±30% on crank.
            e.g., {"coupler_distance": (False, 0, 0)} to disable variation.
        exclude_dimensions: List of dimension names to exclude entirely.
            Convenience alternative to setting overrides with enabled=False.
    
    Example:
        >>> # Default ±50% variation for all dimensions except coupler
        >>> config = DimensionVariationConfig(
        ...     default_variation_range=0.5,
        ...     exclude_dimensions=["coupler_distance"]
        ... )
        
        >>> # Fine-grained control
        >>> config = DimensionVariationConfig(
        ...     default_variation_range=0.3,
        ...     dimension_overrides={
        ...         "crank_distance": (True, -0.5, 0.5),   # ±50% for crank
        ...         "rocker_distance": (True, -0.1, 0.1), # ±10% for rocker
        ...     }
        ... )
    """
    default_variation_range: float = 0.5
    default_enabled: bool = True
    dimension_overrides: dict[str, tuple[bool, float, float]] = field(default_factory=dict)
    exclude_dimensions: list[str] = field(default_factory=list)
    
    def get_variation_for_dimension(
        self, 
        dim_name: str,
    ) -> tuple[bool, float, float]:
        """
        Get variation settings for a specific dimension.
        
        Args:
            dim_name: Name of the dimension
            
        Returns:
            (enabled, min_pct, max_pct) tuple
        """
        # Check explicit overrides first
        if dim_name in self.dimension_overrides:
            return self.dimension_overrides[dim_name]
        
        # Check exclusion list
        if dim_name in self.exclude_dimensions:
            return (False, 0.0, 0.0)
        
        # Return defaults
        return (
            self.default_enabled,
            -self.default_variation_range,
            self.default_variation_range,
        )


@dataclass
class StaticJointMovementConfig:
    """
    Configuration for moving static joints.
    
    By default, static joints (ground/frame joints) are truly static.
    This config allows testing scenarios where static joints can move
    within specified bounds.
    
    Attributes:
        enabled: Master switch for static joint movement.
        max_x_movement: Default maximum X movement (absolute units, e.g., mm).
        max_y_movement: Default maximum Y movement (absolute units, e.g., mm).
        joint_overrides: Per-joint settings.
            Maps joint name to (enabled, max_x, max_y).
            e.g., {"ground": (True, 5.0, 5.0)} to allow ±5 movement.
        linked_joints: Pairs of joints that must move together.
            e.g., [("ground", "frame")] means if ground moves, frame moves too.
    
    Example:
        >>> # Allow static joints to move up to ±10 units
        >>> config = StaticJointMovementConfig(
        ...     enabled=True,
        ...     max_x_movement=10.0,
        ...     max_y_movement=10.0,
        ... )
        
        >>> # Only allow frame joint to move, not ground
        >>> config = StaticJointMovementConfig(
        ...     enabled=True,
        ...     joint_overrides={
        ...         "ground": (False, 0, 0),
        ...         "frame": (True, 5.0, 5.0),
        ...     }
        ... )
    """
    enabled: bool = False
    max_x_movement: float = 10.0
    max_y_movement: float = 10.0
    joint_overrides: dict[str, tuple[bool, float, float]] = field(default_factory=dict)
    linked_joints: list[tuple[str, str]] = field(default_factory=list)
    
    def get_movement_for_joint(
        self,
        joint_name: str,
    ) -> tuple[bool, float, float]:
        """
        Get movement settings for a specific joint.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            (enabled, max_x, max_y) tuple
        """
        if not self.enabled:
            return (False, 0.0, 0.0)
        
        if joint_name in self.joint_overrides:
            return self.joint_overrides[joint_name]
        
        return (True, self.max_x_movement, self.max_y_movement)


@dataclass
class TopologyChangeConfig:
    """
    Configuration for topology modifications (FUTURE/STUB).
    
    This is a placeholder for future functionality to add/remove
    links and nodes during target generation.
    
    WARNING: This functionality is NOT YET IMPLEMENTED. Setting enabled=True
    will raise NotImplementedError when create_achievable_target is called.
    
    Attributes:
        enabled: Master switch (must be False for now).
        add_node_probability: Probability of adding a node (0.0-1.0).
        remove_node_probability: Probability of removing a node (0.0-1.0).
        add_link_probability: Probability of adding a link (0.0-1.0).
        remove_link_probability: Probability of removing a link (0.0-1.0).
        min_nodes: Minimum number of nodes to preserve.
        max_nodes: Maximum number of nodes allowed.
        preserve_crank: Never remove the crank joint.
    
    Future implementation roadmap:
        Phase 1: Implement validate_topology_change() with safety checks
        Phase 2: Implement RemoveLinkChange (simplest - just removes an edge)
        Phase 3: Implement AddLinkChange (adds edge, auto-computes distance)
        Phase 4: Implement RemoveNodeChange (complex - must rewire connections)
        Phase 5: Implement AddNodeChange (most complex - must ensure valid DOF)
    """
    enabled: bool = False
    add_node_probability: float = 0.0
    remove_node_probability: float = 0.0
    add_link_probability: float = 0.0
    remove_link_probability: float = 0.0
    min_nodes: int = 3
    max_nodes: int = 32
    preserve_crank: bool = True


@dataclass
class AchievableTargetConfig:
    """
    Master configuration for achievable target generation.
    
    This is the main configuration object passed to create_achievable_target().
    It aggregates all sub-configurations for dimension variation, static joint
    movement, and topology changes.
    
    Attributes:
        dimension_variation: Config for which dimensions to vary and by how much.
        static_joint_movement: Config for moving static joints (disabled by default).
        topology_changes: Config for topology modifications (NOT IMPLEMENTED).
        max_attempts: Maximum attempts to find a valid configuration per range.
        fallback_ranges: Progressively smaller ranges to try if primary fails.
            The function will try the configured range first, then each fallback
            in order until a valid configuration is found.
        random_seed: Seed for reproducibility (None for random).
    
    Example:
        >>> # Default configuration: ±50% dimension variation
        >>> config = AchievableTargetConfig()
        
        >>> # Conservative: ±25% variation with more retries
        >>> config = AchievableTargetConfig(
        ...     dimension_variation=DimensionVariationConfig(
        ...         default_variation_range=0.25
        ...     ),
        ...     max_attempts=256,
        ... )
        
        >>> # Per-dimension control
        >>> config = AchievableTargetConfig(
        ...     dimension_variation=DimensionVariationConfig(
        ...         default_variation_range=0.3,
        ...         dimension_overrides={
        ...             "crank_link_distance": (True, -0.5, 0.5),
        ...         },
        ...         exclude_dimensions=["ground_frame_distance"],
        ...     ),
        ...     random_seed=42,
        ... )
    """
    dimension_variation: DimensionVariationConfig = field(
        default_factory=DimensionVariationConfig
    )
    static_joint_movement: StaticJointMovementConfig = field(
        default_factory=StaticJointMovementConfig
    )
    topology_changes: TopologyChangeConfig = field(
        default_factory=TopologyChangeConfig
    )
    max_attempts: int = 128
    fallback_ranges: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.15])
    random_seed: int | None = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.topology_changes.enabled:
            raise NotImplementedError(
                "Topology changes are not yet implemented. "
                "Set topology_changes.enabled=False."
            )
