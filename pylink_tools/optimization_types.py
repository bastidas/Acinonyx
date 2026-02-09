"""
optimization_types.py - Data structures for Mechanism-based optimization.

Contains dataclasses used throughout the optimization module:
  - DimensionBoundsSpec: Concrete specification of optimizable dimensions with absolute bounds
  - TargetTrajectory: Target positions for optimization
  - OptimizationResult: Result of an optimization run
  - Solution: Single solution from multi-solution optimization
  - MultiSolutionResult: Container for multiple distinct solutions

Design for Mechanism API efficiency:
  - TargetTrajectory caches numpy array for fast access
  - DimensionBoundsSpec provides numpy-native bounds accessors
  - Lazy conversion to avoid unnecessary allocations
  - OptimizationResult stores Mechanism objects (not dicts)
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism
    from target_gen.variation_config import MechVariationConfig


@dataclass
class DimensionMapping:
    """
    Maps dimension names to joint attributes for fast updates.

    Also contains bounds for mechanism dimensions.

    This allows direct mutation of Linkage joint attributes without
    dict conversion or recompilation.
    """
    names: list[str]
    initial_values: list[float]
    bounds: list[tuple[float, float]]
    # Mapping: dim_name -> (edge_id, property_name) for serialization
    edge_mapping: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Fast path: dim_idx -> (joint_obj, attr_name) for direct mutation
    joint_attrs: list[tuple[object, str] | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class DimensionBoundsSpec:
    """
    Concrete specification of optimizable dimensions with absolute bounds.

    Each dimension corresponds to a link length that can be adjusted
    during optimization. Used by Mechanism.from_pylink_data() to build
    the fast joint attribute mapping.

    Attributes:
        names: Human-readable names for each dimension (e.g., "crank_link_distance")
        initial_values: Current/starting values of each dimension
        bounds: (min, max) tuple for each dimension (absolute bounds)
        edge_mapping: Maps dimension name -> (edge_id, property_name)
                      e.g., {"crank_link_distance": ("crank_link", "distance")}
                      Used for serialization back to pylink_data format.
        weights: Per-dimension weights (higher = more important in optimization).
                 Defaults to uniform weights if not provided.

    Example:
        spec = DimensionBoundsSpec(
            names=['crank_distance', 'coupler_distance', 'rocker_distance'],
            initial_values=[20.0, 50.0, 40.0],
            bounds=[(10, 30), (40, 60), (30, 50)],
            edge_mapping={
                'crank_distance': ('crank_link', 'distance'),
                'coupler_distance': ('coupler', 'distance'),
                'rocker_distance': ('rocker', 'distance'),
            },
            weights=[1.0, 2.0, 1.0],  # coupler twice as important
        )
    """
    names: list[str]
    initial_values: list[float]
    bounds: list[tuple[float, float]]
    edge_mapping: dict[str, tuple[str, str]] | None = None
    weights: list[float] | None = None

    # Cached numpy arrays (created on first access)
    _arrays: dict = field(default_factory=dict, repr=False, compare=False)

    # Minimum allowed link length (prevents degenerate mechanisms)
    MIN_LENGTH: ClassVar[float] = 1.0

    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0] * len(self.names)

        # Enforce minimum link length: clamp all bounds to be at least MIN_LENGTH
        self.bounds = [
            (max(self.MIN_LENGTH, min_bound), max(self.MIN_LENGTH, max_bound))
            for min_bound, max_bound in self.bounds
        ]

    def __len__(self) -> int:
        return len(self.names)

    def _get_cached(self, key: str, values: list) -> np.ndarray:
        """Get or create cached numpy array."""
        if key not in self._arrays:
            self._arrays[key] = np.array(values, dtype=np.float64)
        return self._arrays[key]

    @property
    def initial_array(self) -> np.ndarray:
        """Initial values as numpy array (cached)."""
        return self._get_cached('initial', self.initial_values)

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds as numpy array (cached)."""
        return self._get_cached('lower', [b[0] for b in self.bounds])

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bounds as numpy array (cached)."""
        return self._get_cached('upper', [b[1] for b in self.bounds])

    @property
    def weights_array(self) -> np.ndarray:
        """Weights as numpy array (cached)."""
        return self._get_cached('weights', self.weights)

    def get_bounds_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return bounds in pylinkage format: ((lower...), (upper...))"""
        return (tuple(self.lower_bounds), tuple(self.upper_bounds))

    def get_bounds_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds as numpy arrays: (lower, upper)"""
        return (self.lower_bounds, self.upper_bounds)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/API responses."""
        return {
            'names': self.names,
            'initial_values': self.initial_values,
            'bounds': self.bounds,
            'weights': self.weights,
            'edge_mapping': {k: list(v) for k, v in self.edge_mapping.items()} if self.edge_mapping else None,
            'n_dimensions': len(self),
        }

    @classmethod
    def from_mechanism(
        cls,
        mechanism: Mechanism,
        mech_variation_config: MechVariationConfig | None = None,
    ) -> DimensionBoundsSpec:
        """
        Create DimensionBoundsSpec from mechanism, optionally applying MechVariationConfig.

        If mech_variation_config is provided:
        - Converts relative percentage bounds to absolute bounds
        - Filters out excluded dimensions
        - Applies per-dimension overrides

        If None, returns mechanism's current bounds as-is.

        Args:
            mechanism: Mechanism to extract bounds from
            mech_variation_config: Optional variation config to apply

        Returns:
            DimensionBoundsSpec with absolute bounds
        """
        base_spec = mechanism.get_dimension_bounds_spec()  # Gets current bounds

        if mech_variation_config is None:
            return base_spec

        dim_config = mech_variation_config.dimension_variation

        # Convert percentage variations to absolute bounds
        new_bounds = []
        new_names = []
        new_initial_values = []
        new_edge_mapping = {}

        for name, initial, (old_min, old_max) in zip(
            base_spec.names,
            base_spec.initial_values,
            base_spec.bounds,
        ):
            # Check if excluded
            if name in dim_config.exclude_dimensions:
                continue

            # Get variation settings
            enabled, min_pct, max_pct = dim_config.get_variation_for_dimension(name)

            if not enabled:
                # Use original bounds, but enforce minimum link length
                clamped_min = max(cls.MIN_LENGTH, old_min)
                clamped_max = max(cls.MIN_LENGTH, old_max)
                new_bounds.append((clamped_min, clamped_max))
            else:
                # Convert percentage to absolute bounds
                new_min = initial * (1 + min_pct)
                new_max = initial * (1 + max_pct)
                # Clamp to original bounds
                new_min = max(old_min, new_min)
                new_max = min(old_max, new_max)
                # Enforce minimum link length
                new_min = max(cls.MIN_LENGTH, new_min)
                new_max = max(cls.MIN_LENGTH, new_max)
                new_bounds.append((new_min, new_max))

            new_names.append(name)
            new_initial_values.append(initial)
            if base_spec.edge_mapping and name in base_spec.edge_mapping:
                new_edge_mapping[name] = base_spec.edge_mapping[name]

        return cls(
            names=new_names,
            initial_values=new_initial_values,
            bounds=new_bounds,
            edge_mapping=new_edge_mapping if new_edge_mapping else None,
            weights=base_spec.weights,
        )


@dataclass
class TargetTrajectory:
    """
    Target positions for optimization.

    Specifies where we want a particular joint to be at each timestep.
    Efficiently supports both list and numpy array access patterns.

    Attributes:
        joint_name: Which joint to optimize (match its path to target)
        positions: List of (x, y) positions, one per timestep
        weights: Optional per-timestep weights (higher = more important).
                 Defaults to uniform weights if not provided.

    Example:
        # From list of points
        target = TargetTrajectory(
            joint_name="coupler",
            positions=[(100, 120), (105, 125), (110, 130), ...],
        )

        # From numpy array (more efficient)
        target = TargetTrajectory.from_array("coupler", trajectory_array)

        # Access as numpy for optimization
        positions_np = target.positions_array  # shape (n_steps, 2), cached
    """
    joint_name: str
    positions: list[tuple[float, float]]
    weights: list[float] | None = None

    # Cached numpy array (created on first access)
    _array: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Convert positions to tuples if they're lists
        if self.positions and not isinstance(self.positions[0], tuple):
            self.positions = [tuple(p) for p in self.positions]
        # Set uniform weights if not provided
        if self.weights is None:
            self.weights = [1.0] * len(self.positions)

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.positions)

    @property
    def positions_array(self) -> np.ndarray:
        """
        Positions as numpy array of shape (n_steps, 2).

        Cached for efficient repeated access in optimization loops.
        """
        if self._array is None:
            object.__setattr__(self, '_array', np.array(self.positions, dtype=np.float64))
        return self._array

    @classmethod
    def from_array(cls, joint_name: str, positions: np.ndarray, weights: list[float] | None = None) -> TargetTrajectory:
        """
        Create TargetTrajectory directly from numpy array (efficient).

        Args:
            joint_name: Which joint to optimize
            positions: Numpy array of shape (n_steps, 2)
            weights: Optional per-timestep weights

        Returns:
            TargetTrajectory with pre-cached array (avoids reconversion)
        """
        instance = cls(
            joint_name=joint_name,
            positions=[(float(p[0]), float(p[1])) for p in positions],
            weights=weights,
        )
        # Pre-cache the array to avoid reconversion
        object.__setattr__(instance, '_array', np.asarray(positions, dtype=np.float64))
        return instance

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/API responses."""
        return {
            'joint_name': self.joint_name,
            'positions': [list(p) for p in self.positions],
            'weights': self.weights,
            'n_steps': self.n_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetTrajectory:
        """Create from dictionary (e.g., from JSON/API request)."""
        return cls(joint_name=data['joint_name'], positions=data['positions'], weights=data.get('weights'))


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.

    Attributes:
        success: Whether optimization completed successfully
        optimized_dimensions: Final dimension values {name: value}
        optimized_mechanism: Mechanism object with optimized dimensions (None if failed)
        initial_error: Error before optimization
        final_error: Error after optimization
        best_error: Best error achieved during optimization (may differ from final_error)
        improvement_pct: Percentage improvement from initial to final error
        iterations: Number of iterations/evaluations performed
        convergence_history: Optional list of error values over iterations
        error: Error message if success=False
    """
    success: bool
    optimized_dimensions: dict[str, float]
    optimized_mechanism: Mechanism | None = None
    initial_error: float = 0.0
    final_error: float = 0.0
    best_error: float = 0.0
    improvement_pct: float = 0.0
    iterations: int = 0
    convergence_history: list[float] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """
        Serialize to dictionary for JSON/API responses.

        Converts Mechanism to dict only for JSON serialization (API boundary).
        """
        return {
            'success': self.success,
            'optimized_dimensions': self.optimized_dimensions,
            'optimized_pylink_data': self.optimized_mechanism.to_dict() if self.optimized_mechanism else None,
            'initial_error': self.initial_error,
            'final_error': self.final_error,
            'best_error': self.best_error,
            'improvement_pct': self.improvement_pct,
            'iterations': self.iterations,
            'convergence_history': self.convergence_history,
            'error': self.error,
        }


# =============================================================================
# MULTI-SOLUTION TYPES
# =============================================================================


@dataclass
class Solution(OptimizationResult):
    """
    A single solution found during multi-solution optimization.

    Inherits from OptimizationResult and adds multi-solution specific fields.
    Solutions represent found optima and should typically have success=True.

    Additional Attributes:
        cluster_id: Which cluster this solution belongs to (for grouping similar solutions)
        distance_to_best: L2 distance in dimension space to the global best solution
        uniqueness_score: Measure of how different this is from other solutions (0-1)
            1.0 = completely unique, 0.0 = identical to another solution
        local_search_start: Starting point for local optimization (optional)
    """
    cluster_id: int = 0
    distance_to_best: float = 0.0
    uniqueness_score: float = 1.0
    local_search_start: np.ndarray | None = None


@dataclass
class MultiSolutionResult:
    """
    Results from multi-solution optimization containing multiple distinct optima.

    The solutions list is sorted by final_error (best first). Solutions within
    epsilon_threshold of the best are considered "near-optimal". The clustering
    identifies groups of similar solutions to avoid redundant duplicates.

    Attributes:
        solutions: All distinct solutions found, sorted by quality (best first)
        best_solution: The single best solution (convenience accessor)
        n_unique_clusters: Number of distinct solution clusters found
        epsilon_threshold: Error threshold used for "near-optimal" classification
        search_space_coverage: Fraction of viable search space explored (0-1)
        total_evaluations: Total number of objective function calls
        success: Whether optimization completed successfully
        method: Which algorithm was used ('basin_hopping', 'pso_niching', 'multi_start')
        method_config: Configuration parameters used
        error_message: Error description if success=False
    """
    solutions: list[Solution]
    best_solution: Solution | None
    n_unique_clusters: int
    epsilon_threshold: float
    search_space_coverage: float
    total_evaluations: int
    success: bool
    method: Literal['basin_hopping', 'pso_niching', 'multi_start']
    method_config: dict = field(default_factory=dict)
    error_message: str | None = None

    def get_near_optimal(self, epsilon: float | None = None) -> list[Solution]:
        """
        Get all solutions within epsilon of the best solution.

        Args:
            epsilon: Error threshold (uses self.epsilon_threshold if None)

        Returns:
            Subset of solutions with final_error <= best_error + epsilon
        """
        if not self.best_solution:
            return []
        thresh = epsilon or self.epsilon_threshold
        return [s for s in self.solutions if s.final_error <= self.best_solution.final_error + thresh]

    def get_cluster(self, cluster_id: int) -> list[Solution]:
        """Get all solutions belonging to a specific cluster."""
        return [s for s in self.solutions if s.cluster_id == cluster_id]

    def get_cluster_bests(self) -> list[Solution]:
        """Get one representative solution from each cluster (the best one)."""
        return [
            min(self.get_cluster(i), key=lambda s: s.final_error)
            for i in range(self.n_unique_clusters)
            if self.get_cluster(i)
        ]

    # Properties to provide OptimizationResult-like interface
    @property
    def optimized_dimensions(self) -> dict[str, float]:
        """Return best solution's optimized dimensions."""
        return self.best_solution.optimized_dimensions if self.best_solution else {}

    @property
    def optimized_pylink_data(self) -> dict | None:
        """
        Return best solution's optimized pylink data (for API compatibility).

        Converts Mechanism to dict only for JSON serialization (API boundary).
        """
        if not self.best_solution or not self.best_solution.optimized_mechanism:
            return None
        return self.best_solution.optimized_mechanism.to_dict()

    @property
    def initial_error(self) -> float:
        """Return best solution's initial error."""
        return self.best_solution.initial_error if self.best_solution else float('inf')

    @property
    def final_error(self) -> float:
        """Return best solution's final error."""
        return self.best_solution.final_error if self.best_solution else float('inf')

    @property
    def iterations(self) -> int:
        """Return total evaluations as iterations."""
        return self.total_evaluations

    @property
    def error(self) -> str | None:
        """Return error message (alias for error_message)."""
        return self.error_message
