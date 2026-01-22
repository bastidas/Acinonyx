"""
optimization_types.py - Data structures for trajectory optimization.

Contains dataclasses used throughout the optimization module:
  - DimensionSpec: Describes optimizable dimensions of a linkage
  - TargetTrajectory: Target positions for optimization
  - OptimizationResult: Result of an optimization run
  - ConvergenceStats: Statistics about optimization convergence
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DimensionSpec:
    """
    Describes optimizable dimensions of a linkage.

    Each dimension corresponds to a link length that can be adjusted
    during optimization.

    Attributes:
        names: Human-readable names for each dimension (e.g., "B_distance", "C_distance0")
        initial_values: Current values of each dimension
        bounds: (min, max) tuple for each dimension
        joint_mapping: Maps dimension name -> (joint_name, property_name)
                       e.g., {"B_distance": ("B", "distance")}
                       Used for legacy pylinkage.joints format.
        edge_mapping: Maps dimension name -> (edge_name, property_name)
                      e.g., {"link1_distance": ("link1", "distance")}
                      Used for hypergraph linkage.edges format.
    """
    names: list[str]
    initial_values: list[float]
    bounds: list[tuple[float, float]]
    joint_mapping: dict[str, tuple[str, str]]
    edge_mapping: dict[str, tuple[str, str]] | None = None

    def __len__(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict:
        result = {
            'names': self.names,
            'initial_values': self.initial_values,
            'bounds': self.bounds,
            'joint_mapping': {k: list(v) for k, v in self.joint_mapping.items()},
            'n_dimensions': len(self.names),
        }
        if self.edge_mapping:
            result['edge_mapping'] = {k: list(v) for k, v in self.edge_mapping.items()}
        return result

    def get_bounds_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return bounds in pylinkage format: ((lower...), (upper...))"""
        lower = tuple(b[0] for b in self.bounds)
        upper = tuple(b[1] for b in self.bounds)
        return (lower, upper)


@dataclass
class TargetTrajectory:
    """
    Target positions for optimization.

    Specifies where we want a particular joint to be at each timestep.

    Attributes:
        joint_name: Which joint to optimize (match its path to target)
        positions: List of (x, y) positions, one per timestep
        weights: Optional per-timestep weights (higher = more important)
                 Defaults to uniform weights if not provided.
    """
    joint_name: str
    positions: list[tuple[float, float]]
    weights: list[float] | None = None

    def __post_init__(self):
        # Convert positions to tuples if they're lists
        self.positions = [tuple(p) for p in self.positions]

        # Set uniform weights if not provided
        if self.weights is None:
            self.weights = [1.0] * len(self.positions)

    @property
    def n_steps(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict:
        return {
            'joint_name': self.joint_name,
            'positions': [list(p) for p in self.positions],
            'weights': self.weights,
            'n_steps': self.n_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetTrajectory:
        return cls(
            joint_name=data['joint_name'],
            positions=data['positions'],
            weights=data.get('weights'),
        )


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.

    Attributes:
        success: Whether optimization completed successfully
        optimized_dimensions: Final dimension values {name: value}
        optimized_pylink_data: Updated pylink_data with optimized dimensions
        initial_error: Error before optimization
        final_error: Error after optimization
        iterations: Number of iterations/evaluations performed
        convergence_history: Optional list of error values over iterations
        error: Error message if success=False
    """
    success: bool
    optimized_dimensions: dict[str, float]
    optimized_pylink_data: dict | None = None
    initial_error: float = 0.0
    final_error: float = 0.0
    iterations: int = 0
    convergence_history: list[float] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'optimized_dimensions': self.optimized_dimensions,
            'optimized_pylink_data': self.optimized_pylink_data,
            'initial_error': self.initial_error,
            'final_error': self.final_error,
            'iterations': self.iterations,
            'convergence_history': self.convergence_history,
            'error': self.error,
        }


@dataclass
class ConvergenceStats:
    """
    Statistics about optimization convergence.

    Useful for analyzing and debugging optimization runs.
    """
    initial_error: float
    final_error: float
    best_error: float
    improvement_pct: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    history: list[float]
    improvement_per_iteration: list[float]

    def to_dict(self) -> dict:
        return {
            'initial_error': self.initial_error,
            'final_error': self.final_error,
            'best_error': self.best_error,
            'improvement_pct': self.improvement_pct,
            'n_iterations': self.n_iterations,
            'n_evaluations': self.n_evaluations,
            'converged': self.converged,
            'history': self.history,
            'improvement_per_iteration': self.improvement_per_iteration,
        }
