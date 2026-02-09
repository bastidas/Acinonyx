"""
pylink_tools - Linkage simulation and optimization utilities.

Key components:
  - Mechanism: Fast, mutable linkage wrapper for simulation (recommended)
  - score_trajectory: Unified trajectory scoring (MSE or distance metrics)

Example usage:
    from pylink_tools import Mechanism, score_trajectory

    # Fast repeated simulation (optimization hot loop)
    for dims in candidate_dimensions:
        mechanism.set_dimensions(dims)
        trajectory = mechanism.simulate()
        score = score_trajectory(target, trajectory[:, joint_idx, :], metric='mse')

    # Serialize result
    optimized_data = mechanism.to_dict()
"""
from __future__ import annotations

from pylink_tools.mechanism import create_mechanism_fitness
from pylink_tools.mechanism import Mechanism
from pylink_tools.mechanism import ReflectableLink
from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.optimize import optimize_trajectory
from pylink_tools.trajectory_scoring import score_trajectory

__all__ = [
    # Core mechanism API
    'Mechanism',
    'ReflectableLink',
    'create_mechanism_fitness',
    # Optimization entry point
    'optimize_trajectory',
    # Data types
    'DimensionBoundsSpec',
    'TargetTrajectory',
    # Trajectory scoring
    'score_trajectory',
    # Note: For advanced trajectory distance metrics, import directly from
    # pylink_tools.trajectory_metrics (e.g., distance_sspd, distance_dtw, etc.)
]
