"""
mechanism.py - Fast, mutable linkage wrapper for simulation and optimization.

This module provides the Mechanism class which wraps pylinkage's Linkage object
for efficient trajectory computation.
  - Compiled once, reused for all simulations (no per-call overhead)
  - holds additonal metadata like link color, z-level, weight, etc.
  - Work with Mechanism internally for all simulation/optimization
  - Serialize back to dict only when needed (API responses, saving)

Usage:

    # Fast simulation (repeated)
    mechanism.set_dimensions(new_dims)
    trajectory = mechanism.simulate()

    # Get result
    optimized_data = mechanism.to_dict()
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from typing import Literal
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
from pylinkage.bridge.solver_conversion import update_solver_constraints
from pylinkage.bridge.solver_conversion import update_solver_positions
from pylinkage.hypergraph import graph_from_dict
from pylinkage.hypergraph import graph_to_dict
from pylinkage.joints import Crank
from pylinkage.joints import Revolute
from pylinkage.joints import Static
from pylinkage.linkage.linkage import Linkage
from pylinkage.linkage.serialization import linkage_to_dict

from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import DimensionMapping
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.trajectory_scoring import score_trajectory

if TYPE_CHECKING:
    from target_gen.variation_config import MechVariationConfig

logger = logging.getLogger(__name__)

# Role mapping: our format <-> pylinkage format (GROUND / DRIVER / DRIVEN)
_ROLE_TO_PYLINKAGE = {
    'fixed': 'GROUND',
    'ground': 'GROUND',
    'crank': 'DRIVER',
    'driver': 'DRIVER',
    'follower': 'DRIVEN',
    'driven': 'DRIVEN',
}
_ROLE_FROM_PYLINKAGE = {v: k for k, v in _ROLE_TO_PYLINKAGE.items()}


# Minimum axis length to consider a reflection axis non-degenerate
_REFLECT_AXIS_MIN_LENGTH = 1e-6


def _side_of_line(
    line_p1: tuple[float, float],
    line_p2: tuple[float, float],
    point: tuple[float, float],
) -> float:
    """
    Return signed value: >0 one side, <0 other side, 0 on line.
    Uses cross product (line_p2 - line_p1) x (point - line_p1).
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    px = point[0] - line_p1[0]
    py = point[1] - line_p1[1]
    return dx * py - dy * px


def _connected_component_containing(
    neighbors: dict[str, set[str]],
    start: str,
    exclude_edge: tuple[str, str] | None = None,
) -> set[str]:
    """BFS from start; optionally do not traverse exclude_edge (a, b) (order ignored)."""
    a_ex, b_ex = (exclude_edge[0], exclude_edge[1]) if exclude_edge else (None, None)
    component: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        for nbr in neighbors.get(node, set()):
            if nbr in component:
                continue
            if exclude_edge and (node == a_ex and nbr == b_ex or node == b_ex and nbr == a_ex):
                continue
            stack.append(nbr)
    return component


# Tolerance for edge length preservation after reflection (same link lengths).
# Kept loose so solver/float errors and approximate positions don't reject valid axes.
_REFLECT_LENGTH_TOLERANCE = 0.5


def _reflect_point_over_line(
    point: tuple[float, float],
    line_p1: tuple[float, float],
    line_p2: tuple[float, float],
) -> tuple[float, float]:
    """
    Reflect a point over a line defined by two points.

    Args:
        point: (x, y) point to reflect
        line_p1: First point on the line
        line_p2: Second point on the line

    Returns:
        Reflected (x, y) point
    """
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    dx = x2 - x1
    dy = y2 - y1
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-10:
        return point
    vx = px - x1
    vy = py - y1
    t = (vx * dx + vy * dy) / line_len_sq
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (2 * proj_x - px, 2 * proj_y - py)


@dataclass(frozen=True)
class ReflectableLink:
    """
    A viable reflection axis: only driven joint(s) are reflected; crank and ground stay fixed.

    Used so the crank initial angle is unchanged after reflection (over-center / branch flip).
    """

    axis_node_1: str
    axis_node_2: str
    line_p1: tuple[float, float]
    line_p2: tuple[float, float]
    joints_to_reflect: tuple[str, ...]
    joints_fixed: tuple[str, ...]


def _linkage_document_from_pylinkage_dict(
    pylinkage_dict: dict,
    original_nodes: dict,
    metadata: dict | None = None,
) -> dict:
    """
    Convert a validated pylinkage hypergraph dict to frontend LinkageDocument format.

    Args:
        pylinkage_dict: Dict with 'nodes' (list), 'edges' (list), 'name' (str)
                       from pylinkage (e.g. output of graph_to_dict).
        original_nodes: Map node_id -> original node data for role/angle preservation.
        metadata: Optional metadata; optional keys (drawnObjects, components, etc.) are copied to result.

    Returns:
        LinkageDocument dict: version, name, linkage.{name, nodes, edges, hyperedges}, meta.
    """
    validated_nodes = pylinkage_dict.get('nodes', [])
    validated_edges = pylinkage_dict.get('edges', [])
    linkage_name = pylinkage_dict.get('name', 'optimized')

    nodes = {}
    for node_data in validated_nodes:
        node_id = node_data['id']
        original_node = original_nodes.get(node_id, {})
        original_role = original_node.get('role')

        if original_role:
            our_role = original_role
        else:
            role = node_data.get('role', 'DRIVEN')
            our_role = _ROLE_FROM_PYLINKAGE.get(role, 'follower')

        angle = original_node.get('angle') or node_data.get('angle')

        nodes[node_id] = {
            'id': node_id,
            'position': node_data.get('position', [None, None]),
            'role': our_role,
            'jointType': node_data.get('joint_type', 'REVOLUTE').lower(),
            'name': node_data.get('name', node_id),
        }
        if our_role == 'crank' and angle is not None:
            nodes[node_id]['angle'] = float(angle)
        elif 'angle' in node_data:
            nodes[node_id]['angle'] = node_data['angle']

    edges = {}
    for edge_data in validated_edges:
        edge_id = edge_data['id']
        edges[edge_id] = {
            'id': edge_id,
            'source': edge_data['source'],
            'target': edge_data['target'],
            'distance': edge_data.get('distance'),
        }

    result = {
        'version': '2.0.0',
        'name': linkage_name,
        'linkage': {
            'name': linkage_name,
            'nodes': nodes,
            'edges': edges,
            'hyperedges': {},
        },
        'meta': {'nodes': {}, 'edges': {}},
    }

    if metadata:
        metadata_copy = copy.deepcopy(metadata)
        for key in ['drawnObjects', 'components', 'hierarchy', 'savedAt']:
            if key in metadata_copy:
                result[key] = metadata_copy[key]

    return result


class Mechanism:
    """
    Wrapper around pylinkage Linkage for fast, mutable simulation.

    Linkage:
    https://github.com/HugoFara/pylinkage/blob/main/src/pylinkage/linkage/linkage.py
    linkage.get_coords()
    linkage.set_coords()
    linkage.set_completely() - set both dimension and initial positions.

    Key benefits:
    - Compiled once, reused for all simulations
    - Direct attribute mutation (no dict conversion)
    - step_fast() with numba JIT (4-7x faster)
    - No skip_sync confusion - dimensions are THE source of truth
    """

    def __init__(
        self,
        linkage: Linkage,
        dimension_mapping: DimensionMapping,
        joint_names: list[str],
        n_steps: int = 32,
        metadata: dict | None = None,
    ):
        """
        Initialize Mechanism

        Args:
            linkage: Compiled pylinkage Linkage object
            dimension_mapping: Mapping of dimension names to joint attributes
            joint_names: Ordered list of joint names
            n_steps: Number of simulation steps
            metadata: Additional metadata to store in the mechanism
        """
        self._linkage = linkage
        self._dimension_mapping = dimension_mapping
        self._joint_names = joint_names
        self._n_steps = n_steps
        self._metadata = metadata

        # Cache initial positions for reset
        self._initial_positions = linkage.get_coords()

        # Track current dimension values
        self._current_dimensions = np.array(dimension_mapping.initial_values)

    def set_dimensions(
        self,
        dimensions: dict[str, float] | tuple[float, ...] | np.ndarray,
        validate_bounds: bool = True,
    ) -> None:
        """
        Update link dimensions in-place (fast, no copy).

        Directly mutates joint attributes and syncs to solver data.

        Args:
            dimensions: Either:
                - dict mapping dimension name to value
                - tuple/array of values in dimension order
            validate_bounds: If True, raise ValueError if dimensions are out of bounds.
                            Always logs warning if out of bounds regardless of this flag.
        """

        # Convert to array if dict
        if isinstance(dimensions, dict):
            values = np.array([
                dimensions.get(name, self._current_dimensions[i])
                for i, name in enumerate(self._dimension_mapping.names)
            ])
        else:
            values = np.asarray(dimensions)

        # Always warn about bounds violations, only fail if validate_bounds=True
        # Skip bounds check when no bounds are set (e.g. mechanism created without variation config)
        bounds_list = self._dimension_mapping.bounds
        for i, value in enumerate(values):
            name = self._dimension_mapping.names[i]
            if i >= len(bounds_list):
                continue
            min_bound, max_bound = bounds_list[i]
            if value < min_bound or value > max_bound:
                logger.warning(
                    f"Dimension '{name}' value {value} is out of bounds "
                    f'[{min_bound}, {max_bound}]',
                )
                if validate_bounds:
                    raise ValueError(
                        f"Dimension '{name}' value {value} is out of bounds "
                        f'[{min_bound}, {max_bound}]',
                    )

        # Update joint attributes directly (doesn't invalidate _solver_data)
        for i, ja in enumerate(self._dimension_mapping.joint_attrs):
            if ja is not None:
                joint_obj, attr_name = ja
                setattr(joint_obj, attr_name, values[i])

        # Sync constraints to solver data (fast numpy update)
        update_solver_constraints(self._linkage._solver_data, self._linkage)

        # Reset positions to initial state
        self._linkage.set_coords(self._initial_positions)
        update_solver_positions(self._linkage._solver_data, self._linkage)

        # Track current values
        self._current_dimensions = values.copy()

    def simulate(self) -> np.ndarray:
        """
        Run simulation using step_fast().

        Returns:
            Trajectory array of shape (n_steps, n_joints, 2)
        """
        trajectory = self._linkage.step_fast(iterations=self._n_steps)
        return trajectory

    def simulate_dict(self) -> dict[str, list[tuple[float, float]]]:
        """
        Run simulation and return trajectories as dict.

        Returns:
            Dict mapping joint_name -> list of (x, y) tuples
        """
        trajectory = self.simulate()

        return {
            name: [(trajectory[step, i, 0], trajectory[step, i, 1])
                   for step in range(self._n_steps)]
            for i, name in enumerate(self._joint_names)
        }

    def get_trajectory(self, joint_name: str) -> np.ndarray:
        """
        Get trajectory for a specific joint.

        Args:
            joint_name: Name of joint to get trajectory for

        Returns:
            Array of shape (n_steps, 2) with (x, y) positions
        """
        trajectory = self.simulate()
        idx = self._joint_names.index(joint_name)
        return trajectory[:, idx, :]

    def get_joint_index(self, joint_name: str) -> int:
        """Get the index of a joint by name."""
        return self._joint_names.index(joint_name)

    def _get_joint_position(self, joint_name: str) -> tuple[float, float]:
        """Get current position of a joint by name (from _initial_positions)."""
        idx = self._joint_names.index(joint_name)
        pos = self._initial_positions[idx]
        return (float(pos[0]), float(pos[1]))

    def get_reflectable_edges(
        self,
        length_tolerance: float = 0.1,
        angle_tolerance: float = 0.05,
        allow_move_fixed: bool = False,
        debug: bool = False,
    ) -> list[ReflectableLink]:
        """
        Return reflectable edges found in a graph- and geometry-aware way.

        A reflectable edge is an imaginary line between two nodes such that
        reflecting all joints on one side of the line over the line leaves all
        link lengths unchanged (within length_tolerance), all joint angles
        unchanged except at the axis (within angle_tolerance).

        Strategy:
        1) For every edge/link, use that link's line as the axis. Partition
           nodes by geometry (which side of the line); reflect the side opposite
           the crank (so crank stays fixed). Check lengths and angles.
        2) Additionally, try imaginary lines from crank to each other node;
           same geometric partition and checks.

        Args:
            length_tolerance: Max allowed change in any link length after
                reflection (default 0.5).
            angle_tolerance: Max allowed change in joint angle (radians) at
                non-axis joints (default ~3 deg).
            allow_move_fixed: If False (default), skip axes that would move
                any fixed/ground node.
            debug: If True, log why each candidate axis is rejected.

        Returns:
            List of ReflectableLink (axis line + joints_to_reflect / joints_fixed).
        """
        joint_name_set = set(self._joint_names)
        if self._metadata and '_original_edges' in self._metadata:
            edges_dict = self._metadata['_original_edges']
        else:
            doc = self.to_dict()
            edges_dict = doc.get('linkage', {}).get('edges', {})

        # Build graph: neighbors per node (for angle check)
        neighbors: dict[str, set[str]] = {n: set() for n in joint_name_set}
        for ed in edges_dict.values():
            s, t = ed.get('source'), ed.get('target')
            if s in joint_name_set and t in joint_name_set:
                neighbors[s].add(t)
                neighbors[t].add(s)

        # Crank and fixed (static) names from linkage
        linkage = self._linkage
        crank_names: set[str] = set()
        fixed_names: set[str] = set()
        for j in linkage.joints:
            if isinstance(j, Crank):
                crank_names.add(j.name)
            elif isinstance(j, Static):
                fixed_names.add(j.name)
        if not crank_names:
            crank_names = {self._joint_names[0]} if self._joint_names else set()

        def _pos(name: str) -> tuple[float, float]:
            return self._get_joint_position(name)

        def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
            return float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2))

        def _angle_at_joint(
            j: str,
            pos_map: dict[str, tuple[float, float]],
        ) -> float | None:
            """Angle (radians) at joint j between first two neighbors, or None if deg < 2."""
            nbrs = sorted(neighbors.get(j, set()))
            if len(nbrs) < 2:
                return None
            pj = pos_map[j]
            pa = pos_map[nbrs[0]]
            pb = pos_map[nbrs[1]]
            vax = pa[0] - pj[0]
            vay = pa[1] - pj[1]
            vbx = pb[0] - pj[0]
            vby = pb[1] - pj[1]
            a = np.arctan2(vay, vax)
            b = np.arctan2(vby, vbx)
            diff = abs(a - b)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            return float(diff)

        def lengths_ok_after_reflect(
            line_p1: tuple[float, float],
            line_p2: tuple[float, float],
            reflect_set: set[str],
        ) -> bool:
            new_pos = {}
            for name in self._joint_names:
                p = _pos(name)
                new_pos[name] = (
                    _reflect_point_over_line(p, line_p1, line_p2)
                    if name in reflect_set
                    else p
                )
            for ed in edges_dict.values():
                s, t = ed.get('source'), ed.get('target')
                if s not in joint_name_set or t not in joint_name_set:
                    continue
                want = float(ed.get('distance', 0))
                got = _dist(new_pos[s], new_pos[t])
                if abs(got - want) > length_tolerance:
                    return False
            return True

        def angles_ok_after_reflect(
            line_p1: tuple[float, float],
            line_p2: tuple[float, float],
            reflect_set: set[str],
            axis_nodes: set[str],
        ) -> bool:
            """Non-axis joints must keep the same angle (within angle_tolerance)."""
            orig_pos = {n: _pos(n) for n in self._joint_names}
            new_pos = {}
            for name in self._joint_names:
                p = orig_pos[name]
                new_pos[name] = (
                    _reflect_point_over_line(p, line_p1, line_p2)
                    if name in reflect_set
                    else p
                )
            for j in self._joint_names:
                if j in axis_nodes:
                    continue
                before = _angle_at_joint(j, orig_pos)
                after = _angle_at_joint(j, new_pos)
                if before is None or after is None:
                    continue
                err = abs(after - before)
                if err > np.pi:
                    err = 2 * np.pi - err
                if err > angle_tolerance:
                    return False
            return True

        def axis_key(a: str, b: str) -> tuple[str, str]:
            return (a, b) if a <= b else (b, a)

        def to_reflect_for_line(
            line_p1: tuple[float, float],
            line_p2: tuple[float, float],
            ref_side_val: float,
        ) -> set[str]:
            """Nodes on the opposite side of the line from ref_side_val."""
            out = set()
            for node in self._joint_names:
                side = _side_of_line(line_p1, line_p2, _pos(node))
                if (ref_side_val >= 0 and side < 0) or (ref_side_val <= 0 and side > 0):
                    out.add(node)
            return out

        # Target one axis for detailed debug (crank–coupler_rocker_joint)
        _debug_axis = axis_key('crank', 'coupler_rocker_joint')

        def add_candidate(
            a: str,
            b: str,
            to_reflect: set[str],
            seen: set[tuple[str, str]],
            out: list[ReflectableLink],
            source: str,
        ) -> None:
            key = axis_key(a, b)
            is_debug_axis = key == _debug_axis
            if (debug or is_debug_axis) and key not in seen:
                logger.debug(
                    'get_reflectable_edges candidate axis %s–%s (source=%s) to_reflect=%s',
                    a, b, source, sorted(to_reflect),
                )
            if not to_reflect:
                if debug or is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: empty to_reflect', a, b)
                return
            if not allow_move_fixed and (to_reflect & fixed_names):
                if debug or is_debug_axis:
                    logger.debug(
                        'get_reflectable_edges skip %s–%s: would move fixed %s',
                        a, b, sorted(to_reflect & fixed_names),
                    )
                return
            if key in seen:
                if is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: already seen', a, b)
                return
            try:
                line_p1 = _pos(a)
                line_p2 = _pos(b)
            except (ValueError, KeyError):
                if debug or is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: no position', a, b)
                return
            dx = line_p2[0] - line_p1[0]
            dy = line_p2[1] - line_p1[1]
            if dx * dx + dy * dy < _REFLECT_AXIS_MIN_LENGTH * _REFLECT_AXIS_MIN_LENGTH:
                if debug or is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: degenerate axis', a, b)
                return
            if not lengths_ok_after_reflect(line_p1, line_p2, to_reflect):
                if debug or is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: length check failed', a, b)
                return
            axis_nodes = {a, b}
            if not angles_ok_after_reflect(line_p1, line_p2, to_reflect, axis_nodes):
                if debug or is_debug_axis:
                    logger.debug('get_reflectable_edges skip %s–%s: angle check failed', a, b)
                return
            seen.add(key)
            fixed_set = joint_name_set - to_reflect
            # line_p1/line_p2 stored for reference; reflect_over_axis uses current
            # axis positions so lengths stay within tolerance.
            out.append(
                ReflectableLink(
                    axis_node_1=a,
                    axis_node_2=b,
                    line_p1=line_p1,
                    line_p2=line_p2,
                    joints_to_reflect=tuple(sorted(to_reflect)),
                    joints_fixed=tuple(sorted(fixed_set)),
                ),
            )

        result: list[ReflectableLink] = []
        seen_axes: set[tuple[str, str]] = set()
        crank = next(iter(crank_names)) if crank_names else None
        crank_pos = _pos(crank) if crank else (0.0, 0.0)
        ref_name = next((n for n in fixed_names if n != crank), None) or next(
            (n for n in self._joint_names if n != crank), None,
        )

        # 1) Every edge/link as axis: partition by geometry (side of line), not graph
        for ed in edges_dict.values():
            a = ed.get('source')
            b = ed.get('target')
            if not a or not b or a not in joint_name_set or b not in joint_name_set:
                continue
            line_p1 = _pos(a)
            line_p2 = _pos(b)
            ref_side_val = _side_of_line(
                line_p1, line_p2,
                crank_pos if crank else (_pos(ref_name) if ref_name is not None else (0.0, 0.0)),
            )
            to_reflect = to_reflect_for_line(line_p1, line_p2, ref_side_val)
            if not allow_move_fixed:
                to_reflect = to_reflect - fixed_names
            add_candidate(a, b, to_reflect, seen_axes, result, source='edge')

        # 2) Crank-only rays: axis = (crank, X) for each other node X
        if crank and ref_name:
            for other in self._joint_names:
                if other == crank:
                    continue
                line_p1 = crank_pos
                line_p2 = _pos(other)
                dx = line_p2[0] - line_p1[0]
                dy = line_p2[1] - line_p1[1]
                if dx * dx + dy * dy < _REFLECT_AXIS_MIN_LENGTH * _REFLECT_AXIS_MIN_LENGTH:
                    continue
                key = axis_key(crank, other)
                if key in seen_axes:
                    continue
                ref_pos = _pos(ref_name)
                ref_side_val = _side_of_line(line_p1, line_p2, ref_pos)
                to_reflect = to_reflect_for_line(line_p1, line_p2, ref_side_val)
                if not allow_move_fixed:
                    to_reflect = to_reflect - fixed_names
                add_candidate(crank, other, to_reflect, seen_axes, result, source='crank_ray')

        if debug:
            logger.debug(
                'get_reflectable_edges found %d axes (length_tol=%.4f angle_tol=%.4f)',
                len(result), length_tolerance, angle_tolerance,
            )
        return result

    def reflect_over_axis(self, reflectable: ReflectableLink) -> Mechanism:
        """
        Return a new mechanism with only the driven joint(s) reflected over the axis.

        Crank and ground joints are unchanged, so the crank initial angle is preserved.
        The reflection line is taken from the **current** positions of the axis nodes
        (reflectable.axis_node_1, axis_node_2) on this mechanism so that lengths are
        preserved even if the mechanism was modified after get_reflectable_edges().

        Args:
            reflectable: Viable reflectable axis from get_reflectable_edges().

        Returns:
            New Mechanism instance (copy) with reflected initial configuration.
        """
        copy_mech = self.copy()
        positions = copy_mech._initial_positions
        n = len(copy_mech._joint_names)
        coords_list = [
            (float(positions[i][0]), float(positions[i][1]))
            for i in range(min(n, len(positions)))
        ]
        while len(coords_list) < n:
            coords_list.append((0.0, 0.0))

        # Use current positions of axis endpoints so the line is correct for this
        # mechanism state (avoids length drift if mechanism changed since reflectable
        # was computed).
        idx1 = copy_mech._joint_names.index(reflectable.axis_node_1)
        idx2 = copy_mech._joint_names.index(reflectable.axis_node_2)
        line_p1 = (float(positions[idx1][0]), float(positions[idx1][1]))
        line_p2 = (float(positions[idx2][0]), float(positions[idx2][1]))

        reflect_set = set(reflectable.joints_to_reflect)
        for i, joint_name in enumerate(copy_mech._joint_names):
            if joint_name in reflect_set:
                coords_list[i] = _reflect_point_over_line(
                    coords_list[i], line_p1, line_p2,
                )

        copy_mech._linkage.set_coords(coords_list)
        update_solver_positions(copy_mech._linkage._solver_data, copy_mech._linkage)
        copy_mech._initial_positions = copy_mech._linkage.get_coords()
        return copy_mech

    def _to_pylinkage_dict(self) -> dict:
        """
        Serialize current mechanism state to validated pylinkage hypergraph dict.

        Returns:
            Dict with 'nodes' (list), 'edges' (list), 'name' (str) in pylinkage format.
        """
        linkage_dict = linkage_to_dict(self._linkage)
        coords = self._linkage.get_coords()

        pylinkage_nodes = []
        original_nodes = {}
        if self._metadata and '_original_nodes' in self._metadata:
            original_nodes = copy.deepcopy(self._metadata['_original_nodes'])

        for i, joint_data in enumerate(linkage_dict.get('joints', [])):
            joint_name = joint_data['name']
            joint_type = joint_data.get('type', 'Revolute')

            if i < len(coords):
                x, y = coords[i]
            else:
                x = joint_data.get('x', 0.0)
                y = joint_data.get('y', 0.0)

            original_node = original_nodes.get(joint_name, {})
            original_role = original_node.get('role')

            if original_role:
                if original_role == 'fixed':
                    role = 'fixed'
                elif original_role == 'crank':
                    role = 'crank'
                elif original_role == 'follower':
                    role = 'follower'
                else:
                    role = 'fixed' if joint_type == 'Static' else ('crank' if joint_type == 'Crank' else 'follower')
            else:
                role = 'fixed' if joint_type == 'Static' else ('crank' if joint_type == 'Crank' else 'follower')

            pylinkage_role = _ROLE_TO_PYLINKAGE.get(role, 'DRIVEN')
            node_dict = {
                'id': joint_name,
                'position': [float(x), float(y)],
                'role': pylinkage_role,
                'joint_type': 'REVOLUTE',
                'name': joint_name,
            }
            if role == 'crank' or joint_type == 'Crank':
                angle = original_node.get('angle') or joint_data.get('angle', 0.0)
                node_dict['angle'] = float(angle)
                node_dict['initial_angle'] = float(angle)

            pylinkage_nodes.append(node_dict)

        original_edges = {}
        if self._metadata and '_original_edges' in self._metadata:
            original_edges = copy.deepcopy(self._metadata['_original_edges'])

        optimized_distances = {}
        for dim_idx, dim_name in enumerate(self._dimension_mapping.names):
            if dim_name not in self._dimension_mapping.edge_mapping:
                continue
            edge_id, _ = self._dimension_mapping.edge_mapping[dim_name]
            if dim_idx >= len(self._current_dimensions):
                raise ValueError(
                    f'CRITICAL BUG in _to_pylinkage_dict(): dimension index {dim_idx} out of bounds '
                    f'(current_dimensions length: {len(self._current_dimensions)}, '
                    f'dimension_mapping.names length: {len(self._dimension_mapping.names)})',
                )
            optimized_distances[edge_id] = float(self._current_dimensions[dim_idx])

        edge_id_to_pylinkage = {}
        for edge_id, edge_data in original_edges.items():
            source = edge_data.get('source')
            target = edge_data.get('target')
            if not source or not target:
                continue
            original_distance = edge_data.get('distance', 1.0)
            optimized_distance = optimized_distances.get(edge_id)
            distance = optimized_distance if optimized_distance is not None else original_distance
            edge_id_to_pylinkage[edge_id] = {
                'id': edge_id,
                'source': source,
                'target': target,
                'distance': distance,
            }

        for dim_idx, dim_name in enumerate(self._dimension_mapping.names):
            if dim_name not in self._dimension_mapping.edge_mapping:
                continue
            edge_id, _ = self._dimension_mapping.edge_mapping[dim_name]
            distance = float(self._current_dimensions[dim_idx])
            if edge_id in edge_id_to_pylinkage:
                edge_id_to_pylinkage[edge_id]['distance'] = distance
                continue
            source = None
            target = None
            if dim_idx < len(self._dimension_mapping.joint_attrs):
                ja = self._dimension_mapping.joint_attrs[dim_idx]
                if ja is not None:
                    joint_obj, attr_name = ja
                    if isinstance(joint_obj, Crank):
                        if hasattr(joint_obj, 'joint0') and joint_obj.joint0:
                            source = joint_obj.joint0.name
                        if hasattr(joint_obj, 'name'):
                            target = joint_obj.name
                    elif isinstance(joint_obj, Revolute):
                        if attr_name == 'r0' and hasattr(joint_obj, 'joint0') and joint_obj.joint0:
                            source = joint_obj.joint0.name
                            if hasattr(joint_obj, 'name'):
                                target = joint_obj.name
                        elif attr_name == 'r1' and hasattr(joint_obj, 'joint1') and joint_obj.joint1:
                            source = joint_obj.joint1.name
                            if hasattr(joint_obj, 'name'):
                                target = joint_obj.name
                        else:
                            if hasattr(joint_obj, 'joint0') and joint_obj.joint0:
                                source = joint_obj.joint0.name
                            if hasattr(joint_obj, 'name'):
                                target = joint_obj.name
            if source and target:
                edge_id_to_pylinkage[edge_id] = {
                    'id': edge_id,
                    'source': source,
                    'target': target,
                    'distance': distance,
                }

        pylinkage_edges = list(edge_id_to_pylinkage.values())
        raw_pylinkage = {
            'name': linkage_dict.get('name', ''),
            'nodes': pylinkage_nodes,
            'edges': pylinkage_edges,
            'hyperedges': [],
        }
        hg = graph_from_dict(raw_pylinkage)
        validated_dict = graph_to_dict(hg)
        return {
            'nodes': validated_dict.get('nodes', []),
            'edges': validated_dict.get('edges', []),
            'name': linkage_dict.get('name', validated_dict.get('name', 'optimized')),
        }

    def _validate_dimensions_match_edges(self, edges: dict) -> None:
        """
        Raise ValueError if any dimension in _dimension_mapping does not match its edge distance.
        """
        if not self._dimension_mapping or not self._dimension_mapping.edge_mapping:
            return
        validation_errors = []
        tolerance = 1e-6
        for dim_idx, dim_name in enumerate(self._dimension_mapping.names):
            if dim_name not in self._dimension_mapping.edge_mapping:
                continue
            if dim_idx >= len(self._current_dimensions):
                validation_errors.append(
                    f"Dimension '{dim_name}' index {dim_idx} out of bounds "
                    f'(current_dimensions length: {len(self._current_dimensions)})',
                )
                continue
            edge_id, _ = self._dimension_mapping.edge_mapping[dim_name]
            expected_distance = float(self._current_dimensions[dim_idx])
            if edge_id not in edges:
                validation_errors.append(
                    f"Dimension '{dim_name}' maps to edge '{edge_id}' which is missing from output edges",
                )
                continue
            actual_distance = edges[edge_id].get('distance')
            if actual_distance is None:
                validation_errors.append(
                    f"Dimension '{dim_name}' maps to edge '{edge_id}' which has no distance",
                )
                continue
            if abs(expected_distance - float(actual_distance)) > tolerance:
                validation_errors.append(
                    f"Dimension '{dim_name}': _current_dimensions[{dim_idx}]={expected_distance}, "
                    f"but edge '{edge_id}' distance={actual_distance}, "
                    f'diff={abs(expected_distance - float(actual_distance))}',
                )
        if validation_errors:
            error_msg = (
                'CRITICAL BUG in Mechanism.to_dict(): optimized dimensions do not match edge distances!\n'
                'Errors:\n' + '\n'.join(f'  - {e}' for e in validation_errors) + '\n'
                f'_current_dimensions: {list(self._current_dimensions)}\n'
                f'dimension_mapping.names: {list(self._dimension_mapping.names)}\n'
                f'edge_mapping: {self._dimension_mapping.edge_mapping}\n'
                f"edge distances: {dict((eid, e.get('distance')) for eid, e in edges.items())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def to_dict(self) -> dict:
        """
        Serialize mechanism to frontend LinkageDocument format.

        Use this for API responses or saving to file.
        Returns a dict with updated edge distances from current dimensions.

        Returns:
            LinkageDocument dict: version, name, linkage.{nodes, edges}, meta, and optional metadata.
        """
        pylinkage_dict = self._to_pylinkage_dict()
        original_nodes = {}
        if self._metadata and '_original_nodes' in self._metadata:
            original_nodes = copy.deepcopy(self._metadata['_original_nodes'])
        doc = _linkage_document_from_pylinkage_dict(pylinkage_dict, original_nodes, self._metadata)
        self._validate_dimensions_match_edges(doc['linkage']['edges'])
        return doc

    def reset(self) -> None:
        """Reset mechanism to initial dimensions and positions."""
        self.set_dimensions(tuple(self._dimension_mapping.initial_values))

    def copy(self) -> Mechanism:
        """
        Create a copy of this mechanism.

        Returns a new Mechanism with the same dimensions and configuration.
        The underlying linkage is recreated, so modifications to one won't affect the other.

        Returns:
            New Mechanism instance with identical state.
        """
        import math

        from pylink_tools.hypergraph_adapter import to_simulatable_linkage

        # Serialize to hypergraph format
        pylink_data = self.to_dict()

        # Recreate linkage from hypergraph format
        new_linkage = to_simulatable_linkage(pylink_data)

        # Set crank angle (angular velocity per step)
        angle_per_step = 2 * math.pi / self._n_steps
        for joint in new_linkage.joints:
            if isinstance(joint, Crank):
                joint.angle = angle_per_step

        # Rebuild and compile
        new_linkage.rebuild()
        new_linkage.compile()

        # Copy dimension_mapping (rebuild joint_attrs from edges)
        new_dim_mapping = DimensionMapping(
            names=list(self._dimension_mapping.names),
            initial_values=list(self._current_dimensions),  # Use current, not initial
            bounds=list(self._dimension_mapping.bounds),
            edge_mapping=self._dimension_mapping.edge_mapping.copy(),
        )

        # Rebuild joint_attrs using helper method
        edges_dict = pylink_data.get('linkage', {}).get('edges', {})
        new_dim_mapping.joint_attrs = self._rebuild_joint_attrs(
            new_linkage,
            new_dim_mapping.edge_mapping,
            new_dim_mapping.names,
            edges_dict,
        )

        # Copy metadata and update _original_edges and _original_nodes to current state
        # This ensures that if the copied mechanism is optimized again,
        # to_dict() will include all edges and nodes with the current optimized state
        new_metadata = copy.deepcopy(self._metadata) if self._metadata else {}
        if not isinstance(new_metadata, dict):
            new_metadata = {}

        # Update _original_edges and _original_nodes to current state from to_dict() result
        # This preserves the current optimized state as the "original" for future operations
        current_edges = pylink_data.get('linkage', {}).get('edges', {})
        current_nodes = pylink_data.get('linkage', {}).get('nodes', {})
        new_metadata['_original_edges'] = copy.deepcopy(current_edges)
        new_metadata['_original_nodes'] = copy.deepcopy(current_nodes)

        # Create new mechanism
        new_mechanism = Mechanism(
            linkage=new_linkage,
            dimension_mapping=new_dim_mapping,
            joint_names=[j.name for j in new_linkage.joints],
            n_steps=self._n_steps,
            metadata=new_metadata,
        )

        # Set dimensions to sync solver data
        new_mechanism.set_dimensions(self._current_dimensions)

        return new_mechanism

    def with_dimensions(self, dimensions: dict[str, float] | tuple[float, ...] | np.ndarray) -> Mechanism:
        """
        Create a copy of this mechanism with modified dimensions.

        Args:
            dimensions: New dimension values (dict, tuple, or array)

        Returns:
            New Mechanism instance with modified dimensions.
        """
        new_mechanism = self.copy()
        new_mechanism.set_dimensions(dimensions)
        return new_mechanism

    @property
    def n_steps(self) -> int:
        """Number of simulation steps."""
        return self._n_steps

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of joint names."""
        return self._joint_names

    @property
    def dimension_names(self) -> list[str]:
        """Names of optimizable dimensions."""
        return self._dimension_mapping.names

    @property
    def dimension_bounds(self) -> list[tuple[float, float]]:
        """Bounds for each dimension."""
        return self._dimension_mapping.bounds

    @property
    def current_dimensions(self) -> np.ndarray:
        """Current dimension values."""
        return self._current_dimensions.copy()

    @property
    def n_dimensions(self) -> int:
        """Number of optimizable dimensions."""
        return len(self._dimension_mapping)

    def get_link_lengths(self) -> np.ndarray:
        """
        Get all link lengths as a numpy array.

        Returns:
            np.ndarray of shape (n_dimensions,) with link lengths in order
        """
        return self._current_dimensions.copy()

    def get_link_lengths_dict(self) -> dict[str, float]:
        """
        Get all link lengths as a dict with link names as keys.

        Returns:
            Dict mapping link name -> length (e.g., {'crank_link': 20.0, 'coupler': 50.0})
        """
        return {
            name: float(self._current_dimensions[i])
            for i, name in enumerate(self._dimension_mapping.names)
        }

    @property
    def linkage(self) -> Linkage:
        """Direct access to underlying Linkage (use with caution)."""
        return self._linkage

    def get_dimension_bounds_spec(self) -> DimensionBoundsSpec:
        """
        Get DimensionBoundsSpec from current mechanism state.

        Returns:
            DimensionBoundsSpec with current dimensions, bounds, and edge mapping.
        """
        return DimensionBoundsSpec(
            names=list(self._dimension_mapping.names),
            initial_values=list(self._current_dimensions),
            bounds=list(self._dimension_mapping.bounds),
            edge_mapping=self._dimension_mapping.edge_mapping.copy(),
        )

    def apply_variation_config(
        self,
        mech_variation_config: MechVariationConfig,
    ) -> None:
        """
        Apply MechVariationConfig to set dimension bounds post-initialization.

        This allows creating a mechanism with default bounds, then applying
        custom bounds via variation config. Useful for optimization scenarios
        where you want different bounds than the default.

        Args:
            mech_variation_config: Variation config to apply

        Note:
            - Converts relative percentage bounds to absolute bounds
            - Updates self._dimension_mapping.bounds
            - Filters out excluded dimensions from mapping
            - Static joint movement and topology changes are ignored (not yet supported)
        """
        # Convert variation config to bounds spec using factory method
        # Note: This requires bounds to already exist, so we compute default bounds first if empty
        current_spec = self.get_dimension_bounds_spec()
        if not current_spec.bounds:
            # If bounds are empty, compute default bounds from variation config
            dim_config = mech_variation_config.dimension_variation
            default_bounds = []
            for initial in current_spec.initial_values:
                # Use default_variation_range to compute bounds
                # Equivalent to old bounds_factor=2.0: [initial*0.5, initial*2.0]
                variation_range = dim_config.default_variation_range
                default_bounds.append((
                    initial * (1 - variation_range),
                    initial * (1 + variation_range),
                ))
            # Update current spec with default bounds
            current_spec = DimensionBoundsSpec(
                names=current_spec.names,
                initial_values=current_spec.initial_values,
                bounds=default_bounds,
                edge_mapping=current_spec.edge_mapping,
                weights=current_spec.weights,
            )
            # Update mechanism's dimension mapping with default bounds
            self._dimension_mapping = DimensionMapping(
                names=self._dimension_mapping.names,
                initial_values=self._dimension_mapping.initial_values,
                bounds=default_bounds,
                edge_mapping=self._dimension_mapping.edge_mapping,
                joint_attrs=self._dimension_mapping.joint_attrs,
            )

        new_spec = DimensionBoundsSpec.from_mechanism(self, mech_variation_config)

        # Update dimension mapping with new bounds
        # Filter out excluded dimensions
        dim_config = mech_variation_config.dimension_variation
        excluded = set(dim_config.exclude_dimensions)

        # Rebuild dimension_mapping with filtered dimensions and new bounds
        new_names = [n for n in new_spec.names if n not in excluded]
        new_initial_values = [
            current_spec.initial_values[current_spec.names.index(n)]
            for n in new_names
        ]
        new_bounds = [
            new_spec.bounds[new_spec.names.index(n)]
            for n in new_names
        ]

        # Rebuild joint_attrs for filtered dimensions
        new_joint_attrs = []
        for name in new_names:
            if name in self._dimension_mapping.names:
                idx = self._dimension_mapping.names.index(name)
                if idx < len(self._dimension_mapping.joint_attrs):
                    new_joint_attrs.append(self._dimension_mapping.joint_attrs[idx])
                else:
                    new_joint_attrs.append(None)
            else:
                new_joint_attrs.append(None)

        # Update internal mapping
        self._dimension_mapping = DimensionMapping(
            names=new_names,
            initial_values=new_initial_values,
            bounds=new_bounds,
            edge_mapping={
                k: v for k, v in (self._dimension_mapping.edge_mapping.items())
                if k in new_names
            },
            joint_attrs=new_joint_attrs,
        )

        # Update current dimensions to match new mapping
        self._current_dimensions = np.array(new_initial_values)

    @staticmethod
    def _rebuild_joint_attrs(
        linkage: Linkage,
        edge_mapping: dict[str, tuple[str, str]],
        dimension_names: list[str],
        edges_dict: dict[str, dict],
    ) -> list[tuple[object, str] | None]:
        """
        Rebuild joint_attrs mapping from linkage and edge information.

        Args:
            linkage: The linkage to build mapping for
            edge_mapping: Maps dimension_name -> (edge_id, property_name)
            dimension_names: Ordered list of dimension names
            edges_dict: Dict mapping edge_id -> {source, target, distance}

        Returns:
            List of (joint_obj, attr_name) tuples or None for each dimension
        """
        joint_attrs: list[tuple[object, str] | None] = []

        for dim_name in dimension_names:
            if dim_name not in edge_mapping:
                joint_attrs.append(None)
                continue

            edge_id, _ = edge_mapping[dim_name]
            if edge_id not in edges_dict:
                joint_attrs.append(None)
                continue

            edge = edges_dict[edge_id]
            source_name = edge.get('source')
            target_name = edge.get('target')

            # Find joint that connects source -> target
            found = None
            for joint in linkage.joints:
                if isinstance(joint, Crank):
                    # Crank: joint0 -> this joint (r attribute)
                    if (
                        hasattr(joint, 'joint0') and joint.joint0 and
                        hasattr(joint, 'name') and
                        joint.joint0.name == source_name and
                        joint.name == target_name
                    ):
                        found = (joint, 'r')
                        break
                elif isinstance(joint, Revolute):
                    # Revolute: r0 connects joint0->this, r1 connects joint1->this
                    if (
                        hasattr(joint, 'joint0') and joint.joint0 and
                        hasattr(joint, 'name') and
                        joint.joint0.name == source_name and
                        joint.name == target_name
                    ):
                        found = (joint, 'r0')
                        break
                    elif (
                        hasattr(joint, 'joint1') and joint.joint1 and
                        hasattr(joint, 'name') and
                        joint.joint1.name == source_name and
                        joint.name == target_name
                    ):
                        found = (joint, 'r1')
                        break

            joint_attrs.append(found)

        return joint_attrs


class _FitnessWithMeta(Protocol):
    """Protocol for the callable returned by create_mechanism_fitness (has metadata)."""

    def __call__(self, dimensions: tuple[float, ...] | np.ndarray) -> float: ...
    eval_count: list[int]
    mechanism: Mechanism
    target_joint: str


def create_mechanism_fitness(
    mechanism: Mechanism,
    target: TargetTrajectory | np.ndarray,
    target_joint: str | None = None,
    metric: str = 'mse',
    phase_invariant: bool = True,
    phase_align_method: Literal['fft', 'rotation'] = 'rotation',
    translation_invariant: bool = False,
) -> _FitnessWithMeta:
    """
    Create a fast fitness function using Mechanism.

    Uses compiled Linkage with step_fast() for ~5x speedup.

    Args:
        mechanism: Compiled Mechanism instance
        target: Either TargetTrajectory or numpy array of shape (n_steps, 2)
        target_joint: Joint name (required if target is array, optional if TargetTrajectory)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Alignment method ('rotation', 'fft', 'frechet')
        translation_invariant: Center trajectories before comparison

    Returns:
        Callable that takes dimension tuple/array and returns error float
    """
    # Handle both TargetTrajectory and raw array inputs
    if isinstance(target, TargetTrajectory):
        target_arr = target.positions_array  # Use cached array
        joint_name = target.joint_name
    else:
        target_arr = np.asarray(target, dtype=np.float64)
        if target_joint is None:
            raise ValueError('target_joint required when target is array')
        joint_name = target_joint

    target_idx = mechanism.get_joint_index(joint_name)
    eval_count = [0]

    def fitness(dimensions: tuple[float, ...] | np.ndarray) -> float:
        """Evaluate linkage fitness with given dimensions."""
        eval_count[0] += 1

        try:
            mechanism.set_dimensions(dimensions)
            trajectory = mechanism.simulate()

            if np.isnan(trajectory).any():
                return float('inf')

            # Extract target joint trajectory
            joint_traj = trajectory[:, target_idx, :]

            # Use unified score_trajectory function
            return score_trajectory(
                target_arr, joint_traj,
                metric=metric,
                phase_invariant=phase_invariant,
                phase_align_method=phase_align_method,
                translation_invariant=translation_invariant,
            )

        except Exception:
            return float('inf')

    # Attach metadata for consumers
    out: _FitnessWithMeta = cast(_FitnessWithMeta, fitness)
    out.eval_count = eval_count
    out.mechanism = mechanism
    out.target_joint = joint_name
    return out
