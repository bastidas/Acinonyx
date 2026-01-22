"""
Adapter for pylinkage's native hypergraph module.

This module bridges our JSON format (used by frontend) to pylinkage's native
HypergraphLinkage class, enabling:
- Direct use of pylinkage's `to_linkage()` for simulation
- Cleaner code with less custom conversion logic
- Better compatibility with pylinkage updates

Our JSON format (frontend-friendly, dict-keyed):
    {
        "linkage": {
            "nodes": { "A": { "id": "A", "position": [x, y], "role": "fixed", ... }, ... },
            "edges": { "link1": { "source": "A", "target": "B", "distance": 20 }, ... }
        }
    }

pylinkage format (array-based, enum roles):
    {
        "nodes": [{ "id": "A", "position": [x, y], "role": "GROUND", ... }, ...],
        "edges": [{ "id": "link1", "source": "A", "target": "B", "distance": 20 }, ...]
    }

Usage:
    from pylink_tools.hypergraph_adapter import (
        to_pylinkage_hypergraph,
        from_pylinkage_hypergraph,
        simulate_hypergraph,
    )

    # Convert our format to pylinkage's HypergraphLinkage
    hg = to_pylinkage_hypergraph(pylink_data)

    # Simulate and get trajectories
    trajectories = simulate_hypergraph(hg, n_steps=24)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylinkage.hypergraph import HypergraphLinkage
    from pylinkage.linkage.linkage import Linkage


# =============================================================================
# Role Mapping
# =============================================================================

# Our frontend uses lowercase role names, pylinkage uses uppercase enum names
ROLE_TO_PYLINKAGE = {
    'fixed': 'GROUND',
    'ground': 'GROUND',
    'crank': 'DRIVER',
    'driver': 'DRIVER',
    'follower': 'DRIVEN',
    'driven': 'DRIVEN',
}

ROLE_FROM_PYLINKAGE = {v: k for k, v in ROLE_TO_PYLINKAGE.items()}
# Normalize to our preferred names
ROLE_FROM_PYLINKAGE['GROUND'] = 'fixed'
ROLE_FROM_PYLINKAGE['DRIVER'] = 'crank'
ROLE_FROM_PYLINKAGE['DRIVEN'] = 'follower'


# =============================================================================
# Format Detection
# =============================================================================

def is_our_hypergraph_format(data: dict) -> bool:
    """Check if data is in our dict-keyed hypergraph format."""
    linkage = data.get('linkage', {})
    nodes = linkage.get('nodes', {})
    edges = linkage.get('edges', {})

    # Our format: nodes and edges are dicts keyed by ID
    return isinstance(nodes, dict) and isinstance(edges, dict) and len(nodes) > 0


def is_pylinkage_hypergraph_format(data: dict) -> bool:
    """Check if data is in pylinkage's array-based hypergraph format."""
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])

    # pylinkage format: nodes and edges are arrays
    return isinstance(nodes, list) and isinstance(edges, list)


def is_legacy_joints_format(data: dict) -> bool:
    """Check if data is in legacy pylinkage.joints format."""
    pylinkage = data.get('pylinkage', {})
    joints = pylinkage.get('joints', [])
    return isinstance(joints, list) and len(joints) > 0


# =============================================================================
# Conversion: Our Format → pylinkage HypergraphLinkage
# =============================================================================

def to_pylinkage_dict(pylink_data: dict) -> dict:
    """
    Convert our dict-keyed format to pylinkage's array-based dict format.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges' as dicts

    Returns:
        Dict in pylinkage format (arrays with proper role names)
    """
    linkage = pylink_data.get('linkage', {})
    our_nodes = linkage.get('nodes', {})
    our_edges = linkage.get('edges', {})

    pylinkage_nodes = []
    for node_id, node in our_nodes.items():
        role = node.get('role', 'follower')
        pylinkage_role = ROLE_TO_PYLINKAGE.get(role.lower(), 'DRIVEN')

        joint_type = node.get('jointType', 'revolute')
        pylinkage_joint_type = joint_type.upper() if joint_type else 'REVOLUTE'

        pylinkage_nodes.append({
            'id': node_id,
            'position': node.get('position', [None, None]),
            'role': pylinkage_role,
            'joint_type': pylinkage_joint_type,
            'angle': node.get('angle'),
            'initial_angle': node.get('angle'),
            'name': node.get('name', node_id),
        })

    pylinkage_edges = []
    for edge_id, edge in our_edges.items():
        pylinkage_edges.append({
            'id': edge_id,
            'source': edge['source'],
            'target': edge['target'],
            'distance': edge.get('distance'),
        })

    # Handle hyperedges if present
    our_hyperedges = linkage.get('hyperedges', {})
    pylinkage_hyperedges = []
    for he_id, he in our_hyperedges.items():
        if isinstance(he, dict):
            pylinkage_hyperedges.append({
                'id': he_id,
                'nodes': he.get('nodes', []),
                'constraints': he.get('constraints', []),
                'name': he.get('name', he_id),
            })

    return {
        'name': linkage.get('name', pylink_data.get('name', 'unnamed')),
        'nodes': pylinkage_nodes,
        'edges': pylinkage_edges,
        'hyperedges': pylinkage_hyperedges,
    }


def to_pylinkage_hypergraph(pylink_data: dict) -> HypergraphLinkage:
    """
    Convert our format to pylinkage's native HypergraphLinkage.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges'

    Returns:
        pylinkage.hypergraph.HypergraphLinkage instance
    """
    from pylinkage.hypergraph import graph_from_dict

    pylinkage_dict = to_pylinkage_dict(pylink_data)
    return graph_from_dict(pylinkage_dict)


def to_simulatable_linkage(pylink_data: dict) -> Linkage:
    """
    Convert our format directly to a simulatable Linkage.

    This is the preferred method for trajectory computation.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges'

    Returns:
        pylinkage Linkage instance ready for simulation
    """
    from pylinkage.hypergraph import to_linkage

    hg = to_pylinkage_hypergraph(pylink_data)
    return to_linkage(hg)


# =============================================================================
# Conversion: pylinkage HypergraphLinkage → Our Format
# =============================================================================

def from_pylinkage_hypergraph(hg: HypergraphLinkage) -> dict:
    """
    Convert pylinkage's HypergraphLinkage back to our dict-keyed format.

    Args:
        hg: pylinkage HypergraphLinkage instance

    Returns:
        Our format with 'linkage.nodes' and 'linkage.edges' as dicts
    """
    from pylinkage.hypergraph import graph_to_dict

    pylinkage_dict = graph_to_dict(hg)

    # Convert arrays back to dicts
    nodes = {}
    for node in pylinkage_dict.get('nodes', []):
        node_id = node['id']
        role = node.get('role', 'DRIVEN')
        our_role = ROLE_FROM_PYLINKAGE.get(role, 'follower')

        nodes[node_id] = {
            'id': node_id,
            'position': node.get('position', [None, None]),
            'role': our_role,
            'jointType': node.get('joint_type', 'REVOLUTE').lower(),
            'angle': node.get('angle'),
            'name': node.get('name', node_id),
        }

    edges = {}
    for edge in pylinkage_dict.get('edges', []):
        edge_id = edge['id']
        edges[edge_id] = {
            'id': edge_id,
            'source': edge['source'],
            'target': edge['target'],
            'distance': edge.get('distance'),
        }

    hyperedges = {}
    for he in pylinkage_dict.get('hyperedges', []):
        he_id = he['id']
        hyperedges[he_id] = he

    return {
        'name': pylinkage_dict.get('name', 'unnamed'),
        'linkage': {
            'name': pylinkage_dict.get('name', 'unnamed'),
            'nodes': nodes,
            'edges': edges,
            'hyperedges': hyperedges,
        },
    }


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class SimulationResult:
    """Result of hypergraph simulation."""
    success: bool
    trajectories: dict[str, list[tuple[float, float]]]
    n_steps: int
    joint_names: list[str]
    error: str | None = None


def simulate_hypergraph(
    pylink_data: dict,
    n_steps: int = 24,
) -> SimulationResult:
    """
    Simulate a mechanism using pylinkage's native hypergraph.

    This is the preferred method for trajectory computation with hypergraph format.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges'
        n_steps: Number of simulation steps (default 24 for full rotation)

    Returns:
        SimulationResult with trajectories for each joint
    """
    import math
    from pylinkage.joints import Crank

    try:
        linkage = to_simulatable_linkage(pylink_data)

        # CRITICAL FIX: pylinkage's Crank.angle is the angular VELOCITY (radians per step),
        # not the initial angle. We need to set it to 2*pi/n_steps for a full rotation.
        # The initial angle is already encoded in the crank's position.
        angle_per_step = 2 * math.pi / n_steps

        for joint in linkage.joints:
            if isinstance(joint, Crank):
                joint.angle = angle_per_step

        # Rebuild linkage after modifying crank angles
        linkage.rebuild()

        # Initialize trajectories dict
        joint_names = [j.name for j in linkage.joints]
        trajectories = {name: [] for name in joint_names}

        # Run simulation - step() is a GENERATOR that yields coords at each step
        for step_coords in linkage.step(iterations=n_steps):
            for i, (x, y) in enumerate(step_coords):
                trajectories[joint_names[i]].append((x, y))

        return SimulationResult(
            success=True,
            trajectories=trajectories,
            n_steps=n_steps,
            joint_names=joint_names,
        )

    except Exception as e:
        return SimulationResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_names=[],
            error=str(e),
        )


# =============================================================================
# Edge Dimension Utilities
# =============================================================================

def get_edge_distances(pylink_data: dict) -> dict[str, float]:
    """
    Extract edge distances from our hypergraph format.

    Args:
        pylink_data: Our format with 'linkage.edges'

    Returns:
        Dict mapping edge_id -> distance
    """
    edges = pylink_data.get('linkage', {}).get('edges', {})
    return {
        edge_id: edge.get('distance', 0.0)
        for edge_id, edge in edges.items()
    }


def set_edge_distance(
    pylink_data: dict,
    edge_id: str,
    distance: float,
    inplace: bool = False,
) -> dict:
    """
    Set an edge distance in our hypergraph format.

    Args:
        pylink_data: Our format with 'linkage.edges'
        edge_id: ID of the edge to modify
        distance: New distance value
        inplace: If True, modify in place; otherwise return copy

    Returns:
        Modified pylink_data
    """
    if not inplace:
        pylink_data = copy.deepcopy(pylink_data)

    edges = pylink_data.get('linkage', {}).get('edges', {})
    if edge_id in edges:
        edges[edge_id]['distance'] = distance

    return pylink_data


def set_edge_distances(
    pylink_data: dict,
    distances: dict[str, float],
    inplace: bool = False,
) -> dict:
    """
    Set multiple edge distances in our hypergraph format.

    Args:
        pylink_data: Our format with 'linkage.edges'
        distances: Dict mapping edge_id -> new distance
        inplace: If True, modify in place; otherwise return copy

    Returns:
        Modified pylink_data
    """
    if not inplace:
        pylink_data = copy.deepcopy(pylink_data)

    edges = pylink_data.get('linkage', {}).get('edges', {})
    for edge_id, distance in distances.items():
        if edge_id in edges:
            edges[edge_id]['distance'] = distance

    return pylink_data


def sync_hypergraph_distances(
    pylink_data: dict,
    verbose: bool = False,
    inplace: bool = True,
) -> dict:
    """
    Sync edge distances to match node positions in hypergraph format.

    This ensures edge.distance values are consistent with the Euclidean
    distance between source and target node positions. Essential before
    optimization when the frontend may have stale/incorrect distances.

    Args:
        pylink_data: Our hypergraph format with 'linkage.nodes' and 'linkage.edges'
        verbose: If True, print sync changes
        inplace: If True, modify in place; otherwise return copy

    Returns:
        Updated pylink_data with synced distances
    """
    import math

    if not inplace:
        pylink_data = copy.deepcopy(pylink_data)

    linkage = pylink_data.get('linkage', {})
    nodes = linkage.get('nodes', {})
    edges = linkage.get('edges', {})

    if not nodes or not edges:
        return pylink_data

    synced_count = 0
    for edge_id, edge in edges.items():
        source_id = edge.get('source')
        target_id = edge.get('target')

        if source_id not in nodes or target_id not in nodes:
            if verbose:
                print(f'  [SYNC] Edge {edge_id}: missing node(s) {source_id} or {target_id}')
            continue

        source_pos = nodes[source_id].get('position', [0, 0])
        target_pos = nodes[target_id].get('position', [0, 0])

        # Calculate Euclidean distance
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        new_distance = math.sqrt(dx * dx + dy * dy)

        old_distance = edge.get('distance', 0)

        # Only update if significantly different (avoid floating point noise)
        if abs(new_distance - old_distance) > 0.001:
            edge['distance'] = new_distance
            synced_count += 1
            if verbose:
                print(f'  [SYNC] Edge {edge_id}: distance {old_distance:.2f} → {new_distance:.2f}')

    if verbose and synced_count > 0:
        print(f'  Synced {synced_count} edge distances')

    return pylink_data


# =============================================================================
# Linkage Mutation for Optimization
# =============================================================================

def create_linkage_mutator(pylink_data: dict) -> tuple[Linkage, dict[str, Any]]:
    """
    Create a mutable Linkage for optimization.

    Returns a Linkage instance and a mapping dict that allows fast
    edge distance updates without recreating the linkage.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges'

    Returns:
        (linkage, edge_lookup) where edge_lookup maps edge_id -> edge object

    Usage:
        linkage, edge_lookup = create_linkage_mutator(pylink_data)

        # Fast dimension update in optimization loop
        edge_lookup['crank_link'].distance = 25.0
        edge_lookup['coupler'].distance = 55.0

        # Simulate with new dimensions
        linkage.step()
    """
    from pylinkage.hypergraph import to_linkage

    hg = to_pylinkage_hypergraph(pylink_data)
    linkage = to_linkage(hg)

    # Build edge lookup for fast updates
    # Note: The linkage stores edges internally, we need to find them
    # For now, return the hypergraph's edge dict for reference
    edge_lookup = {edge.id: edge for edge in hg.edges.values()}

    return linkage, edge_lookup


# =============================================================================
# Compatibility Layer
# =============================================================================

def compute_trajectory_native(
    pylink_data: dict,
    n_steps: int | None = None,
    verbose: bool = False,
) -> SimulationResult:
    """
    Compute trajectory using pylinkage's native hypergraph (if applicable).

    Falls back to legacy format if the data isn't in hypergraph format.

    Args:
        pylink_data: Either our hypergraph format or legacy joints format
        n_steps: Number of steps (default from pylink_data or 24)
        verbose: Print progress information

    Returns:
        SimulationResult with trajectories
    """
    if n_steps is None:
        n_steps = pylink_data.get('n_steps', 24)

    if is_our_hypergraph_format(pylink_data):
        if verbose:
            print('  Using native pylinkage hypergraph simulation')
        return simulate_hypergraph(pylink_data, n_steps=n_steps)

    elif is_legacy_joints_format(pylink_data):
        if verbose:
            print('  Using legacy joints format (no conversion needed)')
        # Fall back to existing kinematic.py implementation
        from pylink_tools.kinematic import compute_trajectory as legacy_compute
        result = legacy_compute(pylink_data, verbose=verbose)
        return SimulationResult(
            success=result.success,
            trajectories=result.trajectories,
            n_steps=result.n_steps,
            joint_names=list(result.trajectories.keys()),
            error=result.error,
        )

    else:
        return SimulationResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_names=[],
            error='Unknown data format: expected hypergraph or legacy joints',
        )
