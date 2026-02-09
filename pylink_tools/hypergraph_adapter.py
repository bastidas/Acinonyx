"""
hypergraph_adapter.py - Convert pylink_data to pylinkage Linkage and extract dimensions.

This module bridges our JSON format (used by frontend) to pylinkage's native
HypergraphLinkage class for simulation, and provides utilities for extracting
optimizable dimensions from hypergraph format.

Our JSON format (frontend-friendly, dict-keyed):
    {
        "linkage": {
            "nodes": { "A": { "id": "A", "position": [x, y], "role": "fixed", ... }, ... },
            "edges": { "link1": { "source": "A", "target": "B", "distance": 20 }, ... }
        }
    }

Usage:
    from pylink_tools.hypergraph_adapter import to_simulatable_linkage, extract_dimensions

    linkage = to_simulatable_linkage(pylink_data)
    dim_spec = extract_dimensions(pylink_data)
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pylink_tools.optimization_types import DimensionBoundsSpec

if TYPE_CHECKING:
    from pylinkage.hypergraph import HypergraphLinkage
    from pylinkage.linkage.linkage import Linkage

# Role mapping: our frontend uses lowercase, pylinkage uses uppercase
ROLE_TO_PYLINKAGE = {
    'fixed': 'GROUND',
    'ground': 'GROUND',
    'crank': 'DRIVER',
    'driver': 'DRIVER',
    'follower': 'DRIVEN',
    'driven': 'DRIVEN',
}


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

    This is the preferred method for creating Linkage objects.
    Use Mechanism.from_pylink_data() for the full API with dimension tracking.

    Args:
        pylink_data: Our format with 'linkage.nodes' and 'linkage.edges'

    Returns:
        pylinkage Linkage instance ready for simulation
    """
    from pylinkage.hypergraph import to_linkage

    hg = to_pylinkage_hypergraph(pylink_data)
    return to_linkage(hg)


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
# Dimension Extraction
# =============================================================================


def extract_dimensions(
    pylink_data: dict,
) -> tuple[DimensionBoundsSpec, set[str]]:
    """
    Extract optimizable dimensions (link lengths) from pylink_data.

    This function ONLY extracts what IS - it does not compute bounds,
    exclude edges, or apply any configuration. It simply identifies
    which dimensions exist and their current values.

    Args:
        pylink_data: Full pylink document with 'linkage.edges'

    Returns:
        Tuple of:
        - DimensionBoundsSpec with names, initial_values, edge_mapping (NO BOUNDS - empty list)
        - set of static node IDs (for future static joint variation support)

    Note:
        - Bounds should be set via Mechanism.apply_variation_config() post-init
        - Default bounds can be set using MechVariationConfig(default_variation_range=2.0)
        - Exclusions should be handled via MechVariationConfig
        - This is a pure extraction function - no policy decisions
    """
    linkage = pylink_data.get('linkage', {})
    nodes = linkage.get('nodes', {})
    edges = linkage.get('edges', {})

    names: list[str] = []
    initial_values: list[float] = []
    edge_mapping: dict[str, tuple[str, str]] = {}

    # Get static node IDs (for future static joint variation support)
    static_nodes = {
        node_id for node_id, node in nodes.items()
        if node.get('role', '').lower() in ('fixed', 'ground', 'static')
    }

    # Extract all edge distances (except edges between two static nodes)
    for edge_id, edge in edges.items():
        source = edge.get('source', '')
        target = edge.get('target', '')

        # Skip edges between two static nodes (ground links - not optimizable)
        if source in static_nodes and target in static_nodes:
            continue

        distance = edge.get('distance', 1.0)
        if distance is None:
            distance = 1.0

        dim_name = f'{edge_id}_distance'

        names.append(dim_name)
        initial_values.append(float(distance))
        edge_mapping[dim_name] = (edge_id, 'distance')

    # Return spec WITHOUT bounds - bounds will be set post-init via apply_variation_config()
    return DimensionBoundsSpec(
        names=names,
        initial_values=initial_values,
        bounds=[],  # Empty - will be set via apply_variation_config() with default config
        edge_mapping=edge_mapping,
    ), static_nodes


def _build_joint_attr_mapping(
    linkage: Linkage,
    pylink_data: dict,
    dimension_spec: DimensionBoundsSpec,
) -> list[tuple[object, str] | None]:
    """
    Build mapping from dimension index to (joint, attribute_name).

    This allows direct joint attribute updates without invalidating solver_data.

    Args:
        linkage: Compiled Linkage object
        pylink_data: Original pylink_data dict
        dimension_spec: Dimension specification

    Returns:
        List where index i maps to (joint_obj, attr_name) or None
    """
    from pylinkage.joints import Crank, Revolute, Static

    edges = pylink_data.get('linkage', {}).get('edges', {})
    edge_mapping = getattr(dimension_spec, 'edge_mapping', {}) or {}

    # Build (source, target) -> edge_id lookup
    edge_by_nodes: dict[tuple[str, str], str] = {}
    for edge_id, edge in edges.items():
        edge_by_nodes[(edge['source'], edge['target'])] = edge_id
        edge_by_nodes[(edge['target'], edge['source'])] = edge_id

    # Build edge_id -> (joint, attr_name) mapping
    edge_to_joint_attr: dict[str, tuple[object, str]] = {}

    for joint in linkage.joints:
        if isinstance(joint, Static) and not isinstance(joint, Crank):
            # Static joints (non-crank) don't have distance attributes
            continue
        elif isinstance(joint, Crank):
            # Crank has 'r' attribute for radius
            if joint.joint0:
                key = (joint.joint0.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r')
        elif isinstance(joint, Revolute):
            # Revolute has 'r0' and 'r1' for two link distances
            if joint.joint0:
                key = (joint.joint0.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r0')
            if joint.joint1:
                key = (joint.joint1.name, joint.name)
                if key in edge_by_nodes:
                    edge_to_joint_attr[edge_by_nodes[key]] = (joint, 'r1')

    # Map dimension index to (joint, attr)
    mapping: list[tuple[object, str] | None] = []
    for dim_name in dimension_spec.names:
        if dim_name in edge_mapping:
            edge_id, _ = edge_mapping[dim_name]
            mapping.append(edge_to_joint_attr.get(edge_id))
        else:
            mapping.append(None)

    return mapping
