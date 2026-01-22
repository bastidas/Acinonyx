"""
kinematic.py - Core linkage kinematics and trajectory computation.

This module provides reusable functions for:
  - Building pylinkage Linkage objects from serialized data
  - Running forward kinematics (trajectory simulation)
  - Mechanism validation

Design notes:
  - Uses pylinkage's native hypergraph module for v2.0 format
  - Falls back to pylinkage's serialization for legacy format
  - Compatible with pylinkage's optimization API
"""
from __future__ import annotations

import math
from typing import Any

from pylink_tools.schemas import MechanismGroup
from pylink_tools.schemas import TrajectoryResult


# =============================================================================
# Format Detection
# =============================================================================

def is_hypergraph_format(data: dict) -> bool:
    """
    Check if data is in hypergraph format (v2.0.0 LinkageDocument).

    Hypergraph format has: linkage.nodes, linkage.edges
    Legacy format has: pylinkage.joints
    """
    return 'linkage' in data and 'nodes' in data.get('linkage', {})


def is_legacy_format(data: dict) -> bool:
    """
    Check if data is in legacy format (pylinkage.joints).
    """
    return 'pylinkage' in data and 'joints' in data.get('pylinkage', {})


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_trajectory(
    pylink_data: dict,
    verbose: bool = False,
    skip_sync: bool = False,
) -> TrajectoryResult:
    """
    Compute joint trajectories from LinkageDocument or legacy format.

    This is the main entry point for trajectory computation.

    Args:
        pylink_data: Document in either format:
            - Hypergraph: { 'linkage': { 'nodes': {...}, 'edges': {...} }, ... }
            - Legacy: { 'pylinkage': { 'joints': [...] }, ... }
        verbose: If True, print progress
        skip_sync: If True, skip syncing distances from visual positions.
                   Use this for optimization when you want the stored distances
                   to be used directly.

    Returns:
        TrajectoryResult with trajectories or error
    """
    n_steps = pylink_data.get('n_steps', 12)

    # Route to appropriate handler based on format
    if is_hypergraph_format(pylink_data):
        return _compute_trajectory_hypergraph(pylink_data, n_steps, verbose, skip_sync)
    elif is_legacy_format(pylink_data):
        return _compute_trajectory_legacy(pylink_data, n_steps, verbose)
    else:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error='Unknown data format: expected hypergraph (linkage.nodes) or legacy (pylinkage.joints)',
        )


def _compute_trajectory_hypergraph(
    pylink_data: dict,
    n_steps: int,
    verbose: bool,
    skip_sync: bool,
) -> TrajectoryResult:
    """
    Compute trajectory using pylinkage's native hypergraph module.

    This is the preferred path for v2.0 format.
    """
    from pylink_tools.hypergraph_adapter import simulate_hypergraph
    from pylink_tools.hypergraph_adapter import sync_hypergraph_distances

    if verbose:
        print('  Using native hypergraph format...')

    # Sync distances from node positions if needed
    if not skip_sync:
        pylink_data = sync_hypergraph_distances(pylink_data, verbose=verbose)

    result = simulate_hypergraph(pylink_data, n_steps=n_steps)

    if not result.success:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error=result.error or 'Unknown error in hypergraph simulation',
        )

    # Build joint_types from nodes
    nodes = pylink_data.get('linkage', {}).get('nodes', {})
    joint_types = {}
    for name in result.joint_names:
        node = nodes.get(name, {})
        role = node.get('role', 'follower')
        if role == 'fixed':
            joint_types[name] = 'Static'
        elif role in ('crank', 'driver'):
            joint_types[name] = 'Crank'
        else:
            joint_types[name] = 'Revolute'

    return TrajectoryResult(
        success=True,
        trajectories=result.trajectories,
        n_steps=n_steps,
        joint_types=joint_types,
    )


def _compute_trajectory_legacy(
    pylink_data: dict,
    n_steps: int,
    verbose: bool,
) -> TrajectoryResult:
    """
    Compute trajectory using pylinkage's serialization for legacy format.

    This is a fallback for backward compatibility with old files.
    """
    from pylinkage.joints import Crank
    from pylinkage.linkage.serialization import linkage_from_dict

    if verbose:
        print('  Using legacy format with pylinkage serialization...')

    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])

    if not joints_data:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error='No joints found in pylinkage data',
        )

    try:
        # Use pylinkage's native serialization
        linkage = linkage_from_dict(pylinkage_data)

        # Set crank angle for proper rotation
        angle_per_step = 2 * math.pi / n_steps
        for joint in linkage.joints:
            if isinstance(joint, Crank):
                joint.angle = angle_per_step

        # Rebuild linkage after modifying crank angles
        linkage.rebuild()

        # Run simulation
        joint_names = [j.name for j in linkage.joints]
        trajectories = {name: [] for name in joint_names}

        for step_coords in linkage.step(iterations=n_steps):
            for i, coord in enumerate(step_coords):
                if coord[0] is not None and coord[1] is not None:
                    trajectories[joint_names[i]].append((coord[0], coord[1]))
                else:
                    trajectories[joint_names[i]].append((float('nan'), float('nan')))

        # Build joint_types
        joint_types = {j['name']: j['type'] for j in joints_data}

        return TrajectoryResult(
            success=True,
            trajectories=trajectories,
            n_steps=n_steps,
            joint_types=joint_types,
        )

    except Exception as e:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error=f'Legacy simulation failed: {str(e)}',
        )


# =============================================================================
# Mechanism Validation
# =============================================================================

def find_connected_link_groups(
    edges: dict[str, dict],
    nodes: dict[str, dict],
) -> list[MechanismGroup]:
    """
    Find connected groups of edges (links) in the graph.

    Two edges are connected if they share a node.

    Args:
        edges: linkage.edges dict {edge_id: {source, target, distance}, ...}
        nodes: linkage.nodes dict {node_id: {id, position, role, ...}, ...}

    Returns:
        List of MechanismGroup, one per connected component
    """
    if not edges:
        return []

    # Build adjacency: which edges share nodes
    edge_ids = list(edges.keys())
    edge_nodes = {}
    for edge_id, edge in edges.items():
        edge_nodes[edge_id] = {edge['source'], edge['target']}

    # Union-Find for connected components
    parent = {eid: eid for eid in edge_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Connect edges that share any node
    for i, eid1 in enumerate(edge_ids):
        for eid2 in edge_ids[i+1:]:
            if edge_nodes[eid1] & edge_nodes[eid2]:  # shared nodes
                union(eid1, eid2)

    # Group edges by component
    components: dict[str, list[str]] = {}
    for eid in edge_ids:
        root = find(eid)
        if root not in components:
            components[root] = []
        components[root].append(eid)

    # Build MechanismGroup for each component
    groups = []
    for component_edges in components.values():
        # Gather all nodes in this component
        component_nodes = set()
        for edge_id in component_edges:
            component_nodes.update(edge_nodes[edge_id])

        # Check for crank and ground
        has_crank = any(
            nodes.get(nid, {}).get('role') in ('crank', 'driver')
            for nid in component_nodes
        )
        has_ground = any(
            nodes.get(nid, {}).get('role') in ('fixed', 'ground')
            for nid in component_nodes
        )

        # Validate
        error = None
        is_valid = True

        if len(component_edges) < 3:
            is_valid = False
            error = f'Need at least 3 edges, got {len(component_edges)}'
        elif not has_crank:
            is_valid = False
            error = 'No driver (crank) node found - mechanism needs a driver'
        elif not has_ground:
            is_valid = False
            error = 'No fixed (ground) node found'

        groups.append(
            MechanismGroup(
                links=component_edges,
                joints=list(component_nodes),
                has_crank=has_crank,
                has_ground=has_ground,
                is_valid=is_valid,
                error=error,
            ),
        )

    return groups


def validate_mechanism(pylink_data: dict) -> dict[str, Any]:
    """
    Validate a linkage document and identify valid mechanism groups.

    Works with hypergraph format (linkage.nodes/edges).

    Args:
        pylink_data: Full linkage document

    Returns:
        {
            "valid": bool,
            "groups": [MechanismGroup, ...],
            "valid_groups": [MechanismGroup, ...],
            "errors": [str, ...]
        }
    """
    linkage = pylink_data.get('linkage', {})
    nodes = linkage.get('nodes', {})
    edges = linkage.get('edges', {})

    errors = []

    if not nodes:
        errors.append('No nodes defined')
    if not edges:
        errors.append('No edges defined')

    if errors:
        return {
            'valid': False,
            'groups': [],
            'valid_groups': [],
            'errors': errors,
        }

    # Find connected groups
    groups = find_connected_link_groups(edges, nodes)
    valid_groups = [g for g in groups if g.is_valid]

    # Test simulation on valid groups
    for group in valid_groups:
        # Create a minimal document for this group
        group_nodes = {nid: nodes[nid] for nid in group.joints if nid in nodes}
        group_edges = {eid: edges[eid] for eid in group.links if eid in edges}

        test_data = {
            'linkage': {
                'nodes': group_nodes,
                'edges': group_edges,
            },
            'n_steps': 12,
        }

        # Try simulation
        result = compute_trajectory(test_data, verbose=False)
        if not result.success:
            group.is_valid = False
            group.error = result.error

    # Refilter after simulation check
    valid_groups = [g for g in groups if g.is_valid]

    return {
        'valid': len(valid_groups) > 0,
        'groups': [_group_to_dict(g) for g in groups],
        'valid_groups': [_group_to_dict(g) for g in valid_groups],
        'errors': [g.error for g in groups if g.error],
    }


def _group_to_dict(g: MechanismGroup) -> dict:
    return {
        'links': g.links,
        'joints': g.joints,
        'has_crank': g.has_crank,
        'has_ground': g.has_ground,
        'is_valid': g.is_valid,
        'error': g.error,
    }
