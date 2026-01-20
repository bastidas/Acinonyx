"""
kinematic.py - Core linkage kinematics and trajectory computation.

This module provides reusable functions for:
  - Building pylinkage Joint objects from serialized data
  - Computing joint positions (circle-circle intersection, crank geometry)
  - Running forward kinematics (trajectory simulation)
  - (Future) Optimization routines for linkage synthesis

Design notes:
  - Functions are pure where possible (no side effects)
  - Separation of concerns: parsing, geometry, simulation, optimization
  - Compatible with pylinkage's optimization API (trials, PSO)
"""
from __future__ import annotations

from typing import Any
from typing import Union

import numpy as np
from pylinkage.joints import Crank
from pylinkage.joints import Revolute
from pylinkage.linkage import Linkage

from pylink_tools.schemas import MechanismGroup
from pylink_tools.schemas import TrajectoryResult


# =============================================================================
# Hypergraph Format Conversion (v2.0.0 LinkageDocument -> Legacy PylinkDocument)
# =============================================================================

def is_hypergraph_format(data: dict) -> bool:
    """
    Check if data is in new hypergraph format (v2.0.0 LinkageDocument).

    Hypergraph format has: linkage.nodes, linkage.edges
    Legacy format has: pylinkage.joints
    """
    return 'linkage' in data and 'nodes' in data.get('linkage', {})


def convert_hypergraph_to_legacy(doc: dict, verbose: bool = False) -> dict:
    """
    Convert hypergraph LinkageDocument to legacy PylinkDocument format.

    This allows the backend to support both formats until full migration.

    Hypergraph format (v2.0.0):
        {
            "linkage": {
                "nodes": { "id": { "id", "position", "role", "angle", ... }, ... },
                "edges": { "id": { "id", "source", "target", "distance" }, ... }
            },
            "meta": { "nodes": {...}, "edges": {...} }
        }

    Legacy format:
        {
            "pylinkage": {
                "joints": [{ "type", "name", "x"/"y" or "joint0"/"distance", ... }],
                "solve_order": [...]
            },
            "meta": { "joints": {...}, "links": {...} }
        }
    """
    linkage = doc.get('linkage', {})
    nodes = linkage.get('nodes', {})
    edges = linkage.get('edges', {})
    meta = doc.get('meta', {})
    node_meta = meta.get('nodes', {})
    edge_meta = meta.get('edges', {})

    if verbose:
        print(f'  Converting hypergraph format: {len(nodes)} nodes, {len(edges)} edges')

    # Build adjacency: node_id -> list of (edge_id, other_node_id, distance)
    adjacency: dict[str, list[tuple[str, str, float]]] = {nid: [] for nid in nodes}
    for edge_id, edge in edges.items():
        src, tgt = edge['source'], edge['target']
        dist = edge.get('distance', 0)
        if src in adjacency:
            adjacency[src].append((edge_id, tgt, dist))
        if tgt in adjacency:
            adjacency[tgt].append((edge_id, src, dist))

    # Sort nodes by role priority: fixed -> crank -> follower
    role_order = {'fixed': 0, 'crank': 1, 'driven': 1, 'follower': 2}
    sorted_nodes = sorted(
        nodes.values(),
        key=lambda n: (role_order.get(n.get('role', 'follower'), 3), n.get('id', '')),
    )

    # Track which joints have been processed (for dependency resolution)
    processed = set()
    joints = []
    joints_meta = {}
    validation_issues = []

    # First pass: process fixed and crank nodes
    for node in sorted_nodes:
        nid = node['id']
        role = node.get('role', 'follower')
        pos = node.get('position', [0, 0])
        nmeta = node_meta.get(nid, {})

        if role == 'fixed':
            joints.append({
                'type': 'Static',
                'name': nid,
                'x': pos[0],
                'y': pos[1],
            })
            processed.add(nid)

        elif role in ('crank', 'driven'):
            # Find parent (fixed node connected via edge)
            parent_id = None
            distance = 0

            for edge_id, other_id, dist in adjacency.get(nid, []):
                other_node = nodes.get(other_id, {})
                if other_node.get('role') == 'fixed':
                    parent_id = other_id
                    distance = dist
                    break

            if parent_id:
                joints.append({
                    'type': 'Crank',
                    'name': nid,
                    'joint0': {'ref': parent_id},
                    'distance': distance,
                    'angle': node.get('angle', 0),
                })
                processed.add(nid)
            else:
                validation_issues.append(f"Crank '{nid}' has no fixed parent")

        # Save meta
        joints_meta[nid] = {
            'color': nmeta.get('color', ''),
            'zlevel': nmeta.get('zlevel', 0),
            'x': pos[0],
            'y': pos[1],
            'show_path': nmeta.get('showPath', True),
        }

    # Second pass: process follower nodes (need two processed parents)
    max_iterations = len(sorted_nodes)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        made_progress = False

        for node in sorted_nodes:
            nid = node['id']
            if nid in processed:
                continue

            role = node.get('role', 'follower')
            if role not in ('follower',):
                continue

            # Find two processed parent nodes
            parents = []
            for edge_id, other_id, dist in adjacency.get(nid, []):
                if other_id in processed:
                    parents.append((other_id, dist))
                    if len(parents) >= 2:
                        break

            if len(parents) >= 2:
                joints.append({
                    'type': 'Revolute',
                    'name': nid,
                    'joint0': {'ref': parents[0][0]},
                    'joint1': {'ref': parents[1][0]},
                    'distance0': parents[0][1],
                    'distance1': parents[1][1],
                })
                processed.add(nid)
                made_progress = True
            elif len(parents) == 1:
                # Underconstrained: only one parent
                # This joint cannot be kinematically simulated
                pass
            elif len(parents) == 0:
                # No connections to processed joints yet
                pass

        if not made_progress:
            break

    # Check for unprocessed nodes (underconstrained)
    unprocessed = set(nodes.keys()) - processed
    if unprocessed and verbose:
        print(f'  Warning: Underconstrained nodes (cannot simulate): {unprocessed}')
        for nid in unprocessed:
            edge_count = len(adjacency.get(nid, []))
            validation_issues.append(
                f"Node '{nid}' is underconstrained ({edge_count} edge(s), need 2 for Revolute)",
            )

    # Convert edges to links
    links_meta = {}
    for edge_id, edge in edges.items():
        emeta = edge_meta.get(edge_id, {})
        links_meta[edge_id] = {
            'color': emeta.get('color', '#888888'),
            'connects': [edge['source'], edge['target']],
            'isGround': emeta.get('isGround', False),
        }

    legacy = {
        'name': doc.get('name', 'converted'),
        'pylinkage': {
            'name': linkage.get('name', 'converted'),
            'joints': joints,
            'solve_order': [j['name'] for j in joints],
        },
        'meta': {
            'joints': joints_meta,
            'links': links_meta,
        },
        '_validation_issues': validation_issues,
        '_unprocessed_nodes': list(unprocessed),
    }

    return legacy


def convert_legacy_to_hypergraph(legacy: dict, verbose: bool = False) -> dict:
    """
    Convert legacy PylinkDocument to hypergraph LinkageDocument format (v2.0.0).

    This is used to return results in the new format after internal processing.

    Legacy format:
        {
            "pylinkage": {
                "joints": [{ "type", "name", "x"/"y" or "joint0"/"distance", ... }],
                "solve_order": [...]
            },
            "meta": { "joints": {...}, "links": {...} }
        }

    Hypergraph format (v2.0.0):
        {
            "linkage": {
                "nodes": { "id": { "id", "position", "role", "angle", ... }, ... },
                "edges": { "id": { "id", "source", "target", "distance" }, ... }
            },
            "meta": { "nodes": {...}, "edges": {...} }
        }
    """
    pylinkage = legacy.get('pylinkage', {})
    joints = pylinkage.get('joints', [])
    meta = legacy.get('meta', {})
    joints_meta = meta.get('joints', {})
    links_meta = meta.get('links', {})

    if verbose:
        print(f'  Converting legacy format: {len(joints)} joints, {len(links_meta)} links')

    # Convert joints to nodes
    nodes = {}
    node_meta = {}

    for joint in joints:
        jtype = joint.get('type', 'Static')
        name = joint.get('name', '')
        jmeta = joints_meta.get(name, {})

        # Determine position
        if jtype == 'Static':
            position = [joint.get('x', 0), joint.get('y', 0)]
        else:
            # Use meta position if available, otherwise [0, 0]
            position = [jmeta.get('x', 0), jmeta.get('y', 0)]

        # Map joint type to role
        if jtype == 'Static':
            role = 'fixed'
        elif jtype == 'Crank':
            role = 'crank'
        else:
            role = 'follower'

        # Build node
        node = {
            'id': name,
            'position': position,
            'role': role,
            'jointType': 'revolute',
            'name': name,
        }

        # Add angle for crank nodes
        if jtype == 'Crank':
            node['angle'] = joint.get('angle', 0)

        nodes[name] = node

        # Build node meta
        if jmeta:
            node_meta[name] = {
                'color': jmeta.get('color', ''),
                'zlevel': jmeta.get('zlevel', 0),
                'showPath': jmeta.get('show_path', role != 'fixed'),
            }

    # Convert links to edges
    edges = {}
    edge_meta = {}

    for link_name, link in links_meta.items():
        connects = link.get('connects', [])
        if len(connects) >= 2:
            source_id, target_id = connects[0], connects[1]

            # Calculate distance from node positions
            source_node = nodes.get(source_id, {})
            target_node = nodes.get(target_id, {})
            source_pos = source_node.get('position', [0, 0])
            target_pos = target_node.get('position', [0, 0])

            distance = np.sqrt(
                (target_pos[0] - source_pos[0]) ** 2 +
                (target_pos[1] - source_pos[1]) ** 2,
            )

            edges[link_name] = {
                'id': link_name,
                'source': source_id,
                'target': target_id,
                'distance': distance,
            }

            edge_meta[link_name] = {
                'color': link.get('color', '#888888'),
                'isGround': link.get('isGround', False),
            }

    return {
        'name': legacy.get('name', 'converted'),
        'version': '2.0.0',
        'linkage': {
            'name': pylinkage.get('name', 'converted'),
            'nodes': nodes,
            'edges': edges,
            'hyperedges': {},
        },
        'meta': {
            'nodes': node_meta,
            'edges': edge_meta,
        },
    }


# =============================================================================
# Position Computation (Circle-Circle Intersection, Crank Geometry)
# =============================================================================

def compute_joint_position(
    jdata: dict,
    joint_info: dict[str, dict],
    meta_joints: dict[str, dict],
    computed_positions: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """
    Compute the position for a joint from meta data or geometric constraints.

    Priority:
      1. meta_joints (UI-specified position)
      2. Stored x, y (for Static joints)
      3. Calculated from parent joints (Crank, Revolute)

    Args:
        jdata: Joint data dict with 'name', 'type', and constraint info
        joint_info: Map of joint_name -> joint_data for all joints
        meta_joints: Map of joint_name -> meta info (may contain UI x, y)
        computed_positions: Cache of already-computed positions (for recursion)

    Returns:
        (x, y) position tuple
    """
    if computed_positions is None:
        computed_positions = {}

    name = jdata['name']
    jtype = jdata['type']

    # Check cache first
    if name in computed_positions:
        return computed_positions[name]

    # Priority 1: Check meta for stored UI position
    if name in meta_joints:
        meta_j = meta_joints[name]
        if meta_j.get('x') is not None and meta_j.get('y') is not None:
            pos = (meta_j['x'], meta_j['y'])
            computed_positions[name] = pos
            return pos

    # Priority 2: For Static joints, use stored x, y
    if jtype == 'Static':
        pos = (jdata['x'], jdata['y'])
        computed_positions[name] = pos
        return pos

    # Priority 3: Calculate from parent joints
    if jtype == 'Crank':
        pos = _compute_crank_position(jdata, joint_info, meta_joints, computed_positions)
    elif jtype == 'Revolute':
        pos = _compute_revolute_position(jdata, joint_info, meta_joints, computed_positions)
    else:
        pos = (0.0, 0.0)

    computed_positions[name] = pos
    return pos


def _compute_crank_position(
    jdata: dict,
    joint_info: dict[str, dict],
    meta_joints: dict[str, dict],
    computed_positions: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute crank position from parent + distance + angle."""
    parent_name = jdata['joint0']['ref']
    parent_jdata = joint_info[parent_name]
    parent_pos = compute_joint_position(parent_jdata, joint_info, meta_joints, computed_positions)

    distance = jdata['distance']
    angle = jdata.get('angle', 0)

    x = parent_pos[0] + distance * np.cos(angle)
    y = parent_pos[1] + distance * np.sin(angle)
    return (x, y)


def _compute_revolute_position(
    jdata: dict,
    joint_info: dict[str, dict],
    meta_joints: dict[str, dict],
    computed_positions: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Compute revolute position via circle-circle intersection."""
    parent0_name = jdata['joint0']['ref']
    parent1_name = jdata['joint1']['ref']

    parent0_jdata = joint_info[parent0_name]
    parent1_jdata = joint_info[parent1_name]

    pos0 = compute_joint_position(parent0_jdata, joint_info, meta_joints, computed_positions)
    pos1 = compute_joint_position(parent1_jdata, joint_info, meta_joints, computed_positions)

    d0 = jdata['distance0']
    d1 = jdata['distance1']

    return circle_circle_intersection(pos0, d0, pos1, d1)


def circle_circle_intersection(
    center0: tuple[float, float],
    radius0: float,
    center1: tuple[float, float],
    radius1: float,
    prefer_positive_cross: bool = True,
) -> tuple[float, float]:
    """
    Compute intersection point of two circles.

    Returns one of the two intersection points (or midpoint if no intersection).

    Args:
        center0, radius0: First circle
        center1, radius1: Second circle
        prefer_positive_cross: If True, return the "left" intersection (positive cross product)

    Returns:
        (x, y) intersection point
    """
    dx = center1[0] - center0[0]
    dy = center1[1] - center0[1]
    d = np.sqrt(dx * dx + dy * dy)

    # Check if circles intersect
    if d == 0 or d > radius0 + radius1 or d < abs(radius0 - radius1):
        # No intersection - return midpoint as fallback
        return ((center0[0] + center1[0]) / 2, (center0[1] + center1[1]) / 2)

    # Standard circle-circle intersection
    a = (radius0 * radius0 - radius1 * radius1 + d * d) / (2 * d)
    h_sq = radius0 * radius0 - a * a
    h = np.sqrt(max(0, h_sq))

    # Point on line between centers
    px = center0[0] + (a * dx) / d
    py = center0[1] + (a * dy) / d

    # Two intersection points
    if prefer_positive_cross:
        x = px - (h * dy) / d
        y = py + (h * dx) / d
    else:
        x = px + (h * dy) / d
        y = py - (h * dx) / d

    return (x, y)


# =============================================================================
# Distance Sync (Ensure constraints match visual positions)
# =============================================================================

def sync_pylink_distances(pylink_data: dict, verbose: bool = False) -> dict:
    """
    Sync distances in a pylink_data document to match visual positions.

    This is essential when the frontend saves pylink_data with stale/incorrect
    distances that don't match the visual joint positions. This function
    recomputes distances from the visual positions and updates the pylink_data.

    Args:
        pylink_data: Full pylink document
        verbose: If True, print sync changes

    Returns:
        Updated pylink_data with synced distances (modifies in place AND returns)

    Use this before optimization to ensure starting dimensions are valid.
    """
    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])
    meta = pylink_data.get('meta', {})
    meta_joints = meta.get('joints', {})

    # Build joint info lookup
    joint_info = {jdata['name']: jdata for jdata in joints_data}

    # Compute visual positions
    visual_positions = {}
    for jdata in joints_data:
        visual_positions[jdata['name']] = compute_joint_position(jdata, joint_info, meta_joints)

    # Sync distances
    synced_joints = sync_distances_from_visual(joints_data, visual_positions, verbose=verbose)

    # Update the original pylink_data
    pylink_data['pylinkage']['joints'] = synced_joints

    return pylink_data


def sync_distances_from_visual(
    joints_data: list[dict],
    visual_positions: dict[str, tuple[float, float]],
    verbose: bool = False,
) -> list[dict]:
    """
    Update joint distance constraints to match actual visual positions.

    This fixes cases where stored distances don't match the UI positions.
    Returns a new list with updated distances (does not mutate input).

    Args:
        joints_data: List of joint data dicts
        visual_positions: Map of joint_name -> (x, y) from UI
        verbose: If True, print sync changes

    Returns:
        Updated joints_data list with synced distances
    """
    # Deep copy to avoid mutation
    import copy
    updated = copy.deepcopy(joints_data)

    for jdata in updated:
        jtype = jdata['type']
        name = jdata['name']
        my_pos = visual_positions.get(name)

        if my_pos is None:
            continue

        if jtype == 'Crank':
            parent_name = jdata['joint0']['ref']
            parent_pos = visual_positions.get(parent_name)
            if parent_pos:
                new_distance = np.sqrt(
                    (my_pos[0] - parent_pos[0])**2 +
                    (my_pos[1] - parent_pos[1])**2,
                )
                old_distance = jdata['distance']
                if abs(new_distance - old_distance) > 0.01:
                    if verbose:
                        print(f"  [SYNC] Crank '{name}': distance {old_distance:.2f} → {new_distance:.2f}")
                    jdata['distance'] = new_distance

        elif jtype == 'Revolute':
            parent0_name = jdata['joint0']['ref']
            parent1_name = jdata['joint1']['ref']
            parent0_pos = visual_positions.get(parent0_name)
            parent1_pos = visual_positions.get(parent1_name)

            if parent0_pos and parent1_pos:
                new_distance0 = np.sqrt(
                    (my_pos[0] - parent0_pos[0])**2 +
                    (my_pos[1] - parent0_pos[1])**2,
                )
                new_distance1 = np.sqrt(
                    (my_pos[0] - parent1_pos[0])**2 +
                    (my_pos[1] - parent1_pos[1])**2,
                )

                old_distance0 = jdata['distance0']
                old_distance1 = jdata['distance1']

                if abs(new_distance0 - old_distance0) > 0.01 or abs(new_distance1 - old_distance1) > 0.01:
                    if verbose:
                        print(f"  [SYNC] Revolute '{name}': d0 {old_distance0:.2f} → {new_distance0:.2f}, d1 {old_distance1:.2f} → {new_distance1:.2f}")
                    jdata['distance0'] = new_distance0
                    jdata['distance1'] = new_distance1

    return updated


# =============================================================================
# Solve Order Computation (Topological Sort)
# =============================================================================

def compute_proper_solve_order(joints_data: list[dict], verbose: bool = False) -> list[str]:
    """
    Compute proper solve order using topological sort.

    Ensures:
      1. Static joints come first (they have no dependencies)
      2. Parent joints are processed before children
      3. All dependencies are resolved before a joint is built

    Args:
        joints_data: List of joint data dicts
        verbose: If True, print sorting info

    Returns:
        Properly ordered list of joint names
    """
    joint_info = {j['name']: j for j in joints_data}

    # Build dependency graph
    # dependencies[joint_name] = set of joints that must come before it
    dependencies: dict[str, set] = {}

    for jdata in joints_data:
        name = jdata['name']
        jtype = jdata['type']

        if jtype == 'Static':
            dependencies[name] = set()  # No dependencies
        elif jtype == 'Crank':
            parent = jdata['joint0']['ref']
            dependencies[name] = {parent}
        elif jtype == 'Revolute':
            parent0 = jdata['joint0']['ref']
            parent1 = jdata['joint1']['ref']
            dependencies[name] = {parent0, parent1}
        else:
            dependencies[name] = set()

    # Topological sort using Kahn's algorithm
    # Start with joints that have no dependencies
    in_degree = {name: len(deps) for name, deps in dependencies.items()}
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        # Sort queue to ensure deterministic order (Static joints naturally come first)
        # Prioritize: Static > Crank > Revolute
        def sort_key(name):
            jtype = joint_info[name]['type']
            priority = {'Static': 0, 'Crank': 1, 'Revolute': 2}
            return (priority.get(jtype, 3), name)

        queue.sort(key=sort_key)
        current = queue.pop(0)
        result.append(current)

        # Remove current from all dependency sets
        for name, deps in dependencies.items():
            if current in deps:
                deps.remove(current)
                in_degree[name] -= 1
                if in_degree[name] == 0 and name not in result:
                    queue.append(name)

    # Check for cycles (joints that couldn't be sorted)
    if len(result) != len(joints_data):
        missing = {j['name'] for j in joints_data} - set(result)
        if verbose:
            print(f'  Warning: Could not resolve order for joints: {missing}')
        # Add remaining joints at the end
        result.extend(sorted(missing))

    if verbose:
        print(f'  Computed solve_order: {result}')

    return result


# =============================================================================
# Joint Object Building
# =============================================================================

JointObject = Union[Crank, Revolute, tuple[float, float]]  # Static joints are tuples


def build_joint_objects(
    joints_data: list[dict],
    solve_order: list[str],
    joint_info: dict[str, dict],
    meta_joints: dict[str, dict],
    angle_per_step: float,
    verbose: bool = False,
) -> dict[str, JointObject]:
    """
    Build pylinkage Joint objects from serialized data.

    Args:
        joints_data: List of joint data dicts
        solve_order: Order to build joints (respects dependencies)
        joint_info: Map of joint_name -> joint_data
        meta_joints: Map of joint_name -> meta info
        angle_per_step: Crank rotation per simulation step
        verbose: If True, print joint creation info

    Returns:
        Map of joint_name -> Joint object (or tuple for Static)
    """
    joint_objects: dict[str, JointObject] = {}
    computed_positions: dict[str, tuple[float, float]] = {}

    for joint_name in solve_order:
        if joint_name not in joint_info:
            continue

        jdata = joint_info[joint_name]
        jtype = jdata['type']
        pos = compute_joint_position(jdata, joint_info, meta_joints, computed_positions)

        if jtype == 'Static':
            # Static joints are tuple references (implicit Fixed in pylinkage)
            joint_objects[joint_name] = (jdata['x'], jdata['y'])
            if verbose:
                print(f"  Static '{joint_name}' at ({jdata['x']:.1f}, {jdata['y']:.1f})")

        elif jtype == 'Crank':
            parent_name = jdata['joint0']['ref']
            parent = joint_objects.get(parent_name)

            if parent is None:
                if verbose:
                    print(f"  Warning: Crank '{joint_name}' parent '{parent_name}' not found")
                continue

            joint_objects[joint_name] = Crank(
                x=pos[0],
                y=pos[1],
                joint0=parent,
                distance=jdata['distance'],
                angle=angle_per_step,
                name=joint_name,
            )
            if verbose:
                print(f"  Crank '{joint_name}' at ({pos[0]:.1f}, {pos[1]:.1f}), dist={jdata['distance']:.1f}")

        elif jtype == 'Revolute':
            parent0_name = jdata['joint0']['ref']
            parent1_name = jdata['joint1']['ref']
            parent0 = joint_objects.get(parent0_name)
            parent1 = joint_objects.get(parent1_name)

            if parent0 is None or parent1 is None:
                if verbose:
                    print(f"  Warning: Revolute '{joint_name}' parents not found")
                continue

            joint_objects[joint_name] = Revolute(
                x=pos[0],
                y=pos[1],
                joint0=parent0,
                joint1=parent1,
                distance0=jdata['distance0'],
                distance1=jdata['distance1'],
                name=joint_name,
            )
            if verbose:
                print(f"  Revolute '{joint_name}' at ({pos[0]:.1f}, {pos[1]:.1f}), d0={jdata['distance0']:.1f}, d1={jdata['distance1']:.1f}")

    return joint_objects


# =============================================================================
# Linkage Construction
# =============================================================================

def make_linkage(
    joint_objects: dict[str, JointObject],
    solve_order: list[str],
    name: str = 'linkage',
) -> tuple[Linkage | None, str | None]:
    """
    Build a pylinkage Linkage object from joint objects.

    Args:
        joint_objects: Map of joint_name -> Joint object
        solve_order: Order of joints
        name: Linkage name

    Returns:
        (Linkage, None) on success, (None, error_message) on failure
    """
    # Get non-static joints (only Crank and Revolute can be in Linkage)
    linkage_joints = []
    for joint_name in solve_order:
        joint = joint_objects.get(joint_name)
        if joint is not None and not isinstance(joint, tuple):
            linkage_joints.append(joint)

    if not linkage_joints:
        return None, 'No movable joints (Crank/Revolute) found. Need at least one Crank to drive the mechanism.'

    # Check for at least one Crank
    has_crank = any(isinstance(j, Crank) for j in linkage_joints)
    if not has_crank:
        return None, 'No Crank joint found. A Crank is required to drive the mechanism.'

    linkage = Linkage(
        joints=tuple(linkage_joints),
        order=tuple(linkage_joints),
        name=name,
    )

    return linkage, None


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(
    linkage: Linkage,
    joint_objects: dict[str, JointObject],
    solve_order: list[str],
    n_steps: int,
) -> tuple[dict[str, list[list[float]]], str | None]:
    """
    Run forward kinematics simulation on a linkage.

    Args:
        linkage: The pylinkage Linkage object
        joint_objects: Map of joint_name -> Joint object (includes static joints as tuples)
        solve_order: Order of joints (for consistent output)
        n_steps: Number of simulation steps

    Returns:
        (trajectories dict, None) on success, ({}, error_message) on failure

        trajectories: { joint_name: [[x0, y0], [x1, y1], ...], ... }
    """
    trajectories: dict[str, list[list[float]]] = {}

    # Initialize empty trajectories for all joints
    for joint_name in solve_order:
        trajectories[joint_name] = []

    try:
        linkage.rebuild()  # Reset to initial state

        for step, coords in enumerate(linkage.step(iterations=n_steps)):
            # coords is a list of (x, y) tuples for each joint in linkage.joints
            for joint, coord in zip(linkage.joints, coords):
                if coord[0] is not None and coord[1] is not None:
                    trajectories[joint.name].append([float(coord[0]), float(coord[1])])
                else:
                    # Use last known position or (0,0)
                    last = trajectories[joint.name][-1] if trajectories[joint.name] else [0, 0]
                    trajectories[joint.name].append(last)

            # Add static joint positions (they don't change)
            linkage_joint_names = {j.name for j in linkage.joints}
            for joint_name in solve_order:
                if joint_name not in linkage_joint_names:
                    joint = joint_objects.get(joint_name)
                    if isinstance(joint, tuple):
                        trajectories[joint_name].append([float(joint[0]), float(joint[1])])

        return trajectories, None

    except Exception as e:
        return {}, str(e)


# =============================================================================
# High-Level Orchestrator
# =============================================================================

def compute_trajectory(
    pylink_data: dict,
    verbose: bool = False,
    skip_sync: bool = False,
) -> TrajectoryResult:
    """
    Compute joint trajectories from PylinkDocument or LinkageDocument format.

    This is the main entry point - coordinates all the steps.
    Supports both legacy (pylinkage.joints) and new hypergraph (linkage.nodes/edges) formats.

    Args:
        pylink_data: Document in either format:
            - Legacy: { 'pylinkage': { 'joints': [...] }, 'meta': {...}, 'n_steps': N }
            - Hypergraph: { 'linkage': { 'nodes': {...}, 'edges': {...} }, 'meta': {...}, 'n_steps': N }
        verbose: If True, print progress
        skip_sync: If True, skip syncing distances from visual positions.
                   Use this for optimization when you want the stored distances
                   to be used directly without being overwritten by meta positions.

    Returns:
        TrajectoryResult with trajectories or error
    """
    n_steps = pylink_data.get('n_steps', 12)

    # ─────────────────────────────────────────────────────────────────────────
    # FORMAT DETECTION AND CONVERSION
    # ─────────────────────────────────────────────────────────────────────────
    if is_hypergraph_format(pylink_data):
        if verbose:
            print('  Detected hypergraph format (v2.0.0), converting to legacy...')
        converted = convert_hypergraph_to_legacy(pylink_data, verbose=verbose)

        # Check for validation issues from conversion
        validation_issues = converted.get('_validation_issues', [])
        unprocessed_nodes = converted.get('_unprocessed_nodes', [])

        if unprocessed_nodes:
            # Report underconstrained mechanism
            return TrajectoryResult(
                success=False,
                trajectories={},
                n_steps=n_steps,
                joint_types={},
                error=f"Underconstrained mechanism: nodes {unprocessed_nodes} cannot be simulated. "
                f"Each follower node needs exactly 2 edges connecting to other nodes in the kinematic chain.",
            )

        # Use converted data
        pylink_data = converted

    # Parse input (now always in legacy format)
    pylinkage_data = pylink_data.get('pylinkage', {})
    meta = pylink_data.get('meta', {})
    meta_joints = meta.get('joints', {})

    joints_data = pylinkage_data.get('joints', [])

    if not joints_data:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error='No joints found in pylinkage data',
        )

    # Compute proper solve order using topological sort
    # This ensures parents are always processed before children, regardless of
    # how the joints were created in the UI
    solve_order = compute_proper_solve_order(joints_data, verbose=verbose)

    if verbose:
        input_order = pylinkage_data.get('solve_order', [])
        if input_order != solve_order:
            print(f'  Reordered solve_order: {input_order} → {solve_order}')

    # Build joint info lookup
    joint_info = {jdata['name']: jdata for jdata in joints_data}

    # Compute visual positions (for initial positions if no sync)
    visual_positions = {}
    for jdata in joints_data:
        visual_positions[jdata['name']] = compute_joint_position(jdata, joint_info, meta_joints)

    # Sync distances to match visual positions (SKIP for optimization!)
    # This ensures UI-dragged joints have correct distances, but during optimization
    # we want to use the dimensions we set, not overwrite them from old UI positions.
    if not skip_sync:
        joints_data = sync_distances_from_visual(joints_data, visual_positions, verbose=verbose)
        joint_info = {jdata['name']: jdata for jdata in joints_data}  # Rebuild after sync

    # Calculate angle per step
    angle_per_step = 2 * np.pi / n_steps

    # Build joint objects
    joint_objects = build_joint_objects(
        joints_data, solve_order, joint_info, meta_joints, angle_per_step, verbose=verbose,
    )

    # Create linkage
    linkage, error = make_linkage(joint_objects, solve_order, pylinkage_data.get('name', 'computed'))
    if error:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error=error,
        )

    if verbose:
        print(f'  Created Linkage with {len(linkage.joints)} joints')

    # Run simulation
    trajectories, sim_error = run_simulation(linkage, joint_objects, solve_order, n_steps)
    if sim_error:
        return TrajectoryResult(
            success=False,
            trajectories={},
            n_steps=n_steps,
            joint_types={},
            error=f'Simulation failed: {sim_error}',
        )

    return TrajectoryResult(
        success=True,
        trajectories=trajectories,
        n_steps=n_steps,
        joint_types={name: joint_info[name]['type'] for name in solve_order if name in joint_info},
    )


# =============================================================================
# Mechanism Validation
# =============================================================================

def find_connected_link_groups(
    links: dict[str, dict],
    joints_data: list[dict],
) -> list[MechanismGroup]:
    """
    Find connected groups of links in the graph.

    Two links are connected if they share a joint.

    Args:
        links: meta.links dict {link_name: {connects: [joint1, joint2, ...], ...}}
        joints_data: pylinkage.joints list

    Returns:
        List of MechanismGroup, one per connected component
    """
    if not links:
        return []

    # Build joint type lookup
    joint_types = {j['name']: j['type'] for j in joints_data}

    # Build adjacency: which links share joints
    link_names = list(links.keys())
    link_joints = {name: set(data.get('connects', [])) for name, data in links.items()}

    # Union-Find for connected components
    parent = {name: name for name in link_names}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Connect links that share any joint
    for i, name1 in enumerate(link_names):
        for name2 in link_names[i+1:]:
            if link_joints[name1] & link_joints[name2]:  # shared joints
                union(name1, name2)

    # Group links by component
    components: dict[str, list[str]] = {}
    for name in link_names:
        root = find(name)
        if root not in components:
            components[root] = []
        components[root].append(name)

    # Build MechanismGroup for each component
    groups = []
    for component_links in components.values():
        # Gather all joints in this component
        component_joints = set()
        for link_name in component_links:
            component_joints.update(link_joints[link_name])

        # Check for crank and ground
        has_crank = any(joint_types.get(j) == 'Crank' for j in component_joints)
        has_ground = any(joint_types.get(j) == 'Static' for j in component_joints)

        # Validate
        error = None
        is_valid = True

        if len(component_links) < 3:
            is_valid = False
            error = f'Need at least 3 links, got {len(component_links)}'
        elif not has_crank:
            is_valid = False
            error = 'No Crank joint found - mechanism needs a driver'
        elif not has_ground:
            is_valid = False
            error = 'No Static (ground) joint found'

        groups.append(
            MechanismGroup(
                links=component_links,
                joints=list(component_joints),
                has_crank=has_crank,
                has_ground=has_ground,
                is_valid=is_valid,
                error=error,
            ),
        )

    return groups


def validate_mechanism(pylink_data: dict) -> dict[str, Any]:
    """
    Validate a pylink document and identify valid mechanism groups.

    Args:
        pylink_data: Full pylink document

    Returns:
        {
            "valid": bool,
            "groups": [MechanismGroup, ...],
            "valid_groups": [MechanismGroup, ...],  # only the valid ones
            "errors": [str, ...]
        }
    """
    pylinkage_data = pylink_data.get('pylinkage', {})
    meta = pylink_data.get('meta', {})

    joints_data = pylinkage_data.get('joints', [])
    links = meta.get('links', {})

    errors = []

    if not joints_data:
        errors.append('No joints defined')
    if not links:
        errors.append('No links defined')

    if errors:
        return {
            'valid': False,
            'groups': [],
            'valid_groups': [],
            'errors': errors,
        }

    # Find connected groups
    groups = find_connected_link_groups(links, joints_data)
    valid_groups = [g for g in groups if g.is_valid]

    # Try make_linkage on valid groups to double-check
    for group in valid_groups:
        # Filter joints to only those in this group
        group_joints = [j for j in joints_data if j['name'] in group.joints]
        joint_info = {j['name']: j for j in group_joints}
        meta_joints = meta.get('joints', {})

        # Compute proper solve order for this group using topological sort
        group_solve_order = compute_proper_solve_order(group_joints)

        # Try building
        angle_per_step = 2 * np.pi / 12
        joint_objects = build_joint_objects(
            group_joints, group_solve_order, joint_info, meta_joints, angle_per_step,
        )
        linkage, err = make_linkage(joint_objects, group_solve_order)

        if err:
            group.is_valid = False
            group.error = err

    # Refilter after make_linkage check
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


# # =============================================================================
# # Link Rigidity Check - Detect Over-Constrained Mechanisms
# # =============================================================================

# def check_link_rigidity(pylink_data: dict) -> Dict[str, Any]:
#     """
#     Check if all visual links can maintain their length during simulation.

#     A link connecting a kinematic joint (Crank/Revolute) to a Static joint
#     will "stretch" unless the static joint is kinematically connected.
#     This creates an over-constrained (locked) mechanism.

#     Args:
#         pylink_data: Full pylink document

#     Returns:
#         {
#             "valid": bool,  # True if all links can maintain length
#             "locked_links": [  # Links that would stretch/lock the mechanism
#                 {
#                     "link_name": str,
#                     "kinematic_joint": str,  # The joint that moves
#                     "static_joint": str,     # The joint that's fixed
#                     "current_distance": float,
#                     "message": str
#                 }
#             ],
#             "message": str
#         }
#     """
#     pylinkage_data = pylink_data.get('pylinkage', {})
#     meta = pylink_data.get('meta', {})

#     joints_data = pylinkage_data.get('joints', [])
#     links = meta.get('links', {})
#     meta_joints = meta.get('joints', {})

#     if not joints_data or not links:
#         return {"valid": True, "locked_links": [], "message": "No joints or links to check"}

#     # Build joint lookup
#     joint_by_name = {j['name']: j for j in joints_data}

#     # Find which joints are in the kinematic chain (have Crank or Revolute types)
#     kinematic_joints = set()
#     static_joints = set()

#     for joint in joints_data:
#         if joint['type'] in ('Crank', 'Revolute'):
#             kinematic_joints.add(joint['name'])
#         elif joint['type'] == 'Static':
#             static_joints.add(joint['name'])

#     # Also add parents of kinematic joints as "connected to chain"
#     chain_connected = kinematic_joints.copy()
#     for joint in joints_data:
#         if joint['type'] == 'Crank':
#             chain_connected.add(joint.get('joint0', {}).get('ref', ''))
#         elif joint['type'] == 'Revolute':
#             chain_connected.add(joint.get('joint0', {}).get('ref', ''))
#             chain_connected.add(joint.get('joint1', {}).get('ref', ''))

#     # Get positions for all joints
#     def get_position(jname: str) -> Tuple[float, float] | None:
#         joint = joint_by_name.get(jname)
#         if not joint:
#             return None

#         if joint['type'] == 'Static':
#             return (joint.get('x', 0), joint.get('y', 0))

#         # For kinematic joints, check meta.joints
#         mj = meta_joints.get(jname)
#         if mj and mj.get('x') is not None and mj.get('y') is not None:
#             return (mj['x'], mj['y'])

#         # Fallback for Crank
#         if joint['type'] == 'Crank':
#             parent_name = joint.get('joint0', {}).get('ref', '')
#             parent_pos = get_position(parent_name)
#             if parent_pos:
#                 x = parent_pos[0] + joint.get('distance', 0) * np.cos(joint.get('angle', 0))
#                 y = parent_pos[1] + joint.get('distance', 0) * np.sin(joint.get('angle', 0))
#                 return (x, y)

#         return None

#     # Check each link for rigidity violations
#     locked_links = []

#     for link_name, link_meta in links.items():
#         connects = link_meta.get('connects', [])
#         if len(connects) < 2:
#             continue

#         # Check pairs of connected joints
#         for i in range(len(connects) - 1):
#             j0_name = connects[i]
#             j1_name = connects[i + 1]

#             j0 = joint_by_name.get(j0_name)
#             j1 = joint_by_name.get(j1_name)

#             if not j0 or not j1:
#                 continue

#             # Check if one joint is kinematic and the other is static but NOT in the kinematic chain
#             is_j0_kinematic = j0_name in kinematic_joints
#             is_j1_kinematic = j1_name in kinematic_joints
#             is_j0_unconnected_static = j0['type'] == 'Static' and j0_name not in chain_connected
#             is_j1_unconnected_static = j1['type'] == 'Static' and j1_name not in chain_connected

#             # Problem case: kinematic joint linked to unconnected static joint
#             if (is_j0_kinematic and is_j1_unconnected_static) or (is_j1_kinematic and is_j0_unconnected_static):
#                 kin_joint = j0_name if is_j0_kinematic else j1_name
#                 static_joint = j1_name if is_j1_unconnected_static else j0_name

#                 # Get current distance
#                 pos0 = get_position(j0_name)
#                 pos1 = get_position(j1_name)

#                 if pos0 and pos1:
#                     current_dist = np.sqrt((pos1[0] - pos0[0])**2 + (pos1[1] - pos0[1])**2)
#                 else:
#                     current_dist = -1

#                 locked_links.append({
#                     "link_name": link_name,
#                     "kinematic_joint": kin_joint,
#                     "static_joint": static_joint,
#                     "current_distance": float(current_dist),
#                     "message": f"Link '{link_name}' connects moving joint '{kin_joint}' to static joint '{static_joint}'. "
#                               f"This creates an over-constrained mechanism that would lock or require the link to stretch."
#                 })

#     if locked_links:
#         return {
#             "valid": False,
#             "locked_links": locked_links,
#             "message": f"Found {len(locked_links)} link(s) that would create an over-constrained (locked) mechanism"
#         }

#     return {
#         "valid": True,
#         "locked_links": [],
#         "message": "All links can maintain their length during simulation"
#     }


# =============================================================================
# Optimization Support (Future)
# =============================================================================

# TODO: Add these functions when needed:
#
# def make_linkage_from_dimensions(
#     spec: LinkageSpec,
#     dimensions: Tuple[float, ...]
# ) -> Linkage:
#     """Create a linkage with specific dimension values (for optimization)."""
#     pass
#
#
# def evaluate_trajectory(
#     trajectories: Dict[str, List[List[float]]],
#     target_joint: str,
#     target_path: List[Tuple[float, float]],
#     metric: str = "distance"
# ) -> float:
#     """Score a trajectory against a target path (for optimization objective)."""
#     pass
#
#
# def optimize_linkage(
#     spec: LinkageSpec,
#     target_joint: str,
#     target_path: List[Tuple[float, float]],
#     method: str = "pso"  # "pso" or "trials"
# ) -> Tuple[Tuple[float, ...], float]:
#     """Run optimization to find best dimensions for target path."""
#     pass
