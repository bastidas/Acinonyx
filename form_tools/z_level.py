"""
Compute integer z-levels (layers) for each link in a mechanism.

Ensures directly connected links and links whose segments intersect over time
get different z-levels, minimizing the range and preferring small |z|.

Z-level constraints and heuristics
-----------------------------------
Assignment must satisfy **constraints**: no two conflicting entities (adjacent
or trajectory-intersecting) may share the same z-level; hard-pinned entities
must keep their z; and (when configured) all z must be >= min_z. **Heuristics**
are preferences only, expressed as weighted cost terms:

- **Reduce deltas:** Prefer adjacent-level connections (N with N±1).
- **Reduce height:** Prefer smaller total z span (max(z) - min(z)).
- **Hard pin:** Entity required at a z-level (can cause invalid assignment if
  it conflicts with another hard-pinned entity).
- **Soft pin:** Target z with weight; violation adds cost, does not invalidate.
- **Min z:** Minimum allowed z-level (default 0); all z >= min_z.
- **Role-based:** Prefer crank-incident (root) entity at a given z (default 1).
- **Sandwich bodies:** Prefer minimum |Δz| between two bodies sharing a connector; when a body
  has several connectors, nudge toward co-endpoints already assigned on other connectors.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import cast
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np

from form_tools.overlap import segments_intersect

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism


@dataclass
class ZLevelHeuristicConfig:
    """
    Config for z-level constraints and heuristic weights.

    Constraints (hard pins, min_z) must be satisfied. Heuristic weights
    are used to prefer one valid assignment over another: reduce deltas,
    reduce height, soft pins, and crank preference.
    """

    weight_reduce_deltas: float = 1.0
    """Weight for preferring adjacent-level connections (N↔N±1)."""

    weight_reduce_height: float = 0.3
    """Weight for preferring smaller max(z)−min(z)."""

    hard_pins: dict[str, int] = field(default_factory=dict)
    """Entity id → required z (must satisfy conflicts and min_z)."""

    soft_pins: dict[str, tuple[int, float]] = field(default_factory=dict)
    """Entity id → (target_z, weight). Violation adds cost, does not invalidate."""

    min_z: int = 0
    """Minimum allowed z-level; all assignments must have z >= min_z."""

    crank_z: int | None = 1
    """Preferred z for crank-incident (root) entity; None = no role-based preference."""

    weight_crank: float = 1.0
    """Weight for preferring root entity at crank_z (when crank_z is not None)."""

    weight_prefer_sandwich: float = 5.0
    """Weight for preferring connector entities between connected neighbors."""


DEFAULT_Z_LEVEL_CONFIG = ZLevelHeuristicConfig()

# Penalty scale for sandwich body pairs with |Δz| > 2 (room for connector is already satisfied).
# Full weight_prefer_sandwich would overpower reduce_deltas and push bodies to extreme z (e.g. z=5 vs z=0).
_SANDWICH_EXCESS_BODY_SPAN_FRAC = 0.35

# When a sandwich partner is not yet assigned, nudge z toward that partner's *other* assigned
# sandwich endpoint (same partner on a different connector) so multi-connector bodies can pick
# one z that satisfies both sandwiches (e.g. legzeta: rigid_group_3 vs rigid_group_0 at rigid_group_2).
_SANDWICH_UNASSIGNED_PARTNER_ALIGN_FRAC = 0.12


class _ZLevelConfigDict(TypedDict):
    weight_reduce_deltas: float
    weight_reduce_height: float
    hard_pins: dict[str, int]
    soft_pins: dict[str, tuple[int, float]]
    min_z: int
    crank_z: int | None
    weight_crank: float
    weight_prefer_sandwich: float


def config_from_request(data: dict | None) -> ZLevelHeuristicConfig | None:
    """
    Build ZLevelHeuristicConfig from a request-friendly dict (e.g. from API JSON).
    Accepts optional keys; missing keys leave defaults. Returns None if data is None or empty.
    soft_pins: entity_id -> [target_z, weight] (list) or (target_z, weight) tuple.
    hard_pins: entity_id -> int.
    """
    if not data or not isinstance(data, dict):
        return None
    out: _ZLevelConfigDict = {
        'weight_reduce_deltas': DEFAULT_Z_LEVEL_CONFIG.weight_reduce_deltas,
        'weight_reduce_height': DEFAULT_Z_LEVEL_CONFIG.weight_reduce_height,
        'hard_pins': dict(DEFAULT_Z_LEVEL_CONFIG.hard_pins),
        'soft_pins': dict(DEFAULT_Z_LEVEL_CONFIG.soft_pins),
        'min_z': DEFAULT_Z_LEVEL_CONFIG.min_z,
        'crank_z': DEFAULT_Z_LEVEL_CONFIG.crank_z,
        'weight_crank': DEFAULT_Z_LEVEL_CONFIG.weight_crank,
        'weight_prefer_sandwich': DEFAULT_Z_LEVEL_CONFIG.weight_prefer_sandwich,
    }
    if 'weight_reduce_deltas' in data and data['weight_reduce_deltas'] is not None:
        out['weight_reduce_deltas'] = float(data['weight_reduce_deltas'])
    if 'weight_reduce_height' in data and data['weight_reduce_height'] is not None:
        out['weight_reduce_height'] = float(data['weight_reduce_height'])
    if 'min_z' in data and data['min_z'] is not None:
        out['min_z'] = int(data['min_z'])
    if 'crank_z' in data:
        v = data['crank_z']
        out['crank_z'] = None if v is None or v == 'null' else int(v)
    if 'weight_crank' in data and data['weight_crank'] is not None:
        out['weight_crank'] = float(data['weight_crank'])
    if 'weight_prefer_sandwich' in data and data['weight_prefer_sandwich'] is not None:
        out['weight_prefer_sandwich'] = float(data['weight_prefer_sandwich'])
    if 'hard_pins' in data and isinstance(data['hard_pins'], dict):
        out['hard_pins'] = {str(k): int(v) for k, v in data['hard_pins'].items()}
    if 'soft_pins' in data and isinstance(data['soft_pins'], dict):
        sp: dict[str, tuple[int, float]] = {}
        for k, v in data['soft_pins'].items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                sp[str(k)] = (int(v[0]), float(v[1]))
            elif isinstance(v, (list, tuple)) and len(v) == 1:
                sp[str(k)] = (int(v[0]), 1.0)
        out['soft_pins'] = sp
    return ZLevelHeuristicConfig(
        weight_reduce_deltas=out['weight_reduce_deltas'],
        weight_reduce_height=out['weight_reduce_height'],
        hard_pins=out['hard_pins'],
        soft_pins=out['soft_pins'],
        min_z=out['min_z'],
        crank_z=out['crank_z'],
        weight_crank=out['weight_crank'],
        weight_prefer_sandwich=out['weight_prefer_sandwich'],
    )


def compute_link_z_levels(
    mechanism: Mechanism | None = None,
    path: str | Path | None = None,
    pylink_data: dict | None = None,
    n_steps: int = 32,
    use_trajectory_conflicts: bool = True,
    max_assignments: int = 10,
    drawn_objects: list[dict] | None = None,
    trajectories: dict | None = None,
    extra_entity_conflict_pairs: list[tuple[str, str]] | None = None,
    fixed_entity_z_levels: dict[str, int] | None = None,
    z_level_config: ZLevelHeuristicConfig | None = None,
) -> list[dict[str, int]] | tuple[list[dict[str, int]], dict[str, int]]:
    """
    Compute one or more valid z-level assignments for all links.

    Exactly one of mechanism, path, or pylink_data must be provided.

    Args:
        mechanism: Existing Mechanism instance (used as-is for simulation).
        path: Path to a JSON file (linkage document).
        pylink_data: Linkage document dict with linkage.nodes and linkage.edges.
        n_steps: Number of simulation steps (used when building mechanism from path/dict).
        use_trajectory_conflicts: If True, run simulation and add conflicts when
            two non-adjacent links' segments intersect at any timestep.
        max_assignments: Maximum number of valid assignments to return when
            multiple orderings exist.
        drawn_objects: Optional list of polygons with id and contained_links. When
            provided, each polygon with non-empty contained_links is one "entity"
            (all its links share one z); links not in any polygon are standalone
            entities. Return value becomes (assignments, polygon_z_levels).
        trajectories: Optional dict joint_name -> list of [x, y] for each step.
            When provided (e.g. from frontend), used for conflict detection instead
            of re-simulating, so z-order matches the same trajectory as polygon bounds.
        extra_entity_conflict_pairs: Optional list of (entity_id, entity_id) that
            must have different z-levels.
        fixed_entity_z_levels: Hard pin: entity_id -> required z. If two conflicting
            entities share the same fixed z, assignment is invalid (ValueError).
            Merged with z_level_config.hard_pins when both are provided.
        z_level_config: Optional config for heuristics and constraints (hard_pins,
            soft_pins, min_z, crank_z, weights). When None, DEFAULT_Z_LEVEL_CONFIG
            is used. Supplies hard_pins, soft_pins, min_z, crank_z, and weights;
            cost logic using it is applied in a follow-up phase.

    Returns:
        If drawn_objects is None: list of assignments (each dict link_id -> z_level).
        If drawn_objects is provided: (assignments, polygon_z_levels) for the first
        assignment only; polygon_z_levels maps polygon_id -> z_level.

    Raises:
        ValueError: If zero or more than one of mechanism/path/pylink_data is set.
    """
    config = z_level_config or DEFAULT_Z_LEVEL_CONFIG
    effective_fixed = dict(config.hard_pins)
    if fixed_entity_z_levels:
        effective_fixed.update(fixed_entity_z_levels)

    mech, edges_list, joint_names, nodes_roles = _normalize_input(
        mechanism=mechanism,
        path=path,
        pylink_data=pylink_data,
        n_steps=n_steps,
    )
    if not edges_list:
        if drawn_objects:
            return ([{}], {})
        return [{}]

    link_ids = [e[0] for e in edges_list]
    link_conflict_graph = _build_conflict_graph(
        edges_list=edges_list,
        joint_names=joint_names,
        nodes_roles=nodes_roles,
        mechanism=mech if (use_trajectory_conflicts and not trajectories) else None,
        n_steps=n_steps,
        trajectory_override=trajectories,
    )

    if drawn_objects:
        assignments, polygon_z_levels = _compute_z_levels_with_polygons(
            edges_list=edges_list,
            link_ids=link_ids,
            link_conflict_graph=link_conflict_graph,
            nodes_roles=nodes_roles,
            drawn_objects=drawn_objects,
            max_assignments=1,
            extra_entity_conflict_pairs=extra_entity_conflict_pairs or [],
            fixed_entity_z_levels=effective_fixed or None,
            z_level_config=config,
        )
        return (assignments, polygon_z_levels)

    assignments = _assign_z_levels(
        link_ids=link_ids,
        conflict_graph=link_conflict_graph,
        edges_list=edges_list,
        nodes_roles=nodes_roles,
        max_assignments=max_assignments,
        fixed_assignments=effective_fixed or None,
        z_level_config=config,
    )
    return assignments


def _normalize_input(
    mechanism: Mechanism | None = None,
    path: str | Path | None = None,
    pylink_data: dict | None = None,
    n_steps: int = 32,
) -> tuple[
    Mechanism | None,
    list[tuple[str, str, str]],
    list[str],
    dict[str, str],
]:
    """
    Return (mechanism or None, list of (link_id, source_joint, target_joint), joint_names, nodes_roles).
    If path or pylink_data is given, mechanism is built and returned; otherwise mechanism is used.
    """
    n_sources = sum(1 for x in (mechanism, path, pylink_data) if x is not None)
    if n_sources == 0:
        raise ValueError('Exactly one of mechanism, path, or pylink_data must be provided')
    if n_sources > 1:
        raise ValueError('Only one of mechanism, path, or pylink_data may be provided')

    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        with open(path) as f:
            pylink_data = json.load(f)
        mechanism = None

    if pylink_data is not None:
        from pylink_tools.hypergraph_adapter import sync_hypergraph_distances
        from pylink_tools.mechanism import create_mechanism_from_dict

        if 'linkage' in pylink_data and 'nodes' in pylink_data.get('linkage', {}):
            pylink_data = sync_hypergraph_distances(pylink_data, verbose=False)
        data = dict(pylink_data)
        data['n_steps'] = n_steps
        mechanism = create_mechanism_from_dict(data, n_steps=n_steps)

    # mechanism is set here (either passed in or built from path/dict)
    assert mechanism is not None
    meta = getattr(mechanism, '_metadata', None) or {}
    original_edges = meta.get('_original_edges', {})
    original_nodes = meta.get('_original_nodes', {})

    if not original_edges:
        joint_names = list(mechanism._joint_names)
        nodes_roles = {n: 'follower' for n in joint_names}
        return mechanism, [], joint_names, nodes_roles

    edges_list = []
    for link_id, edge in original_edges.items():
        src = edge.get('source')
        tgt = edge.get('target')
        if src is not None and tgt is not None:
            edges_list.append((link_id, src, tgt))

    joint_names = list(mechanism._joint_names)
    nodes_roles = {}
    for nid, node in original_nodes.items():
        nodes_roles[nid] = (node.get('role') or 'follower').lower()

    return mechanism, edges_list, joint_names, nodes_roles


def _build_entity_graph(
    drawn_objects: list[dict],
    link_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, list[str]], list[str]]:
    """Build polygon_links, link_to_entity, entity_to_links, entity_ids from drawn_objects."""
    polygon_links: dict[str, list[str]] = {}
    for obj in drawn_objects:
        if obj.get('type') != 'polygon':
            continue
        pid = obj.get('id')
        contained = obj.get('contained_links') or []
        if not pid or not contained:
            continue
        polygon_links[pid] = [lid for lid in contained if lid in link_ids]

    link_to_entity: dict[str, str] = {}
    entity_to_links: dict[str, list[str]] = {}
    for pid, lids in polygon_links.items():
        if not lids:
            continue
        eid = 'polygon:' + pid
        entity_to_links[eid] = lids
        for lid in lids:
            link_to_entity[lid] = eid
    for lid in link_ids:
        if lid not in link_to_entity:
            link_to_entity[lid] = lid
            entity_to_links[lid] = [lid]

    entity_ids = list(entity_to_links.keys())
    return polygon_links, link_to_entity, entity_to_links, entity_ids


def _build_entity_conflict_and_structural(
    entity_ids: list[str],
    entity_to_links: dict[str, list[str]],
    link_to_entity: dict[str, str],
    link_conflict_graph: dict[str, set[str]],
    edges_list: list[tuple[str, str, str]],
    extra_entity_conflict_pairs: list[tuple[str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build entity conflict graph and structural neighbors (entities that share a link joint)."""
    entity_conflict: dict[str, set[str]] = {e: set() for e in entity_ids}
    for e1 in entity_ids:
        for e2 in entity_ids:
            if e1 >= e2:
                continue
            for l1 in entity_to_links[e1]:
                for l2 in entity_to_links[e2]:
                    if l2 in link_conflict_graph.get(l1, set()):
                        entity_conflict[e1].add(e2)
                        entity_conflict[e2].add(e1)
                        break
                else:
                    continue
                break

    for e1, e2 in extra_entity_conflict_pairs:
        if e1 in entity_conflict and e2 in entity_conflict and e1 != e2:
            entity_conflict[e1].add(e2)
            entity_conflict[e2].add(e1)

    link_to_endpoints = {e[0]: (e[1], e[2]) for e in edges_list}
    joint_to_links: dict[str, list[str]] = {}
    for lid, sa, ta in edges_list:
        joint_to_links.setdefault(sa, []).append(lid)
        joint_to_links.setdefault(ta, []).append(lid)
    entity_structural: dict[str, set[str]] = {e: set() for e in entity_ids}
    for e1 in entity_ids:
        for lid in entity_to_links[e1]:
            source_joint, target_joint = cast(
                tuple[str | None, str | None],
                link_to_endpoints.get(lid, (None, None)),
            )
            if source_joint is None:
                continue
            neighbor_links = list(joint_to_links.get(source_joint, []))
            if target_joint is not None:
                neighbor_links.extend(joint_to_links.get(target_joint, []))
            for neighbor_link in neighbor_links:
                neighbor_entity: str | None = link_to_entity.get(neighbor_link)
                if neighbor_entity and neighbor_entity != e1:
                    entity_structural[e1].add(neighbor_entity)

    return entity_conflict, entity_structural


def _entity_bfs_order(
    entity_ids: list[str],
    entity_structural: dict[str, set[str]],
    root_entity: str,
) -> list[str]:
    """Return entity IDs in BFS order from root_entity; unreached entities appended at end."""
    order: list[str] = []
    seen: set[str] = set()
    queue = [root_entity]
    while queue:
        eid = queue.pop(0)
        if eid in seen:
            continue
        seen.add(eid)
        order.append(eid)
        for neighbor in entity_structural.get(eid, set()):
            if neighbor not in seen:
                queue.append(neighbor)
    for eid in entity_ids:
        if eid not in seen:
            order.append(eid)
    return order


def _entity_order_respecting_sandwich_prereqs(
    entity_ids: list[str],
    base_order: list[str],
    sandwich_pairs: dict[str, list[tuple[str, str]]],
    *,
    entity_structural: dict[str, set[str]] | None = None,
) -> list[str]:
    """
    Topological order so connector entities are assigned after both neighbors
    in each sandwich pair; ties follow base_order (BFS). Falls back to base_order on cycle.

    When entity_structural is set, sandwich *body* pairs (the two flanking entities) are
    ordered so the structurally simpler body (fewer neighbors) is assigned first; the
    second body can then use sandwich span cost against the first. Ties use base_order.
    """
    entity_set = set(entity_ids)
    pos = {e: i for i, e in enumerate(base_order)}
    prereqs: dict[str, set[str]] = {e: set() for e in entity_ids}
    for eid, pairs in sandwich_pairs.items():
        if eid not in entity_set or not pairs:
            continue
        for n1, n2 in pairs:
            if n1 in entity_set and n2 in entity_set and n1 != eid and n2 != eid:
                prereqs[eid].add(n1)
                prereqs[eid].add(n2)
    if entity_structural is not None:
        for eid, pairs in sandwich_pairs.items():
            if eid not in entity_set or not pairs:
                continue
            for n1, n2 in pairs:
                if n1 not in entity_set or n2 not in entity_set or n1 == n2:
                    continue
                d1 = len(entity_structural.get(n1, ()))
                d2 = len(entity_structural.get(n2, ()))
                if d1 < d2:
                    prereqs[n2].add(n1)
                elif d2 < d1:
                    prereqs[n1].add(n2)
                elif pos[n1] < pos[n2]:
                    prereqs[n2].add(n1)
                else:
                    prereqs[n1].add(n2)
    graph: dict[str, list[str]] = {e: [] for e in entity_ids}
    in_degree = {e: len(prereqs[e]) for e in entity_ids}
    for eid, ps in prereqs.items():
        for p in ps:
            graph[p].append(eid)
    ind = dict(in_degree)
    heap: list[tuple[int, str]] = [(pos[e], e) for e in entity_ids if ind[e] == 0]
    heapq.heapify(heap)
    out: list[str] = []
    while heap:
        _, e = heapq.heappop(heap)
        out.append(e)
        for succ in graph[e]:
            ind[succ] -= 1
            if ind[succ] == 0:
                heapq.heappush(heap, (pos[succ], succ))
    if len(out) != len(entity_ids):
        return base_order
    return out


def _validate_fixed_entity_z_levels(
    fixed_entity_z_levels: dict[str, int],
    entity_conflict: dict[str, set[str]],
) -> None:
    """Raise ValueError if two conflicting fixed entities share the same z-level."""
    for eid, z in fixed_entity_z_levels.items():
        for other in entity_conflict.get(eid, set()):
            if other in fixed_entity_z_levels and fixed_entity_z_levels[other] == z:
                raise ValueError(
                    'Z-level assignment impossible with current fixed layers; '
                    'two conflicting forms share the same z-level. Try unfixing some forms or changing their order.',
                )


def _compute_z_levels_with_polygons(
    edges_list: list[tuple[str, str, str]],
    link_ids: list[str],
    link_conflict_graph: dict[str, set[str]],
    nodes_roles: dict[str, str],
    drawn_objects: list[dict],
    max_assignments: int,
    extra_entity_conflict_pairs: list[tuple[str, str]] | None = None,
    fixed_entity_z_levels: dict[str, int] | None = None,
    z_level_config: ZLevelHeuristicConfig | None = None,
) -> tuple[list[dict[str, int]], dict[str, int]]:
    """
    Build entity-level conflict graph (polygon = one entity, standalone link = one entity),
    assign z to entities, return link_id -> z and polygon_id -> z.
    """
    polygon_links, link_to_entity, entity_to_links, entity_ids = _build_entity_graph(
        drawn_objects, link_ids,
    )
    entity_conflict, entity_structural = _build_entity_conflict_and_structural(
        entity_ids,
        entity_to_links,
        link_to_entity,
        link_conflict_graph,
        edges_list,
        extra_entity_conflict_pairs or [],
    )

    crank_nodes = {n for n, r in nodes_roles.items() if r == 'crank'}
    root_link = None
    for lid, sa, ta in edges_list:
        if sa in crank_nodes or ta in crank_nodes:
            root_link = lid
            break
    root_link = root_link or link_ids[0]
    root_entity = link_to_entity[root_link]

    entity_order_bfs = _entity_bfs_order(entity_ids, entity_structural, root_entity)
    link_to_endpoints = {e[0]: (e[1], e[2]) for e in edges_list}
    joint_to_links: dict[str, list[str]] = {}
    for lid, sa, ta in edges_list:
        joint_to_links.setdefault(sa, []).append(lid)
        joint_to_links.setdefault(ta, []).append(lid)
    sandwich_pairs: dict[str, list[tuple[str, str]]] = {eid: [] for eid in entity_ids}
    for eid, lids in entity_to_links.items():
        # Hard connector constraints apply only to single-link polygon forms.
        if len(lids) != 1 or not eid.startswith('polygon:'):
            continue
        lid = lids[0]
        sa, ta = link_to_endpoints.get(lid, (None, None))
        if sa is None or ta is None:
            continue
        left_entities = {
            link_to_entity[nl]
            for nl in joint_to_links.get(sa, [])
            if nl != lid and nl in link_to_entity and link_to_entity[nl] != eid
        }
        right_entities = {
            link_to_entity[nl]
            for nl in joint_to_links.get(ta, [])
            if nl != lid and nl in link_to_entity and link_to_entity[nl] != eid
        }
        if not left_entities or not right_entities:
            continue
        for left in left_entities:
            for right in right_entities:
                if left != right:
                    sandwich_pairs[eid].append((left, right))

    entity_order = _entity_order_respecting_sandwich_prereqs(
        entity_ids,
        entity_order_bfs,
        sandwich_pairs,
        entity_structural=entity_structural,
    )

    if fixed_entity_z_levels:
        _validate_fixed_entity_z_levels(fixed_entity_z_levels, entity_conflict)

    entity_assignments = _assign_entity_z_levels(
        entity_ids=entity_order,
        conflict_graph=entity_conflict,
        structural_neighbors=entity_structural,
        max_assignments=max_assignments,
        fixed_assignments=fixed_entity_z_levels,
        z_level_config=z_level_config,
        sandwich_pairs=sandwich_pairs,
        hard_connector_between_pairs=sandwich_pairs,
        crank_preferred_entity=root_entity,
    )
    if not entity_assignments:
        raise ValueError(
            'Z-level assignment impossible with current fixed layers; '
            'try unfixing some forms or changing their order.',
        )

    ass = entity_assignments[0]
    link_assignments = {lid: ass[link_to_entity[lid]] for lid in link_ids}
    polygon_z_levels = {}
    for pid in polygon_links:
        eid = 'polygon:' + pid
        if eid in ass:
            polygon_z_levels[pid] = ass[eid]
    return ([link_assignments], polygon_z_levels)


def _z_level_palette(min_z: int | None = None):
    """
    Yield candidate z-levels. If min_z is not None, yield only z >= min_z
    in order min_z, min_z+1, min_z+2, ... Otherwise yield 0, 1, -1, 2, -2, ...
    for minimal range.
    """
    if min_z is not None:
        k = 0
        while True:
            yield min_z + k
            k += 1
        return
    yield 0
    k = 1
    while True:
        yield k
        yield -k
        k += 1


def _candidate_z_levels(
    eid: str,
    assignment: dict[str, int],
    used_by_neighbors: set[int],
    structural_neighbors: dict[str, set[str]],
    min_z_for_palette: int | None,
    max_candidates: int,
) -> list[int]:
    """
    Build candidate z values for eid: include z adjacent to assigned
    structural neighbors (so pinned/high z get neighbors at z±1), then
    fill from palette. Ensures we consider 9 and 11 when a neighbor is at 10.
    """
    min_z = min_z_for_palette if min_z_for_palette is not None else 0
    candidates: list[int] = []
    seen: set[int] = set()

    def add(z: int) -> bool:
        if z < min_z or z in used_by_neighbors or z in seen:
            return False
        seen.add(z)
        candidates.append(z)
        return len(candidates) >= max_candidates

    # First: z adjacent to structural neighbors (prefer N±1 next to assigned)
    for n in structural_neighbors.get(eid, set()):
        if n not in assignment:
            continue
        zn = assignment[n]
        if add(zn - 1) or add(zn + 1):
            return candidates

    # Then: fill from palette
    for z in _z_level_palette(min_z=min_z_for_palette):
        if add(z):
            break

    return candidates


def _partner_z_allowed_for_conflicts(
    partner_z: int,
    partner_eid: str,
    conflict_graph: dict[str, set[str]],
    assignment: dict[str, int],
    min_z: int,
) -> bool:
    if partner_z < min_z:
        return False
    for c in conflict_graph.get(partner_eid, set()):
        if c in assignment and assignment[c] == partner_z:
            return False
    return True


def _has_feasible_tight_sandwich_z_for_partner(
    z_self: int,
    partner_eid: str,
    conflict_graph: dict[str, set[str]],
    assignment: dict[str, int],
    min_z: int,
) -> bool:
    """True if partner could later be placed at z_self±2 without conflicting assigned entities."""
    for zb in (z_self - 2, z_self + 2):
        if _partner_z_allowed_for_conflicts(zb, partner_eid, conflict_graph, assignment, min_z):
            return True
    return False


def _assign_z_levels_generic(
    order: list[str],
    conflict_graph: dict[str, set[str]],
    structural_neighbors: dict[str, set[str]],
    max_assignments: int,
    fixed_assignments: dict[str, int] | None = None,
    z_level_config: ZLevelHeuristicConfig | None = None,
    sandwich_pairs: dict[str, list[tuple[str, str]]] | None = None,
    hard_connector_between_pairs: dict[str, list[tuple[str, str]]] | None = None,
    crank_preferred_entity: str | None = None,
) -> list[dict[str, int]]:
    """
    Return up to max_assignments valid z-level assignments for the given order.
    Uses palette (respecting min_z when set) and total cost: delta, height,
    soft pin, and crank preference per z_level_config.
    """
    cfg = z_level_config or DEFAULT_Z_LEVEL_CONFIG
    fixed = fixed_assignments or {}
    for eid, z in fixed.items():
        if z < cfg.min_z:
            raise ValueError(
                f'Hard-pinned entity {eid} has z={z} but min_z={cfg.min_z}; '
                'all z-levels must be >= min_z.',
            )
    results: list[dict[str, int]] = []
    initial_assignment = dict(fixed)
    initial_used = set(fixed.values())
    # When min_z is set (including 0), use ascending palette so all z >= min_z
    min_z_for_palette: int | None = cfg.min_z
    connector_pairs = sandwich_pairs or {}
    hard_connector_pairs = hard_connector_between_pairs or {}
    # Endpoints that sandwich the same connector (not the connector id itself).
    sandwich_body_neighbors: dict[str, set[str]] = {}
    for pairs in connector_pairs.values():
        for a, b in pairs:
            sandwich_body_neighbors.setdefault(a, set()).add(b)
            sandwich_body_neighbors.setdefault(b, set()).add(a)

    def _violates_hard_connector_between(
        z: int,
        eid: str,
        assignment: dict[str, int],
    ) -> bool:
        for n1, n2 in hard_connector_pairs.get(eid, []):
            if n1 not in assignment or n2 not in assignment:
                continue
            lo = min(assignment[n1], assignment[n2])
            hi = max(assignment[n1], assignment[n2])
            if z < lo or z > hi:
                return True
        return False

    def total_cost(z: int, eid: str, assignment: dict[str, int]) -> float:
        cost = 0.0
        # Delta: prefer adjacent levels to structural neighbors
        if cfg.weight_reduce_deltas != 0:
            delta = 0
            for n in structural_neighbors.get(eid, set()):
                if n in assignment:
                    delta += abs(z - assignment[n])
            cost += cfg.weight_reduce_deltas * delta
        # Soft pin: prefer target z for this entity
        if eid in cfg.soft_pins:
            target_z, w = cfg.soft_pins[eid]
            cost += w * abs(z - target_z)
        # Crank: prefer the designated crank-preferred entity at crank_z.
        crank_entity = crank_preferred_entity if crank_preferred_entity is not None else (order[0] if order else None)
        if crank_entity is not None and eid == crank_entity and cfg.crank_z is not None and cfg.weight_crank != 0:
            cost += cfg.weight_crank * abs(z - cfg.crank_z)
        # Sandwich: prefer connector entity z within connected-neighbor interval
        if cfg.weight_prefer_sandwich != 0:
            for n1, n2 in connector_pairs.get(eid, []):
                if n1 not in assignment or n2 not in assignment:
                    continue
                z1 = assignment[n1]
                z2 = assignment[n2]
                lo = min(z1, z2)
                hi = max(z1, z2)
                if z < lo:
                    cost += cfg.weight_prefer_sandwich * (lo - z)
                elif z > hi:
                    cost += cfg.weight_prefer_sandwich * (z - hi)
                else:
                    cost += cfg.weight_prefer_sandwich * 0.1 * min(z - lo, hi - z)
        # Sandwich bodies: two entities flanking the same connector need |Δz| >= 2 so the
        # connector can sit strictly between; prefer the minimum (2), penalize excess span.
        if cfg.weight_prefer_sandwich != 0:
            for other in sandwich_body_neighbors.get(eid, set()):
                if other in assignment:
                    zo = assignment[other]
                    span = abs(z - zo)
                    if span < 2:
                        cost += cfg.weight_prefer_sandwich * float(2 - span)
                    elif span > 2:
                        cost += (
                            cfg.weight_prefer_sandwich
                            * _SANDWICH_EXCESS_BODY_SPAN_FRAC
                            * float(span - 2)
                        )
                elif not _has_feasible_tight_sandwich_z_for_partner(
                    z,
                    other,
                    conflict_graph,
                    assignment,
                    cfg.min_z,
                ):
                    # Partner not yet assigned: avoid z where both z±2 are blocked for them.
                    cost += cfg.weight_prefer_sandwich * 2.0
                else:
                    # Partner still unassigned: prefer z near other sandwich co-endpoints of that partner.
                    seen_co: set[tuple[str, str]] = set()
                    for pairs in connector_pairs.values():
                        for a, b in pairs:
                            if a == other and b != eid:
                                anchor = b
                            elif b == other and a != eid:
                                anchor = a
                            else:
                                continue
                            if anchor not in assignment:
                                continue
                            key = (other, anchor)
                            if key in seen_co:
                                continue
                            seen_co.add(key)
                            cost += (
                                _SANDWICH_UNASSIGNED_PARTNER_ALIGN_FRAC
                                * cfg.weight_prefer_sandwich
                                * abs(z - assignment[anchor])
                            )
        # Height: prefer smaller span (max - min over assignment including z)
        if cfg.weight_reduce_height != 0 and assignment:
            vals = set(assignment.values()) | {z}
            span = max(vals) - min(vals)
            cost += cfg.weight_reduce_height * span
        elif cfg.weight_reduce_height != 0:
            cost += 0.0  # single value, no span
        return cost

    def backtrack(idx: int, assignment: dict[str, int], used_vals: set[int]) -> None:
        if len(results) >= max_assignments:
            return
        if idx == len(order):
            results.append(dict(assignment))
            return
        eid = order[idx]
        if eid in fixed:
            backtrack(idx + 1, assignment, used_vals)
            return
        neighbors = conflict_graph.get(eid, set())
        used_by_neighbors = {assignment[n] for n in neighbors if n in assignment}
        choices = _candidate_z_levels(
            eid,
            assignment,
            used_by_neighbors,
            structural_neighbors,
            min_z_for_palette,
            max_candidates=5,
        )
        choices.sort(key=lambda z: total_cost(z, eid, assignment))
        for z in choices:
            if _violates_hard_connector_between(z, eid, assignment):
                continue
            assignment[eid] = z
            used_vals.add(z)
            backtrack(idx + 1, assignment, used_vals)
            if len(results) >= max_assignments:
                return
            del assignment[eid]
            used_vals.discard(z)

    backtrack(0, initial_assignment, initial_used)
    if not results:
        assignment = dict(initial_assignment)
        used_vals = set(initial_used)
        for eid in order:
            if eid in fixed:
                continue
            neighbors = conflict_graph.get(eid, set())
            used = {assignment[n] for n in neighbors if n in assignment}
            candidates = _candidate_z_levels(
                eid,
                assignment,
                used,
                structural_neighbors,
                min_z_for_palette,
                max_candidates=10,
            )
            candidates.sort(key=lambda z: total_cost(z, eid, assignment))
            candidates = [z for z in candidates if not _violates_hard_connector_between(z, eid, assignment)]
            if not candidates:
                return []
            assignment[eid] = candidates[0]
            used_vals.add(candidates[0])
        results.append(assignment)
    return results


def _assign_entity_z_levels(
    entity_ids: list[str],
    conflict_graph: dict[str, set[str]],
    structural_neighbors: dict[str, set[str]],
    max_assignments: int,
    fixed_assignments: dict[str, int] | None = None,
    z_level_config: ZLevelHeuristicConfig | None = None,
    sandwich_pairs: dict[str, list[tuple[str, str]]] | None = None,
    hard_connector_between_pairs: dict[str, list[tuple[str, str]]] | None = None,
    crank_preferred_entity: str | None = None,
) -> list[dict[str, int]]:
    """Assign z-levels to entities (polygon or link IDs). Uses generic assign with optional fixed pins."""
    return _assign_z_levels_generic(
        order=entity_ids,
        conflict_graph=conflict_graph,
        structural_neighbors=structural_neighbors,
        max_assignments=max_assignments,
        fixed_assignments=fixed_assignments,
        z_level_config=z_level_config,
        sandwich_pairs=sandwich_pairs,
        hard_connector_between_pairs=hard_connector_between_pairs,
        crank_preferred_entity=crank_preferred_entity,
    )


def _build_conflict_graph(
    edges_list: list[tuple[str, str, str]],
    joint_names: list[str],
    nodes_roles: dict[str, str],
    mechanism: Mechanism | None,
    n_steps: int,
    trajectory_override: dict | None = None,
) -> dict[str, set[str]]:
    """
    Build conflict graph: link_id -> set of link_ids that must have different z-level.
    Adjacent links (share a joint) always conflict. If mechanism or trajectory_override
    is set, add conflicts for non-adjacent links whose segments intersect at any timestep.
    When trajectory_override is provided (joint_name -> list of [x,y]), use it instead of
    simulating so conflict detection matches the same trajectory as polygon bounds.
    """
    joint_to_index = {name: i for i, name in enumerate(joint_names)}
    # Adjacency: links that share a joint
    link_ids = [e[0] for e in edges_list]
    conflict: dict[str, set[str]] = {lid: set() for lid in link_ids}
    for i, (lid_a, sa, ta) in enumerate(edges_list):
        for j, (lid_b, sb, tb) in enumerate(edges_list):
            if i >= j:
                continue
            if sa == sb or sa == tb or ta == sb or ta == tb:
                conflict[lid_a].add(lid_b)
                conflict[lid_b].add(lid_a)

    if trajectory_override and joint_names:
        # Use provided trajectory (e.g. from frontend) so z-order matches polygon bounds
        lengths = [len(p) for p in trajectory_override.values() if p]
        n_steps_actual = min(lengths) if lengths else 0
        if n_steps_actual == 0:
            return conflict
        trajectory = np.zeros((n_steps_actual, len(joint_names), 2), dtype=float)
        for step in range(n_steps_actual):
            for i, jname in enumerate(joint_names):
                if jname in trajectory_override and step < len(trajectory_override[jname]):
                    trajectory[step, i, :] = trajectory_override[jname][step]
    elif mechanism is not None:
        try:
            trajectory = mechanism.simulate()
        except Exception:
            return conflict
    else:
        return conflict

    # trajectory shape (n_steps, n_joints, 2)
    for step in range(trajectory.shape[0]):
        for i, (lid_a, sa, ta) in enumerate(edges_list):
            ia = joint_to_index.get(sa)
            ib = joint_to_index.get(ta)
            if ia is None or ib is None:
                continue
            a1 = trajectory[step, ia, :]
            a2 = trajectory[step, ib, :]
            for j, (lid_b, sb, tb) in enumerate(edges_list):
                if i >= j:
                    continue
                if lid_b in conflict[lid_a]:
                    continue
                ja = joint_to_index.get(sb)
                jb = joint_to_index.get(tb)
                if ja is None or jb is None:
                    continue
                b1 = trajectory[step, ja, :]
                b2 = trajectory[step, jb, :]
                if segments_intersect(
                    a1, a2, b1, b2,
                    exclude_shared_endpoints=True,
                ):
                    conflict[lid_a].add(lid_b)
                    conflict[lid_b].add(lid_a)
    return conflict


def _assign_z_levels(
    link_ids: list[str],
    conflict_graph: dict[str, set[str]],
    edges_list: list[tuple[str, str, str]],
    nodes_roles: dict[str, str],
    max_assignments: int,
    fixed_assignments: dict[str, int] | None = None,
    z_level_config: ZLevelHeuristicConfig | None = None,
) -> list[dict[str, int]]:
    """
    Return up to max_assignments valid colorings. Palette order: 0, 1, -1, 2, -2, ...
    Order links by BFS from a crank-incident link so we "step back" along chains.
    """
    if not link_ids:
        return [{}]

    # Crank node(s) and root link for BFS
    crank_nodes = {n for n, r in nodes_roles.items() if r == 'crank'}
    root_link = None
    for lid, sa, ta in edges_list:
        if sa in crank_nodes or ta in crank_nodes:
            root_link = lid
            break
    if root_link is None:
        root_link = link_ids[0]

    link_to_endpoints = {e[0]: (e[1], e[2]) for e in edges_list}
    joint_to_links: dict[str, list[str]] = {}
    for lid, sa, ta in edges_list:
        joint_to_links.setdefault(sa, []).append(lid)
        joint_to_links.setdefault(ta, []).append(lid)

    order = []
    seen = set()
    queue = [root_link]
    while queue:
        lid = queue.pop(0)
        if lid in seen:
            continue
        seen.add(lid)
        order.append(lid)
        sa, ta = link_to_endpoints[lid]
        for neighbor in (joint_to_links.get(sa, []) + joint_to_links.get(ta, [])):
            if neighbor not in seen:
                queue.append(neighbor)
    for lid in link_ids:
        if lid not in seen:
            order.append(lid)

    structural_neighbors: dict[str, set[str]] = {}
    for lid, sa, ta in edges_list:
        structural_neighbors[lid] = (
            set(joint_to_links.get(sa, [])) | set(joint_to_links.get(ta, []))
        ) - {lid}
    sandwich_pairs: dict[str, list[tuple[str, str]]] = {lid: [] for lid in link_ids}
    for lid, sa, ta in edges_list:
        left_links = {n for n in joint_to_links.get(sa, []) if n != lid}
        right_links = {n for n in joint_to_links.get(ta, []) if n != lid}
        if not left_links or not right_links:
            continue
        for left in left_links:
            for right in right_links:
                if left != right:
                    sandwich_pairs[lid].append((left, right))

    order = _entity_order_respecting_sandwich_prereqs(
        link_ids,
        order,
        sandwich_pairs,
        entity_structural=structural_neighbors,
    )

    return _assign_z_levels_generic(
        order=order,
        conflict_graph=conflict_graph,
        structural_neighbors=structural_neighbors,
        max_assignments=max_assignments,
        fixed_assignments=fixed_assignments,
        z_level_config=z_level_config,
        sandwich_pairs=sandwich_pairs,
        crank_preferred_entity=root_link,
    )
