"""
topology_validity.py - Topology identity, bad-topology cache, and validity checks.

Provides:
  - topology_id_from_edges: Canonical topology id (frozenset of node pairs)
  - BadTopologyCache: Store and query invalid/locked topologies
  - is_grashof_compliant: Four-bar Grashof condition (s + l <= p + q)
  - check_mechanism_validity: Grashof (if 4-bar) + simulation; optionally record bad
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism


def topology_id_from_edges(edges: dict) -> frozenset[tuple[str, str]]:
    """
    Return a canonical topology id from an edges dict.

    Same graph with different edge IDs or order maps to the same id.
    Each edge is represented as (min(source, target), max(source, target)).

    Args:
        edges: Dict mapping edge_id -> {source, target, ...} (e.g. linkage.edges)

    Returns:
        Frozenset of (node_a, node_b) with node_a <= node_b for each edge
    """
    pairs: set[tuple[str, str]] = set()
    for edge in edges.values():
        if not isinstance(edge, dict):
            continue
        src = edge.get('source')
        tgt = edge.get('target')
        if src is None or tgt is None:
            continue
        a, b = str(src).strip(), str(tgt).strip()
        pairs.add((min(a, b), max(a, b)))
    return frozenset(pairs)


class BadTopologyCache:
    """
    In-memory cache of topologies that are invalid (locked, non-Grashof, etc.).

    Use one instance per optimization run. Before building or evaluating
    a topology, call is_bad(topology_id); if True, skip. When a topology
    fails validity, call add_bad(topology_id, reason).
    """

    def __init__(self) -> None:
        self._bad: dict[frozenset[tuple[str, str]], str] = {}

    def add_bad(self, topology_id: frozenset[tuple[str, str]], reason: str) -> None:
        """Record a topology as invalid."""
        self._bad[topology_id] = reason

    def is_bad(self, topology_id: frozenset[tuple[str, str]]) -> bool:
        """Return True if this topology was previously marked bad."""
        return topology_id in self._bad

    def get_reason(self, topology_id: frozenset[tuple[str, str]]) -> str | None:
        """Return the reason a topology was marked bad, or None if not bad."""
        return self._bad.get(topology_id)

    def __len__(self) -> int:
        return len(self._bad)

    def to_report_list(self) -> list[dict]:
        """List of {topology_id_repr, reason} for API/logging."""
        return [
            {'topology_id': repr(tid), 'reason': reason}
            for tid, reason in self._bad.items()
        ]


def is_grashof_compliant(link_lengths: tuple[float, ...] | list[float]) -> bool:
    """
    Grashof condition for a four-bar: at least one link can rotate fully iff
    s + l <= p + q, where s = shortest, l = longest, p,q = other two.

    Args:
        link_lengths: Exactly four link lengths (any order)

    Returns:
        True if Grashof compliant (crank can rotate fully)
    """
    if len(link_lengths) != 4:
        return True  # Not a four-bar; do not reject
    sorted_lengths = sorted(link_lengths)
    s, p, q, l = sorted_lengths[0], sorted_lengths[1], sorted_lengths[2], sorted_lengths[3]
    return (s + l) <= (p + q)


def check_mechanism_validity(
    mechanism: Mechanism,
    topology_id: frozenset[tuple[str, str]],
    cache: BadTopologyCache,
    check_grashof: bool = True,
) -> tuple[bool, str | None]:
    """
    Run Grashof (if 4-bar) and simulation; if invalid, add to cache and return (False, reason).

    Args:
        mechanism: Mechanism with current dimensions (will not be modified)
        topology_id: Canonical id for this topology (used if invalid)
        cache: BadTopologyCache to update when invalid
        check_grashof: If True, check Grashof for 4-link mechanisms first

    Returns:
        (valid, reason): valid=True and reason=None if OK; valid=False and reason set if invalid
    """
    # Grashof for four-bar: exactly 4 dimensions => single loop 4-bar
    if check_grashof and len(mechanism._dimension_mapping.names) == 4:
        lengths = tuple(mechanism._dimension_mapping.initial_values)
        if not is_grashof_compliant(lengths):
            cache.add_bad(topology_id, 'non_grashof')
            return (False, 'non_grashof')

    try:
        mechanism.set_dimensions(mechanism._current_dimensions)
        trajectory = mechanism.simulate()
    except Exception:
        cache.add_bad(topology_id, 'simulation_failed')
        return (False, 'simulation_failed')

    if np.isnan(trajectory).any() or np.isinf(trajectory).any():
        cache.add_bad(topology_id, 'simulation_failed')
        return (False, 'simulation_failed')

    return (True, None)
