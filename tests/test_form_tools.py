"""
Tests for form_tools: overlap, z_level, polygon_utils.

See docs/form_tools_improvements_plan.md (Phase 1.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from form_tools.overlap import segments_intersect
from form_tools.polygon_utils import bounding_polygon_for_links
from form_tools.polygon_utils import build_polygon_entity_conflict_pairs
from form_tools.polygon_utils import contained_links
from form_tools.polygon_utils import merge_two_polygons_geometry
from form_tools.polygon_utils import validate_polygon_rigidity
from form_tools.z_level import compute_link_z_levels
from form_tools.z_level import DEFAULT_Z_LEVEL_CONFIG
from form_tools.z_level import ZLevelHeuristicConfig
from pylink_tools.mechanism import create_mechanism_from_dict


# -----------------------------------------------------------------------------
# Overlap: segments_intersect
# -----------------------------------------------------------------------------


class TestSegmentsIntersect:
    """Tests for segments_intersect (crossing, collinear, touching, shared endpoint)."""

    def test_crossing_segments(self):
        """Two segments that cross in the interior intersect."""
        # (0,0)-(2,2) and (0,2)-(2,0) cross at (1,1)
        assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    def test_disjoint_segments(self):
        """Two segments that do not meet do not intersect."""
        assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False

    def test_collinear_overlap(self):
        """Collinear overlapping segments intersect."""
        # (0,0)-(3,0) and (1,0)-(2,0) overlap
        assert segments_intersect((0, 0), (3, 0), (1, 0), (2, 0)) is True

    def test_collinear_no_overlap(self):
        """Collinear but non-overlapping segments do not intersect."""
        assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False

    def test_touching_at_endpoint_exclude_false(self):
        """Segments that meet at one endpoint intersect when not excluding shared endpoints."""
        # (0,0)-(1,0) and (1,0)-(1,1) share (1,0); with exclude=False we get True
        assert (
            segments_intersect((0, 0), (1, 0), (1, 0), (1, 1), exclude_shared_endpoints=False)
            is True
        )

    def test_shared_endpoint_only_exclude_true(self):
        """When exclude_shared_endpoints=True, only shared endpoint counts as no intersection."""
        # (0,0)-(1,0) and (1,0)-(2,0) only share (1,0)
        assert (
            segments_intersect(
                (0, 0), (1, 0), (1, 0), (2, 0), exclude_shared_endpoints=True,
            )
            is False
        )

    def test_shared_endpoint_only_exclude_false(self):
        """When exclude_shared_endpoints=False, shared endpoint counts as intersection."""
        assert (
            segments_intersect(
                (0, 0), (1, 0), (1, 0), (2, 0), exclude_shared_endpoints=False,
            )
            is True
        )

    def test_degenerate_segment(self):
        """Degenerate segment (zero length) that lies on other segment."""
        # a1==a2 at (1,0), segment b is (0,0)-(2,0)
        assert segments_intersect((1, 0), (1, 0), (0, 0), (2, 0)) is True


# -----------------------------------------------------------------------------
# Z-level: compute_link_z_levels
# -----------------------------------------------------------------------------


@pytest.fixture
def fourbar_pylink_data():
    """Load 4-bar linkage from demo/test_graphs."""
    path = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'
    with open(path) as f:
        data = json.load(f)
    data['n_steps'] = 24
    return data


class TestComputeLinkZLevels:
    """Tests for compute_link_z_levels (mechanism and pylink_data)."""

    def test_with_mechanism(self, fourbar_pylink_data):
        """compute_link_z_levels(mechanism=...) returns at least one assignment with all links."""
        mechanism = create_mechanism_from_dict(fourbar_pylink_data, n_steps=24)
        assignments = compute_link_z_levels(mechanism=mechanism, n_steps=24)
        assert len(assignments) >= 1
        first = assignments[0]
        # 4-bar has 4 edges: ground, crank_link, coupler, rocker
        linkage = fourbar_pylink_data.get('linkage', {})
        edge_ids = set(linkage.get('edges', {}).keys())
        assert edge_ids
        for eid in edge_ids:
            assert eid in first, f'missing z-level for link {eid}'
            assert isinstance(first[eid], int)

    def test_with_pylink_data(self, fourbar_pylink_data):
        """compute_link_z_levels(pylink_data=...) returns same shape as mechanism path."""
        assignments = compute_link_z_levels(pylink_data=fourbar_pylink_data, n_steps=24)
        assert len(assignments) >= 1
        linkage = fourbar_pylink_data.get('linkage', {})
        edge_ids = set(linkage.get('edges', {}).keys())
        for eid in edge_ids:
            assert eid in assignments[0]
            assert isinstance(assignments[0][eid], int)

    def test_with_z_level_config(self, fourbar_pylink_data):
        """compute_link_z_levels accepts z_level_config and returns valid assignments."""
        assignments = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=DEFAULT_Z_LEVEL_CONFIG,
        )
        assert len(assignments) >= 1
        linkage = fourbar_pylink_data.get('linkage', {})
        edge_ids = set(linkage.get('edges', {}).keys())
        for eid in edge_ids:
            assert eid in assignments[0]
            assert isinstance(assignments[0][eid], int)

        # Custom config (min_z=0, crank_z=1) is accepted
        custom = ZLevelHeuristicConfig(min_z=0, crank_z=1, weight_reduce_deltas=1.0)
        assignments2 = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=custom,
        )
        assert len(assignments2) >= 1
        for eid in edge_ids:
            assert eid in assignments2[0]
            assert isinstance(assignments2[0][eid], int)

    def test_min_z_constraint(self, fourbar_pylink_data):
        """With min_z=2, all assigned z-levels are >= 2."""
        config = ZLevelHeuristicConfig(min_z=2, crank_z=2)
        assignments = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=config,
        )
        assert len(assignments) >= 1
        for link_id, z in assignments[0].items():
            assert z >= 2, f'link {link_id} got z={z}'

    def test_crank_z_preference(self, fourbar_pylink_data):
        """With crank_z=2 and weight_crank=1, root (crank-incident) link tends to get z=2."""
        config = ZLevelHeuristicConfig(
            min_z=0, crank_z=2, weight_crank=2.0, weight_reduce_deltas=1.0,
        )
        assignments = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=config,
            max_assignments=1,
        )
        assert len(assignments) >= 1
        z_vals = list(assignments[0].values())
        assert 2 in z_vals, 'crank_z=2 should appear in assignment'

    def test_soft_pin_preference(self, fourbar_pylink_data):
        """Soft pin a link to target z=3 with high weight; that link gets z=3."""
        linkage = fourbar_pylink_data.get('linkage', {})
        edge_ids = list(linkage.get('edges', {}).keys())
        assert len(edge_ids) >= 1
        link_id = edge_ids[0]
        config = ZLevelHeuristicConfig(
            min_z=0,
            soft_pins={link_id: (3, 10.0)},  # target z=3, weight 10
        )
        assignments = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=config,
            max_assignments=1,
        )
        assert len(assignments) >= 1
        assert assignments[0][link_id] == 3

    def test_hard_pin_below_min_z_raises(self, fourbar_pylink_data):
        """Fixed entity z-level below min_z raises ValueError."""
        linkage = fourbar_pylink_data.get('linkage', {})
        edge_ids = list(linkage.get('edges', {}).keys())
        assert len(edge_ids) >= 1
        config = ZLevelHeuristicConfig(min_z=2)
        with pytest.raises(ValueError, match='min_z'):
            compute_link_z_levels(
                pylink_data=fourbar_pylink_data,
                n_steps=24,
                fixed_entity_z_levels={edge_ids[0]: 1},  # 1 < min_z=2
                z_level_config=config,
            )

    def test_reduce_height_assigns_smaller_span(self, fourbar_pylink_data):
        """With weight_reduce_height > 0, assignment uses a bounded z span."""
        config = ZLevelHeuristicConfig(
            min_z=0,
            weight_reduce_height=1.0,
            weight_reduce_deltas=1.0,
        )
        assignments = compute_link_z_levels(
            pylink_data=fourbar_pylink_data,
            n_steps=24,
            z_level_config=config,
            max_assignments=1,
        )
        assert len(assignments) >= 1
        z_vals = list(assignments[0].values())
        span = max(z_vals) - min(z_vals)
        # 4-bar has 4 links; with conflicts we need at least a few levels
        assert span >= 0
        assert span <= 4  # heuristic should keep span small

    def test_sandwich_weight_prefers_between_neighbors(self, fourbar_pylink_data):
        """High sandwich weight pulls coupler z between pinned crank_link and rocker despite weak opposite soft pin."""
        data = dict(fourbar_pylink_data)
        data['n_steps'] = 24
        # Pin neighbors so coupler may use z=0 and z=3 (avoid ground@0 conflicting via trajectory).
        fixed = {'ground': 5, 'crank_link': 2, 'rocker': 4}
        no_sandwich = ZLevelHeuristicConfig(
            min_z=0,
            weight_reduce_deltas=0.0,
            weight_reduce_height=0.0,
            weight_prefer_sandwich=0.0,
            crank_z=None,
            weight_crank=0.0,
            soft_pins={'coupler': (0, 1_000.0)},
        )
        with_sandwich = ZLevelHeuristicConfig(
            min_z=0,
            weight_reduce_deltas=0.0,
            weight_reduce_height=0.0,
            weight_prefer_sandwich=500.0,
            crank_z=None,
            weight_crank=0.0,
            soft_pins={'coupler': (0, 1.0)},
        )
        a0 = compute_link_z_levels(
            pylink_data=data,
            n_steps=24,
            fixed_entity_z_levels=fixed,
            z_level_config=no_sandwich,
            max_assignments=1,
        )[0]
        a1 = compute_link_z_levels(
            pylink_data=data,
            n_steps=24,
            fixed_entity_z_levels=fixed,
            z_level_config=with_sandwich,
            max_assignments=1,
        )[0]
        assert a0['coupler'] == 0
        assert a1['coupler'] == 3


# -----------------------------------------------------------------------------
# Polygon utils: contained_links
# -----------------------------------------------------------------------------


@pytest.fixture
def minimal_linkage():
    """Minimal linkage with nodes and edges for contained_links tests."""
    return {
        'nodes': {
            'A': {'id': 'A', 'position': [0, 0]},
            'B': {'id': 'B', 'position': [2, 0]},
            'C': {'id': 'C', 'position': [2, 2]},
            'D': {'id': 'D', 'position': [0, 2]},
        },
        'edges': {
            'e1': {'source': 'A', 'target': 'B'},
            'e2': {'source': 'B', 'target': 'C'},
            'e3': {'source': 'C', 'target': 'D'},
            'e4': {'source': 'D', 'target': 'A'},
        },
    }


class TestContainedLinks:
    """Tests for contained_links (polygon contains 0, 1, or 2 link endpoints)."""

    def test_polygon_contains_no_links(self, minimal_linkage):
        """Polygon away from all nodes returns no contained links."""
        # Polygon far to the right
        polygon = [(10, 10), (12, 10), (12, 12), (10, 12)]
        result = contained_links(minimal_linkage, polygon)
        assert result == []

    def test_polygon_contains_one_link(self, minimal_linkage):
        """Polygon that contains both endpoints of one edge returns that edge."""
        # Triangle containing A and B (e1)
        polygon = [(0, 0), (2, 0), (1, -0.5)]
        result = contained_links(minimal_linkage, polygon)
        assert 'e1' in result
        assert len(result) == 1

    def test_polygon_contains_two_links(self, minimal_linkage):
        """Polygon containing two adjacent edges returns both."""
        # Square that contains A, B, C (so e1 and e2)
        polygon = [(0, 0), (3, 0), (3, 3), (0, 3)]
        result = contained_links(minimal_linkage, polygon)
        assert 'e1' in result
        assert 'e2' in result
        assert len(result) >= 2


# -----------------------------------------------------------------------------
# Polygon utils: validate_polygon_rigidity
# -----------------------------------------------------------------------------


class TestValidatePolygonRigidity:
    """Tests for validate_polygon_rigidity (rigid vs non-rigid pair)."""

    def test_single_link_returns_true(self):
        """Zero or one link returns (True, None)."""
        linkage = {'edges': {'e1': {'source': 'A', 'target': 'B'}}}
        traj = {'A': [[0, 0], [1, 0]], 'B': [[1, 0], [2, 0]]}
        ok, msg = validate_polygon_rigidity(linkage, [], traj)
        assert ok is True
        assert msg is None
        ok, msg = validate_polygon_rigidity(linkage, ['e1'], traj)
        assert ok is True
        assert msg is None

    def test_rigid_pair_returns_true(self):
        """Two links sharing a joint with constant relative angle return (True, None)."""
        # Links e1 (A-B) and e2 (B-C). Keep angle between (A->B) and (B->C) constant.
        linkage = {
            'edges': {
                'e1': {'source': 'A', 'target': 'B'},
                'e2': {'source': 'B', 'target': 'C'},
            },
        }
        # Rigid: B at origin, A and C rotate together (same angle)
        n = 8
        trajectory = {
            'A': [[1, 0], [0.7, 0.7], [0, 1], [-0.7, 0.7], [-1, 0], [-0.7, -0.7], [0, -1], [0.7, -0.7]],
            'B': [[0, 0]] * n,
            'C': [[0, 1], [-0.7, 0.7], [-1, 0], [-0.7, -0.7], [0, -1], [0.7, -0.7], [1, 0], [0.7, 0.7]],
        }
        # Normalize length for C so it's like a rigid bar
        trajectory['C'] = [[p[0] * 0.5, p[1] * 0.5] for p in trajectory['C']]
        ok, msg = validate_polygon_rigidity(linkage, ['e1', 'e2'], trajectory)
        assert ok is True
        assert msg is None

    def test_non_rigid_pair_returns_false(self):
        """Two links with changing relative angle return (False, message)."""
        linkage = {
            'edges': {
                'e1': {'source': 'A', 'target': 'B'},
                'e2': {'source': 'B', 'target': 'C'},
            },
        }
        # B fixed; A and C move so angle between (A->B) and (B->C) changes
        trajectory = {
            'A': [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
            'B': [[0, 0], [0, 0], [0, 0], [0, 0]],
            'C': [[0, 1], [0.1, 0.9], [0.3, 0.8], [0.5, 0.7]],  # different angular motion
        }
        ok, msg = validate_polygon_rigidity(linkage, ['e1', 'e2'], trajectory)
        assert ok is False
        assert msg is not None
        assert 'e1' in msg or 'e2' in msg or 'relative' in msg.lower() or 'joint' in msg.lower()


# -----------------------------------------------------------------------------
# Polygon utils: merge_two_polygons_geometry
# -----------------------------------------------------------------------------


class TestMergeTwoPolygonsGeometry:
    """Tests for merge_two_polygons_geometry."""

    def test_two_overlapping_triangles(self):
        """Two overlapping triangles yield one polygon bounding both."""
        tri_a = [(0, 0), (2, 0), (1, 2)]
        tri_b = [(1, 0), (3, 0), (2, 2)]
        result = merge_two_polygons_geometry(tri_a, tri_b)
        assert result is not None
        assert len(result) >= 3
        # Result should be a single polygon (list of vertices)
        assert all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in result)

    def test_disjoint_polygons_return_hull(self):
        """Two disjoint polygons return convex hull of union (one polygon)."""
        tri_a = [(0, 0), (1, 0), (0.5, 1)]
        tri_b = [(3, 0), (4, 0), (3.5, 1)]
        result = merge_two_polygons_geometry(tri_a, tri_b)
        assert result is not None
        assert len(result) >= 3


# -----------------------------------------------------------------------------
# Polygon utils: bounding_polygon_for_links
# -----------------------------------------------------------------------------


class TestBoundingPolygonForLinks:
    """Tests for bounding_polygon_for_links."""

    def test_returns_polygon_containing_endpoints(self):
        """Minimal link set and trajectory yield polygon with >=3 vertices containing endpoints."""
        linkage = {
            'edges': {
                'e1': {'source': 'A', 'target': 'B'},
                'e2': {'source': 'B', 'target': 'C'},
            },
        }
        trajectory = {
            'A': [[0, 0]],
            'B': [[2, 0]],
            'C': [[2, 2]],
        }
        result = bounding_polygon_for_links(
            ['e1', 'e2'],
            linkage,
            trajectory,
            margin_fraction=0.1,
            step=0,
        )
        assert len(result) >= 3
        # All points (0,0), (2,0), (2,2) should be inside or on boundary
        from form_tools.polygon_utils import is_point_in_polygon
        for pt in [(0, 0), (2, 0), (2, 2)]:
            assert is_point_in_polygon(pt, result), f'point {pt} not in result polygon'


# -----------------------------------------------------------------------------
# Polygon utils: build_polygon_entity_conflict_pairs (custom points for z-level)
# -----------------------------------------------------------------------------


class TestBuildPolygonEntityConflictPairs:
    """Tests that custom polygon points are used for collision detection."""

    def test_custom_polygon_points_detect_overlap(self):
        """Two forms with overlapping custom geometry (larger than link hull) get a conflict pair."""
        linkage = {
            'edges': {
                'e1': {'source': 'A', 'target': 'B'},
                'e2': {'source': 'B', 'target': 'C'},
                'e3': {'source': 'C', 'target': 'D'},
                'e4': {'source': 'D', 'target': 'A'},
            },
        }
        # Form 1: link e1 only; custom points are a large box that extends into form 2
        # Form 2: link e3 only; custom points are a large box overlapping form 1
        # Link hulls would be small (single segment each); custom shapes overlap.
        trajectories = {
            'A': [[0, 0], [0, 0]],
            'B': [[2, 0], [2, 0]],
            'C': [[2, 2], [2, 2]],
            'D': [[0, 2], [0, 2]],
        }
        polygons_for_z = [
            {
                'id': 'p1',
                'contained_links': ['e1'],
                'points': [(0, 0), (2, 0), (2, 1.5), (0, 1.5)],  # wide box overlapping below
            },
            {
                'id': 'p2',
                'contained_links': ['e3'],
                'points': [(2, 2), (2, 0.5), (0, 0.5), (0, 2)],  # box overlapping above
            },
        ]
        pairs = build_polygon_entity_conflict_pairs(
            polygons_for_z, linkage, trajectories, margin_fraction=0.05,
        )
        # Should detect overlap of custom shapes (e.g. at step 0 they overlap in y in [0.5, 1.5])
        assert any(
            ('polygon:p1', 'polygon:p2') == tuple(sorted((a, b))) for a, b in pairs
        ), 'expected conflict pair (polygon:p1, polygon:p2) when custom shapes overlap'

    def test_fallback_to_bounding_when_no_custom_points(self):
        """When polygon has no 'points', bounding_polygon_for_links is used."""
        linkage = {
            'edges': {'e1': {'source': 'A', 'target': 'B'}, 'e2': {'source': 'B', 'target': 'C'}},
        }
        trajectories = {'A': [[0, 0]], 'B': [[1, 0]], 'C': [[1, 1]]}
        polygons_for_z = [
            {'id': 'p1', 'contained_links': ['e1']},  # no points
            {'id': 'p2', 'contained_links': ['e2']},  # no points
        ]
        pairs = build_polygon_entity_conflict_pairs(
            polygons_for_z, linkage, trajectories, margin_fraction=0.05,
        )
        # No overlap at step 0: e1 is (0,0)-(1,0), e2 is (1,0)-(1,1); hulls may touch at B
        # So we may or may not get a pair depending on margin; just ensure no error
        assert isinstance(pairs, list)

    def test_connector_joint_inside_other_body_adds_conflict(self):
        """Connector endpoint inside another form body creates a hard conflict pair."""
        linkage = {
            'edges': {
                'e_body': {'source': 'A', 'target': 'B'},
                'e_conn': {'source': 'B', 'target': 'C'},
            },
        }
        trajectories = {
            'A': [[0, 0]],
            'B': [[1, 0]],
            'C': [[2, 0]],
        }
        polygons_for_z = [
            {
                'id': 'body',
                'contained_links': ['e_body'],
                'points': [(-0.2, -0.2), (1.2, -0.2), (1.2, 0.2), (-0.2, 0.2)],
            },
            {
                'id': 'connector',
                'contained_links': ['e_conn'],
                'points': [(0.9, -0.1), (2.1, -0.1), (2.1, 0.1), (0.9, 0.1)],
            },
        ]
        pairs = build_polygon_entity_conflict_pairs(
            polygons_for_z, linkage, trajectories, margin_fraction=0.0,
        )
        expected = tuple(sorted(('polygon:body', 'polygon:connector')))
        assert any(tuple(sorted((a, b))) == expected for a, b in pairs)

    def test_connector_joint_swept_path_intersects_other_swept_body_adds_conflict(self):
        """Hard guard uses swept areas: asynchronous overlap still creates conflict pair."""
        linkage = {
            'edges': {
                'e_left': {'source': 'A', 'target': 'B'},
                'e_conn': {'source': 'B', 'target': 'C'},
                'e_right': {'source': 'C', 'target': 'D'},
            },
        }
        # Connector joint B moves from x=0 -> x=2 while body polygon only occupies
        # x≈0 at step 0 and x≈2 at step 1 (never same-step overlap with B if sampled sparsely),
        # but the swept-area rule should still force a hard conflict.
        trajectories = {
            'A': [[-1.0, 0.0], [1.0, 0.0]],
            'B': [[0.0, 0.0], [2.0, 0.0]],
            'C': [[1.0, 0.0], [3.0, 0.0]],
            'D': [[2.0, 0.0], [4.0, 0.0]],
        }
        polygons_for_z = [
            {
                'id': 'connector',
                'contained_links': ['e_conn'],
                'points': [(-0.1, -0.1), (1.1, -0.1), (1.1, 0.1), (-0.1, 0.1)],
            },
            {
                'id': 'body',
                'contained_links': ['e_left', 'e_right'],
                # Sweeps from x≈0 to x≈2 across the two timesteps.
                'points': [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
            },
        ]
        pairs = build_polygon_entity_conflict_pairs(
            polygons_for_z, linkage, trajectories, margin_fraction=0.0,
        )
        expected = tuple(sorted(('polygon:body', 'polygon:connector')))
        assert any(tuple(sorted((a, b))) == expected for a, b in pairs)
