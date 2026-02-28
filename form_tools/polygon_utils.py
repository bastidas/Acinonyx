"""
Polygon containment and form utilities (form_tools).

Used by merge-polygon, find-associated-polygons, create-polygons-from-rigid-groups,
compute-link-z-levels, merge-two-polygons, validate-polygon-rigidity.

Uses Shapely for point-in-polygon and polygon dilation (buffer).
Provides rigid-group detection, bounding-polygon creation, and polygon-aware
z-level conflict pairs (same rigidity criterion as validate_polygon_rigidity).
"""
from __future__ import annotations

import logging
import math
from typing import Any, cast

import numpy as np
from shapely import contains_xy as shapely_contains_xy
from shapely import make_valid as shapely_make_valid
from shapely import prepare as shapely_prepare
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import MultiPoint as ShapelyMultiPoint
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient as shapely_orient

logger = logging.getLogger(__name__)

# Max retries when buffered polygon does not contain all points
BOUNDING_POLYGON_CONTAINMENT_RETRIES = 8

# Radians: max allowed change in relative angle between two links at a shared joint across steps
RIGIDITY_ANGLE_TOLERANCE = 1e-3

# Half-circle segment count for solo link form (pill shape) at each end
LINK_FORM_HALFCIRCLE_SIDES = 16


def is_point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """
    Return True if point is inside or on the boundary of the polygon.
    Uses Shapely for reliable containment (same semantics as frontend).

    Args:
        point: [x, y]
        polygon: list of [x, y] vertices (closed path; need not repeat first point).
    """
    if len(polygon) < 3:
        return False
    try:
        shp = ShapelyPolygon(polygon)
        if not shp.is_valid:
            shp = shp.buffer(0)
        pt = ShapelyPoint(point[0], point[1])
        return bool(pt.within(shp) or pt.touches(shp))
    except Exception:
        return False


def polygons_intersect(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """
    Return True if the two polygons overlap (interior or boundary).
    Used for z-level conflict: if two forms/polygons intersect at any timestep,
    they must get different z-levels.
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False
    try:
        shp_a = ShapelyPolygon(poly_a)
        shp_b = ShapelyPolygon(poly_b)
        if not shp_a.is_valid:
            shp_a = shp_a.buffer(0)
        if not shp_b.is_valid:
            shp_b = shp_b.buffer(0)
        if shp_a.is_empty or shp_b.is_empty:
            return False
        return bool(shp_a.intersects(shp_b))
    except Exception:
        return False


def merge_two_polygons_geometry(
    points_a: list[tuple[float, float]],
    points_b: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """
    Compute the outer bounding polygon that is the merge of two polygons:
    the union of both, so interior boundaries (points/lines inside mutual bounds) are removed.
    Returns a single polygon that perfectly bounds both, or None if invalid/empty.

    Uses Shapely: union of the two polygons; if the result is MultiPolygon (disjoint),
    returns the convex hull of the union so one polygon bounds both.
    """
    if len(points_a) < 3 or len(points_b) < 3:
        return None
    try:
        poly_a = ShapelyPolygon(points_a)
        poly_b = ShapelyPolygon(points_b)
        if not poly_a.is_valid:
            poly_a = poly_a.buffer(0)
        if not poly_b.is_valid:
            poly_b = poly_b.buffer(0)
        if poly_a.is_empty or poly_b.is_empty:
            return None
        union_geom = poly_a.union(poly_b)
        if union_geom.is_empty:
            return None
        if union_geom.geom_type == "Polygon":
            coords = list(union_geom.exterior.coords)
        elif union_geom.geom_type == "MultiPolygon":
            hull_geom = union_geom.convex_hull
            if hull_geom.is_empty or hull_geom.geom_type != "Polygon":
                return None
            coords = list(hull_geom.exterior.coords)
        else:
            return None
        if not coords or len(coords) < 3:
            return None
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        return [(float(c[0]), float(c[1])) for c in coords]
    except Exception as e:
        logger.debug("merge_two_polygons_geometry failed: %s", e)
        return None


def get_link_endpoints(
    linkage: dict[str, Any], edge_id: str
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Get (start, end) positions for an edge from linkage.nodes and linkage.edges."""
    edges = linkage.get("edges") or {}
    nodes = linkage.get("nodes") or {}
    edge = edges.get(edge_id)
    if not edge:
        return None
    src_id = edge.get("source") or edge.get("sourceId")
    tgt_id = edge.get("target") or edge.get("targetId")
    if not src_id or not tgt_id:
        return None
    src_node = nodes.get(src_id)
    tgt_node = nodes.get(tgt_id)
    if not src_node or not tgt_node:
        return None
    pos_src = src_node.get("position")
    pos_tgt = tgt_node.get("position")
    if not pos_src or not pos_tgt or len(pos_src) < 2 or len(pos_tgt) < 2:
        return None
    return (tuple(pos_src[:2]), tuple(pos_tgt[:2]))


def contained_links(
    linkage: dict[str, Any],
    polygon_points: list[tuple[float, float]],
) -> list[str]:
    """
    Return list of edge IDs for which both endpoints lie inside the polygon.

    Builds the polygon once, prepares it for fast predicate tests, then uses
    vectorized contains_xy plus boundary fallback (covers) for "inside or on boundary" semantics.

    Args:
        linkage: dict with "nodes" (id -> { position: [x,y] }) and "edges" (id -> { source, target })
        polygon_points: list of [x, y] or (x, y) vertices
    """
    poly = [(float(p[0]), float(p[1])) for p in polygon_points]
    if len(poly) < 3:
        logger.debug("contained_links: polygon has fewer than 3 points, returning []")
        return []

    try:
        shp = ShapelyPolygon(poly)
        if not shp.is_valid:
            shp = shp.buffer(0)
        if shp.is_empty:
            return []
        shapely_prepare(shp)
    except Exception:
        return []

    edges = linkage.get("edges") or {}
    xs: list[float] = []
    ys: list[float] = []
    edge_data: list[tuple[str, int, int]] = []  # (edge_id, start_idx, end_idx)

    for edge_id in edges:
        endpoints = get_link_endpoints(linkage, edge_id)
        if not endpoints:
            continue
        start, end = endpoints
        i_start = len(xs)
        xs.append(start[0])
        ys.append(start[1])
        xs.append(end[0])
        ys.append(end[1])
        edge_data.append((edge_id, i_start, i_start + 1))

    if not edge_data:
        logger.debug("contained_links: no edges with endpoints, returning []")
        return []

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    inside = shapely_contains_xy(shp, x_arr, y_arr)
    # contains_xy is interior-only; include boundary via covers for points that are False
    for i in range(len(inside)):
        if not inside[i]:
            inside[i] = shp.covers(ShapelyPoint(x_arr[i], y_arr[i]))

    result = [edge_id for edge_id, i_s, i_e in edge_data if inside[i_s] and inside[i_e]]
    logger.debug("contained_links: polygon has %d vertices, found %d contained links", len(poly), len(result))
    return result


def validate_polygon_rigidity(
    linkage: dict[str, Any],
    contained_links: list[str],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
) -> tuple[bool, str | None]:
    """
    Check that the set of contained links behaves as a rigid body: relative angles
    between links at shared joints must be constant over time.

    Only run when len(contained_links) > 1; for 0 or 1 link return (True, None).

    Args:
        linkage: dict with "edges" (id -> { source, target }); node IDs are joint names.
        contained_links: list of edge IDs in the polygon.
        trajectory_dict: joint_name -> list of [x, y] per step (same as compute-pylink-trajectory).

    Returns:
        (True, None) if rigid or single/empty; (False, message) if non-rigid.
    """
    if len(contained_links) < 2:
        return True, None

    edges = linkage.get("edges") or {}
    link_endpoints: dict[str, tuple[str, str]] = {}
    for eid in contained_links:
        edge = edges.get(eid)
        if not edge:
            logger.warning("validate_polygon_rigidity: edge %s not in linkage", eid)
            return False, f"Edge {eid} not in linkage"
        src = edge.get("source") or edge.get("sourceId")
        tgt = edge.get("target") or edge.get("targetId")
        if not src or not tgt:
            return False, f"Edge {eid} missing source/target"
        link_endpoints[eid] = (src, tgt)

    link_ids = list(link_endpoints.keys())
    for i in range(len(link_ids)):
        for j in range(i + 1, len(link_ids)):
            lid_a, lid_b = link_ids[i], link_ids[j]
            sa, ta = link_endpoints[lid_a]
            sb, tb = link_endpoints[lid_b]
            if not ({sa, ta} & {sb, tb}):
                continue
            if not _pair_is_rigid(lid_a, lid_b, link_endpoints, trajectory_dict):
                joint = next(iter({sa, ta} & {sb, tb}))
                logger.info(
                    "validate_polygon_rigidity: links %s and %s non-rigid at joint %s",
                    lid_a, lid_b, joint,
                )
                return False, f"Links {lid_a} and {lid_b} move relative to each other at joint {joint}"
    return True, None


def _get_link_endpoint_joints(linkage: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Return link_id -> (source_joint_id, target_joint_id) from linkage.edges."""
    edges = linkage.get("edges") or {}
    result: dict[str, tuple[str, str]] = {}
    for eid, edge in edges.items():
        src = edge.get("source") or edge.get("sourceId")
        tgt = edge.get("target") or edge.get("targetId")
        if src and tgt:
            result[eid] = (src, tgt)
    return result


def _pair_is_rigid(
    lid_a: str,
    lid_b: str,
    link_endpoints: dict[str, tuple[str, str]],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
) -> bool:
    """
    Return True if the two links (which must share a joint) have constant relative 
    angle over time.
    """
    sa, ta = link_endpoints.get(lid_a, (None, None))
    sb, tb = link_endpoints.get(lid_b, (None, None))
    if not sa or not ta or not sb or not tb:
        return False
    shared = {sa, ta} & {sb, tb}
    if not shared:
        return False
    joint = next(iter(shared))
    other_a = ta if joint == sa else sa
    other_b = tb if joint == sb else sb

    n_steps = None
    for positions in trajectory_dict.values():
        if positions and len(positions) > 0:
            n_steps = len(positions)
            break
    if not n_steps or n_steps < 2:
        return True

    def get_pos(jid: str, step: int) -> tuple[float, float] | None:
        pts = trajectory_dict.get(jid)
        if not pts or step >= len(pts):
            return None
        p = pts[step]
        return (float(p[0]), float(p[1]))

    angles = []
    for step in range(n_steps):
        p_j = get_pos(joint, step)
        p_oa = get_pos(other_a, step)
        p_ob = get_pos(other_b, step)
        if p_j is None or p_oa is None or p_ob is None:
            continue
        vax = p_oa[0] - p_j[0]
        vay = p_oa[1] - p_j[1]
        vbx = p_ob[0] - p_j[0]
        vby = p_ob[1] - p_j[1]
        ang_a = math.atan2(vay, vax)
        ang_b = math.atan2(vby, vbx)
        diff = ang_a - ang_b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        angles.append(diff)
    if len(angles) < 2:
        return True
    mean_ang = sum(angles) / len(angles)
    max_dev = max(abs(a - mean_ang) for a in angles)
    return max_dev <= RIGIDITY_ANGLE_TOLERANCE


def detect_rigid_groups(
    linkage: dict[str, Any],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
) -> list[list[str]]:
    """
    Partition links into rigid groups: two links in the same group iff they are
    connected (share a joint) and the pair is rigid (relative angle constant over time).
    Uses union-find to merge links that are rigid with each other.

    Returns:
        List of groups, each a list of link IDs.
    """
    link_endpoints = _get_link_endpoint_joints(linkage)
    link_ids = list(link_endpoints.keys())
    if not link_ids:
        return []

    # Union-find parent
    parent: dict[str, str] = {lid: lid for lid in link_ids}

    def find(lid: str) -> str:
        if parent[lid] != lid:
            parent[lid] = find(parent[lid])
        return parent[lid]

    def union(lid_a: str, lid_b: str) -> None:
        ra, rb = find(lid_a), find(lid_b)
        if ra != rb:
            parent[ra] = rb

    # Joint -> list of link IDs
    joint_to_links: dict[str, list[str]] = {}
    for lid, (sa, ta) in link_endpoints.items():
        joint_to_links.setdefault(sa, []).append(lid)
        joint_to_links.setdefault(ta, []).append(lid)

    for joint, lids in joint_to_links.items():
        for i, la in enumerate(lids):
            for lb in lids[i + 1 :]:
                if _pair_is_rigid(la, lb, link_endpoints, trajectory_dict):
                    union(la, lb)

    # Group by root
    groups_map: dict[str, list[str]] = {}
    for lid in link_ids:
        root = find(lid)
        groups_map.setdefault(root, []).append(lid)
    return list(groups_map.values())


def _ensure_ccw(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return vertices in CCW order (positive signed area). Uses Shapely orient."""
    if len(vertices) < 3:
        return vertices
    try:
        poly = ShapelyPolygon(vertices)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return vertices
        oriented = shapely_orient(poly, sign=1.0)
        coords = list(oriented.exterior.coords)
        if coords and len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        return [(float(c[0]), float(c[1])) for c in coords]
    except Exception:
        return vertices


def _polygon_contains_hull(
    buffered_geom: Any,
    hull_vertices: list[tuple[float, float]],
    tol: float = 1e-9,
) -> bool:
    """Return True if buffered_geom contains all hull vertices (with tolerance)."""
    shapely_prepare(buffered_geom)
    for (x, y) in hull_vertices:
        pt = ShapelyPoint(x, y)
        if buffered_geom.contains(pt):
            continue
        if hasattr(buffered_geom, "boundary") and buffered_geom.boundary is not None:
            if buffered_geom.boundary.distance(pt) <= tol:
                continue
        return False
    return True


def _polygon_contains_all_points(
    geom: Any, points: list[tuple[float, float]], tol: float = 1e-9
) -> bool:
    """Return True if geom contains every point (interior or boundary within tol)."""
    shapely_prepare(geom)
    for (x, y) in points:
        pt = ShapelyPoint(x, y)
        if geom.contains(pt):
            continue
        if hasattr(geom, "boundary") and geom.boundary is not None:
            try:
                if geom.boundary.distance(pt) <= tol:
                    continue
            except Exception:
                pass
        return False
    return True


def _points_outside_geom(
    geom: Any, points: list[tuple[float, float]], tol: float = 1e-9
) -> list[tuple[float, float]]:
    """Return list of points that are not contained in geom (for logging)."""
    shapely_prepare(geom)
    outside: list[tuple[float, float]] = []
    for (x, y) in points:
        pt = ShapelyPoint(x, y)
        if geom.contains(pt):
            continue
        if hasattr(geom, "boundary") and geom.boundary is not None:
            try:
                if geom.boundary.distance(pt) <= tol:
                    continue
            except Exception:
                pass
        outside.append((x, y))
    return outside


def _hull_from_points_shapely(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """
    Build convex hull using Shapely's convex_hull on a MultiPoint.
    Guaranteed to contain all input points. Returns hull exterior as list of (x,y)
    or None if degenerate (e.g. < 3 points or collinear).
    """
    logger.debug("_hull_from_points_shapely: input points=%d", len(points))
    if len(points) < 3:
        logger.debug("_hull_from_points_shapely: degenerate (< 3 points), returning None")
        return None
    try:
        mp = ShapelyMultiPoint(points)
        hull_geom = mp.convex_hull
        if hull_geom.is_empty:
            logger.debug("_hull_from_points_shapely: convex_hull is empty, returning None")
            return None
        if hull_geom.geom_type == "Polygon":
            coords = list(hull_geom.exterior.coords)
            if coords and len(coords) >= 3:
                if coords[0] == coords[-1]:
                    coords = coords[:-1]
                out = [(float(c[0]), float(c[1])) for c in coords]
                logger.debug("_hull_from_points_shapely: hull vertices=%d", len(out))
                return out
        if hull_geom.geom_type == "LineString":
            # Collinear: buffer the line by a tiny amount to get a polygon
            try:
                poly = hull_geom.buffer(1e-10)
                if not poly.is_empty and hasattr(poly, "exterior"):
                    coords = list(poly.exterior.coords)
                    if len(coords) >= 3:
                        if coords[0] == coords[-1]:
                            coords = coords[:-1]
                        out = [(float(c[0]), float(c[1])) for c in coords]
                        logger.debug("_hull_from_points_shapely: collinear, buffered line -> %d vertices", len(out))
                        return out
            except Exception:
                pass
        if hull_geom.geom_type == "Point":
            logger.debug("_hull_from_points_shapely: degenerate (Point), returning None")
            return None
    except Exception as e:
        logger.debug("Shapely convex_hull failed: %s", e)
    logger.debug("_hull_from_points_shapely: degenerate or failed, returning None")
    return None


def _limit_polygon_sides(coords: list[tuple[float, float]], max_sides: int = 32) -> list[tuple[float, float]]:
    """Reduce vertex count to at most max_sides using simplify. Returns list of (x,y).
    Uses Douglas-Peucker (preserve_topology=False) for speed; result may differ slightly from topology-preserving simplify.
    """
    if len(coords) <= max_sides:
        return coords
    try:
        poly = ShapelyPolygon(coords)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return coords[:max_sides]  # fallback: truncate
        # Tolerance in same units as coords; use a fraction of bbox size
        try:
            bounds = poly.bounds
            span = max(
                abs(bounds[2] - bounds[0]) if bounds[2] != bounds[0] else 1.0,
                abs(bounds[3] - bounds[1]) if bounds[3] != bounds[1] else 1.0,
                1e-6,
            )
        except Exception:
            span = 1.0
        lo, hi = 0.0, span * 0.5
        best = coords
        for _ in range(25):
            mid = (lo + hi) / 2
            simplified = poly.simplify(mid, preserve_topology=False)
            if simplified.is_empty:
                hi = mid
                continue
            ext = simplified.exterior
            npts = len(ext.coords) - (1 if ext.coords[0] == ext.coords[-1] else 0)
            if npts <= max_sides:
                best = [(float(c[0]), float(c[1])) for c in ext.coords]
                if best and len(best) > 1 and best[0] == best[-1]:
                    best = best[:-1]
                lo = mid
            else:
                hi = mid
        return best if len(best) <= max_sides else coords[:max_sides]
    except Exception:
        return coords[:max_sides]


def _buffer_hull_and_check(
    poly: Any, pad: float, check_points: list[tuple[float, float]]
) -> Any | None:
    """Buffer polygon by pad, return the (single) buffered geom if it contains all check_points, else None."""
    try:
        if getattr(poly, "is_empty", True):
            logger.debug("_buffer_hull_and_check: poly is empty, pad=%.6f", pad)
            return None
        buffered = poly.buffer(pad, join_style=1, quad_segs=4)
        if buffered.is_empty:
            logger.debug("_buffer_hull_and_check: buffered is empty, pad=%.6f", pad)
            return None
        geom = buffered
        if buffered.geom_type == "MultiPolygon" and len(buffered.geoms) > 0:
            geom = max(buffered.geoms, key=lambda g: g.area)
        if _polygon_contains_all_points(geom, check_points):
            logger.debug("_buffer_hull_and_check: success, pad=%.6f, check_points=%d", pad, len(check_points))
            return geom
        outside = _points_outside_geom(geom, check_points)
        logger.debug(
            "_buffer_hull_and_check: containment failed, pad=%.6f, %d points outside: %s",
            pad,
            len(outside),
            outside[:10] if len(outside) > 10 else outside,
        )
    except Exception as e:
        logger.debug("_buffer_hull_and_check: exception pad=%.6f: %s", pad, e)
    return None


def _to_valid_polygon_geom(vertices: list[tuple[float, float]]) -> Any | None:
    """Build a valid Shapely Polygon from vertices; return None if degenerate or empty."""
    p = ShapelyPolygon(vertices)
    if not p.is_valid:
        p = shapely_make_valid(p)
        gt = getattr(p, "geom_type", "")
        if gt == "GeometryCollection":
            polys = [g for g in p.geoms if g.geom_type == "Polygon" and not g.is_empty]
            p = max(polys, key=lambda g: g.area) if polys else None
        elif gt == "MultiPolygon" and not p.is_empty and len(p.geoms) > 0:
            p = max(p.geoms, key=lambda g: g.area)
        elif gt != "Polygon" or getattr(p, "is_empty", True):
            p = None
    return p if p is not None and not getattr(p, "is_empty", True) else None


def _bounding_polygon_shapely_buffer(
    hull_vertices: list[tuple[float, float]],
    padding: float,
    max_sides: int = 32,
    source_points: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """
    Dilation: build polygon from hull (CCW), buffer by padding, return exterior as (x,y) list.
    Verifies containment of source_points; retries with larger pad; fallback to parallel_offset.
    """
    logger.info(
        "bounding_polygon_shapely_buffer: hull_vertices=%d, padding=%.6f, max_sides=%d, source_points=%s",
        len(hull_vertices), padding, max_sides, len(source_points) if source_points is not None else 0,
    )
    if len(hull_vertices) < 3 or padding <= 0:
        logger.warning("bounding_polygon_shapely_buffer: skipping (hull_vertices=%d or padding<=0), returning hull as-is", len(hull_vertices))
        return hull_vertices
    check_points = source_points if source_points is not None else hull_vertices
    hull_ccw = _ensure_ccw(hull_vertices)

    for attempt in range(BOUNDING_POLYGON_CONTAINMENT_RETRIES):
        pad = padding * (1.0 + 0.2 * attempt)
        logger.debug("bounding_polygon_shapely_buffer: attempt %d, pad=%.6f", attempt + 1, pad)
        for poly in (_to_valid_polygon_geom(hull_ccw), _to_valid_polygon_geom(list(reversed(hull_ccw)))):
            if poly is None:
                continue
            geom = _buffer_hull_and_check(poly, pad, check_points)
            if geom is not None:
                coords = list(geom.exterior.coords)
                if coords and len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                out = [(float(c[0]), float(c[1])) for c in coords]
                limited = _limit_polygon_sides(out, max_sides)
                logger.info("bounding_polygon_shapely_buffer: success after attempt %d, output vertices=%d", attempt + 1, len(limited))
                return limited
        if attempt > 0:
            logger.warning(
                "Bounding polygon does not contain all rigid-group points (attempt %d/%d), retrying with larger padding",
                attempt + 1, BOUNDING_POLYGON_CONTAINMENT_RETRIES,
            )

    logger.warning("Bounding polygon failed containment after %d retries; using parallel-offset fallback", BOUNDING_POLYGON_CONTAINMENT_RETRIES)
    fallback = _bounding_polygon_parallel_offset(hull_ccw, padding)
    try:
        if len(fallback) >= 3:
            closed = list(fallback) + [fallback[0]] if fallback[0] != fallback[-1] else list(fallback)
            fallback_poly = ShapelyPolygon(closed)
            if not fallback_poly.is_valid:
                fallback_poly = shapely_make_valid(fallback_poly)
            if fallback_poly and not fallback_poly.is_empty:
                outside_fallback = _points_outside_geom(fallback_poly, check_points)
                if outside_fallback:
                    logger.warning("bounding_polygon_parallel_offset fallback does not contain %d points: %s", len(outside_fallback), outside_fallback[:15])
    except Exception as e:
        logger.debug("Could not verify fallback polygon containment: %s", e)
    logger.info("bounding_polygon_shapely_buffer: returning parallel_offset fallback, vertices=%d", len(fallback))
    return fallback


def _collect_link_points_at_step(
    link_ids: list[str],
    linkage: dict[str, Any],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
    step: int,
) -> list[tuple[float, float]]:
    """Collect (x, y) of all link endpoints at the given trajectory step."""
    link_endpoints = _get_link_endpoint_joints(linkage)
    points: list[tuple[float, float]] = []
    for lid in link_ids:
        st = link_endpoints.get(lid)
        if not st:
            continue
        for jid in st:
            pts = trajectory_dict.get(jid)
            if pts and step < len(pts):
                p = pts[step]
                points.append((float(p[0]), float(p[1])))
    return points


def _collect_joint_positions_ordered(
    link_ids: list[str],
    linkage: dict[str, Any],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
    step: int,
) -> list[tuple[float, float]]:
    """
    Return joint positions at the given step for all joints of the given links,
    in deterministic order (sorted by joint id). Used for rigid transform correspondence.
    """
    link_endpoints = _get_link_endpoint_joints(linkage)
    joint_ids = sorted(
        {jid for lid in link_ids for jid in (link_endpoints.get(lid) or ())}
    )
    out: list[tuple[float, float]] = []
    for jid in joint_ids:
        pts = trajectory_dict.get(jid)
        if pts and step < len(pts):
            p = pts[step]
            out.append((float(p[0]), float(p[1])))
    return out


def _rigid_transform_2d(
    points_src: list[tuple[float, float]],
    points_tgt: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Best-fit rigid transform (rotation R, translation t) such that points_tgt ≈ R @ points_src + t.
    Uses Kabsch (SVD). Returns (R 2x2, t 2,) or None if degenerate.
    """
    if len(points_src) < 2 or len(points_src) != len(points_tgt):
        return None
    P = np.array(points_src, dtype=float)
    Q = np.array(points_tgt, dtype=float)
    if P.shape[0] < 2:
        return None
    centroid_p = P.mean(axis=0)
    centroid_q = Q.mean(axis=0)
    P_c = P - centroid_p
    Q_c = Q - centroid_q
    H = P_c.T @ Q_c
    try:
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_q - R @ centroid_p
        return (R, t)
    except Exception:
        return None


def _apply_rigid_transform(
    points: list[tuple[float, float]],
    R: np.ndarray,
    t: np.ndarray,
) -> list[tuple[float, float]]:
    """Apply rigid transform to each point: R @ p + t."""
    out: list[tuple[float, float]] = []
    for (px, py) in points:
        p = np.array([px, py])
        q = R @ p + t
        out.append((float(q[0]), float(q[1])))
    return out


def _polygon_geometry_at_step(
    polygon_obj: dict[str, Any],
    link_ids: list[str],
    linkage: dict[str, Any],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
    step: int,
    margin_fraction: float = 0.05,
    margin_units: float | None = None,
) -> list[tuple[float, float]]:
    """
    Return polygon geometry at the given trajectory step for z-level conflict detection.

    When the drawn object has custom "points" (user-drawn or merged polygon), use those
    points transformed by the rigid motion of the contained links from step 0 to this step,
    so that collisions reflect the actual form shape. Otherwise fall back to
    bounding_polygon_for_links (hull of link endpoints at this step).
    """
    custom_points = polygon_obj.get("points")
    if isinstance(custom_points, (list, tuple)) and len(custom_points) >= 3:
        pts0 = [(float(p[0]), float(p[1])) for p in custom_points]
        ref_pos = _collect_joint_positions_ordered(
            link_ids, linkage, trajectory_dict, 0
        )
        step_pos = _collect_joint_positions_ordered(
            link_ids, linkage, trajectory_dict, step
        )
        if len(ref_pos) >= 2 and len(ref_pos) == len(step_pos):
            transform = _rigid_transform_2d(ref_pos, step_pos)
            if transform is not None:
                R, t = transform
                return _apply_rigid_transform(pts0, R, t)
        # Fallback: custom points invalid or transform failed (e.g. single joint)
        logger.debug(
            "polygon_geometry_at_step: using bounding_polygon_for_links fallback (custom points not transformable)"
        )
    return bounding_polygon_for_links(
        link_ids,
        linkage,
        trajectory_dict,
        margin_fraction=margin_fraction,
        margin_units=margin_units,
        step=step,
    )


def _pill_polygon_for_two_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    padding: float,
) -> list[tuple[float, float]]:
    """Return a pill-shaped polygon (two half-circles) around the segment p0-p1 with given padding."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return [p0, p1]
    u = (dx / length, dy / length)
    n = (-u[1], u[0])
    angle_n = math.atan2(n[1], n[0])
    n_sides = LINK_FORM_HALFCIRCLE_SIDES
    arc0 = [
        (p0[0] + padding * math.cos(angle_n + math.pi * i / n_sides), p0[1] + padding * math.sin(angle_n + math.pi * i / n_sides))
        for i in range(n_sides + 1)
    ]
    arc1 = [
        (p1[0] + padding * math.cos(angle_n + math.pi + math.pi * i / n_sides), p1[1] + padding * math.sin(angle_n + math.pi + math.pi * i / n_sides))
        for i in range(n_sides + 1)
    ]
    return arc0 + arc1


def _bounding_polygon_parallel_offset(
    hull_vertices: list[tuple[float, float]], padding: float
) -> list[tuple[float, float]]:
    """
    Build a bounding polygon by taking each edge of the convex hull and
    offsetting it outward by padding (a line parallel to the link at padding distance).
    Each hull edge (V_i, V_{i+1}) becomes a segment with endpoints
    V_i + n*padding and V_{i+1} + n*padding, where n is the outward normal.
    Used as fallback when Shapely buffer is not applicable.
    """
    logger.debug("_bounding_polygon_parallel_offset: hull_vertices=%d, padding=%.6f", len(hull_vertices), padding)
    if len(hull_vertices) < 3 or padding <= 0:
        return hull_vertices
    n = len(hull_vertices)
    result: list[tuple[float, float]] = []

    for i in range(n):
        v_i = np.array(hull_vertices[i])
        v_next = np.array(hull_vertices[(i + 1) % n])
        edge = v_next - v_i
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-12:
            continue
        # Outward normal for CCW hull: right perpendicular of edge direction
        # (x, y) -> (y, -x) normalized
        n_x = float(edge[1]) / length
        n_y = -float(edge[0]) / length
        pt_a = (v_i[0] + n_x * padding, v_i[1] + n_y * padding)
        pt_b = (v_next[0] + n_x * padding, v_next[1] + n_y * padding)
        result.append(pt_a)
        result.append(pt_b)
    logger.debug("_bounding_polygon_parallel_offset: result vertices=%d (edge list, not closed polygon)", len(result))
    return result


def _verify_bounding_contains_points(
    result: list[tuple[float, float]],
    points_tuples: list[tuple[float, float]],
) -> None:
    """Log ERROR if result polygon does not contain all points_tuples; otherwise debug log."""
    if len(result) < 3 or not points_tuples:
        return
    try:
        closed = list(result) + [result[0]] if (not result or result[0] != result[-1]) else list(result)
        result_poly = ShapelyPolygon(closed)
        if not result_poly.is_valid:
            result_poly = shapely_make_valid(result_poly)
        if result_poly and not result_poly.is_empty:
            outside = _points_outside_geom(result_poly, points_tuples)
            if outside:
                logger.error(
                    "bounding_polygon_for_links: final polygon does NOT contain all %d source points; %d outside: %s",
                    len(points_tuples),
                    len(outside),
                    outside[:20],
                )
            else:
                logger.debug("bounding_polygon_for_links: verified final polygon contains all %d source points", len(points_tuples))
    except Exception as e:
        logger.debug("bounding_polygon_for_links: could not verify containment: %s", e)


def bounding_polygon_for_links(
    link_ids: list[str],
    linkage: dict[str, Any],
    trajectory_dict: dict[str, list[list[float] | tuple[float, float]]],
    margin_fraction: float = 0.05,
    margin_units: float | None = None,
    step: int = 0,
) -> list[tuple[float, float]]:
    """
    Compute a convex hull of all link endpoints at the given trajectory step,
    then expand: by margin_units (constant padding) if set, else by margin_fraction (scale from centroid).

    For "create polygons from rigid groups" and merge consistency, only step=0 should be used.

    Returns:
        List of [x, y] vertices (closed polygon; first point not repeated at end).
    """
    logger.info(
        "bounding_polygon_for_links: link_ids=%d, step=%d, margin_units=%s, margin_fraction=%.4f",
        len(link_ids), step, margin_units, margin_fraction,
    )
    points_tuples = _collect_link_points_at_step(link_ids, linkage, trajectory_dict, step)
    logger.info("bounding_polygon_for_links: collected %d points from trajectory (step=%d)", len(points_tuples), step)

    if len(points_tuples) == 2 and margin_units is not None and margin_units > 0:
        result = _pill_polygon_for_two_points(points_tuples[0], points_tuples[1], margin_units)
        if len(result) <= 2:
            logger.warning("bounding_polygon_for_links: 2-point degenerate (zero length), returning as-is")
            return points_tuples
        logger.info("bounding_polygon_for_links: 2-point pill -> %d vertices", len(result))
        return result

    if len(points_tuples) < 3:
        logger.warning("bounding_polygon_for_links: fewer than 3 points, returning as-is")
        return points_tuples

    hull_pts = _hull_from_points_shapely(points_tuples)
    if hull_pts is None:
        logger.warning("bounding_polygon_for_links: hull degenerate (collinear/point), returning %d points as-is", len(points_tuples))
        return points_tuples
    logger.info("bounding_polygon_for_links: hull has %d vertices", len(hull_pts))

    if margin_units is not None and margin_units > 0:
        result = _bounding_polygon_shapely_buffer(hull_pts, margin_units, max_sides=32, source_points=points_tuples)
    else:
        centroid = np.array(hull_pts).mean(axis=0)
        scale = 1.0 + margin_fraction
        result = [
            cast(tuple[float, float], tuple(float(x) for x in (centroid + scale * (np.array(p) - centroid))))
            for p in hull_pts
        ]
        logger.info("bounding_polygon_for_links: margin_fraction path, result vertices=%d", len(result))

    _verify_bounding_contains_points(result, points_tuples)
    logger.info("bounding_polygon_for_links: returning polygon with %d vertices", len(result))
    return result


def build_polygon_entity_conflict_pairs(
    polygons_for_z: list[dict],
    linkage: dict[str, Any],
    trajectories: dict[str, list[list[float] | tuple[float, float]]],
    margin_fraction: float = 0.05,
    margin_units: float | None = None,
) -> list[tuple[str, str]]:
    """
    Build extra entity conflict pairs for z-level assignment when forms/polygons
    overlap over time (segment-segment may not intersect but polygon bounds do).

    Used by compute-link-z-levels: for each timestep, compute bounding polygon
    per form; if two forms' polygons intersect at any step, add (eid1, eid2)
    so they get different z-levels. Entity IDs are "polygon:" + polygon id.

    Args:
        polygons_for_z: List of drawn-object dicts with "id" and "contained_links".
        linkage: Linkage dict (nodes, edges).
        trajectories: joint_name -> list of [x, y] per step.
        margin_fraction: Passed to bounding_polygon_for_links when margin_units not set.
        margin_units: Optional constant padding for bounding polygon.

    Returns:
        Deduplicated list of (eid1, eid2) with eid1 < eid2.
    """
    n_steps_actual = min(len(v) for v in trajectories.values() if v) if trajectories else 0
    if not linkage or n_steps_actual <= 0 or len(polygons_for_z) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    for step in range(n_steps_actual):
        polys_at_step: dict[str, list[tuple[float, float]]] = {}
        for o in polygons_for_z:
            pid = o.get("id")
            contained = o.get("contained_links") or []
            if not pid or not contained:
                continue
            # Use custom polygon points when present (merged/user-drawn shapes) so
            # collisions reflect actual form geometry; otherwise use link-based bounding.
            pts = _polygon_geometry_at_step(
                o,
                contained,
                linkage,
                trajectories,
                step,
                margin_fraction=margin_fraction,
                margin_units=margin_units,
            )
            if len(pts) >= 3:
                polys_at_step["polygon:" + pid] = pts
        keys = list(polys_at_step.keys())
        for i, eid1 in enumerate(keys):
            for eid2 in keys[i + 1 :]:
                if polygons_intersect(polys_at_step[eid1], polys_at_step[eid2]):
                    pairs.append((eid1, eid2))
    return list({cast(tuple[str, str], tuple(sorted((a, b)))) for a, b in pairs})
