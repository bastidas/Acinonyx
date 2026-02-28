"""
Overlap checks for link geometry.

Phase 1: segment-segment intersection (links as line segments).
Phase 2 (future): polygon-polygon overlap when links have attached polygons.
"""
from __future__ import annotations

from typing import Callable, Protocol, TypeAlias

import numpy as np

# Type for a 2D point (x, y)
Point2D: TypeAlias = tuple[float, float] | list[float] | np.ndarray


def segments_intersect(
    a1: Point2D,
    a2: Point2D,
    b1: Point2D,
    b2: Point2D,
    *,
    exclude_shared_endpoints: bool = True,
    tol: float = 1e-9,
) -> bool:
    """
    Return True if the segment (a1, a2) and the segment (b1, b2) intersect.

    Intersection includes: crossing in the interior, collinear overlap,
    or touching at an endpoint. When exclude_shared_endpoints is True,
    returns False if the only common point is a shared endpoint (a1==b1 etc.).
    Use that when links are adjacent (share a joint) and you only care about
    non-adjacent link crossings.

    Uses parametric form and orientation tests. Collinear overlapping segments
    are detected by interval overlap.

    Args:
        a1, a2: Endpoints of first segment.
        b1, b2: Endpoints of second segment.
        exclude_shared_endpoints: If True, return False when the only
            intersection is a shared endpoint.
        tol: Tolerance for numeric comparisons.

    Returns:
        True if the segments intersect (optionally excluding shared-endpoint-only).
    """
    a1_arr = np.asarray(a1, dtype=float)
    a2_arr = np.asarray(a2, dtype=float)
    b1_arr = np.asarray(b1, dtype=float)
    b2_arr = np.asarray(b2, dtype=float)

    # Helper: orientation of (p, q, r): +1 ccw, -1 cw, 0 collinear
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if abs(v) <= tol:
            return 0.0
        return float(np.sign(v))

    o1 = orient(a1_arr, a2_arr, b1_arr)
    o2 = orient(a1_arr, a2_arr, b2_arr)
    o3 = orient(b1_arr, b2_arr, a1_arr)
    o4 = orient(b1_arr, b2_arr, a2_arr)

    # General case: segments cross if (b1, b2) straddle line (a1,a2) and vice versa
    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        if o1 != o2 and o3 != o4:
            return True
        return False

    # Collinear or endpoint cases
    def on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(
            min(float(a[0]), float(b[0])) - tol <= float(p[0]) <= max(float(a[0]), float(b[0])) + tol
            and min(float(a[1]), float(b[1])) - tol <= float(p[1]) <= max(float(a[1]), float(b[1])) + tol
        )

    def same_point(p: np.ndarray, q: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(p) - np.asarray(q))) <= tol

    # All four collinear
    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        # Check overlap of collinear segments via parameter t in [0,1] for segment a
        da = a2_arr - a1_arr
        dd = float(np.dot(da, da))
        if dd <= tol:
            # a is degenerate
            return bool(
                on_segment(a1_arr, b1_arr, b2_arr)
                and _not_shared_only(
                    a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints
                )
            )
        tb1 = float(np.dot(b1_arr - a1_arr, da) / dd)
        tb2 = float(np.dot(b2_arr - a1_arr, da) / dd)
        t_b_min, t_b_max = min(tb1, tb2), max(tb1, tb2)
        # Segment a is t in [0, 1]
        if t_b_max < -tol or t_b_min > 1.0 + tol:
            return False
        # They overlap on the line
        return _not_shared_only(a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints)

    # One of b1, b2 lies on (a1,a2)
    if o1 == 0 and on_segment(b1_arr, a1_arr, a2_arr):
        return _not_shared_only(a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints)
    if o2 == 0 and on_segment(b2_arr, a1_arr, a2_arr):
        return _not_shared_only(a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints)
    if o3 == 0 and on_segment(a1_arr, b1_arr, b2_arr):
        return _not_shared_only(a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints)
    if o4 == 0 and on_segment(a2_arr, b1_arr, b2_arr):
        return _not_shared_only(a1_arr, a2_arr, b1_arr, b2_arr, same_point, exclude_shared_endpoints)

    return False


def _not_shared_only(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    same_point: Callable[[np.ndarray, np.ndarray], bool],
    exclude: bool,
) -> bool:
    if not exclude:
        return True
    # Return False if the only intersection would be a shared endpoint
    shared = 0
    if same_point(a1, b1) or same_point(a1, b2):
        shared += 1
    if same_point(a2, b1) or same_point(a2, b2):
        shared += 1
    # If they overlap in the interior we have more than endpoint touch
    # If they only touch at one endpoint, that's "shared only"
    if shared >= 2:
        return False  # both endpoints same -> same segment
    if shared == 1:
        return False  # only shared endpoint
    return True


class PolygonOverlapChecker(Protocol):
    """
    Protocol for future polygon-polygon overlap checks.

    When links have attached polygons (e.g. from drawnObjects merged to links),
    implement this to replace or supplement segment-segment checks.
    """

    def __call__(
        self,
        polygon_a: list[tuple[float, float]],
        polygon_b: list[tuple[float, float]],
    ) -> bool:
        """Return True if the two polygons overlap (interior or boundary)."""
        ...
