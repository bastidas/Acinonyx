"""
Form and layout helpers for Acinonyx mechanisms. Tools for
computing which physical layer each part would occupy in a real 3D 
build (polygon- and shape-aware), so you can design mechanisms that are 
constructible and collision-free. You can also compute z-levels for link 
and polygon forms to ensure they are rendered in the correct order.

- polygon_utils: Polygon containment, rigid groups, bounding polygons (merge-polygon,
  create-polygons-from-rigid-groups, compute-link-z-levels).
- overlap: Segment-segment intersection (and future polygon overlap).
- z_level: Compute integer z-levels (layers) for each link for rendering order.
"""
from form_tools.overlap import segments_intersect
from form_tools.z_level import (
    DEFAULT_Z_LEVEL_CONFIG,
    ZLevelHeuristicConfig,
    compute_link_z_levels,
)

__all__ = [
    "compute_link_z_levels",
    "DEFAULT_Z_LEVEL_CONFIG",
    "ZLevelHeuristicConfig",
    "segments_intersect",
]
