"""
Visualization tools for linkage mechanisms.

Modules:
- demo_viz: Unified plotting for demos (variation_plot, convergence, bounds)
- opt_viz: Optimization-specific visualization
- viz: General mechanism visualization
- animate: Animation utilities
"""
from __future__ import annotations

from viz_tools.demo_viz import DemoVizStyle
from viz_tools.demo_viz import STYLE
# from viz_tools.demo_viz import plot_convergence_comparison
# from viz_tools.demo_viz import plot_dimension_bounds
# from viz_tools.demo_viz import variation_plot

__all__ = [
    # 'variation_plot',
    # 'plot_convergence_comparison',
    # 'plot_dimension_bounds',
    'DemoVizStyle',
    'STYLE',
]
