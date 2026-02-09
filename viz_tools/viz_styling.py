"""
viz_styling.py - Unified styling configuration for all visualization modules.

Consolidates styling from demo_viz.py, opt_viz.py, and viz.py into a single
module for consistency and maintainability.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pydantic import BaseModel
from pydantic import Field

from configs.matplotlib_config import configure_matplotlib_for_backend
# Configure matplotlib for backend use BEFORE any matplotlib imports

configure_matplotlib_for_backend()


# =============================================================================
# GLOBAL MATPLOTLIB CONFIGURATION
# =============================================================================

# Use a clean style as base
plt.style.use('seaborn-v0_8-whitegrid')

# Override with custom settings
mpl.rcParams.update({
    # Figure
    'figure.figsize': (12, 10),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'savefig.dpi': 150,
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',

    # Fonts
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,

    # Lines
    'lines.linewidth': 2.0,
    'lines.markersize': 6,

    # Axes
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'axes.axisbelow': True,

    # Grid
    'grid.alpha': 0.4,
    'grid.linewidth': 0.8,
    'grid.linestyle': '-',

    # Legend
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.fancybox': True,
})

# Set consistent seaborn style for all plots
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
sns.set_palette('colorblind')


# =============================================================================
# STYLE DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class DemoVizStyle:
    """
    Style configuration for demo visualizations.

    Uses a carefully selected color palette optimized for:
    - Colorblind accessibility
    - Print and screen readability
    - Clear visual hierarchy
    """
    # Primary colors (high contrast, distinct)
    base_color: str = '#2C3E50'       # Dark blue-gray (base/original)
    target_color: str = '#E74C3C'     # Red (target)
    optimized_color: str = '#9B59B6'  # Purple (optimized results)
    initial_color: str = '#27AE60'    # Green (initial/start)

    # Variation colors (for multiple items, colorblind-friendly)
    variation_colors: tuple = (
        '#3498DB',  # Blue
        '#E67E22',  # Orange
        '#1ABC9C',  # Teal
        '#F1C40F',  # Yellow
        '#9B59B6',  # Purple
        '#E91E63',  # Pink
        '#00BCD4',  # Cyan
        '#8BC34A',  # Light green
        '#FF5722',  # Deep orange
        '#607D8B',  # Blue gray
    )

    # Neutral colors
    bounds_color: str = '#BDC3C7'     # Light gray
    grid_color: str = '#ECF0F1'       # Very light gray
    text_color: str = '#2C3E50'       # Dark text
    muted_color: str = '#95A5A6'      # Muted gray

    # Alpha values
    base_alpha: float = 0.8
    target_alpha: float = 0.65
    variation_alpha: float = 0.5
    faded_alpha: float = 0.3

    # Line widths
    base_linewidth: float = 3.5
    target_linewidth: float = 2.5
    variation_linewidth: float = 1.8
    link_linewidth: float = 4.0

    # Marker sizes
    marker_size: float = 80
    base_marker_size: float = 80
    variation_marker_size: float = 50

    # Figure settings
    figsize: tuple = (12, 10)
    figsize_wide: tuple = (14, 8)
    figsize_tall: tuple = (10, 14)
    dpi: int = 150


@dataclass
class OptVizStyle:
    """
    Style configuration for optimization visualizations.

    Uses seaborn's colorblind-friendly palette for all colors.
    """
    # Get seaborn's colorblind palette
    _palette = sns.color_palette('colorblind', 10)

    # Assign semantic colors from palette
    target_color: str = sns.color_palette('colorblind')[3]      # Red/coral
    current_color: str = sns.color_palette('colorblind')[0]     # Blue
    initial_color: str = sns.color_palette('colorblind')[2]     # Green
    optimized_color: str = sns.color_palette('colorblind')[4]   # Purple
    bounds_color: str = sns.color_palette('gray', 5)[2]         # Gray

    # Additional colors for multi-trajectory plots
    accent_colors: tuple = tuple(sns.color_palette('colorblind', 10))

    # Line properties
    target_linewidth: float = 3.0
    current_linewidth: float = 2.5
    trajectory_alpha: float = 0.85

    # Marker properties
    marker_size: float = 80
    start_marker: str = 'o'
    end_marker: str = 's'

    # Figure properties
    figsize: tuple[int, int] = (12, 10)
    dpi: int = 150

    # Bounds bar properties
    bar_height: float = 0.6
    bar_alpha: float = 0.7


class PlotStyleConfig(BaseModel):
    """Pydantic configuration for plot styling (used by viz.py)"""

    # Colors and colormaps
    colormap: str = Field(default='Spectral', description='Matplotlib colormap name')
    color_palette: str = Field(default='tab10', description='Seaborn color palette')
    fixed_node_color: str = Field(default='#1f77b4', description='Color for fixed nodes')
    free_node_color: str = Field(default='#ff7f0e', description='Color for free nodes')

    # Line and marker properties
    linewidth: float = Field(default=2.0, ge=0.1, le=10.0, description='Line width')
    markersize: float = Field(default=15.0, ge=1.0, le=50.0, description='Marker size')
    alpha: float = Field(default=0.9, ge=0.0, le=1.0, description='Transparency')

    # Node properties
    node_size: int = Field(default=300, ge=50, le=1000, description='Node size for graph plots')

    # Display options
    show_grid: bool = Field(default=True, description='Show grid')
    show_legend: bool = Field(default=True, description='Show legend')
    equal_aspect: bool = Field(default=True, description='Use equal aspect ratio')

    # Font properties
    font_weight: str = Field(default='bold', description='Font weight')

    # Output properties
    dpi: int = Field(default=300, ge=72, le=600, description='DPI for saved figures')
    bbox_inches: str = Field(default='tight', description='Bounding box for saved figures')


# =============================================================================
# DEFAULT STYLE INSTANCES
# =============================================================================

# Global style instances
STYLE = DemoVizStyle()  # For demo_viz.py
DEFAULT_STYLE = OptVizStyle()  # For opt_viz.py
DEFAULT_PLOT_STYLE = PlotStyleConfig()  # For viz.py


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _save_or_show(fig: plt.Figure, out_path: Path | str | None, dpi: int = 150):
    """
    Save figure to path or show interactively.

    Args:
        fig: Matplotlib figure
        out_path: Path to save (None to show)
        dpi: DPI for saved figures
    """
    from pathlib import Path

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Saved: {out_path}')
    else:
        try:
            plt.show()
        except Exception:
            # If show fails (e.g., no display available), just close the figure
            plt.close(fig)
        plt.close(fig)


def _setup_plot_style(title: str, style: PlotStyleConfig = DEFAULT_PLOT_STYLE) -> None:
    """
    Setup common plot styling (for viz.py).

    Args:
        title: Plot title
        style: Style configuration
    """
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    if style.show_grid:
        plt.grid(True, zorder=-1)
    if style.equal_aspect:
        plt.axis('equal')


def _handle_output(
    title: str, out_path: str | Path | None = None,
    style: PlotStyleConfig = DEFAULT_PLOT_STYLE,
) -> None:
    """
    Handle plot output - save if out_path provided, otherwise show (for viz.py).

    Args:
        title: Plot title
        out_path: Output path for saving (None to display)
        style: Style configuration
    """
    from pathlib import Path

    if out_path is not None:
        out_path = Path(out_path)
        if out_path.is_dir():
            # If out_path is a directory, create filename from title
            filename = f'{title}.png'
            full_path = out_path / filename
        else:
            # If out_path includes filename, use as is
            full_path = out_path

        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(full_path, dpi=style.dpi, bbox_inches=style.bbox_inches)
        plt.close()  # Close to free memory
    else:
        # Only show plot if we're in an interactive environment (not in backend)
        try:
            plt.show()
        except Exception:
            # If show fails (e.g., no display available), just close the figure
            plt.close()
