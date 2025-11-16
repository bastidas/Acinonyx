
import numpy as np
# Set matplotlib to use non-GUI backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use('Agg', force=True)  # force=True to override existing backend  
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import cycle
from pathlib import Path
import networkx as nx
import os
from pydantic import BaseModel, Field
from configs.link_models import Link


class PlotStyleConfig(BaseModel):
    """Pydantic configuration for plot styling"""
    
    # Colors and colormaps
    colormap: str = Field(default="Spectral", description="Matplotlib colormap name")
    color_palette: str = Field(default="tab10", description="Seaborn color palette")
    fixed_node_color: str = Field(default="#1f77b4", description="Color for fixed nodes")
    free_node_color: str = Field(default="#ff7f0e", description="Color for free nodes")
    
    # Line and marker properties
    linewidth: float = Field(default=2.0, ge=0.1, le=10.0, description="Line width")
    markersize: float = Field(default=15.0, ge=1.0, le=50.0, description="Marker size")
    alpha: float = Field(default=0.9, ge=0.0, le=1.0, description="Transparency")
    
    # Node properties
    node_size: int = Field(default=300, ge=50, le=1000, description="Node size for graph plots")
    
    # Display options
    show_grid: bool = Field(default=True, description="Show grid")
    show_legend: bool = Field(default=True, description="Show legend")
    equal_aspect: bool = Field(default=True, description="Use equal aspect ratio")
    
    # Font properties
    font_weight: str = Field(default="bold", description="Font weight")
    
    # Output properties
    dpi: int = Field(default=300, ge=72, le=600, description="DPI for saved figures")
    bbox_inches: str = Field(default="tight", description="Bounding box for saved figures")


# Default style configuration
DEFAULT_STYLE = PlotStyleConfig()


def _setup_plot_style(title: str, style: PlotStyleConfig = DEFAULT_STYLE) -> None:
    """Setup common plot styling"""
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    if style.show_grid:
        plt.grid(True, zorder=-1)
    if style.equal_aspect:
        plt.axis('equal')


def _handle_output(title: str, out_path: Optional[Union[str, Path]] = None, 
                  style: PlotStyleConfig = DEFAULT_STYLE) -> None:
    """Handle plot output - save if out_path provided, otherwise show"""
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.is_dir():
            # If out_path is a directory, create filename from title
            filename = f"{title}.png"
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
        except:
            # If show fails (e.g., no display available), just close the figure
            plt.close()


def plot_graph(lg: nx.Graph, 
               title: str = "Link Graph", 
               out_path: Optional[Union[str, Path]] = None,
               style: PlotStyleConfig = DEFAULT_STYLE) -> None:
    """
    Plot a networkx graph representing the linkage system
    
    Args:
        lg: NetworkX graph with node attributes
        title: Plot title
        out_path: Output path for saving (None to display)
        style: Style configuration
    """
    plt.figure(figsize=(8, 6))
    
    colors = [style.fixed_node_color if lg.nodes[n].get('fixed', False) 
              else style.free_node_color for n in lg.nodes]
    
    options = {
        'node_color': colors,
        'node_size': style.node_size,
        'width': style.linewidth,
        'font_weight': style.font_weight,
        'alpha': style.alpha
    }
    
    nx.draw(lg, with_labels=True, **options)
    
    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)


def plot_static_arrows(links: List[Link],
                      i: int,
                      title: str = "Linkage Arrows",
                      out_path: Optional[Union[str, Path]] = None,
                      style: PlotStyleConfig = DEFAULT_STYLE) -> None:
    """
    Plot static arrows showing link directions at a specific time index
    
    Args:
        links: List of Link objects
        i: Time index to plot
        title: Plot title
        out_path: Output path for saving (None to display)
        style: Style configuration
    """
    plt.figure(figsize=(10, 8))
    
    link_colors = sns.color_palette(style.color_palette)
    link_color_cycle = cycle(link_colors)

    hs = links[0].length * 0.1
    for link in links:
        plt.arrow(
            x=link.pos1[i][0],
            y=link.pos1[i][1],
            dx=link.pos2[i][0] - link.pos1[i][0],
            dy=link.pos2[i][1] - link.pos1[i][1],
            color=next(link_color_cycle),
            head_width=hs,
            head_length=hs,
            linewidth=style.linewidth,
            label=link.name,
            alpha=style.alpha
        )
    
    if style.show_legend:
        plt.legend()
    
    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)


def plot_static_pos(links: List[Link],
                   times: np.ndarray,
                   title: str = "Linkage Positions",
                   out_path: Optional[Union[str, Path]] = None,
                   style: PlotStyleConfig = DEFAULT_STYLE,
                   show_paths: bool = True,
                   show_fixed_links: bool = True,
                   show_free_links: bool = True,
                   show_pivot_points: bool = True,
                   show_end_points: bool = False) -> None:
    """
    Plot static positions showing link paths over time
    
    Args:
        links: List of Link objects
        times: Time array
        title: Plot title
        out_path: Output path for saving (None to display)
        style: Style configuration
        show_paths: Show link paths
        show_fixed_links: Show fixed links
        show_free_links: Show free links
        show_pivot_points: Show pivot points
        show_end_points: Show end points
    """
    plt.figure(figsize=(12, 10))
    
    markers = ['o', 'p', 's', 'P', 'v', 'd', '*', 'h']
    phase_colors = plt.cm.get_cmap(style.colormap)
    phase_colors = plt.cm.get_cmap("Spectral")
    marker_cycle = cycle(markers)
    
    for i, time_sample in enumerate(times):
        #rotation_fraction = np.mod(time_sample, 2*np.pi) / (2*np.pi)
        rotation_fraction = np.mod(time_sample, 2*np.pi)
        marker = '.'
        
        for link in links:
            color = phase_colors(rotation_fraction)
            # print(color)
            
            if show_end_points:
                plt.scatter(link.pos2[i][0], link.pos2[i][1],
                           color=color, s=style.markersize, marker=marker, alpha=style.alpha)

            if show_paths and i > 0:
                plt.plot([link.pos2[i-1][0], link.pos2[i][0]],
                        [link.pos2[i-1][1], link.pos2[i][1]],
                        color=color, linewidth=style.linewidth, alpha=style.alpha)
                        
            if show_free_links and not link.has_fixed:
                plt.plot([link.pos1[i][0], link.pos2[i][0]],
                        [link.pos1[i][1], link.pos2[i][1]],
                        color=color, linewidth=style.linewidth, alpha=style.alpha)
            
            if show_fixed_links and link.has_fixed:
                plt.plot([link.pos1[i][0], link.pos2[i][0]],
                        [link.pos1[i][1], link.pos2[i][1]],
                        color=color, linewidth=style.linewidth, alpha=style.alpha)

    # Plot pivot points
    if show_pivot_points:
        link_colors = sns.color_palette(style.color_palette)
        link_color_cycle = cycle(link_colors)
        
        for link in links:
            marker = next(marker_cycle)
            color = next(link_color_cycle)
            
            if link.has_fixed and link.fixed_loc is not None:
                plt.scatter(link.fixed_loc[0], link.fixed_loc[1],
                           color=color, zorder=10, marker=marker, 
                           s=style.markersize, alpha=style.alpha, label=link.name)
        
        if style.show_legend:
            plt.legend()
    
    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)

