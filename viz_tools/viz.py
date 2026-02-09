from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from configs.link_models import Link
from viz_tools.viz_styling import _handle_output
from viz_tools.viz_styling import _setup_plot_style
from viz_tools.viz_styling import DEFAULT_PLOT_STYLE
from viz_tools.viz_styling import PlotStyleConfig


def plot_graph(
    lg: nx.Graph,
    title: str = 'Link Graph',
    out_path: str | Path | None = None,
    style: PlotStyleConfig = DEFAULT_STYLE,
) -> None:
    """
    Plot a networkx graph representing the linkage system

    Args:
        lg: NetworkX graph with node attributes
        title: Plot title
        out_path: Output path for saving (None to display)
        style: Style configuration
    """
    plt.figure(figsize=(8, 6))

    colors = [
        style.fixed_node_color if lg.nodes[n].get('fixed', False)
        else style.free_node_color for n in lg.nodes
    ]

    options = {
        'node_color': colors,
        'node_size': style.node_size,
        'width': style.linewidth,
        'font_weight': style.font_weight,
        'alpha': style.alpha,
    }

    nx.draw(lg, with_labels=True, **options)

    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)


def plot_static_arrows(
    links: list[Link],
    i: int,
    title: str = 'Linkage Arrows',
    out_path: str | Path | None = None,
    style: PlotStyleConfig = DEFAULT_STYLE,
) -> None:
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
            alpha=style.alpha,
        )

    if style.show_legend:
        # Only show legend if there are labeled artists
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()

    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)


def plot_static_pos(
    links: list[Link],
    times: np.ndarray,
    title: str = 'Linkage Positions',
    out_path: str | Path | None = None,
    style: PlotStyleConfig = DEFAULT_STYLE,
    show_paths: bool = True,
    show_fixed_links: bool = True,
    show_free_links: bool = True,
    show_pivot_points: bool = True,
    show_end_points: bool = False,
) -> None:
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
    phase_colors = plt.cm.get_cmap('Spectral')
    marker_cycle = cycle(markers)

    for i, time_sample in enumerate(times):
        # rotation_fraction = np.mod(time_sample, 2*np.pi) / (2*np.pi)
        rotation_fraction = np.mod(time_sample, 2*np.pi)
        marker = '.'

        for link in links:
            color = phase_colors(rotation_fraction)
            # print(color)

            if show_end_points:
                plt.scatter(
                    link.pos2[i][0], link.pos2[i][1],
                    color=color, s=style.markersize, marker=marker, alpha=style.alpha,
                )

            if show_paths and i > 0:
                plt.plot(
                    [link.pos2[i-1][0], link.pos2[i][0]],
                    [link.pos2[i-1][1], link.pos2[i][1]],
                    color=color, linewidth=style.linewidth, alpha=style.alpha,
                )

            if show_free_links and not link.has_fixed:
                plt.plot(
                    [link.pos1[i][0], link.pos2[i][0]],
                    [link.pos1[i][1], link.pos2[i][1]],
                    color=color, linewidth=style.linewidth, alpha=style.alpha,
                )

            if show_fixed_links and link.has_fixed:
                plt.plot(
                    [link.pos1[i][0], link.pos2[i][0]],
                    [link.pos1[i][1], link.pos2[i][1]],
                    color=color, linewidth=style.linewidth, alpha=style.alpha,
                )

    # Plot pivot points
    if show_pivot_points:
        link_colors = sns.color_palette(style.color_palette)
        link_color_cycle = cycle(link_colors)

        for link in links:
            marker = next(marker_cycle)
            color = next(link_color_cycle)

            if link.has_fixed and link.fixed_loc is not None:
                plt.scatter(
                    link.fixed_loc[0], link.fixed_loc[1],
                    color=color, zorder=10, marker=marker,
                    s=style.markersize, alpha=style.alpha, label=link.name,
                )

        if style.show_legend:
            # Only show legend if there are labeled artists
            if plt.gca().get_legend_handles_labels()[0]:
                plt.legend()

    _setup_plot_style(title, style)
    _handle_output(title, out_path, style)
