"""
Visualization utilities for design space exploration.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def render_contours(
    # New interface: scattered points
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    values: np.ndarray | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    x_label: str = 'Dimension 1',
    y_label: str = 'Dimension 2',
    value_label: str = 'Value',
    show_points: bool = True,
    point_labels: list[str] | np.ndarray | None = None,  # Labels like ["valid", "invalid", "epsilon_close"]
    # Legacy interface: gridded data
    transpose: bool = False,
    value_feature: str | None = None,
    value_min_max: tuple[float, float] | None = None,
    show_contour_lines: bool = True,
    show_contour: bool = True,
    qualitative_cmap: bool = True,
    n_contour_levels: int = 20,
    n_contour_line_levels: int = 10,
    fig=None,
    ax=None,
    x_axis_space: np.ndarray | None = None,
    y_axis_space: np.ndarray | None = None,
    verbose: int = 0,
    # Grid interpolation parameters
    grid_resolution: int = 100,
    interpolation_method: str = 'linear',  # usic cubic shows bad results for sparse scores
) -> None:
    """
    Render contours on a plot from either scattered points or gridded data.

    New interface (scattered points):
        x_coords: X coordinates of sample points (1D array)
        y_coords: Y coordinates of sample points (1D array)
        values: Values at each sample point (1D array). If None, contours are skipped.
        x_bounds: (min, max) bounds for x-axis
        y_bounds: (min, max) bounds for y-axis
        x_label: Label for x-axis
        y_label: Label for y-axis
        value_label: Label for colorbar
        show_points: Whether to scatter plot the sample points
        point_labels: Optional array of labels for each point (e.g., ["valid", "invalid", "epsilon_close"]).
                     If provided, points are colored/marked by label and a legend is added.

    Legacy interface (gridded data):
        x_axis_space: Grid X coordinates (2D array)
        y_axis_space: Grid Y coordinates (2D array)
        values: Grid values (2D array)
        transpose: Whether to transpose values
        value_feature: Label for colorbar (legacy)
        value_min_max: (min, max) for value range

    Args:
        grid_resolution: Number of grid points per dimension for interpolation
        interpolation_method: 'linear', 'cubic', or 'nearest' for griddata
    """
    import matplotlib.pyplot as plt

    # Determine which interface is being used
    use_scattered = (x_coords is not None and y_coords is not None)
    use_gridded = (
        x_axis_space is not None and y_axis_space is not None and
        values is not None and len(values.shape) == 2
    )

    if not use_scattered and not use_gridded:
        raise ValueError(
            'Must provide either scattered points (x_coords, y_coords) '
            'or gridded data (x_axis_space, y_axis_space, 2D values)',
        )

    # Store original scattered values for scatter plot (before interpolation)
    original_values = None
    original_x_coords = None
    original_y_coords = None
    original_point_labels = None

    if use_scattered:
        # Store original scattered values before interpolation
        original_x_coords = x_coords.copy()
        original_y_coords = y_coords.copy()
        if values is not None:
            original_values = values.copy()
        if point_labels is not None:
            original_point_labels = np.array(point_labels) if not isinstance(point_labels, np.ndarray) else point_labels.copy()

        # Set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Skip contour plotting if values is None
        if values is None:
            if verbose >= 1:
                print('Warning: values is None, skipping contour plotting')
            show_contour = False
            show_contour_lines = False
        else:
            # Interpolate scattered points to grid
            if x_bounds is None:
                x_bounds = (np.min(x_coords), np.max(x_coords))
            if y_bounds is None:
                y_bounds = (np.min(y_coords), np.max(y_coords))

            # Create grid
            x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
            y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

            # Interpolate values to grid (catch qhull errors)
            try:
                grid_values = griddata(
                    (x_coords, y_coords),
                    values,
                    (X_grid, Y_grid),
                    method=interpolation_method,
                    fill_value=np.nan,
                )

                # Use interpolated grid for plotting
                x_axis_space = X_grid
                y_axis_space = Y_grid
                values = grid_values
            except Exception as e:
                # Catch qhull errors and other interpolation errors
                error_msg = str(e).lower()
                if 'qhull' in error_msg or 'qh' in error_msg:
                    if verbose >= 1:
                        print(f'Warning: Qhull error during interpolation: {e}')
                        print('Skipping contour plotting. This often happens with insufficient or collinear points.')
                else:
                    if verbose >= 1:
                        print(f'Warning: Interpolation error: {e}')
                        print('Skipping contour plotting.')
                show_contour = False
                show_contour_lines = False
                values = None  # Mark as None to skip contour plotting

        # Set labels
        if value_feature is None:
            value_feature = value_label

    # Legacy gridded data path
    if values is not None and transpose:
        values = np.transpose(values)

    # Plot contours only if values are available
    if values is not None and (show_contour or show_contour_lines):
        # Determine value range, handling NaN values
        values_min = np.nanmin(values) if value_min_max is None else value_min_max[0]
        values_max = np.nanmax(values) if value_min_max is None else value_min_max[1]

        # Check for NaN or invalid ranges
        if np.isnan(values_min) or np.isnan(values_max) or not np.isfinite(values_min) or not np.isfinite(values_max):
            if verbose >= 1:
                print('Warning: NaN or invalid values detected in the data. Skipping contour plotting.')
                print(f'  values_min: {values_min}, values_max: {values_max}')
            # Skip contour plotting but continue with point plotting
            show_contour = False
            show_contour_lines = False
        elif values_min == values_max:
            if verbose >= 1:
                print('Warning: All values are the same. Skipping contour plotting.')
            show_contour = False
            show_contour_lines = False
        else:
            # Mask NaN values for contour plotting
            values_masked = np.ma.masked_invalid(values)

            # Check if we have any valid values after masking
            if np.ma.count_masked(values_masked) == values_masked.size:
                if verbose >= 1:
                    print('Warning: All values are NaN after masking. Skipping contour plotting.')
                show_contour = False
                show_contour_lines = False
            else:
                # Plot contour lines
                if show_contour_lines:
                    try:
                        contour_lines_keywords = {
                            'colors': 'white' if show_contour else 'black',
                            'linewidths': 1 if show_contour else 2,
                            'levels': np.linspace(values_min, values_max, n_contour_line_levels + 2),
                        }

                        contour_line_obj = ax.contour(
                            x_axis_space, y_axis_space, values_masked, **contour_lines_keywords,
                        )

                        contour_line_obj.set_clim(values_min, vmax=values_max)
                        if not show_contour:
                            _ = fig.colorbar(
                                contour_line_obj,
                                ax=ax,
                                fraction=0.02,
                                pad=0.1,
                                label=value_feature or value_label,
                            )
                    except (ValueError, RuntimeError) as e:
                        if verbose >= 1:
                            print(f'Warning: Error plotting contour lines: {e}. Skipping contour lines.')
                        show_contour_lines = False

                # Plot filled contours
                if show_contour:
                    try:
                        cmap = 'viridis' if not qualitative_cmap else 'RdYlGn_r'
                        contour_keywords = {
                            'cmap': cmap,
                            'vmin': values_min,
                            'vmax': values_max,
                            'levels': np.linspace(values_min, values_max, n_contour_levels + 1),
                        }

                        contour_obj = ax.contourf(
                            x_axis_space, y_axis_space, values_masked, **contour_keywords,
                        )
                        contour_obj.set_clim(values_min, vmax=values_max)
                        _ = fig.colorbar(
                            contour_obj, ax=ax, fraction=0.02, pad=0.1, label=value_feature or value_label,
                        )
                    except (ValueError, RuntimeError) as e:
                        if verbose >= 1:
                            print(f'Warning: Error plotting filled contours: {e}. Skipping filled contours.')
                        show_contour = False

    # Plot sample points if requested (for scattered interface)
    if use_scattered and show_points:
        if original_point_labels is not None:
            # Plot points by label with different colors/markers
            unique_labels = np.unique(original_point_labels)

            # Define color and marker mapping for common labels
            label_styles = {
                'valid': {'color': 'green', 'marker': 'o', 'facecolors': 'none', 'edgecolors': 'green', 's': 50},
                'invalid': {'color': 'red', 'marker': 'x', 'facecolors': 'red', 'edgecolors': 'red', 's': 30},
                'epsilon_close': {'color': 'blue', 'marker': 'o', 'facecolors': 'blue', 'edgecolors': 'blue', 's': 40},
                'good': {'color': 'green', 'marker': 'o', 'facecolors': 'none', 'edgecolors': 'green', 's': 50},
                'not_good': {'color': 'orange', 'marker': 'x', 'facecolors': 'orange', 'edgecolors': 'orange', 's': 30},
            }

            # Default style for unknown labels
            default_style = {'color': 'gray', 'marker': 'o', 'facecolors': 'gray', 'edgecolors': 'gray', 's': 30}

            for label in unique_labels:
                mask = original_point_labels == label
                style = label_styles.get(label.lower(), default_style)

                if 'facecolors' in style and style['facecolors'] == 'none':
                    # Open circle
                    ax.scatter(
                        original_x_coords[mask], original_y_coords[mask],
                        marker=style['marker'],
                        facecolors='none',
                        edgecolors=style['edgecolors'],
                        linewidths=1.5,
                        s=style['s'],
                        alpha=0.8,
                        label=label,
                        zorder=5,
                    )
                else:
                    # Filled marker
                    ax.scatter(
                        original_x_coords[mask], original_y_coords[mask],
                        marker=style['marker'],
                        color=style['color'],
                        s=style['s'],
                        linewidths=1.0,
                        alpha=0.6,
                        label=label,
                        zorder=5,
                    )

            # Add legend
            ax.legend(loc='upper right', fontsize=10)
        elif original_values is not None:
            # Plot points colored by values (original behavior)
            values_min = np.nanmin(original_values) if value_min_max is None else value_min_max[0]
            values_max = np.nanmax(original_values) if value_min_max is None else value_min_max[1]
            scatter = ax.scatter(
                original_x_coords, original_y_coords,
                c=original_values,
                s=30,
                edgecolors='black',
                linewidths=0.2,
                alpha=0.5,
                cmap='viridis',
                vmin=values_min,
                vmax=values_max,
                zorder=5,
            )
        else:
            # Just plot points without coloring
            ax.scatter(
                original_x_coords, original_y_coords,
                s=30,
                color='blue',
                alpha=0.5,
                zorder=5,
            )

    return None
