#!/usr/bin/env python3
"""
Trajectory Scoring Demo - Visualize how different scoring methods work.

WHAT THIS DEMO DOES:
====================
Demonstrates how different trajectory scoring methods handle:
1. Same shape (no transformation) vs rotation/phase differences
2. Rotation (same shape, different phase/rotation)

Test cases include identical shapes, rotation-only, scrambled order, and shape differences.

For each case, computes scores using all combinations of:
- phase_invariant: True/False
- translation_invariant: True/False
- phase_align_method: 'rotation', 'fft'
- metric: 'mse' and distance metrics (sspd, dtw, hausdorff, etc.)

Creates plots showing:
- Trajectories with points labeled 1-10
- Scoring method and results

WHY THIS MATTERS:
=================
- Shows how different scoring modes affect optimization
- Demonstrates when translation/phase invariance helps
- Helps choose the right scoring method for your use case

RUN THIS DEMO:
==============
    python demo/trajectory_scoring_demo.py

Output saved to: user/demo/trajectory_scoring/
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from configs.appconfig import USER_DIR
from configs.matplotlib_config import configure_matplotlib_for_backend

# Configure matplotlib BEFORE importing pyplot
configure_matplotlib_for_backend()

import matplotlib.pyplot as plt
import seaborn as sns
from demo.helpers import print_section
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.trajectory_scoring import score_trajectory


# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of points in the circle
N_POINTS = 16

# Circle radius
RADIUS = 1.0

# Rotation angle (in degrees) for cases that use rotation
ROTATION_ANGLE = 90.0

# Random seed for scrambling points in case 6
SCRAMBLE_SEED = 42

# Number of times to run each scorer for timing (can be changed)
TIMING_ITERATIONS = 10

# Output directory
OUTPUT_DIR = USER_DIR / 'demo' / 'trajectory_scoring'
TIMESTAMP = ""  # Set to datetime.now().strftime('%Y%m%d_%H%M%S') for timestamps


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_circle_points(n_points: int, radius: float, center: tuple[float, float] = (0, 0)) -> list[tuple[float, float]]:
    """Create a circle of points."""
    points = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points


def create_circle_nonuniform_density(n_points: int, radius: float, center: tuple[float, float] = (0, 0), 
                                     cluster_size: int = 6, cluster_spacing: float = 0.1) -> list[tuple[float, float]]:
    """
    Create a circle with non-uniform point density.
    
    The first cluster_size points (e.g., 1-6) are clustered closely together with spacing cluster_spacing,
    while the remaining points are evenly distributed around the rest of the circle.
    
    Args:
        n_points: Total number of points
        radius: Circle radius
        center: Center of the circle
        cluster_size: Number of points to cluster together (default 6)
        cluster_spacing: Spacing between clustered points along the circle (default 0.1)
    
    Returns:
        List of (x, y) points with non-uniform density
    """
    points = []
    
    # Start angle for the cluster (at angle 0, which is (radius, 0) in Cartesian)
    start_angle = 0.0
    
    # Create clustered points (1 to cluster_size)
    # These are spaced closely together along a small arc of the circle
    for i in range(cluster_size):
        # Calculate angle increment: cluster_spacing is the arc length, convert to angle
        angle_increment = cluster_spacing / radius
        angle = start_angle + i * angle_increment
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    # Calculate the arc angle used by the cluster
    cluster_arc_angle = (cluster_size - 1) * cluster_spacing / radius
    
    # Remaining points to distribute evenly around the rest of the circle
    remaining_points = n_points - cluster_size
    
    if remaining_points > 0:
        # Remaining arc is the full circle minus the cluster arc
        remaining_arc = 2 * math.pi - cluster_arc_angle
        angle_step = remaining_arc / remaining_points
        
        # Start the remaining points right after the cluster
        start_angle_remaining = start_angle + cluster_arc_angle
        
        for i in range(remaining_points):
            angle = start_angle_remaining + i * angle_step
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
    
    return points


def rotate_points(points: list[tuple[float, float]], angle_deg: float) -> list[tuple[float, float]]:
    """Rotate points around origin by angle_deg degrees."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated = []
    for x, y in points:
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        rotated.append((x_new, y_new))
    return rotated


def shift_points_phase(
    points: list[tuple[float, float]],
    phase_fraction: float,
) -> list[tuple[float, float]]:
    """
    Cyclically shift points so the trajectory starts phase_fraction through the loop.
    E.g. phase_fraction=0.5 gives a half-phase shift (candidate "starts" halfway along).
    """
    n = len(points)
    k = int(round(phase_fraction * n)) % n
    if k == 0:
        return list(points)
    return list(points[k:]) + list(points[:k])


def scramble_points(points: list[tuple[float, float]], seed: int | None = None) -> list[tuple[float, float]]:
    """
    Scramble the order of points (shuffle) while keeping the same points.
    
    Args:
        points: List of (x, y) points
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        List of same points in scrambled order
    """
    import random
    scrambled = list(points)  # Make a copy
    if seed is not None:
        random.seed(seed)
    random.shuffle(scrambled)
    return scrambled


def create_organic_trajectory(n_points: int, scale: float = 1.0, center: tuple[float, float] = (0, 0)) -> list[tuple[float, float]]:
    """
    Create an organic, flowing trajectory that looks natural and curved.
    
    Creates a figure-8-like path with smooth curves and natural variations.
    
    Args:
        n_points: Number of points in the trajectory
        scale: Scale factor for the trajectory size
        center: Center point of the trajectory
    
    Returns:
        List of (x, y) points forming an organic trajectory
    """
    points = []
    for i in range(n_points):
        t = 2 * math.pi * i / n_points
        
        # Create a figure-8 (lemniscate) with some organic variation
        # Base figure-8 shape
        x_base = scale * math.sin(t)
        y_base = scale * math.sin(t) * math.cos(t)
        
        # Add organic variations with multiple frequencies
        variation1 = 0.15 * scale * math.sin(3 * t)  # Higher frequency variation
        variation2 = 0.1 * scale * math.cos(5 * t)  # Another frequency
        
        x = center[0] + x_base + variation1
        y = center[1] + y_base + variation2
        
        points.append((x, y))
    
    return points


def create_bitten_circle_points(n_points: int, radius: float, center: tuple[float, float] = (0, 0), 
                                bite_size: float = 0.3) -> list[tuple[float, float]]:
    """
    Create a circle of points with a flat 'bite' taken out at the top.
    
    Creates n_points evenly spaced on a circle, but modifies some points
    to create a flat horizontal top, making the bite flatter.
    
    Args:
        n_points: Total number of points (should be 12 for this demo)
        radius: Circle radius
        center: Center of the circle
        bite_size: How much to indent the bite (fraction of radius)
    
    Returns:
        List of (x, y) points with a flat bite taken out at the top
    """
    points = []
    bite_start_idx = 2  # Start the bite at point index 2 (in first quadrant)
    bite_count = 5  # Modify 5 consecutive points for a flatter bite
    
    bite_radius = radius * (1 - bite_size)  # Inner radius for the bite
    flat_y = center[1] + bite_radius  # Y coordinate of the flat top
    
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        
        # Check if this point should be part of the bite
        if bite_start_idx <= i < bite_start_idx + bite_count:
            # Create a flat bite by placing points on a horizontal line
            # Calculate the x position to maintain spacing along the flat top
            # Map the bite indices to x positions from right to left
            bite_idx = i - bite_start_idx
            # Create evenly spaced x positions across the flat top
            # From right edge (bite_radius) to left edge (-bite_radius)
            x_span = 2 * bite_radius
            x_offset = bite_radius - (bite_idx / (bite_count - 1)) * x_span if bite_count > 1 else 0
            x = center[0] + x_offset
            y = flat_y
        else:
            # Normal circle point
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
        
        points.append((x, y))
    
    return points


def compute_all_scores(
    target: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    timing_iterations: int = TIMING_ITERATIONS,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Compute scores for the new table structure.
    
    Returns:
        Tuple of (error_scores_dict, error_timings_dict, distance_scores_dict, distance_timings_dict)
        where timing is in milliseconds
        Keys are structured as: "metric_name | no_phase_center" or "metric_name | phase_center"
    """
    target_traj = TargetTrajectory(joint_name="test", positions=target)
    candidate_list = candidate
    
    error_results = {}
    error_timings = {}
    distance_results = {}
    distance_timings = {}
    
    # Phase alignment methods for error metrics (rotation, fft)
    phase_methods = ['rotation', 'fft']
    
    # Distance metrics: all available distance metrics
    distance_metrics = [
        'sspd',
        'dtw',
        'hausdorff', 
        'frechet',
        'frechet_discrete',
        #'lcss',
        #'edr',
        'erp'
    ]
    
    # Compute error metrics - only MSE for each phase method
    for phase_method in phase_methods:
        # Value without phase/center
        key_base = f"mse | {phase_method} | no_phase_center"
        try:
            score = score_trajectory(
                traj1=target_traj,
                traj2=candidate_list,
                metric='mse',
                phase_invariant=False,
                phase_align_method=phase_method,  # Won't be used when phase_invariant=False
                translation_invariant=False,
            )
            error_results[key_base] = score
            
            # Time with phase/center OFF
            start_time = time.perf_counter()
            for _ in range(timing_iterations):
                score_trajectory(
                    traj1=target_traj,
                    traj2=candidate_list,
                    metric='mse',
                    phase_invariant=False,
                    phase_align_method=phase_method,
                    translation_invariant=False,
                )
            elapsed_ms = (time.perf_counter() - start_time) * 1000 / timing_iterations
            error_timings[key_base] = elapsed_ms
        except Exception as e:
            error_results[key_base] = float('inf')
            error_timings[key_base] = float('inf')
            print(f"Error computing {key_base}: {e}")
        
        # Value with phase alignment only (no centering)
        key_center_only = f"mse | {phase_method} | center_only"
        try:
            score = score_trajectory(
                traj1=target_traj,
                traj2=candidate_list,
                metric='mse',
                phase_invariant=True,
                phase_align_method=phase_method,
                translation_invariant=False,
            )
            error_results[key_center_only] = score
        except Exception as e:
            error_results[key_center_only] = float('inf')
            print(f"Error computing {key_center_only}: {e}")
    
    # Compute distance metrics
    for metric in distance_metrics:
        # Handle metric-specific kwargs
        metric_kwargs = {}
        if metric in ['lcss', 'edr']:
            metric_kwargs['eps'] = 200.0
        elif metric == 'erp':
            metric_kwargs['g'] = None
        
        # Value without phase/center
        key_base = f"{metric} | no_phase_center"
        try:
            distance = score_trajectory(
                traj1=target_traj,
                traj2=candidate_list,
                metric=metric,
                phase_invariant=False,
                phase_align_method='fft',  # Won't be used when phase_invariant=False
                translation_invariant=False,
                **metric_kwargs,
            )
            distance_results[key_base] = distance
            
            # Time with phase/center OFF
            start_time = time.perf_counter()
            for _ in range(timing_iterations):
                score_trajectory(
                    traj1=target_traj,
                    traj2=candidate_list,
                    metric=metric,
                    phase_invariant=False,
                    phase_align_method='fft',
                    translation_invariant=False,
                    **metric_kwargs,
                )
            elapsed_ms = (time.perf_counter() - start_time) * 1000 / timing_iterations
            distance_timings[key_base] = elapsed_ms
        except Exception as e:
            distance_results[key_base] = float('inf')
            distance_timings[key_base] = float('inf')
            print(f"Error computing {key_base}: {e}")
        
        # Value with phase alignment only (no centering)
        key_center_only = f"{metric} | center_only"
        try:
            distance = score_trajectory(
                traj1=target_traj,
                traj2=candidate_list,
                metric=metric,
                phase_invariant=True,
                phase_align_method='fft',
                translation_invariant=False,
                **metric_kwargs,
            )
            distance_results[key_center_only] = distance
        except Exception as e:
            distance_results[key_center_only] = float('inf')
            print(f"Error computing {key_center_only}: {e}")
    
    return error_results, error_timings, distance_results, distance_timings


def run_case(
    case_name: str,
    case_number: int,
    target: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    title: str,
    description: str,
) -> None:
    """
    Helper function to run a test case: compute scores and plot.
    
    Args:
        case_name: Short name for the case (e.g., 'case1_translation')
        case_number: Case number for display
        target: Target trajectory points
        candidate: Candidate trajectory points
        title: Plot title
        description: Description to print
    """
    print_section(f'Case {case_number}: {description}')
    print(f'  Target: {len(target)} points')
    print(f'  Candidate: {description.lower()}')
    
    # Compute all scores
    print('  Computing scores for all combinations...')
    print(f'  Timing each scorer {TIMING_ITERATIONS} times...')
    error_scores, error_timings, distance_scores, distance_timings = compute_all_scores(
        target, candidate, TIMING_ITERATIONS
    )
    print(f'  Computed {len(error_scores)} error scores and {len(distance_scores)} distance scores')
    if len(error_scores) == 0 and len(distance_scores) == 0:
        print('  WARNING: No scores computed!')
    else:
        if error_scores:
            print(f'  Sample error keys: {list(error_scores.keys())[:3]}')
        if distance_scores:
            print(f'  Sample distance keys: {list(distance_scores.keys())[:3]}')
    
    # Plot
    plot_trajectory_comparison(
        target=target,
        candidate=candidate,
        error_scores=error_scores,
        error_timings=error_timings,
        distance_scores=distance_scores,
        distance_timings=distance_timings,
        title=title,
        case_name=case_name,
        output_path=OUTPUT_DIR / f'{case_name}_{TIMESTAMP}.png',
    )


def plot_trajectory_comparison(
    target: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    error_scores: dict[str, float],
    error_timings: dict[str, float],
    distance_scores: dict[str, float],
    distance_timings: dict[str, float],
    title: str,
    case_name: str,
    output_path: Path,
):
    """Create a plot showing trajectories with labeled points and scores."""
    # Set seaborn style for elegant plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with main plot on left, two tables on right (stacked)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])  # Main plot spans both rows
    ax_error_table = fig.add_subplot(gs[0, 1])
    ax_error_table.axis('off')  # Hide axes for table subplot
    ax_distance_table = fig.add_subplot(gs[1, 1])
    ax_distance_table.axis('off')  # Hide axes for table subplot
    
    target_arr = np.array(target)
    candidate_arr = np.array(candidate)
    
    # Use elegant colors from seaborn palette
    target_color = sns.color_palette("husl", 8)[1]
    candidate_color = sns.color_palette("husl", 8)[4]
    
    # Plot trajectories: target solid, candidate dashed so both are visible when overlapping
    ax.plot(
        target_arr[:, 0], target_arr[:, 1],
        'o-', color=target_color, linewidth=2.5, markersize=10, linestyle='-',
        label='Target', alpha=0.8, zorder=2, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor=target_color,
    )
    ax.plot(
        candidate_arr[:, 0], candidate_arr[:, 1],
        's-', color=candidate_color, linewidth=2.5, markersize=10, linestyle='--',
        label='Candidate', alpha=0.8, zorder=2, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor=candidate_color,
    )
    
    # Label points with jiggled offsets so target vs candidate numbers are visible when overlapping
    # Use deterministic per-point offsets: target labels one side, candidate the other, with small variation by index
    for i in range(len(target)):
        # Jiggle: small offset variation by index so labels don't stack when points overlap
        jiggle_tx = (i % 5) * 2 - 4
        jiggle_ty = (i % 3) * 2 - 2
        # Target labels: offset toward upper-right, with jiggle
        ax.annotate(
            str(i + 1),
            (target_arr[i, 0], target_arr[i, 1]),
            xytext=(6 + jiggle_tx, 6 + jiggle_ty), textcoords='offset points',
            fontsize=10, color=target_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                     edgecolor=target_color, linewidth=1.5),
        )
        # Candidate labels: offset toward lower-left, with opposite jiggle so they separate from target
        ax.annotate(
            str(i + 1),
            (candidate_arr[i, 0], candidate_arr[i, 1]),
            xytext=(-6 - jiggle_tx, -6 - jiggle_ty), textcoords='offset points',
            fontsize=10, color=candidate_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                     edgecolor=candidate_color, linewidth=1.5),
        )
    
    # Set equal aspect, no grid
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('X Position', fontsize=12, fontweight='medium')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add elegant legend
    #ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
    #          edgecolor='gray', fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)
    
    # ============================================================================
    # Build Error Metrics Table
    # ============================================================================
    error_table_data = []
    error_headers = ['Method', 'MSE', 'MSE+phase', 'Time (ms)']
    
    # Process each phase method (rotation, fft)
    phase_methods = ['rotation', 'fft']
    error_table_raw_data = []  # Store raw values for percentile calculation
    
    for phase_method in phase_methods:
        # Get MSE without phase/center
        key_base = f"mse | {phase_method} | no_phase_center"
        key_center_only = f"mse | {phase_method} | center_only"
        key_time = key_base  # Time is measured with phase/center off
        
        if key_base in error_scores and key_center_only in error_scores:
            mse_base = error_scores[key_base]
            mse_center = error_scores[key_center_only]
            timing_ms = error_timings.get(key_time, 0.0)
            
            # Format method display
            method_display = phase_method.capitalize()
            
            # Format values
            mse_str = "ERROR" if (mse_base == float('inf') or np.isnan(mse_base)) else f"{mse_base:.3f}"
            mse_center_str = "ERROR" if (mse_center == float('inf') or np.isnan(mse_center)) else f"{mse_center:.3f}"
            time_str = "ERROR" if (timing_ms == float('inf') or np.isnan(timing_ms)) else f"{timing_ms:.3f}"
            
            # Store raw values for percentile calculation
            error_table_raw_data.append({
                'row': len(error_table_data),
                'mse_base': mse_base if (mse_base != float('inf') and not np.isnan(mse_base)) else None,
                'mse_center': mse_center if (mse_center != float('inf') and not np.isnan(mse_center)) else None,
                'time': timing_ms if (timing_ms != float('inf') and not np.isnan(timing_ms)) else None,
            })
            
            error_table_data.append([method_display, mse_str, mse_center_str, time_str])
    
    # Calculate percentiles for coloring
    # Collect all MSE values (both columns)
    all_mse_values = []
    for row_data in error_table_raw_data:
        if row_data['mse_base'] is not None:
            all_mse_values.append(row_data['mse_base'])
        if row_data['mse_center'] is not None:
            all_mse_values.append(row_data['mse_center'])
    
    # Collect all time values
    all_time_values = []
    for row_data in error_table_raw_data:
        if row_data['time'] is not None:
            all_time_values.append(row_data['time'])
    
    # Calculate 10th percentile thresholds
    mse_threshold = np.percentile(all_mse_values, 10) if all_mse_values else float('inf')
    time_threshold = np.percentile(all_time_values, 10) if all_time_values else float('inf')
    
    # Create error metrics table
    if error_table_data:
        error_table_data_display = error_table_data  # No need to remove flags anymore
        
        # Set column widths - make Method column wider
        error_col_widths = [0.25, 0.25, 0.30, 0.20]  # Method, MSE, MSE+phase, Time
        
        error_table = ax_error_table.table(
            cellText=error_table_data_display,
            colLabels=error_headers,
            cellLoc='left',
            loc='upper center',
            bbox=[0, 0, 1, 1],
            colWidths=error_col_widths,
        )
        
        error_table.auto_set_font_size(False)
        error_table.set_fontsize(10)
        error_table.scale(1, 2.0)
        
        # Style header row
        for i in range(len(error_headers)):
            cell = error_table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
        
        # Style data rows - color individual cells based on percentiles
        for row_idx, row_data in enumerate(error_table_raw_data):
            i = row_idx + 1  # Table row index (0 is header)
            
            # Method column (column 0) - no special coloring
            cell = error_table[(i, 0)]
            cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # MSE column (column 1)
            cell = error_table[(i, 1)]
            if row_data['mse_base'] is not None and row_data['mse_base'] <= mse_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # MSE+phase column (column 2)
            cell = error_table[(i, 2)]
            if row_data['mse_center'] is not None and row_data['mse_center'] <= mse_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # Time column (column 3)
            cell = error_table[(i, 3)]
            if row_data['time'] is not None and row_data['time'] <= time_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
        
        ax_error_table.set_title('Error Metrics\n(Green=Top 10%)', fontsize=11, fontweight='bold', pad=10)
    else:
        ax_error_table.text(0.5, 0.5, 'No error metrics data', 
                           transform=ax_error_table.transAxes,
                           ha='center', va='center', fontsize=10, color='red')
    
    # ============================================================================
    # Build Distance Metrics Table
    # ============================================================================
    distance_table_data = []
    distance_headers = ['Method', 'Distance', 'Distance+phase', 'Time (ms)']
    
    # Distance metrics to show
    distance_metrics = [
        ('sspd', 'SSPD'),
        ('dtw', 'DTW'),
        ('hausdorff', 'Hausdorff'),
        ('frechet', 'Frèchet'),
        ('frechet_discrete', 'Frèchet-D'),
        ('lcss', 'LCSS'),
        ('edr', 'EDR'),
        ('erp', 'ERP'),
    ]
    
    # Process each distance metric
    distance_table_raw_data = []  # Store raw values for percentile calculation
    
    for metric_key, metric_display in distance_metrics:
        # Get distance without phase/center
        key_base = f"{metric_key} | no_phase_center"
        key_center_only = f"{metric_key} | center_only"
        key_time = key_base  # Time is measured with phase/center off
        
        if key_base in distance_scores and key_center_only in distance_scores:
            dist_base = distance_scores[key_base]
            dist_center = distance_scores[key_center_only]
            timing_ms = distance_timings.get(key_time, 0.0)
            
            # Format method display
            method_display = metric_display
            
            # Format values
            dist_str = "ERROR" if (dist_base == float('inf') or np.isnan(dist_base)) else f"{dist_base:.3f}"
            dist_center_str = "ERROR" if (dist_center == float('inf') or np.isnan(dist_center)) else f"{dist_center:.3f}"
            time_str = "ERROR" if (timing_ms == float('inf') or np.isnan(timing_ms)) else f"{timing_ms:.3f}"
            
            # Store raw values for percentile calculation
            distance_table_raw_data.append({
                'row': len(distance_table_data),
                'dist_base': dist_base if (dist_base != float('inf') and not np.isnan(dist_base)) else None,
                'dist_center': dist_center if (dist_center != float('inf') and not np.isnan(dist_center)) else None,
                'time': timing_ms if (timing_ms != float('inf') and not np.isnan(timing_ms)) else None,
            })
            
            distance_table_data.append([method_display, dist_str, dist_center_str, time_str])
    
    # Calculate percentiles for coloring
    # Collect all distance values (both columns)
    all_distance_values = []
    for row_data in distance_table_raw_data:
        if row_data['dist_base'] is not None:
            all_distance_values.append(row_data['dist_base'])
        if row_data['dist_center'] is not None:
            all_distance_values.append(row_data['dist_center'])
    
    # Collect all time values
    all_distance_time_values = []
    for row_data in distance_table_raw_data:
        if row_data['time'] is not None:
            all_distance_time_values.append(row_data['time'])
    
    # Calculate 10th percentile thresholds
    distance_threshold = np.percentile(all_distance_values, 10) if all_distance_values else float('inf')
    distance_time_threshold = np.percentile(all_distance_time_values, 10) if all_distance_time_values else float('inf')
    
    # Create distance metrics table
    if distance_table_data:
        distance_table_data_display = distance_table_data  # No need to remove flags anymore
        
        # Set column widths - make Method column wider
        distance_col_widths = [0.25, 0.25, 0.30, 0.20]  # Method, Distance, Distance+phase, Time
        
        distance_table = ax_distance_table.table(
            cellText=distance_table_data_display,
            colLabels=distance_headers,
            cellLoc='left',
            loc='upper center',
            bbox=[0, 0, 1, 1],
            colWidths=distance_col_widths,
        )
        
        distance_table.auto_set_font_size(False)
        distance_table.set_fontsize(10)
        distance_table.scale(1, 2.0)
        
        # Style header row
        for i in range(len(distance_headers)):
            cell = distance_table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
        
        # Style data rows - color individual cells based on percentiles
        for row_idx, row_data in enumerate(distance_table_raw_data):
            i = row_idx + 1  # Table row index (0 is header)
            
            # Method column (column 0) - no special coloring
            cell = distance_table[(i, 0)]
            cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # Distance column (column 1)
            cell = distance_table[(i, 1)]
            if row_data['dist_base'] is not None and row_data['dist_base'] <= distance_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # Distance+phase column (column 2)
            cell = distance_table[(i, 2)]
            if row_data['dist_center'] is not None and row_data['dist_center'] <= distance_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
            
            # Time column (column 3)
            cell = distance_table[(i, 3)]
            if row_data['time'] is not None and row_data['time'] <= distance_time_threshold:
                cell.set_facecolor('#D4EDDA')  # Green for top 10%
            else:
                cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5F5F5')
            cell.set_edgecolor('#D0D0D0')
            cell.set_linewidth(1)
        
        ax_distance_table.set_title('Distance Metrics\n(Green=Top 10%)', fontsize=11, fontweight='bold', pad=10)
    else:
        ax_distance_table.text(0.5, 0.5, 'No distance metrics data', 
                               transform=ax_distance_table.transAxes,
                               ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print_section('TRAJECTORY SCORING DEMO')
    print(f'Output: {OUTPUT_DIR}')
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Case 1: Same shape (no transformation)
    # -------------------------------------------------------------------------
    target_circle = create_circle_points(N_POINTS, RADIUS, center=(0, 0))
    candidate_circle = create_circle_points(N_POINTS, RADIUS, center=(0, 0))
    
    run_case(
        case_name='case1_translation',
        case_number=1,
        target=target_circle,
        candidate=candidate_circle,
        title='Case 1: Same Shape\nTarget (red) vs Candidate (blue) - Identical circles',
        description='Same Shape (No Transformation)',
    )
    
    # -------------------------------------------------------------------------
    # Case 2: Rotation only
    # -------------------------------------------------------------------------
    target_circle2 = create_circle_points(N_POINTS, RADIUS, center=(0, 0))
    candidate_circle2 = rotate_points(target_circle2, ROTATION_ANGLE)
    
    run_case(
        case_name='case2_translation_rotation',
        case_number=2,
        target=target_circle2,
        candidate=candidate_circle2,
        title=f'Case 2: Rotation Only\nTarget (red) vs Candidate (blue) - Rotated {ROTATION_ANGLE}°',
        description='Rotation Only',
    )
    
    # -------------------------------------------------------------------------
    # Case 3: Non-circularly symmetric trajectory (bitten circle)
    # -------------------------------------------------------------------------
    target_bitten = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    candidate_bitten = rotate_points(target_bitten, ROTATION_ANGLE)
    
    run_case(
        case_name='case3_bitten_circle',
        case_number=3,
        target=target_bitten,
        candidate=candidate_bitten,
        title=f'Case 3: Non-Circularly Symmetric\nTarget (red) vs Candidate (blue) - Rotated {ROTATION_ANGLE}°',
        description='Non-Circularly Symmetric (Bitten Circle)',
    )
    
    # -------------------------------------------------------------------------
    # Case 4: Non-circularly symmetric trajectory (bitten circle) - same shape
    # -------------------------------------------------------------------------
    target_bitten4 = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    candidate_bitten4 = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    
    run_case(
        case_name='case4_bitten_circle_vertical',
        case_number=4,
        target=target_bitten4,
        candidate=candidate_bitten4,
        title='Case 4: Non-Circularly Symmetric - Same Shape\nTarget (red) vs Candidate (blue) - Identical bitten circles',
        description='Non-Circularly Symmetric (Bitten Circle) - Same Shape',
    )
    
    # -------------------------------------------------------------------------
    # Case 5: Non-circularly symmetric trajectory (bitten circle) - Rotation only
    # -------------------------------------------------------------------------
    target_bitten5 = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    candidate_bitten5 = rotate_points(target_bitten5, ROTATION_ANGLE)
    
    run_case(
        case_name='case5_bitten_circle_translation_rotation',
        case_number=5,
        target=target_bitten5,
        candidate=candidate_bitten5,
        title=f'Case 5: Non-Circularly Symmetric - Rotation Only\nTarget (red) vs Candidate (blue) - Rotated {ROTATION_ANGLE}°',
        description='Non-Circularly Symmetric (Bitten Circle) - Rotation Only',
    )
    
    # -------------------------------------------------------------------------
    # Case 6: Non-circularly symmetric trajectory (bitten circle) - Scrambled order
    # -------------------------------------------------------------------------
    target_bitten6 = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    target_bitten6_scrambled = scramble_points(target_bitten6, seed=SCRAMBLE_SEED)
    candidate_bitten6 = create_bitten_circle_points(12, RADIUS, center=(0, 0), bite_size=0.3)
    
    run_case(
        case_name='case6_bitten_circle_scrambled',
        case_number=6,
        target=target_bitten6_scrambled,
        candidate=candidate_bitten6,
        title=f'Case 6: Non-Circularly Symmetric - Scrambled Order\nTarget (red) vs Candidate (blue) - Same points, random order',
        description='Non-Circularly Symmetric (Bitten Circle) - Scrambled Order',
    )
    
    # -------------------------------------------------------------------------
    # Case 7: Organic trajectory vs geometric shape (very different)
    # -------------------------------------------------------------------------
    target_organic = create_organic_trajectory(N_POINTS, scale=1.5, center=(0, 0))
    candidate_circle7 = create_circle_points(N_POINTS, RADIUS * 1.2, center=(0, 0))
    
    run_case(
        case_name='case7_organic_vs_circle',
        case_number=7,
        target=target_organic,
        candidate=candidate_circle7,
        title=f'Case 7: Organic vs Geometric\nTarget (red) - Organic trajectory vs Candidate (blue) - Circle',
        description='Organic Trajectory vs Geometric Circle (Very Different)',
    )
    
    # -------------------------------------------------------------------------
    # Case 8: Non-uniform density circle - Rotation only
    # -------------------------------------------------------------------------
    target_nonuniform = create_circle_nonuniform_density(N_POINTS, RADIUS, center=(0, 0), cluster_size=6, cluster_spacing=0.1)
    candidate_nonuniform = rotate_points(target_nonuniform, ROTATION_ANGLE)
    
    run_case(
        case_name='case8_nonuniform_density',
        case_number=8,
        target=target_nonuniform,
        candidate=candidate_nonuniform,
        title=f'Case 8: Non-Uniform Density - Rotation Only\nTarget (red) vs Candidate (blue) - Points 1-6 clustered, Rotated {ROTATION_ANGLE}°',
        description='Non-Uniform Density Circle - Rotation Only',
    )
    
    # -------------------------------------------------------------------------
    # Case 9: Non-uniform density target vs uniform density candidate
    # -------------------------------------------------------------------------
    target_nonuniform9 = create_circle_nonuniform_density(N_POINTS, RADIUS, center=(0, 0), cluster_size=6, cluster_spacing=0.1)
    candidate_uniform9 = create_circle_points(N_POINTS, RADIUS, center=(0, 0))
    
    run_case(
        case_name='case9_nonuniform_target_uniform_candidate',
        case_number=9,
        target=target_nonuniform9,
        candidate=candidate_uniform9,
        title=f'Case 9: Non-Uniform Target vs Uniform Candidate\nTarget (red) - Points 1-6 clustered vs Candidate (blue) - Uniform density',
        description='Non-Uniform Density Target vs Uniform Density Candidate',
    )
    
    # -------------------------------------------------------------------------
    # Case 10: Organic trajectory - candidate half phase ahead of target
    # -------------------------------------------------------------------------
    target_organic10 = create_organic_trajectory(N_POINTS, scale=1.5, center=(0, 0))
    candidate_organic10 = shift_points_phase(target_organic10, 0.5)
    
    run_case(
        case_name='case10_organic_half_phase',
        case_number=10,
        target=target_organic10,
        candidate=candidate_organic10,
        title='Case 10: Organic - Half Phase Shift\nTarget (red) vs Candidate (blue) - Same shape, candidate start ½ phase ahead',
        description='Organic Trajectory - Candidate Half Phase Ahead',
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section('COMPLETE')
    print(f'\nOutput files saved to: {OUTPUT_DIR}')
    for f in sorted(OUTPUT_DIR.glob(f'*{TIMESTAMP}*')):
        print(f'  - {f.name}')
    
    print('\nKey Insights:')
    print('  - Phase-invariant scoring finds best alignment regardless of starting point')
    print('  - Different phase alignment methods (rotation, FFT) may give different results')
    print('  - Different metrics (MSE and distance metrics) emphasize different aspects')


if __name__ == '__main__':
    main()
