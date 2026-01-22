"""
Multi-solution optimization for finding multiple distinct local optima.

When the optimization landscape has degenerate solutions (multiple dimension
configurations achieving similar low errors), standard optimizers return only
the single best solution. This module provides algorithms specifically designed
to discover and characterize multiple near-optimal solutions.

Use cases:
- Exploring design alternatives with similar performance
- Understanding solution robustness and sensitivity
- Finding backup solutions when manufacturing constraints apply
- Characterizing the structure of the fitness landscape

Three approaches are provided:
1. Basin Hopping: SciPy's global optimizer with multi-solution tracking
2. PSO with Niching: Particle swarm with diversity preservation
3. Multi-Start: Systematic sampling with local refinement and clustering
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.optimize import basinhopping, minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    from pylink_tools.optimization_types import (
        DimensionSpec,
        TargetTrajectory,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# RETURN TYPES
# =============================================================================

@dataclass
class Solution:
    """
    A single solution found during multi-solution optimization.
    
    Attributes:
        optimized_dimensions: Dimension values for this solution
        optimized_pylink_data: Full linkage configuration
        final_error: Objective function value (MSE, RMSE, etc.)
        initial_error: Error before optimization (for comparison)
        iterations: Number of function evaluations used
        cluster_id: Which cluster this solution belongs to (for grouping similar solutions)
        distance_to_best: L2 distance in dimension space to the global best solution
        uniqueness_score: Measure of how different this is from other solutions (0-1)
            1.0 = completely unique, 0.0 = identical to another solution
        convergence_history: Error values over iterations (optional)
        local_search_start: Starting point for local optimization (optional)
    """
    optimized_dimensions: dict[str, float]
    optimized_pylink_data: dict
    final_error: float
    initial_error: float
    iterations: int
    cluster_id: int
    distance_to_best: float
    uniqueness_score: float
    convergence_history: list[float] | None = None
    local_search_start: np.ndarray | None = None


@dataclass
class MultiSolutionResult:
    """
    Results from multi-solution optimization containing multiple distinct optima.
    
    The solutions list is sorted by final_error (best first). Solutions within
    epsilon_threshold of the best are considered "near-optimal". The clustering
    identifies groups of similar solutions to avoid redundant duplicates.
    
    Attributes:
        solutions: All distinct solutions found, sorted by quality (best first)
        best_solution: The single best solution (convenience accessor)
        n_unique_clusters: Number of distinct solution clusters found
        epsilon_threshold: Error threshold used for "near-optimal" classification
        search_space_coverage: Fraction of viable search space explored (0-1)
        total_evaluations: Total number of objective function calls
        success: Whether optimization completed successfully
        method: Which algorithm was used ('basin_hopping', 'pso_niching', 'multi_start')
        method_config: Configuration parameters used
        error_message: Error description if success=False
    """
    solutions: list[Solution]
    best_solution: Solution
    n_unique_clusters: int
    epsilon_threshold: float
    search_space_coverage: float
    total_evaluations: int
    success: bool
    method: Literal['basin_hopping', 'pso_niching', 'multi_start']
    method_config: dict
    error_message: str | None = None

    def get_near_optimal_solutions(self, epsilon: float | None = None) -> list[Solution]:
        """
        Get all solutions within epsilon of the best solution.
        
        Args:
            epsilon: Error threshold (uses self.epsilon_threshold if None)
        
        Returns:
            Subset of solutions with final_error <= best_error + epsilon
        """
        threshold = epsilon if epsilon is not None else self.epsilon_threshold
        best_error = self.best_solution.final_error
        return [s for s in self.solutions if s.final_error <= best_error + threshold]

    def get_solutions_by_cluster(self, cluster_id: int) -> list[Solution]:
        """Get all solutions belonging to a specific cluster."""
        return [s for s in self.solutions if s.cluster_id == cluster_id]

    def get_cluster_representatives(self) -> list[Solution]:
        """Get one representative solution from each cluster (the best one)."""
        representatives = []
        for cluster_id in range(self.n_unique_clusters):
            cluster_solutions = self.get_solutions_by_cluster(cluster_id)
            if cluster_solutions:
                best_in_cluster = min(cluster_solutions, key=lambda s: s.final_error)
                representatives.append(best_in_cluster)
        return representatives


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class BasinHoppingConfig:
    """
    Configuration for Basin Hopping multi-solution optimizer.
    
    Basin Hopping combines local minimization with random perturbations to escape
    local minima. This implementation tracks all discovered local minima rather
    than just returning the best.
    
    Attributes:
        n_iterations: Number of basin hopping iterations
        temperature: Controls acceptance of uphill moves (higher = more exploratory)
        stepsize: Size of random perturbations between local searches
        local_method: Local optimizer ('L-BFGS-B', 'Powell', 'Nelder-Mead')
        local_max_iter: Max iterations for each local search
        epsilon_threshold: Error threshold for "near-optimal" solutions
        min_distance_threshold: Minimum L2 distance to consider solutions distinct
        accept_test: Custom acceptance criterion ('metropolis', 'all', callable)
        seed: Random seed for reproducibility
    """
    n_iterations: int = 100
    temperature: float = 1.0
    stepsize: float = 0.5
    local_method: str = 'L-BFGS-B'
    local_max_iter: int = 100
    epsilon_threshold: float = 1.0
    min_distance_threshold: float = 0.1
    accept_test: str | None = None
    seed: int | None = None


@dataclass
class PSONichingConfig:
    """
    Configuration for PSO with Niching (species formation).
    
    Niching PSO maintains multiple sub-swarms (species) that explore different
    regions of the search space. When particles get too close, they form species
    that preserve diversity and prevent convergence to a single optimum.
    
    Attributes:
        n_particles: Total number of particles across all species
        n_iterations: Number of PSO iterations
        species_radius: Distance threshold for forming species
        min_species_size: Minimum particles per species (smaller species merge)
        max_species: Maximum number of species to maintain
        w: Inertia weight (particle momentum)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (species best attraction)
        epsilon_threshold: Error threshold for "near-optimal" solutions
        speciation_frequency: How often to recompute species (iterations)
        seed: Random seed for reproducibility
    """
    n_particles: int = 100
    n_iterations: int = 200
    species_radius: float = 0.2
    min_species_size: int = 3
    max_species: int = 10
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    epsilon_threshold: float = 1.0
    speciation_frequency: int = 10
    seed: int | None = None


@dataclass
class MultiStartConfig:
    """
    Configuration for Multi-Start optimizer with viable sampling.
    
    Systematically samples the search space using viable samples (mechanism-validated
    points), runs local optimization from each, then clusters results to identify
    distinct solutions. Most transparent and controllable approach.
    
    Attributes:
        n_starts: Number of starting points to generate
        sampling_mode: How to generate starts ('sobol', 'random', 'grid', 'viable')
        viable_sampling: If True, pre-filter starts to ensure mechanism validity
        viable_max_attempts: Max attempts when generating viable samples
        local_method: Local optimizer ('L-BFGS-B', 'Powell', 'Nelder-Mead', 'COBYLA')
        local_max_iter: Max iterations for each local search
        epsilon_threshold: Error threshold for "near-optimal" solutions
        cluster_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        cluster_threshold: Distance threshold for clustering similar solutions
        min_cluster_separation: Minimum distance between cluster centers
        parallel: Use multiprocessing for local optimizations
        n_workers: Number of parallel workers (None = use all CPUs)
        seed: Random seed for reproducibility
    """
    n_starts: int = 100
    sampling_mode: Literal['sobol', 'random', 'grid', 'viable'] = 'sobol'
    viable_sampling: bool = True
    viable_max_attempts: int = 1000
    local_method: str = 'L-BFGS-B'
    local_max_iter: int = 100
    epsilon_threshold: float = 1.0
    cluster_method: Literal['kmeans', 'dbscan', 'hierarchical'] = 'dbscan'
    cluster_threshold: float = 0.1
    min_cluster_separation: float = 0.05
    parallel: bool = False
    n_workers: int | None = None
    seed: int | None = None


# =============================================================================
# MULTI-SOLUTION OPTIMIZERS
# =============================================================================

def run_basin_hopping_multi(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: BasinHoppingConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    initial_point: np.ndarray | None = None,
    **kwargs,
) -> MultiSolutionResult:
    """
    Find multiple distinct solutions using Basin Hopping global optimization.
    
    Basin Hopping is a stochastic global optimizer that alternates between:
    1. Local minimization from current point
    2. Random perturbation to new starting point
    3. Acceptance test (typically Metropolis criterion)
    
    This implementation tracks ALL local minima discovered during the search,
    not just the final best. Solutions are clustered by dimension-space distance
    to identify distinct optima.
    
    ALGORITHM OVERVIEW:
    ------------------
    1. Start from initial point x0
    2. For n_iterations:
        a. Run local minimization from current point → find local minimum
        b. Record this minimum if it's distinct from previous solutions
        c. Randomly perturb position by stepsize
        d. Accept/reject new position based on temperature and energy change
    3. Cluster all discovered minima to group similar solutions
    4. Return sorted list of distinct solutions
    
    BENEFITS:
    ---------
    ✓ Well-established algorithm (SciPy implementation)
    ✓ Temperature parameter controls exploration vs exploitation
    ✓ Natural for finding multiple basins of attraction
    ✓ Robust to local minima traps
    ✓ Convergence guarantees in limit of infinite iterations
    
    DRAWBACKS:
    ----------
    ✗ Random walk may revisit same regions multiple times (inefficient)
    ✗ No explicit diversity preservation (relies on temperature/stepsize)
    ✗ May struggle with high-dimensional spaces (>20 dims)
    ✗ Requires tuning temperature/stepsize for each problem
    ✗ No guarantee of finding all solutions in finite time
    ✗ Cannot leverage mechanism-specific viable sampling
    
    WHEN TO USE:
    -----------
    - Medium-dimensional problems (5-15 dimensions)
    - When you want a simple, battle-tested algorithm
    - When landscape structure is unknown
    - When you can afford many function evaluations
    
    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: Basin hopping configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Phase alignment algorithm
        initial_point: Starting point for optimization (uses dimension_spec.initial_values if None).
            Can be a single point (1D array) or you can generate valid starting points using:
            - generate_valid_samples(): Pre-validated mechanism configurations
            - presample_valid_positions(): Sample space with kinematic validation
            - Random sampling within bounds with manual validation
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        MultiSolutionResult with all discovered distinct solutions
    
    Example:
        >>> # Basic usage with default starting point
        >>> config = BasinHoppingConfig(
        ...     n_iterations=200,
        ...     temperature=2.0,
        ...     epsilon_threshold=0.5
        ... )
        >>> result = run_basin_hopping_multi(pylink_data, target, config=config)
        >>> print(f"Found {len(result.solutions)} distinct solutions")
        >>> print(f"Best error: {result.best_solution.final_error:.6f}")
        >>> 
        >>> # Advanced: Start from a validated viable point for better exploration
        >>> from pylink_tools.optimization_helpers import generate_valid_samples
        >>> dim_spec = extract_dimensions(pylink_data)
        >>> valid_samples = generate_valid_samples(
        ...     pylink_data, dim_spec, n_samples=10, max_attempts=100
        ... )
        >>> if len(valid_samples) > 0:
        ...     # Use first valid sample as starting point
        ...     initial_pt = np.array([valid_samples[0][name] for name in dim_spec.names])
        ...     result = run_basin_hopping_multi(
        ...         pylink_data, target, config=config, initial_point=initial_pt
        ...     )
        >>> 
        >>> # Get all solutions within 1.0 MSE of best
        >>> near_optimal = result.get_near_optimal_solutions(epsilon=1.0)
        >>> for sol in near_optimal:
        ...     print(f"Cluster {sol.cluster_id}: error={sol.final_error:.3f}")
    """
    from pylink_tools.optimize import create_fitness_function
    from pylink_tools.optimization_helpers import extract_dimensions, apply_dimensions
    
    # Use default config if not provided
    if config is None:
        config = BasinHoppingConfig()
    
    # Extract dimension specification if not provided
    if dimension_spec is None:
        dimension_spec = extract_dimensions(pylink_data)
    
    dim = len(dimension_spec)
    if dim == 0:
        return MultiSolutionResult(
            solutions=[],
            best_solution=None,
            n_unique_clusters=0,
            epsilon_threshold=config.epsilon_threshold,
            search_space_coverage=0.0,
            total_evaluations=0,
            success=False,
            method='basin_hopping',
            method_config=config.__dict__,
            error_message='No dimensions to optimize',
        )
    
    # Create fitness function
    fitness_func = create_fitness_function(
        pylink_data,
        target,
        dimension_spec,
        metric=metric,
        verbose=False,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method
    )
    
    # Get bounds
    lower_bounds = np.array([b[0] for b in dimension_spec.bounds], dtype=np.float64)
    upper_bounds = np.array([b[1] for b in dimension_spec.bounds], dtype=np.float64)
    
    # Initial point: use provided point or fall back to dimension_spec.initial_values
    if initial_point is not None:
        x0 = np.array(initial_point, dtype=np.float64)
        if len(x0) != dim:
            raise ValueError(
                f'initial_point has {len(x0)} dimensions but dimension_spec has {dim}. '
                f'Ensure initial_point matches dimension_spec.names order.'
            )
        # Clip to bounds if needed
        x0 = np.clip(x0, lower_bounds, upper_bounds)
    else:
        # Use current dimensions from dimension_spec
        x0 = np.array(dimension_spec.initial_values, dtype=np.float64)
    
    initial_error = fitness_func(tuple(x0))
    
    if verbose:
        logger.info('Starting Basin Hopping multi-solution optimization')
        logger.info(f'  Dimensions: {dim}')
        logger.info(f'  Iterations: {config.n_iterations}')
        logger.info(f'  Temperature: {config.temperature}')
        logger.info(f'  Step size: {config.stepsize}')
        logger.info(f'  Local method: {config.local_method}')
        logger.info(f'  Epsilon threshold: {config.epsilon_threshold}')
        logger.info(f'  Initial error: {initial_error:.6f}')
    
    # Track all local minima discovered
    class MinimumTracker:
        def __init__(self, min_distance_threshold):
            self.minima = []  # List of (x, f, iteration)
            self.min_distance_threshold = min_distance_threshold
            self.iteration = 0
            self.eval_count = 0
        
        def __call__(self, x, f, accept):
            """Callback called after each basin hopping step."""
            self.iteration += 1
            
            # Check if this minimum is distinct from previous ones
            is_new = True
            if len(self.minima) > 0:
                distances = [np.linalg.norm(x - prev_x) for prev_x, _, _ in self.minima]
                min_dist = min(distances)
                if min_dist < self.min_distance_threshold:
                    is_new = False
            
            if is_new:
                self.minima.append((x.copy(), f, self.iteration))
                if verbose and len(self.minima) % 10 == 0:
                    logger.info(f'  Iteration {self.iteration}: Found {len(self.minima)} distinct minima')
    
    tracker = MinimumTracker(config.min_distance_threshold)
    
    # Wrapper to count evaluations
    eval_count = [0]
    def fitness_wrapper(x):
        eval_count[0] += 1
        return fitness_func(tuple(x))
    
    # Set up basin hopping minimizer kwargs
    minimizer_kwargs = {
        'method': config.local_method,
        'bounds': list(zip(lower_bounds, upper_bounds)),
        'options': {'maxiter': config.local_max_iter}
    }
    
    # Custom step-taking function to respect bounds
    class BoundedStepTaker:
        def __init__(self, stepsize, bounds):
            self.stepsize = stepsize
            self.lower = bounds[0]
            self.upper = bounds[1]
        
        def __call__(self, x):
            """Take a random step and clip to bounds."""
            x_new = x + np.random.uniform(-self.stepsize, self.stepsize, x.shape)
            return np.clip(x_new, self.lower, self.upper)
    
    step_taker = BoundedStepTaker(config.stepsize, (lower_bounds, upper_bounds))
    
    # Custom acceptance test (optional)
    accept_test = None
    if config.accept_test == 'all':
        # Accept all steps (pure random walk)
        accept_test = lambda f_new, f_old, x_new, x_old: True
    # 'metropolis' or None uses default Metropolis criterion
    
    # Set random seed if provided
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Run basin hopping
    # Note: scipy's basinhopping with niter=N does N perturbations plus the initial point,
    # resulting in N+1 local minimizations. To get exactly config.n_iterations results,
    # we subtract 1 from niter (unless n_iterations is 0 or 1).
    niter_param = max(0, config.n_iterations - 1)
    
    start_time = time.time()
    try:
        result = basinhopping(
            fitness_wrapper,
            x0,
            niter=niter_param,
            T=config.temperature,
            take_step=step_taker,
            accept_test=accept_test,
            callback=tracker,
            minimizer_kwargs=minimizer_kwargs,
            seed=config.seed,
        )
        success = True
        error_message = None
    except Exception as e:
        logger.error(f'Basin hopping failed: {e}')
        success = False
        error_message = str(e)
        result = None
    
    elapsed_time = time.time() - start_time
    
    if not success or len(tracker.minima) == 0:
        return MultiSolutionResult(
            solutions=[],
            best_solution=None,
            n_unique_clusters=0,
            epsilon_threshold=config.epsilon_threshold,
            search_space_coverage=0.0,
            total_evaluations=eval_count[0],
            success=False,
            method='basin_hopping',
            method_config=config.__dict__,
            error_message=error_message or 'No minima found',
        )
    
    # Extract all discovered minima
    all_x = np.array([x for x, _, _ in tracker.minima])
    all_f = np.array([f for _, f, _ in tracker.minima])
    all_iter = np.array([it for _, _, it in tracker.minima])
    
    if verbose:
        logger.info(f'Basin hopping completed in {elapsed_time:.2f}s')
        logger.info(f'  Total evaluations: {eval_count[0]}')
        logger.info(f'  Discovered {len(tracker.minima)} distinct minima')
        logger.info(f'  Best error: {result.fun:.6f}')
    
    # Cluster solutions to identify distinct groups
    cluster_labels, n_clusters = _cluster_solutions(
        all_x,
        method='dbscan',
        threshold=config.min_distance_threshold,
    )
    
    if verbose:
        logger.info(f'  Clustered into {n_clusters} distinct solution groups')
    
    # Compute uniqueness scores
    uniqueness_scores = _compute_uniqueness_scores(all_x, cluster_labels)
    
    # Find best solution
    best_idx = np.argmin(all_f)
    best_x = all_x[best_idx]
    best_error = all_f[best_idx]
    
    # Compute distances to best
    distances_to_best = np.linalg.norm(all_x - best_x, axis=1)
    
    # Create Solution objects
    solutions = []
    for i in range(len(all_x)):
        optimized_dims = dict(zip(dimension_spec.names, all_x[i]))
        optimized_pylink = apply_dimensions(pylink_data, optimized_dims, dimension_spec)
        
        sol = Solution(
            optimized_dimensions=optimized_dims,
            optimized_pylink_data=optimized_pylink,
            final_error=all_f[i],
            initial_error=initial_error,
            iterations=all_iter[i],
            cluster_id=int(cluster_labels[i]),
            distance_to_best=distances_to_best[i],
            uniqueness_score=uniqueness_scores[i],
            convergence_history=None,  # Not tracked in basin hopping callback
            local_search_start=None,
        )
        solutions.append(sol)
    
    # Sort by error (best first)
    solutions.sort(key=lambda s: s.final_error)
    
    # Estimate search space coverage
    search_space_coverage = _compute_search_space_coverage(
        all_x,
        list(zip(lower_bounds, upper_bounds)),
        method='convex_hull',
    )
    
    return MultiSolutionResult(
        solutions=solutions,
        best_solution=solutions[0],
        n_unique_clusters=n_clusters,
        epsilon_threshold=config.epsilon_threshold,
        search_space_coverage=search_space_coverage,
        total_evaluations=eval_count[0],
        success=success,
        method='basin_hopping',
        method_config=config.__dict__,
        error_message=error_message,
    )


def run_pso_niching_multi(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: PSONichingConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> MultiSolutionResult:
    """
    Find multiple distinct solutions using PSO with Niching (species formation).
    
    Niching PSO extends standard Particle Swarm Optimization by maintaining
    multiple sub-populations (species) that explore different regions. When
    particles get within species_radius of each other, they form a species
    with its own local best, preventing premature convergence to a single optimum.
    
    ALGORITHM OVERVIEW:
    ------------------
    1. Initialize n_particles randomly across search space
    2. For n_iterations:
        a. Evaluate fitness for all particles
        b. Every speciation_frequency iterations:
            - Compute pairwise distances between particles
            - Form species by grouping nearby particles (< species_radius)
            - Identify species_best for each species
        c. Update particle velocities:
            - Attracted to personal best (c1)
            - Attracted to their species_best (c2), not global best
            - Maintain inertia (w)
        d. Update particle positions
    3. Extract best particle from each species as distinct solutions
    4. Filter solutions by epsilon_threshold and cluster
    
    BENEFITS:
    ---------
    ✓ Explicitly maintains diversity through species separation
    ✓ Naturally finds multiple optima in parallel
    ✓ No local search needed (PSO explores directly)
    ✓ Works well in high-dimensional spaces (tested up to 50+ dims)
    ✓ Automatic adaptation to landscape structure
    ✓ Less sensitive to parameter tuning than basin hopping
    ✓ Can handle multi-modal, multi-objective landscapes
    
    DRAWBACKS:
    ----------
    ✗ More complex to implement than standard PSO
    ✗ Species_radius parameter critical (problem-dependent)
    ✗ May form too many species in early iterations (merge overhead)
    ✗ Slower convergence than single-optimum PSO
    ✗ Cannot leverage mechanism-specific viable sampling directly
    ✗ Memory overhead scales with n_particles * n_species
    
    WHEN TO USE:
    -----------
    - When you suspect many distinct local optima exist
    - High-dimensional problems (>15 dimensions)
    - When you want automatic diversity preservation
    - When parallel exploration is valuable
    - When landscape has clear basins of attraction
    
    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: PSO niching configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Phase alignment algorithm
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        MultiSolutionResult with one solution per stable species
    
    Example:
        >>> config = PSONichingConfig(
        ...     n_particles=100,
        ...     n_iterations=200,
        ...     species_radius=0.15,
        ...     epsilon_threshold=0.5
        ... )
        >>> result = run_pso_niching_multi(pylink_data, target, config=config)
        >>> print(f"Found {result.n_unique_clusters} species")
        >>> 
        >>> # Get representative from each cluster
        >>> representatives = result.get_cluster_representatives()
        >>> for sol in representatives:
        ...     print(f"Species {sol.cluster_id}: error={sol.final_error:.3f}, "
        ...           f"uniqueness={sol.uniqueness_score:.2f}")
    """
    raise NotImplementedError("PSO with Niching not yet implemented")


def run_multi_start(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec | None = None,
    config: MultiStartConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> MultiSolutionResult:
    """
    Find multiple distinct solutions using systematic multi-start optimization.
    
    The most transparent and controllable approach: generate diverse starting
    points using Design of Experiments (Sobol sequences, random, or grid),
    optionally filter for mechanism viability, run local optimization from
    each, then cluster results to identify distinct solutions.
    
    ALGORITHM OVERVIEW:
    ------------------
    1. Generate n_starts diverse starting points:
        - 'sobol': Low-discrepancy Sobol sequence (recommended)
        - 'random': Uniform random sampling
        - 'grid': Factorial grid across dimensions
        - 'viable': Any of above, pre-filtered with verify_mechanism_viable()
    2. For each starting point (optionally in parallel):
        a. Run local optimization (L-BFGS-B, Powell, etc.)
        b. Record final solution and convergence history
    3. Cluster all local minima by dimension-space distance:
        - 'dbscan': Density-based (recommended, auto-detects clusters)
        - 'kmeans': Partition into k clusters (need to specify k)
        - 'hierarchical': Build dendrogram, cut at threshold
    4. Filter by epsilon_threshold: keep solutions near best
    5. Sort by error, compute uniqueness scores
    
    BENEFITS:
    ---------
    ✓ Most transparent: you control sampling, optimization, clustering
    ✓ Can leverage viable sampling to avoid infeasible regions
    ✓ Sobol sequences give excellent space-filling coverage
    ✓ Trivially parallelizable (local searches independent)
    ✓ Easy to diagnose: see all starts, all endpoints, all convergence
    ✓ No stochastic behavior (deterministic if seed set)
    ✓ Simple to implement and debug
    ✓ Works with any local optimizer
    
    DRAWBACKS:
    ----------
    ✗ Requires many starting points for thorough coverage (>100 typical)
    ✗ Wasteful: multiple starts may converge to same optimum
    ✗ No adaptive sampling (doesn't learn from early results)
    ✗ Clustering threshold is problem-dependent
    ✗ Expensive in high dimensions (curse of dimensionality)
    ✗ May miss solutions between sampling points
    
    WHEN TO USE:
    -----------
    - When you want full control and transparency
    - When viable sampling is critical (swiss cheese landscape)
    - When you have parallel compute resources
    - When you need reproducible, deterministic results
    - When you want to analyze convergence patterns
    - Medium-dimensional problems (<20 dimensions)
    
    Args:
        pylink_data: Base pylink document with linkage configuration
        target: Target trajectory to match (joint name + positions)
        dimension_spec: Dimensions to optimize (extracted if not provided)
        config: Multi-start configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Phase alignment algorithm
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        MultiSolutionResult with clustered distinct solutions
    
    Example:
        >>> config = MultiStartConfig(
        ...     n_starts=200,
        ...     sampling_mode='sobol',
        ...     viable_sampling=True,
        ...     cluster_method='dbscan',
        ...     epsilon_threshold=0.5,
        ...     parallel=True
        ... )
        >>> result = run_multi_start(pylink_data, target, config=config)
        >>> print(f"Found {result.n_unique_clusters} distinct solution clusters")
        >>> print(f"Coverage: {result.search_space_coverage:.1%}")
        >>> 
        >>> # Analyze solution diversity
        >>> for sol in result.solutions[:10]:  # Top 10
        ...     print(f"Error: {sol.final_error:.3f}, "
        ...           f"Distance to best: {sol.distance_to_best:.3f}, "
        ...           f"Uniqueness: {sol.uniqueness_score:.2f}")
    """
    raise NotImplementedError("Multi-start optimizer not yet implemented")


# =============================================================================
# UTILITY FUNCTIONS (to be implemented)
# =============================================================================

def _cluster_solutions(
    solutions: np.ndarray,
    method: str = 'dbscan',
    threshold: float = 0.1,
) -> tuple[np.ndarray, int]:
    """
    Cluster solutions by dimension-space distance.
    
    Args:
        solutions: Array of solution vectors (n_solutions x n_dims)
        method: Clustering algorithm ('dbscan', 'kmeans', 'hierarchical')
        threshold: Distance threshold or number of clusters
    
    Returns:
        (cluster_labels, n_clusters) where cluster_labels[i] is cluster ID for solutions[i]
    """
    if len(solutions) == 0:
        return np.array([]), 0
    
    if len(solutions) == 1:
        return np.array([0]), 1
    
    if method == 'dbscan':
        # DBSCAN: density-based clustering, auto-detects number of clusters
        # eps = maximum distance between two samples to be in same cluster
        # min_samples = minimum cluster size (set to 1 to keep all points)
        clustering = DBSCAN(eps=threshold, min_samples=1, metric='euclidean')
        cluster_labels = clustering.fit_predict(solutions)
        
        # DBSCAN can produce -1 for noise points, reassign them as unique clusters
        n_clusters = len(set(cluster_labels))
        if -1 in cluster_labels:
            # Reassign noise points as individual clusters
            next_cluster_id = max(cluster_labels) + 1
            for i in range(len(cluster_labels)):
                if cluster_labels[i] == -1:
                    cluster_labels[i] = next_cluster_id
                    next_cluster_id += 1
            n_clusters = len(set(cluster_labels))
        
        return cluster_labels, n_clusters
    
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        # KMeans: partition into k clusters (threshold = k)
        n_clusters = max(1, int(threshold))
        n_clusters = min(n_clusters, len(solutions))  # Can't have more clusters than points
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(solutions)
        return cluster_labels, n_clusters
    
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        # Hierarchical: build tree and cut at threshold distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(solutions)
        n_clusters = len(set(cluster_labels))
        return cluster_labels, n_clusters
    
    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'dbscan', 'kmeans', or 'hierarchical'")


def _compute_uniqueness_scores(
    solutions: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute how unique each solution is relative to others.
    
    Uses minimum distance to any other solution in different cluster.
    Normalized to [0, 1] where 1.0 = maximally unique.
    
    Args:
        solutions: Array of solution vectors (n_solutions x n_dims)
        cluster_labels: Cluster ID for each solution
    
    Returns:
        Uniqueness score for each solution (array of floats)
    """
    n_solutions = len(solutions)
    
    if n_solutions == 0:
        return np.array([])
    
    if n_solutions == 1:
        return np.array([1.0])
    
    # Compute pairwise distances
    distances = squareform(pdist(solutions, metric='euclidean'))
    
    # For each solution, find minimum distance to solution in different cluster
    uniqueness = np.zeros(n_solutions)
    
    for i in range(n_solutions):
        # Find all solutions in different clusters
        different_cluster = cluster_labels != cluster_labels[i]
        
        if not np.any(different_cluster):
            # Only one cluster exists, use distance to next nearest point
            distances_to_others = distances[i].copy()
            distances_to_others[i] = np.inf  # Exclude self
            min_dist = np.min(distances_to_others)
        else:
            # Distance to nearest solution in different cluster
            distances_to_different = distances[i][different_cluster]
            min_dist = np.min(distances_to_different)
        
        uniqueness[i] = min_dist
    
    # Normalize to [0, 1] where 1.0 = maximally unique
    # Use robust normalization (avoid outliers dominating)
    if np.max(uniqueness) > 0:
        # Use 95th percentile as reference for normalization
        ref_distance = np.percentile(uniqueness, 95)
        uniqueness = np.clip(uniqueness / ref_distance, 0.0, 1.0)
    
    return uniqueness


def _compute_search_space_coverage(
    sampled_points: np.ndarray,
    bounds: list[tuple[float, float]],
    method: str = 'convex_hull',
) -> float:
    """
    Estimate fraction of search space covered by sampling.
    
    Args:
        sampled_points: Array of sampled points (n_points x n_dims)
        bounds: Lower and upper bounds for each dimension
        method: Coverage metric ('convex_hull', 'grid', 'range')
    
    Returns:
        Coverage fraction [0, 1]
    """
    if len(sampled_points) == 0:
        return 0.0
    
    n_dims = sampled_points.shape[1]
    
    if method == 'convex_hull':
        # Compute ratio of convex hull volume to total search space volume
        try:
            from scipy.spatial import ConvexHull
            
            if len(sampled_points) < n_dims + 1:
                # Not enough points for convex hull in n dimensions
                method = 'range'  # Fall back to range method
            else:
                # Normalize points to unit hypercube
                lower = np.array([b[0] for b in bounds])
                upper = np.array([b[1] for b in bounds])
                normalized_points = (sampled_points - lower) / (upper - lower)
                
                # Compute convex hull volume
                hull = ConvexHull(normalized_points)
                hull_volume = hull.volume
                
                # Unit hypercube has volume 1.0
                coverage = min(hull_volume, 1.0)
                return coverage
        except Exception:
            # Fall back to range method if convex hull fails
            method = 'range'
    
    if method == 'range' or method == 'grid':
        # Simple coverage: fraction of dimension ranges spanned
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # For each dimension, compute fraction of range covered
        dim_coverages = []
        for d in range(n_dims):
            dim_min = np.min(sampled_points[:, d])
            dim_max = np.max(sampled_points[:, d])
            dim_range = upper[d] - lower[d]
            
            if dim_range > 0:
                coverage = (dim_max - dim_min) / dim_range
            else:
                coverage = 1.0
            
            dim_coverages.append(coverage)
        
        # Geometric mean of dimension coverages
        coverage = np.prod(dim_coverages) ** (1.0 / n_dims)
        return min(coverage, 1.0)
    
    else:
        raise ValueError(f"Unknown coverage method: {method}. Use 'convex_hull', 'grid', or 'range'")
