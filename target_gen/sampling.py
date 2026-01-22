"""
sampling.py - Design of Experiments (DOE) and sampling utilities for mechanism optimization.

This module provides various sampling strategies for exploring mechanism design spaces:
- Full combinatoric grids (Cartesian product)
- Box-Behnken designs (response surface modeling)
- Sobol sequences (quasi-random low-discrepancy)
- Viable sample filtering (mechanism-aware sampling)

Functions are organized into three categories:
1. Basic sampling: Generate sample points in design space
2. Validation: Filter samples based on mechanism viability
3. Pre-sampling for optimization: Generate valid starting points

Main functions:
    - get_combinatoric_gradations(): Generate evenly-spaced values per dimension
    - get_mech_variations(): Cartesian product of gradations
    - get_mech_variations_from_spec(): DOE sampling from DimensionSpec
    - presample_valid_positions(): Pre-sample and score configurations
    - generate_viable_sobol_samples(): Generate valid Sobol samples for MLSL
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylink_tools.optimization_types import DimensionSpec, TargetTrajectory

logger = logging.getLogger(__name__)


# =============================================================================
# BASIC SAMPLING FUNCTIONS
# =============================================================================

def get_combinatoric_gradations(
    names: list[str],
    bounds: list[tuple[float, float]],
    n: int,
) -> dict[str, list[float]]:
    """
    Generate N evenly-spaced values for each link within its bounds.

    Creates a dictionary where each key is a link name and each value
    is a list of N values ranging from the lower to upper bound.

    Args:
        names: List of link/dimension names
        bounds: List of (lower, upper) tuples, one per name
        n: Number of gradations to generate for each link

    Returns:
        Dict mapping link name -> list of N values from low to high

    Example:
        >>> names = ['crank', 'coupler', 'rocker']
        >>> bounds = [(10, 20), (30, 50), (15, 25)]
        >>> gradations = get_combinatoric_gradations(names, bounds, n=3)
        >>> gradations
        {
            'crank': [10.0, 15.0, 20.0],
            'coupler': [30.0, 40.0, 50.0],
            'rocker': [15.0, 20.0, 25.0]
        }
    """
    if len(names) != len(bounds):
        raise ValueError(f'Length mismatch: {len(names)} names vs {len(bounds)} bounds')

    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')

    gradations = {}
    for name, (lower, upper) in zip(names, bounds):
        if n == 1:
            # Single value: use midpoint
            gradations[name] = [(lower + upper) / 2]
        else:
            # N evenly-spaced values from lower to upper (inclusive)
            gradations[name] = list(np.linspace(lower, upper, n))

    return gradations


def get_mech_variations(
    gradations: dict[str, list[float]],
) -> list[dict[str, float]]:
    """
    Generate all combinatorial variations of mechanism dimensions.

    Takes the gradation values for each link and produces the Cartesian
    product of all combinations.

    Args:
        gradations: Dict from get_combinatoric_gradations, mapping link name -> list of values

    Returns:
        List of dicts, each representing one mechanism configuration.
        Each dict maps link name -> specific value.

    Example:
        >>> gradations = {
        ...     'crank': [10.0, 20.0],
        ...     'coupler': [30.0, 40.0],
        ... }
        >>> variations = get_mech_variations(gradations)
        >>> len(variations)
        4
        >>> variations[0]
        {'crank': 10.0, 'coupler': 30.0}
        >>> variations[3]
        {'crank': 20.0, 'coupler': 40.0}

    Note:
        The number of variations grows exponentially: N^(num_links).
        For 3 links with N=10, you get 1000 variations.
        For 5 links with N=10, you get 100,000 variations.
    """
    import itertools

    if not gradations:
        return []

    # Get ordered list of names and their value lists
    names = list(gradations.keys())
    value_lists = [gradations[name] for name in names]

    # Generate Cartesian product of all value combinations
    variations = []
    for combo in itertools.product(*value_lists):
        variation = dict(zip(names, combo))
        variations.append(variation)

    return variations


def get_mech_variations_from_spec(
    spec: DimensionSpec,
    n: int,
    mode: str = 'full_combinatoric',
    center: int | None = None,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """
    Generate mechanism variations from a DimensionSpec using various sampling strategies.

    Args:
        spec: DimensionSpec containing names and bounds
        n: Meaning depends on mode:
           - 'full_combinatoric': Number of gradations per dimension (N^d total points)
           - 'behnken': Number of center points (ignored if center is specified)
           - 'sobol': Total number of sample points to generate
        mode: Sampling strategy, one of:
           - 'full_combinatoric': Cartesian product of evenly-spaced values.
                 WARNING: Grows as N^d (e.g., 5 dims × 10 gradations = 100,000 points)
           - 'behnken': Box-Behnken design. Efficient for 3+ factors,
                 generates ~2*d*(d-1) + center points. Good for response surface modeling.
           - 'sobol': Sobol quasi-random sequence. Low-discrepancy
                 sampling that fills space uniformly. Good for global exploration.
        center: For 'behnken' mode only - number of center points (default: auto)
        seed: For 'sobol' mode only - random seed for reproducibility

    Returns:
        List of mechanism configuration dicts

    Example:
        >>> spec = extract_dimensions(pylink_data)
        >>> # Full combinatoric (careful - grows fast!)
        >>> variations = get_mech_variations_from_spec(spec, n=5, mode='full_combinatoric')
        >>> # Box-Behnken (efficient for response surfaces)
        >>> variations = get_mech_variations_from_spec(spec, n=3, mode='behnken')
        >>> # Sobol sequence (good space coverage)
        >>> variations = get_mech_variations_from_spec(spec, n=100, mode='sobol')

    References:
    - Box-Behnken: https://pydoe3.readthedocs.io/en/latest/reference/response_surface/
    - Sobol: https://pydoe3.readthedocs.io/en/latest/reference/low_discrepancy_sequences/
    """

    num_dims = len(spec.names)
    if num_dims == 0:
        return []

    # Convert bounds to numpy arrays for easier manipulation
    lower_bounds = np.array([b[0] for b in spec.bounds])
    upper_bounds = np.array([b[1] for b in spec.bounds])

    if mode == 'full_combinatoric':
        # Original behavior: Cartesian product of gradations
        gradations = get_combinatoric_gradations(spec.names, spec.bounds, n)
        return get_mech_variations(gradations)

    elif mode == 'behnken':
        # Box-Behnken design - efficient for response surface modeling
        if num_dims < 3:
            logger.warning(
                f'Box-Behnken requires at least 3 factors, got {num_dims}. '
                'Falling back to full_combinatoric.',
            )
            gradations = get_combinatoric_gradations(spec.names, spec.bounds, n)
            return get_mech_variations(gradations)

        try:
            from pyDOE3 import bbdesign
        except ImportError as e:
            logger.error(
                f'pyDOE3 not installed. Install with: pip install pyDOE3. Error: {e}',
            )
            raise ImportError(
                "pyDOE3 is required for 'behnken' mode. Install with: pip install pyDOE3",
            ) from e

        try:
            # Generate Box-Behnken design (returns values in [-1, 1])
            if center is not None:
                design = bbdesign(num_dims, center=center)
            else:
                design = bbdesign(num_dims)

            # Scale from [-1, 1] to actual bounds
            # x_scaled = lower + (x + 1) / 2 * (upper - lower)
            scaled_design = lower_bounds + (design + 1) / 2 * (upper_bounds - lower_bounds)

            # Convert to list of dicts
            variations = []
            for row in scaled_design:
                variation = dict(zip(spec.names, row.tolist()))
                variations.append(variation)

            logger.info(f'Box-Behnken design generated {len(variations)} points for {num_dims} factors')
            return variations

        except Exception as e:
            logger.error(f'Box-Behnken design failed: {e}')
            raise

    elif mode == 'sobol':
        # Sobol quasi-random sequence - good space-filling properties
        # Note: Sobol rounds n up to the next power of 2 by default
        try:
            from pyDOE3 import sobol_sequence
        except ImportError as e:
            logger.error(
                f'pyDOE3 not installed. Install with: pip install pyDOE3. Error: {e}',
            )
            raise ImportError(
                "pyDOE3 is required for 'sobol' mode. Install with: pip install pyDOE3",
            ) from e

        try:
            # Generate Sobol sequence with bounds scaling built-in
            # sobol_sequence(n, d, scramble, seed, bounds, skip, use_pow_of_2)
            # bounds format: array of (min, max) pairs per dimension
            bounds_array = np.array(spec.bounds)  # Shape: (num_dims, 2)

            design = sobol_sequence(
                n=n,
                d=num_dims,
                scramble=True,  # Scrambling improves statistical properties
                seed=seed,
                bounds=bounds_array,  # pyDOE3 handles scaling for us
            )

            # Convert to list of dicts
            variations = []
            for row in design:
                variation = dict(zip(spec.names, row.tolist()))
                variations.append(variation)

            logger.info(f'Sobol sequence generated {len(variations)} points for {num_dims} dimensions')
            return variations

        except Exception as e:
            logger.error(f'Sobol sequence generation failed: {e}')
            raise

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: 'full_combinatoric', 'behnken', 'sobol'",
        )


# =============================================================================
# VALIDATION-AWARE SAMPLING
# =============================================================================

def generate_valid_samples(
    pylink_data: dict,
    dimension_spec: DimensionSpec,
    n_requested: int = 64,
    max_attempts: int | None = None,
    mode: str = 'sobol',
    validation: str = 'viability',
    selection: str = 'first',
    target: TargetTrajectory | None = None,
    target_joint: str | None = None,
    metric: str = 'mse',
    seed: int | None = None,
    phase_invariant: bool = True,
    min_viable_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """
    Generate valid mechanism samples using various strategies.

    This unified function combines viability checking and fitness-based presampling:
    - Generate samples using DOE methods (Sobol, Box-Behnken, full grid)
    - Validate samples (either mechanism viability or fitness scoring)
    - Select samples (either first N found or best N by score)

    Args:
        pylink_data: Base mechanism configuration
        dimension_spec: Specification of dimensions to sample
        n_requested: Number of valid samples to return
        max_attempts: Maximum samples to generate before stopping.
                     If None, uses n_requested * 10 for 'viability' mode,
                     or n_requested for 'fitness' mode (all samples evaluated).
        mode: Sampling strategy:
            - 'sobol': Sobol quasi-random sequence (good space coverage)
            - 'behnken': Box-Behnken design (good for response surfaces, 3+ dims)
            - 'full_combinatoric': Full grid (warning: grows as N^d)
        validation: Validation method:
            - 'viability': Check if mechanism can complete full rotation
            - 'fitness': Evaluate trajectory fit to target (requires target parameter)
        selection: Selection strategy:
            - 'first': Return first n_requested valid samples found
            - 'best': Return n_requested samples with best scores (requires validation='fitness')
        target: Target trajectory (required if validation='fitness')
        target_joint: Optional joint name to verify (used with validation='viability')
        metric: Error metric if validation='fitness' ('mse', 'rmse', 'total', 'max')
        seed: Random seed for reproducibility
        phase_invariant: Use phase-aligned scoring (used with validation='fitness')
        min_viable_ratio: Minimum acceptable ratio of valid/total samples.
                         Warning issued if ratio drops below this threshold.

    Returns:
        (samples, scores, n_generated): Tuple of:
            - samples: ndarray of shape (n_valid, n_dims) where n_valid <= n_requested
            - scores: ndarray of shape (n_valid,) with fitness scores if validation='fitness',
                     None if validation='viability'
            - n_generated: Total number of samples generated (including invalid)

    Raises:
        ValueError: If no valid samples found, or if invalid parameter combinations

    Example - Viable samples for MLSL initialization:
        >>> samples, _, n_total = generate_valid_samples(
        ...     pylink_data, dim_spec,
        ...     n_requested=64,
        ...     mode='sobol',
        ...     validation='viability',
        ...     selection='first',
        ...     target_joint='final_joint'
        ... )
        >>> print(f"Found {len(samples)} viable samples from {n_total} attempts")

    Example - Best samples for PSO initialization:
        >>> samples, scores, _ = generate_valid_samples(
        ...     pylink_data, dim_spec,
        ...     n_requested=32,
        ...     mode='sobol',
        ...     validation='fitness',
        ...     selection='best',
        ...     target=target_trajectory
        ... )
        >>> print(f"Best score: {scores[0]:.4f}")

    Example - Quick viability check with behnken:
        >>> samples, _, _ = generate_valid_samples(
        ...     pylink_data, dim_spec,
        ...     n_requested=50,
        ...     mode='behnken',
        ...     validation='viability'
        ... )
    """
    # Validate parameters
    if validation == 'fitness' and target is None:
        raise ValueError("validation='fitness' requires target parameter")
    
    if selection == 'best' and validation != 'fitness':
        raise ValueError("selection='best' requires validation='fitness'")
    
    num_dims = len(dimension_spec)
    if num_dims == 0:
        raise ValueError('DimensionSpec has no dimensions')
    
    # Set default max_attempts based on mode
    if max_attempts is None:
        if validation == 'viability':
            # May need multiple attempts to find viable samples
            max_attempts = n_requested * 10
        else:
            # Fitness mode: evaluate all requested samples
            max_attempts = n_requested
    
    logger.info(
        f'Generating valid samples: mode={mode}, validation={validation}, '
        f'selection={selection}, requesting {n_requested}, max_attempts={max_attempts}'
    )
    
    # Import validation functions as needed
    if validation == 'viability':
        from target_gen.achievable_target import verify_mechanism_viable
        from pylink_tools.optimization_helpers import apply_dimensions_from_array
    else:  # fitness
        from pylink_tools.optimize import create_fitness_function
        fitness_func = create_fitness_function(
            pylink_data, target, dimension_spec,
            metric=metric, phase_invariant=phase_invariant,
        )
    
    valid_samples: list[np.ndarray] = []
    valid_scores: list[float] = []
    n_generated = 0
    n_invalid = 0
    
    # Generate and validate samples in batches
    batch_size = min(256, max_attempts)
    
    while n_generated < max_attempts and len(valid_samples) < n_requested:
        # Determine batch size
        remaining_attempts = max_attempts - n_generated
        if selection == 'first':
            # For 'first' selection, stop when we have enough
            remaining_needed = n_requested - len(valid_samples)
            current_batch_size = min(batch_size, remaining_attempts, remaining_needed * 3)
        else:
            # For 'best' selection, generate all max_attempts samples
            current_batch_size = min(batch_size, remaining_attempts)
        
        if current_batch_size <= 0:
            break
        
        # Generate sample batch
        try:
            variations = get_mech_variations_from_spec(
                spec=dimension_spec,
                n=current_batch_size,
                mode=mode,
                seed=seed + n_generated if seed is not None else None,
            )
        except Exception as e:
            logger.warning(f"Sampling with mode='{mode}' failed: {e}. Using random sampling.")
            # Fallback to random
            lower = np.array([b[0] for b in dimension_spec.bounds])
            upper = np.array([b[1] for b in dimension_spec.bounds])
            rng = np.random.default_rng(seed + n_generated if seed is not None else None)
            random_samples = rng.uniform(lower, upper, (current_batch_size, num_dims))
            variations = [
                dict(zip(dimension_spec.names, row))
                for row in random_samples
            ]
        
        # Validate each sample
        for var in variations:
            n_generated += 1
            
            # Convert dict to array
            sample_array = np.array([var[name] for name in dimension_spec.names])
            
            try:
                if validation == 'viability':
                    # Just check if mechanism is viable
                    test_config = apply_dimensions_from_array(
                        pylink_data, tuple(sample_array), dimension_spec, inplace=False
                    )
                    is_valid = verify_mechanism_viable(test_config, target_joint)
                    
                    if is_valid:
                        valid_samples.append(sample_array)
                        if selection == 'first' and len(valid_samples) >= n_requested:
                            break
                    else:
                        n_invalid += 1
                
                else:  # fitness validation
                    # Evaluate fitness score
                    dims_tuple = tuple(sample_array)
                    score = fitness_func(dims_tuple)
                    
                    if np.isfinite(score):
                        valid_samples.append(sample_array)
                        valid_scores.append(score)
                    else:
                        n_invalid += 1
            
            except Exception:
                n_invalid += 1
                continue
        
        # Check viable ratio after reasonable sample size
        if n_generated >= 100 and validation == 'viability':
            viable_ratio = len(valid_samples) / n_generated
            if viable_ratio < min_viable_ratio:
                logger.warning(
                    f'Valid ratio ({viable_ratio:.1%}) below threshold '
                    f'({min_viable_ratio:.1%}) after {n_generated} samples. '
                    f'Consider tightening bounds or adjusting constraints.'
                )
                if len(valid_samples) < n_requested * 0.25:
                    logger.error(
                        f'Only found {len(valid_samples)}/{n_requested} valid samples '
                        f'with very low valid ratio. May indicate highly constrained space.'
                    )
                    break
    
    # Compute final statistics
    valid_ratio = len(valid_samples) / n_generated if n_generated > 0 else 0
    logger.info(
        f'Sampling complete: {len(valid_samples)}/{n_requested} requested, '
        f'{n_generated} total attempts, {valid_ratio:.1%} valid ratio'
    )
    
    if len(valid_samples) == 0:
        raise ValueError(
            f'No valid samples found after {n_generated} attempts. '
            'Mechanism may be over-constrained or bounds may be too wide.'
        )
    
    # Apply selection strategy
    if selection == 'best' and validation == 'fitness':
        # Sort by score and take best n_requested
        sorted_indices = np.argsort(valid_scores)[:n_requested]
        selected_samples = [valid_samples[i] for i in sorted_indices]
        selected_scores = [valid_scores[i] for i in sorted_indices]
        
        samples_array = np.array(selected_samples)
        scores_array = np.array(selected_scores)
        
        logger.info(f'Returning {len(samples_array)} best samples (best score: {scores_array[0]:.4f})')
        return samples_array, scores_array, n_generated
    
    elif selection == 'first':
        # Return first n_requested found
        selected_samples = valid_samples[:n_requested]
        samples_array = np.array(selected_samples)
        
        if validation == 'fitness':
            selected_scores = valid_scores[:n_requested]
            scores_array = np.array(selected_scores)
            return samples_array, scores_array, n_generated
        else:
            return samples_array, None, n_generated
    
    else:
        # Default: return all found (up to n_requested)
        samples_array = np.array(valid_samples[:n_requested])
        
        if validation == 'fitness':
            scores_array = np.array(valid_scores[:n_requested])
            return samples_array, scores_array, n_generated
        else:
            return samples_array, None, n_generated


def presample_valid_positions(
    pylink_data: dict,
    target: TargetTrajectory,
    dimension_spec: DimensionSpec,
    n_samples: int = 128,
    n_best: int = 32,
    mode: str = 'sobol',
    metric: str = 'mse',
    seed: int | None = None,
    phase_invariant: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-sample the design space and return the best valid configurations.

    DEPRECATED: Use generate_valid_samples() with validation='fitness' and selection='best' instead.
    This function is kept for backward compatibility.

    This function generates sample points using DOE methods (Sobol, Box-Behnken,
    or full combinatoric), evaluates each with the fitness function, filters
    to keep only valid configurations, and returns the best ones sorted by score.

    Args:
        pylink_data: Base linkage configuration
        target: Target trajectory to match
        dimension_spec: Dimensions to sample
        n_samples: How many samples to generate (for Sobol/combinatoric)
        n_best: Maximum number of best positions to return
        mode: Sampling strategy ('sobol', 'behnken', 'full_combinatoric')
        metric: Error metric for scoring ('mse', 'rmse', 'total', 'max')
        seed: Random seed for reproducibility
        phase_invariant: Use phase-aligned scoring

    Returns:
        (positions, scores): Tuple of:
            - positions: ndarray of shape (n_valid, n_dims) with best valid positions
            - scores: ndarray of shape (n_valid,) with corresponding fitness scores

    Example:
        >>> positions, scores = presample_valid_positions(
        ...     pylink_data, target, dim_spec,
        ...     n_samples=256, n_best=64, mode='sobol'
        ... )
        >>> print(f"Found {len(positions)} valid configurations")
        >>> print(f"Best score: {scores[0]:.4f}")
    """
    logger.warning(
        'presample_valid_positions is deprecated. '
        'Use generate_valid_samples(validation="fitness", selection="best") instead.'
    )
    
    samples, scores, _ = generate_valid_samples(
        pylink_data=pylink_data,
        dimension_spec=dimension_spec,
        n_requested=n_best,
        max_attempts=n_samples,
        mode=mode,
        validation='fitness',
        selection='best',
        target=target,
        metric=metric,
        seed=seed,
        phase_invariant=phase_invariant,
    )
    
    return samples, scores


def generate_viable_sobol_samples(
    pylink_data: dict,
    dimension_spec: DimensionSpec,
    n_requested: int = 32,
    max_attempts: int = 1000,
    target_joint: str | None = None,
    seed: int | None = None,
    min_viable_ratio: float = 0.1,
) -> tuple[np.ndarray, int]:
    """
    Generate Sobol samples that pass mechanism viability checks.

    DEPRECATED: Use generate_valid_samples() with validation='viability' instead.
    This function is kept for backward compatibility.

    This function generates Sobol quasi-random samples within the specified bounds,
    validates each configuration using verify_mechanism_viable(), and returns only
    the viable samples.

    Args:
        pylink_data: Base mechanism configuration (will not be modified)
        dimension_spec: Specification of dimensions to sample
        n_requested: Number of viable samples to return (if possible)
        max_attempts: Maximum total samples to generate before giving up
        target_joint: Optional joint name to verify exists in trajectories
        seed: Random seed for reproducibility
        min_viable_ratio: Minimum acceptable ratio of viable/total samples

    Returns:
        (viable_samples, n_generated): Tuple of:
            - viable_samples: ndarray of shape (n_viable, n_dims)
            - n_generated: Total number of samples generated

    Example:
        >>> samples, n_total = generate_viable_sobol_samples(
        ...     pylink_data, dim_spec,
        ...     n_requested=64, target_joint='final_joint'
        ... )
        >>> print(f"Found {len(samples)} viable samples from {n_total} attempts")
    """
    logger.warning(
        'generate_viable_sobol_samples is deprecated. '
        'Use generate_valid_samples(validation="viability", mode="sobol") instead.'
    )
    
    samples, _, n_generated = generate_valid_samples(
        pylink_data=pylink_data,
        dimension_spec=dimension_spec,
        n_requested=n_requested,
        max_attempts=max_attempts,
        mode='sobol',
        validation='viability',
        selection='first',
        target_joint=target_joint,
        seed=seed,
        min_viable_ratio=min_viable_ratio,
    )
    
    return samples, n_generated
