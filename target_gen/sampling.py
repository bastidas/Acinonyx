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
    - get_mech_variations_from_spec(): DOE sampling from DimensionBoundsSpec
    - presample_valid_positions(): Pre-sample and score configurations
    - generate_viable_sobol_samples(): Generate valid Sobol samples for MLSL
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from pylink_tools.mechanism import Mechanism
from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import TargetTrajectory

if TYPE_CHECKING:
    from target_gen.variation_config import DimensionVariationConfig

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """
    Result of mechanism sampling operation.

    Designed for both backend processing and frontend visualization.
    When return_mechanisms=True, includes Mechanism objects for each valid sample.
    When return_trajectories=True, includes trajectory data for frontend rendering.

    Frontend Visualization:
    - For each valid sample, frontend can display:
      1. Mechanism structure (from mechanisms[i].to_dict())
      2. Trajectory paths (from trajectories[i][joint_name])
    - Invalid samples are marked but mechanisms/trajectories are None for them.

    Memory Considerations:
    - mechanisms: Can be large (full Mechanism objects) - only include if needed
    - trajectories: Moderate size (dict per sample) - include for visualization
    - Both are optional to allow lightweight queries when only samples/scores needed

    Attributes:
        samples: Array of shape (n_samples, n_dims) containing all samples.
                 If return_all=True, includes all generated samples up to max_attempts.
                 If return_all=False, includes only selected samples up to n_requested.
        is_valid: Boolean array of shape (n_samples,) marking which samples are valid.
                  True means the mechanism is viable (can complete full rotation).
        scores: Optional array of shape (n_samples,) with fitness scores.
                Only present if return_scores=True or validation_mode='fitness'.
                None for invalid samples or when scores not calculated.
        n_generated: Total number of samples generated (including invalid).
        n_valid: Number of valid samples found.
        n_invalid: Number of invalid samples found.
        mechanisms: Optional list of Mechanism objects, one per sample.
                    Only present if return_mechanisms=True.
                    mechanisms[i] is None if is_valid[i] == False.
                    WARNING: Can be large in memory - only include if needed.
        trajectories: Optional list of trajectory dictionaries, one per sample.
                      Only present if return_trajectories=True.
                      Format: dict[str, list[list[float]]] mapping joint_name -> positions.
                      Each position is [x, y] as list[float].
                      trajectories[i] is None if is_valid[i] == False.
                      This matches the format expected by frontend trajectory renderers.

    Example:
        >>> result = generate_valid_samples(
        ...     mechanism, dimension_bounds_spec,
        ...     n_requested=32,
        ...     return_mechanisms=True,
        ...     return_trajectories=True,
        ... )
        >>> # Get only valid samples
        >>> valid_samples = result.valid_samples
        >>> # Get mechanisms for valid samples
        >>> valid_mechanisms = result.valid_mechanisms
        >>> # Get trajectories for valid samples
        >>> valid_trajectories = result.valid_trajectories
        >>> # Serialize for API/JSON
        >>> api_data = result.to_dict()
    """
    samples: np.ndarray  # Shape: (n_samples, n_dims)
    is_valid: np.ndarray  # Shape: (n_samples,) - boolean array
    scores: np.ndarray | None = None  # Shape: (n_samples,) - optional
    n_generated: int = 0
    n_valid: int = 0
    n_invalid: int = 0

    # Optional: Mechanism objects (only for valid samples, None for invalid)
    # WARNING: Can be large in memory - only include if return_mechanisms=True
    mechanisms: list[Mechanism | None] | None = None
    # Shape: (n_samples,) - mechanisms[i] is None if is_valid[i] == False

    # Optional: Trajectory data for frontend visualization (only for valid samples)
    # Format: list[dict[str, list[list[float]]] | None]
    # Each dict maps joint_name -> [[x0, y0], [x1, y1], ...]
    # trajectories[i] is None if is_valid[i] == False or return_trajectories=False
    trajectories: list[dict[str, list[list[float]]] | None] | None = None
    # Shape: (n_samples,) - trajectories[i] is None for invalid samples

    @property
    def valid_samples(self) -> np.ndarray:
        """
        Get only valid samples.

        Returns:
            Array of shape (n_valid, n_dims) containing only samples where is_valid==True.
        """
        return self.samples[self.is_valid]

    @property
    def valid_scores(self) -> np.ndarray | None:
        """
        Get scores for valid samples only.

        Returns:
            Array of shape (n_valid,) with scores for valid samples, or None if scores not available.
        """
        if self.scores is None:
            return None
        return self.scores[self.is_valid]

    @property
    def valid_mechanisms(self) -> list[Mechanism] | None:
        """
        Get only valid mechanisms (filters out None values).

        Returns:
            List of Mechanism objects for valid samples only, or None if mechanisms not available.
        """
        if self.mechanisms is None:
            return None
        return [m for m, valid in zip(self.mechanisms, self.is_valid) if valid and m is not None]

    @property
    def valid_trajectories(self) -> list[dict[str, list[list[float]]]] | None:
        """
        Get only valid trajectories (filters out None values).

        Returns:
            List of trajectory dictionaries for valid samples only, or None if trajectories not available.
            Each dictionary maps joint_name -> list of [x, y] positions.
        """
        if self.trajectories is None:
            return None
        return [t for t, valid in zip(self.trajectories, self.is_valid) if valid and t is not None]

    def to_dict(self) -> dict:
        """
        Serialize to dictionary for API/JSON responses.

        Converts Mechanism objects to dict format for frontend consumption.
        Frontend can use this directly for visualization.

        Returns:
            Dictionary with all data serialized for JSON/API:
            - samples: list of lists (n_samples, n_dims)
            - is_valid: list of booleans (n_samples,)
            - scores: list of floats (n_samples,) - only if available
            - mechanisms: list of dicts (n_samples,) - only if available, None for invalid samples
            - trajectories: list of dicts (n_samples,) - only if available, None for invalid samples
            - n_generated: int
            - n_valid: int
            - n_invalid: int

        Example:
            >>> result = generate_valid_samples(...)
            >>> api_data = result.to_dict()
            >>> # Send to frontend via JSON
            >>> json_response = json.dumps(api_data)
        """
        result: dict = {
            'samples': self.samples.tolist(),
            'is_valid': self.is_valid.tolist(),
            'n_generated': self.n_generated,
            'n_valid': self.n_valid,
            'n_invalid': self.n_invalid,
        }

        if self.scores is not None:
            result['scores'] = self.scores.tolist()

        if self.mechanisms is not None:
            result['mechanisms'] = [
                m.to_dict() if m is not None else None
                for m in self.mechanisms
            ]

        if self.trajectories is not None:
            result['trajectories'] = self.trajectories

        return result


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
    logger.info(f'\nget_combinatoric_gradations called: n={n}, names={len(names)}, bounds={len(bounds)}')
    logger.info(f'Generating {n} gradations for {len(names)} names and {len(bounds)} bounds')
    logger.info(f'Names: {names}')
    logger.info(f'Bounds: {bounds}')
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
    spec: DimensionBoundsSpec,
    n: int,
    mode: str = 'full_combinatoric',
    center: int | None = None,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """
    Generate mechanism variations from a DimensionBoundsSpec using various sampling strategies.

    Args:
        spec: DimensionBoundsSpec containing names and bounds
        n: Meaning depends on mode:
           - 'meshgrid' or 'full_combinatoric': Number of gradations per dimension (N^d total points)
           - 'behnken': Number of center points (ignored if center is specified)
           - 'sobol': Total number of sample points to generate
        mode: Sampling strategy, one of:
           - 'meshgrid': Regular grid (Cartesian product of evenly-spaced values).
                 WARNING: Grows as N^d (e.g., 5 dims × 10 gradations = 100,000 points)
           - 'full_combinatoric': Alias for 'meshgrid' (deprecated, use 'meshgrid')
           - 'behnken': Box-Behnken design. Efficient for 3+ factors,
                 generates ~2*d*(d-1) + center points. Good for response surface modeling.
           - 'sobol': Sobol quasi-random sequence. Low-discrepancy
                 sampling that fills space uniformly. Good for global exploration.
        center: For 'behnken' mode only - number of center points (default: auto)
        seed: For 'sobol' mode only - random seed for reproducibility

    Returns:
        List of mechanism configuration dicts

    Example:
        >>> spec = mechanism.get_dimension_bounds_spec()
        >>> # Meshgrid (careful - grows fast! n=5 with 3 dims = 125 samples)
        >>> variations = get_mech_variations_from_spec(spec, n=5, mode='meshgrid')
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

    if mode == 'full_combinatoric' or mode == 'meshgrid':
        # Regular grid: Cartesian product of gradations
        # n = number of gradations per dimension
        # Total samples = n^num_dims
        gradations = get_combinatoric_gradations(spec.names, spec.bounds, n)
        variations = get_mech_variations(gradations)
        logger.debug(f'Meshgrid generated {len(variations)} samples from {n} gradations per dimension')
        return variations

    elif mode == 'behnken':
        # Box-Behnken design - efficient for response surface modeling
        if num_dims < 3:
            logger.warning(
                f'Box-Behnken requires at least 3 factors, got {num_dims}. '
                f'Varying dimensions: {spec.names}. '
                'Falling back to full_combinatoric.',
            )
            gradations = get_combinatoric_gradations(spec.names, spec.bounds, n)
            return get_mech_variations(gradations)

        # Debug: Log dimension bounds to check if any are constant
        logger.debug(f'Box-Behnken: {num_dims} varying dimensions: {spec.names}')
        for i, name in enumerate(spec.names):
            min_bound, max_bound = spec.bounds[i]
            if min_bound == max_bound:
                logger.warning(
                    f'Box-Behnken: Dimension "{name}" has identical bounds [{min_bound}, {max_bound}]. '
                    f'This dimension will be constant (not vary).',
                )
            else:
                logger.debug(f'Box-Behnken: {name} bounds=[{min_bound:.4f}, {max_bound:.4f}], range={max_bound - min_bound:.4f}')

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

            # Debug: Check design structure
            logger.debug(f'Box-Behnken design shape: {design.shape}')
            logger.debug(f'Box-Behnken design unique values per column:')
            for i in range(design.shape[1]):
                unique_vals = np.unique(design[:, i])
                logger.debug(
                    f'  Column {i} ({spec.names[i] if i < len(spec.names) else "unknown"}): {unique_vals} (range: [{np.min(design[:, i]):.3f}, {np.max(design[:, i]):.3f}])',
                )

            # Check if any columns are constant (all same value)
            for i in range(design.shape[1]):
                if len(np.unique(design[:, i])) == 1:
                    logger.warning(
                        f'Box-Behnken design column {i} ({spec.names[i] if i < len(spec.names) else "unknown"}) is constant! '
                        f'All values = {design[0, i]:.3f}. This dimension will not vary.',
                    )

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
            f"Unknown mode '{mode}'. Choose from: 'meshgrid', 'full_combinatoric' (deprecated), 'behnken', 'sobol'",
        )


# =============================================================================
# DIMENSION VARIATION HELPERS
# =============================================================================


def _identify_constant_dimensions(
    dimension_bounds_spec: DimensionBoundsSpec,
    dimension_variation_config: DimensionVariationConfig | None,
) -> dict[str, float]:
    """
    Identify dimensions that are held constant (excluded from variation).

    Returns a dictionary mapping constant dimension names to their fixed values.

    Args:
        dimension_bounds_spec: Dimension specification
        dimension_variation_config: Variation configuration (None means all vary)

    Returns:
        Dictionary mapping constant dimension names to their fixed values
    """
    if dimension_variation_config is None:
        return {}

    constant_dims = {}
    for i, name in enumerate(dimension_bounds_spec.names):
        enabled, _, _ = dimension_variation_config.get_variation_for_dimension(name)
        if not enabled:
            # This dimension is constant - use its initial value
            constant_dims[name] = dimension_bounds_spec.initial_values[i]

    return constant_dims


def _create_varying_dimension_spec(
    dimension_bounds_spec: DimensionBoundsSpec,
    constant_dims: dict[str, float],
) -> DimensionBoundsSpec:
    """
    Create a filtered DimensionBoundsSpec containing only varying dimensions.

    Args:
        dimension_bounds_spec: Original dimension specification
        constant_dims: Dictionary of constant dimension names to exclude

    Returns:
        New DimensionBoundsSpec with only varying dimensions
    """
    if not constant_dims:
        # No constant dimensions, return original
        return dimension_bounds_spec

    # Filter out constant dimensions
    varying_names = []
    varying_initial_values = []
    varying_bounds = []
    varying_edge_mapping = {}
    varying_weights = []

    for i, name in enumerate(dimension_bounds_spec.names):
        if name not in constant_dims:
            varying_names.append(name)
            varying_initial_values.append(dimension_bounds_spec.initial_values[i])
            varying_bounds.append(dimension_bounds_spec.bounds[i])
            if name in dimension_bounds_spec.edge_mapping:
                varying_edge_mapping[name] = dimension_bounds_spec.edge_mapping[name]
            if dimension_bounds_spec.weights is not None:
                varying_weights.append(dimension_bounds_spec.weights[i])

    return DimensionBoundsSpec(
        names=varying_names,
        initial_values=varying_initial_values,
        bounds=varying_bounds,
        edge_mapping=varying_edge_mapping,
        weights=varying_weights if dimension_bounds_spec.weights is not None else None,
    )


def _inject_constant_dimensions(
    sample_variations: list[dict[str, float]],
    constant_dims: dict[str, float],
) -> list[dict[str, float]]:
    """
    Inject constant dimension values back into sample variations.

    Args:
        sample_variations: List of variation dicts (only varying dimensions)
        constant_dims: Dictionary mapping constant dimension names to their fixed values

    Returns:
        List of variation dicts with constant dimensions added
    """
    if not constant_dims:
        return sample_variations

    # Add constant dimensions to each variation
    return [
        {**var_dict, **constant_dims}
        for var_dict in sample_variations
    ]


def apply_dimension_variation_config(
    dimension_bounds_spec: DimensionBoundsSpec,
    config: DimensionVariationConfig | None,
) -> DimensionBoundsSpec:
    """
    Apply DimensionVariationConfig to modify DimensionBoundsSpec bounds.

    This function takes a DimensionBoundsSpec and a DimensionVariationConfig,
    and creates a new DimensionBoundsSpec with bounds adjusted according to
    the variation settings. Excluded dimensions have bounds set to
    their initial value (no variation).

    Args:
        dimension_bounds_spec: Original dimension specification
        config: Variation configuration (None means no modification)

    Returns:
        New DimensionBoundsSpec with modified bounds

    Example:
        >>> from target_gen.variation_config import DimensionVariationConfig
        >>> config = DimensionVariationConfig(
        ...     default_variation_range=0.5,  # ±50%
        ...     exclude_dimensions=['rocker_distance']
        ... )
        >>> modified_spec = apply_dimension_variation_config(dimension_bounds_spec, config)
    """
    if config is None:
        return dimension_bounds_spec

    # Import here to avoid circular dependencies
    from target_gen.variation_config import DimensionVariationConfig

    modified_bounds = []
    for i, (name, initial, (min_bound, max_bound)) in enumerate(
        zip(dimension_bounds_spec.names, dimension_bounds_spec.initial_values, dimension_bounds_spec.bounds),
    ):
        enabled, min_pct, max_pct = config.get_variation_for_dimension(name)

        if not enabled:
            # Excluded dimension: set bounds to initial value (no variation)
            modified_bounds.append((initial, initial))
        else:
            # Apply percentage variation relative to initial value
            new_min = initial * (1.0 + min_pct)
            new_max = initial * (1.0 + max_pct)

            # Clamp to original bounds to ensure we don't exceed them
            new_min = max(min_bound, new_min)
            new_max = min(max_bound, new_max)

            modified_bounds.append((new_min, new_max))

    # Create new DimensionBoundsSpec with modified bounds
    return DimensionBoundsSpec(
        names=dimension_bounds_spec.names,
        initial_values=dimension_bounds_spec.initial_values,
        bounds=modified_bounds,
        edge_mapping=dimension_bounds_spec.edge_mapping,
        weights=dimension_bounds_spec.weights,
    )


# =============================================================================
# VALIDATION-AWARE SAMPLING
# =============================================================================


def _validate_sample_viability(
    base_mechanism: Mechanism,
    sample_dimensions: dict[str, float],
) -> bool:
    """
    Check if mechanism with given dimensions is viable.

    A mechanism is viable if it can complete a full rotation without
    encountering singularities or constraint violations.

    Args:
        base_mechanism: Base mechanism to modify
        sample_dimensions: Dictionary mapping dimension names to values

    Returns:
        True if mechanism is viable, False otherwise
    """
    test_mechanism = base_mechanism.with_dimensions(sample_dimensions)
    from target_gen.achievable_target import verify_mechanism_viable
    return verify_mechanism_viable(test_mechanism)


def _evaluate_sample_fitness(
    fitness_function: Callable[[tuple[float, ...] | np.ndarray], float],
    sample_array: np.ndarray,
) -> float | None:
    """
    Evaluate fitness score for a sample.

    Args:
        fitness_function: Function that takes dimensions tuple and returns fitness score
        sample_array: Array of dimension values for the sample

    Returns:
        Fitness score if valid and finite, None otherwise
    """
    try:
        dims_tuple = tuple(sample_array)
        score = fitness_function(dims_tuple)
        return score if np.isfinite(score) else None
    except Exception:
        return None


def _generate_sample_batch(
    dimension_bounds_spec: DimensionBoundsSpec,
    batch_size: int,
    sampling_mode: str,
    seed: int | None,
    fallback_rng: np.random.Generator | None = None,
) -> list[dict[str, float]]:
    """
    Generate a batch of sample variations.

    Args:
        dimension_bounds_spec: Specification of dimensions to sample
        batch_size: Desired number of samples to generate (approximate for meshgrid)
        sampling_mode: Sampling strategy ('sobol', 'behnken', 'meshgrid', 'full_combinatoric')
        seed: Random seed for reproducibility
        fallback_rng: Optional RNG for fallback random sampling

    Returns:
        List of dictionaries, each mapping dimension names to values

    Note:
        For 'meshgrid' mode, batch_size is approximate. The actual number of samples
        will be n^d where n is calculated to approximate batch_size.
    """
    try:
        # For meshgrid (full_combinatoric), calculate appropriate n (gradations per dimension)
        # to approximate the desired batch_size
        if sampling_mode == 'meshgrid' or sampling_mode == 'full_combinatoric':
            num_dims = len(dimension_bounds_spec.names)
            if num_dims == 0:
                return []

            # Calculate n such that n^num_dims ≈ batch_size
            # n = batch_size^(1/num_dims)
            n_gradations = max(2, int(round(batch_size ** (1.0 / num_dims))))

            # Ensure we don't generate too many samples
            total_samples = n_gradations ** num_dims
            if total_samples > batch_size * 2:
                # If we'd generate way too many, reduce n
                n_gradations = max(2, int(round((batch_size * 1.5) ** (1.0 / num_dims))))
                total_samples = n_gradations ** num_dims

            logger.debug(
                f'Meshgrid mode: batch_size={batch_size}, num_dims={num_dims}, '
                f'n_gradations={n_gradations}, total_samples={total_samples}',
            )

            # Use 'full_combinatoric' mode internally (for backward compatibility)
            return get_mech_variations_from_spec(
                spec=dimension_bounds_spec,  # Fixed: parameter name is 'spec', not 'dimension_bounds_spec'
                n=n_gradations,
                mode='full_combinatoric',
                seed=seed,
            )
        else:
            # For other modes, batch_size is the actual number of samples
            return get_mech_variations_from_spec(
                spec=dimension_bounds_spec,  # Fixed: parameter name is 'spec', not 'dimension_bounds_spec'
                n=batch_size,
                mode=sampling_mode,
                seed=seed,
            )
    except Exception as e:
        logger.warning(f"Sampling with mode='{sampling_mode}' failed: {e}. Using random sampling.")
        # Fallback to random sampling
        if fallback_rng is None:
            fallback_rng = np.random.default_rng(seed)

        lower_bounds = np.array([b[0] for b in dimension_bounds_spec.bounds])
        upper_bounds = np.array([b[1] for b in dimension_bounds_spec.bounds])
        random_samples = fallback_rng.uniform(lower_bounds, upper_bounds, (batch_size, len(dimension_bounds_spec)))

        return [
            dict(zip(dimension_bounds_spec.names, row))
            for row in random_samples
        ]


def _process_sample_batch(
    sample_variations: list[dict[str, float]],
    base_mechanism: Mechanism,
    dimension_bounds_spec: DimensionBoundsSpec,
    validation_mode: str,
    fitness_function: Callable[[tuple[float, ...] | np.ndarray], float] | None,
    return_scores: bool,
    return_mechanisms: bool,
    return_trajectories: bool,
) -> tuple[
    list[np.ndarray],  # samples
    list[bool],  # is_valid
    list[float | None],  # scores
    list[Mechanism | None] | None,  # mechanisms
    list[dict[str, list[list[float]]] | None] | None,  # trajectories
]:
    """
    Process a batch of sample variations.

    Validates each sample and optionally creates mechanisms and trajectories
    for valid samples. This function handles both viability and fitness validation modes.

    Args:
        sample_variations: List of dictionaries, each mapping dimension names to values
        base_mechanism: Base mechanism to modify for each sample
        dimension_bounds_spec: Specification of dimensions (for array conversion)
        validation_mode: 'viability' or 'fitness'
        fitness_function: Optional fitness function (required if validation_mode='fitness')
        return_scores: Whether to calculate and return fitness scores
        return_mechanisms: Whether to create and return Mechanism objects for valid samples
        return_trajectories: Whether to compute and return trajectories for valid samples

    Returns:
        Tuple of:
        - samples: List of sample arrays (all samples, including invalid)
        - is_valid: List of validity flags (True for viable/finite-fitness samples)
        - scores: List of scores (None if not calculated or invalid)
        - mechanisms: List of Mechanism objects (None if return_mechanisms=False or invalid)
        - trajectories: List of trajectory dicts (None if return_trajectories=False or invalid)

    Note:
        - Invalid samples still appear in the results with is_valid[i]=False
        - mechanisms[i] and trajectories[i] are None for invalid samples
        - Trajectories are in frontend format: dict[str, list[list[float]]]
    """
    samples: list[np.ndarray] = []
    is_valid: list[bool] = []
    scores: list[float | None] = []
    mechanisms: list[Mechanism | None] | None = [] if return_mechanisms else None
    trajectories: list[dict[str, list[list[float]]] | None] | None = [] if return_trajectories else None

    for var_dict in sample_variations:
        # Convert dict to array
        sample_array = np.array([var_dict[name] for name in dimension_bounds_spec.names])
        samples.append(sample_array)

        # Validate sample
        valid = False
        score = None

        try:
            if validation_mode == 'viability':
                # Check viability
                valid = _validate_sample_viability(base_mechanism, var_dict)

                # Calculate score if requested (even in viability mode)
                if valid and return_scores and fitness_function is not None:
                    score = _evaluate_sample_fitness(fitness_function, sample_array)
                    # Note: score can be None if fitness evaluation fails

            else:  # validation_mode == 'fitness'
                # Evaluate fitness score
                if fitness_function is None:
                    raise ValueError("fitness_function required for validation_mode='fitness'")

                score = _evaluate_sample_fitness(fitness_function, sample_array)
                valid = score is not None and np.isfinite(score)

        except Exception:
            # Any exception during validation means invalid
            valid = False
            score = None

        is_valid.append(valid)

        # Store score (None for invalid or if not requested)
        if return_scores:
            scores.append(score)
        else:
            scores.append(None)

        # Create mechanism and trajectory if needed and valid
        if valid:
            if return_mechanisms or return_trajectories:
                test_mechanism = base_mechanism.with_dimensions(var_dict)

                if return_mechanisms:
                    # mechanisms is guaranteed to be a list (not None) when return_mechanisms=True
                    assert mechanisms is not None, 'mechanisms should not be None when return_mechanisms=True'
                    mechanisms.append(test_mechanism)

                if return_trajectories:
                    # Get trajectories for all joints (frontend format)
                    traj_dict = test_mechanism.simulate_dict()
                    # Convert to frontend format: dict[str, list[list[float]]]
                    # Each position tuple (x, y) becomes [x, y] as list[float]
                    # trajectories is guaranteed to be a list (not None) when return_trajectories=True
                    assert trajectories is not None, 'trajectories should not be None when return_trajectories=True'
                    trajectories.append({
                        joint_name: [[float(x), float(y)] for x, y in positions]
                        for joint_name, positions in traj_dict.items()
                    })
        else:
            # Invalid sample: set mechanisms and trajectories to None
            if return_mechanisms:
                assert mechanisms is not None, 'mechanisms should not be None when return_mechanisms=True'
                mechanisms.append(None)
            if return_trajectories:
                assert trajectories is not None, 'trajectories should not be None when return_trajectories=True'
                trajectories.append(None)

    return samples, is_valid, scores, mechanisms, trajectories


def _select_samples(
    all_samples: np.ndarray,
    is_valid: np.ndarray,
    scores: np.ndarray | None,
    mechanisms: list[Mechanism | None] | None,
    trajectories: list[dict[str, list[list[float]]] | None] | None,
    n_requested: int,
    selection_strategy: str,
    return_all: bool,
) -> tuple[
    np.ndarray,  # selected_samples
    np.ndarray,  # selected_is_valid
    np.ndarray | None,  # selected_scores
    list[Mechanism | None] | None,  # selected_mechanisms
    list[dict[str, list[list[float]]] | None] | None,  # selected_trajectories
]:
    """
    Apply selection strategy to choose final samples.

    If return_all=True, returns all samples (up to max_attempts) with validity flags.
    If return_all=False, returns selected samples (up to n_requested) based on strategy.

    Args:
        all_samples: Array of all samples (n_total, n_dims)
        is_valid: Boolean array marking validity (n_total,)
        scores: Optional array of scores (n_total,)
        mechanisms: Optional list of Mechanism objects (n_total,)
        trajectories: Optional list of trajectory dicts (n_total,)
        n_requested: Number of samples to return (if return_all=False)
        selection_strategy: 'first' or 'best'
        return_all: If True, return all samples; if False, apply selection strategy

    Returns:
        Tuple of:
        - selected_samples: Array of selected samples
        - selected_is_valid: Boolean array for selected samples
        - selected_scores: Scores for selected samples (None if not available)
        - selected_mechanisms: Mechanisms for selected samples (None if not available)
        - selected_trajectories: Trajectories for selected samples (None if not available)
    """
    if return_all:
        # Return all samples with validity flags
        selected_samples = all_samples
        selected_is_valid = is_valid
        selected_scores = scores
        selected_mechanisms = mechanisms
        selected_trajectories = trajectories
    else:
        # Apply selection strategy
        if selection_strategy == 'best' and scores is not None:
            # Sort by score and take best n_requested valid samples
            valid_indices = np.where(is_valid)[0]
            if len(valid_indices) == 0:
                # No valid samples - return empty arrays
                n_dims = all_samples.shape[1] if len(all_samples) > 0 else 0
                return (
                    np.empty((0, n_dims), dtype=all_samples.dtype),
                    np.array([], dtype=bool),
                    None,
                    None,
                    None,
                )

            # Sort valid samples by score (ascending - lower is better)
            sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[:n_requested]]
            selected_indices = sorted_valid_indices
        else:
            # 'first' strategy: take first n_requested valid samples
            valid_indices = np.where(is_valid)[0][:n_requested]
            selected_indices = valid_indices

        # Select corresponding data
        selected_samples = all_samples[selected_indices]
        selected_is_valid = is_valid[selected_indices]
        selected_scores = scores[selected_indices] if scores is not None else None

        if mechanisms is not None:
            selected_mechanisms = [mechanisms[i] for i in selected_indices]
        else:
            selected_mechanisms = None

        if trajectories is not None:
            selected_trajectories = [trajectories[i] for i in selected_indices]
        else:
            selected_trajectories = None

    return selected_samples, selected_is_valid, selected_scores, selected_mechanisms, selected_trajectories


# =============================================================================
# NEW SAMPLING FUNCTIONS (Refactored)
# =============================================================================


def generate_samples(
    mechanism: Mechanism,
    dimension_bounds_spec: DimensionBoundsSpec,
    n_requested: int,
    sampling_mode: str = 'sobol',
    dimension_variation_config: DimensionVariationConfig | None = None,
    target_trajectory: TargetTrajectory | None = None,
    target_joint: str | None = None,
    metric: str = 'mse',
    phase_invariant: bool = True,
    seed: int | None = None,
    return_mechanisms: bool = False,
    return_trajectories: bool = False,
) -> SamplingResult:
    """
    Generate mechanism samples with viability validation.

    Generates `n_requested` mechanism samples within the bounds defined by
    `dimension_variation_config`. Validates viability (checks if mechanism can
    complete full rotation) but does not filter - returns all samples with
    validity flags.

    Args:
        mechanism: Mechanism to sample
        dimension_bounds_spec: Specification of dimensions to sample
        n_requested: Number of samples to generate
        sampling_mode: Sampling strategy ('sobol', 'behnken', 'meshgrid', 'full_combinatoric')
        dimension_variation_config: Optional configuration to control which dimensions vary
                                   and by how much. Excluded dimensions have bounds set to their
                                   initial value (no variation).
        target_trajectory: Optional target trajectory. If provided, fitness scores will be calculated.
        target_joint: Optional joint name for viability checking and fitness evaluation
        metric: Error metric for fitness scoring ('mse', 'rmse', 'total', 'max')
        phase_invariant: Use phase-aligned scoring for fitness evaluation
        seed: Random seed for reproducibility
        return_mechanisms: If True, include Mechanism objects for valid samples
        return_trajectories: If True, include trajectory data for valid samples (None for invalid)

    Returns:
        SamplingResult: Dataclass containing:
            - samples: Array of all generated samples (shape: (n_requested, n_varying_dims))
            - is_valid: Boolean array marking which samples are viable
            - scores: Array of fitness scores if target_trajectory provided, None otherwise
            - mechanisms: Optional list of Mechanism objects (only for valid samples)
            - trajectories: Optional list of trajectory dictionaries (only for valid samples, None for invalid)
            - n_generated: Always equals n_requested
            - n_valid: Number of viable samples found
            - n_invalid: Number of non-viable samples

    Example:
        >>> result = generate_samples(
        ...     mechanism=my_mechanism,
        ...     dimension_bounds_spec=dim_spec,
        ...     n_requested=100,
        ...     sampling_mode='sobol',
        ...     dimension_variation_config=config,  # Excludes 'rocker_distance'
        ...     target_trajectory=target,  # Optional: enables score calculation
        ...     return_trajectories=True,
        ... )
        >>> # result.n_generated == 100
        >>> # result.n_valid <= 100  # Some may be invalid
        >>> # result.scores is not None if target_trajectory provided
        >>> # result.trajectories[i] is None if is_valid[i] == False
    """
    # Apply dimension variation config if provided and identify constant dimensions
    constant_dims: dict[str, float] = {}
    if dimension_variation_config is not None:
        from target_gen.variation_config import DimensionVariationConfig
        dimension_bounds_spec = apply_dimension_variation_config(dimension_bounds_spec, dimension_variation_config)
        constant_dims = _identify_constant_dimensions(dimension_bounds_spec, dimension_variation_config)
        logger.debug(
            f'Applied dimension_variation_config: '
            f'excluded={dimension_variation_config.exclude_dimensions if dimension_variation_config else []}',
        )
        if constant_dims:
            logger.info(
                f'Constant dimensions (excluded from sampling): {list(constant_dims.keys())}',
            )

    # Create filtered dimension spec with only varying dimensions for sampling
    varying_dimension_spec = _create_varying_dimension_spec(dimension_bounds_spec, constant_dims)

    num_dims = len(varying_dimension_spec)
    if num_dims == 0:
        raise ValueError('DimensionBoundsSpec has no varying dimensions (all dimensions are constant)')

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Determine if deterministic mode
    deterministic_modes = ('meshgrid', 'full_combinatoric', 'behnken')
    is_deterministic = sampling_mode in deterministic_modes

    # Generate samples
    if is_deterministic:
        # For deterministic modes, generate all samples at once
        logger.debug(f'Deterministic mode {sampling_mode}: generating {n_requested} samples at once')

        sample_variations = _generate_sample_batch(
            dimension_bounds_spec=varying_dimension_spec,
            batch_size=n_requested,
            sampling_mode=sampling_mode,
            seed=seed,
            fallback_rng=rng,
        )

        # Limit to n_requested if we generated more (e.g., meshgrid generates n^d)
        if len(sample_variations) > n_requested:
            sample_variations = sample_variations[:n_requested]
    else:
        # For non-deterministic modes, generate exactly n_requested samples
        sample_variations = _generate_sample_batch(
            dimension_bounds_spec=varying_dimension_spec,
            batch_size=n_requested,
            sampling_mode=sampling_mode,
            seed=seed,
            fallback_rng=rng,
        )

    # Inject constant dimensions back into sample variations
    sample_variations = _inject_constant_dimensions(sample_variations, constant_dims)

    # Set up fitness function if target_trajectory is provided
    fitness_func = None
    if target_trajectory is not None:
        from pylink_tools.mechanism import create_mechanism_fitness
        score_target_joint = target_joint
        if score_target_joint is None:
            score_target_joint = target_trajectory.joint_name

        fitness_func = create_mechanism_fitness(
            mechanism,
            target_trajectory,
            target_joint=score_target_joint,
            metric=metric,
            phase_invariant=phase_invariant,
            phase_align_method='rotation',
            translation_invariant=False,
        )

    # Process samples: validate viability and optionally create mechanisms/trajectories
    # Calculate scores if target_trajectory is provided
    batch_samples, batch_is_valid, batch_scores, batch_mechanisms, batch_trajectories = _process_sample_batch(
        sample_variations=sample_variations,
        base_mechanism=mechanism,
        dimension_bounds_spec=dimension_bounds_spec,  # Use full spec for array conversion
        validation_mode='viability',
        fitness_function=fitness_func,
        return_scores=(target_trajectory is not None),  # Calculate scores if target provided
        return_mechanisms=return_mechanisms,
        return_trajectories=return_trajectories,
    )

    # Convert to arrays
    all_samples_array = np.array(batch_samples)
    all_is_valid_array = np.array(batch_is_valid, dtype=bool)

    # Convert scores to array if calculated
    if target_trajectory is not None and batch_scores:
        all_scores_array = np.array(
            [
                float('inf') if s is None else s
                for s in batch_scores
            ], dtype=np.float64,
        )
    else:
        all_scores_array = None

    # Count valid/invalid
    n_valid = int(np.sum(all_is_valid_array))
    n_invalid = len(all_samples_array) - n_valid

    # Create SamplingResult
    return SamplingResult(
        samples=all_samples_array,
        is_valid=all_is_valid_array,
        scores=all_scores_array,
        n_generated=len(all_samples_array),
        n_valid=n_valid,
        n_invalid=n_invalid,
        mechanisms=batch_mechanisms,
        trajectories=batch_trajectories,
    )


def generate_valid_samples(
    mechanism: Mechanism,
    dimension_bounds_spec: DimensionBoundsSpec,
    n_valid_requested: int,
    max_attempts: int | None = None,  # Default: n_valid_requested * 100
    sampling_mode: str = 'sobol',
    dimension_variation_config: DimensionVariationConfig | None = None,
    target_trajectory: TargetTrajectory | None = None,
    target_joint: str | None = None,
    metric: str = 'mse',
    phase_invariant: bool = True,
    seed: int | None = None,
    return_all: bool = False,
    return_mechanisms: bool = False,
    return_trajectories: bool = False,
) -> SamplingResult:
    """
    Generate viable mechanism samples.

    Generates `n_valid_requested` samples that are viable (can complete a full rotation
    without singularities). This function batches sample generation and validation until
    it finds enough valid samples or hits `max_attempts`.

    Args:
        mechanism: Mechanism to sample
        dimension_bounds_spec: Specification of dimensions to sample
        n_valid_requested: Number of valid samples to return
        max_attempts: Maximum samples to generate before stopping.
                     Default: n_valid_requested * 100
        sampling_mode: Sampling strategy:
            - 'sobol': Sobol quasi-random sequence (good space coverage)
            - 'behnken': Box-Behnken design (good for response surfaces, 3+ dims)
            - 'meshgrid': Regular grid (Cartesian product, grows as N^d)
            - 'full_combinatoric': Alias for 'meshgrid' (deprecated, use 'meshgrid')
        dimension_variation_config: Optional configuration to control which dimensions vary
                                   and by how much. Excluded dimensions have bounds set to their
                                   initial value (no variation).
        target_trajectory: Optional target trajectory. If provided, fitness scores will be calculated.
        target_joint: Optional joint name for viability checking and fitness evaluation
        metric: Error metric for fitness scoring ('mse', 'rmse', 'total', 'max')
        phase_invariant: Use phase-aligned scoring for fitness evaluation
        seed: Random seed for reproducibility
        return_all: If True, return ALL samples generated (up to max_attempts) with validity flags.
                   If False, return only valid samples (up to n_valid_requested).
        return_mechanisms: If True, include Mechanism objects for valid samples
        return_trajectories: If True, include trajectory data for valid samples

    Returns:
        SamplingResult: Dataclass containing:
            - samples: Array of samples (if return_all=True, includes invalid samples too)
            - is_valid: Boolean array marking which samples are viable
            - scores: Array of fitness scores if target_trajectory provided, None otherwise
            - mechanisms: Optional list of Mechanism objects (only for valid samples)
            - trajectories: Optional list of trajectory dictionaries (only for valid samples)
            - n_generated: Total samples generated (including invalid)
            - n_valid: Number of valid samples found
            - n_invalid: Number of invalid samples

    Example:
        >>> result = generate_valid_samples(
        ...     mechanism=my_mechanism,
        ...     dimension_bounds_spec=dim_spec,
        ...     n_valid_requested=32,
        ...     max_attempts=1000,
        ...     sampling_mode='sobol',
        ...     dimension_variation_config=config,
        ...     target_trajectory=target,  # Optional: enables score calculation
        ...     return_all=True,  # Get all samples (valid + invalid) for visualization
        ...     return_trajectories=True,
        ... )
        >>> # result.n_valid >= 32 (if found within max_attempts)
        >>> # result.n_generated <= 1000
        >>> # result.is_valid marks which samples are viable
        >>> # result.scores is not None if target_trajectory provided
    """
    # Set default max_attempts
    if max_attempts is None:
        max_attempts = n_valid_requested * 100

    # Apply dimension variation config if provided and identify constant dimensions
    constant_dims: dict[str, float] = {}
    if dimension_variation_config is not None:
        from target_gen.variation_config import DimensionVariationConfig
        dimension_bounds_spec = apply_dimension_variation_config(dimension_bounds_spec, dimension_variation_config)
        constant_dims = _identify_constant_dimensions(dimension_bounds_spec, dimension_variation_config)
        logger.debug(
            f'Applied dimension_variation_config: '
            f'excluded={dimension_variation_config.exclude_dimensions}, '
            f'default_range=±{dimension_variation_config.default_variation_range*100:.0f}%',
        )
        if constant_dims:
            logger.info(
                f'Constant dimensions (excluded from sampling): {list(constant_dims.keys())}',
            )

    # Create filtered dimension spec with only varying dimensions for sampling
    varying_dimension_spec = _create_varying_dimension_spec(dimension_bounds_spec, constant_dims)

    num_dims = len(varying_dimension_spec)
    if num_dims == 0:
        raise ValueError('DimensionBoundsSpec has no varying dimensions (all dimensions are constant)')

    if constant_dims:
        logger.info(
            f'Sampling {num_dims} varying dimensions (out of {len(dimension_bounds_spec)} total). '
            f'Constant dimensions will be injected after sampling.',
        )

    logger.info(
        f'Generating valid samples: sampling_mode={sampling_mode}, '
        f'requesting {n_valid_requested} valid samples, max_attempts={max_attempts}, '
        f'return_all={return_all}, return_mechanisms={return_mechanisms}, return_trajectories={return_trajectories}',
    )

    # Debug: Log dimension bounds at start (only varying dimensions)
    logger.debug('=== Varying Dimension Bounds ===')
    for i, name in enumerate(varying_dimension_spec.names):
        min_bound, max_bound = varying_dimension_spec.bounds[i]
        initial = varying_dimension_spec.initial_values[i]
        logger.debug(
            f'  {name}: bounds=[{min_bound:.4f}, {max_bound:.4f}], '
            f'initial={initial:.4f}, range={max_bound - min_bound:.4f}',
        )
    if constant_dims:
        logger.debug('=== Constant Dimensions ===')
        for name, value in constant_dims.items():
            logger.debug(f'  {name}: fixed={value:.4f}')

    # Set up fitness function if target_trajectory is provided
    fitness_func = None
    if target_trajectory is not None:
        from pylink_tools.mechanism import create_mechanism_fitness
        score_target_joint = target_joint
        if score_target_joint is None:
            score_target_joint = target_trajectory.joint_name

        fitness_func = create_mechanism_fitness(
            mechanism,
            target_trajectory,
            target_joint=score_target_joint,
            metric=metric,
            phase_invariant=phase_invariant,
            phase_align_method='rotation',
            translation_invariant=False,
        )

    # Initialize accumulation lists
    all_samples_list: list[np.ndarray] = []
    all_is_valid_list: list[bool] = []
    all_scores_list: list[float | None] = [] if target_trajectory is not None else []
    all_mechanisms_list: list[Mechanism | None] | None = [] if return_mechanisms else None
    all_trajectories_list: list[dict[str, list[list[float]]] | None] | None = [] if return_trajectories else None

    n_generated = 0
    n_valid_found = 0

    # Determine if deterministic mode
    deterministic_modes = ('meshgrid', 'full_combinatoric', 'behnken')
    is_deterministic = sampling_mode in deterministic_modes

    if is_deterministic:
        # Progressive grid refinement for deterministic modes
        # Start with small grid (n=3), then increase density until we find enough valid samples
        logger.info(
            f'Deterministic mode {sampling_mode}: using progressive grid refinement '
            f'(starting with n=3, up to {max_attempts} total samples)',
        )

        n = 3  # Start with 3 gradations per dimension
        total_generated = 0

        while n_valid_found < n_valid_requested and total_generated < max_attempts:
            # Calculate grid size for this n
            grid_size = n ** num_dims

            # Check if next grid would exceed max_attempts
            if total_generated + grid_size > max_attempts:
                # Try to fit what we can
                remaining = max_attempts - total_generated
                if remaining <= 0:
                    break
                # For meshgrid, we can't partially generate, so skip if too large
                if grid_size > remaining:
                    logger.warning(
                        f'Grid size {grid_size} would exceed max_attempts. Stopping.',
                    )
                    break

            logger.debug(f'Generating grid with n={n} (total samples: {grid_size})')

            # Generate grid samples
            sample_variations = get_mech_variations_from_spec(
                spec=varying_dimension_spec,
                n=n,
                mode='full_combinatoric' if sampling_mode in ('meshgrid', 'full_combinatoric') else sampling_mode,
                seed=seed,
            )

            # Inject constant dimensions
            sample_variations = _inject_constant_dimensions(sample_variations, constant_dims)

            # Validate samples
            batch_samples, batch_is_valid, batch_scores, batch_mechanisms, batch_trajectories = _process_sample_batch(
                sample_variations=sample_variations,
                base_mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                validation_mode='viability',
                fitness_function=fitness_func,
                return_scores=(target_trajectory is not None),
                return_mechanisms=return_mechanisms,
                return_trajectories=return_trajectories,
            )

            # Accumulate results
            all_samples_list.extend(batch_samples)
            all_is_valid_list.extend(batch_is_valid)
            if target_trajectory is not None:
                all_scores_list.extend(batch_scores)

            if return_mechanisms and batch_mechanisms is not None:
                if all_mechanisms_list is None:
                    all_mechanisms_list = []
                all_mechanisms_list.extend(batch_mechanisms)
            if return_trajectories and batch_trajectories is not None:
                if all_trajectories_list is None:
                    all_trajectories_list = []
                all_trajectories_list.extend(batch_trajectories)

            total_generated += len(batch_samples)
            n_generated = total_generated
            n_valid_found = sum(all_is_valid_list)

            logger.debug(
                f'Grid n={n}: generated {len(batch_samples)}, valid: {sum(batch_is_valid)}, '
                f'total valid so far: {n_valid_found}/{n_valid_requested}',
            )

            # Increase grid density for next iteration
            n += 1
    else:
        # Non-deterministic modes: call generate_samples() in batches
        # Each call generates unique samples (sobol sequence continues)
        logger.info(
            f'Non-deterministic mode {sampling_mode}: using batched generate_samples()',
        )

        # Start with batch size = n_valid_requested * 2 (expect ~50% valid ratio)
        base_batch_size = n_valid_requested * 2
        batch_size = min(base_batch_size, max_attempts)

        while n_valid_found < n_valid_requested and n_generated < max_attempts:
            # Calculate remaining attempts
            remaining_attempts = max_attempts - n_generated
            remaining_needed = n_valid_requested - n_valid_found

            # Adjust batch size based on observed valid ratio
            if n_generated > 0:
                observed_ratio = n_valid_found / n_generated
                if observed_ratio > 0:
                    # Estimate how many samples we need
                    estimated_needed = int(remaining_needed / observed_ratio)
                    current_batch_size = min(batch_size, remaining_attempts, estimated_needed)
                else:
                    # Very low valid ratio, use larger batches
                    current_batch_size = min(batch_size, remaining_attempts)
            else:
                current_batch_size = min(batch_size, remaining_attempts)

            if current_batch_size <= 0:
                break

            # Call generate_samples() to get a batch
            # Use seed offset to ensure unique samples across batches
            batch_seed = seed + n_generated if seed is not None else None
            batch_result = generate_samples(
                mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,  # Full spec (generate_samples handles constant dims)
                n_requested=current_batch_size,
                sampling_mode=sampling_mode,
                dimension_variation_config=dimension_variation_config,
                target_trajectory=target_trajectory,  # Pass target to calculate scores
                target_joint=target_joint,
                metric=metric,
                phase_invariant=phase_invariant,
                seed=batch_seed,
                return_mechanisms=return_mechanisms,
                return_trajectories=return_trajectories,
            )

            # Accumulate results
            all_samples_list.extend(batch_result.samples)
            all_is_valid_list.extend(batch_result.is_valid)
            if target_trajectory is not None and batch_result.scores is not None:
                all_scores_list.extend(batch_result.scores)

            if return_mechanisms and batch_result.mechanisms is not None:
                if all_mechanisms_list is None:
                    all_mechanisms_list = []
                all_mechanisms_list.extend(batch_result.mechanisms)
            if return_trajectories and batch_result.trajectories is not None:
                if all_trajectories_list is None:
                    all_trajectories_list = []
                all_trajectories_list.extend(batch_result.trajectories)

            n_generated += batch_result.n_generated
            n_valid_found = sum(all_is_valid_list)

            logger.debug(
                f'Batch: generated {batch_result.n_generated}, valid: {batch_result.n_valid}, '
                f'total valid so far: {n_valid_found}/{n_valid_requested}',
            )

    # Validate we generated samples
    if len(all_samples_list) == 0:
        raise ValueError(
            f'No samples generated. Check max_attempts and dimension_bounds_spec.',
        )

    # Convert to arrays efficiently
    all_samples_array = np.array(all_samples_list)
    all_is_valid_array = np.array(all_is_valid_list, dtype=bool)

    # Convert scores to array if calculated
    if target_trajectory is not None and all_scores_list:
        all_scores_array = np.array(
            [
                float('inf') if s is None else s
                for s in all_scores_list
            ], dtype=np.float64,
        )
    else:
        all_scores_array = None

    # Compute final statistics
    n_valid = int(np.sum(all_is_valid_array))
    n_invalid = len(all_is_valid_array) - n_valid

    # Validate we have valid samples
    if n_valid == 0:
        raise ValueError(
            f'No valid samples found after {n_generated} attempts. '
            'Mechanism may be over-constrained or bounds may be too wide.',
        )

    # Select samples based on return_all flag
    if return_all:
        # Return all samples with validity flags
        selected_samples = all_samples_array
        selected_is_valid = all_is_valid_array
        selected_scores = all_scores_array
        selected_mechanisms = all_mechanisms_list
        selected_trajectories = all_trajectories_list
        # n_invalid is already calculated above based on all_is_valid_array
    else:
        # Return only valid samples (up to n_valid_requested)
        valid_indices = np.where(all_is_valid_array)[0][:n_valid_requested]
        selected_samples = all_samples_array[valid_indices]
        selected_is_valid = all_is_valid_array[valid_indices]
        selected_scores = all_scores_array[valid_indices] if all_scores_array is not None else None

        if return_mechanisms and all_mechanisms_list is not None:
            selected_mechanisms = [all_mechanisms_list[i] for i in valid_indices]
        else:
            selected_mechanisms = None

        if return_trajectories and all_trajectories_list is not None:
            selected_trajectories = [all_trajectories_list[i] for i in valid_indices]
        else:
            selected_trajectories = None

        # Update n_valid and n_invalid based on returned samples
        n_valid = len(valid_indices)
        # When return_all=False, we only return valid samples, so n_invalid = 0
        n_invalid = 0

    # Log comprehensive summary
    valid_ratio = n_valid / n_generated if n_generated > 0 else 0.0
    logger.info(
        f'=== Sampling Summary ===\n'
        f'  Sampling mode: {sampling_mode}\n'
        f'  Requested: {n_valid_requested}, Found: {n_valid}, Generated: {n_generated}\n'
        f'  Valid ratio: {valid_ratio:.1%} ({n_valid}/{n_generated})\n'
        f'  Return all: {return_all}, Return mechanisms: {return_mechanisms}, Return trajectories: {return_trajectories}',
    )

    # Warning if insufficient samples found
    if n_valid < n_valid_requested and not return_all:
        logger.warning(
            f'  Only found {n_valid}/{n_valid_requested} valid samples '
            f'({(n_valid/n_valid_requested)*100:.1f}% of requested)',
        )

    # Debug: Log ranges of final returned samples (only if debug logging enabled)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('=== Final Returned Sample Ranges ===')
        for i, name in enumerate(dimension_bounds_spec.names):
            dim_values = selected_samples[:, i]
            min_found = float(np.min(dim_values))
            max_found = float(np.max(dim_values))
            mean_found = float(np.mean(dim_values))
            logger.debug(
                f'  {name}: range=[{min_found:.4f}, {max_found:.4f}], mean={mean_found:.4f}',
            )

    # Return SamplingResult
    return SamplingResult(
        samples=selected_samples,
        is_valid=selected_is_valid,
        scores=selected_scores,  # None if target_trajectory not provided
        n_generated=n_generated,
        n_valid=n_valid,
        n_invalid=n_invalid,
        mechanisms=selected_mechanisms,
        trajectories=selected_trajectories,
    )


def generate_good_samples(
    mechanism: Mechanism,
    dimension_bounds_spec: DimensionBoundsSpec,
    target_trajectory: TargetTrajectory,
    n_good_requested: int,
    epsilon: float = 500.0,
    max_attempts: int | None = None,  # Default: n_good_requested * 200
    sampling_mode: str = 'sobol',
    dimension_variation_config: DimensionVariationConfig | None = None,
    target_joint: str | None = None,
    metric: str = 'mse',
    phase_invariant: bool = True,
    seed: int | None = None,
    return_all: bool = False,
    return_mechanisms: bool = False,
    return_trajectories: bool = False,
) -> SamplingResult:
    """
    Generate samples that are epsilon-close to a target trajectory.

    Generates `n_good_requested` samples that have fitness score <= `epsilon`
    (i.e., are "good" matches to the target trajectory). This function batches
    sample generation and fitness evaluation until it finds enough good samples
    or hits `max_attempts`.

    Args:
        mechanism: Mechanism to sample
        dimension_spec: Specification of dimensions to sample
        target_trajectory: Target trajectory to match (required)
        n_good_requested: Number of good samples to return (score <= epsilon)
        epsilon: Fitness score threshold (default: 500.0). Sample is "good" if score <= epsilon.
        max_attempts: Maximum samples to generate before stopping.
                     Default: n_good_requested * 200
        sampling_mode: Sampling strategy ('sobol', 'behnken', 'meshgrid', 'full_combinatoric')
        dimension_variation_config: Optional configuration to control which dimensions vary
                                   and by how much. Excluded dimensions have bounds set to their
                                   initial value (no variation).
        target_joint: Optional joint name for fitness evaluation
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        phase_invariant: Use phase-aligned scoring
        seed: Random seed for reproducibility
        return_all: If True, return ALL samples generated (up to max_attempts) with validity flags.
                   If False, return only good samples (up to n_good_requested).
        return_mechanisms: If True, include Mechanism objects for good samples
        return_trajectories: If True, include trajectory data for good samples

    Returns:
        SamplingResult: Dataclass containing:
            - samples: Array of samples (if return_all=True, includes all samples)
            - is_valid: Boolean array marking which samples are good (score <= epsilon)
            - scores: Array of fitness scores for all samples
            - mechanisms: Optional list of Mechanism objects (only for good samples)
            - trajectories: Optional list of trajectory dictionaries (only for good samples)
            - n_generated: Total samples generated
            - n_valid: Number of good samples found (score <= epsilon)
            - n_invalid: Number of samples with score > epsilon

    Example:
        >>> result = generate_good_samples(
        ...     mechanism=my_mechanism,
        ...     dimension_bounds_spec=dim_spec,
        ...     target_trajectory=target_trajectory,
        ...     n_good_requested=16,
        ...     epsilon=500.0,
        ...     max_attempts=5000,
        ...     sampling_mode='sobol',
        ...     dimension_variation_config=config,
        ...     metric='mse',
        ...     return_all=True,  # See all samples and their scores
        ...     return_trajectories=True,
        ... )
        >>> # result.n_valid >= 16 (if found within max_attempts)
        >>> # result.scores contains fitness scores
        >>> # result.is_valid marks samples with score <= 500
    """
    # Set default max_attempts
    if max_attempts is None:
        max_attempts = n_good_requested * 200

    # Apply dimension variation config if provided and identify constant dimensions
    constant_dims: dict[str, float] = {}
    if dimension_variation_config is not None:
        from target_gen.variation_config import DimensionVariationConfig
        dimension_bounds_spec = apply_dimension_variation_config(dimension_bounds_spec, dimension_variation_config)
        constant_dims = _identify_constant_dimensions(dimension_bounds_spec, dimension_variation_config)
        logger.debug(
            f'Applied dimension_variation_config: '
            f'excluded={dimension_variation_config.exclude_dimensions}, '
            f'default_range=±{dimension_variation_config.default_variation_range*100:.0f}%',
        )
        if constant_dims:
            logger.info(
                f'Constant dimensions (excluded from sampling): {list(constant_dims.keys())}',
            )

    # Create filtered dimension spec with only varying dimensions for sampling
    varying_dimension_spec = _create_varying_dimension_spec(dimension_bounds_spec, constant_dims)

    num_dims = len(varying_dimension_spec)
    if num_dims == 0:
        raise ValueError('DimensionBoundsSpec has no varying dimensions (all dimensions are constant)')

    if constant_dims:
        logger.info(
            f'Sampling {num_dims} varying dimensions (out of {len(dimension_bounds_spec)} total). '
            f'Constant dimensions will be injected after sampling.',
        )

    logger.info(
        f'Generating good samples: sampling_mode={sampling_mode}, '
        f'requesting {n_good_requested} good samples (epsilon={epsilon}), max_attempts={max_attempts}, '
        f'return_all={return_all}, return_mechanisms={return_mechanisms}, return_trajectories={return_trajectories}',
    )

    # Set up fitness function
    from pylink_tools.mechanism import create_mechanism_fitness
    score_target_joint = target_joint
    if score_target_joint is None:
        score_target_joint = target_trajectory.joint_name

    fitness_func = create_mechanism_fitness(
        mechanism,
        target_trajectory,
        target_joint=score_target_joint,
        metric=metric,
        phase_invariant=phase_invariant,
        phase_align_method='rotation',
        translation_invariant=False,
    )

    # Initialize accumulation lists
    all_samples_list: list[np.ndarray] = []
    all_is_valid_list: list[bool] = []
    all_scores_list: list[float] = []
    all_mechanisms_list: list[Mechanism | None] | None = [] if return_mechanisms else None
    all_trajectories_list: list[dict[str, list[list[float]]] | None] | None = [] if return_trajectories else None

    n_generated = 0
    n_good_found = 0

    # Determine if deterministic mode
    deterministic_modes = ('meshgrid', 'full_combinatoric', 'behnken')
    is_deterministic = sampling_mode in deterministic_modes

    if is_deterministic:
        # Progressive grid refinement for deterministic modes
        # Use generate_samples for each grid level to leverage its score calculation
        logger.info(
            f'Deterministic mode {sampling_mode}: using progressive grid refinement '
            f'(starting with n=3, up to {max_attempts} total samples)',
        )

        n = 3  # Start with 3 gradations per dimension
        total_generated = 0

        while n_good_found < n_good_requested and total_generated < max_attempts:
            # Calculate grid size for this n
            grid_size = n ** num_dims

            # Check if next grid would exceed max_attempts
            if total_generated + grid_size > max_attempts:
                remaining = max_attempts - total_generated
                if remaining <= 0:
                    break
                if grid_size > remaining:
                    logger.warning(
                        f'Grid size {grid_size} would exceed max_attempts. Stopping.',
                    )
                    break

            logger.debug(f'Generating grid with n={n} (total samples: {grid_size})')

            # Use generate_samples for this grid level - it handles score calculation
            # For meshgrid, we need to pass n_requested that results in exactly n gradations
            # But generate_samples calculates n_gradations from n_requested, so we need to work backwards
            # Actually, we can't easily control exact n with generate_samples, so we'll generate
            # the grid directly and then use _process_sample_batch for consistency
            sample_variations = get_mech_variations_from_spec(
                spec=varying_dimension_spec,
                n=n,
                mode='full_combinatoric' if sampling_mode in ('meshgrid', 'full_combinatoric') else sampling_mode,
                seed=seed,
            )

            # Inject constant dimensions
            sample_variations = _inject_constant_dimensions(sample_variations, constant_dims)

            # Process samples: evaluate fitness and mark as "good" if score <= epsilon
            batch_samples, batch_is_valid, batch_scores, batch_mechanisms, batch_trajectories = _process_sample_batch(
                sample_variations=sample_variations,
                base_mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                validation_mode='fitness',
                fitness_function=fitness_func,
                return_scores=True,
                return_mechanisms=return_mechanisms,
                return_trajectories=return_trajectories,
            )

            # Mark samples as "good" if score <= epsilon
            batch_scores_array = np.array([
                float('inf') if s is None else s
                for s in batch_scores
            ])
            batch_is_good = batch_is_valid & (batch_scores_array <= epsilon)

            # Accumulate results
            all_samples_list.extend(batch_samples)
            all_is_valid_list.extend(batch_is_good)
            all_scores_list.extend(batch_scores)

            if return_mechanisms and batch_mechanisms is not None:
                if all_mechanisms_list is None:
                    all_mechanisms_list = []
                all_mechanisms_list.extend(batch_mechanisms)
            if return_trajectories and batch_trajectories is not None:
                if all_trajectories_list is None:
                    all_trajectories_list = []
                all_trajectories_list.extend(batch_trajectories)

            total_generated += len(batch_samples)
            n_generated = total_generated
            n_good_found = sum(all_is_valid_list)

            logger.debug(
                f'Grid n={n}: generated {len(batch_samples)}, good: {sum(batch_is_good)}, '
                f'total good so far: {n_good_found}/{n_good_requested}',
            )

            # Increase grid density for next iteration
            n += 1
    else:
        # Non-deterministic modes: call generate_samples() in batches, then evaluate fitness
        logger.info(
            f'Non-deterministic mode {sampling_mode}: using batched generate_samples() with fitness evaluation',
        )

        # Start with batch size = n_good_requested * 5 (expect ~20% good ratio)
        base_batch_size = n_good_requested * 5
        batch_size = min(base_batch_size, max_attempts)

        while n_good_found < n_good_requested and n_generated < max_attempts:
            # Calculate remaining attempts
            remaining_attempts = max_attempts - n_generated
            remaining_needed = n_good_requested - n_good_found

            # Adjust batch size based on observed good ratio
            if n_generated > 0:
                observed_ratio = n_good_found / n_generated
                if observed_ratio > 0:
                    estimated_needed = int(remaining_needed / observed_ratio)
                    current_batch_size = min(batch_size, remaining_attempts, estimated_needed)
                else:
                    current_batch_size = min(batch_size, remaining_attempts)
            else:
                current_batch_size = min(batch_size, remaining_attempts)

            if current_batch_size <= 0:
                break

            # Call generate_samples() to get a batch with scores already calculated
            batch_seed = seed + n_generated if seed is not None else None
            batch_result = generate_samples(
                mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                n_requested=current_batch_size,
                sampling_mode=sampling_mode,
                dimension_variation_config=dimension_variation_config,
                target_trajectory=target_trajectory,  # Pass target to calculate scores
                target_joint=target_joint,
                metric=metric,
                phase_invariant=phase_invariant,
                seed=batch_seed,
                return_mechanisms=return_mechanisms,
                return_trajectories=return_trajectories,
            )

            # Filter by epsilon threshold: mark samples as "good" if score <= epsilon
            # batch_result.scores is already calculated by generate_samples
            if batch_result.scores is not None:
                batch_scores_array = batch_result.scores
                # Mark as "good" if viable AND score <= epsilon
                batch_is_good = batch_result.is_valid & (batch_scores_array <= epsilon)
            else:
                # Should not happen if target_trajectory is provided, but handle gracefully
                batch_scores_array = np.full(len(batch_result.samples), float('inf'))
                batch_is_good = np.zeros(len(batch_result.samples), dtype=bool)

            # Accumulate results
            all_samples_list.extend(batch_result.samples)
            all_is_valid_list.extend(batch_is_good)
            all_scores_list.extend(batch_scores_array)

            if return_mechanisms and batch_result.mechanisms is not None:
                if all_mechanisms_list is None:
                    all_mechanisms_list = []
                all_mechanisms_list.extend(batch_result.mechanisms)
            if return_trajectories and batch_result.trajectories is not None:
                if all_trajectories_list is None:
                    all_trajectories_list = []
                all_trajectories_list.extend(batch_result.trajectories)

            n_generated += batch_result.n_generated
            n_good_found = sum(all_is_valid_list)

            logger.debug(
                f'Batch: generated {batch_result.n_generated}, good: {sum(batch_is_good)}, '
                f'total good so far: {n_good_found}/{n_good_requested}',
            )

    # Validate we generated samples
    if len(all_samples_list) == 0:
        raise ValueError(
            f'No samples generated. Check max_attempts and dimension_bounds_spec.',
        )

    # Convert to arrays efficiently
    all_samples_array = np.array(all_samples_list)
    all_is_valid_array = np.array(all_is_valid_list, dtype=bool)
    all_scores_array = np.array(all_scores_list, dtype=np.float64)

    # Compute final statistics
    n_valid = int(np.sum(all_is_valid_array))
    n_invalid = len(all_is_valid_array) - n_valid

    # Validate we have good samples
    if n_valid == 0:
        raise ValueError(
            f'No good samples found (score <= {epsilon}) after {n_generated} attempts. '
            'Consider increasing epsilon or max_attempts.',
        )

    # Select samples based on return_all flag
    if return_all:
        # Return all samples with validity flags
        selected_samples = all_samples_array
        selected_is_valid = all_is_valid_array
        selected_scores = all_scores_array
        selected_mechanisms = all_mechanisms_list
        selected_trajectories = all_trajectories_list
    else:
        # Return only good samples (up to n_good_requested)
        good_indices = np.where(all_is_valid_array)[0][:n_good_requested]
        selected_samples = all_samples_array[good_indices]
        selected_is_valid = all_is_valid_array[good_indices]
        selected_scores = all_scores_array[good_indices]

        if return_mechanisms and all_mechanisms_list is not None:
            selected_mechanisms = [all_mechanisms_list[i] for i in good_indices]
        else:
            selected_mechanisms = None

        if return_trajectories and all_trajectories_list is not None:
            selected_trajectories = [all_trajectories_list[i] for i in good_indices]
        else:
            selected_trajectories = None

        # Update n_valid to actual number returned
        n_valid = len(good_indices)
        n_invalid = n_generated - n_valid

    # Log comprehensive summary
    valid_ratio = n_valid / n_generated if n_generated > 0 else 0.0
    logger.info(
        f'=== Sampling Summary ===\n'
        f'  Sampling mode: {sampling_mode}\n'
        f'  Requested: {n_good_requested} good samples (epsilon={epsilon}), Found: {n_valid}, Generated: {n_generated}\n'
        f'  Good ratio: {valid_ratio:.1%} ({n_valid}/{n_generated})\n'
        f'  Return all: {return_all}, Return mechanisms: {return_mechanisms}, Return trajectories: {return_trajectories}',
    )

    # Log fitness score statistics
    if len(selected_scores) > 0:
        good_scores_mask = selected_is_valid & np.isfinite(selected_scores)
        if np.any(good_scores_mask):
            good_scores_only = selected_scores[good_scores_mask]
            min_score = float(np.min(good_scores_only))
            max_score = float(np.max(good_scores_only))
            mean_score = float(np.mean(good_scores_only))
            logger.info(
                f'  Fitness scores - Min: {min_score:.4f}, Max: {max_score:.4f}, Mean: {mean_score:.4f}',
            )

    # Warning if insufficient samples found
    if n_valid < n_good_requested and not return_all:
        logger.warning(
            f'  Only found {n_valid}/{n_good_requested} good samples '
            f'({(n_valid/n_good_requested)*100:.1f}% of requested)',
        )

    # Return SamplingResult
    return SamplingResult(
        samples=selected_samples,
        is_valid=selected_is_valid,
        scores=selected_scores,
        n_generated=n_generated,
        n_valid=n_valid,
        n_invalid=n_invalid,
        mechanisms=selected_mechanisms,
        trajectories=selected_trajectories,
    )
