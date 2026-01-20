"""
optimization_helpers.py - Helper functions for linkage dimension extraction and application.

Contains utility functions for working with linkage dimensions:
  - extract_dimensions: Extract optimizable dimensions from pylink_data
  - apply_dimensions: Apply dimension values to pylink_data
  - Conversion utilities between dict and tuple formats
"""
from __future__ import annotations

import copy
import logging

from pylink_tools.optimization_types import DimensionSpec
from pylink_tools.optimization_types import TargetTrajectory
logger = logging.getLogger(__name__)


def extract_dimensions(
    pylink_data: dict,
    bounds_factor: float = 2.0,
    min_length: float = 0.1,
) -> DimensionSpec:
    """
    Extract optimizable dimensions (link lengths) from pylink_data.

    Identifies all link lengths that can be adjusted:
      - Crank: distance (length from parent to crank point)
      - Revolute: distance0, distance1 (lengths to each parent)

    Static joints are NOT included (they are fixed ground points).

    Args:
        pylink_data: Full pylink document with 'pylinkage' section
        bounds_factor: Multiplier for suggested bounds (e.g., 2.0 means
                       bounds are [value/2, value*2])
        min_length: Minimum allowed link length

    Returns:
        DimensionSpec with names, initial values, bounds, and joint mapping

    Example:
        >>> spec = extract_dimensions(pylink_data)
        >>> print(spec.names)
        ['B_distance', 'C_distance0', 'C_distance1']
        >>> print(spec.initial_values)
        [1.0, 3.0, 1.0]
    """
    pylinkage_data = pylink_data.get('pylinkage', {})
    joints_data = pylinkage_data.get('joints', [])

    names: list[str] = []
    initial_values: list[float] = []
    bounds: list[tuple[float, float]] = []
    joint_mapping: dict[str, tuple[str, str]] = {}

    for jdata in joints_data:
        jtype = jdata.get('type')
        jname = jdata.get('name')

        if jtype == 'Crank':
            # Crank has a single 'distance' parameter
            distance = jdata.get('distance', 1.0)
            dim_name = f'{jname}_distance'

            names.append(dim_name)
            initial_values.append(distance)
            bounds.append(_compute_bounds(distance, bounds_factor, min_length))
            joint_mapping[dim_name] = (jname, 'distance')

        elif jtype == 'Revolute':
            # Revolute has distance0 and distance1
            distance0 = jdata.get('distance0', 1.0)
            distance1 = jdata.get('distance1', 1.0)

            dim_name0 = f'{jname}_distance0'
            dim_name1 = f'{jname}_distance1'

            names.append(dim_name0)
            initial_values.append(distance0)
            bounds.append(_compute_bounds(distance0, bounds_factor, min_length))
            joint_mapping[dim_name0] = (jname, 'distance0')

            names.append(dim_name1)
            initial_values.append(distance1)
            bounds.append(_compute_bounds(distance1, bounds_factor, min_length))
            joint_mapping[dim_name1] = (jname, 'distance1')

        # Static joints are skipped (fixed ground points)

    return DimensionSpec(
        names=names,
        initial_values=initial_values,
        bounds=bounds,
        joint_mapping=joint_mapping,
    )


def _compute_bounds(
    value: float,
    factor: float,
    min_length: float,
) -> tuple[float, float]:
    """Compute bounds for a dimension value."""
    lower = max(min_length, value / factor)
    upper = value * factor
    return (lower, upper)


def extract_dimensions_with_custom_bounds(
    pylink_data: dict,
    custom_bounds: dict[str, tuple[float, float]],
) -> DimensionSpec:
    """
    Extract dimensions with user-specified bounds.

    Args:
        pylink_data: Full pylink document
        custom_bounds: Dict of {dimension_name: (min, max)}
                       Dimensions not in this dict use default bounds.

    Returns:
        DimensionSpec with custom bounds applied
    """
    spec = extract_dimensions(pylink_data)

    # Override bounds where specified
    for i, name in enumerate(spec.names):
        if name in custom_bounds:
            spec.bounds[i] = custom_bounds[name]

    return spec


def apply_dimensions(
    pylink_data: dict,
    dimension_values: dict[str, float],
    dimension_spec: DimensionSpec | None = None,
    inplace: bool = False,
) -> dict:
    """
    Apply dimension values to pylink_data.

    Updates the link lengths in the pylinkage.joints list OR linkage.edges dict
    based on the provided dimension values. Automatically detects the format.

    Args:
        pylink_data: Original pylink document (legacy pylinkage.joints or hypergraph linkage.edges)
        dimension_values: Dict of {dimension_name: new_value}
        dimension_spec: Optional DimensionSpec for validation/mapping.
                        If not provided, uses naming convention to infer mapping.
        inplace: If True, mutate pylink_data directly (faster for optimization).
                 If False (default), return a deep copy with changes.

    Returns:
        Updated pylink_data dict (same object if inplace=True, new copy if False)

    Example:
        >>> updated = apply_dimensions(pylink_data, {"B_distance": 2.5})
        >>> # Joint B now has distance=2.5 (original unchanged)

        >>> # For optimization hot loops, use inplace=True:
        >>> apply_dimensions(working_copy, {"B_distance": 2.5}, inplace=True)
    """
    if inplace:
        updated = pylink_data
    else:
        updated = copy.deepcopy(pylink_data)

    # Detect format: hypergraph (linkage.edges) vs legacy (pylinkage.joints)
    is_hypergraph = 'linkage' in updated and 'edges' in updated.get('linkage', {})

    if is_hypergraph:
        # Hypergraph format: apply to linkage.edges
        edges_data = updated.get('linkage', {}).get('edges', {})

        # Get edge mapping from dimension_spec
        if dimension_spec is not None and hasattr(dimension_spec, 'edge_mapping'):
            edge_mapping = dimension_spec.edge_mapping
        else:
            # Infer edge mapping: "edge_name_property" -> ("edge_name", "property")
            edge_mapping = {}
            for dim_name in dimension_values.keys():
                # Try to find matching edge by stripping _distance suffix
                if dim_name.endswith('_distance'):
                    edge_name = dim_name[:-9]  # Remove '_distance'
                    if edge_name in edges_data:
                        edge_mapping[dim_name] = (edge_name, 'distance')

        # Apply each dimension to edges
        for dim_name, new_value in dimension_values.items():
            if dim_name in edge_mapping:
                edge_name, prop_name = edge_mapping[dim_name]
                if edge_name in edges_data:
                    edges_data[edge_name][prop_name] = new_value

    else:
        # Legacy format: apply to pylinkage.joints
        joints_data = updated.get('pylinkage', {}).get('joints', [])
        joint_by_name = {j['name']: j for j in joints_data}

        # Get mapping from dimension_spec or infer from naming convention
        if dimension_spec is not None:
            mapping = dimension_spec.joint_mapping
        else:
            mapping = _infer_mapping_from_names(dimension_values.keys())

        # Apply each dimension
        for dim_name, new_value in dimension_values.items():
            if dim_name in mapping:
                joint_name, prop_name = mapping[dim_name]
                if joint_name in joint_by_name:
                    joint_by_name[joint_name][prop_name] = new_value
            else:
                # Try to infer from naming convention
                parts = dim_name.rsplit('_', 1)
                if len(parts) == 2:
                    joint_name, prop_name = parts
                    if joint_name in joint_by_name:
                        joint_by_name[joint_name][prop_name] = new_value

    return updated


def apply_dimensions_from_array(
    pylink_data: dict,
    values: tuple[float, ...],
    dimension_spec: DimensionSpec,
    inplace: bool = False,
) -> dict:
    """
    Apply dimension values from a numeric array (for optimizer callbacks).

    Args:
        pylink_data: Original pylink document
        values: Tuple/list of values in same order as dimension_spec.names
        dimension_spec: Spec defining the mapping
        inplace: If True, mutate pylink_data directly (faster for optimization).
                 If False (default), return a deep copy with changes.

    Returns:
        Updated pylink_data with new dimensions
    """
    if len(values) != len(dimension_spec.names):
        raise ValueError(
            f'Expected {len(dimension_spec.names)} values, got {len(values)}',
        )

    dimension_values = dict(zip(dimension_spec.names, values))
    return apply_dimensions(pylink_data, dimension_values, dimension_spec, inplace=inplace)


def _infer_mapping_from_names(
    dim_names: list[str],
) -> dict[str, tuple[str, str]]:
    """
    Infer joint mapping from dimension names using naming convention.

    Expected format: "{joint_name}_{property_name}"
    e.g., "B_distance" -> ("B", "distance")
          "C_distance0" -> ("C", "distance0")
    """
    mapping = {}
    for name in dim_names:
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            joint_name, prop_name = parts
            mapping[name] = (joint_name, prop_name)
    return mapping


# =============================================================================
# Utility Functions
# =============================================================================

def dimensions_to_dict(
    values: tuple[float, ...],
    spec: DimensionSpec,
) -> dict[str, float]:
    """Convert dimension array to named dict."""
    return dict(zip(spec.names, values))


def dict_to_dimensions(
    values_dict: dict[str, float],
    spec: DimensionSpec,
) -> tuple[float, ...]:
    """Convert named dict to dimension array (in spec order)."""
    return tuple(
        values_dict.get(name, spec.initial_values[i])
        for i, name in enumerate(spec.names)
    )


def validate_bounds(
    values: tuple[float, ...],
    spec: DimensionSpec,
) -> list[str]:
    """
    Check if values are within bounds.

    Returns:
        List of violation messages (empty if all valid)
    """
    violations = []
    for i, (value, (lower, upper)) in enumerate(zip(values, spec.bounds)):
        if value < lower:
            violations.append(f'{spec.names[i]}: {value} < {lower} (min)')
        elif value > upper:
            violations.append(f'{spec.names[i]}: {value} > {upper} (max)')
    return violations


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
        >>> gradations = get_link_gradations(names, bounds, n=3)
        >>> gradations
        {
            'crank': [10.0, 15.0, 20.0],
            'coupler': [30.0, 40.0, 50.0],
            'rocker': [15.0, 20.0, 25.0]
        }
    """
    import numpy as np

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
        gradations: Dict from get_link_gradations, mapping link name -> list of values

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
    import numpy as np
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
                f"Box-Behnken requires at least 3 factors, got {num_dims}. "
                'Falling back to full_combinatoric.',
            )
            gradations = get_combinatoric_gradations(spec.names, spec.bounds, n)
            return get_mech_variations(gradations)

        try:
            from pyDOE3 import bbdesign
        except ImportError as e:
            logger.error(
                f"pyDOE3 not installed. Install with: pip install pyDOE3. Error: {e}",
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

            logger.info(f"Box-Behnken design generated {len(variations)} points for {num_dims} factors")
            return variations

        except Exception as e:
            logger.error(f"Box-Behnken design failed: {e}")
            raise

    elif mode == 'sobol':
        # Sobol quasi-random sequence - good space-filling properties
        # Note: Sobol rounds n up to the next power of 2 by default
        try:
            from pyDOE3 import sobol_sequence
        except ImportError as e:
            logger.error(
                f"pyDOE3 not installed. Install with: pip install pyDOE3. Error: {e}",
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

            logger.info(f"Sobol sequence generated {len(variations)} points for {num_dims} dimensions")
            return variations

        except Exception as e:
            logger.error(f"Sobol sequence generation failed: {e}")
            raise

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: 'full_combinatoric', 'behnken', 'sobol'",
        )


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
):
    """
    Pre-sample the design space and return the best valid configurations.

    This function generates sample points using DOE methods (Sobol, Box-Behnken,
    or full combinatoric), evaluates each with the fitness function, filters
    to keep only valid configurations, and returns the best ones sorted by score.

    Use this to initialize PSO particles in valid regions of the search space,
    dramatically improving convergence for constrained mechanisms.

    Args:
        pylink_data: Base linkage configuration
        target: Target trajectory to match
        dimension_spec: Dimensions to sample
        n_samples: How many samples to generate (for Sobol/combinatoric)
        n_best: Maximum number of best positions to return
        mode: Sampling strategy:
            - 'sobol': Sobol quasi-random sequence (good space coverage)
            - 'behnken': Box-Behnken design (good for response surfaces, 3+ dims)
            - 'full_combinatoric': Full grid (warning: grows as N^d)
        metric: Error metric for scoring ('mse', 'rmse', 'total', 'max')
        seed: Random seed for reproducibility
        phase_invariant: Use phase-aligned scoring

    Returns:
        (positions, scores): Tuple of:
            - positions: ndarray of shape (n_valid, n_dims) with best valid positions
            - scores: ndarray of shape (n_valid,) with corresponding fitness scores
            where n_valid <= n_best (may be fewer if not enough valid samples found)

    Example:
        >>> positions, scores = presample_valid_positions(
        ...     pylink_data, target, dim_spec,
        ...     n_samples=256, n_best=64, mode='sobol'
        ... )
        >>> print(f"Found {len(positions)} valid configurations")
        >>> print(f"Best score: {scores[0]:.4f}")
    """
    import numpy as np
    from pylink_tools.optimize import create_fitness_function

    # Generate samples using the specified DOE method
    try:
        variations = get_mech_variations_from_spec(
            spec=dimension_spec,
            n=n_samples,
            mode=mode,
            seed=seed,
        )
    except Exception as e:
        logger.warning(f"Presampling with mode='{mode}' failed: {e}. Using random sampling.")
        # Fallback to random sampling
        lower = np.array([b[0] for b in dimension_spec.bounds])
        upper = np.array([b[1] for b in dimension_spec.bounds])
        if seed is not None:
            np.random.seed(seed)
        random_samples = np.random.uniform(lower, upper, (n_samples, len(dimension_spec)))
        variations = [
            dict(zip(dimension_spec.names, row))
            for row in random_samples
        ]

    logger.info(f"Presampling {len(variations)} configurations using mode='{mode}'")

    # Create fitness function
    fitness = create_fitness_function(
        pylink_data, target, dimension_spec,
        metric=metric, phase_invariant=phase_invariant,
    )

    # Evaluate all samples and keep valid ones
    results = []
    n_invalid = 0
    for var in variations:
        # Convert dict to tuple in spec order
        dims = tuple(var[name] for name in dimension_spec.names)
        score = fitness(dims)

        if np.isfinite(score):
            results.append((score, dims))
        else:
            n_invalid += 1

    valid_rate = (len(results) / len(variations) * 100) if variations else 0
    logger.info(f"Presampling: {len(results)}/{len(variations)} valid ({valid_rate:.1f}%), {n_invalid} invalid")

    if not results:
        logger.warning('No valid configurations found during presampling!')
        return np.array([]).reshape(0, len(dimension_spec)), np.array([])

    # Sort by score (best first) and take top n_best
    results.sort(key=lambda x: x[0])
    best_results = results[:n_best]

    # Convert to numpy arrays
    positions = np.array([r[1] for r in best_results])
    scores = np.array([r[0] for r in best_results])

    logger.info(f"Presampling: returning {len(positions)} best positions (best score: {scores[0]:.4f})")

    return positions, scores
