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
    exclude_edges: set[str] | None = None,
    custom_bounds: dict[str, tuple[float, float]] | None = None,
) -> DimensionSpec:
    """
    Extract optimizable dimensions (link lengths) from pylink_data.

    Supports both formats:
      - Legacy (pylinkage.joints): extracts Crank.distance, Revolute.distance0/1
      - Hypergraph (linkage.edges): extracts edge distances

    Static joints / ground edges are NOT included (they are fixed).

    Args:
        pylink_data: Full pylink document (either format)
        bounds_factor: Multiplier for suggested bounds (e.g., 2.0 means
                       bounds are [value/2, value*2])
        min_length: Minimum allowed link length
        exclude_edges: Edge IDs to exclude (for hypergraph format, defaults to {'ground'})
        custom_bounds: Optional dict of {dimension_name: (min, max)} to override
                       auto-computed bounds. Dimensions not in this dict use defaults.

    Returns:
        DimensionSpec with names, initial values, bounds, and mapping

    Example:
        >>> spec = extract_dimensions(pylink_data)
        >>> print(spec.names)
        ['B_distance', 'C_distance0', 'C_distance1']
        >>> print(spec.initial_values)
        [1.0, 3.0, 1.0]

        >>> # With custom bounds for specific dimensions:
        >>> spec = extract_dimensions(pylink_data, custom_bounds={'B_distance': (0.5, 10.0)})
    """
    # Auto-detect format: hypergraph (linkage.nodes/edges) vs legacy (pylinkage.joints)
    is_hypergraph = 'linkage' in pylink_data and 'edges' in pylink_data.get('linkage', {})

    if is_hypergraph:
        # Delegate to hypergraph extraction
        spec = extract_dimensions_from_edges(
            pylink_data,
            bounds_factor=bounds_factor,
            min_length=min_length,
            exclude_edges=exclude_edges or {'ground'},
        )
        # Apply custom bounds overrides if provided
        if custom_bounds:
            for i, name in enumerate(spec.names):
                if name in custom_bounds:
                    spec.bounds[i] = custom_bounds[name]
        return spec

    # Legacy format: pylinkage.joints
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

    # Apply custom bounds overrides if provided
    if custom_bounds:
        for i, name in enumerate(names):
            if name in custom_bounds:
                bounds[i] = custom_bounds[name]

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


def extract_dimensions_from_edges(
    pylink_data: dict,
    bounds_factor: float = 2.0,
    min_length: float = 0.1,
    exclude_edges: set[str] | None = None,
) -> DimensionSpec:
    """
    Extract optimizable dimensions from hypergraph linkage.edges format.

    This handles the 'linkage.edges' format where each edge has a 'distance' property.

    Args:
        pylink_data: Mechanism data with 'linkage.edges' structure
        bounds_factor: Multiplier for bounds (e.g., 2.0 = [val/2, val*2])
        min_length: Minimum allowed link length
        exclude_edges: Edge IDs to exclude (e.g., {'ground'})

    Returns:
        DimensionSpec with edge distances as optimizable parameters
    """
    if exclude_edges is None:
        exclude_edges = {'ground'}  # Ground link is typically fixed

    linkage = pylink_data.get('linkage', {})
    edges = linkage.get('edges', {})

    names: list[str] = []
    initial_values: list[float] = []
    bounds: list[tuple[float, float]] = []
    edge_mapping: dict[str, tuple[str, str]] = {}

    for edge_id, edge_data in edges.items():
        if edge_id in exclude_edges:
            continue

        distance = edge_data.get('distance', 1.0)
        dim_name = f'{edge_id}_distance'

        names.append(dim_name)
        initial_values.append(distance)
        bounds.append(_compute_bounds(distance, bounds_factor, min_length))
        edge_mapping[dim_name] = (edge_id, 'distance')

    return DimensionSpec(
        names=names,
        initial_values=initial_values,
        bounds=bounds,
        joint_mapping={},  # Empty for hypergraph format
        edge_mapping=edge_mapping,  # Used for linkage.edges format
    )


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

        # Get edge mapping from dimension_spec or infer it
        edge_mapping: dict[str, tuple[str, str]] = {}
        if dimension_spec is not None and dimension_spec.edge_mapping:
            edge_mapping = dimension_spec.edge_mapping
        else:
            # Infer edge mapping: "edge_name_property" -> ("edge_name", "property")
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
            mapping = _infer_mapping_from_names(list(dimension_values.keys()))

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
    dim_names: list[str] | tuple[str, ...],
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


# =============================================================================
# DEPRECATED: Sampling functions have moved to target_gen.sampling
# These are kept for backward compatibility but will be removed in a future version
# =============================================================================

def get_combinatoric_gradations(
    names: list[str],
    bounds: list[tuple[float, float]],
    n: int,
) -> dict[str, list[float]]:
    """
    DEPRECATED: Use target_gen.sampling.get_combinatoric_gradations instead.
    
    This function has been moved to the target_gen.sampling module for better organization.
    """
    from target_gen.sampling import get_combinatoric_gradations as _new_func
    logger.warning(
        'get_combinatoric_gradations has moved to target_gen.sampling. '
        'Please update your imports: from target_gen.sampling import get_combinatoric_gradations'
    )
    return _new_func(names, bounds, n)


def get_mech_variations(
    gradations: dict[str, list[float]],
) -> list[dict[str, float]]:
    """
    DEPRECATED: Use target_gen.sampling.get_mech_variations instead.
    
    This function has been moved to the target_gen.sampling module for better organization.
    """
    from target_gen.sampling import get_mech_variations as _new_func
    logger.warning(
        'get_mech_variations has moved to target_gen.sampling. '
        'Please update your imports: from target_gen.sampling import get_mech_variations'
    )
    return _new_func(gradations)


def get_mech_variations_from_spec(
    spec: DimensionSpec,
    n: int,
    mode: str = 'full_combinatoric',
    center: int | None = None,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """
    DEPRECATED: Use target_gen.sampling.get_mech_variations_from_spec instead.
    
    This function has been moved to the target_gen.sampling module for better organization.
    """
    from target_gen.sampling import get_mech_variations_from_spec as _new_func
    logger.warning(
        'get_mech_variations_from_spec has moved to target_gen.sampling. '
        'Please update your imports: from target_gen.sampling import get_mech_variations_from_spec'
    )
    return _new_func(spec, n, mode, center, seed)


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
    DEPRECATED: Use target_gen.sampling.presample_valid_positions instead.
    
    This function has been moved to the target_gen.sampling module for better organization.
    """
    from target_gen.sampling import presample_valid_positions as _new_func
    logger.warning(
        'presample_valid_positions has moved to target_gen.sampling. '
        'Please update your imports: from target_gen.sampling import presample_valid_positions'
    )
    return _new_func(pylink_data, target, dimension_spec, n_samples, n_best, mode, metric, seed, phase_invariant)
