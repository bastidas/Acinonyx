"""
Main entry point for trajectory optimization.

Provides a unified interface to all optimization methods using Mechanism objects.
All optimizers work with Mechanism instances directly (no pylink_data).
"""
from __future__ import annotations

import logging
from typing import Literal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism
    from pylink_tools.optimization_types import (
        DimensionBoundsSpec,
        OptimizationResult,
        TargetTrajectory,
    )
    from target_gen.variation_config import MechVariationConfig

logger = logging.getLogger(__name__)


def _bounds_spec_compatible(
    spec: DimensionBoundsSpec,
    mechanism_spec: DimensionBoundsSpec,
) -> bool:
    """
    Check if bounds spec is compatible with mechanism.

    Spec can have fewer dimensions (filtered) but all must exist in mechanism.
    """
    mechanism_names = set(mechanism_spec.names)
    spec_names = set(spec.names)

    # All spec dimensions must exist in mechanism
    if not spec_names.issubset(mechanism_names):
        return False

    # Order must match mechanism's order (for indexing)
    mechanism_order = {name: i for i, name in enumerate(mechanism_spec.names)}

    # Check that spec maintains relative order
    spec_indices = [mechanism_order[name] for name in spec.names]
    return spec_indices == sorted(spec_indices)


def optimize_trajectory(
    mechanism: Mechanism,
    target: TargetTrajectory,
    dimension_bounds_spec: DimensionBoundsSpec | None = None,
    mech_variation_config: MechVariationConfig | None = None,  # NEW
    method: str = 'pylinkage',
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    **kwargs,
) -> OptimizationResult:
    """
    Main entry point for trajectory optimization.

    Finds link lengths that make the mechanism's trajectory best match
    the target positions. All optimizers use Mechanism objects directly.

    Args:
        mechanism: Mechanism to optimize (will be modified in place)
        target: Target trajectory to match (joint name + positions)
        dimension_bounds_spec: Pre-computed dimension bounds spec (extracted from mechanism if None)
        mech_variation_config: Optional variation config to derive bounds from
        method: Optimization method:
                - "pylinkage": Particle Swarm Optimization (default, robust)
                - "pso": Standalone PSO implementation
                - "scipy": scipy.optimize.minimize with L-BFGS-B (faster)
                - "l-bfgs-b": Alias for scipy with L-BFGS-B
                - "powell": scipy Powell method (gradient-free)
                - "nelder-mead": Nelder-Mead simplex (gradient-free)
                - "nlopt" / "nlopt_mlsl" / "mlsl": NLopt MLSL with L-BFGS (global)
                - "nlopt_gf": NLopt MLSL with BOBYQA (gradient-free)
                - "scip": SCIP mixed-integer solver (for discrete params)
        metric: Error metric ("mse", "rmse", "total", "max")
        verbose: Print progress
        phase_invariant: Use phase-aligned scoring (recommended for external targets)
        phase_align_method: Phase alignment algorithm ('rotation'|'fft'|'frechet')
        **kwargs: Method-specific arguments:
                  PSO: n_particles (32), iterations (512),
                       init_mode ('random'|'sobol'|'behnken'), init_samples (128)
                  scipy: max_iterations (100), tolerance (1e-6), method (override)
                  nlopt: max_eval (1000), local_max_eval (100), ftol_rel (1e-6)
                  scip: time_limit (300), gap_limit (0.01)

    Returns:
        OptimizationResult with optimized dimensions and updated mechanism

    Example:
        >>> from pylink_tools.mechanism import Mechanism
        >>> from pylink_tools.optimization_types import TargetTrajectory
        >>> mechanism = Mechanism(...)  # Create mechanism
        >>> target = TargetTrajectory(
        ...     joint_name="coupler",
        ...     positions=[(100, 120), (105, 125), ...]
        ... )
        >>> result = optimize_trajectory(mechanism, target, method="scipy")
        >>> if result.success:
        ...     print(f"Reduced error from {result.initial_error:.2f} "
        ...           f"to {result.final_error:.2f}")
        ...     optimized_mechanism = mechanism  # Mechanism updated in place
    """
    # Log MechVariationConfig when optimizer is instantiated
    if mech_variation_config:
        print('\n[optimize_trajectory] Received MechVariationConfig:')
        print(f'  Type: {type(mech_variation_config)}')
        print(f'  Has dimension_variation: {hasattr(mech_variation_config, "dimension_variation")}')
        if hasattr(mech_variation_config, 'dimension_variation'):
            dim_var = mech_variation_config.dimension_variation
            print(f'    dimension_variation type: {type(dim_var)}')
            print(f'    default_variation_range: {getattr(dim_var, "default_variation_range", "MISSING")}')
            print(f'    default_enabled: {getattr(dim_var, "default_enabled", "MISSING")}')
            print(f'    dimension_overrides: {len(getattr(dim_var, "dimension_overrides", {}))} overrides')
            print(f'    exclude_dimensions: {len(getattr(dim_var, "exclude_dimensions", []))} excluded')
        print(f'  Has static_joint_movement: {hasattr(mech_variation_config, "static_joint_movement")}')
        if hasattr(mech_variation_config, 'static_joint_movement'):
            static_joint = mech_variation_config.static_joint_movement
            print(f'    static_joint_movement type: {type(static_joint)}')
            print(f'    enabled: {getattr(static_joint, "enabled", "MISSING")}')
            print(f'    max_x_movement: {getattr(static_joint, "max_x_movement", "MISSING")}')
            print(f'    max_y_movement: {getattr(static_joint, "max_y_movement", "MISSING")}')
        print(f'  Has topology_changes: {hasattr(mech_variation_config, "topology_changes")}')
        print(f'  Has max_attempts: {hasattr(mech_variation_config, "max_attempts")}')
        print(f'  Has fallback_ranges: {hasattr(mech_variation_config, "fallback_ranges")}')
        print(f'  Has random_seed: {hasattr(mech_variation_config, "random_seed")}')
        # #region agent log
        import json
        import time
        try:
            from dataclasses import asdict
            config_dict = asdict(mech_variation_config) if hasattr(mech_variation_config, '__dataclass_fields__') else {}
        except Exception:
            config_dict = {}
        with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
            f.write(
                json.dumps({
                    'location': 'optimize.py:111', 'message': 'optimize_trajectory received MechVariationConfig', 'data': {
                        'hasConfig': mech_variation_config is not None, 'configType': type(mech_variation_config).__name__, 'configDict': config_dict,
                        'hasDimensionVariation': hasattr(mech_variation_config, 'dimension_variation') if mech_variation_config else False,
                    }, 'timestamp': int(time.time() * 1000), 'sessionId': 'debug-session', 'runId': 'run6', 'hypothesisId': 'C',
                }) + '\n',
            )
        # #endregion

    # Priority: dimension_bounds_spec > mech_variation_config > mechanism default
    if dimension_bounds_spec is None:
        if mech_variation_config is not None:
            from target_gen.variation_config import MechVariationConfig
            dimension_bounds_spec = DimensionBoundsSpec.from_mechanism(
                mechanism, mech_variation_config,
            )
        else:
            dimension_bounds_spec = mechanism.get_dimension_bounds_spec()

    # Validate bounds spec is compatible with mechanism
    # Spec can have fewer dimensions (filtered) - this is a performance optimization
    mechanism_spec = mechanism.get_dimension_bounds_spec()
    if not _bounds_spec_compatible(dimension_bounds_spec, mechanism_spec):
        raise ValueError(
            f"DimensionBoundsSpec dimensions don't match mechanism. "
            f'Spec has: {dimension_bounds_spec.names}, '
            f'mechanism has: {mechanism_spec.names}',
        )

    # IMPORTANT: If spec has fewer dimensions, optimizers must use static values
    # for excluded dimensions. This is a performance optimization - fewer
    # dimensions to optimize = faster optimization.
    # Optimizers should handle this by:
    # 1. Only optimizing dimensions in spec.names
    # 2. Using mechanism's current values for dimensions not in spec

    if verbose:
        logger.info('Optimizing trajectory fit')
        logger.info(f'  Target joint: {target.joint_name}')
        logger.info(f'  Target steps: {target.n_steps}')
        logger.info(f'  Dimensions: {len(dimension_bounds_spec)}')
        logger.info(f'  Method: {method}')

    # Route to appropriate optimizer
    method_lower = method.lower()

    # Scipy-based optimizers (L-BFGS-B, Powell, Nelder-Mead)
    if method_lower in ('scipy', 'l-bfgs-b', 'powell', 'nelder-mead'):
        from optimizers.scipy_optimizer import run_scipy_optimization, ScipyConfig

        # Determine scipy method
        if method_lower == 'l-bfgs-b':
            scipy_method = 'L-BFGS-B'
        elif method_lower == 'powell':
            scipy_method = 'Powell'
        elif method_lower == 'nelder-mead':
            scipy_method = 'Nelder-Mead'
        else:
            scipy_method = kwargs.get('method', 'L-BFGS-B')

        config = kwargs.get('config')
        if config is None:
            config = ScipyConfig(
                method=scipy_method,
                max_iterations=kwargs.get('max_iterations', 100),
                tolerance=kwargs.get('tolerance', 1e-6),
            )

        return run_scipy_optimization(
            mechanism=mechanism,
            target=target,
            dimension_bounds_spec=dimension_bounds_spec,
            config=config,
            metric=metric,
            verbose=verbose,
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    # NLopt MLSL optimizers
    elif method_lower in ('nlopt', 'nlopt_mlsl', 'mlsl', 'nlopt_gf'):
        from optimizers.mlsl_optimizer import NLoptMLSLConfig, run_nlopt_mlsl, run_nlopt_mlsl_gf

        # Use gradient-free variant if requested
        if method_lower == 'nlopt_gf':
            config = kwargs.get('config')
            if config is None:
                config = NLoptMLSLConfig(
                    max_eval=kwargs.get('max_eval', 1000),
                    local_max_eval=kwargs.get('local_max_eval', 100),
                    ftol_rel=kwargs.get('ftol_rel', 1e-6),
                    local_algorithm='bobyqa',  # Gradient-free
                )
            return run_nlopt_mlsl_gf(
                mechanism=mechanism,
                target=target,
                dimension_bounds_spec=dimension_bounds_spec,
                config=config,
                metric=metric,
                verbose=verbose,
                phase_invariant=phase_invariant,
                phase_align_method=phase_align_method,
            )
        else:
            config = kwargs.get('config')
            if config is None:
                config = NLoptMLSLConfig(
                    max_eval=kwargs.get('max_eval', 1000),
                    local_max_eval=kwargs.get('local_max_eval', 100),
                    ftol_rel=kwargs.get('ftol_rel', 1e-6),
                )

            return run_nlopt_mlsl(
                mechanism=mechanism,
                target=target,
                dimension_bounds_spec=dimension_bounds_spec,
                config=config,
                metric=metric,
                verbose=verbose,
                phase_invariant=phase_invariant,
                phase_align_method=phase_align_method,
            )

    # # PSO optimizers
    # elif method_lower == 'pso':
    #     from optimizers.pso_optimizer import run_pso_optimization, PSOConfig

    #     config = kwargs.get('config')
    #     if config is None:
    #         config = PSOConfig(
    #             n_particles=kwargs.get('n_particles', 32),
    #             iterations=kwargs.get('iterations', 512),
    #             init_mode=kwargs.get('init_mode', 'random'),
    #             init_samples=kwargs.get('init_samples', 128),
    #         )

    #     return run_pso_optimization(
    #         mechanism=mechanism,
    #         target=target,
    #         dimension_spec=dimension_spec,
    #         config=config,
    #         metric=metric,
    #         verbose=verbose,
    #         phase_invariant=phase_invariant,
    #         phase_align_method=phase_align_method,
    #     )

    elif method_lower == 'pylinkage':
        from optimizers.pylinkage_pso import run_pylinkage_pso, PylinkagePSOConfig

        config = kwargs.get('config')
        if config is None:
            config = PylinkagePSOConfig(
                n_particles=kwargs.get('n_particles', 32),
                iterations=kwargs.get('iterations', 512),
                init_mode=kwargs.get('init_mode', 'random'),
                init_samples=kwargs.get('init_samples', 128),
            )

        return run_pylinkage_pso(
            mechanism=mechanism,
            target=target,
            dimension_bounds_spec=dimension_bounds_spec,
            config=config,
            metric=metric,
            verbose=verbose,
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    # SCIP optimizer
    elif method_lower == 'scip':
        from optimizers.scip_optimizer import run_scip_optimization, SCIPConfig

        config = kwargs.get('config')
        if config is None:
            config = SCIPConfig(
                time_limit=kwargs.get('time_limit', 300.0),
                gap_limit=kwargs.get('gap_limit', 0.01),
            )

        return run_scip_optimization(
            mechanism=mechanism,
            target=target,
            dimension_bounds_spec=dimension_bounds_spec,
            config=config,
            metric=metric,
            verbose=verbose,
            phase_invariant=phase_invariant,
            phase_align_method=phase_align_method,
        )

    else:
        available_methods = [
            'pylinkage',
            'scipy',
            'l-bfgs-b',
            'powell',
            'nelder-mead',
            'nlopt',
            'nlopt_mlsl',
            'mlsl',
            'nlopt_gf',
            'scip',
        ]
        from pylink_tools.optimization_types import OptimizationResult
        return OptimizationResult(
            success=False,
            optimized_dimensions={},
            error=f'Unknown optimization method: {method}. Available: {", ".join(available_methods)}',
        )
