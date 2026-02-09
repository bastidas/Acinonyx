"""
Optimizer implementations for linkage trajectory optimization.

Each optimizer follows a consistent interface:
- Takes Mechanism object, target trajectory, dimension spec
- Returns OptimizationResult
- All optimizers use Mechanism-based fast path (no pylink_data)

Available optimizers:
- scipy: Scipy-based optimization (L-BFGS-B, Powell, Nelder-Mead)
- pso: Standalone Particle Swarm Optimization
- pylinkage: Pylinkage-native PSO with kinematic validation
- nlopt: Multi-Level Single-Linkage with L-BFGS local search (NLopt)
- nlopt_gf: Gradient-free variant using BOBYQA (NLopt)
- scip: Mixed-integer nonlinear programming (PySCIPOpt)
"""
from __future__ import annotations

from optimizers.mlsl_optimizer import NLoptMLSLConfig
from optimizers.mlsl_optimizer import run_nlopt_mlsl
from optimizers.mlsl_optimizer import run_nlopt_mlsl_gf
from optimizers.pylinkage_pso import PylinkagePSOConfig
from optimizers.pylinkage_pso import run_pylinkage_pso
from optimizers.scip_optimizer import run_scip_optimization
from optimizers.scip_optimizer import SCIPConfig
from optimizers.scipy_optimizer import run_scipy_optimization
from optimizers.scipy_optimizer import ScipyConfig
# from optimizers.pso_optimizer import PSOConfig
# from optimizers.pso_optimizer import run_pso_optimization
# NLopt optimizers
# PSO optimizers
# Pylinkage PSO
# SCIP optimizer
# Scipy optimizer

# Registry of available optimizers for optimize_trajectory
AVAILABLE_OPTIMIZERS = {
    'scipy': {
        'function': run_scipy_optimization,
        'config_class': ScipyConfig,
        'description': 'Scipy-based optimization (L-BFGS-B, Powell, Nelder-Mead)',
        'package': 'scipy',
        'gradient': True,
        'global': False,
    },
    'l-bfgs-b': {
        'function': run_scipy_optimization,
        'config_class': ScipyConfig,
        'description': 'L-BFGS-B gradient-based optimizer',
        'package': 'scipy',
        'gradient': True,
        'global': False,
    },
    'powell': {
        'function': run_scipy_optimization,
        'config_class': ScipyConfig,
        'description': 'Powell gradient-free optimizer',
        'package': 'scipy',
        'gradient': False,
        'global': False,
    },
    'nelder-mead': {
        'function': run_scipy_optimization,
        'config_class': ScipyConfig,
        'description': 'Nelder-Mead simplex optimizer',
        'package': 'scipy',
        'gradient': False,
        'global': False,
    },
    # 'pso': {
    #     'function': run_pso_optimization,
    #     'config_class': PSOConfig,
    #     'description': 'Standalone Particle Swarm Optimization',
    #     'package': None,  # Built-in
    #     'gradient': False,
    #     'global': True,
    # },
    'pylinkage': {
        'function': run_pylinkage_pso,
        'config_class': PylinkagePSOConfig,
        'description': 'Pylinkage-native PSO with kinematic validation',
        'package': 'pylinkage',
        'gradient': False,
        'global': True,
    },
    'nlopt': {
        'function': run_nlopt_mlsl,
        'config_class': NLoptMLSLConfig,
        'description': 'Multi-Level Single-Linkage with L-BFGS local search',
        'package': 'nlopt',
        'gradient': True,
        'global': True,
    },
    'nlopt_gf': {
        'function': run_nlopt_mlsl_gf,
        'config_class': NLoptMLSLConfig,
        'description': 'Multi-Level Single-Linkage with BOBYQA (gradient-free)',
        'package': 'nlopt',
        'gradient': False,
        'global': True,
    },
    'scip': {
        'function': run_scip_optimization,
        'config_class': SCIPConfig,
        'description': 'SCIP mixed-integer nonlinear programming solver',
        'package': 'pyscipopt',
        'gradient': False,
        'global': True,
    },
}


__all__ = [
    # Scipy
    'run_scipy_optimization',
    'ScipyConfig',
    # PSO
    'run_pso_optimization',
    'PSOConfig',
    # Pylinkage PSO
    'run_pylinkage_pso',
    'PylinkagePSOConfig',
    # NLopt
    'run_nlopt_mlsl',
    'run_nlopt_mlsl_gf',
    'NLoptMLSLConfig',
    # SCIP
    'run_scip_optimization',
    'SCIPConfig',
    # Registry
    'AVAILABLE_OPTIMIZERS',
]
