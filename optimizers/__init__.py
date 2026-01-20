"""
Optimizer implementations for linkage trajectory optimization.

Each optimizer follows a consistent interface:
- Takes pylink_data, target trajectory, dimension spec
- Returns OptimizationResult

Available optimizers:
- nlopt_mlsl: Multi-Level Single-Linkage with L-BFGS local search (NLopt)
- nlopt_mlsl_gf: Gradient-free variant using BOBYQA (NLopt)
- scip: Mixed-integer nonlinear programming (PySCIPOpt)
"""
from __future__ import annotations

from optimizers.nlopt_mlsl import NLoptMLSLConfig
from optimizers.nlopt_mlsl import run_nlopt_mlsl
from optimizers.nlopt_mlsl import run_nlopt_mlsl_gf
from optimizers.scip_optimizer import run_scip_optimization
from optimizers.scip_optimizer import SCIPConfig

# Registry of available optimizers for optimize_trajectory
AVAILABLE_OPTIMIZERS = {
    'nlopt': {
        'function': run_nlopt_mlsl,
        'description': 'Multi-Level Single-Linkage with L-BFGS local search',
        'package': 'nlopt',
        'gradient': True,
        'global': True,
    },
    'nlopt_gf': {
        'function': run_nlopt_mlsl_gf,
        'description': 'Multi-Level Single-Linkage with BOBYQA (gradient-free)',
        'package': 'nlopt',
        'gradient': False,
        'global': True,
    },
    'scip': {
        'function': run_scip_optimization,
        'description': 'SCIP mixed-integer nonlinear programming solver',
        'package': 'pyscipopt',
        'gradient': False,
        'global': True,
    },
}

__all__ = [
    'run_nlopt_mlsl',
    'run_nlopt_mlsl_gf',
    'NLoptMLSLConfig',
    'run_scip_optimization',
    'SCIPConfig',
    'AVAILABLE_OPTIMIZERS',
]
