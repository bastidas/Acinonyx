"""
traj_dist - Pure Python implementation of trajectory distance metrics.

This module is based on the traj-dist package by Brendan Guillouet:
https://github.com/bguillouet/traj-dist

Original code is MIT licensed. This is a pure Python port using the pydist
implementations instead of the Cython (cydist) implementations.

Original repository: https://github.com/bguillouet/traj-dist
License: MIT
"""
from __future__ import annotations
__author__ = 'bguillouet'
from .distance import (
    pdist, cdist, sspd, sowd_grid, frechet, discret_frechet,
    hausdorff, dtw, lcss, edr, erp,
)

__all__ = [
    'pdist', 'cdist', 'sspd', 'sowd_grid', 'frechet', 'discret_frechet',
    'hausdorff', 'dtw', 'lcss', 'edr', 'erp',
]
