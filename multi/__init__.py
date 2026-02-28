"""
Multi-dimensional analysis utilities.

This package provides tools for working with high-dimensional data,
including dimensionality reduction for visualization and clustering.
"""
from __future__ import annotations

from multi.dim_tools import normalize_samples
from multi.dim_tools import reduce_dimensions

__all__ = ['reduce_dimensions', 'normalize_samples']
