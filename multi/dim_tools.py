"""
Dimensionality reduction utilities for high-dimensional data.

This module provides modular functions for reducing high-dimensional samples
to lower dimensions. Designed for reuse across the codebase for:
- Visualization (2D plotting)
- Clustering (preprocessing before clustering algorithms)
- Analysis (exploring high-dimensional spaces)

Example usage for visualization:
    >>> from multi.dim_tools import reduce_dimensions
    >>> reduced, reducer, metadata = reduce_dimensions(
    ...     samples=high_dim_samples,
    ...     method='pca',
    ...     n_components=2,
    ...     normalize=True,
    ... )
    >>> # Use reduced[:, 0] and reduced[:, 1] for 2D plotting

Example usage for clustering:
    >>> from multi.dim_tools import reduce_dimensions
    >>> reduced, reducer, metadata = reduce_dimensions(
    ...     samples=high_dim_samples,
    ...     method='pca',
    ...     n_components=10,  # Keep more dimensions for clustering
    ...     normalize=True,
    ... )
    >>> # Use reduced samples for clustering algorithms
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def normalize_samples(
    samples: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Normalize samples to [0, 1] range.
    
    Args:
        samples: Samples to normalize (n_samples, n_dims)
        bounds: Optional list of (min, max) tuples for each dimension.
                If None, uses min/max from samples.
    
    Returns:
        tuple: (normalized_samples, normalization_params)
        - normalized_samples: Normalized samples in [0, 1]
        - normalization_params: Dict with 'min' and 'max' arrays for inverse transform
    """
    samples = np.asarray(samples)
    n_samples, n_dims = samples.shape
    
    if bounds is not None:
        if len(bounds) != n_dims:
            raise ValueError(f"bounds length ({len(bounds)}) must match number of dimensions ({n_dims})")
        min_vals = np.array([b[0] for b in bounds])
        max_vals = np.array([b[1] for b in bounds])
    else:
        min_vals = np.min(samples, axis=0)
        max_vals = np.max(samples, axis=0)
    
    # Handle constant dimensions (max == min)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Avoid division by zero
    
    normalized = (samples - min_vals) / range_vals
    
    normalization_params = {
        'min': min_vals,
        'max': max_vals,
        'range': range_vals,
    }
    
    return normalized, normalization_params


def reduce_dimensions(
    samples: np.ndarray,
    method: str = 'pca',
    n_components: int = 2,
    fit_samples_mask: np.ndarray | None = None,
    transform_samples_mask: np.ndarray | None = None,
    normalize: bool = True,
    dimension_bounds: list[tuple[float, float]] | None = None,
    random_state: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, object, dict]:
    """
    Reduce high-dimensional samples to lower dimensions.
    
    Modular utility for dimensionality reduction that can be used for:
    - Visualization (2D plotting)
    - Clustering (preprocessing before clustering algorithms)
    - Analysis (exploring high-dimensional spaces)
    
    Args:
        samples: High-dimensional samples (n_samples, n_dims)
        method: Reduction method ('pca', 'tsne', 'umap')
        n_components: Number of output dimensions (default: 2 for visualization)
        fit_samples_mask: Boolean mask for which samples to use for fitting the reducer.
                         Options:
                         - None: Use all samples
                         - result.is_valid: Use only valid samples
                         - result.is_valid & (result.scores <= epsilon): Use only good samples
        transform_samples_mask: Which samples to transform (default: all samples)
        normalize: Whether to normalize samples before reduction
        dimension_bounds: List of (min, max) tuples for each dimension (for normalization).
                         If None and normalize=True, uses min/max from samples.
        random_state: Random seed for reproducibility
        **kwargs: Method-specific parameters (e.g., perplexity for t-SNE, n_neighbors for UMAP)
    
    Returns:
        tuple: (reduced_samples, reducer_object, metadata_dict)
        - reduced_samples: (n_samples, n_components) array
        - reducer_object: Fitted reducer (for inverse transform if needed, PCA only)
        - metadata: Dict with:
            - 'variance_explained': Array of explained variance ratios (for PCA)
            - 'method': Method used
            - 'n_components': Number of output dimensions
            - 'normalized': Whether normalization was applied
            - 'normalization_params': Dict with min/max values if normalized
    """
    samples = np.asarray(samples)
    n_samples, n_dims = samples.shape
    
    if n_components >= n_dims:
        raise ValueError(f"n_components ({n_components}) must be less than n_dims ({n_dims})")
    
    # Normalize if requested
    normalized_samples = samples
    normalization_params = None
    if normalize:
        normalized_samples, normalization_params = normalize_samples(samples, bounds=dimension_bounds)
    
    # Determine which samples to use for fitting
    if fit_samples_mask is not None:
        fit_samples_mask = np.asarray(fit_samples_mask, dtype=bool)
        if len(fit_samples_mask) != n_samples:
            raise ValueError(f"fit_samples_mask length ({len(fit_samples_mask)}) must match n_samples ({n_samples})")
        fit_samples = normalized_samples[fit_samples_mask]
        if len(fit_samples) == 0:
            logger.warning("No samples selected for fitting, using all samples")
            fit_samples = normalized_samples
    else:
        fit_samples = normalized_samples
    
    # Determine which samples to transform
    if transform_samples_mask is not None:
        transform_samples_mask = np.asarray(transform_samples_mask, dtype=bool)
        if len(transform_samples_mask) != n_samples:
            raise ValueError(f"transform_samples_mask length ({len(transform_samples_mask)}) must match n_samples ({n_samples})")
        transform_samples = normalized_samples[transform_samples_mask]
    else:
        transform_samples = normalized_samples
        transform_samples_mask = None
    
    # Apply dimensionality reduction
    reducer = None
    metadata: dict[str, Any] = {
        'method': method,
        'n_components': n_components,
        'normalized': normalize,
        'normalization_params': normalization_params,
    }
    
    if method == 'pca':
        try:
            from sklearn.decomposition import PCA
            
            # Extract PCA-specific kwargs
            svd_solver = kwargs.pop('svd_solver', 'auto')
            
            reducer = PCA(n_components=n_components, random_state=random_state, svd_solver=svd_solver)
            reducer.fit(fit_samples)
            
            # Transform all samples (or subset if mask provided)
            if transform_samples_mask is not None:
                # Transform only selected samples, then place in full array
                reduced_subset = reducer.transform(transform_samples)
                reduced_samples = np.full((n_samples, n_components), np.nan)
                reduced_samples[transform_samples_mask] = reduced_subset
            else:
                reduced_samples = reducer.transform(transform_samples)
            
            metadata['variance_explained'] = reducer.explained_variance_ratio_
            logger.info(
                f"PCA reduction: {n_dims}D -> {n_components}D, "
                f"variance explained: {reducer.explained_variance_ratio_[:n_components].sum():.1%} "
                f"(PC1: {reducer.explained_variance_ratio_[0]:.1%}, "
                f"PC2: {reducer.explained_variance_ratio_[1]:.1%})"
            )
            
        except ImportError:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
    
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            
            # Extract t-SNE-specific kwargs
            perplexity = kwargs.pop('perplexity', 30)
            learning_rate = kwargs.pop('learning_rate', 'auto')
            n_iter = kwargs.pop('n_iter', 1000)
            
            # t-SNE doesn't support separate fit/transform - it only does fit_transform
            # So we need to fit_transform on all samples we want to transform
            if transform_samples_mask is not None:
                # We need to transform only selected samples
                # Since t-SNE doesn't support transform, we fit_transform on transform_samples
                # but use fit_samples for fitting if they're different
                if len(fit_samples) == len(transform_samples) and np.array_equal(fit_samples, transform_samples):
                    samples_to_fit_transform = transform_samples
                else:
                    # Fit on fit_samples, but transform all samples we need
                    # Since t-SNE doesn't support this, we fit_transform on all unique samples
                    logger.warning(
                        "t-SNE doesn't support separate fit/transform. "
                        "Fitting and transforming on all samples that need transformation."
                    )
                    samples_to_fit_transform = transform_samples
            else:
                # Transform all samples - use fit_samples if they're the same, otherwise all samples
                if len(fit_samples) == len(normalized_samples) and np.array_equal(fit_samples, normalized_samples):
                    samples_to_fit_transform = normalized_samples
                else:
                    # Fit on fit_samples, but we need to transform all samples
                    # Since t-SNE doesn't support this, we fit_transform on all samples
                    logger.warning(
                        "t-SNE doesn't support separate fit/transform. "
                        "Fitting and transforming on all samples."
                    )
                    samples_to_fit_transform = normalized_samples
            
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(samples_to_fit_transform) - 1),  # Perplexity must be < n_samples
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                **kwargs,
            )
            
            reduced_fit_transform = reducer.fit_transform(samples_to_fit_transform)
            
            # Place results in full array
            if transform_samples_mask is not None:
                reduced_samples = np.full((n_samples, n_components), np.nan)
                reduced_samples[transform_samples_mask] = reduced_fit_transform
            else:
                reduced_samples = reduced_fit_transform
            
            metadata['variance_explained'] = None  # t-SNE doesn't provide variance explained
            logger.info(f"t-SNE reduction: {n_dims}D -> {n_components}D")
            
        except ImportError:
            raise ImportError("sklearn is required for t-SNE. Install with: pip install scikit-learn")
    
    elif method == 'umap':
        try:
            import umap
            
            # Extract UMAP-specific kwargs
            n_neighbors = kwargs.pop('n_neighbors', 15)
            min_dist = kwargs.pop('min_dist', 0.1)
            metric = kwargs.pop('metric', 'euclidean')
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, len(fit_samples) - 1),  # n_neighbors must be < n_samples
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                **kwargs,
            )
            
            # UMAP supports fit/transform
            reducer.fit(fit_samples)
            
            if transform_samples_mask is not None:
                reduced_subset = reducer.transform(transform_samples)
                reduced_samples = np.full((n_samples, n_components), np.nan)
                reduced_samples[transform_samples_mask] = reduced_subset
            else:
                reduced_samples = reducer.transform(transform_samples)
            
            metadata['variance_explained'] = None  # UMAP doesn't provide variance explained
            logger.info(f"UMAP reduction: {n_dims}D -> {n_components}D")
            
        except ImportError:
            logger.warning("umap-learn not available, falling back to PCA")
            # Fall back to PCA
            return reduce_dimensions(
                samples=samples,
                method='pca',
                n_components=n_components,
                fit_samples_mask=fit_samples_mask,
                transform_samples_mask=transform_samples_mask,
                normalize=normalize,
                dimension_bounds=dimension_bounds,
                random_state=random_state,
                **kwargs,
            )
    
    else:
        raise ValueError(f"Unknown reduction method: {method}. Supported methods: 'pca', 'tsne', 'umap'")
    
    return reduced_samples, reducer, metadata
