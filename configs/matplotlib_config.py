"""
matplotlib_config.py - Configure matplotlib for backend/headless use.
"""
from __future__ import annotations

import os

import matplotlib


def configure_matplotlib_for_backend():
    """
    Configure matplotlib to use a non-interactive backend.

    This should be called BEFORE importing matplotlib.pyplot
    to avoid issues in headless environments.
    """
    # Check if we're in a headless environment
    if os.environ.get('DISPLAY') is None and os.name != 'nt':
        # Use Agg backend for non-interactive rendering
        matplotlib.use('Agg')

    # Set some sensible defaults
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 150
    matplotlib.rcParams['figure.figsize'] = [10, 8]
    matplotlib.rcParams['font.size'] = 10


def get_safe_backend():
    """Return a safe backend for the current environment."""
    try:
        import matplotlib.pyplot as plt
        return matplotlib.get_backend()
    except:
        return 'Agg'
