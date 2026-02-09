"""
Pytest configuration - runs before test collection.

Adds project root to sys.path so local modules can be imported.
Configures logging for test output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path for local module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
# Default to INFO level - use pytest -s --log-cli-level=DEBUG for more verbose output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S',
)

# Set specific loggers to appropriate levels
# Set target_gen to DEBUG to see detailed sampling information
logging.getLogger('target_gen').setLevel(logging.INFO)
logging.getLogger('pylink_tools').setLevel(logging.WARNING)  # Reduce noise from pylink_tools
