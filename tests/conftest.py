"""
Pytest configuration - runs before test collection.

Adds project root to sys.path so local modules can be imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for local module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
