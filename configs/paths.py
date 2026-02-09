"""
Configuration file for directory paths
"""
from __future__ import annotations

from pathlib import Path

# Define the base project directory
BASE_DIR = Path(__file__).parent.parent

# Define the user directory where outputs will be saved
user_dir = BASE_DIR / 'user'

# Ensure the user directory exists
user_dir.mkdir(exist_ok=True)
