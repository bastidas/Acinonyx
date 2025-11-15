import os
from pathlib import Path

# Define the user directory relative to this file's location
user_dir = Path(__file__).parent.parent / "user"

# Ensure the user directory exists
user_dir.mkdir(exist_ok=True)