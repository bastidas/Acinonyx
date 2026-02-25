# Application Configuration
# Centralized configuration for all ports and endpoints.
# When changing BACKEND_PORT or FRONTEND_PORT, also update configs/appconfig.js
# so the Vite dev server proxy targets the correct backend port.
from __future__ import annotations

import os
from pathlib import Path


class AppConfig:
    """Centralized application configuration"""

    # Project root: directory containing configs/ (e.g. .../Acinonyx)
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    USER_DIR: Path = _PROJECT_ROOT / 'user'
    # Graphs (working) directory for save/load .json files.
    # Default: <project_root>/user/graphs (e.g. /Users/.../Acinonyx/user/graphs).
    # Override via environment variable ACINONYX_GRAPHS_DIR (absolute or relative path).
    _GRAPHS_DIR_OVERRIDE = os.environ.get('ACINONYX_GRAPHS_DIR')
    GRAPHS_DIR: Path = (
        Path(_GRAPHS_DIR_OVERRIDE).resolve() if _GRAPHS_DIR_OVERRIDE
        else (USER_DIR / 'graphs').resolve()
    )

    # Port Configuration
    FRONTEND_PORT = 5173
    BACKEND_PORT = 8022

    # URLs (derived from ports)
    FRONTEND_URL = f'http://localhost:{FRONTEND_PORT}'
    BACKEND_URL = f'http://localhost:{BACKEND_PORT}'

    # API Configuration
    API_PREFIX = '/api'

    @classmethod
    def get_frontend_url(cls):
        return cls.FRONTEND_URL

    @classmethod
    def get_backend_url(cls):
        return cls.BACKEND_URL

    @classmethod
    def get_api_base_url(cls):
        return f'{cls.BACKEND_URL}{cls.API_PREFIX}'


# For backward compatibility and easy imports
FRONTEND_PORT = AppConfig.FRONTEND_PORT
BACKEND_PORT = AppConfig.BACKEND_PORT
FRONTEND_URL = AppConfig.FRONTEND_URL
BACKEND_URL = AppConfig.BACKEND_URL
USER_DIR = AppConfig.USER_DIR
GRAPHS_DIR = AppConfig.GRAPHS_DIR
