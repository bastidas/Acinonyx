#!/usr/bin/env python3
"""
Acinonyx Backend Server
Uses centralized port configuration from configs.appconfig
"""
from __future__ import annotations

import os
import sys

import uvicorn

from configs.appconfig import BACKEND_PORT

# Add parent directory to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    import logging
    import os

    # Set log level from environment variable, default to DEBUG
    log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    uvicorn_log_level = log_level.lower() if log_level in log_level_map else 'info'

    print(f'🐆 Starting Acinonyx Backend Server on port {BACKEND_PORT}...')
    print(f'📊 Log level: {log_level} (set LOG_LEVEL env var to change)')
    uvicorn.run(
        'acinonyx_api:app',
        host='0.0.0.0',
        port=BACKEND_PORT,
        reload=True,
        log_level=uvicorn_log_level,
    )
