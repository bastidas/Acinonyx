"""
Logging configuration for the automata project.

Usage:
    from configs.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("This will log to both console and backend.log")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from configs.paths import BASE_DIR

# Log file path
LOG_FILE = BASE_DIR / 'backend.log'

# Module-level flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    console: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: backend.log in project root)
        console: Whether to also log to console (default: True)
    """
    global _logging_configured

    if _logging_configured:
        return

    if log_file is None:
        log_file = LOG_FILE

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Get root logger for our project
    root_logger = logging.getLogger('pylink_tools')
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Automatically sets up logging if not already configured.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Debug details: %s", details)
        >>> logger.warning("Something unexpected")
        >>> logger.error("Operation failed: %s", error)
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()

    # Return a child logger under pylink_tools namespace
    if name.startswith('pylink_tools'):
        return logging.getLogger(name)
    else:
        return logging.getLogger(f'pylink_tools.{name}')


def log_separator(logger: logging.Logger, title: str = '') -> None:
    """Log a visual separator line for readability."""
    if title:
        logger.info('=' * 20 + f' {title} ' + '=' * 20)
    else:
        logger.info('=' * 50)
