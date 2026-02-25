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


def _reset_logger(logger: logging.Logger) -> None:
    """Detach and close all handlers on a logger."""
    if not isinstance(logger, logging.Logger):
        return

    for handler in list(logger.handlers):
        try:
            handler.flush()
        except Exception:
            pass
        handler.close()
        logger.removeHandler(handler)


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

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Clear handlers from root and any previously instantiated loggers so we own output destinations
    root_logger = logging.getLogger()
    _reset_logger(root_logger)
    root_logger.setLevel(level)

    for logger_obj in list(logging.root.manager.loggerDict.values()):
        if isinstance(logger_obj, logging.PlaceHolder):  # Skip placeholders created by logging module
            continue
        _reset_logger(logger_obj)
        logger_obj.propagate = True

    # File handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
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

    return logging.getLogger(name)


def log_separator(logger: logging.Logger, title: str = '') -> None:
    """Log a visual separator line for readability."""
    if title:
        logger.info('=' * 20 + f' {title} ' + '=' * 20)
    else:
        logger.info('=' * 50)
