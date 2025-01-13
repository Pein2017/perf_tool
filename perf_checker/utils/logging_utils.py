"""Logging utilities for the performance analysis tool."""

import logging
import os
from datetime import datetime
from typing import Optional

# Global timestamp for the current run
_CURRENT_TIMESTAMP: Optional[str] = None


def get_timestamp() -> str:
    """Get the current timestamp or create a new one if not exists."""
    global _CURRENT_TIMESTAMP
    if _CURRENT_TIMESTAMP is None:
        _CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CURRENT_TIMESTAMP


def create_results_structure(base_dir: str = "results") -> dict[str, str]:
    """Create a structured directory for all results.

    Args:
        base_dir: Base directory for results

    Returns:
        Dictionary with paths for different result types
    """
    timestamp = get_timestamp()
    timestamp_dir = os.path.join(base_dir, timestamp)
    paths = {
        "base": base_dir,
        "timestamp": timestamp_dir,
        "time": os.path.join(timestamp_dir, "time"),
        "dump": os.path.join(timestamp_dir, "dump"),
        "logs": timestamp_dir,
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def setup_base_logger(log_dir: str = "results", level: int = logging.DEBUG) -> None:
    """Set up the base logger configuration that all other loggers will inherit.

    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    root_logger = logging.getLogger("perf_monitor")
    root_logger.setLevel(level)

    # Reset existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Create formatter with a detailed format
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Set up file handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "perf_monitor.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Set up console handler with a simpler format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def setup_logger(name: str) -> logging.Logger:
    """Get a logger that inherits from the base configuration.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return logging.getLogger(
        f"perf_monitor.{name}" if not name.startswith("perf_monitor.") else name
    )
