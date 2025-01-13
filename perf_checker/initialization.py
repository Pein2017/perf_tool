"""Initialization utilities for the performance monitoring system."""

from typing import Dict, Tuple

from .core.decorators.monitor import set_default_manager
from .core.manager.collector_manager import CollectorManager
from .utils import create_results_structure, setup_base_logger


def initialize_monitoring(
    enable_time: bool = True,
    enable_precision: bool = True,
    precision_config: str = "default",
    result_dir: str = "perf_results",
    log_level: str = "DEBUG",
) -> Tuple[CollectorManager, Dict[str, str]]:
    """Initialize the monitoring system.

    This must be called before using any @monitor decorators.
    Creates a unified directory structure for all monitoring outputs (logs, dumps, timing data)
    under a timestamped directory.

    Args:
        enable_time: Whether to enable time monitoring
        enable_precision: Whether to enable precision monitoring
        precision_config: Path to precision monitoring config file or 'default' to use default config
        result_dir: Base directory for all results (logs, dumps, timing data)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        A tuple containing:
            - The initialized CollectorManager instance
            - Dictionary of paths for different result types:
                - base: Base directory
                - time: Directory for timing results
                - dump: Directory for precision monitoring dumps
                - logs: Directory for log files
    """
    # Create unified result directory structure
    paths = create_results_structure(result_dir)

    # Set up logging in the timestamp directory
    setup_base_logger(paths["logs"], level=log_level)

    # Initialize the manager
    manager = CollectorManager(
        enable_time=enable_time,
        enable_precision=enable_precision,
        precision_config=precision_config,
    )
    set_default_manager(manager)
    return manager, paths
