"""Performance monitoring tools for ML inference pipelines."""

from .core.decorators.monitor import monitor, set_default_manager
from .core.manager.collector_manager import CollectorManager
from .utils.logging_utils import (
    setup_base_logger,
)

__version__ = "1.0.0"
__all__ = [
    "CollectorManager",
    "monitor",
    "set_default_manager",
    "setup_base_logger",
]
