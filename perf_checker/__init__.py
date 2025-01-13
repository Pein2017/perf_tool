"""Performance monitoring tools for Deep Learning inference pipelines."""

__version__ = "0.1.0"

from .core.decorators.monitor import monitor, set_default_manager
from .core.manager.collector_manager import CollectorManager
from .initialization import initialize_monitoring
from .utils import setup_base_logger

__all__ = [
    "monitor",
    "CollectorManager",
    "set_default_manager",
    "setup_base_logger",
    "initialize_monitoring",
]
