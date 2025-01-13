"""Decorators for function monitoring with thread-safe global state management."""

from functools import wraps
from threading import Lock
from typing import Callable, Optional, ParamSpec, TypeVar, cast

from ..manager.collector_manager import CollectorManager

# Type variables for better type hints
P = ParamSpec("P")
T = TypeVar("T")


class _GlobalManager:
    """Thread-safe singleton for managing the default collector."""

    def __init__(self):
        self._lock = Lock()
        self._instance: Optional[CollectorManager] = None

    def set(self, manager: CollectorManager) -> None:
        with self._lock:
            self._instance = manager

    def get(self) -> CollectorManager:
        with self._lock:
            if self._instance is None:
                raise RuntimeError(
                    "No default manager has been set. Call set_default_manager() "
                    "with a CollectorManager instance before using @monitor."
                )
            return self._instance


_global = _GlobalManager()


def set_default_manager(manager: CollectorManager) -> None:
    """Set the default manager instance to be used by all @monitor decorators.

    Args:
        manager: The CollectorManager instance to use as default
    """
    _global.set(manager)


def monitor(
    stage: str,
    tag: str,
    *,  # Force keyword arguments
    enable_time: bool = True,
    enable_precision: bool = False,
    manager: Optional[CollectorManager] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to monitor function execution with improved type safety.

    Uses the manager provided or falls back to the default manager set via set_default_manager().
    All monitoring configuration is done through keyword arguments for clarity.

    Args:
        stage: The monitoring stage name (e.g., "inference", "preprocessing")
        tag: The monitoring tag (will be suffixed for multiple calls)
        enable_time: Whether to enable time monitoring (default: True)
        enable_precision: Whether to enable precision monitoring (default: False)
        manager: Optional collector manager instance. If None, uses default manager.

    Returns:
        Decorated function with monitoring enabled

    Raises:
        RuntimeError: If no manager is provided and no default manager has been set

    Example:
        @monitor("inference", "decode", enable_time=True)
        def decode_prediction(logits: torch.Tensor) -> torch.Tensor:
            return process_logits(logits)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        active_manager = manager if manager is not None else _global.get()
        active_manager.register(
            stage=stage,
            tag=tag,
            enable_time=enable_time,
            enable_precision=enable_precision,
        )

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return cast(T, active_manager.monitor(stage, tag)(func)(*args, **kwargs))

        return wrapper

    return decorator
