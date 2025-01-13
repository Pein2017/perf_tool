from typing import Any, Protocol


class CollectorManagerProtocol(Protocol):
    """Protocol defining the interface required by MonitoringContext."""

    def start(self, stage: str, tag: str) -> None: ...
    def end(self) -> None: ...


class MonitoringContext:
    """Context manager for monitoring code blocks."""

    def __init__(self, manager: CollectorManagerProtocol, stage: str, tag: str):
        """Initialize the monitoring context.

        Args:
            manager: The collector manager instance
            stage: The monitoring stage name
            tag: The monitoring tag
        """
        self.manager = manager
        self.stage = stage
        self.tag = tag

    def __enter__(self) -> None:
        """Start monitoring when entering the context."""
        self.manager.start(self.stage, self.tag)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End monitoring when exiting the context."""
        self.manager.end()
