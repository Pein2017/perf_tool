"""Manager for coordinating multiple collectors in the performance analysis tool."""

from functools import wraps
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
)

from ...utils import setup_logger
from ..base.monitor_config import MonitorConfig
from ..collectors.precision_collector import PrecisionCollector
from ..collectors.time_collector import TimeCollector
from .context import MonitoringContext

T = TypeVar("T")  # Return type for the decorated function
P = ParamSpec("P")  # Parameter specification for the decorated function


class CollectorManager:
    """Unified manager for hierarchical time and precision monitoring."""

    def __init__(
        self,
        enable_time: bool = True,
        enable_precision: bool = False,
        precision_config: Optional[str] = None,
    ):
        """Initialize the collector manager.

        Args:
            enable_time: Whether to enable time monitoring
            enable_precision: Whether to enable precision monitoring
            precision_config: Path to precision monitoring config file
        """
        self.logger = setup_logger("manager")
        self.time_collector = TimeCollector() if enable_time else None
        self.precision_collector = None
        if enable_precision and precision_config:
            self.precision_collector = PrecisionCollector(precision_config)

        self.default_enable_time = enable_time
        self.default_enable_precision = enable_precision
        self.configs: Dict[str, Dict[str, MonitorConfig]] = {}
        self.current_run = 0  # Matches dataloader batch index

    def register(
        self,
        stage: str,
        tag: str,
        enable_time: Optional[bool] = None,
        enable_precision: Optional[bool] = None,
    ) -> None:
        """Register monitoring configuration for a function."""
        if stage not in self.configs:
            self.configs[stage] = {}

        self.configs[stage][tag] = MonitorConfig(
            stage=stage,
            tag=tag,
            enable_time=self.default_enable_time
            if enable_time is None
            else enable_time,
            enable_precision=self.default_enable_precision
            if enable_precision is None
            else enable_precision,
        )
        self.logger.debug(f"Registered config for {stage}/{tag}")

    def _ensure_registered(self, stage: str, tag: str) -> MonitorConfig:
        """Ensure operator is registered with default settings if not already registered."""
        if stage not in self.configs or tag not in self.configs[stage]:
            self.register(stage, tag)
        return self.configs[stage][tag]

    def monitor_context(self, stage: str, tag: str) -> MonitoringContext:
        """Create a context manager for monitoring a code block."""
        self._ensure_registered(stage, tag)
        return MonitoringContext(self, stage, tag)

    def set_run(self, batch_idx: int) -> None:
        """Set the current run ID to match the dataloader batch index.

        Each batch in the dataloader represents one run, and its index
        determines the run ID. The run ID is used to track operator calls
        within each batch independently.

        Args:
            batch_idx: The current batch index from the dataloader
        """
        self.current_run = batch_idx + 1

        # Set the new run ID in collectors
        if self.time_collector:
            self.time_collector.set_run(self.current_run)

        if self.precision_collector:
            self.precision_collector.set_run(self.current_run)

        self.logger.debug(f"Set run ID to {self.current_run}")

    def start(self, stage: str, tag: str) -> None:
        """Start monitoring a code block."""
        config = self._ensure_registered(stage, tag)
        self.logger.debug(
            f"Starting monitoring for {stage}/{tag}, "
            f"time enabled: {config.enable_time and self.time_collector is not None}, "
            f"precision enabled: {config.enable_precision and self.precision_collector is not None}"
        )

        if self.time_collector and config.enable_time:
            self.time_collector.start(stage, tag)
            self.logger.debug(
                f"Started timing collection for stage: {stage}, tag: {tag}"
            )
        if self.precision_collector and config.enable_precision:
            self.precision_collector.start(stage, tag)
            self.logger.debug(
                f"Started precision collection for stage: {stage}, tag: {tag}"
            )

    def end(self) -> None:
        """End monitoring the current code block."""
        self.logger.debug(
            f"Ending monitoring, time collector: {self.time_collector is not None}, "
            f"precision collector: {self.precision_collector is not None}"
        )

        # End time collection if active
        if self.time_collector:
            self.time_collector.end()

        if self.precision_collector:
            self.precision_collector.end()

    def get_time_stats(
        self, stage: Optional[str] = None, tag: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get timing statistics for specified stage and tag."""
        if not self.time_collector:
            return {}
        return self.time_collector.get_stats(stage, tag)

    def monitor(
        self, stage: str, tag: str
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator factory for monitoring functions.

        This method is used internally by the @monitor decorator to create
        function-specific monitoring decorators. It ensures the function
        is registered and creates a decorator that uses monitor_context.

        Args:
            stage: The monitoring stage name
            tag: The monitoring tag

        Returns:
            A decorator function that wraps the target function with monitoring
        """
        self._ensure_registered(stage, tag)

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with self.monitor_context(stage, tag):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_call_stack(self) -> List[Tuple[str, str]]:
        """Get the current call stack of monitored functions."""
        return (
            [] if not self.time_collector else self.time_collector.get_current_stack()
        )

    def get_monitored_functions(self) -> Set[Tuple[str, str]]:
        """Get the set of functions being monitored."""
        time_funcs = (
            set()
            if not self.time_collector
            else self.time_collector.get_monitored_functions()
        )
        precision_funcs = (
            set()
            if not self.precision_collector
            else self.precision_collector.get_monitored_functions()
        )
        return time_funcs.union(precision_funcs)

    def export_time_stats(
        self, output_dir: str = "results", filename: Optional[str] = None
    ) -> Optional[str]:
        """Export timing data to CSV format."""
        return (
            None
            if not self.time_collector
            else self.time_collector.export_results(output_dir, filename)
        )
