"""Base interface for all collectors in the performance analysis tool."""

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    TypeVar,
)

from ...utils import setup_logger
from ..storage.tag_manager import TagManager

# Type variables for better type hints
P = ParamSpec("P")
T = TypeVar("T")


class BaseCollector(ABC):
    """Abstract base class for all collectors with improved thread safety."""

    def __init__(self, collector_type: str):
        """Initialize the collector with thread-safe context tracking.

        Args:
            collector_type: Type of collector ("time" or "precision")
        """
        self._context_lock = Lock()
        self._context_stacks: Dict[int, List[Tuple[str, str]]] = {}
        self._run_lock = Lock()
        self.current_run = 1
        self.tag_manager = TagManager()
        self.logger = setup_logger(collector_type)
        self.monitored_funcs: Dict[int, Set[Tuple[str, str]]] = {1: set()}
        self._collector_type = collector_type

    @property
    def current_context(self) -> Optional[Tuple[str, str]]:
        """Get the current monitoring context (stage, tag) for the current thread."""
        thread_id = threading.get_ident()
        with self._context_lock:
            stack = self._context_stacks.get(thread_id, [])
            return stack[-1] if stack else None

    def _manage_context_stack(
        self, operation: str, context: Optional[Tuple[str, str]] = None
    ) -> Optional[Tuple[str, str]]:
        """Manage the context stack with thread safety.

        Args:
            operation: Either 'push' or 'pop'
            context: The (stage, tag) tuple to push if operation is 'push'

        Returns:
            The popped context if operation is 'pop', None otherwise
        """
        thread_id = threading.get_ident()
        with self._context_lock:
            if thread_id not in self._context_stacks:
                self._context_stacks[thread_id] = []

            if operation == "push" and context:
                self._context_stacks[thread_id].append(context)
                return None
            elif operation == "pop":
                return (
                    self._context_stacks[thread_id].pop()
                    if self._context_stacks[thread_id]
                    else None
                )

        return None

    def push_context(self, stage: str, tag: str) -> None:
        """Push a new monitoring context for the current thread."""
        self._manage_context_stack("push", (stage, tag))

    def pop_context(self) -> Optional[Tuple[str, str]]:
        """Pop the current monitoring context for the current thread."""
        return self._manage_context_stack("pop")

    def set_run(self, run_id: int) -> None:
        """Set the current run ID with thread safety."""
        with self._run_lock:
            self.current_run = max(1, run_id)
            self.tag_manager.set_run(self.current_run)
            # Clear context stacks for all threads
            with self._context_lock:
                self._context_stacks.clear()
            if self.current_run not in self.monitored_funcs:
                self.monitored_funcs[self.current_run] = set()
            self._on_run_change(run_id)

    def _on_run_change(self, run_id: int) -> None:
        """Hook for subclasses to implement run-specific initialization."""
        pass

    def get_monitored_functions(self) -> Set[Tuple[str, str]]:
        """Get the set of monitored functions across all runs."""
        monitored = set()
        for run_funcs in self.monitored_funcs.values():
            monitored.update(run_funcs)
        return monitored

    def get_current_stack(self) -> List[Tuple[str, str]]:
        """Get the current call stack of monitored functions for the current thread."""
        thread_id = threading.get_ident()
        with self._context_lock:
            return list(self._context_stacks.get(thread_id, []))

    @abstractmethod
    def _start_collection(self, stage: str, tag: str, suffixed_tag: str) -> None:
        """Start actual data collection.

        Args:
            stage: The monitoring stage name
            tag: The base tag name
            suffixed_tag: The tag with unique suffix for this call
        """
        pass

    @abstractmethod
    def _end_collection(self) -> None:
        """End actual data collection."""
        pass

    @contextmanager
    def monitor_context(self, stage: str, tag: str) -> Generator[None, None, None]:
        """Context manager for monitoring code blocks with automatic cleanup.

        Args:
            stage: The monitoring stage name
            tag: The monitoring tag

        Yields:
            None
        """
        self.start(stage, tag)
        try:
            yield
        finally:
            self.end()

    def start(self, stage: str, tag: str) -> None:
        """Start monitoring a code block with improved error handling."""
        try:
            suffixed_tag = self.tag_manager.get_unique_tag(stage, tag)
            self._start_collection(stage, tag, suffixed_tag)
            self.monitored_funcs[self.current_run].add((stage, tag))
            self.logger.debug(
                f"Started monitoring block - Stage: {stage}, Tag: {suffixed_tag}"
            )
            self.push_context(stage, suffixed_tag)
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            raise

    def end(self) -> None:
        """End monitoring the current code block."""
        if not self.current_context:
            self.logger.warning(
                f"[{self._collector_type}] No active monitoring block to end"
            )
            return

        stage, suffixed_tag = self.current_context
        base_tag = suffixed_tag.rsplit("_", 1)[0]
        column_list = self.tag_manager.get_column_list(
            self.current_run, stage, base_tag
        )

        if not column_list:
            self.logger.warning(
                f"[{self._collector_type}] No column list found for {stage}/{base_tag}"
            )
            self._end_collection()
            self.pop_context()
            return

        suffixed_tag = column_list[-1]
        self._end_collection()
        self.logger.debug(
            f"[{self._collector_type}] Ended monitoring block - Stage: {stage}, Tag: {suffixed_tag}"
        )
        self.pop_context()

    def __call__(
        self, stage: str, tag: str
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator to monitor function execution with improved type safety."""

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                current = self.current_context
                if current is not None:
                    self.logger.debug(
                        f"[{self._collector_type}] Skipping decorator monitoring for {stage}/{tag} "
                        f"due to active manual monitoring {current[0]}/{current[1]}"
                    )
                    return func(*args, **kwargs)

                with self.monitor_context(stage, tag):
                    return func(*args, **kwargs)

            return wrapper

        return decorator
