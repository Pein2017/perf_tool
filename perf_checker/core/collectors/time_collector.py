import time
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from ...utils import export_timing_to_csv
from ..base.base_collector import BaseCollector


class TimeCollector(BaseCollector):
    """Collects execution time for functions with hierarchical support."""

    def __init__(self):
        """Initialize the time collector."""
        super().__init__("time")
        self.run_times: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {1: {}}
        self._active_timers: Dict[Tuple[str, str], float] = {}
        self.logger.info("Initialized TimeCollector")

    def _on_run_change(self, run_id: int) -> None:
        """Initialize data structures for new run."""
        if run_id not in self.run_times:
            self.run_times[run_id] = {}
        self._active_timers.clear()

    def _start_collection(self, stage: str, tag: str, suffixed_tag: str) -> None:
        """Start timing collection."""
        self._active_timers[(stage, suffixed_tag)] = time.time()
        self.logger.debug(f"Started timing collection for {stage}/{suffixed_tag}")

    def _end_collection(self) -> None:
        """End timing collection."""
        if not self.current_context:
            return

        stage, suffixed_tag = self.current_context
        start_time = self._active_timers.pop((stage, suffixed_tag), None)
        if start_time is None:
            return

        elapsed_time = time.time() - start_time
        base_tag = suffixed_tag.rsplit("_", 1)[0]

        # Ensure data structures exist
        if stage not in self.run_times[self.current_run]:
            self.run_times[self.current_run][stage] = {}
        if base_tag not in self.run_times[self.current_run][stage]:
            self.run_times[self.current_run][stage][base_tag] = {}

        self.run_times[self.current_run][stage][base_tag][suffixed_tag] = elapsed_time
        self.logger.debug(
            f"Ended timing collection for {stage}/{suffixed_tag}, elapsed: {elapsed_time:.6f}s"
        )

    def get_stats(
        self, stage: Optional[str] = None, tag: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get timing statistics for specified stage and tag."""
        stats = {}
        stages = [stage] if stage else self.tag_manager.get_all_stages()

        for s in stages:
            stats[s] = {}
            base_tags = [tag] if tag else self.tag_manager.get_all_tags(s)

            for base_tag in base_tags:
                # Initialize stats for all columns
                max_cols = self.tag_manager.get_max_columns(s, base_tag)
                for col in range(1, max_cols + 1):
                    suffixed_tag = f"{base_tag}_{col}"
                    stats[s][suffixed_tag] = {
                        "mean": 0.0,
                        "min": float("inf"),
                        "max": float("-inf"),
                        "count": 0,
                        "total": 0.0,
                        "times_by_run": {},
                    }

                # Collect stats across runs
                for run_id in self.tag_manager.get_all_runs():
                    run_data = (
                        self.run_times.get(run_id, {}).get(s, {}).get(base_tag, {})
                    )
                    for suffixed_tag, time_value in run_data.items():
                        stat_entry = stats[s][suffixed_tag]
                        stat_entry["times_by_run"][run_id] = time_value
                        stat_entry["count"] += 1
                        stat_entry["total"] += time_value
                        stat_entry["min"] = min(stat_entry["min"], time_value)
                        stat_entry["max"] = max(stat_entry["max"], time_value)

                # Calculate means for non-empty entries
                for suffixed_tag in list(stats[s].keys()):
                    stat_entry = stats[s][suffixed_tag]
                    if stat_entry["count"] > 0:
                        stat_entry["mean"] = stat_entry["total"] / stat_entry["count"]
                    else:
                        # Remove empty entries
                        del stats[s][suffixed_tag]

        return stats

    def export_results(
        self, output_dir: str, filename: Optional[str] = None
    ) -> Optional[str]:
        """Export timing data to CSV format."""
        return export_timing_to_csv(self.run_times, output_dir, filename)

    def __call__(self, stage: str, tag: str) -> Callable:
        """Decorator to monitor function execution."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                current = self.current_context
                self.logger.debug(
                    f"Decorator called for {stage}/{tag}, current context: {current}, "
                    f"stack: {self.get_current_stack()}"
                )
                if current is not None:
                    self.logger.debug(
                        f"Skipping decorator monitoring for {stage}/{tag} "
                        f"due to active manual monitoring {current[0]}/{current[1]}"
                    )
                    return func(*args, **kwargs)

                self.start(stage, tag)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.logger.debug(
                        f"Decorator finally block for {stage}/{tag}, "
                        f"stack before end: {self.get_current_stack()}"
                    )
                    self.end()

            return wrapper

        return decorator
