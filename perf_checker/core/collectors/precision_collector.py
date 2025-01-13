import json
import os
from typing import (
    Any,
    Dict,
    Optional,
)

from msprobe.pytorch import PrecisionDebugger

from ..base.base_collector import BaseCollector


class PrecisionCollector(BaseCollector):
    """Collects precision information using PrecisionDebugger with hierarchical support."""

    def __init__(self, precision_config: str, result_dir: Optional[str] = None):
        """Initialize the precision collector."""
        super().__init__("precision")
        self.config_path = precision_config
        self._active_debugger = None
        self.result_dir = result_dir
        # Load and validate the configuration
        with open(precision_config, "r") as f:
            self.base_config = json.load(f)

        # Set up base dump path
        if result_dir:
            base_dir = self.result_dir
        else:
            base_dir = self.base_config.get("dump_path", "./results")
        self.base_dump_path = os.path.join(base_dir, "dump")
        os.makedirs(self.base_dump_path, exist_ok=True)

    def _get_dump_path(self, stage: str, suffixed_tag: str, run_id: int) -> str:
        """Get the dump path for the current run and function.

        Args:
            stage: The monitoring stage name
            suffixed_tag: The tag with suffix (e.g. "model1.forward_1")
            run_id: The current run ID
        """
        # Extract base tag and suffix from the suffixed tag
        base_tag, suffix = suffixed_tag.rsplit("_", 1)
        # Remove the call_x level, just use the suffix in the base folder
        func_dump_path = os.path.join(
            self.base_dump_path, f"run_{run_id}", stage, base_tag
        )
        os.makedirs(func_dump_path, exist_ok=True)
        return func_dump_path

    def _on_run_change(self, run_id: int) -> None:
        """Clean up debugger when changing runs."""
        if self._active_debugger is not None:
            self._end_collection()

    def _start_collection(self, stage: str, tag: str, suffixed_tag: str) -> None:
        """Start precision collection."""
        if self._active_debugger is not None:
            self._end_collection()

        dump_path = self._get_dump_path(stage, suffixed_tag, self.current_run)
        self._active_debugger = PrecisionDebugger(config_path=self.config_path)

        if hasattr(self._active_debugger, "config"):
            self._active_debugger.config.dump_path = dump_path

        self._active_debugger.start()
        self.logger.debug(f"Started precision collection for {stage}/{suffixed_tag}")

    def _end_collection(self) -> None:
        """End precision collection."""
        if self._active_debugger is None:
            return

        try:
            self._active_debugger.step()
            self._active_debugger.stop()
            self.logger.debug(
                f"Ended precision collection for {self.current_context[0]}/{self.current_context[1]}"
                if self.current_context
                else "Ended precision collection"
            )
        finally:
            self._active_debugger = None

    def get_stats(
        self, stage: Optional[str] = None, tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get precision statistics (placeholder as stats are in dump files)."""
        return {}

    def export_results(
        self, output_dir: str, filename: Optional[str] = None
    ) -> Optional[str]:
        """Export precision data (automatically exported to dump files)."""
        return None
