from typing import Dict, List


class TagManager:
    """Manages unique tags for monitoring operations across all runs."""

    def __init__(self):
        """Initialize tDictag manager."""
        self.current_run = 1
        # Store run-specific data
        self.run_data: Dict[
            int, Dict[str, Dict[str, List[str]]]
        ] = {}  # run_id -> stage -> tag -> list of suffixed tags
        # Store max counts across all runs for each stage/tag combination
        self.max_counts: Dict[str, Dict[str, int]] = {}  # stage -> tag -> max count
        # Initialize first run
        self.run_data[self.current_run] = {}

    def set_run(self, run_id: int) -> None:
        """Set the current run ID and initialize if needed.

        Args:
            run_id: The run identifier (1-based indexing)
        """
        self.current_run = max(1, run_id)
        # Initialize run data if not exists
        if self.current_run not in self.run_data:
            self.run_data[self.current_run] = {}

    def get_unique_tag(self, stage: str, tag: str) -> str:
        """Get a unique tag with suffix for the current run.

        Args:
            stage: The monitoring stage name
            tag: The base tag name

        Returns:
            A unique tag with suffix (e.g., tag_1, tag_2)
        """
        # Ensure run data exists
        if self.current_run not in self.run_data:
            self.run_data[self.current_run] = {}

        # Initialize stage dictionaries if needed
        if stage not in self.run_data[self.current_run]:
            self.run_data[self.current_run][stage] = {}
        if stage not in self.max_counts:
            self.max_counts[stage] = {}

        # Initialize tag lists and counts if needed
        if tag not in self.run_data[self.current_run][stage]:
            self.run_data[self.current_run][stage][tag] = []
        if tag not in self.max_counts[stage]:
            self.max_counts[stage][tag] = 0

        # Get next count for this run
        current_count = len(self.run_data[self.current_run][stage][tag]) + 1

        # Update max count if needed
        self.max_counts[stage][tag] = max(self.max_counts[stage][tag], current_count)

        # Create suffixed tag
        suffixed_tag = f"{tag}_{current_count}"

        # Add to run data
        self.run_data[self.current_run][stage][tag].append(suffixed_tag)

        return suffixed_tag

    def get_column_list(self, run_id: int, stage: str, tag: str) -> List[str]:
        """Get the ordered list of suffixed tags for a specific run.

        Args:
            run_id: The run identifier (1-based indexing)
            stage: The monitoring stage name
            tag: The base tag name

        Returns:
            List of suffixed tags in order of creation
        """
        run_id = max(1, run_id)
        # Initialize if not exists
        if run_id not in self.run_data:
            self.run_data[run_id] = {}
        if stage not in self.run_data[run_id]:
            self.run_data[run_id][stage] = {}
        if tag not in self.run_data[run_id][stage]:
            self.run_data[run_id][stage][tag] = []

        return self.run_data[run_id][stage][tag]

    def get_max_columns(self, stage: str, tag: str) -> int:
        """Get the maximum number of columns needed for a stage/tag combination.

        Args:
            stage: The monitoring stage name
            tag: The base tag name

        Returns:
            Maximum number of columns needed
        """
        return self.max_counts.get(stage, {}).get(tag, 0)

    def get_all_runs(self) -> List[int]:
        """Get a sorted list of all run IDs.

        Returns:
            Sorted list of run IDs
        """
        return sorted(self.run_data.keys())

    def get_all_stages(self) -> List[str]:
        """Get a sorted list of all stages.

        Returns:
            Sorted list of stages
        """
        stages = set()
        for run_data in self.run_data.values():
            stages.update(run_data.keys())
        return sorted(stages)

    def get_all_tags(self, stage: str) -> List[str]:
        """Get a sorted list of all base tags for a stage.

        Args:
            stage: The monitoring stage name

        Returns:
            Sorted list of base tags
        """
        tags = set()
        for run_data in self.run_data.values():
            if stage in run_data:
                tags.update(run_data[stage].keys())
        return sorted(tags)
