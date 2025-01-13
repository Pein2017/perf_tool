"""Export utilities for the performance analysis tool."""

import csv
import os
from datetime import datetime
from typing import Dict, Optional, Set

from .logging_utils import setup_logger

logger = setup_logger("export")


def _get_base_operator_names(
    times: Dict[int, Dict[str, Dict[str, Dict[str, float]]]],
) -> Set[str]:
    """Get all unique base operator names without suffixes.

    Args:
        times: Nested dictionary of timing data

    Returns:
        Set of unique operator names in 'stage/tag' format
    """
    base_ops = set()
    for run_data in times.values():
        for stage in run_data:
            for tag in run_data[stage]:
                base_ops.add(f"{stage}/{tag}")
    return base_ops


def export_timing_to_csv(
    times: Dict[int, Dict[str, Dict[str, Dict[str, float]]]],
    output_dir: str = "results/time",
    filename: Optional[str] = None,
) -> str:
    """Export timing data to CSV format.

    Each row represents one experiment run, with columns for each operator.
    Operator names are suffixed with the run number (e.g., op_1, op_2).
    Missing values (when an operator isn't called in a run) are set to NULL.

    Args:
        times: Nested dictionary of timing data (run_id -> stage -> base_tag -> suffixed_tag -> time)
        output_dir: Directory to save the CSV file
        filename: Optional filename for the CSV file

    Returns:
        Path to the created CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"timing_results_{datetime.now().strftime('%Y%m%d_%H')}.csv"
    filepath = os.path.join(output_dir, filename)

    # Get all base operator names (without suffixes)
    base_ops = _get_base_operator_names(times)

    # Get all run IDs and determine max suffix for each operator
    run_ids = sorted(times.keys())
    max_suffixes = {}
    for stage_tag in base_ops:
        stage, tag = stage_tag.split("/")
        max_suffix = 0
        for run_id in run_ids:
            if stage in times[run_id] and tag in times[run_id][stage]:
                max_suffix = max(
                    max_suffix,
                    max(
                        int(suffixed_tag.split("_")[-1])
                        for suffixed_tag in times[run_id][stage][tag].keys()
                    ),
                )
        max_suffixes[stage_tag] = max_suffix

    # Create headers
    headers = ["run_id"]
    for base_op in sorted(base_ops):
        headers.extend([f"{base_op}_{i+1}" for i in range(max_suffixes[base_op])])

    # Create rows
    rows = []
    for run_id in run_ids:
        row = [run_id]  # run_id is already 1-based
        row_data = {}

        # Go through all stage/tag combinations
        for stage_tag in sorted(base_ops):
            stage, tag = stage_tag.split("/")
            if stage in times[run_id] and tag in times[run_id][stage]:
                for suffixed_tag, value in times[run_id][stage][tag].items():
                    suffix = int(suffixed_tag.split("_")[-1])
                    col_name = f"{stage_tag}_{suffix}"
                    row_data[col_name] = value

        # Fill row with values or None
        for header in headers[1:]:  # Skip run_id
            row.append(row_data.get(header, None))

        rows.append(row)

    # Write to CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    logger.info(f"Exported timing data to {filepath}")
    return filepath
