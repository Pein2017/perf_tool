"""Configuration classes for monitoring functionality."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MonitorConfig:
    """Configuration for function monitoring."""

    stage: str
    tag: str
    enable_time: bool = True
    enable_precision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
