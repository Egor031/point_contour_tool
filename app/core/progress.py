from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


ProgressStage = Literal["statistics", "density"]


@dataclass(frozen=True)
class ProcessingProgress:
    """Byte progress for one source-reading processing stage."""

    stage: ProgressStage
    completed: int
    total: int
    fraction: float


ProgressCallback = Callable[[ProcessingProgress], None]
