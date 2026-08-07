from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.progress import ProcessingProgress, ProgressStage
from app.services.coarse_processing import DensityProcessingResult, prepare_density
from app.ui.density_workflow import density_grid_to_preview, preview_to_texture_rgba


@dataclass(frozen=True)
class DensityWorkerProgress:
    progress: ProcessingProgress


@dataclass(frozen=True)
class DensityWorkerResult:
    result: DensityProcessingResult
    preview: np.ndarray
    texture_rgba: np.ndarray


@dataclass(frozen=True)
class DensityWorkerError:
    error: Exception


DensityWorkerMessage = DensityWorkerProgress | DensityWorkerResult | DensityWorkerError
PrepareDensityCallable = Callable[..., DensityProcessingResult]


def format_byte_count(byte_count: int) -> str:
    value = float(max(0, byte_count))
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def format_progress_bytes(progress: ProcessingProgress) -> str:
    return (
        f"{format_byte_count(progress.completed)} / "
        f"{format_byte_count(progress.total)}"
    )


def progress_stage_label(stage: ProgressStage) -> str:
    if stage == "statistics":
        return "Reading point cloud / calculating statistics"
    return "Building density map"


class DensityWorker:
    def __init__(
        self,
        prepare_density_func: PrepareDensityCallable = prepare_density,
    ) -> None:
        self._prepare_density = prepare_density_func
        self._messages: queue.Queue[DensityWorkerMessage] = queue.Queue()
        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def start(self, source_path: str | Path, cell_size: float) -> bool:
        source_snapshot = Path(source_path)
        cell_snapshot = float(cell_size)

        with self._lock:
            if self._active:
                return False
            self._active = True
            self._thread = threading.Thread(
                target=self._run,
                args=(source_snapshot, cell_snapshot),
                name="density-processing-worker",
                daemon=True,
            )
            thread = self._thread

        try:
            thread.start()
        except Exception:
            with self._lock:
                self._active = False
                self._thread = None
            raise
        return True

    def drain_messages(self) -> list[DensityWorkerMessage]:
        messages: list[DensityWorkerMessage] = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def mark_finished(self) -> None:
        with self._lock:
            self._active = False

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self, source_path: Path, cell_size: float) -> None:
        def forward_progress(progress: ProcessingProgress) -> None:
            self._messages.put(DensityWorkerProgress(progress=progress))

        try:
            result = self._prepare_density(
                source_path,
                cell_size=cell_size,
                progress_callback=forward_progress,
            )
            preview = density_grid_to_preview(result.grid)
            texture_rgba = preview_to_texture_rgba(preview)
            self._messages.put(
                DensityWorkerResult(
                    result=result,
                    preview=preview,
                    texture_rgba=texture_rgba,
                )
            )
        except Exception as exc:
            self._messages.put(DensityWorkerError(error=exc))
