from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

from tqdm import tqdm

from app.core.progress import ProcessingProgress, ProgressCallback, ProgressStage


_MIN_PROGRESS_INTERVAL_BYTES = 1024 * 1024
_MAX_PROGRESS_UPDATES = 1000


@dataclass
class PointCloudStats:
    file_path: Path
    point_count: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


def parse_xyz_line(line: str) -> Tuple[float, float, float] | None:
    line = line.strip()
    if not line:
        return None

    # ВАЖНО:
    # Если в будущем попадутся числа с десятичной запятой,
    # этот replace(",", " ") сломает дробную часть.
    # Пока оставляем, так как у текущих файлов формат X Y Z.
    line = line.replace(";", " ").replace(",", " ")
    parts = line.split()

    if len(parts) < 3:
        return None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def iter_xyz_points(
    file_path: str | Path,
    show_progress: bool = False,
    desc: str = "Reading",
    progress_callback: ProgressCallback | None = None,
    progress_stage: ProgressStage | None = None,
) -> Iterator[Tuple[float, float, float]]:
    path = Path(file_path)
    total_size = path.stat().st_size

    if progress_callback is not None and progress_stage is None:
        raise ValueError("progress_stage is required when progress_callback is set")

    progress_interval = max(
        _MIN_PROGRESS_INTERVAL_BYTES,
        (total_size + _MAX_PROGRESS_UPDATES - 1) // _MAX_PROGRESS_UPDATES,
    )
    next_progress_at = progress_interval
    completed_bytes = 0

    if progress_callback is not None:
        progress_callback(
            ProcessingProgress(
                stage=progress_stage,
                completed=0,
                total=total_size,
                fraction=0.0,
            )
        )

    with path.open("rb") as file:

        if show_progress:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=desc,
            )
        else:
            progress = None

        try:
            for raw_line in file:
                completed_bytes += len(raw_line)
                if progress is not None:
                    progress.update(len(raw_line))

                if (
                    progress_callback is not None
                    and completed_bytes >= next_progress_at
                    and completed_bytes < total_size
                ):
                    progress_callback(
                        ProcessingProgress(
                            stage=progress_stage,
                            completed=completed_bytes,
                            total=total_size,
                            fraction=completed_bytes / total_size,
                        )
                    )
                    next_progress_at = (
                        completed_bytes // progress_interval + 1
                    ) * progress_interval

                line = raw_line.decode("utf-8", errors="ignore")
                point = parse_xyz_line(line)
                if point is not None:
                    yield point

            if progress_callback is not None:
                progress_callback(
                    ProcessingProgress(
                        stage=progress_stage,
                        completed=total_size,
                        total=total_size,
                        fraction=1.0,
                    )
                )
        finally:
            if progress is not None:
                progress.close()


def compute_stats(
    file_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> PointCloudStats:
    path = Path(file_path)

    point_count = 0

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_z = float("inf")
    max_z = float("-inf")

    for x, y, z in iter_xyz_points(
        path,
        show_progress=progress_callback is None,
        desc="Stats pass",
        progress_callback=progress_callback,
        progress_stage="statistics",
    ):
        point_count += 1

        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x

        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z

    if point_count == 0:
        raise ValueError(f"Файл не содержит корректных XYZ-точек: {path}")

    return PointCloudStats(
        file_path=path,
        point_count=point_count,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
    )


def export_decimated_points(
    input_path: str | Path,
    output_path: str | Path,
    stats: PointCloudStats,
    decimate_cell_mm: float,
) -> int:
    if decimate_cell_mm <= 0:
        raise ValueError("decimate_cell_mm must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_cells: set[tuple[int, int]] = set()
    exported_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for x, y, z in iter_xyz_points(
            input_path,
            show_progress=True,
            desc="Decimated export",
        ):
            ix = math.floor((x - stats.min_x) / decimate_cell_mm)
            iy = math.floor((y - stats.min_y) / decimate_cell_mm)
            cell_key = (ix, iy)
            if cell_key in seen_cells:
                continue

            seen_cells.add(cell_key)
            output_file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            exported_count += 1

    return exported_count
