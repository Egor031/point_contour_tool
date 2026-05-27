from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.xyz_reader import PointCloudStats, iter_xyz_points


@dataclass
class DensityGrid:
    density: np.ndarray
    cell_size: float
    min_x: float
    min_y: float

    @property
    def height(self) -> int:
        return self.density.shape[0]

    @property
    def width(self) -> int:
        return self.density.shape[1]


def build_density_grid(
    file_path: str | Path,
    stats: PointCloudStats,
    cell_size: float,
) -> DensityGrid:
    if cell_size <= 0:
        raise ValueError("cell_size должен быть больше 0")

    width_cells = int(np.ceil(stats.width / cell_size)) + 1
    height_cells = int(np.ceil(stats.height / cell_size)) + 1

    if width_cells <= 0 or height_cells <= 0:
        raise ValueError("Некорректный размер сетки")

    print(f"Размер density grid: {width_cells} x {height_cells}")
    print(f"Всего ячеек: {width_cells * height_cells:,}")

    density = np.zeros((height_cells, width_cells), dtype=np.uint32)

    for x, y, _z in iter_xyz_points(file_path, show_progress=True, desc="Density pass"):
        ix = int((x - stats.min_x) / cell_size)
        iy = int((y - stats.min_y) / cell_size)

        if 0 <= ix < width_cells and 0 <= iy < height_cells:
            density[iy, ix] += 1

    return DensityGrid(
        density=density,
        cell_size=cell_size,
        min_x=stats.min_x,
        min_y=stats.min_y,
    )