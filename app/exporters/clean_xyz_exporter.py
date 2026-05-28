from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.density_grid import DensityGrid
from app.core.xyz_reader import iter_xyz_points


def export_clean_points(
    input_path: str | Path,
    output_path: str | Path,
    grid: DensityGrid,
    mask: np.ndarray,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = mask.shape
    written_count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for x, y, z in iter_xyz_points(
            input_path,
            show_progress=True,
            desc="Clean export",
        ):
            ix = int((x - grid.min_x) / grid.cell_size)
            iy = int((y - grid.min_y) / grid.cell_size)

            if not (0 <= ix < width and 0 <= iy < height):
                continue

            if mask[iy, ix] == 0:
                continue

            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            written_count += 1

    return written_count
