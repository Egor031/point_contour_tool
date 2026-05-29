from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.density_grid import DensityGrid
from app.core.xyz_reader import iter_xyz_points


def build_boundary_band_mask(mask: np.ndarray, width_cells: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)

    if width_cells <= 0:
        width_cells = 1

    contours, _hierarchy = cv2.findContours(
        mask_u8,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )

    boundary = np.zeros_like(mask_u8, dtype=np.uint8)

    if not contours:
        return boundary

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(
        boundary,
        [largest],
        contourIdx=-1,
        color=1,
        thickness=1,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary_mask = cv2.dilate(boundary, kernel, iterations=width_cells)
    return (boundary_mask > 0).astype(np.uint8)


def export_boundary_points(
    input_path: str | Path,
    output_path: str | Path,
    grid: DensityGrid,
    boundary_mask: np.ndarray,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = boundary_mask.shape
    written_count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for x, y, z in iter_xyz_points(
            input_path,
            show_progress=True,
            desc="Boundary export",
        ):
            ix = int((x - grid.min_x) / grid.cell_size)
            iy = int((y - grid.min_y) / grid.cell_size)

            if not (0 <= ix < width and 0 <= iy < height):
                continue

            if boundary_mask[iy, ix] == 0:
                continue

            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            written_count += 1

    return written_count
