from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.density_grid import DensityGrid
from app.core.xyz_reader import PointCloudStats


def density_to_image(
    density: np.ndarray,
    max_size: int = 3000,
    flip_y: bool = True,
) -> np.ndarray:
    density_float = density.astype(np.float32)

    # Логарифм помогает видеть и слабые, и плотные области одновременно.
    image = np.log1p(density_float)

    max_value = float(image.max())
    if max_value > 0:
        image = image / max_value

    image = (image * 255).astype(np.uint8)

    if flip_y:
        image = np.flipud(image)

    h, w = image.shape
    scale = min(max_size / max(w, h), 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def smooth_density(
    density: np.ndarray,
    sigma_cells: float,
) -> np.ndarray:
    """
    Сглаживает карту плотности.

    sigma_cells задаётся в ячейках, не в миллиметрах.
    Например:
      cell = 1.0 мм, sigma = 2 → сглаживание примерно на 2 мм.
      cell = 0.5 мм, sigma = 4 → сглаживание примерно на 2 мм.
    """
    if sigma_cells <= 0:
        return density

    density_float = density.astype(np.float32)
    return cv2.GaussianBlur(
        density_float,
        ksize=(0, 0),
        sigmaX=sigma_cells,
        sigmaY=sigma_cells,
    )


def save_density_preview(
    grid: DensityGrid,
    output_path: str | Path,
    max_size: int = 3000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = density_to_image(grid.density, max_size=max_size)
    cv2.imwrite(str(output_path), image)


def save_smoothed_density_preview(
    grid: DensityGrid,
    output_path: str | Path,
    sigma_mm: float,
    max_size: int = 3000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sigma_cells = sigma_mm / grid.cell_size
    smoothed = smooth_density(grid.density, sigma_cells=sigma_cells)

    image = density_to_image(smoothed, max_size=max_size)
    cv2.imwrite(str(output_path), image)


def save_report(
    stats: PointCloudStats,
    grid: DensityGrid,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nonzero_cells = int(np.count_nonzero(grid.density))
    max_density = int(grid.density.max())
    mean_density_nonzero = (
        float(grid.density[grid.density > 0].mean())
        if nonzero_cells > 0
        else 0.0
    )

    text = f"""Point cloud report

File: {stats.file_path}

Point count: {stats.point_count:,}

Bounding box:
  X: {stats.min_x:.6f} .. {stats.max_x:.6f}
  Y: {stats.min_y:.6f} .. {stats.max_y:.6f}
  Z: {stats.min_z:.6f} .. {stats.max_z:.6f}

Size:
  Width:  {stats.width:.6f}
  Height: {stats.height:.6f}

Density grid:
  Cell size: {grid.cell_size}
  Width cells:  {grid.width}
  Height cells: {grid.height}
  Total cells:  {grid.width * grid.height:,}
  Non-empty cells: {nonzero_cells:,}
  Max density in cell: {max_density}
  Mean density in non-empty cells: {mean_density_nonzero:.3f}
"""

    output_path.write_text(text, encoding="utf-8")

def save_mask_preview(
    mask: np.ndarray,
    output_path: str | Path,
    max_size: int = 3000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = (mask.astype(np.uint8) * 255)

    # Переворачиваем по Y, чтобы совпадало с density preview.
    image = np.flipud(image)

    h, w = image.shape
    scale = min(max_size / max(w, h), 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(output_path), image)