from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.density_grid import DensityGrid


@dataclass
class ContourResult:
    contour_pixels: np.ndarray
    contour_world: np.ndarray
    point_count: int


def extract_external_contour(mask: np.ndarray) -> np.ndarray:
    """
    Ищет внешний контур крупнейшей белой области.

    mask:
      0 = фон
      1 = деталь

    Возвращает массив точек в пиксельных координатах:
      shape = (N, 2)
      columns = [x_pixel, y_pixel]
    """
    mask_u8 = (mask.astype(np.uint8) * 255)

    contours, _hierarchy = cv2.findContours(
        mask_u8,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("Внешний контур не найден")

    largest = max(contours, key=cv2.contourArea)

    # OpenCV возвращает shape (N, 1, 2), приводим к (N, 2)
    contour_pixels = largest[:, 0, :].astype(np.float32)

    return contour_pixels


def simplify_contour_pixels(
    contour_pixels: np.ndarray,
    simplify_cells: float,
) -> np.ndarray:
    """
    Упрощает контур алгоритмом Douglas-Peucker.

    simplify_cells задаётся в ячейках grid.
    Например:
      cell = 1.8 мм
      simplify_mm = 2.0 мм
      simplify_cells = 2.0 / 1.8
    """
    if simplify_cells <= 0:
        return contour_pixels

    contour_cv = contour_pixels.reshape((-1, 1, 2)).astype(np.float32)

    simplified = cv2.approxPolyDP(
        contour_cv,
        epsilon=float(simplify_cells),
        closed=True,
    )

    return simplified[:, 0, :].astype(np.float32)


def contour_pixels_to_world(
    contour_pixels: np.ndarray,
    grid: DensityGrid,
) -> np.ndarray:
    """
    Переводит пиксельные координаты контура в координаты исходного .xyz.

    В density grid:
      x_pixel = индекс столбца
      y_pixel = индекс строки

    Координата берётся по центру ячейки.
    """
    x = grid.min_x + (contour_pixels[:, 0] + 0.5) * grid.cell_size
    y = grid.min_y + (contour_pixels[:, 1] + 0.5) * grid.cell_size

    return np.column_stack([x, y]).astype(np.float64)


def build_external_contour(
    mask: np.ndarray,
    grid: DensityGrid,
    simplify_mm: float = 0.0,
) -> ContourResult:
    raw_pixels = extract_external_contour(mask)

    simplify_cells = simplify_mm / grid.cell_size if grid.cell_size > 0 else 0.0
    simplified_pixels = simplify_contour_pixels(raw_pixels, simplify_cells)

    world = contour_pixels_to_world(simplified_pixels, grid)

    return ContourResult(
        contour_pixels=simplified_pixels,
        contour_world=world,
        point_count=int(len(simplified_pixels)),
    )


def save_contour_csv(
    contour_world: np.ndarray,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in contour_world:
            f.write(f"{x:.6f},{y:.6f}\n")