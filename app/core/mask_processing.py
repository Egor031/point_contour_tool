from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.core.density_grid import DensityGrid


ThresholdMode = Literal["auto", "manual"]


@dataclass
class MaskResult:
    mask: np.ndarray
    threshold: float
    mode: ThresholdMode
    nonzero_density_cells: int
    mask_cells: int


def compute_auto_threshold(density: np.ndarray) -> float:
    """
    Автоматический подбор порога по непустым ячейкам.

    Идея:
    - 0 обычно означает пустоту/фон, его не учитываем.
    - Берём распределение плотностей среди непустых ячеек.
    - Нижние значения чаще соответствуют редкому мусору/слабым линиям.
    - Типичная область детали обычно плотнее.
    """
    nonzero = density[density > 0]

    if nonzero.size == 0:
        return 1.0

    # Осторожный стартовый вариант.
    # Берём 25-й процентиль непустых ячеек.
    # То есть отбрасываем самые слабые 25% непустых ячеек.
    threshold = float(np.percentile(nonzero, 25))

    # Порог не должен быть меньше 1.
    return max(1.0, threshold)


def build_mask_from_density(
    density: np.ndarray,
    mode: ThresholdMode = "auto",
    manual_threshold: float | None = None,
) -> MaskResult:
    if mode == "auto":
        threshold = compute_auto_threshold(density)
    else:
        if manual_threshold is None:
            raise ValueError("Для manual-режима нужен manual_threshold")
        threshold = float(manual_threshold)

    mask = density >= threshold

    return MaskResult(
        mask=mask.astype(np.uint8),
        threshold=threshold,
        mode=mode,
        nonzero_density_cells=int(np.count_nonzero(density)),
        mask_cells=int(np.count_nonzero(mask)),
    )


def remove_small_components(
    mask: np.ndarray,
    min_area_cells: int,
) -> np.ndarray:
    """
    Удаляет маленькие связные области.

    mask:
      0 = фон
      1 = предполагаемая деталь

    min_area_cells:
      минимальная площадь области в ячейках.
    """
    if min_area_cells <= 0:
        return mask

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    # label 0 — фон, его пропускаем.
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area_cells:
            cleaned[labels == label_id] = 1

    return cleaned


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Оставляет только крупнейшую связную область.
    Это полезно как первый автоматический режим, но не всегда правильно,
    если деталь состоит из нескольких раздельных областей.
    """
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(np.uint8)

    largest_label = None
    largest_area = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    if largest_label is not None:
        cleaned[labels == largest_label] = 1

    return cleaned


def apply_roi_to_mask(
    mask: np.ndarray,
    grid: DensityGrid,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    min_x, min_y, max_x, max_y = roi

    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y

    height, width = mask.shape

    x = grid.min_x + (np.arange(width) + 0.5) * grid.cell_size
    y = grid.min_y + (np.arange(height) + 0.5) * grid.cell_size

    x_inside = (x >= min_x) & (x <= max_x)
    y_inside = (y >= min_y) & (y <= max_y)

    roi_mask = y_inside[:, np.newaxis] & x_inside[np.newaxis, :]

    filtered = mask.astype(np.uint8).copy()
    filtered[~roi_mask] = 0

    return filtered


def apply_polygon_roi_to_mask(
    mask: np.ndarray,
    grid: DensityGrid,
    polygon_points: list[tuple[float, float]],
) -> np.ndarray:
    height, width = mask.shape

    if len(polygon_points) < 3:
        raise ValueError("polygon ROI must contain at least 3 points")

    points_pixels = []
    for x, y in polygon_points:
        ix = (x - grid.min_x) / grid.cell_size
        iy = (y - grid.min_y) / grid.cell_size
        points_pixels.append([int(round(ix)), int(round(iy))])

    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(points_pixels, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(polygon_mask, [polygon], 1)

    filtered = mask.astype(np.uint8).copy()
    filtered[polygon_mask == 0] = 0

    return filtered

def fill_small_holes(
    mask: np.ndarray,
    max_hole_area_cells: int,
) -> np.ndarray:
    """
    Заполняет маленькие чёрные области внутри белой маски.

    mask:
      0 = фон / пустота
      1 = деталь

    max_hole_area_cells:
      максимальная площадь внутренней пустоты в ячейках,
      которую нужно заполнить.
    """
    if max_hole_area_cells <= 0:
        return mask.astype(np.uint8)

    mask_u8 = mask.astype(np.uint8)

    # Инвертируем:
    # было: 1 = деталь, 0 = фон/дырки
    # стало: 1 = фон/дырки, 0 = деталь
    inverted = 1 - mask_u8

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8,
    )

    filled = mask_u8.copy()

    height, width = mask_u8.shape

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area > max_hole_area_cells:
            continue

        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        right = left + w - 1
        bottom = top + h - 1

        # Если область касается края изображения, это внешний фон,
        # а не внутренняя дырка. Не заполняем.
        touches_border = (
            left == 0
            or top == 0
            or right == width - 1
            or bottom == height - 1
        )

        if touches_border:
            continue

        filled[labels == label_id] = 1

    return filled
