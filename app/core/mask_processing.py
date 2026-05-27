from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


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