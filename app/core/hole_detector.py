from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.density_grid import DensityGrid


@dataclass
class HoleCandidate:
    id: int

    center_x: float
    center_y: float
    radius: float
    diameter: float

    center_px: float
    center_py: float
    radius_px: float

    area_cells: int
    area_mm2: float

    bbox_width_mm: float
    bbox_height_mm: float
    aspect_ratio: float

    circularity: float
    mean_error_mm: float
    max_error_mm: float
    error_ratio: float

    accepted: bool
    reject_reason: str = ""


def _fit_circle_least_squares(points_xy: np.ndarray) -> tuple[float, float, float]:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy должен иметь форму (N, 2)")

    if len(points_xy) < 3:
        raise ValueError("Для окружности нужно минимум 3 точки")

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    A, B, C = params

    cx = -A / 2.0
    cy = -B / 2.0
    r_sq = cx * cx + cy * cy - C

    if r_sq <= 0:
        raise ValueError("Некорректный радиус окружности")

    r = float(np.sqrt(r_sq))
    return float(cx), float(cy), r


def _component_touches_border(
    left: int,
    top: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> bool:
    right = left + width - 1
    bottom = top + height - 1

    return (
        left == 0
        or top == 0
        or right == image_width - 1
        or bottom == image_height - 1
    )


def _pixel_points_to_world(points_pixels: np.ndarray, grid: DensityGrid) -> np.ndarray:
    x = grid.min_x + (points_pixels[:, 0] + 0.5) * grid.cell_size
    y = grid.min_y + (points_pixels[:, 1] + 0.5) * grid.cell_size

    return np.column_stack([x, y]).astype(np.float64)


def detect_circular_holes(
    mask: np.ndarray,
    grid: DensityGrid,
    min_diameter_mm: float = 8.0,
    max_diameter_mm: float | None = None,
    min_circularity: float = 0.55,
    max_aspect_ratio_deviation: float = 0.35,
    max_error_ratio: float = 0.18,
) -> list[HoleCandidate]:
    mask_u8 = mask.astype(np.uint8)

    image_height, image_width = mask_u8.shape
    inverted = 1 - mask_u8

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        inverted,
        connectivity=8,
    )

    holes: list[HoleCandidate] = []
    next_id = 1

    for label_id in range(1, num_labels):
        area_cells = int(stats[label_id, cv2.CC_STAT_AREA])

        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if _component_touches_border(
            left=left,
            top=top,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        ):
            continue

        area_mm2 = area_cells * grid.cell_size * grid.cell_size
        equivalent_radius = float(np.sqrt(area_mm2 / np.pi))
        equivalent_diameter = equivalent_radius * 2.0

        bbox_width_mm = width * grid.cell_size
        bbox_height_mm = height * grid.cell_size

        if bbox_width_mm <= 0 or bbox_height_mm <= 0:
            continue

        aspect_ratio = bbox_width_mm / bbox_height_mm
        if aspect_ratio < 1.0:
            aspect_ratio = 1.0 / aspect_ratio

        accepted = True
        reject_reason = ""

        if equivalent_diameter < min_diameter_mm:
            accepted = False
            reject_reason = "too_small"

        if max_diameter_mm is not None and equivalent_diameter > max_diameter_mm:
            accepted = False
            reject_reason = "too_large"

        if aspect_ratio > (1.0 + max_aspect_ratio_deviation):
            accepted = False
            reject_reason = "bad_aspect_ratio"

        component_mask = (labels == label_id).astype(np.uint8)

        contours, _hierarchy = cv2.findContours(
            component_mask * 255,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        perimeter_pixels = cv2.arcLength(contour, closed=True)
        perimeter_mm = perimeter_pixels * grid.cell_size

        if perimeter_mm <= 0:
            continue

        circularity = float(4.0 * np.pi * area_mm2 / (perimeter_mm * perimeter_mm))

        if circularity < min_circularity:
            accepted = False
            reject_reason = "low_circularity"

        contour_pixels = contour[:, 0, :].astype(np.float64)
        contour_world = _pixel_points_to_world(contour_pixels, grid)

        try:
            cx, cy, r = _fit_circle_least_squares(contour_world)
        except ValueError:
            continue

        distances = np.sqrt(
            (contour_world[:, 0] - cx) ** 2
            + (contour_world[:, 1] - cy) ** 2
        )

        errors = np.abs(distances - r)

        mean_error_mm = float(errors.mean())
        max_error_mm = float(errors.max())
        error_ratio = float(mean_error_mm / r) if r > 0 else float("inf")

        if error_ratio > max_error_ratio:
            accepted = False
            reject_reason = "bad_circle_fit"

        center_px = (cx - grid.min_x) / grid.cell_size - 0.5
        center_py = (cy - grid.min_y) / grid.cell_size - 0.5
        radius_px = r / grid.cell_size

        holes.append(
            HoleCandidate(
                id=next_id,
                center_x=cx,
                center_y=cy,
                radius=r,
                diameter=2.0 * r,
                center_px=center_px,
                center_py=center_py,
                radius_px=radius_px,
                area_cells=area_cells,
                area_mm2=area_mm2,
                bbox_width_mm=bbox_width_mm,
                bbox_height_mm=bbox_height_mm,
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                mean_error_mm=mean_error_mm,
                max_error_mm=max_error_mm,
                error_ratio=error_ratio,
                accepted=accepted,
                reject_reason=reject_reason,
            )
        )

        next_id += 1

    return holes


def save_holes_csv(
    holes: list[HoleCandidate],
    output_path: str | Path,
    only_accepted: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "id,accepted,reject_reason,"
            "center_x,center_y,radius,diameter,center_px,center_py,radius_px,"
            "area_cells,area_mm2,"
            "bbox_width_mm,bbox_height_mm,aspect_ratio,"
            "circularity,mean_error_mm,max_error_mm,error_ratio\n"
        )

        for hole in holes:
            if only_accepted and not hole.accepted:
                continue

            f.write(
                f"{hole.id},"
                f"{int(hole.accepted)},"
                f"{hole.reject_reason},"
                f"{hole.center_x:.6f},"
                f"{hole.center_y:.6f},"
                f"{hole.radius:.6f},"
                f"{hole.diameter:.6f},"
                f"{hole.center_px:.6f},"
                f"{hole.center_py:.6f},"
                f"{hole.radius_px:.6f},"
                f"{hole.area_cells},"
                f"{hole.area_mm2:.6f},"
                f"{hole.bbox_width_mm:.6f},"
                f"{hole.bbox_height_mm:.6f},"
                f"{hole.aspect_ratio:.6f},"
                f"{hole.circularity:.6f},"
                f"{hole.mean_error_mm:.6f},"
                f"{hole.max_error_mm:.6f},"
                f"{hole.error_ratio:.6f}\n"
            )